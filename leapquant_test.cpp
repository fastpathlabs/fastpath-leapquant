// =============================================================================
// leapquant_test.cpp
//
// Tests for leapquant.cpp, and a synthetic comparison of storage methods.
//
// Build and run:   make test
//            or:   g++ -O2 -std=c++17 leapquant.cpp reference_heads.cpp leapquant_test.cpp -o leapquant_test
//                  ./leapquant_test            (comparison over 16,384 tokens)
//                  ./leapquant_test 65536      (comparison over 65,536 tokens)
//
// What "passing" means
//   Tests 1 to 7 compare the LeapQuant head with plain fp32 arithmetic and fail
//   the program (exit code 1) if any number is outside its allowed error.
//   Test 8 prints a comparison table. It feeds every storage method the same
//   stream of tokens. There is no model in the loop, so the table measures
//   arithmetic error, not answer quality.
// =============================================================================

#include "leapquant.h"
#include "reference_heads.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using leapquant::FullPrecisionHead;
using leapquant::Head;
using leapquant::Settings;
using leapquant::convert_float_to_half;
using leapquant::convert_half_to_float;
using leapquant::largest_absolute_value;
using leapquant::length_of;
using leapquant::length_of_difference;

// -----------------------------------------------------------------------------
// A tiny test framework: count checks, print the first failures.
// -----------------------------------------------------------------------------
static long g_checks_run = 0;
static long g_checks_failed = 0;

static void check(bool condition, const char* what_was_checked, double observed_value) {
    g_checks_run = g_checks_run + 1;
    if (!condition) {
        g_checks_failed = g_checks_failed + 1;
        if (g_checks_failed <= 20) {
            std::printf("  FAILED: %s (observed %g)\n", what_was_checked, observed_value);
        }
    }
}

// -----------------------------------------------------------------------------
// Singular values of a matrix, by the one-sided Jacobi method, in double.
// Slow and simple. It is the oracle for "how much of the matrix could
// `compensator_count` patterns capture at best".
// -----------------------------------------------------------------------------
static std::vector<double> singular_values_of(const std::vector<float>& matrix, int rows, int columns) {
    // The method works on the columns of a tall matrix. If the input is wide,
    // use its transpose: the singular values are the same.
    const bool transpose = rows < columns;
    const int tall_rows = transpose ? columns : rows;
    const int tall_columns = transpose ? rows : columns;

    std::vector<double> work(static_cast<std::size_t>(tall_rows) * tall_columns);
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < columns; ++column) {
            const double entry = matrix[static_cast<std::size_t>(row) * columns + column];
            if (transpose) {
                work[static_cast<std::size_t>(column) * tall_columns + row] = entry;
            } else {
                work[static_cast<std::size_t>(row) * tall_columns + column] = entry;
            }
        }
    }

    // Rotate pairs of columns until all columns are perpendicular to each other.
    for (int sweep = 0; sweep < 60; ++sweep) {
        double largest_remaining_overlap = 0.0;
        for (int first = 0; first < tall_columns - 1; ++first) {
            for (int second = first + 1; second < tall_columns; ++second) {
                double first_squared = 0.0;
                double second_squared = 0.0;
                double overlap = 0.0;
                for (int row = 0; row < tall_rows; ++row) {
                    const double a = work[static_cast<std::size_t>(row) * tall_columns + first];
                    const double b = work[static_cast<std::size_t>(row) * tall_columns + second];
                    first_squared += a * a;
                    second_squared += b * b;
                    overlap += a * b;
                }
                if (first_squared == 0.0 || second_squared == 0.0) {
                    continue;
                }
                const double relative_overlap = std::fabs(overlap) / std::sqrt(first_squared * second_squared);
                if (relative_overlap <= 1e-15) {
                    continue;
                }
                if (relative_overlap > largest_remaining_overlap) {
                    largest_remaining_overlap = relative_overlap;
                }
                const double zeta = (second_squared - first_squared) / (2.0 * overlap);
                const double sign = zeta >= 0.0 ? 1.0 : -1.0;
                const double tangent = sign / (std::fabs(zeta) + std::sqrt(1.0 + zeta * zeta));
                const double cosine = 1.0 / std::sqrt(1.0 + tangent * tangent);
                const double sine = cosine * tangent;
                for (int row = 0; row < tall_rows; ++row) {
                    const double a = work[static_cast<std::size_t>(row) * tall_columns + first];
                    const double b = work[static_cast<std::size_t>(row) * tall_columns + second];
                    work[static_cast<std::size_t>(row) * tall_columns + first] = cosine * a - sine * b;
                    work[static_cast<std::size_t>(row) * tall_columns + second] = sine * a + cosine * b;
                }
            }
        }
        if (largest_remaining_overlap < 1e-12) {
            break;
        }
    }

    // The singular values are the lengths of the columns.
    std::vector<double> values(static_cast<std::size_t>(tall_columns));
    for (int column = 0; column < tall_columns; ++column) {
        double squared = 0.0;
        for (int row = 0; row < tall_rows; ++row) {
            const double entry = work[static_cast<std::size_t>(row) * tall_columns + column];
            squared += entry * entry;
        }
        values[column] = std::sqrt(squared);
    }
    std::sort(values.begin(), values.end());
    std::reverse(values.begin(), values.end());
    return values;
}

// -----------------------------------------------------------------------------
// A synthetic stream of tokens for one Gated DeltaNet head.
//
// structured = true gives the stream the two properties the paper reports for
// real states: a few dominant patterns (keys and values follow a slow
// 6-dimensional signal) and channels of very different size.
// structured = false is plain random noise with neither property.
//
// decay = 1 - 10^(decay_exponent_low + decay_exponent_span * U), U uniform in [0, 1).
// A decay closer to 1 means the state forgets more slowly.
// -----------------------------------------------------------------------------
class SyntheticTokenStream {
public:
    SyntheticTokenStream(int key_channels,
                         int value_channels,
                         bool structured,
                         float decay_exponent_low,
                         float decay_exponent_span,
                         unsigned seed)
        : key_channels_(key_channels),
          value_channels_(value_channels),
          structured_(structured),
          decay_exponent_low_(decay_exponent_low),
          decay_exponent_span_(decay_exponent_span),
          random_engine_(seed),
          normal_(0.0f, 1.0f),
          uniform_(0.0f, 1.0f) {
        key_mixing_.resize(static_cast<std::size_t>(key_channels_) * kSignalSize);
        value_mixing_.resize(static_cast<std::size_t>(value_channels_) * kSignalSize);
        key_channel_gain_.resize(static_cast<std::size_t>(key_channels_));
        value_channel_gain_.resize(static_cast<std::size_t>(value_channels_));
        slow_signal_.resize(kSignalSize);
        for (std::size_t index = 0; index < key_mixing_.size(); ++index) key_mixing_[index] = normal_(random_engine_);
        for (std::size_t index = 0; index < value_mixing_.size(); ++index) value_mixing_[index] = normal_(random_engine_);
        for (std::size_t index = 0; index < key_channel_gain_.size(); ++index) {
            key_channel_gain_[index] = std::exp(0.8f * normal_(random_engine_));
        }
        for (std::size_t index = 0; index < value_channel_gain_.size(); ++index) {
            value_channel_gain_[index] = std::exp(0.8f * normal_(random_engine_));
        }
        for (int index = 0; index < kSignalSize; ++index) slow_signal_[index] = normal_(random_engine_);
    }

    // Produces the inputs of the next token.
    void next_token(float& alpha, float& beta, float* key, float* value, float* query) {
        // The slow signal changes a little with every token.
        for (int index = 0; index < kSignalSize; ++index) {
            slow_signal_[index] = 0.95f * slow_signal_[index] + 0.3122f * normal_(random_engine_);
        }
        make_unit_length_key_like_vector(key);
        make_unit_length_key_like_vector(query);
        for (int value_index = 0; value_index < value_channels_; ++value_index) {
            float entry = 0.7f * normal_(random_engine_);
            if (structured_) {
                for (int signal = 0; signal < kSignalSize; ++signal) {
                    entry += value_mixing_[static_cast<std::size_t>(value_index) * kSignalSize + signal] * slow_signal_[signal];
                }
                entry = entry * value_channel_gain_[value_index];
            }
            value[value_index] = entry;
        }
        alpha = 1.0f - std::pow(10.0f, decay_exponent_low_ + decay_exponent_span_ * uniform_(random_engine_));
        beta = 0.2f + 0.8f * uniform_(random_engine_);
    }

private:
    static const int kSignalSize = 6;

    // Keys and queries are unit-length vectors, as in the real models.
    void make_unit_length_key_like_vector(float* vector) {
        double squared_length = 0.0;
        for (int key_index = 0; key_index < key_channels_; ++key_index) {
            float entry = 0.7f * normal_(random_engine_);
            if (structured_) {
                for (int signal = 0; signal < kSignalSize; ++signal) {
                    entry += key_mixing_[static_cast<std::size_t>(key_index) * kSignalSize + signal] * slow_signal_[signal];
                }
                entry = entry * key_channel_gain_[key_index];
            }
            vector[key_index] = entry;
            squared_length += static_cast<double>(entry) * entry;
        }
        const float one_over_length = 1.0f / static_cast<float>(std::sqrt(squared_length));
        for (int key_index = 0; key_index < key_channels_; ++key_index) {
            vector[key_index] = vector[key_index] * one_over_length;
        }
    }

    int key_channels_;
    int value_channels_;
    bool structured_;
    float decay_exponent_low_;
    float decay_exponent_span_;
    std::mt19937 random_engine_;
    std::normal_distribution<float> normal_;
    std::uniform_real_distribution<float> uniform_;
    std::vector<float> key_mixing_;
    std::vector<float> value_mixing_;
    std::vector<float> key_channel_gain_;
    std::vector<float> value_channel_gain_;
    std::vector<float> slow_signal_;
};

// =============================================================================
// Test 1. fp16 conversion
// =============================================================================
static void test_half_conversion() {
    std::printf("[1] fp16 conversion\n");

    // Every finite fp16 bit pattern: converting to float must give the value
    // the format defines, and converting back must give the same bits.
    for (std::uint32_t pattern = 0; pattern < 65536; ++pattern) {
        const int sign = static_cast<int>((pattern >> 15) & 1u);
        const int exponent = static_cast<int>((pattern >> 10) & 31u);
        const int fraction = static_cast<int>(pattern & 1023u);
        if (exponent == 31) {
            continue;  // infinity and not-a-number are never stored by the core
        }
        double defined_value = 0.0;
        if (exponent == 0) {
            defined_value = std::ldexp(static_cast<double>(fraction), -24);
        } else {
            defined_value = std::ldexp(1024.0 + fraction, exponent - 25);
        }
        if (sign == 1) {
            defined_value = -defined_value;
        }
        const float converted = convert_half_to_float(static_cast<std::uint16_t>(pattern));
        check(static_cast<double>(converted) == defined_value, "half -> float gives the defined value", converted);
        check(convert_float_to_half(converted) == static_cast<std::uint16_t>(pattern), "half -> float -> half is the identity", pattern);
    }

#if defined(__FLT16_MANT_DIG__)
    // Two million random floats, including exact ties, against the compiler's own fp16.
    std::mt19937 random_engine(1);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    std::uniform_int_distribution<int> exponent_choice(-30, 15);
    long compared = 0;
    for (int trial = 0; trial < 2000000; ++trial) {
        float value = std::ldexp(uniform(random_engine), exponent_choice(random_engine));
        if (trial % 7 == 0) {
            // Move the value exactly half-way between two neighbouring fp16 values.
            const std::uint16_t lower_bits = convert_float_to_half(value);
            const float lower = convert_half_to_float(lower_bits);
            const float upper = convert_half_to_float(static_cast<std::uint16_t>(lower_bits + 1));
            if (std::isfinite(upper) && (lower_bits & 0x7C00u) != 0x7C00u) {
                value = 0.5f * (lower + upper);
            }
        }
        if (std::fabs(value) >= 65520.0f) {
            continue;  // the compiler gives infinity here; this code saturates on purpose
        }
        const _Float16 compiler_half = static_cast<_Float16>(value);
        std::uint16_t compiler_bits = 0;
        std::memcpy(&compiler_bits, &compiler_half, sizeof(compiler_bits));
        check(convert_float_to_half(value) == compiler_bits, "float -> half matches the compiler", value);
        compared = compared + 1;
    }
    std::printf("    63,488 finite fp16 values and %ld random floats match the compiler's fp16\n", compared);
#else
    std::printf("    63,488 finite fp16 values checked (this compiler has no _Float16 to compare against)\n");
#endif

    check(convert_half_to_float(convert_float_to_half(1.0e9f)) == 65504.0f, "a too-large value saturates at 65504", 0);

    // Why the decay is stored as a logarithm.
    const float decay_stored_directly = convert_half_to_float(convert_float_to_half(0.9999f));
    const float decay_stored_as_log = std::exp(convert_half_to_float(convert_float_to_half(std::log(0.9999f))));
    check(decay_stored_directly == 1.0f, "0.9999 stored directly in fp16 becomes 1.0", decay_stored_directly);
    check(std::fabs(decay_stored_as_log - 0.9999f) < 1e-6f, "0.9999 stored as a logarithm survives", decay_stored_as_log);
    std::printf("    decay 0.9999 stored directly reads back as %.4f; stored as a logarithm it reads back as %.6f\n",
                decay_stored_directly, decay_stored_as_log);
}

// =============================================================================
// Test 2. The model mappings of Table 4 reproduce the reference kernels.
//         Pure fp32 on both sides: no rounding involved.
// =============================================================================
static void test_model_mappings() {
    std::printf("[2] equation 1 with the Table 4 mapping equals the flash-linear-attention reference kernels\n");
    const int key_channels = 32;
    const int value_channels = 48;
    std::mt19937 random_engine(2);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

    std::vector<float> key(key_channels), query(key_channels), value(value_channels);
    std::vector<float> write_key(key_channels), read_vector(key_channels), alpha_per_channel(key_channels);
    std::vector<float> output_from_equation_1(value_channels), output_from_reference_kernel(value_channels);

    FullPrecisionHead gdn_by_equation_1(key_channels, value_channels);
    FullPrecisionHead gdn_by_reference_kernel(key_channels, value_channels);
    FullPrecisionHead kda_by_equation_1(key_channels, value_channels);
    FullPrecisionHead kda_by_reference_kernel(key_channels, value_channels);

    double worst_relative_difference = 0.0;
    for (int token = 0; token < 200; ++token) {
        for (int index = 0; index < key_channels; ++index) key[index] = normal(random_engine) / std::sqrt(static_cast<float>(key_channels));
        for (int index = 0; index < key_channels; ++index) query[index] = normal(random_engine);
        for (int index = 0; index < value_channels; ++index) value[index] = normal(random_engine);
        const float beta = uniform(random_engine);

        // Gated DeltaNet: decay = alpha, write_key = beta * key, read_vector = alpha * key.
        const float alpha = 0.9f + 0.1f * uniform(random_engine);
        for (int index = 0; index < key_channels; ++index) {
            write_key[index] = beta * key[index];
            read_vector[index] = alpha * key[index];
        }
        gdn_by_equation_1.process_one_token(&alpha, 1, write_key.data(), read_vector.data(), value.data(), query.data(),
                                            output_from_equation_1.data());
        gdn_by_reference_kernel.process_one_token_gated_delta_net(alpha, beta, key.data(), value.data(), query.data(),
                                                                  output_from_reference_kernel.data());
        worst_relative_difference = std::max(
            worst_relative_difference,
            length_of_difference(output_from_equation_1, output_from_reference_kernel) / length_of(output_from_reference_kernel));

        // KDA: decay = alpha per channel, write_key = beta * key, read_vector = alpha * key (per channel).
        for (int index = 0; index < key_channels; ++index) {
            alpha_per_channel[index] = 0.9f + 0.1f * uniform(random_engine);
            read_vector[index] = alpha_per_channel[index] * key[index];
        }
        kda_by_equation_1.process_one_token(alpha_per_channel.data(), key_channels, write_key.data(), read_vector.data(),
                                            value.data(), query.data(), output_from_equation_1.data());
        kda_by_reference_kernel.process_one_token_kda(alpha_per_channel.data(), beta, key.data(), value.data(), query.data(),
                                                      output_from_reference_kernel.data());
        worst_relative_difference = std::max(
            worst_relative_difference,
            length_of_difference(output_from_equation_1, output_from_reference_kernel) / length_of(output_from_reference_kernel));
    }
    check(worst_relative_difference < 1e-4, "outputs agree", worst_relative_difference);
    check(length_of_difference(gdn_by_equation_1.state, gdn_by_reference_kernel.state) / length_of(gdn_by_reference_kernel.state) < 1e-5,
          "Gated DeltaNet states agree", 0);
    check(length_of_difference(kda_by_equation_1.state, kda_by_reference_kernel.state) / length_of(kda_by_reference_kernel.state) < 1e-5,
          "KDA states agree", 0);
    std::printf("    worst relative output difference over 200 tokens: %.1e\n", worst_relative_difference);
}

// =============================================================================
// Test 3. The window bookkeeping, token by token, against plain matrix arithmetic.
//         The state is 24 x 40 (not square) so that a swapped index cannot hide.
// =============================================================================
static void test_window_bookkeeping(bool one_decay_per_token) {
    std::printf("[3] window bookkeeping, %s, 24 x 40 state\n",
                one_decay_per_token ? "one decay per token" : "one decay per key channel");
    Settings settings;
    settings.key_channels = 24;
    settings.value_channels = 40;
    settings.one_decay_per_token = one_decay_per_token;
    const int key_channels = settings.key_channels;
    const int value_channels = settings.value_channels;
    const int decay_count = one_decay_per_token ? 1 : key_channels;
    const std::size_t entry_count = static_cast<std::size_t>(key_channels) * value_channels;

    Head head(settings);
    std::mt19937 random_engine(3);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

    std::vector<float> starting_state(entry_count);
    for (std::size_t index = 0; index < entry_count; ++index) starting_state[index] = normal(random_engine);
    head.load_full_precision_state(starting_state.data());

    std::vector<float> state_before(entry_count), state_after(entry_count), expected_state(entry_count);
    std::vector<float> decay(decay_count), decay_as_stored(decay_count);
    std::vector<float> write_key(key_channels), write_key_as_stored(key_channels), read_vector(key_channels);
    std::vector<float> query(key_channels), probe(key_channels);
    std::vector<float> value(value_channels), output(value_channels), held(value_channels), correction_as_stored(value_channels);
    std::vector<float> expected_output(value_channels), read_by_head(value_channels), read_by_matrix(value_channels);

    double worst_output_error = 0.0;        // relative
    double worst_read_error = 0.0;          // relative
    double worst_update_error = 0.0;        // as a fraction of the allowed error
    double worst_fold_error = 0.0;          // as a fraction of the allowed error
    int folds_seen = 0;

    for (int token = 0; token < 200; ++token) {
        for (int index = 0; index < decay_count; ++index) decay[index] = 0.8f + 0.2f * uniform(random_engine);
        for (int index = 0; index < key_channels; ++index) write_key[index] = normal(random_engine) / std::sqrt(static_cast<float>(key_channels));
        for (int index = 0; index < key_channels; ++index) read_vector[index] = normal(random_engine) / std::sqrt(static_cast<float>(key_channels));
        for (int index = 0; index < key_channels; ++index) query[index] = normal(random_engine);
        for (int index = 0; index < key_channels; ++index) probe[index] = normal(random_engine);
        for (int index = 0; index < value_channels; ++index) value[index] = normal(random_engine);

        head.copy_state_to_full_precision(state_before.data());
        head.process_one_token(decay.data(), write_key.data(), read_vector.data(), value.data(), query.data(), output.data());
        head.copy_state_to_full_precision(state_after.data());

        // What equation 1 says the state must now be, using the numbers as the head stored them.
        for (int index = 0; index < decay_count; ++index) {
            decay_as_stored[index] = std::exp(convert_half_to_float(convert_float_to_half(std::log(decay[index]))));
        }
        for (int index = 0; index < key_channels; ++index) {
            write_key_as_stored[index] = convert_half_to_float(convert_float_to_half(write_key[index]));
        }
        for (int value_index = 0; value_index < value_channels; ++value_index) held[value_index] = 0.0f;
        for (int key_index = 0; key_index < key_channels; ++key_index) {
            for (int value_index = 0; value_index < value_channels; ++value_index) {
                held[value_index] += read_vector[key_index] * state_before[static_cast<std::size_t>(key_index) * value_channels + value_index];
            }
        }
        for (int value_index = 0; value_index < value_channels; ++value_index) {
            correction_as_stored[value_index] = convert_half_to_float(convert_float_to_half(value[value_index] - held[value_index]));
        }
        for (int key_index = 0; key_index < key_channels; ++key_index) {
            const float decay_value = one_decay_per_token ? decay_as_stored[0] : decay_as_stored[key_index];
            for (int value_index = 0; value_index < value_channels; ++value_index) {
                const std::size_t position = static_cast<std::size_t>(key_index) * value_channels + value_index;
                expected_state[position] = decay_value * state_before[position] + write_key_as_stored[key_index] * correction_as_stored[value_index];
            }
        }
        for (int value_index = 0; value_index < value_channels; ++value_index) expected_output[value_index] = 0.0f;
        for (int key_index = 0; key_index < key_channels; ++key_index) {
            for (int value_index = 0; value_index < value_channels; ++value_index) {
                expected_output[value_index] += query[key_index] * expected_state[static_cast<std::size_t>(key_index) * value_channels + value_index];
            }
        }

        // (a) The output must be the new state read with the query.
        worst_output_error = std::max(worst_output_error, length_of_difference(output, expected_output) / length_of(expected_output));

        if (head.tokens_in_window() != 0) {
            // (b) Inside a window the state must follow equation 1. The allowed
            //     error is one fp16 step of the correction (the head and this
            //     test compute `held` in a different order of additions).
            const double allowed = 1e-3 * largest_absolute_value(write_key_as_stored) * largest_absolute_value(correction_as_stored) +
                                   1e-5 * largest_absolute_value(expected_state);
            double largest_difference = 0.0;
            for (std::size_t index = 0; index < entry_count; ++index) {
                largest_difference = std::max(largest_difference, static_cast<double>(std::fabs(state_after[index] - expected_state[index])));
            }
            worst_update_error = std::max(worst_update_error, largest_difference / allowed);
        } else {
            // (c) The window was just folded. Every entry must be within half a
            //     rounding step of the exact state.
            folds_seen = folds_seen + 1;
            for (int key_index = 0; key_index < key_channels; ++key_index) {
                for (int value_index = 0; value_index < value_channels; ++value_index) {
                    const std::size_t position = static_cast<std::size_t>(key_index) * value_channels + value_index;
                    const double half_a_step = 0.5 * head.key_row_scale(key_index) * head.value_channel_step(value_index);
                    const double allowed = half_a_step * 1.001 +
                                           1e-3 * std::fabs(write_key_as_stored[key_index] * correction_as_stored[value_index]) +
                                           1e-5 * largest_absolute_value(expected_state);
                    const double difference = std::fabs(state_after[position] - expected_state[position]);
                    worst_fold_error = std::max(worst_fold_error, difference / allowed);
                }
            }
        }

        // (d) read_state() must agree with reading the full matrix.
        head.read_state(probe.data(), read_by_head.data());
        for (int value_index = 0; value_index < value_channels; ++value_index) read_by_matrix[value_index] = 0.0f;
        for (int key_index = 0; key_index < key_channels; ++key_index) {
            for (int value_index = 0; value_index < value_channels; ++value_index) {
                read_by_matrix[value_index] += probe[key_index] * state_after[static_cast<std::size_t>(key_index) * value_channels + value_index];
            }
        }
        worst_read_error = std::max(worst_read_error, length_of_difference(read_by_head, read_by_matrix) / length_of(read_by_matrix));
    }

    check(folds_seen == 200 / settings.window_length, "one fold every window_length tokens", folds_seen);
    check(worst_output_error < 2e-3, "output equals new_state^T * query", worst_output_error);
    check(worst_update_error <= 1.0, "state inside a window follows equation 1", worst_update_error);
    check(worst_fold_error <= 1.0, "state after a fold is within half a rounding step", worst_fold_error);
    check(worst_read_error < 1e-4, "read_state() equals reading the full matrix", worst_read_error);
    std::printf("    200 tokens, %d folds. output error %.1e, read error %.1e (relative). "
                "update %.2f, fold %.2f (share of the allowed error)\n",
                folds_seen, worst_output_error, worst_read_error, worst_update_error, worst_fold_error);
}

// =============================================================================
// Test 4. Loading a state and reading it back: every entry within half a step.
// =============================================================================
static void test_rounding_error_bound() {
    std::printf("[4] load a state, read it back: every entry within half a rounding step (8, 6 and 4 bits)\n");
    const int key_channels = 96;
    const int value_channels = 128;
    const std::size_t entry_count = static_cast<std::size_t>(key_channels) * value_channels;
    std::mt19937 random_engine(4);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    // A state with rows and columns of very different size, plus a strong rank-1 part.
    std::vector<float> row_gain(key_channels), column_gain(value_channels), state(entry_count), read_back(entry_count);
    for (int index = 0; index < key_channels; ++index) row_gain[index] = std::exp(1.2f * normal(random_engine));
    for (int index = 0; index < value_channels; ++index) column_gain[index] = std::exp(1.2f * normal(random_engine));
    for (int key_index = 0; key_index < key_channels; ++key_index) {
        for (int value_index = 0; value_index < value_channels; ++value_index) {
            state[static_cast<std::size_t>(key_index) * value_channels + value_index] =
                row_gain[key_index] * column_gain[value_index] * (normal(random_engine) + 3.0f);
        }
    }

    const int bit_choices[3] = {8, 6, 4};
    for (int choice = 0; choice < 3; ++choice) {
        for (int smoothing = 0; smoothing < 2; ++smoothing) {
            Settings settings;
            settings.key_channels = key_channels;
            settings.value_channels = value_channels;
            settings.bits_per_entry = bit_choices[choice];
            settings.use_smoothing = (smoothing == 1);
            Head head(settings);
            head.load_full_precision_state(state.data());
            head.copy_state_to_full_precision(read_back.data());

            double worst = 0.0;
            for (int key_index = 0; key_index < key_channels; ++key_index) {
                for (int value_index = 0; value_index < value_channels; ++value_index) {
                    const std::size_t position = static_cast<std::size_t>(key_index) * value_channels + value_index;
                    const double allowed = 0.5 * head.key_row_scale(key_index) * head.value_channel_step(value_index) * 1.001 +
                                           1e-5 * largest_absolute_value(state);
                    worst = std::max(worst, std::fabs(state[position] - read_back[position]) / allowed);
                }
            }
            check(worst <= 1.0, "rounding error within half a step", worst);
            check(head.tokens_in_window() == 0, "window is empty after loading", head.tokens_in_window());
        }
    }

    // Groups of 8 at 6 and 4 bits, with no compensators. The one-argument step
    // is only group 0, so the bound has to use the group that owns the row.
    const int grouped_bits[2] = {6, 4};
    for (int choice = 0; choice < 2; ++choice) {
        Settings grouped;
        grouped.key_channels = key_channels;
        grouped.value_channels = value_channels;
        grouped.bits_per_entry = grouped_bits[choice];
        grouped.compensator_count = 0;
        grouped.scale_group = 8;
        Head grouped_head(grouped);
        grouped_head.load_full_precision_state(state.data());
        grouped_head.copy_state_to_full_precision(read_back.data());
        const int group_length = leapquant::scale_group_length_of(grouped);
        double worst = 0.0;
        for (int key_index = 0; key_index < key_channels; ++key_index) {
            const int group = key_index / group_length;
            for (int value_index = 0; value_index < value_channels; ++value_index) {
                const std::size_t position = static_cast<std::size_t>(key_index) * value_channels + value_index;
                const double allowed = 0.5 * grouped_head.key_row_scale(key_index) *
                                           grouped_head.value_channel_step(value_index, group) * 1.001 +
                                       1e-5 * largest_absolute_value(state);
                worst = std::max(worst, std::fabs(state[position] - read_back[position]) / allowed);
            }
        }
        check(worst <= 1.0, "grouped rounding stays within half a step of that group", worst);
    }

    // An all-zero state must stay all zero and must not produce not-a-number.
    Settings settings;
    settings.key_channels = key_channels;
    settings.value_channels = value_channels;
    Head head(settings);
    std::fill(state.begin(), state.end(), 0.0f);
    head.load_full_precision_state(state.data());
    head.copy_state_to_full_precision(read_back.data());
    check(largest_absolute_value(read_back) == 0.0f, "a zero state stays zero", largest_absolute_value(read_back));
}

// =============================================================================
// Test 5. Stored bytes match the paper (section 4.3).
// =============================================================================
static void test_stored_bytes() {
    std::printf("[5] stored bytes\n");
    Settings gated_delta_net_settings;   // defaults: 128 x 128, window 16, 4 compensators, one decay per token
    Head gated_delta_net_head(gated_delta_net_settings);
    check(gated_delta_net_head.boundary_state_bytes() == 19456, "boundary state is 19,456 bytes",
          static_cast<double>(gated_delta_net_head.boundary_state_bytes()));
    check(gated_delta_net_head.window_buffer_bytes() == 8224, "window buffer is 8,224 bytes",
          static_cast<double>(gated_delta_net_head.window_buffer_bytes()));

    Settings kda_settings;
    kda_settings.one_decay_per_token = false;
    Head kda_head(kda_settings);
    check(kda_head.window_buffer_bytes() == 12288, "KDA window buffer is 12,288 bytes",
          static_cast<double>(kda_head.window_buffer_bytes()));

    std::printf("    boundary state: %zu bytes = %.4f bytes per entry (paper: 1.19; fp32: 4)\n",
                gated_delta_net_head.boundary_state_bytes(),
                static_cast<double>(gated_delta_net_head.boundary_state_bytes()) / (128.0 * 128.0));
    std::printf("    window buffer : %zu bytes for Gated DeltaNet (paper: about 8 KiB), %zu bytes for KDA\n",
                gated_delta_net_head.window_buffer_bytes(), kda_head.window_buffer_bytes());
}

// =============================================================================
// Test 6. Do the compensators capture as much as any 4 patterns could?
// =============================================================================
static void test_compensator_quality() {
    std::printf("[6] compensators against the best possible 4 patterns (from the singular values)\n");
    const int key_channels = 128;
    const int value_channels = 128;
    const std::size_t entry_count = static_cast<std::size_t>(key_channels) * value_channels;

    // Build a realistic state by running 3,000 structured tokens through the fp32 head.
    SyntheticTokenStream stream(key_channels, value_channels, true, -3.0f, 1.7f, 6);
    FullPrecisionHead full_precision(key_channels, value_channels);
    std::vector<float> key(key_channels), query(key_channels), value(value_channels), output(value_channels);
    for (int token = 0; token < 3000; ++token) {
        float alpha = 0.0f;
        float beta = 0.0f;
        stream.next_token(alpha, beta, key.data(), value.data(), query.data());
        full_precision.process_one_token_gated_delta_net(alpha, beta, key.data(), value.data(), query.data(), output.data());
    }

    const std::vector<double> singular_values = singular_values_of(full_precision.state, key_channels, value_channels);
    double total_energy = 0.0;
    double best_possible_energy = 0.0;
    for (std::size_t index = 0; index < singular_values.size(); ++index) {
        total_energy += singular_values[index] * singular_values[index];
        if (index < 4) {
            best_possible_energy += singular_values[index] * singular_values[index];
        }
    }
    const double state_length = length_of(full_precision.state);
    check(std::fabs(total_energy - state_length * state_length) < 1e-6 * total_energy, "the singular-value oracle is consistent", 0);

    const int sweep_choices[4] = {2, 4, 8, 16};
    for (int choice = 0; choice < 4; ++choice) {
        Settings settings;
        settings.power_iteration_sweeps = sweep_choices[choice];
        Head head(settings);
        head.load_full_precision_state(full_precision.state.data());

        // What is left after subtracting the compensators.
        std::vector<float> residual = full_precision.state;
        for (int compensator = 0; compensator < settings.compensator_count; ++compensator) {
            for (int key_index = 0; key_index < key_channels; ++key_index) {
                for (int value_index = 0; value_index < value_channels; ++value_index) {
                    residual[static_cast<std::size_t>(key_index) * value_channels + value_index] -=
                        head.compensator_key(compensator, key_index) * head.compensator_correction(compensator, value_index);
                }
            }
        }
        const double residual_length = length_of(residual);
        const double captured_energy = total_energy - residual_length * residual_length;
        std::printf("    %2d sweeps: captured %.2f%% of the state energy; best possible %.2f%% (ratio %.4f)\n",
                    sweep_choices[choice], 100.0 * captured_energy / total_energy, 100.0 * best_possible_energy / total_energy,
                    captured_energy / best_possible_energy);
        if (sweep_choices[choice] == 8) {
            check(captured_energy / best_possible_energy > 0.98, "the default setting captures at least 98% of the optimum",
                  captured_energy / best_possible_energy);
        }
    }
    (void)entry_count;
}

// =============================================================================
// Test 7. Small checks that can be verified by hand, and bad settings.
// =============================================================================
static void test_small_cases() {
    std::printf("[7] hand-checkable case and rejected settings\n");

    // One key direction, beta = 1, query = key. The delta rule must store the
    // value exactly, so the output must be the value. Twice in a row.
    Settings settings;
    settings.key_channels = 8;
    settings.value_channels = 5;
    settings.compensator_count = 0;
    settings.window_length = 1;   // fold after every token: the hardest case for rounding
    Head head(settings);
    const float key[8] = {1, 0, 0, 0, 0, 0, 0, 0};
    const float value[5] = {1, 2, 3, 4, 5};
    float output[5] = {0, 0, 0, 0, 0};
    for (int repeat = 0; repeat < 2; ++repeat) {
        leapquant::process_one_token_gated_delta_net(head, 0.9f, 1.0f, key, value, key, output);
        for (int index = 0; index < 5; ++index) {
            check(std::fabs(output[index] - value[index]) < 1e-3f, "output equals the stored value", output[index]);
        }
    }

    // Settings out of range must be rejected with an exception.
    bool rejected = false;
    try {
        Settings bad_settings;
        bad_settings.bits_per_entry = 1;
        Head bad_head(bad_settings);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "bits_per_entry = 1 is rejected", 0);
}

// =============================================================================
// Test 8. Comparison of storage methods on a synthetic token stream.
//         This is the ablation of the paper (Table 2) at the level of one head.
// =============================================================================
static void compare_storage_methods(int token_count,
                                    const char* description,
                                    bool structured,
                                    float decay_exponent_low,
                                    float decay_exponent_span) {
    const int key_channels = 128;
    const int value_channels = 128;
    const std::size_t entry_count = static_cast<std::size_t>(key_channels) * value_channels;

    // The methods, in the order they are printed.
    enum MethodKind { kFullPrecision, kBfloat16EveryToken, kBfloat16EveryWindow, kInt8EveryToken, kLeapQuantHead };
    struct Method {
        std::string name;
        MethodKind kind;
        Settings settings;         // used only by kLeapQuantHead
        double bytes_per_entry;
    };
    std::vector<Method> methods;

    methods.push_back({"fp32 (reference)", kFullPrecision, Settings(), 4.0});
    methods.push_back({"bf16, rounded every token", kBfloat16EveryToken, Settings(), 2.0});
    methods.push_back({"bf16, rounded every 16 tokens", kBfloat16EveryWindow, Settings(), 2.0});
    methods.push_back({"int8, rounded every token", kInt8EveryToken, Settings(), 1.0 + 4.0 / key_channels});
    const std::size_t index_of_int8_every_token = methods.size() - 1;

    Settings window_only;
    window_only.compensator_count = 0;
    window_only.use_smoothing = false;
    methods.push_back({"int8, rounded every 16 tokens", kLeapQuantHead, window_only, 1.0 + 4.0 / key_channels});
    const std::size_t index_of_window_only = methods.size() - 1;

    Settings with_compensators;
    with_compensators.use_smoothing = false;
    methods.push_back({"  + 4 compensators", kLeapQuantHead, with_compensators,
                       (16384.0 + 4.0 * value_channels + 2.0 * 4 * (key_channels + value_channels)) / entry_count});

    Settings full_method;
    methods.push_back({"  + smoothing  (= LeapQuant)", kLeapQuantHead, full_method,
                       static_cast<double>(Head(full_method).boundary_state_bytes()) / entry_count});
    const std::size_t index_of_leapquant = methods.size() - 1;

    const std::size_t method_count = methods.size();
    std::vector<FullPrecisionHead> full_precision_heads;
    std::vector<Head> leapquant_heads;
    for (std::size_t method = 0; method < method_count; ++method) {
        full_precision_heads.push_back(FullPrecisionHead(key_channels, value_channels));
        leapquant_heads.push_back(Head(methods[method].settings));
    }

    // The state error is reported after 1,024, 4,096, ... tokens and at the end.
    std::vector<int> report_points;
    for (int point = 1024; point < token_count; point = point * 4) {
        report_points.push_back(point);
    }
    report_points.push_back(token_count);

    SyntheticTokenStream stream(key_channels, value_channels, structured, decay_exponent_low, decay_exponent_span, 7);
    std::vector<float> key(key_channels), query(key_channels), value(value_channels);
    std::vector<float> output(value_channels), reference_output(value_channels), state_copy(entry_count);
    std::vector<double> squared_output_error(method_count, 0.0);
    double squared_reference_output = 0.0;
    std::vector<std::vector<double> > state_error(method_count);
    std::size_t next_report_point = 0;

    for (int token = 1; token <= token_count; ++token) {
        float alpha = 0.0f;
        float beta = 0.0f;
        stream.next_token(alpha, beta, key.data(), value.data(), query.data());

        for (std::size_t method = 0; method < method_count; ++method) {
            if (methods[method].kind == kLeapQuantHead) {
                leapquant::process_one_token_gated_delta_net(leapquant_heads[method], alpha, beta, key.data(), value.data(),
                                                             query.data(), output.data());
            } else {
                FullPrecisionHead& head = full_precision_heads[method];
                head.process_one_token_gated_delta_net(alpha, beta, key.data(), value.data(), query.data(), output.data());
                if (methods[method].kind == kBfloat16EveryToken) {
                    leapquant::round_trip_through_bfloat16(head.state);
                }
                if (methods[method].kind == kBfloat16EveryWindow && token % 16 == 0) {
                    leapquant::round_trip_through_bfloat16(head.state);
                }
                if (methods[method].kind == kInt8EveryToken) {
                    leapquant::round_trip_through_int8_per_value_channel(head.state, key_channels, value_channels);
                }
            }

            if (method == 0) {
                reference_output = output;
                const double length = length_of(reference_output);
                squared_reference_output += length * length;
            } else {
                const double difference = length_of_difference(output, reference_output);
                squared_output_error[method] += difference * difference;
            }
        }

        if (next_report_point < report_points.size() && token == report_points[next_report_point]) {
            for (std::size_t method = 1; method < method_count; ++method) {
                if (methods[method].kind == kLeapQuantHead) {
                    leapquant_heads[method].copy_state_to_full_precision(state_copy.data());
                } else {
                    state_copy = full_precision_heads[method].state;
                }
                state_error[method].push_back(length_of_difference(state_copy, full_precision_heads[0].state) /
                                              length_of(full_precision_heads[0].state));
            }
            next_report_point = next_report_point + 1;
        }
    }

    // Describe the reference state, so the reader can judge how realistic the inputs are.
    const std::vector<double> singular_values = singular_values_of(full_precision_heads[0].state, key_channels, value_channels);
    double total_energy = 0.0;
    double top_four_energy = 0.0;
    for (std::size_t index = 0; index < singular_values.size(); ++index) {
        total_energy += singular_values[index] * singular_values[index];
        if (index < 4) top_four_energy += singular_values[index] * singular_values[index];
    }
    std::vector<float> magnitudes(entry_count);
    for (std::size_t index = 0; index < entry_count; ++index) magnitudes[index] = std::fabs(full_precision_heads[0].state[index]);
    std::nth_element(magnitudes.begin(), magnitudes.begin() + static_cast<long>(entry_count / 2), magnitudes.end());
    const float median_magnitude = magnitudes[entry_count / 2];

    std::printf("\n[8] Storage methods, one Gated DeltaNet head, %d x %d, %d tokens. %s\n", key_channels, value_channels,
                token_count, description);
    std::printf("    fp32 state at the end: the top 4 patterns hold %.0f%% of its energy; largest entry = %.0fx the median entry\n",
                100.0 * top_four_energy / total_energy, largest_absolute_value(full_precision_heads[0].state) / median_magnitude);
    std::printf("    %-30s %6s | state error after N tokens", "", "bytes");
    for (std::size_t point = 3; point < report_points.size(); ++point) std::printf("%9s", "");
    std::printf(" | output\n");
    std::printf("    %-30s %6s |", "method", "/entry");
    for (std::size_t point = 0; point < report_points.size(); ++point) std::printf(" %8d", report_points[point]);
    std::printf(" | error\n");
    for (std::size_t method = 0; method < method_count; ++method) {
        std::printf("    %-30s %6.2f |", methods[method].name.c_str(), methods[method].bytes_per_entry);
        for (std::size_t point = 0; point < report_points.size(); ++point) {
            if (method == 0) {
                std::printf(" %8s", "0");
            } else {
                std::printf(" %7.3f%%", 100.0 * state_error[method][point]);
            }
        }
        if (method == 0) {
            std::printf(" | %7s\n", "0");
        } else {
            std::printf(" | %6.3f%%\n", 100.0 * std::sqrt(squared_output_error[method] / squared_reference_output));
        }
    }
    std::printf("    (error = distance from the fp32 state or fp32 outputs, as a share of their size)\n");

    if (structured) {
        const double error_int8_every_token = std::sqrt(squared_output_error[index_of_int8_every_token] / squared_reference_output);
        const double error_window_only = std::sqrt(squared_output_error[index_of_window_only] / squared_reference_output);
        const double error_leapquant = std::sqrt(squared_output_error[index_of_leapquant] / squared_reference_output);
        check(error_window_only < error_int8_every_token, "rounding every 16 tokens beats rounding every token", error_window_only);
        check(error_leapquant < error_window_only, "full LeapQuant beats the window alone", error_leapquant);
        check(error_leapquant < 0.01, "full LeapQuant output error is below 1%", error_leapquant);
    }
}

// -----------------------------------------------------------------------------
// [9] A weakness of the one-at-a-time compensator search, and the joint fit that removes it.
//
// Found by an external review (13 November 2026). The power iteration starts
// from the longest row of the matrix. The comment in leapquant.cpp said that row
// "always contains some of the strongest pattern". That is false. Here is a
// matrix where it contains none of it:
//     4 rows with a single entry 2.0 (on the diagonal, columns 0..3)
//     124 rows with a single entry 1.5, all in column 4
// The strongest pattern is column 4 (strength 1.5 * sqrt(124) = 16.7). The four
// longest rows are the diagonal ones (length 2), each exactly orthogonal to it.
// No number of sweeps finds a pattern the start has zero overlap with.
//
// The joint fit (Settings::fit_compensators_together, the default) adds a rescue
// vector, the energy of every column, which cannot be perpendicular to a pattern
// that carries most of a column. It must capture at least 99% of the best possible.
// -----------------------------------------------------------------------------
static void test_compensator_seed_weakness() {
    std::printf("\n[9] A matrix whose strongest pattern the longest-row start misses\n");
    leapquant::Settings settings;
    const int size = 128;
    std::vector<float> matrix(static_cast<std::size_t>(size) * size, 0.0f);
    for (int row = 0; row < 4; ++row) {
        matrix[static_cast<std::size_t>(row) * size + row] = 2.0f;
    }
    for (int row = 4; row < size; ++row) {
        matrix[static_cast<std::size_t>(row) * size + 4] = 1.5f;
    }
    for (int method = 0; method < 2; ++method) {
    settings.fit_compensators_together = (method == 1);
    leapquant::Head head(settings);
    head.load_full_precision_state(matrix.data());

    // Energy of the matrix minus energy of (matrix - the four stored compensators).
    double energy = 0.0, energy_left = 0.0;
    for (int row = 0; row < size; ++row) {
        for (int column = 0; column < size; ++column) {
            double entry = matrix[static_cast<std::size_t>(row) * size + column];
            energy += entry * entry;
            for (int compensator = 0; compensator < settings.compensator_count; ++compensator) {
                entry -= static_cast<double>(head.compensator_key(compensator, row)) *
                         head.compensator_correction(compensator, column);
            }
            energy_left += entry * entry;
        }
    }
    // The best four: column 4 (1.5^2 * 124 = 279) and three of the four 2.0 entries (3 * 4 = 12).
    const double best_possible = 279.0 + 12.0;
    const double captured = energy - energy_left;
    std::printf("    %-34s energy captured: %.1f of a best possible %.1f (%.1f%%)\n",
                method == 1 ? "all together, with a rescue vector:" : "one at a time, from the longest row:", captured,
                best_possible, 100.0 * captured / best_possible);
    if (method == 1) {
        check(captured >= 0.99 * best_possible, "the joint fit finds the dominant pattern the old start misses",
              captured / best_possible);
    }
    // What must hold even here: the stored state is still a valid rounding of the matrix.
    std::vector<float> stored(matrix.size());
    head.copy_state_to_full_precision(stored.data());
    double worst = 0.0;
    for (std::size_t index = 0; index < matrix.size(); ++index) {
        worst = std::max(worst, std::fabs(static_cast<double>(stored[index]) - matrix[index]));
    }
    check(worst < 0.02, "with either method, every stored entry is within a rounding step", worst);
    }
}

// A heavy-tailed state: rank 8, then two spikes per column. One step per column
// lets one spike set the step, and 6-bit and 4-bit then drift. Rank 6 with a
// step per 8 key rows is what this file ships for those widths.
static std::vector<float> spike_state(int key_channels, int value_channels, unsigned seed) {
    std::mt19937 engine(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> state(static_cast<std::size_t>(key_channels) * value_channels, 0.0f);
    for (int pattern = 0; pattern < 8; ++pattern) {
        std::vector<float> left(key_channels), right(value_channels);
        for (float& entry : left) {
            entry = normal(engine);
        }
        for (float& entry : right) {
            entry = normal(engine);
        }
        const float strength = 4.0f / std::sqrt(static_cast<float>(pattern + 1));
        for (int key_index = 0; key_index < key_channels; ++key_index) {
            for (int value_index = 0; value_index < value_channels; ++value_index) {
                state[static_cast<std::size_t>(key_index) * value_channels + value_index] +=
                    strength * left[key_index] * right[value_index];
            }
        }
    }
    for (int key_index = 0; key_index < key_channels; ++key_index) {
        const float gain = std::exp(1.4f * normal(engine));
        for (int value_index = 0; value_index < value_channels; ++value_index) {
            state[static_cast<std::size_t>(key_index) * value_channels + value_index] *= gain;
        }
    }
    for (int value_index = 0; value_index < value_channels; ++value_index) {
        for (int spike = 0; spike < 2; ++spike) {
            const int row = (value_index * 3 + spike * 17) % key_channels;
            state[static_cast<std::size_t>(row) * value_channels + value_index] +=
                (spike == 0 ? 60.0f : -45.0f) * (0.5f + std::fabs(normal(engine)));
        }
    }
    return state;
}

struct SpikeError {
    double output = 0.0;
    double state = 0.0;
};

static SpikeError replay_spike(int bits, int rank, int group, int tokens, bool rotate = false) {
    const int key_channels = 64;
    const int value_channels = 64;
    Settings settings;
    settings.key_channels = key_channels;
    settings.value_channels = value_channels;
    settings.bits_per_entry = bits;
    settings.compensator_count = rank;
    settings.rotate_keys = rotate;
    settings.power_iteration_sweeps = 2;
    settings.scale_group = group;
    Head head(settings);
    FullPrecisionHead full(key_channels, value_channels);
    const std::vector<float> initial = spike_state(key_channels, value_channels, 4);
    head.load_full_precision_state(initial.data());
    full.load_state(initial.data());

    std::mt19937 engine(2);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    const int signal = 8;
    std::vector<float> mix(static_cast<std::size_t>(key_channels) * signal);
    std::vector<float> slow(signal);
    for (float& entry : mix) {
        entry = normal(engine);
    }
    for (float& entry : slow) {
        entry = normal(engine);
    }
    std::vector<float> key(key_channels), query(key_channels), value(value_channels);
    std::vector<float> output(value_channels), reference(value_channels);
    double squared_error = 0.0;
    double squared_reference = 0.0;
    for (int token = 0; token < tokens; ++token) {
        for (int index = 0; index < signal; ++index) {
            slow[index] = 0.98f * slow[index] + 0.2f * normal(engine);
        }
        auto unit = [&](float* vector) {
            double squared = 0.0;
            for (int key_index = 0; key_index < key_channels; ++key_index) {
                float entry = 0.2f * normal(engine);
                for (int component = 0; component < signal; ++component) {
                    entry += mix[static_cast<std::size_t>(key_index) * signal + component] * slow[component];
                }
                vector[key_index] = entry;
                squared += static_cast<double>(entry) * entry;
            }
            const float scale = 1.0f / static_cast<float>(std::sqrt(squared));
            for (int key_index = 0; key_index < key_channels; ++key_index) {
                vector[key_index] *= scale;
            }
        };
        unit(key.data());
        unit(query.data());
        for (float& entry : value) {
            entry = 0.25f * normal(engine);
        }
        const float alpha = 0.998f;
        const float beta = 0.15f + 0.25f * uniform(engine);
        full.process_one_token_gated_delta_net(alpha, beta, key.data(), value.data(), query.data(), reference.data());
        leapquant::process_one_token_gated_delta_net(head, alpha, beta, key.data(), value.data(), query.data(),
                                                     output.data());
        for (int index = 0; index < value_channels; ++index) {
            const double gap = static_cast<double>(output[index]) - reference[index];
            squared_error += gap * gap;
            squared_reference += static_cast<double>(reference[index]) * reference[index];
        }
    }
    std::vector<float> copied(initial.size());
    head.copy_state_to_full_precision(copied.data());
    double state_gap = 0.0;
    double state_mag = 0.0;
    for (std::size_t index = 0; index < copied.size(); ++index) {
        const double gap = static_cast<double>(copied[index]) - full.state[index];
        state_gap += gap * gap;
        state_mag += static_cast<double>(full.state[index]) * full.state[index];
    }
    SpikeError error;
    error.output = std::sqrt(squared_error / squared_reference);
    error.state = std::sqrt(state_gap / state_mag);
    return error;
}

static void test_low_bit_accuracy() {
    std::printf("[10] 6-bit and 4-bit on a spiked state (768 tokens, 64 x 64)\n");
    check(leapquant::paper_compensator_count(8) == 4, "8-bit keeps 4 compensators", 4);
    check(leapquant::paper_compensator_count(6) == 6, "6-bit uses 6 compensators", 6);
    check(leapquant::paper_compensator_count(4) == 6, "4-bit uses 6 compensators", 6);
    check(leapquant::low_bit_scale_group(8) == 0, "8-bit keeps one step per column", 0);
    check(leapquant::low_bit_scale_group(6) == 8, "6-bit groups 8 key rows", 8);
    check(leapquant::low_bit_scale_group(4) == 8, "4-bit groups 8 key rows", 8);

    const int tokens = 768;
    const SpikeError eight = replay_spike(8, 4, 0, tokens);
    const SpikeError six_old = replay_spike(6, 4, 0, tokens);
    const SpikeError six = replay_spike(6, 6, 8, tokens);
    const SpikeError four_old = replay_spike(4, 4, 0, tokens);
    const SpikeError four = replay_spike(4, 6, 8, tokens);
    std::printf("    output %%   8-bit r4 %.3f   6-bit r4 %.3f   6-bit r6 g8 %.3f   4-bit r4 %.3f   4-bit r6 g8 %.3f\n",
                100.0 * eight.output, 100.0 * six_old.output, 100.0 * six.output, 100.0 * four_old.output,
                100.0 * four.output);
    std::printf("    state %%    8-bit r4 %.3f   6-bit r4 %.3f   6-bit r6 g8 %.3f   4-bit r4 %.3f   4-bit r6 g8 %.3f\n",
                100.0 * eight.state, 100.0 * six_old.state, 100.0 * six.state, 100.0 * four_old.state,
                100.0 * four.state);
    // Measured on this stream: 6-bit rank 4 is about 5x the 8-bit output error;
    // rank 6 with groups of 8 is about 1.5x. 4-bit rank 4 is about 30% and still
    // growing; rank 6 with groups of 8 stays near 8% output.
    check(six.output < 2.0 * eight.output, "6-bit with rank 6 and group 8 stays near the 8-bit error",
          six.output / eight.output);
    check(six_old.output > 3.5 * eight.output, "6-bit rank 4 with one step per column is several times worse",
          six_old.output / eight.output);
    check(four.output < 0.12, "4-bit with rank 6 and group 8 stays under 12% output error", four.output);
    check(four_old.output > 0.20, "4-bit rank 4 with one step per column breaks down", four_old.output);
    check(four.state < 0.30, "4-bit grouped state error stays under 30%", four.state);
    check(four_old.state > 0.40, "4-bit one-step state error is already large", four_old.state);
}

// [11] The rotation option (state-cache plan v2, task TR.1). It must be invisible to
// every caller: states go in and come out in the original basis, reads see the
// same matrix. Whether it lowers the error is measured on the recorded traces
// (runbook/rotation_experiment.sh); here only that it works and is not worse by much.
static void test_rotation_option() {
    std::printf("[11] Hadamard rotation of keys and state (option rotate_keys)\n");
    const int key_channels = 64;
    const int value_channels = 64;
    Settings settings;
    settings.key_channels = key_channels;
    settings.value_channels = value_channels;
    settings.power_iteration_sweeps = 2;
    settings.rotate_keys = true;
    Head head(settings);
    const std::vector<float> initial = spike_state(key_channels, value_channels, 4);
    head.load_full_precision_state(initial.data());
    std::vector<float> copied(initial.size());
    head.copy_state_to_full_precision(copied.data());
    double gap = 0.0;
    double magnitude = 0.0;
    for (std::size_t index = 0; index < initial.size(); ++index) {
        gap += (static_cast<double>(copied[index]) - initial[index]) * (static_cast<double>(copied[index]) - initial[index]);
        magnitude += static_cast<double>(initial[index]) * initial[index];
    }
    const double stored_error = std::sqrt(gap / magnitude);
    check(stored_error < 0.02, "a rotated head returns the state it took over (original basis)", stored_error);

    // read_state(q) must equal the returned matrix read with q.
    std::mt19937 engine(5);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> probe(key_channels), by_head(value_channels);
    double worst_read = 0.0;
    for (int trial = 0; trial < 8; ++trial) {
        for (float& entry : probe) {
            entry = normal(engine);
        }
        head.read_state(probe.data(), by_head.data());
        for (int value_index = 0; value_index < value_channels; ++value_index) {
            double sum = 0.0;
            double size = 0.0;
            for (int key_index = 0; key_index < key_channels; ++key_index) {
                const double term = static_cast<double>(copied[static_cast<std::size_t>(key_index) * value_channels + value_index]) *
                                    probe[key_index];
                sum += term;
                size += std::fabs(term);
            }
            worst_read = std::max(worst_read, std::fabs(sum - by_head[value_index]) / std::max(size, 1e-30));
        }
    }
    check(worst_read < 1e-4, "read_state() of a rotated head equals reading the returned matrix", worst_read);

    bool rejected = false;
    try {
        Settings per_key = settings;
        per_key.one_decay_per_token = false;
        Head invalid(per_key);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "rotation is refused with one decay per key channel", rejected ? 1.0 : 0.0);

    const int tokens = 768;
    const SpikeError plain4 = replay_spike(8, 4, 0, tokens);
    const SpikeError rotated4 = replay_spike(8, 4, 0, tokens, true);
    const SpikeError plain2 = replay_spike(8, 2, 0, tokens);
    const SpikeError rotated2 = replay_spike(8, 2, 0, tokens, true);
    std::printf("    output %%   r4 %.3f   r4 rotated %.3f   r2 %.3f   r2 rotated %.3f\n", 100.0 * plain4.output,
                100.0 * rotated4.output, 100.0 * plain2.output, 100.0 * rotated2.output);
    std::printf("    state %%    r4 %.3f   r4 rotated %.3f   r2 %.3f   r2 rotated %.3f\n", 100.0 * plain4.state,
                100.0 * rotated4.state, 100.0 * plain2.state, 100.0 * rotated2.state);
    check(std::isfinite(rotated4.output) && rotated4.output < 2.0 * plain4.output + 0.005,
          "rotated 8-bit rank 4 is finite and not much worse than unrotated", rotated4.output / plain4.output);
}

int main(int argument_count, char** arguments) {
    int token_count = 16384;
    if (argument_count > 1) {
        token_count = std::atoi(arguments[1]);
    }

    test_half_conversion();
    test_model_mappings();
    test_window_bookkeeping(true);
    test_window_bookkeeping(false);
    test_rounding_error_bound();
    test_stored_bytes();
    test_compensator_quality();
    test_small_cases();
    test_compensator_seed_weakness();
    test_low_bit_accuracy();
    test_rotation_option();
    compare_storage_methods(token_count, "Structured inputs, long memory (decay 0.99999 to 0.999999).", true, -6.0f, 1.0f);
    compare_storage_methods(token_count, "Structured inputs, short memory (decay 0.95 to 0.999).", true, -3.0f, 1.7f);
    compare_storage_methods(token_count, "Plain random inputs, short memory (decay 0.95 to 0.999).", false, -3.0f, 1.7f);

    std::printf("\n%ld checks run, %ld failed\n", g_checks_run, g_checks_failed);
    return g_checks_failed == 0 ? 0 : 1;
}
