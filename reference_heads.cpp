// =============================================================================
// reference_heads.cpp - see reference_heads.h
// =============================================================================

#include "reference_heads.h"

#include <cmath>
#include <cstdint>
#include <cstring>

namespace leapquant {

FullPrecisionHead::FullPrecisionHead(int key_channels, int value_channels)
    : state(static_cast<std::size_t>(key_channels) * static_cast<std::size_t>(value_channels), 0.0f),
      key_channels_(key_channels),
      value_channels_(value_channels) {}

void FullPrecisionHead::load_state(const float* state_matrix) {
    for (std::size_t index = 0; index < state.size(); ++index) {
        state[index] = state_matrix[index];
    }
}

void FullPrecisionHead::read_state(const float* input_vector, float* result) const {
    for (int value_index = 0; value_index < value_channels_; ++value_index) {
        result[value_index] = 0.0f;
    }
    for (int key_index = 0; key_index < key_channels_; ++key_index) {
        const float input_value = input_vector[key_index];
        const float* state_row = state.data() + static_cast<std::size_t>(key_index) * value_channels_;
        for (int value_index = 0; value_index < value_channels_; ++value_index) {
            result[value_index] += input_value * state_row[value_index];
        }
    }
}

void FullPrecisionHead::process_one_token(const float* decay,
                                          int decay_value_count,
                                          const float* write_key,
                                          const float* read_vector,
                                          const float* value,
                                          const float* query,
                                          float* output) {
    // held = state^T * read_vector
    std::vector<float> held(static_cast<std::size_t>(value_channels_));
    read_state(read_vector, held.data());

    // state = Diag(decay) * state + write_key * (value - held)^T
    for (int key_index = 0; key_index < key_channels_; ++key_index) {
        float decay_value = decay[0];
        if (decay_value_count != 1) {
            decay_value = decay[key_index];
        }
        float* state_row = state.data() + static_cast<std::size_t>(key_index) * value_channels_;
        for (int value_index = 0; value_index < value_channels_; ++value_index) {
            state_row[value_index] =
                decay_value * state_row[value_index] + write_key[key_index] * (value[value_index] - held[value_index]);
        }
    }

    // output = state^T * query
    read_state(query, output);
}

void FullPrecisionHead::process_one_token_gated_delta_net(float alpha,
                                                          float beta,
                                                          const float* key,
                                                          const float* value,
                                                          const float* query,
                                                          float* output) {
    // h = h * exp(g)                       (alpha = exp(g))
    for (std::size_t index = 0; index < state.size(); ++index) {
        state[index] = state[index] * alpha;
    }

    // v_new = (v - h^T k) * beta
    std::vector<float> new_value(static_cast<std::size_t>(value_channels_));
    read_state(key, new_value.data());
    for (int value_index = 0; value_index < value_channels_; ++value_index) {
        new_value[value_index] = (value[value_index] - new_value[value_index]) * beta;
    }

    // h = h + k * v_new^T
    for (int key_index = 0; key_index < key_channels_; ++key_index) {
        float* state_row = state.data() + static_cast<std::size_t>(key_index) * value_channels_;
        for (int value_index = 0; value_index < value_channels_; ++value_index) {
            state_row[value_index] += key[key_index] * new_value[value_index];
        }
    }

    // o = h^T q
    read_state(query, output);
}

void FullPrecisionHead::process_one_token_kda(const float* alpha,
                                              float beta,
                                              const float* key,
                                              const float* value,
                                              const float* query,
                                              float* output) {
    // S = S * exp(g)   with one decay per key row
    for (int key_index = 0; key_index < key_channels_; ++key_index) {
        float* state_row = state.data() + static_cast<std::size_t>(key_index) * value_channels_;
        for (int value_index = 0; value_index < value_channels_; ++value_index) {
            state_row[value_index] = state_row[value_index] * alpha[key_index];
        }
    }

    // S = S + (beta * k) * (v - S^T k)^T
    std::vector<float> held(static_cast<std::size_t>(value_channels_));
    read_state(key, held.data());
    for (int key_index = 0; key_index < key_channels_; ++key_index) {
        float* state_row = state.data() + static_cast<std::size_t>(key_index) * value_channels_;
        for (int value_index = 0; value_index < value_channels_; ++value_index) {
            state_row[value_index] += beta * key[key_index] * (value[value_index] - held[value_index]);
        }
    }

    // o = S^T q
    read_state(query, output);
}

int FullPrecisionHead::key_channels() const {
    return key_channels_;
}

int FullPrecisionHead::value_channels() const {
    return value_channels_;
}

void round_trip_through_int8_per_value_channel(std::vector<float>& state, int key_channels, int value_channels) {
    for (int value_index = 0; value_index < value_channels; ++value_index) {
        float largest_absolute_entry = 0.0f;
        for (int key_index = 0; key_index < key_channels; ++key_index) {
            const float entry = state[static_cast<std::size_t>(key_index) * value_channels + value_index];
            if (std::fabs(entry) > largest_absolute_entry) {
                largest_absolute_entry = std::fabs(entry);
            }
        }
        if (largest_absolute_entry == 0.0f) {
            continue;
        }
        const float step = largest_absolute_entry / 127.0f;
        for (int key_index = 0; key_index < key_channels; ++key_index) {
            const std::size_t position = static_cast<std::size_t>(key_index) * value_channels + value_index;
            state[position] = std::nearbyint(state[position] / step) * step;
        }
    }
}

void round_trip_through_bfloat16(std::vector<float>& state) {
    for (std::size_t index = 0; index < state.size(); ++index) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &state[index], sizeof(bits));
        // Round to the nearest value whose low 16 bits are zero (ties to even).
        bits = bits + 0x7FFFu + ((bits >> 16) & 1u);
        bits = bits & 0xFFFF0000u;
        std::memcpy(&state[index], &bits, sizeof(bits));
    }
}

double length_of(const std::vector<float>& values) {
    double sum_of_squares = 0.0;
    for (std::size_t index = 0; index < values.size(); ++index) {
        sum_of_squares += static_cast<double>(values[index]) * static_cast<double>(values[index]);
    }
    return std::sqrt(sum_of_squares);
}

double length_of_difference(const std::vector<float>& first, const std::vector<float>& second) {
    double sum_of_squares = 0.0;
    for (std::size_t index = 0; index < first.size(); ++index) {
        const double difference = static_cast<double>(first[index]) - static_cast<double>(second[index]);
        sum_of_squares += difference * difference;
    }
    return std::sqrt(sum_of_squares);
}

float largest_absolute_value(const std::vector<float>& values) {
    float largest = 0.0f;
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (std::fabs(values[index]) > largest) {
            largest = std::fabs(values[index]);
        }
    }
    return largest;
}

}  // namespace leapquant
