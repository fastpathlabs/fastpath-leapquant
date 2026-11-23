// =============================================================================
// kernel_test.cpp
//
// Checks the batched kernel (CPU build) against the reference core.
//
// Build and run:   make kernel-test
//
// What is checked
//   [1] The kernel's fp16 conversion equals the reference core's on every value.
//   [2] Token by token, the kernel's outputs equal the reference core's outputs,
//       for several shapes, with requests that join at different times, pool
//       slots in scrambled order, padded rows and forced folds.
//   [3] After the run, the kernel's state equals the reference core's state.
//       Built with -ffp-contract=off -DLQ_EXPECT_BIT_IDENTICAL (as the Makefile
//       does), kernel and reference must agree bit for bit over the whole run.
//   [4] Run as an emulated GPU thread block (several real threads and a barrier
//       per block), the kernel gives the same bytes as run in one thread.
//   [5] Bad arguments are rejected.
//
// Run with --emulate-only under ThreadSanitizer (make kernel-race-check) to
// look for data races inside a block.
// =============================================================================

#include "../leapquant.h"
#include "../reference_heads.h"
#include "leapquant_kernel.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

long g_checks_run = 0;
long g_checks_failed = 0;

void check(bool condition, const std::string& what) {
    g_checks_run = g_checks_run + 1;
    if (!condition) {
        g_checks_failed = g_checks_failed + 1;
        if (g_checks_failed <= 20) {
            std::printf("    FAILED: %s\n", what.c_str());
        }
    }
}

// The memory of one state pool, owned by the test.
struct PoolMemory {
    std::vector<std::int8_t> rounded_residual;
    std::vector<float> value_channel_step;
    std::vector<float> key_row_scale;
    std::vector<std::uint16_t> compensator_keys;
    std::vector<std::uint16_t> compensator_corrections;
    std::vector<std::uint16_t> window_keys;
    std::vector<std::uint16_t> window_corrections;
    std::vector<std::uint16_t> window_log_decays;
    LqPool pool;
};

void make_pool(PoolMemory& memory, int slots, int value_heads, const leapquant::Settings& settings) {
    const std::size_t heads = static_cast<std::size_t>(slots) * value_heads;
    const std::size_t key_channels = static_cast<std::size_t>(settings.key_channels);
    const std::size_t value_channels = static_cast<std::size_t>(settings.value_channels);
    const std::size_t compensators = static_cast<std::size_t>(settings.compensator_count);
    const std::size_t window = static_cast<std::size_t>(settings.window_length);

    // The zero state: all levels 0, all steps 0, all row scales 1.
    // One step per column is value_channels floats. A scale group stores one
    // step per band of key rows, laid out [value][group].
    const int scale_groups = leapquant::scale_group_count_of(settings);
    memory.rounded_residual.assign(heads * value_channels * key_channels, 0);
    memory.value_channel_step.assign(heads * value_channels * static_cast<std::size_t>(scale_groups), 0.0f);
    memory.key_row_scale.assign(heads * key_channels, 1.0f);
    memory.compensator_keys.assign(heads * compensators * key_channels + 1, 0);
    memory.compensator_corrections.assign(heads * compensators * value_channels + 1, 0);
    memory.window_keys.assign(heads * window * key_channels, 0);
    memory.window_corrections.assign(heads * window * value_channels, 0);
    memory.window_log_decays.assign(heads * window, 0);

    LqPool& pool = memory.pool;
    pool.rounded_residual = memory.rounded_residual.data();
    pool.value_channel_step = memory.value_channel_step.data();
    pool.key_row_scale = memory.key_row_scale.data();
    pool.compensator_keys = memory.compensator_keys.data();
    pool.compensator_corrections = memory.compensator_corrections.data();
    pool.window_keys = memory.window_keys.data();
    pool.window_corrections = memory.window_corrections.data();
    pool.window_log_decays = memory.window_log_decays.data();
    pool.slot_count = slots;
    pool.value_heads = value_heads;
    pool.key_channels = settings.key_channels;
    pool.value_channels = settings.value_channels;
    pool.window_length = settings.window_length;
    pool.compensator_count = settings.compensator_count;
    pool.largest_integer_level = (1 << (settings.bits_per_entry - 1)) - 1;
    pool.power_iteration_sweeps = settings.power_iteration_sweeps;
    pool.use_smoothing = settings.use_smoothing ? 1 : 0;
    pool.fit_together = settings.fit_compensators_together ? 1 : 0;
    pool.profile_stop_phase = 0;
    pool.scale_group = settings.scale_group;
}

bool pools_hold_the_same_bytes(const PoolMemory& first, const PoolMemory& second) {
    return first.rounded_residual == second.rounded_residual &&
           first.compensator_keys == second.compensator_keys &&
           first.compensator_corrections == second.compensator_corrections &&
           first.window_keys == second.window_keys && first.window_corrections == second.window_corrections &&
           first.window_log_decays == second.window_log_decays &&
           std::memcmp(first.value_channel_step.data(), second.value_channel_step.data(),
                       first.value_channel_step.size() * sizeof(float)) == 0 &&
           std::memcmp(first.key_row_scale.data(), second.key_row_scale.data(),
                       first.key_row_scale.size() * sizeof(float)) == 0;
}

double length_of(const float* values, int count) {
    double sum = 0.0;
    for (int index = 0; index < count; ++index) {
        sum += static_cast<double>(values[index]) * values[index];
    }
    return std::sqrt(sum);
}

double length_of_difference(const float* first, const float* second, int count) {
    double sum = 0.0;
    for (int index = 0; index < count; ++index) {
        const double difference = static_cast<double>(first[index]) - second[index];
        sum += difference * difference;
    }
    return std::sqrt(sum);
}

struct Scenario {
    std::string name;
    int rows = 1;                 // requests decoded together
    int slots = 1;                // pool slots (>= rows)
    int value_heads = 2;
    int key_heads = 1;
    int tokens = 64;              // decode steps
    bool start_from_prefill_state = true;
    bool staggered_starts = false;   // row r joins at step 3 * r
    int forced_fold_every = 0;       // 0 = never; otherwise row r is forced every N + r steps
    leapquant::Settings settings;
};

struct ScenarioResult {
    // Kernel against a reference head that starts every window from the same state.
    double worst_output_difference = 0.0;
    double worst_state_difference = 0.0;
    long folds_compared = 0;
    long folds_not_identical = 0;
    // Kernel and reference, each running on its own, against the fp32 truth.
    double kernel_output_error = 0.0;
    double reference_output_error = 0.0;
    double worst_free_running_difference = 0.0;
    std::vector<float> all_outputs;   // every output of the run, for bit comparisons
    PoolMemory memory;
};

// Copies a [key][value] matrix into [value][key] order, or back (swap the two counts).
void transpose(const float* source, int source_rows, int source_columns, float* destination) {
    for (int row = 0; row < source_rows; ++row) {
        for (int column = 0; column < source_columns; ++column) {
            destination[static_cast<std::size_t>(column) * source_rows + row] =
                source[static_cast<std::size_t>(row) * source_columns + column];
        }
    }
}

// true if the boundary state of one kernel head holds exactly the numbers of a reference head.
bool boundary_states_are_identical(const LqPool& pool, int slot, int head, const leapquant::Head& reference_head) {
    const std::size_t head_index = static_cast<std::size_t>(slot) * pool.value_heads + head;
    const std::size_t key_channels = static_cast<std::size_t>(pool.key_channels);
    const std::size_t value_channels = static_cast<std::size_t>(pool.value_channels);
    const std::size_t compensators = static_cast<std::size_t>(pool.compensator_count);
    const int scale_groups = leapquant::scale_group_count_of(reference_head.settings());
    bool identical = true;
    for (std::size_t key_index = 0; key_index < key_channels; ++key_index) {
        identical = identical && pool.key_row_scale[head_index * key_channels + key_index] ==
                                     reference_head.key_row_scale(static_cast<int>(key_index));
    }
    for (std::size_t value_index = 0; value_index < value_channels; ++value_index) {
        for (int group = 0; group < scale_groups; ++group) {
            const std::size_t step_index =
                (head_index * value_channels + value_index) * static_cast<std::size_t>(scale_groups) +
                static_cast<std::size_t>(group);
            identical = identical && pool.value_channel_step[step_index] ==
                                         reference_head.value_channel_step(static_cast<int>(value_index), group);
        }
        for (std::size_t key_index = 0; key_index < key_channels; ++key_index) {
            identical = identical &&
                        pool.rounded_residual[(head_index * value_channels + value_index) * key_channels + key_index] ==
                            reference_head.rounded_residual(static_cast<int>(key_index), static_cast<int>(value_index));
        }
    }
    for (std::size_t compensator = 0; compensator < compensators; ++compensator) {
        for (std::size_t key_index = 0; key_index < key_channels; ++key_index) {
            identical = identical &&
                        lq_cpu_half_to_float(pool.compensator_keys[(head_index * compensators + compensator) * key_channels + key_index]) ==
                            reference_head.compensator_key(static_cast<int>(compensator), static_cast<int>(key_index));
        }
        for (std::size_t value_index = 0; value_index < value_channels; ++value_index) {
            identical = identical &&
                        lq_cpu_half_to_float(pool.compensator_corrections[(head_index * compensators + compensator) * value_channels + value_index]) ==
                            reference_head.compensator_correction(static_cast<int>(compensator), static_cast<int>(value_index));
        }
    }
    return identical;
}

// Runs one scenario through the kernel.
//
// If `compare_with_reference` is true, the same inputs also go through three
// sets of reference heads:
//   synchronised  A reference Head that is given the kernel's own full-precision
//                 state at every fold. Inside a window, kernel and reference then
//                 start from the same numbers, and their outputs must agree
//                 closely. At the fold, both turn the SAME matrix into a boundary
//                 state, and the two boundary states must be identical.
//   free-running  A reference Head that runs on its own from start to end.
//   truth         The fp32 state.
// Why not simply compare kernel and free-running reference? Because rounding
// amplifies: a difference in the last bit of one float sooner or later tips one
// entry to the other int8 level, and from then on the two runs hold different,
// equally valid roundings of the same state. They drift apart up to the size of
// the rounding error itself. What must hold instead is that both stay equally
// close to the truth.
void run_scenario(const Scenario& scenario, bool compare_with_reference, ScenarioResult& result) {
    const leapquant::Settings& settings = scenario.settings;
    const int rows = scenario.rows;
    const int value_heads = scenario.value_heads;
    const int key_heads = scenario.key_heads;
    const int key_channels = settings.key_channels;
    const int value_channels = settings.value_channels;
    const int window_length = settings.window_length;
    const int heads_per_key_head = value_heads / key_heads;
    const std::size_t entry_count = static_cast<std::size_t>(key_channels) * value_channels;

    std::mt19937 generator(12345);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

    make_pool(result.memory, scenario.slots, value_heads, settings);
    const LqPool* pool = &result.memory.pool;

    // Row r lives in pool slot (slots - 1 - r): not the identity, on purpose.
    std::vector<std::int32_t> slot_of_row(rows);
    for (int row = 0; row < rows; ++row) {
        slot_of_row[row] = scenario.slots - 1 - row;
    }

    std::vector<leapquant::Head> synchronised_heads, free_running_heads;
    std::vector<leapquant::FullPrecisionHead> truth_heads;
    for (int index = 0; index < rows * value_heads; ++index) {
        synchronised_heads.push_back(leapquant::Head(settings));
        free_running_heads.push_back(leapquant::Head(settings));
        truth_heads.push_back(leapquant::FullPrecisionHead(key_channels, value_channels));
    }

    std::vector<float> state_key_major(entry_count);   // [key][value], the reference layout
    std::vector<float> states_value_major(static_cast<std::size_t>(rows) * value_heads * entry_count);

    // ---- Starting state (use case UC3: prefill ends) -------------------------
    if (scenario.start_from_prefill_state) {
        for (int row = 0; row < rows; ++row) {
            for (int head = 0; head < value_heads; ++head) {
                // A state with a few strong patterns plus noise, and rows of very different size.
                for (std::size_t index = 0; index < entry_count; ++index) {
                    state_key_major[index] = 0.05f * normal(generator);
                }
                for (int pattern = 0; pattern < 3; ++pattern) {
                    std::vector<float> key_side(key_channels), value_side(value_channels);
                    for (int key_index = 0; key_index < key_channels; ++key_index) {
                        key_side[key_index] = normal(generator) * (key_index % 7 == 0 ? 4.0f : 0.5f);
                    }
                    for (int value_index = 0; value_index < value_channels; ++value_index) {
                        value_side[value_index] = normal(generator);
                    }
                    for (int key_index = 0; key_index < key_channels; ++key_index) {
                        for (int value_index = 0; value_index < value_channels; ++value_index) {
                            state_key_major[static_cast<std::size_t>(key_index) * value_channels + value_index] +=
                                key_side[key_index] * value_side[value_index] / static_cast<float>(pattern + 1);
                        }
                    }
                }
                if (compare_with_reference) {
                    synchronised_heads[row * value_heads + head].load_full_precision_state(state_key_major.data());
                    free_running_heads[row * value_heads + head].load_full_precision_state(state_key_major.data());
                    truth_heads[row * value_heads + head].load_state(state_key_major.data());
                }
                transpose(state_key_major.data(), key_channels, value_channels,
                          states_value_major.data() + (static_cast<std::size_t>(row) * value_heads + head) * entry_count);
            }
        }
        LqStore store;
        std::memset(&store, 0, sizeof(store));
        store.rows = rows;
        store.slot = slot_of_row.data();
        store.state = states_value_major.data();
        check(lq_cpu_store(pool, &store) == 0, scenario.name + ": lq_cpu_store returns 0");
        if (compare_with_reference) {
            for (int row = 0; row < rows; ++row) {
                for (int head = 0; head < value_heads; ++head) {
                    check(boundary_states_are_identical(*pool, slot_of_row[row], head, synchronised_heads[row * value_heads + head]),
                          scenario.name + ": stored prefill state is identical to the reference");
                }
            }
        }
    }

    // ---- Decode ------------------------------------------------------------------
    std::vector<std::int32_t> tokens_in_window(rows, 0), tokens_after_decode(rows, 0);
    std::vector<std::int32_t> step_slot(rows), force(rows);
    std::vector<float> beta(static_cast<std::size_t>(rows) * value_heads);
    std::vector<float> alpha(static_cast<std::size_t>(rows) * value_heads);
    std::vector<float> key(static_cast<std::size_t>(rows) * key_heads * key_channels);
    std::vector<float> query(static_cast<std::size_t>(rows) * key_heads * key_channels);
    std::vector<float> value(static_cast<std::size_t>(rows) * value_heads * value_channels);
    std::vector<float> output(static_cast<std::size_t>(rows) * value_heads * value_channels);
    std::vector<float> scratch(static_cast<std::size_t>(rows) * value_heads * entry_count);
    std::vector<float> state_before_fold(static_cast<std::size_t>(rows) * value_heads * entry_count);
    std::vector<float> synchronised_output(value_channels), free_running_output(value_channels), truth_output(value_channels);
    double kernel_squared_error = 0.0, reference_squared_error = 0.0, truth_squared_length = 0.0;

    for (int step_index = 0; step_index < scenario.tokens; ++step_index) {
        // Which rows decode in this step, and which are forced to fold after it.
        for (int row = 0; row < rows; ++row) {
            const bool has_joined = !scenario.staggered_starts || step_index >= 3 * row;
            step_slot[row] = has_joined ? slot_of_row[row] : -1;
            force[row] = 0;
            if (scenario.forced_fold_every > 0 && (step_index + 1) % (scenario.forced_fold_every + row) == 0) {
                force[row] = 1;
            }
        }
        // Random inputs, shaped like those of a model: unit keys, scaled unit queries.
        for (std::size_t index = 0; index < alpha.size(); ++index) {
            alpha[index] = 1.0f - std::pow(10.0f, -3.0f + 1.7f * uniform(generator));
            beta[index] = 0.2f + 0.8f * uniform(generator);
        }
        for (int vector_index = 0; vector_index < rows * key_heads; ++vector_index) {
            float* key_vector = key.data() + static_cast<std::size_t>(vector_index) * key_channels;
            float* query_vector = query.data() + static_cast<std::size_t>(vector_index) * key_channels;
            for (int key_index = 0; key_index < key_channels; ++key_index) {
                key_vector[key_index] = normal(generator);
                query_vector[key_index] = normal(generator);
            }
            const float key_length = static_cast<float>(length_of(key_vector, key_channels));
            const float query_length = static_cast<float>(length_of(query_vector, key_channels)) *
                                       std::sqrt(static_cast<float>(key_channels));
            for (int key_index = 0; key_index < key_channels; ++key_index) {
                key_vector[key_index] = key_vector[key_index] / key_length;
                query_vector[key_index] = query_vector[key_index] / query_length;
            }
        }
        for (std::size_t index = 0; index < value.size(); ++index) {
            value[index] = normal(generator);
        }
        for (std::size_t index = 0; index < output.size(); ++index) {
            output[index] = 123.0f;   // a marker: padded rows must overwrite it with zero
        }

        LqStep step;
        std::memset(&step, 0, sizeof(step));
        step.rows = rows;
        step.key_heads = key_heads;
        step.slot = step_slot.data();
        step.tokens_in_window = tokens_in_window.data();
        step.decay = alpha.data();
        step.beta = beta.data();
        step.key = key.data();
        step.query = query.data();
        step.value = value.data();
        step.output = output.data();
        check(lq_cpu_decode_step(pool, &step) == 0, scenario.name + ": lq_cpu_decode_step returns 0");

        // The kernel's full-precision state after this token, before any fold.
        if (compare_with_reference) {
            for (int row = 0; row < rows; ++row) {
                tokens_after_decode[row] = tokens_in_window[row] + 1;
            }
            LqExport before_fold;
            before_fold.rows = rows;
            before_fold.slot = step_slot.data();
            before_fold.tokens_in_window = tokens_after_decode.data();
            before_fold.state = state_before_fold.data();
            check(lq_cpu_export(pool, &before_fold) == 0, scenario.name + ": lq_cpu_export returns 0");
        }

        LqFold fold;
        std::memset(&fold, 0, sizeof(fold));
        fold.rows = rows;
        fold.tokens_added = 1;
        fold.slot = step_slot.data();
        fold.tokens_in_window = tokens_in_window.data();
        fold.force = scenario.forced_fold_every > 0 ? force.data() : nullptr;
        fold.scratch = scratch.data();
        check(lq_cpu_fold(pool, &fold) == 0, scenario.name + ": lq_cpu_fold returns 0");

        result.all_outputs.insert(result.all_outputs.end(), output.begin(), output.end());

        for (int row = 0; row < rows; ++row) {
            const float* row_output = output.data() + static_cast<std::size_t>(row) * value_heads * value_channels;
            if (step_slot[row] < 0) {
                check(length_of(row_output, value_heads * value_channels) == 0.0, scenario.name + ": padded row gives zeros");
                continue;
            }
            // The engine's cursor update.
            const int records = tokens_in_window[row] + 1;
            const bool folded = records >= window_length || (force[row] != 0 && scenario.forced_fold_every > 0);
            tokens_in_window[row] = folded ? 0 : records;

            if (!compare_with_reference) {
                continue;
            }
            for (int head = 0; head < value_heads; ++head) {
                const int head_index = row * value_heads + head;
                const int key_head = head / heads_per_key_head;
                const float head_alpha = alpha[head_index];
                const float head_beta = beta[head_index];
                const float* head_key = key.data() + (static_cast<std::size_t>(row) * key_heads + key_head) * key_channels;
                const float* head_query = query.data() + (static_cast<std::size_t>(row) * key_heads + key_head) * key_channels;
                const float* head_value = value.data() + static_cast<std::size_t>(head_index) * value_channels;
                const float* kernel_output = row_output + static_cast<std::size_t>(head) * value_channels;

                truth_heads[head_index].process_one_token_gated_delta_net(head_alpha, head_beta, head_key, head_value,
                                                                          head_query, truth_output.data());
                leapquant::process_one_token_gated_delta_net(free_running_heads[head_index], head_alpha, head_beta,
                                                             head_key, head_value, head_query, free_running_output.data());
                if (folded && free_running_heads[head_index].tokens_in_window() > 0) {
                    free_running_heads[head_index].fold_window_into_boundary_state();   // the forced fold
                }
                // Keep the head as it is before the token: if this token closes the window, the
                // head folds by itself and replaces its compensators. The kernel's fold started
                // its fit from the OLD compensators, and the comparison below must do the same.
                const leapquant::Head head_before_token = synchronised_heads[head_index];
                leapquant::process_one_token_gated_delta_net(synchronised_heads[head_index], head_alpha, head_beta,
                                                             head_key, head_value, head_query, synchronised_output.data());

                // Inside a window: the kernel's output against the synchronised reference.
                const double truth_length = length_of(truth_output.data(), value_channels);
                const double difference =
                    length_of_difference(kernel_output, synchronised_output.data(), value_channels) /
                    length_of(synchronised_output.data(), value_channels);
                if (difference > result.worst_output_difference) {
                    result.worst_output_difference = difference;
                }
                const double free_difference =
                    length_of_difference(kernel_output, free_running_output.data(), value_channels) /
                    length_of(free_running_output.data(), value_channels);
                if (free_difference > result.worst_free_running_difference) {
                    result.worst_free_running_difference = free_difference;
                }
                const double kernel_error = length_of_difference(kernel_output, truth_output.data(), value_channels);
                const double reference_error =
                    length_of_difference(free_running_output.data(), truth_output.data(), value_channels);
                kernel_squared_error += kernel_error * kernel_error;
                reference_squared_error += reference_error * reference_error;
                truth_squared_length += truth_length * truth_length;

                // At a fold: give the reference the matrix the kernel folded, and
                // require the two boundary states to hold the same numbers.
                if (folded) {
                    transpose(state_before_fold.data() + static_cast<std::size_t>(head_index) * entry_count,
                              value_channels, key_channels, state_key_major.data());
                    synchronised_heads[head_index] = head_before_token;
                    synchronised_heads[head_index].load_full_precision_state(state_key_major.data(), true);
                    result.folds_compared = result.folds_compared + 1;
                    if (!boundary_states_are_identical(*pool, slot_of_row[row], head, synchronised_heads[head_index])) {
                        result.folds_not_identical = result.folds_not_identical + 1;
                    }
                }
                check(synchronised_heads[head_index].tokens_in_window() == tokens_in_window[row],
                      scenario.name + ": cursor matches the reference");
            }
        }
    }

    // ---- State at the end ----------------------------------------------------------
    if (compare_with_reference) {
        result.kernel_output_error = std::sqrt(kernel_squared_error / truth_squared_length);
        result.reference_output_error = std::sqrt(reference_squared_error / truth_squared_length);

        std::vector<float> exported(static_cast<std::size_t>(rows) * value_heads * entry_count);
        std::vector<float> reference_state(entry_count);
        LqExport request;
        request.rows = rows;
        request.slot = slot_of_row.data();
        request.tokens_in_window = tokens_in_window.data();
        request.state = exported.data();
        check(lq_cpu_export(pool, &request) == 0, scenario.name + ": lq_cpu_export returns 0");
        for (int row = 0; row < rows; ++row) {
            for (int head = 0; head < value_heads; ++head) {
                synchronised_heads[row * value_heads + head].copy_state_to_full_precision(reference_state.data());
                transpose(exported.data() + (static_cast<std::size_t>(row) * value_heads + head) * entry_count,
                          value_channels, key_channels, state_key_major.data());
                const double difference =
                    length_of_difference(state_key_major.data(), reference_state.data(), static_cast<int>(entry_count)) /
                    length_of(reference_state.data(), static_cast<int>(entry_count));
                if (difference > result.worst_state_difference) {
                    result.worst_state_difference = difference;
                }
            }
        }
    }
}

std::vector<Scenario> make_scenarios() {
    std::vector<Scenario> scenarios;

    Scenario small;
    small.name = "small, odd sizes";
    small.rows = 1;
    small.slots = 3;
    small.value_heads = 2;
    small.key_heads = 1;
    small.tokens = 200;
    small.settings.key_channels = 24;
    small.settings.value_channels = 40;
    scenarios.push_back(small);

    Scenario zero_start = small;
    zero_start.name = "small, starts at zero";
    zero_start.start_from_prefill_state = false;
    zero_start.settings.key_channels = 40;
    zero_start.settings.value_channels = 24;
    scenarios.push_back(zero_start);

    Scenario batch;
    batch.name = "batch of 5, staggered, forced folds";
    batch.rows = 5;
    batch.slots = 9;
    batch.value_heads = 4;
    batch.key_heads = 2;
    batch.tokens = 150;
    batch.staggered_starts = true;
    batch.forced_fold_every = 11;
    batch.settings.key_channels = 32;
    batch.settings.value_channels = 48;
    scenarios.push_back(batch);

    Scenario ablation = batch;
    ablation.name = "window 8, no compensators, no smoothing, 4 bits";
    ablation.settings.window_length = 8;
    ablation.settings.compensator_count = 0;
    ablation.settings.use_smoothing = false;
    ablation.settings.bits_per_entry = 4;
    scenarios.push_back(ablation);

    Scenario model_shape;
    model_shape.name = "Qwen3.5-9B shape: 32 value heads, 16 key heads, 128 x 128";
    model_shape.rows = 2;
    model_shape.slots = 2;
    model_shape.value_heads = 32;
    model_shape.key_heads = 16;
    model_shape.tokens = 100;
    model_shape.staggered_starts = true;
    scenarios.push_back(model_shape);

    // Paper rank at 6 bits, with a step per 8 key rows. 24 key channels divides by 8.
    // The 4-bit ablation above keeps one step per column and rank 0, on purpose.
    Scenario grouped = small;
    grouped.name = "6-bit, 6 compensators, scale group 8";
    grouped.tokens = 80;
    grouped.settings.bits_per_entry = 6;
    grouped.settings.compensator_count = 6;
    grouped.settings.scale_group = 8;
    grouped.settings.power_iteration_sweeps = 2;
    scenarios.push_back(grouped);

    return scenarios;
}

void test_half_conversion() {
    std::printf("[1] fp16 conversion equals the reference core\n");
    for (std::uint32_t half_bits = 0; half_bits < 65536u; ++half_bits) {
        const float from_kernel = lq_cpu_half_to_float(static_cast<std::uint16_t>(half_bits));
        const float from_reference = leapquant::convert_half_to_float(static_cast<std::uint16_t>(half_bits));
        check(std::memcmp(&from_kernel, &from_reference, sizeof(float)) == 0, "half to float");
    }
    std::mt19937 generator(7);
    for (int trial = 0; trial < 2000000; ++trial) {
        const std::uint32_t float_bits = generator();
        float value = 0.0f;
        std::memcpy(&value, &float_bits, sizeof(value));
        check(lq_cpu_float_to_half(value) == leapquant::convert_float_to_half(value), "float to half");
    }
}

void test_against_reference() {
    std::printf("[2][3] kernel against the reference core\n");
    std::printf("    in a window : worst difference of one output from the reference, when both start the window\n");
    std::printf("                  from the same state (limit 1e-4, the stage D limit of the plan)\n");
    std::printf("    folds       : folds whose boundary state is not identical to the reference's / folds compared\n");
    std::printf("    vs fp32     : output error against the fp32 truth over the whole run, kernel | reference\n");
    std::printf("    free run    : worst difference when kernel and reference never synchronise (not a pass criterion)\n\n");
    std::printf("    %-58s %11s %12s %9s %9s %9s\n", "scenario", "in a window", "folds", "kernel", "reference", "free run");
    const std::vector<Scenario> scenarios = make_scenarios();
    for (std::size_t index = 0; index < scenarios.size(); ++index) {
        ScenarioResult result;
        run_scenario(scenarios[index], true, result);
        char folds[64];
        std::snprintf(folds, sizeof(folds), "%ld/%ld", result.folds_not_identical, result.folds_compared);
        std::printf("    %-58s %11.1e %12s %8.3f%% %8.3f%% %9.1e\n", scenarios[index].name.c_str(),
                    result.worst_output_difference, folds, 100.0 * result.kernel_output_error,
                    100.0 * result.reference_output_error, result.worst_free_running_difference);
        check(result.worst_output_difference < 1e-4, scenarios[index].name + ": outputs within 1e-4 inside a window");
        check(result.worst_state_difference < 1e-4, scenarios[index].name + ": final state within 1e-4");
        check(result.folds_compared > 0 && result.folds_not_identical == 0,
              scenarios[index].name + ": every fold is identical to the reference's");
        // Running freely, the kernel must be as close to the truth as the reference is.
        const double ratio = result.kernel_output_error / result.reference_output_error;
        check(ratio > 0.9 && ratio < 1.1, scenarios[index].name + ": kernel is as accurate as the reference");
#if defined(LQ_EXPECT_BIT_IDENTICAL)
        // Built without fused multiply-add, kernel and reference do the same
        // arithmetic in the same order, so nothing may differ at all.
        check(result.worst_free_running_difference == 0.0 && result.worst_output_difference == 0.0 &&
                  result.worst_state_difference == 0.0,
              scenarios[index].name + ": kernel is bit-identical to the reference over the whole run");
#endif
    }
}

void compare_emulated_thread_block(Scenario scenario) {
    scenario.tokens = 40;
    scenario.rows = 3;
    scenario.slots = 4;

    ScenarioResult one_thread;
    lq_cpu_set_threads_per_block(1);
    run_scenario(scenario, false, one_thread);

    const int thread_counts[3] = {5, 32, 128};   // fewer than, equal to and more than the channel counts
    for (int index = 0; index < 3; ++index) {
        ScenarioResult emulated;
        lq_cpu_set_threads_per_block(thread_counts[index]);
        lq_cpu_set_block_matrix(index != 0);     // also with the block's own matrix, as on a GPU
        run_scenario(scenario, false, emulated);
        lq_cpu_set_threads_per_block(1);
        lq_cpu_set_block_matrix(0);
        const bool same_outputs =
            one_thread.all_outputs.size() == emulated.all_outputs.size() &&
            std::memcmp(one_thread.all_outputs.data(), emulated.all_outputs.data(),
                        one_thread.all_outputs.size() * sizeof(float)) == 0;
        const bool same_pool = pools_hold_the_same_bytes(one_thread.memory, emulated.memory);
        check(same_outputs, scenario.name + ": emulated block outputs are bit-identical");
        check(same_pool, scenario.name + ": emulated block pool is byte-identical");
        std::printf("    %3d threads, %-42s %s\n", thread_counts[index], scenario.name.c_str(),
                    same_outputs && same_pool ? "identical" : "DIFFERENT");
    }
}

void test_emulated_thread_block() {
    std::printf("[4] emulated GPU thread block gives the same bytes as one thread\n");
    const std::vector<Scenario> scenarios = make_scenarios();
    compare_emulated_thread_block(scenarios[2]);   // batch, staggered, forced folds: one step per column
    compare_emulated_thread_block(scenarios.back());   // 6-bit, rank 6, scale group 8
}

void test_rejected_arguments() {
    std::printf("[5] bad arguments are rejected, sizes match the plan\n");
    leapquant::Settings settings;
    PoolMemory memory;
    make_pool(memory, 1, 32, settings);

    // Stored bytes per head: the figures of the implementation plan.
    const std::size_t boundary_bytes = memory.rounded_residual.size() + 4 * memory.value_channel_step.size() +
                                       4 * memory.key_row_scale.size() +
                                       2 * (memory.compensator_keys.size() - 1 + memory.compensator_corrections.size() - 1);
    const std::size_t window_bytes =
        2 * (memory.window_keys.size() + memory.window_corrections.size() + memory.window_log_decays.size());
    check(boundary_bytes / 32 == 19456, "boundary state is 19,456 bytes per head");
    check(window_bytes / 32 == 8224, "window buffer is 8,224 bytes per head");

    LqStep step;
    std::memset(&step, 0, sizeof(step));
    check(lq_cpu_decode_step(nullptr, &step) == LQ_ERROR_BAD_POOL, "null pool");
    check(lq_cpu_decode_step(&memory.pool, &step) == LQ_ERROR_BAD_ARGUMENT, "null step pointers");

    LqPool too_wide = memory.pool;
    too_wide.key_channels = LQ_MAX_CHANNELS + 1;
    check(lq_check_pool(&too_wide) == LQ_ERROR_BAD_POOL, "key_channels above the compile-time limit");

    LqPool bad_group = memory.pool;
    bad_group.scale_group = -1;
    check(lq_check_pool(&bad_group) == LQ_ERROR_BAD_POOL, "negative scale_group");

    std::int32_t slot = 0, cursor = 0;
    float one_float = 0.0f;
    step.rows = 1;
    step.key_heads = 5;   // 32 value heads are not a multiple of 5
    step.slot = &slot;
    step.tokens_in_window = &cursor;
    step.decay = step.beta = step.key = step.query = step.value = &one_float;
    step.output = &one_float;
    check(lq_cpu_decode_step(&memory.pool, &step) == LQ_ERROR_BAD_ARGUMENT, "key_heads does not divide value_heads");
}

}  // unnamed namespace

int main(int argument_count, char** arguments) {
    const bool emulate_only = argument_count > 1 && std::string(arguments[1]) == "--emulate-only";
    if (!emulate_only) {
        test_half_conversion();
        test_against_reference();
    }
    test_emulated_thread_block();
    if (!emulate_only) {
        test_rejected_arguments();
    }
    std::printf("\n%ld checks run, %ld failed\n", g_checks_run, g_checks_failed);
    return g_checks_failed == 0 ? 0 : 1;
}
