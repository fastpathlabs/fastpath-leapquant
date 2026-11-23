// =============================================================================
// leapquant_replay.cpp
//
// Replays a recorded trace of one linear-attention layer through every storage
// method and reports how far each one drifts from full precision.
//
// Why this tool exists
//   The synthetic test in leapquant_test.cpp cannot tell how LeapQuant behaves
//   on the states of a real model. This tool can: record the inputs a real
//   model feeds to its recurrent kernel (record_trace.py does that), then run
//   them through the reference core here. No GPU is needed for the replay.
//
// Build:   make leapquant_replay
// Run:     ./leapquant_replay trace_layer_00.bin
//          ./leapquant_replay trace_layer_00.bin --max-tokens 4096 --max-heads 8
//          ./leapquant_replay trace_layer_00.bin --window 16 --compensators 4 --bits 8
//          ./leapquant_replay trace_layer_00.bin --write-outputs outputs_layer_00.bin
//
// --write-outputs PATH
//   Also writes the output of the full LeapQuant method for every token and
//   head to PATH, so that another implementation can be compared with the
//   reference (kernel/verify_kernel.py --reference-outputs PATH). Format, all
//   little-endian:
//     8 bytes   the text LQOUTPUT
//     int32     head_count (the heads that were replayed)
//     int32     value_channels
//     int32     token_count
//     then token_count * head_count * value_channels float32
//
// Trace file format (all numbers little-endian)
//
//   Header, 32 bytes:
//     8 bytes   the text LQTRACE1
//     int32     key_channels
//     int32     value_channels
//     int32     head_count
//     int32     decay_values_per_head   1 = Gated DeltaNet, key_channels = KDA
//     int32     has_initial_state       0 or 1
//     int32     has_expected_output     0 or 1
//
//   If has_initial_state is 1:
//     head_count * key_channels * value_channels float32
//     The state of every head before the first token, row after row.
//
//   Then one block per token until the end of the file. A block holds, for
//   every head in order:
//     float32 x decay_values_per_head   decay (alpha), a number in (0, 1]
//     float32 x 1                       beta
//     float32 x key_channels            key,   exactly as the kernel uses it
//     float32 x value_channels          value
//     float32 x key_channels            query, exactly as the kernel uses it
//     float32 x value_channels          the model's own kernel output for this
//                                       token (only if has_expected_output is 1)
// =============================================================================

#include "leapquant.h"
#include "reference_heads.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using leapquant::FullPrecisionHead;
using leapquant::Head;
using leapquant::Settings;
using leapquant::length_of;
using leapquant::length_of_difference;

namespace {

struct TraceHeader {
    int key_channels = 0;
    int value_channels = 0;
    int head_count = 0;
    int decay_values_per_head = 0;
    int has_initial_state = 0;
    int has_expected_output = 0;
};

// Reads one int32 from the file. Returns false at the end of the file.
bool read_int32(std::ifstream& file, int& value) {
    std::int32_t raw = 0;
    file.read(reinterpret_cast<char*>(&raw), sizeof(raw));
    if (!file) {
        return false;
    }
    value = static_cast<int>(raw);
    return true;
}

// Reads `count` floats from the file. Returns false if the file ends first.
bool read_floats(std::ifstream& file, float* values, std::size_t count) {
    file.read(reinterpret_cast<char*>(values), static_cast<std::streamsize>(count * sizeof(float)));
    return static_cast<bool>(file);
}

bool read_header(std::ifstream& file, TraceHeader& header) {
    char magic[8];
    file.read(magic, 8);
    if (!file || std::memcmp(magic, "LQTRACE1", 8) != 0) {
        std::fprintf(stderr, "error: this is not a LeapQuant trace file (it does not start with LQTRACE1)\n");
        return false;
    }
    if (!read_int32(file, header.key_channels) || !read_int32(file, header.value_channels) ||
        !read_int32(file, header.head_count) || !read_int32(file, header.decay_values_per_head) ||
        !read_int32(file, header.has_initial_state) || !read_int32(file, header.has_expected_output)) {
        std::fprintf(stderr, "error: the trace file ends inside its header\n");
        return false;
    }
    const bool decay_count_is_valid =
        header.decay_values_per_head == 1 || header.decay_values_per_head == header.key_channels;
    if (header.key_channels < 1 || header.value_channels < 1 || header.head_count < 1 || !decay_count_is_valid) {
        std::fprintf(stderr, "error: the trace header holds impossible sizes\n");
        return false;
    }
    return true;
}

// The storage methods that are compared, in the order they are printed.
enum MethodKind { kFullPrecision, kBfloat16EveryToken, kBfloat16EveryWindow, kInt8EveryToken, kLeapQuantHead };

struct Method {
    std::string name;
    MethodKind kind;
    Settings settings;       // used only by kLeapQuantHead
    double bytes_per_entry;
};

// Value of the entry at position `share` (0 = smallest, 1 = largest) of a sorted copy.
double value_at_share(std::vector<double> values, double share) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const std::size_t position = static_cast<std::size_t>(share * static_cast<double>(values.size() - 1) + 0.5);
    return values[position];
}

}  // unnamed namespace

int main(int argument_count, char** arguments) {
    if (argument_count < 2) {
        std::printf("usage: %s <trace file> [--window N] [--compensators N] [--bits N] [--scale-group N] [--max-tokens N] [--max-heads N] [--sweeps N] [--fit together|sequential]\n"
                    "       [--write-outputs PATH] [--rotate-keys 0|1] [--fp32-log-decay 0|1] [--fp32-records 0|1]\n"
                    "       --rotate-keys 1: keys and state in a Hadamard-rotated basis (state-cache plan, TR.1)\n"
                    "       omitted --compensators and --scale-group follow the paper for --bits\n"
                    "       (rank 4 and one step per column at 8 bits; rank 6 and groups of 8 at 6 and 4 bits)\n",
                    arguments[0]);
        return 2;
    }

    // ---- Step 1. Read the command line ---------------------------------------
    const std::string trace_path = arguments[1];
    int window_length = 16;
    int compensator_count = 4;
    int bits_per_entry = 8;
    int scale_group = 0;
    bool compensators_given = false;
    bool scale_group_given = false;
    int power_iteration_sweeps = 8;
    bool fit_together = true;
    bool rotate_keys = false;
    bool log_decay_fp32 = false;
    bool records_fp32 = false;
    long max_tokens = -1;   // -1 = all tokens in the file
    int max_heads = -1;     // -1 = all heads in the file
    std::string outputs_path;   // empty = do not write the outputs
    for (int index = 2; index + 1 < argument_count; index += 2) {
        const std::string option = arguments[index];
        const long number = std::atol(arguments[index + 1]);
        if (option == "--write-outputs") {
            outputs_path = arguments[index + 1];
        } else if (option == "--window") {
            window_length = static_cast<int>(number);
        } else if (option == "--compensators") {
            compensator_count = static_cast<int>(number);
            compensators_given = true;
        } else if (option == "--bits") {
            bits_per_entry = static_cast<int>(number);
        } else if (option == "--scale-group") {
            scale_group = static_cast<int>(number);
            scale_group_given = true;
        } else if (option == "--sweeps") {
            power_iteration_sweeps = static_cast<int>(number);
        } else if (option == "--fit") {
            // "together" (the default) or "sequential": see Settings::fit_compensators_together.
            fit_together = std::string(arguments[index + 1]) != "sequential";
        } else if (option == "--rotate-keys") {
            rotate_keys = number != 0;
        } else if (option == "--fp32-log-decay") {
            log_decay_fp32 = number != 0;
        } else if (option == "--fp32-records") {
            records_fp32 = number != 0;
        } else if (option == "--max-tokens") {
            max_tokens = number;
        } else if (option == "--max-heads") {
            max_heads = static_cast<int>(number);
        } else {
            std::fprintf(stderr, "error: unknown option %s\n", option.c_str());
            return 2;
        }
    }

    // ---- Step 2. Open the trace and read its header --------------------------
    std::ifstream file(trace_path.c_str(), std::ios::binary);
    if (!file) {
        std::fprintf(stderr, "error: cannot open %s\n", trace_path.c_str());
        return 2;
    }
    TraceHeader header;
    if (!read_header(file, header)) {
        return 2;
    }
    const int key_channels = header.key_channels;
    const int value_channels = header.value_channels;
    const int heads_in_file = header.head_count;
    const int decay_count = header.decay_values_per_head;
    const bool one_decay_per_token = (decay_count == 1);
    if (!compensators_given) {
        compensator_count = leapquant::paper_compensator_count(bits_per_entry);
    }
    if (!scale_group_given) {
        scale_group = leapquant::low_bit_scale_group(bits_per_entry);
    }
    const std::size_t entry_count = static_cast<std::size_t>(key_channels) * value_channels;

    int heads_used = heads_in_file;
    if (max_heads > 0 && max_heads < heads_in_file) {
        heads_used = max_heads;
    }

    std::printf("trace: %s\n", trace_path.c_str());
    std::printf("  %d heads (%d used), state %d x %d, %s\n", heads_in_file, heads_used, key_channels, value_channels,
                one_decay_per_token ? "one decay per token (Gated DeltaNet)" : "one decay per key channel (KDA)");

    // ---- Step 3. Describe the methods -----------------------------------------
    std::vector<Method> methods;
    methods.push_back({"fp32 (what the model does today)", kFullPrecision, Settings(), 4.0});
    methods.push_back({"bf16, rounded every token", kBfloat16EveryToken, Settings(), 2.0});
    methods.push_back({"bf16, rounded every window", kBfloat16EveryWindow, Settings(), 2.0});
    methods.push_back({"int8, rounded every token", kInt8EveryToken, Settings(), 1.0 + 4.0 / key_channels});

    Settings base_settings;
    base_settings.key_channels = key_channels;
    base_settings.value_channels = value_channels;
    base_settings.window_length = window_length;
    base_settings.bits_per_entry = bits_per_entry;
    base_settings.power_iteration_sweeps = power_iteration_sweeps;
    base_settings.fit_compensators_together = fit_together;
    base_settings.one_decay_per_token = one_decay_per_token;
    base_settings.scale_group = scale_group;
    base_settings.rotate_keys = rotate_keys;
    base_settings.log_decay_fp32 = log_decay_fp32;
    base_settings.records_fp32 = records_fp32;
    if (records_fp32) {
        std::printf("  records of the window kept in fp32 (--fp32-records 1)\n");
    }
    if (log_decay_fp32) {
        std::printf("  log decays of the window kept in fp32 (--fp32-log-decay 1)\n");
    }
    if (rotate_keys) {
        std::printf("  keys and state in a Hadamard-rotated basis (--rotate-keys 1)\n");
    }
    const int group_count = leapquant::scale_group_count_of(base_settings);
    const double step_bytes_per_entry = 4.0 * static_cast<double>(group_count) / static_cast<double>(key_channels);

    Settings window_only = base_settings;
    window_only.compensator_count = 0;
    window_only.use_smoothing = false;
    methods.push_back({"int8, rounded every window", kLeapQuantHead, window_only, 1.0 + step_bytes_per_entry});

    Settings with_compensators = base_settings;
    with_compensators.compensator_count = compensator_count;
    with_compensators.use_smoothing = false;
    methods.push_back({"  + compensators", kLeapQuantHead, with_compensators,
                       (static_cast<double>(entry_count) + 4.0 * value_channels * group_count +
                        2.0 * compensator_count * (key_channels + value_channels)) / static_cast<double>(entry_count)});

    Settings full_method = base_settings;
    full_method.compensator_count = compensator_count;
    full_method.use_smoothing = true;
    methods.push_back({"  + smoothing  (= LeapQuant)", kLeapQuantHead, full_method,
                       static_cast<double>(Head(full_method).boundary_state_bytes()) / static_cast<double>(entry_count)});

    const std::size_t method_count = methods.size();

    // ---- Step 4. Create one head per method per attention head ---------------
    // full_precision_heads[method][head] and leapquant_heads[method][head].
    // Only one of the two is used for a given method; the other stays idle.
    std::vector<std::vector<FullPrecisionHead> > full_precision_heads(method_count);
    std::vector<std::vector<Head> > leapquant_heads(method_count);
    for (std::size_t method = 0; method < method_count; ++method) {
        for (int head = 0; head < heads_used; ++head) {
            full_precision_heads[method].push_back(FullPrecisionHead(key_channels, value_channels));
            leapquant_heads[method].push_back(Head(methods[method].settings));
        }
    }

    // ---- Step 5. Load the starting state of every head -----------------------
    if (header.has_initial_state == 1) {
        std::vector<float> initial_state(entry_count);
        for (int head = 0; head < heads_in_file; ++head) {
            if (!read_floats(file, initial_state.data(), entry_count)) {
                std::fprintf(stderr, "error: the trace file ends inside the initial states\n");
                return 2;
            }
            if (head >= heads_used) {
                continue;
            }
            for (std::size_t method = 0; method < method_count; ++method) {
                if (methods[method].kind == kLeapQuantHead) {
                    leapquant_heads[method][head].load_full_precision_state(initial_state.data());
                } else {
                    full_precision_heads[method][head].load_state(initial_state.data());
                    // The naive formats also start from a rounded state.
                    if (methods[method].kind == kBfloat16EveryToken || methods[method].kind == kBfloat16EveryWindow) {
                        leapquant::round_trip_through_bfloat16(full_precision_heads[method][head].state);
                    }
                    if (methods[method].kind == kInt8EveryToken) {
                        leapquant::round_trip_through_int8_per_value_channel(full_precision_heads[method][head].state,
                                                                             key_channels, value_channels);
                    }
                }
            }
        }
        std::printf("  starting states loaded from the trace\n");
    } else {
        std::printf("  no starting states in the trace: every head starts at zero\n");
    }

    // ---- Step 6. Replay the tokens ---------------------------------------------
    const std::size_t floats_per_head = static_cast<std::size_t>(decay_count) + 1 + key_channels + value_channels +
                                        key_channels + (header.has_expected_output == 1 ? value_channels : 0);
    std::vector<float> record(floats_per_head);
    std::vector<float> output(value_channels), reference_output(value_channels), expected_output(value_channels);
    std::vector<float> state_copy(entry_count);

    std::vector<double> squared_output_error(method_count, 0.0);
    double squared_reference_output = 0.0;
    double worst_difference_from_recorded_output = 0.0;

    // Points at which the state error is reported: 1,024, 4,096, 16,384, ...
    std::vector<long> report_points;
    std::vector<std::vector<double> > pooled_state_error(method_count);
    long next_report_point = 1024;

    // The outputs file (--write-outputs). The token count in its header is
    // filled in at the end, when it is known.
    std::ofstream outputs_file;
    const std::size_t leapquant_method = method_count - 1;   // the full method is the last one
    if (!outputs_path.empty()) {
        outputs_file.open(outputs_path.c_str(), std::ios::binary);
        if (!outputs_file) {
            std::fprintf(stderr, "error: cannot write %s\n", outputs_path.c_str());
            return 2;
        }
        const std::int32_t output_header[3] = {static_cast<std::int32_t>(heads_used),
                                               static_cast<std::int32_t>(value_channels), 0};
        outputs_file.write("LQOUTPUT", 8);
        outputs_file.write(reinterpret_cast<const char*>(output_header), sizeof(output_header));
    }
    std::vector<float> outputs_of_token(static_cast<std::size_t>(heads_used) * value_channels);

    long tokens_replayed = 0;
    bool file_has_more = true;
    while (file_has_more && (max_tokens < 0 || tokens_replayed < max_tokens)) {
        for (int head = 0; head < heads_in_file; ++head) {
            if (!read_floats(file, record.data(), floats_per_head)) {
                file_has_more = false;   // normal end of the file (or a cut-off last block)
                break;
            }
            if (head >= heads_used) {
                continue;
            }

            // Split the record into its fields.
            const float* decay = record.data();
            const float beta = record[static_cast<std::size_t>(decay_count)];
            const float* key = record.data() + decay_count + 1;
            const float* value = key + key_channels;
            const float* query = value + value_channels;
            if (header.has_expected_output == 1) {
                const float* recorded = query + key_channels;
                for (int value_index = 0; value_index < value_channels; ++value_index) {
                    expected_output[value_index] = recorded[value_index];
                }
            }

            for (std::size_t method = 0; method < method_count; ++method) {
                if (methods[method].kind == kLeapQuantHead) {
                    Head& leapquant_head = leapquant_heads[method][head];
                    if (one_decay_per_token) {
                        leapquant::process_one_token_gated_delta_net(leapquant_head, decay[0], beta, key, value, query,
                                                                     output.data());
                    } else {
                        leapquant::process_one_token_kda(leapquant_head, decay, beta, key, value, query, output.data());
                    }
                } else {
                    FullPrecisionHead& full_precision_head = full_precision_heads[method][head];
                    if (one_decay_per_token) {
                        full_precision_head.process_one_token_gated_delta_net(decay[0], beta, key, value, query, output.data());
                    } else {
                        full_precision_head.process_one_token_kda(decay, beta, key, value, query, output.data());
                    }
                    if (methods[method].kind == kBfloat16EveryToken) {
                        leapquant::round_trip_through_bfloat16(full_precision_head.state);
                    }
                    if (methods[method].kind == kBfloat16EveryWindow && (tokens_replayed + 1) % window_length == 0) {
                        leapquant::round_trip_through_bfloat16(full_precision_head.state);
                    }
                    if (methods[method].kind == kInt8EveryToken) {
                        leapquant::round_trip_through_int8_per_value_channel(full_precision_head.state, key_channels,
                                                                             value_channels);
                    }
                }

                if (method == leapquant_method) {
                    for (int value_index = 0; value_index < value_channels; ++value_index) {
                        outputs_of_token[static_cast<std::size_t>(head) * value_channels + value_index] = output[value_index];
                    }
                }

                if (method == 0) {
                    reference_output = output;
                    const double reference_length = length_of(reference_output);
                    squared_reference_output += reference_length * reference_length;

                    // Self-check of the trace: this program's fp32 arithmetic must
                    // reproduce the output the model's own kernel produced.
                    if (header.has_expected_output == 1) {
                        const double expected_length = length_of(expected_output);
                        if (expected_length > 0.0) {
                            const double difference = length_of_difference(reference_output, expected_output) / expected_length;
                            if (difference > worst_difference_from_recorded_output) {
                                worst_difference_from_recorded_output = difference;
                            }
                        }
                    }
                } else {
                    const double difference = length_of_difference(output, reference_output);
                    squared_output_error[method] += difference * difference;
                }
            }
        }
        if (!file_has_more) {
            break;
        }
        tokens_replayed = tokens_replayed + 1;
        if (outputs_file.is_open()) {
            outputs_file.write(reinterpret_cast<const char*>(outputs_of_token.data()),
                               static_cast<std::streamsize>(outputs_of_token.size() * sizeof(float)));
        }

        if (tokens_replayed == next_report_point) {
            report_points.push_back(tokens_replayed);
            for (std::size_t method = 1; method < method_count; ++method) {
                double squared_difference = 0.0;
                double squared_reference = 0.0;
                for (int head = 0; head < heads_used; ++head) {
                    if (methods[method].kind == kLeapQuantHead) {
                        leapquant_heads[method][head].copy_state_to_full_precision(state_copy.data());
                    } else {
                        state_copy = full_precision_heads[method][head].state;
                    }
                    const double difference = length_of_difference(state_copy, full_precision_heads[0][head].state);
                    const double reference = length_of(full_precision_heads[0][head].state);
                    squared_difference += difference * difference;
                    squared_reference += reference * reference;
                }
                pooled_state_error[method].push_back(std::sqrt(squared_difference / squared_reference));
            }
            next_report_point = next_report_point * 4;
        }
    }

    if (tokens_replayed == 0) {
        std::fprintf(stderr, "error: the trace holds no complete token\n");
        return 2;
    }
    if (outputs_file.is_open()) {
        const std::int32_t token_count = static_cast<std::int32_t>(tokens_replayed);
        outputs_file.seekp(16);
        outputs_file.write(reinterpret_cast<const char*>(&token_count), sizeof(token_count));
        outputs_file.close();
        std::printf("  outputs of the LeapQuant method written to %s\n", outputs_path.c_str());
    }

    // ---- Step 7. State error of every head at the end ------------------------
    std::vector<std::vector<double> > final_state_error(method_count);
    for (std::size_t method = 1; method < method_count; ++method) {
        for (int head = 0; head < heads_used; ++head) {
            if (methods[method].kind == kLeapQuantHead) {
                leapquant_heads[method][head].copy_state_to_full_precision(state_copy.data());
            } else {
                state_copy = full_precision_heads[method][head].state;
            }
            const double reference = length_of(full_precision_heads[0][head].state);
            double relative_error = 0.0;
            if (reference > 0.0) {
                relative_error = length_of_difference(state_copy, full_precision_heads[0][head].state) / reference;
            }
            final_state_error[method].push_back(relative_error);
        }
    }

    // ---- Step 8. Print the report ----------------------------------------------
    bool trace_is_valid = true;
    std::printf("  %ld tokens replayed, window %d, %d compensators, %d bits, scale group %d, %d sweeps, compensators fitted %s\n",
                tokens_replayed, window_length, compensator_count, bits_per_entry, scale_group, power_iteration_sweeps,
                fit_together ? "together" : "one at a time");
    if (header.has_expected_output == 1) {
        std::printf("  self-check: fp32 replay vs the model's own kernel output, worst relative difference %.2e", 
                    worst_difference_from_recorded_output);
        if (worst_difference_from_recorded_output < 1e-3) {
            std::printf("  (ok)\n");
        } else {
            trace_is_valid = false;
            std::printf("\n  WARNING: the difference is large. The trace does not describe what the model computed.\n"
                        "           Fix the recorder before trusting the table below.\n");
        }
    }

    // First header line.
    std::printf("\n  %-34s %6s |", "", "bytes");
    for (std::size_t point = 0; point < report_points.size(); ++point) {
        std::printf(" %8s", point == 0 ? "after" : "");
    }
    if (!report_points.empty()) {
        std::printf(" |");
    }
    std::printf(" %-19s | %s\n", "state error at end", "output");

    // Second header line.
    std::printf("  %-34s %6s |", "method", "/entry");
    for (std::size_t point = 0; point < report_points.size(); ++point) {
        std::printf(" %8ld", report_points[point]);
    }
    if (!report_points.empty()) {
        std::printf(" |");
    }
    std::printf(" %9s %9s | %s\n", "median", "worst", "error");

    // One line per method.
    for (std::size_t method = 0; method < method_count; ++method) {
        std::printf("  %-34s %6.2f |", methods[method].name.c_str(), methods[method].bytes_per_entry);
        for (std::size_t point = 0; point < report_points.size(); ++point) {
            if (method == 0) {
                std::printf(" %8s", "0");
            } else {
                std::printf(" %7.3f%%", 100.0 * pooled_state_error[method][point]);
            }
        }
        if (!report_points.empty()) {
            std::printf(" |");
        }
        if (method == 0) {
            std::printf(" %9s %9s | %s\n", "0", "0", "0");
        } else {
            std::printf(" %8.3f%% %8.3f%% | %.3f%%\n", 100.0 * value_at_share(final_state_error[method], 0.5),
                        100.0 * value_at_share(final_state_error[method], 1.0),
                        100.0 * std::sqrt(squared_output_error[method] / squared_reference_output));
        }
    }
    if (!report_points.empty()) {
        std::printf("\n  after N       : state error of all heads together after N tokens\n");
    } else {
        std::printf("\n");
    }
    std::printf("  median, worst : state error across heads, after the last token\n");
    std::printf("  output error  : all heads and all tokens together\n");
    std::printf("  (error = distance from the fp32 result, as a share of its size)\n");
    if (!trace_is_valid) {
        std::printf("\n  RESULT: INVALID TRACE (exit status 3). The table above must not be used.\n");
        return 3;
    }
    return 0;
}
