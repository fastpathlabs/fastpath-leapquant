// =============================================================================
// reference_shim.cpp
//
// A plain-C doorway to the reference core, so that Python can call it.
//
// It wraps one LAYER: `heads` attention heads of one request. For every head it
// keeps two things side by side:
//     a leapquant::Head              the reference LeapQuant state
//     a leapquant::FullPrecisionHead the fp32 truth
//
// kernel/verify_kernel.py uses it to check the batched kernel (CPU or GPU)
// against the reference core in one process, with no files in between.
//
// Layouts follow the reference core: a state is [head][key][value], except
// where a function says "pool layout", which is the kernel's [head][value][key].
// =============================================================================

#include "../leapquant.h"
#include "../reference_heads.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

struct ReferenceLayer {
    leapquant::Settings settings;
    std::vector<leapquant::Head> leapquant_heads;
    // The LeapQuant heads as they were before the last lqref_step (see lqref_load_state, flag 4).
    std::vector<leapquant::Head> heads_before_step;
    std::vector<leapquant::FullPrecisionHead> truth_heads;
};

}  // unnamed namespace

extern "C" {

// Returns null if a setting is out of range.
void* lqref_create(int heads, int key_channels, int value_channels, int window_length, int compensator_count,
                   int bits_per_entry, int power_iteration_sweeps, int use_smoothing, int scale_group) {
    ReferenceLayer* layer = new ReferenceLayer();
    layer->settings.key_channels = key_channels;
    layer->settings.value_channels = value_channels;
    layer->settings.window_length = window_length;
    layer->settings.compensator_count = compensator_count;
    layer->settings.bits_per_entry = bits_per_entry;
    layer->settings.power_iteration_sweeps = power_iteration_sweeps;
    layer->settings.use_smoothing = use_smoothing != 0;
    layer->settings.scale_group = scale_group;
    layer->settings.one_decay_per_token = true;
    try {
        for (int head = 0; head < heads; ++head) {
            layer->leapquant_heads.push_back(leapquant::Head(layer->settings));
            layer->truth_heads.push_back(leapquant::FullPrecisionHead(key_channels, value_channels));
        }
    } catch (...) {
        delete layer;
        return nullptr;
    }
    return layer;
}

void lqref_destroy(void* handle) {
    delete static_cast<ReferenceLayer*>(handle);
}

// Loads a full-precision state into every head. states: [head][key][value].
// which: 1 = LeapQuant heads, 2 = truth heads, 3 = both. Add 4 if the state continues the same
// request (the fit then starts from the stored compensators, as at a fold).
void lqref_load_state(void* handle, const float* states, int which) {
    ReferenceLayer* layer = static_cast<ReferenceLayer*>(handle);
    const std::size_t entry_count =
        static_cast<std::size_t>(layer->settings.key_channels) * layer->settings.value_channels;
    for (std::size_t head = 0; head < layer->leapquant_heads.size(); ++head) {
        if ((which & 1) != 0) {
            // which & 4: the state continues the same request. The head goes back to what it was
            // before the last step (so that its own fold of that step is undone) and then takes the
            // state, starting its fit from the compensators it had then, exactly as a fold does.
            if ((which & 4) != 0 && head < layer->heads_before_step.size()) {
                layer->leapquant_heads[head] = layer->heads_before_step[head];
            }
            layer->leapquant_heads[head].load_full_precision_state(states + head * entry_count, (which & 4) != 0);
        }
        if ((which & 2) != 0) {
            layer->truth_heads[head].load_state(states + head * entry_count);
        }
    }
}

// One Gated DeltaNet token for every head.
//   alpha, beta        [head]
//   key, query         [head][key]
//   value              [head][value]
//   leapquant_output   [head][value], filled
//   truth_output       [head][value], filled; may be null to skip the truth heads
void lqref_step(void* handle, const float* alpha, const float* beta, const float* key, const float* value,
                const float* query, float* leapquant_output, float* truth_output) {
    ReferenceLayer* layer = static_cast<ReferenceLayer*>(handle);
    const std::size_t key_channels = static_cast<std::size_t>(layer->settings.key_channels);
    const std::size_t value_channels = static_cast<std::size_t>(layer->settings.value_channels);
    layer->heads_before_step = layer->leapquant_heads;
    for (std::size_t head = 0; head < layer->leapquant_heads.size(); ++head) {
        leapquant::process_one_token_gated_delta_net(layer->leapquant_heads[head], alpha[head], beta[head],
                                                     key + head * key_channels, value + head * value_channels,
                                                     query + head * key_channels,
                                                     leapquant_output + head * value_channels);
        if (truth_output != nullptr) {
            layer->truth_heads[head].process_one_token_gated_delta_net(alpha[head], beta[head], key + head * key_channels,
                                                                       value + head * value_channels,
                                                                       query + head * key_channels,
                                                                       truth_output + head * value_channels);
        }
    }
}

// Records in the window of the LeapQuant heads (the same for every head).
int lqref_tokens_in_window(void* handle) {
    ReferenceLayer* layer = static_cast<ReferenceLayer*>(handle);
    return layer->leapquant_heads[0].tokens_in_window();
}

// Forces a fold of every LeapQuant head.
void lqref_fold(void* handle) {
    ReferenceLayer* layer = static_cast<ReferenceLayer*>(handle);
    for (std::size_t head = 0; head < layer->leapquant_heads.size(); ++head) {
        layer->leapquant_heads[head].fold_window_into_boundary_state();
    }
}

// Writes the current state of every head in full precision. states: [head][key][value].
// which: 1 = LeapQuant heads, 2 = truth heads.
void lqref_copy_state(void* handle, float* states, int which) {
    ReferenceLayer* layer = static_cast<ReferenceLayer*>(handle);
    const std::size_t entry_count =
        static_cast<std::size_t>(layer->settings.key_channels) * layer->settings.value_channels;
    for (std::size_t head = 0; head < layer->leapquant_heads.size(); ++head) {
        if (which == 1) {
            layer->leapquant_heads[head].copy_state_to_full_precision(states + head * entry_count);
        } else {
            const std::vector<float>& truth = layer->truth_heads[head].state;
            for (std::size_t index = 0; index < entry_count; ++index) {
                states[head * entry_count + index] = truth[index];
            }
        }
    }
}

// Writes the boundary state of every LeapQuant head in POOL layout:
//   rounded_residual         int8  [head][value][key]
//   value_channel_step       fp32  [head][value][group]   one group is [head][value]
//   key_row_scale            fp32  [head][key]
//   compensator_keys         fp16  [head][compensator][key]
//   compensator_corrections  fp16  [head][compensator][value]
void lqref_copy_boundary_state(void* handle, std::int8_t* rounded_residual, float* value_channel_step,
                               float* key_row_scale, std::uint16_t* compensator_keys,
                               std::uint16_t* compensator_corrections) {
    ReferenceLayer* layer = static_cast<ReferenceLayer*>(handle);
    const int key_channels = layer->settings.key_channels;
    const int value_channels = layer->settings.value_channels;
    const int compensator_count = layer->settings.compensator_count;
    const int group_count = leapquant::scale_group_count_of(layer->settings);
    for (std::size_t head = 0; head < layer->leapquant_heads.size(); ++head) {
        const leapquant::Head& source = layer->leapquant_heads[head];
        for (int value_index = 0; value_index < value_channels; ++value_index) {
            for (int group = 0; group < group_count; ++group) {
                const std::size_t step_index =
                    (head * static_cast<std::size_t>(value_channels) + static_cast<std::size_t>(value_index)) *
                        static_cast<std::size_t>(group_count) +
                    static_cast<std::size_t>(group);
                value_channel_step[step_index] = source.value_channel_step(value_index, group);
            }
            for (int key_index = 0; key_index < key_channels; ++key_index) {
                rounded_residual[(head * value_channels + value_index) * key_channels + key_index] =
                    static_cast<std::int8_t>(source.rounded_residual(key_index, value_index));
            }
        }
        for (int key_index = 0; key_index < key_channels; ++key_index) {
            key_row_scale[head * key_channels + key_index] = source.key_row_scale(key_index);
        }
        for (int compensator = 0; compensator < compensator_count; ++compensator) {
            for (int key_index = 0; key_index < key_channels; ++key_index) {
                // Every stored value is an fp16 value, so this conversion is exact.
                compensator_keys[(head * compensator_count + compensator) * key_channels + key_index] =
                    leapquant::convert_float_to_half(source.compensator_key(compensator, key_index));
            }
            for (int value_index = 0; value_index < value_channels; ++value_index) {
                compensator_corrections[(head * compensator_count + compensator) * value_channels + value_index] =
                    leapquant::convert_float_to_half(source.compensator_correction(compensator, value_index));
            }
        }
    }
}

}  // extern "C"
