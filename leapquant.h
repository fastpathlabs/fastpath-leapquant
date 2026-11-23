// =============================================================================
// leapquant.h
//
// LeapQuant reference implementation in plain C++17.
//
// What this is
//   A small, slow, easy-to-read implementation of the method in the paper
//   "LeapQuant: Efficient Linear Attention with Accurate Recurrent State
//   Quantization" (arXiv 2609.38166). It is the specification that a GPU
//   kernel must match, number for number. It is not meant to be fast.
//
// Background in four sentences
//   A linear-attention layer (Gated DeltaNet in Qwen, KDA in Kimi) keeps one
//   matrix per attention head as its memory of every token seen so far.
//   The matrix is called the recurrent state. It has key_channels rows and
//   value_channels columns (128 x 128 in today's models).
//   For every new token the model changes the matrix a little and reads it.
//
// The update the model performs for every token (equation 1 of the paper)
//
//       state  <-  Diag(decay) * state  +  write_key * (value - state^T * read_vector)^T
//       output  =  state^T * query
//
//   In words:
//     1. Fade the old content:        multiply every row of the state by its decay.
//     2. Ask the state what it holds:  held = state^T * read_vector.
//     3. Compute the correction:       correction = value - held.
//     4. Write the correction:         state += write_key * correction^T   (a rank-1 update).
//     5. Read the answer:              output = state^T * query.
//
// What LeapQuant stores instead of the full-precision matrix (equation 7)
//
//   BOUNDARY STATE - written once every `window_length` tokens:
//     rounded_residual     int8   [key_channels][value_channels]  the bulk of the state
//     value_channel_step   fp32   [value_channels][groups]        step size of each column,
//                                                                or of each band of key rows
//     key_row_scale        fp32   [key_channels]                  "smoothing" scale of each row
//     compensators         fp16   compensator_count pairs (key vector, correction vector)
//                                 the few largest patterns of the state, kept more precisely
//
//   WINDOW BUFFER - one record appended per token, emptied at every fold:
//     window_keys          fp16   [token][key_channels]
//     window_corrections   fp16   [token][value_channels]
//     window_log_decays    fp16   [token][1 or key_channels]
//
//   The state at any moment equals:
//
//       decay_of_whole_window * ( key_row_scale * rounded_residual * value_channel_step
//                                 + sum of compensator outer products )
//       + sum over window tokens of ( later_decays * key * correction^T )
//
// The two operations
//   process_one_token()                  runs for every token. It only READS the int8 matrix.
//   fold_window_into_boundary_state()    runs every `window_length` tokens. It rebuilds the
//                                        full matrix, splits it again and WRITES the int8 matrix.
//
// How to use it
//
//       leapquant::Settings settings;                 // defaults: 128 x 128, window 16, 4 compensators, 8 bits
//       leapquant::Head head(settings);               // one Head per attention head, per layer, per request
//       head.load_full_precision_state(prefill_state); // optional; without it the state starts at zero
//       for every generated token:
//           leapquant::process_one_token_gated_delta_net(head, alpha, beta, key, value, query, output);
//
// Memory layout convention
//   Every matrix is stored row after row ("row-major"). The entry in key row
//   `key_index` and value column `value_index` is at position
//   key_index * value_channels + value_index.
// =============================================================================

#ifndef LEAPQUANT_H
#define LEAPQUANT_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace leapquant {

// -----------------------------------------------------------------------------
// 16-bit floating point ("half", "fp16") conversion.
//
// The paper keeps the window records and the compensators in fp16. Standard C++17
// has no 16-bit float type, so the conversion is written out by hand here.
//
//   convert_float_to_half  rounds to the nearest fp16 value (ties go to the even
//                          one). Values too large for fp16 become +-65504, the
//                          largest fp16 value, instead of infinity.
//   convert_half_to_float  is exact: every fp16 value is also a float value.
// -----------------------------------------------------------------------------
std::uint16_t convert_float_to_half(float value);
float convert_half_to_float(std::uint16_t half_bits);

// -----------------------------------------------------------------------------
// Settings of one head. The defaults are the values the paper uses.
// -----------------------------------------------------------------------------
struct Settings {
    // Number of rows of the state matrix (the paper calls it d_k).
    int key_channels = 128;

    // Number of columns of the state matrix (the paper calls it d_v).
    int value_channels = 128;

    // How many tokens are buffered before the int8 matrix is rewritten
    // (the paper calls it p). The paper uses 16.
    int window_length = 16;

    // How many large patterns are kept in fp16 beside the int8 matrix
    // (the paper calls it r). Paper v2, section 4.1: 4, and 6 at 6 and 4 bits.
    // paper_compensator_count() is that rule. An explicit count is left alone,
    // so an 8-bit run and a 4-compensator ablation stay what the caller asked for.
    int compensator_count = 4;

    // Number of bits used for each entry of the int8 matrix: 2 to 8.
    // This reference always stores one byte per entry; a lower setting only
    // restricts the integer levels that are used (for accuracy experiments).
    int bits_per_entry = 8;

    // Number of sweeps of the compensator fit. One sweep is two walks over the
    // matrix. The paper does not state its value.
    int power_iteration_sweeps = 8;

    // How the compensators are found.
    // true:  all of them together ("block iteration"), starting from the
    //        compensators of the previous fold plus one rescue vector. A fold
    //        then needs 2 walks over the matrix per sweep for ALL compensators,
    //        and cannot miss a dominant pattern the way the old method can
    //        (leapquant_test.cpp, test [9]).
    // false: one after the other, each with its own power iteration starting
    //        from the longest row. The method of the first version; kept for
    //        comparison.
    bool fit_compensators_together = true;

    // true:  the model gives one decay number per token      (Gated DeltaNet, Qwen).
    // false: the model gives one decay number per key channel (KDA, Kimi).
    bool one_decay_per_token = true;

    // Residual smoothing (section 3.4 of the paper). Switch off only to
    // measure how much it helps.
    bool use_smoothing = true;

    // How many key rows share one value-column step. 0 (the default) is one
    // step for the whole column: appendix B.3, and what every 8-bit state uses.
    // A smaller group gives each band of rows its own step, so one spike does
    // not set the step for the rest of the column. low_bit_scale_group() is 8
    // at 6 and 4 bits. The 8-bit layout is unchanged.
    int scale_group = 0;

    // Rotation experiment (state-cache plan v2, task TR.1). true: every key-side
    // vector (write key, read vector, query) is multiplied by a fixed orthonormal
    // Hadamard matrix H before use, and the state is kept in that rotated basis:
    // S' = H S. Reads are unchanged in exact arithmetic (S'^T (H q) = S^T q), and
    // so is the delta rule; only the rounding sees the rotated rows, in which a
    // large key row is spread over all rows. load_full_precision_state() rotates
    // the matrix it takes over; copy_state_to_full_precision() returns the state
    // in the original basis. Needs one decay per token and a power-of-two number
    // of key channels. Off by default; nothing else changes when it is off.
    bool rotate_keys = false;

    // Precision ablation (plan 5 T3.5, backlog B20): keep each record's log decay in fp32 instead of fp16
    // (4 bytes per record and decay value instead of 2). Off by default: fp16, as every kernel stores it.
    bool log_decay_fp32 = false;

    // Precision ablation (plan v6 N05, backlog B20): keep each record's key and correction in fp32 instead of fp16
    // (the window doubles: 16,384 instead of 8,192 bytes per 128 x 128 head). Off by default.
    bool records_fp32 = false;
};

// 0, or a group at least as long as the column, means one step per column.
inline int scale_group_length_of(const Settings& settings) {
    if (settings.scale_group <= 0 || settings.scale_group >= settings.key_channels) {
        return settings.key_channels;
    }
    return settings.scale_group;
}

inline int scale_group_count_of(const Settings& settings) {
    const int length = scale_group_length_of(settings);
    return (settings.key_channels + length - 1) / length;
}

// Paper v2, section 4.1. 8-bit keeps 4 compensators. 6-bit and 4-bit use 6.
inline int paper_compensator_count(int bits_per_entry) {
    if (bits_per_entry == 4 || bits_per_entry == 6) {
        return 6;
    }
    return 4;
}

// Key rows per value-column step. 0 means the whole column (8-bit, appendix B.3).
// At 6 and 4 bits a group of 8 keeps one spike from setting the step for the
// other rows of the column. The paper states the compensator count, not this
// grouping; the grouping is what stops a 7-level column from rounding the rest
// to zero. Group 8 is the measured point: on a spiked state, 6-bit then stays
// near the 8-bit error and 4-bit stops diverging, at about half an extra byte
// per entry. Group 16 left both widths short of that.
inline int low_bit_scale_group(int bits_per_entry) {
    if (bits_per_entry == 4 || bits_per_entry == 6) {
        return 8;
    }
    return 0;
}

// -----------------------------------------------------------------------------
// The recurrent state of ONE attention head, stored the LeapQuant way.
// -----------------------------------------------------------------------------
class Head {
public:
    // Creates a head whose state is all zeros.
    // Throws std::invalid_argument if a setting is out of range.
    explicit Head(const Settings& settings);

    // Sets the state to all zeros and empties the window buffer.
    void clear_to_zero_state();

    // Takes over a full-precision state, for example the one the prefill
    // phase produced. `state_matrix` holds key_channels * value_channels floats,
    // row after row. The window buffer is emptied.
    //
    // `continue_from_previous`: say true only if the state is the SAME request's
    // state a moment later (a test does this to hand a head the matrix another
    // implementation is about to fold). The compensators stored now are then
    // used as the starting guess of the fit, exactly as at a fold.
    void load_full_precision_state(const float* state_matrix, bool continue_from_previous = false);

    // Processes one token (equation 1 of the paper).
    //
    //   decay        1 float if settings.one_decay_per_token, else key_channels floats.
    //                Every value must be greater than 0 and at most 1.
    //   write_key    key_channels floats   - where the new content is written
    //   read_vector  key_channels floats   - used to ask what the state already holds
    //   value        value_channels floats - the content the model wants to store
    //   query        key_channels floats   - used to read the answer
    //   output       value_channels floats - filled by this function
    //
    // Model-specific wrappers that build these arguments are at the end of this file.
    void process_one_token(const float* decay,
                           const float* write_key,
                           const float* read_vector,
                           const float* value,
                           const float* query,
                           float* output);

    // result = state^T * input_vector (equation 10 of the paper).
    // input_vector has key_channels floats, result has value_channels floats.
    // The state is not changed.
    void read_state(const float* input_vector, float* result) const;

    // Writes the current state as a full-precision matrix into `state_matrix`
    // (key_channels * value_channels floats). Used by tests, by debugging and
    // by the fold. The state is not changed.
    void copy_state_to_full_precision(float* state_matrix) const;

    // Closes the current window: rebuilds the full matrix, picks new
    // compensators, smooths, rounds, and empties the window buffer.
    // process_one_token() calls this by itself when the window is full.
    void fold_window_into_boundary_state();

    // Bytes of the boundary state. This is what a prefix-cache checkpoint keeps.
    std::size_t boundary_state_bytes() const;

    // Bytes of the window buffer. Only requests that are generating need it.
    std::size_t window_buffer_bytes() const;

    // --- Read-only access for tests and tools ---------------------------------
    const Settings& settings() const;
    int tokens_in_window() const;
    // The integer level stored for one entry of the int8 matrix.
    int rounded_residual(int key_index, int value_index) const;
    float value_channel_step(int value_index) const;
    // group is the band of key rows. It is 0 when the column has one step.
    float value_channel_step(int value_index, int group) const;
    float key_row_scale(int key_index) const;
    float compensator_key(int compensator_index, int key_index) const;
    float compensator_correction(int compensator_index, int value_index) const;

private:
    // read_state and copy_state_to_full_precision in the basis the state is
    // stored in (the rotated one when settings.rotate_keys is set).
    void read_stored_state(const float* input_vector, float* result) const;
    void copy_stored_state(float* state_matrix) const;
    // The token step with key-side vectors already in the stored basis.
    void process_stored_token(const float* decay, const float* write_key, const float* read_vector,
                              const float* value, const float* query, float* output);

    // Turns a full-precision matrix into a new boundary state.
    // `full_state` is used as working memory and is destroyed.
    // `continue_from_previous`: the compensators stored right now describe the
    // same request a moment ago (true at a fold; false when a new state is loaded).
    void store_boundary_state(std::vector<float>& full_state, bool continue_from_previous);

    Settings settings_;

    // Largest integer level of the rounded matrix: 127 when bits_per_entry is 8.
    int largest_integer_level_;

    // 1 when one_decay_per_token, otherwise key_channels.
    int decay_values_per_token_;

    // Number of records currently in the window buffer: 0 to window_length.
    int tokens_in_window_;

    // ---- boundary state -------------------------------------------------------
    std::vector<std::int8_t> rounded_residual_;               // [key][value]
    std::vector<float> value_channel_step_;                   // [value][group]
    std::vector<float> key_row_scale_;                        // [key]
    std::vector<std::uint16_t> compensator_keys_;             // [compensator][key]    fp16
    std::vector<std::uint16_t> compensator_corrections_;      // [compensator][value]  fp16

    // ---- window buffer --------------------------------------------------------
    std::vector<std::uint16_t> window_keys_;                  // [token][key]    fp16
    std::vector<std::uint16_t> window_corrections_;           // [token][value]  fp16
    // The decay is stored as its natural logarithm. fp16 cannot hold a number
    // such as 0.9999 (it rounds to 1.0, which would switch the decay off), but
    // it holds log(0.9999) = -0.0001 with a relative error below 0.05%.
    std::vector<std::uint16_t> window_log_decays_;            // [token][1 or key]  fp16
    std::vector<float> window_log_decays_fp32_;               // the same in fp32, with settings.log_decay_fp32
    std::vector<float> window_keys_fp32_;                     // records in fp32, with settings.records_fp32
    std::vector<float> window_corrections_fp32_;
    // Record `record` as the head stores it (fp16, or fp32 with the option), as floats.
    void load_record(int record, float* key, float* correction) const;
    // The log decay of record `record`, value `index`, as the head stores it (fp16, or fp32 with the option).
    float stored_log_decay(int record, int index) const;
};

// -----------------------------------------------------------------------------
// Model-specific wrappers. They turn the numbers a model produces into the
// (decay, write_key, read_vector) of equation 1, following Table 4 of the paper.
// Both mappings are checked in leapquant_test.cpp against the reference kernels
// of the flash-linear-attention library.
//
// Not done here: the model scales the query by 1 / sqrt(key_channels) and
// normalises key and query to unit length BEFORE this point.
// -----------------------------------------------------------------------------

// Gated DeltaNet (Qwen3-Next, Qwen3.5):
//     state <- alpha * (I - beta * key * key^T) * state + beta * key * value^T
// alpha is exp(g), where g is the log-space gate the model computes.
// The head must have been created with one_decay_per_token = true.
void process_one_token_gated_delta_net(Head& head,
                                       float alpha,
                                       float beta,
                                       const float* key,
                                       const float* value,
                                       const float* query,
                                       float* output);

// KDA (Kimi Linear, GLM-5.3-Flash):
//     state <- (I - beta * key * key^T) * Diag(alpha) * state + beta * key * value^T
// alpha has key_channels values.
// The head must have been created with one_decay_per_token = false.
void process_one_token_kda(Head& head,
                           const float* alpha,
                           float beta,
                           const float* key,
                           const float* value,
                           const float* query,
                           float* output);

}  // namespace leapquant

#endif  // LEAPQUANT_H
