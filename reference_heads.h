// =============================================================================
// reference_heads.h
//
// Things LeapQuant is compared against. Used by the tests and by the trace
// replay tool. Nothing in this file is part of the method itself.
//
//   FullPrecisionHead      the state kept as a plain fp32 matrix. This is what
//                          serving engines do today, and it is the ground truth.
//
//   round_trip_...         "store the matrix in a smaller format and read it
//                          back". Applied after every token, this gives the
//                          naive baselines of the paper (equation 2).
// =============================================================================

#ifndef LEAPQUANT_REFERENCE_HEADS_H
#define LEAPQUANT_REFERENCE_HEADS_H

#include <vector>

namespace leapquant {

class FullPrecisionHead {
public:
    FullPrecisionHead(int key_channels, int value_channels);

    // Copies key_channels * value_channels floats into the state.
    void load_state(const float* state_matrix);

    // Equation 1 of the paper, written out directly:
    //     held   = state^T * read_vector
    //     state  = Diag(decay) * state + write_key * (value - held)^T
    //     output = state^T * query
    // decay_value_count is 1 (one decay for the whole state) or key_channels.
    void process_one_token(const float* decay,
                           int decay_value_count,
                           const float* write_key,
                           const float* read_vector,
                           const float* value,
                           const float* query,
                           float* output);

    // One token of Gated DeltaNet, in the same order of operations as
    // fla/ops/gated_delta_rule/naive.py of the flash-linear-attention library.
    void process_one_token_gated_delta_net(float alpha,
                                           float beta,
                                           const float* key,
                                           const float* value,
                                           const float* query,
                                           float* output);

    // One token of KDA, in the same order of operations as
    // fla/ops/kda/naive.py of the flash-linear-attention library.
    void process_one_token_kda(const float* alpha,
                               float beta,
                               const float* key,
                               const float* value,
                               const float* query,
                               float* output);

    // result = state^T * input_vector
    void read_state(const float* input_vector, float* result) const;

    int key_channels() const;
    int value_channels() const;

    // The matrix itself, row after row. Public so that tests and baselines
    // can read it and round it.
    std::vector<float> state;

private:
    int key_channels_;
    int value_channels_;
};

// Stores the matrix as int8 with one step size per value column, then reads
// it back. No window, no compensators, no smoothing.
void round_trip_through_int8_per_value_channel(std::vector<float>& state, int key_channels, int value_channels);

// Stores every entry as bfloat16 (the top 16 bits of a float), then reads it back.
void round_trip_through_bfloat16(std::vector<float>& state);

// Small measurement helpers.
double length_of(const std::vector<float>& values);                                         // sqrt(sum of squares)
double length_of_difference(const std::vector<float>& first, const std::vector<float>& second);
float largest_absolute_value(const std::vector<float>& values);

}  // namespace leapquant

#endif  // LEAPQUANT_REFERENCE_HEADS_H
