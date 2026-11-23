// =============================================================================
// leapquant.cpp
//
// Implementation of leapquant.h. Read the header first: it explains the idea,
// the stored data and the two operations.
//
// Style rules of this file
//   - Plain loops and plain arrays. No templates, no lambdas, no SIMD.
//   - Temporary arrays are local std::vector objects, so no function has
//     hidden state. This costs a few small allocations per token, which is
//     fine for a reference implementation.
//   - Every step names the equation of the paper it implements.
// =============================================================================

#include "leapquant.h"
#include "leapquant_small_matrix.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace leapquant {

// =============================================================================
// Part 1. 16-bit floating point conversion
// =============================================================================
//
// Bit layout of the two formats (most significant bit first):
//
//   float (32 bits):  1 sign bit | 8 exponent bits (bias 127) | 23 fraction bits
//   half  (16 bits):  1 sign bit | 5 exponent bits (bias  15) | 10 fraction bits
//
// A half can hold magnitudes from about 6e-8 up to 65504, with about 3 decimal
// digits of precision.

std::uint16_t convert_float_to_half(float value) {
    // Step 1. Look at the float as 32 raw bits.
    std::uint32_t float_bits = 0;
    std::memcpy(&float_bits, &value, sizeof(float_bits));

    // Step 2. Split off the sign. The sign bit moves from bit 31 to bit 15.
    const std::uint32_t sign_bit_of_half = (float_bits >> 16) & 0x8000u;
    const std::uint32_t magnitude_bits = float_bits & 0x7FFFFFFFu;

    // Step 3. Special case: not-a-number stays not-a-number.
    const std::uint32_t float_infinity_bits = 0x7F800000u;
    if (magnitude_bits > float_infinity_bits) {
        return static_cast<std::uint16_t>(sign_bit_of_half | 0x7E00u);
    }

    // Step 4. Special case: too large for a half.
    // 65520 is the first value that would round up to infinity in fp16.
    // We return the largest finite half (65504) instead, so one large number
    // can never turn the whole state into infinities.
    const std::uint32_t bits_of_65520 = 0x477FF000u;
    const std::uint16_t largest_finite_half = 0x7BFFu;
    if (magnitude_bits >= bits_of_65520) {
        return static_cast<std::uint16_t>(sign_bit_of_half | largest_finite_half);
    }

    // Step 5. Special case: smaller than 2^-14, the smallest normal half.
    // Such values are stored as a whole number of steps of 2^-24.
    const std::uint32_t bits_of_two_to_minus_14 = 0x38800000u;
    if (magnitude_bits < bits_of_two_to_minus_14) {
        float magnitude = 0.0f;
        std::memcpy(&magnitude, &magnitude_bits, sizeof(magnitude));
        const float number_of_steps = std::nearbyint(magnitude * 16777216.0f);  // 16777216 = 2^24
        return static_cast<std::uint16_t>(sign_bit_of_half | static_cast<std::uint32_t>(number_of_steps));
    }

    // Step 6. The normal case.
    // New exponent = old exponent - 127 + 15 = old exponent - 112.
    // New fraction = the top 10 of the 23 fraction bits.
    const std::uint32_t float_exponent = magnitude_bits >> 23;
    const std::uint32_t half_exponent = float_exponent - 112u;
    const std::uint32_t top_ten_fraction_bits = (magnitude_bits >> 13) & 0x3FFu;
    std::uint32_t half_bits = (half_exponent << 10) | top_ten_fraction_bits;

    // Step 7. Round using the 13 fraction bits that were cut off.
    // More than half a step: round up. Exactly half a step: round to the even
    // neighbour. If the fraction overflows, the +1 carries into the exponent,
    // which is exactly the right result.
    const std::uint32_t dropped_bits = magnitude_bits & 0x1FFFu;
    const std::uint32_t exactly_half_a_step = 0x1000u;
    const bool round_up = (dropped_bits > exactly_half_a_step) ||
                          (dropped_bits == exactly_half_a_step && (half_bits & 1u) == 1u);
    if (round_up) {
        half_bits = half_bits + 1u;
    }

    return static_cast<std::uint16_t>(sign_bit_of_half | half_bits);
}

float convert_half_to_float(std::uint16_t half_bits) {
    const std::uint32_t sign_bit_of_float = static_cast<std::uint32_t>(half_bits & 0x8000u) << 16;
    const std::uint32_t half_exponent = (half_bits >> 10) & 0x1Fu;
    const std::uint32_t half_fraction = half_bits & 0x3FFu;

    std::uint32_t float_bits = 0;

    if (half_exponent == 0u) {
        // Zero, or a very small number stored as `half_fraction` steps of 2^-24.
        const float magnitude = static_cast<float>(half_fraction) * (1.0f / 16777216.0f);
        std::memcpy(&float_bits, &magnitude, sizeof(float_bits));
        float_bits = float_bits | sign_bit_of_float;
    } else if (half_exponent == 31u) {
        // Infinity or not-a-number. The core never stores these, but the
        // conversion is complete so that it can be tested against a compiler.
        float_bits = sign_bit_of_float | 0x7F800000u | (half_fraction << 13);
    } else {
        // The normal case: exponent + 112, fraction moved up by 13 bits.
        float_bits = sign_bit_of_float | ((half_exponent + 112u) << 23) | (half_fraction << 13);
    }

    float result = 0.0f;
    std::memcpy(&result, &float_bits, sizeof(result));
    return result;
}

// =============================================================================
// Part 2. Small vector and matrix helpers (only visible inside this file)
// =============================================================================
namespace {

// Smallest allowed smoothing scale. It only exists to avoid a division by zero
// for a row that is entirely zero.
const float kSmallestKeyRowScale = 1e-6f;

// Returns first[0]*second[0] + first[1]*second[1] + ...
//
// The products are added up in FOUR CHUNKS: the first quarter of the indices,
// the second quarter, and so on, each from left to right; then the four chunk
// totals are added from left to right. In exact arithmetic that is the same
// sum. In floating point the order matters in the last bit, so it is fixed
// here, as part of the specification, and for a reason: a GPU kernel can give
// the four chunks to four threads and still produce the same bits. (Measured
// on an H100: with one thread per sum, a third of the decode step was spent
// in sums that kept 21 of 128 threads busy.)
float dot_product(const float* first, const float* second, int count) {
    const int chunk_count = 4;
    float chunk_total[chunk_count];
    for (int chunk = 0; chunk < chunk_count; ++chunk) {
        const int begin = (chunk * count) / chunk_count;
        const int end = ((chunk + 1) * count) / chunk_count;
        float sum = 0.0f;
        for (int index = begin; index < end; ++index) {
            sum += first[index] * second[index];
        }
        chunk_total[chunk] = sum;
    }
    return ((chunk_total[0] + chunk_total[1]) + chunk_total[2]) + chunk_total[3];
}

// matrix += scale * row_vector * column_vector^T
// `matrix` has `rows` rows and `columns` columns, stored row after row.
void add_outer_product(float* matrix,
                       int rows,
                       int columns,
                       const float* row_vector,
                       const float* column_vector,
                       float scale) {
    for (int row = 0; row < rows; ++row) {
        const float row_factor = scale * row_vector[row];
        float* matrix_row = matrix + static_cast<std::size_t>(row) * columns;
        for (int column = 0; column < columns; ++column) {
            matrix_row[column] += row_factor * column_vector[column];
        }
    }
}

// Converts `count` fp16 values to float.
void convert_half_array_to_float(const std::uint16_t* half_values, int count, float* float_values) {
    for (int index = 0; index < count; ++index) {
        float_values[index] = convert_half_to_float(half_values[index]);
    }
}

// Finds the single strongest pattern of a matrix (equation 5 of the paper).
//
// "Pattern" means a pair of vectors (key_side, value_side) such that the
// outer product  key_side * value_side^T  is as close to the matrix as one
// outer product can be. In linear-algebra terms this is the largest singular
// value and its two singular vectors.
//
// Method: power iteration. Start with a guess for value_side, then repeat
//     key_side   = matrix   * value_side      (and normalise key_side)
//     value_side = matrix^T * key_side
// Each repetition pulls the two vectors toward the strongest pattern.
//
// On return the matrix is approximately key_side * value_side^T.
// If the matrix is all zeros, both vectors are returned as zeros.
void find_strongest_pattern(const std::vector<float>& matrix,
                            int rows,
                            int columns,
                            int sweeps,
                            std::vector<float>& key_side,
                            std::vector<float>& value_side) {
    key_side.assign(static_cast<std::size_t>(rows), 0.0f);
    value_side.assign(static_cast<std::size_t>(columns), 0.0f);

    // Step 1. Starting guess: the row with the largest length.
    // WEAKNESS of this one-at-a-time method: that row usually contains some of
    // the strongest pattern, but not always. If it is exactly perpendicular to
    // it, the iteration never finds it (leapquant_test.cpp, test [9]). The
    // stored state is still correct then, only rounded less accurately.
    // find_strongest_patterns_together() below does not have this weakness.
    int longest_row = 0;
    float longest_row_squared_length = -1.0f;
    for (int row = 0; row < rows; ++row) {
        const float* matrix_row = matrix.data() + static_cast<std::size_t>(row) * columns;
        const float squared_length = dot_product(matrix_row, matrix_row, columns);
        if (squared_length > longest_row_squared_length) {
            longest_row_squared_length = squared_length;
            longest_row = row;
        }
    }
    if (!(longest_row_squared_length > 0.0f)) {
        return;  // The matrix is all zeros: there is no pattern.
    }
    const float one_over_length = 1.0f / std::sqrt(longest_row_squared_length);
    for (int column = 0; column < columns; ++column) {
        value_side[column] = matrix[static_cast<std::size_t>(longest_row) * columns + column] * one_over_length;
    }

    // Step 2. The sweeps.
    float strength = 0.0f;  // length of value_side after the last sweep = the singular value
    for (int sweep = 0; sweep < sweeps; ++sweep) {
        // key_side = matrix * value_side
        for (int row = 0; row < rows; ++row) {
            const float* matrix_row = matrix.data() + static_cast<std::size_t>(row) * columns;
            key_side[row] = dot_product(matrix_row, value_side.data(), columns);
        }
        // Normalise key_side to length 1.
        const float key_side_length = std::sqrt(dot_product(key_side.data(), key_side.data(), rows));
        if (!(key_side_length > 0.0f)) {
            key_side.assign(static_cast<std::size_t>(rows), 0.0f);
            value_side.assign(static_cast<std::size_t>(columns), 0.0f);
            return;
        }
        for (int row = 0; row < rows; ++row) {
            key_side[row] = key_side[row] / key_side_length;
        }

        // value_side = matrix^T * key_side
        for (int column = 0; column < columns; ++column) {
            value_side[column] = 0.0f;
        }
        for (int row = 0; row < rows; ++row) {
            const float key_value = key_side[row];
            const float* matrix_row = matrix.data() + static_cast<std::size_t>(row) * columns;
            for (int column = 0; column < columns; ++column) {
                value_side[column] += key_value * matrix_row[column];
            }
        }
        strength = std::sqrt(dot_product(value_side.data(), value_side.data(), columns));
        if (!(strength > 0.0f)) {
            key_side.assign(static_cast<std::size_t>(rows), 0.0f);
            value_side.assign(static_cast<std::size_t>(columns), 0.0f);
            return;
        }

        // Normalise value_side for the next sweep. After the LAST sweep it is
        // left as it is, because then key_side * value_side^T is the best fit
        // for the key_side we ended with, even if the iteration has not fully
        // converged.
        const bool is_last_sweep = (sweep == sweeps - 1);
        if (!is_last_sweep) {
            for (int column = 0; column < columns; ++column) {
                value_side[column] = value_side[column] / strength;
            }
        }
    }

    // Step 3. Balance the two vectors.
    // Now key_side has length 1 and value_side has length `strength`.
    // Give each side sqrt(strength), so both stay well inside the fp16 range.
    const float balance = std::sqrt(strength);
    for (int row = 0; row < rows; ++row) {
        key_side[row] = key_side[row] * balance;
    }
    for (int column = 0; column < columns; ++column) {
        value_side[column] = value_side[column] / balance;
    }
}

// first[0]*second[0] + ... where `second` is every `stride`-th element: the
// same four chunks as dot_product().
float dot_product_with_stride(const float* first, const float* second, std::size_t stride, int count) {
    const int chunk_count = 4;
    float chunk_total[chunk_count];
    for (int chunk = 0; chunk < chunk_count; ++chunk) {
        const int begin = (chunk * count) / chunk_count;
        const int end = ((chunk + 1) * count) / chunk_count;
        float sum = 0.0f;
        for (int index = begin; index < end; ++index) {
            sum += first[index] * second[static_cast<std::size_t>(index) * stride];
        }
        chunk_total[chunk] = sum;
    }
    return ((chunk_total[0] + chunk_total[1]) + chunk_total[2]) + chunk_total[3];
}

// values[0]^2 + values[stride]^2 + ... in the same four chunks.
float dot_product_with_stride_squared(const float* values, std::size_t stride, int count) {
    const int chunk_count = 4;
    float chunk_total[chunk_count];
    for (int chunk = 0; chunk < chunk_count; ++chunk) {
        const int begin = (chunk * count) / chunk_count;
        const int end = ((chunk + 1) * count) / chunk_count;
        float sum = 0.0f;
        for (int index = begin; index < end; ++index) {
            const float value = values[static_cast<std::size_t>(index) * stride];
            sum += value * value;
        }
        chunk_total[chunk] = sum;
    }
    return ((chunk_total[0] + chunk_total[1]) + chunk_total[2]) + chunk_total[3];
}

// Makes `count` vectors of length `length` orthonormal (perpendicular to each
// other, each of length 1). vectors[j] is vector j. A vector that is (almost) a
// combination of the vectors before it becomes the zero vector.
//
// Method ("Cholesky QR", done twice for accuracy): compute the table of all dot
// products, factor it, and apply the inverse factor to every coordinate.
void make_orthonormal(std::vector<std::vector<float> >& vectors, int count, int length) {
    float gram[LQ_SMALL * LQ_SMALL];
    float factor[LQ_SMALL * LQ_SMALL];
    int alive[LQ_SMALL];
    float values[LQ_SMALL];
    for (int pass = 0; pass < 2; ++pass) {
        for (int first = 0; first < count; ++first) {
            for (int second = first; second < count; ++second) {
                const float product = dot_product(vectors[first].data(), vectors[second].data(), length);
                gram[first * LQ_SMALL + second] = product;
                gram[second * LQ_SMALL + first] = product;
            }
        }
        lq_small_cholesky(gram, count, factor, alive);
        for (int index = 0; index < length; ++index) {
            for (int vector = 0; vector < count; ++vector) {
                values[vector] = vectors[vector][index];
            }
            lq_small_apply_inverse(factor, alive, count, values);
            for (int vector = 0; vector < count; ++vector) {
                vectors[vector][index] = values[vector];
            }
        }
    }
}

// Finds the `pattern_count` strongest patterns of a matrix TOGETHER (equation 5
// of the paper, for all compensators at once).
//
//   start_value_sides   pattern_count vectors of length `columns`: a first guess
//                       for the value sides. A zero vector means "no guess".
//   key_sides, value_sides   the result; the matrix is approximately
//                       sum over patterns of  key_side * value_side^T.
//
// Method: block (subspace) iteration with one extra "rescue" vector.
//   1. Start vectors: the guesses. Where there is no guess, the longest rows of
//      the matrix. Plus one rescue vector: the energy of every column. It is
//      positive everywhere a column has content, so it overlaps a dominant
//      pattern that the other start vectors may all be perpendicular to.
//   2. Each sweep:  U = matrix * V;  make U orthonormal;  V = matrix^T * U.
//      Two walks over the matrix serve all vectors.
//   3. The vectors now span a subspace that holds the strongest patterns, but
//      mixed. A small eigenvalue problem (pattern_count + 1 unknowns) unmixes
//      them and orders them by strength; the weakest is dropped.
void find_strongest_patterns_together(const std::vector<float>& matrix,
                                      int rows,
                                      int columns,
                                      int sweeps,
                                      int pattern_count,
                                      const std::vector<std::vector<float> >& start_value_sides,
                                      std::vector<std::vector<float> >& key_sides,
                                      std::vector<std::vector<float> >& value_sides) {
    const int vector_count = pattern_count + 1;   // + the rescue vector
    std::vector<std::vector<float> > u(static_cast<std::size_t>(vector_count), std::vector<float>(static_cast<std::size_t>(rows), 0.0f));
    std::vector<std::vector<float> > v(static_cast<std::size_t>(vector_count), std::vector<float>(static_cast<std::size_t>(columns), 0.0f));

    // Step 1a. The guesses, and which of them are missing.
    bool some_guess_is_missing = false;
    std::vector<int> has_guess(static_cast<std::size_t>(pattern_count), 0);
    for (int pattern = 0; pattern < pattern_count; ++pattern) {
        v[pattern] = start_value_sides[pattern];
        const float squared_length = dot_product(v[pattern].data(), v[pattern].data(), columns);
        has_guess[pattern] = squared_length > 0.0f ? 1 : 0;
        if (has_guess[pattern] == 0) {
            some_guess_is_missing = true;
        }
    }

    // Step 1b. Missing guesses are replaced by the longest rows of the matrix.
    if (some_guess_is_missing) {
        std::vector<float> row_squared_length(static_cast<std::size_t>(rows));
        for (int row = 0; row < rows; ++row) {
            const float* matrix_row = matrix.data() + static_cast<std::size_t>(row) * columns;
            row_squared_length[row] = dot_product(matrix_row, matrix_row, columns);
        }
        for (int pattern = 0; pattern < pattern_count; ++pattern) {
            if (has_guess[pattern] != 0) {
                continue;
            }
            int longest_row = -1;
            float longest = 0.0f;
            for (int row = 0; row < rows; ++row) {
                if (row_squared_length[row] > longest) {
                    longest = row_squared_length[row];
                    longest_row = row;
                }
            }
            if (longest_row < 0) {
                break;      // no row with content is left
            }
            row_squared_length[longest_row] = 0.0f;      // each row is used once
            for (int column = 0; column < columns; ++column) {
                v[pattern][column] = matrix[static_cast<std::size_t>(longest_row) * columns + column];
            }
        }
    }

    // Step 1c. The rescue vector: the energy of every column.
    for (int column = 0; column < columns; ++column) {
        v[pattern_count][column] =
            dot_product_with_stride_squared(matrix.data() + column, static_cast<std::size_t>(columns), rows);
    }

    // Step 2. The sweeps. Without guesses the start is poorer: two more sweeps.
    const int sweeps_used = some_guess_is_missing ? sweeps + 2 : sweeps;
    for (int sweep = 0; sweep < sweeps_used; ++sweep) {
        for (int vector = 0; vector < vector_count; ++vector) {          // U = matrix * V
            for (int row = 0; row < rows; ++row) {
                u[vector][row] = dot_product(matrix.data() + static_cast<std::size_t>(row) * columns, v[vector].data(), columns);
            }
        }
        make_orthonormal(u, vector_count, rows);
        for (int vector = 0; vector < vector_count; ++vector) {          // V = matrix^T * U
            for (int column = 0; column < columns; ++column) {
                v[vector][column] =
                    dot_product_with_stride(u[vector].data(), matrix.data() + column, static_cast<std::size_t>(columns), rows);
            }
        }
    }

    // Step 3. Unmix: eigenvectors of the table of dot products of the V vectors.
    float gram[LQ_SMALL * LQ_SMALL];
    float mix[LQ_SMALL * LQ_SMALL];
    float strengths[LQ_SMALL];
    for (int first = 0; first < vector_count; ++first) {
        for (int second = first; second < vector_count; ++second) {
            const float product = dot_product(v[first].data(), v[second].data(), columns);
            gram[first * LQ_SMALL + second] = product;
            gram[second * LQ_SMALL + first] = product;
        }
    }
    lq_small_eigenvectors(gram, vector_count, mix, strengths);

    for (int pattern = 0; pattern < pattern_count; ++pattern) {
        key_sides[pattern].assign(static_cast<std::size_t>(rows), 0.0f);
        value_sides[pattern].assign(static_cast<std::size_t>(columns), 0.0f);
        for (int row = 0; row < rows; ++row) {
            float sum = 0.0f;
            for (int vector = 0; vector < vector_count; ++vector) {
                sum += u[vector][row] * mix[vector * LQ_SMALL + pattern];
            }
            key_sides[pattern][row] = sum;
        }
        for (int column = 0; column < columns; ++column) {
            float sum = 0.0f;
            for (int vector = 0; vector < vector_count; ++vector) {
                sum += v[vector][column] * mix[vector * LQ_SMALL + pattern];
            }
            value_sides[pattern][column] = sum;
        }
        // key_side has length 1 and value_side carries the strength.
        // Give each side sqrt(strength), so both stay well inside the fp16 range.
        const float strength = std::sqrt(dot_product(value_sides[pattern].data(), value_sides[pattern].data(), columns));
        if (!(strength > 0.0f)) {
            key_sides[pattern].assign(static_cast<std::size_t>(rows), 0.0f);
            value_sides[pattern].assign(static_cast<std::size_t>(columns), 0.0f);
            continue;
        }
        const float balance = std::sqrt(strength);
        for (int row = 0; row < rows; ++row) {
            key_sides[pattern][row] = key_sides[pattern][row] * balance;
        }
        for (int column = 0; column < columns; ++column) {
            value_sides[pattern][column] = value_sides[pattern][column] / balance;
        }
    }
}

// x <- H x for the orthonormal Hadamard matrix H of size n (a power of two):
// the fast Walsh-Hadamard transform, then a scale of 1/sqrt(n). H is symmetric
// and H H = I, so the same call also undoes itself.
void rotate_by_hadamard(float* x, int n) {
    for (int half = 1; half < n; half <<= 1) {
        for (int start = 0; start < n; start += 2 * half) {
            for (int index = start; index < start + half; ++index) {
                const float first = x[index];
                const float second = x[index + half];
                x[index] = first + second;
                x[index + half] = first - second;
            }
        }
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(n));
    for (int index = 0; index < n; ++index) {
        x[index] = x[index] * scale;
    }
}

// The same rotation applied to the key dimension (the rows) of a
// key_channels x value_channels matrix stored row after row.
void rotate_rows_by_hadamard(float* matrix, int key_channels, int value_channels) {
    std::vector<float> column(static_cast<std::size_t>(key_channels));
    for (int value_index = 0; value_index < value_channels; ++value_index) {
        for (int key_index = 0; key_index < key_channels; ++key_index) {
            column[key_index] = matrix[static_cast<std::size_t>(key_index) * value_channels + value_index];
        }
        rotate_by_hadamard(column.data(), key_channels);
        for (int key_index = 0; key_index < key_channels; ++key_index) {
            matrix[static_cast<std::size_t>(key_index) * value_channels + value_index] = column[key_index];
        }
    }
}

}  // unnamed namespace

// =============================================================================
// Part 3. The Head class
// =============================================================================

Head::Head(const Settings& settings) : settings_(settings) {
    // Reject settings that would make the arithmetic below meaningless.
    if (settings_.key_channels < 1) {
        throw std::invalid_argument("leapquant: key_channels must be at least 1");
    }
    if (settings_.value_channels < 1) {
        throw std::invalid_argument("leapquant: value_channels must be at least 1");
    }
    if (settings_.window_length < 1) {
        throw std::invalid_argument("leapquant: window_length must be at least 1");
    }
    if (settings_.compensator_count < 0 || settings_.compensator_count > LQ_SMALL - 1) {
        throw std::invalid_argument("leapquant: compensator_count must be between 0 and 8");
    }
    if (settings_.bits_per_entry < 2 || settings_.bits_per_entry > 8) {
        throw std::invalid_argument("leapquant: bits_per_entry must be between 2 and 8");
    }
    if (settings_.power_iteration_sweeps < 1) {
        throw std::invalid_argument("leapquant: power_iteration_sweeps must be at least 1");
    }
    if (settings_.scale_group < 0) {
        throw std::invalid_argument("leapquant: scale_group must be 0 or a positive group length");
    }
    if (settings_.rotate_keys &&
        (!settings_.one_decay_per_token || (settings_.key_channels & (settings_.key_channels - 1)) != 0)) {
        throw std::invalid_argument("leapquant: rotate_keys needs one decay per token and a power-of-two key_channels");
    }

    // With b bits the integer levels are -(2^(b-1) - 1) ... +(2^(b-1) - 1).
    // For 8 bits that is -127 ... +127.
    largest_integer_level_ = (1 << (settings_.bits_per_entry - 1)) - 1;

    if (settings_.one_decay_per_token) {
        decay_values_per_token_ = 1;
    } else {
        decay_values_per_token_ = settings_.key_channels;
    }

    const std::size_t key_channels = static_cast<std::size_t>(settings_.key_channels);
    const std::size_t value_channels = static_cast<std::size_t>(settings_.value_channels);
    const std::size_t compensator_count = static_cast<std::size_t>(settings_.compensator_count);
    const std::size_t window_length = static_cast<std::size_t>(settings_.window_length);

    rounded_residual_.resize(key_channels * value_channels);
    value_channel_step_.resize(value_channels * static_cast<std::size_t>(scale_group_count_of(settings_)));
    key_row_scale_.resize(key_channels);
    compensator_keys_.resize(compensator_count * key_channels);
    compensator_corrections_.resize(compensator_count * value_channels);
    window_keys_.resize(window_length * key_channels);
    window_corrections_.resize(window_length * value_channels);
    window_log_decays_.resize(window_length * static_cast<std::size_t>(decay_values_per_token_));
    if (settings_.log_decay_fp32) {
        window_log_decays_fp32_.assign(window_length * static_cast<std::size_t>(decay_values_per_token_), 0.0f);
    }
    if (settings_.records_fp32) {
        window_keys_fp32_.assign(window_length * key_channels, 0.0f);
        window_corrections_fp32_.assign(window_length * value_channels, 0.0f);
    }

    clear_to_zero_state();
}

void Head::clear_to_zero_state() {
    for (std::size_t index = 0; index < rounded_residual_.size(); ++index) {
        rounded_residual_[index] = 0;
    }
    for (std::size_t index = 0; index < value_channel_step_.size(); ++index) {
        value_channel_step_[index] = 0.0f;
    }
    for (std::size_t index = 0; index < key_row_scale_.size(); ++index) {
        key_row_scale_[index] = 1.0f;
    }
    for (std::size_t index = 0; index < compensator_keys_.size(); ++index) {
        compensator_keys_[index] = 0;
    }
    for (std::size_t index = 0; index < compensator_corrections_.size(); ++index) {
        compensator_corrections_[index] = 0;
    }
    for (std::size_t index = 0; index < window_keys_.size(); ++index) {
        window_keys_[index] = 0;
    }
    for (std::size_t index = 0; index < window_corrections_.size(); ++index) {
        window_corrections_[index] = 0;
    }
    for (std::size_t index = 0; index < window_log_decays_.size(); ++index) {
        window_log_decays_[index] = 0;
    }
    for (std::size_t index = 0; index < window_log_decays_fp32_.size(); ++index) {
        window_log_decays_fp32_[index] = 0.0f;
    }
    for (std::size_t index = 0; index < window_keys_fp32_.size(); ++index) {
        window_keys_fp32_[index] = 0.0f;
    }
    for (std::size_t index = 0; index < window_corrections_fp32_.size(); ++index) {
        window_corrections_fp32_[index] = 0.0f;
    }
    tokens_in_window_ = 0;
}

void Head::load_full_precision_state(const float* state_matrix, bool continue_from_previous) {
    const std::size_t entry_count =
        static_cast<std::size_t>(settings_.key_channels) * static_cast<std::size_t>(settings_.value_channels);

    // Work on a copy, because store_boundary_state() destroys its input.
    std::vector<float> working_copy(state_matrix, state_matrix + entry_count);
    if (settings_.rotate_keys) {
        rotate_rows_by_hadamard(working_copy.data(), settings_.key_channels, settings_.value_channels);
    }

    tokens_in_window_ = 0;
    store_boundary_state(working_copy, continue_from_previous);
}

// -----------------------------------------------------------------------------
// process_one_token: equation 1 of the paper, done the LeapQuant way.
// With settings.rotate_keys the three key-side vectors are rotated first.
// -----------------------------------------------------------------------------
void Head::process_one_token(const float* decay,
                             const float* write_key,
                             const float* read_vector,
                             const float* value,
                             const float* query,
                             float* output) {
    if (!settings_.rotate_keys) {
        process_stored_token(decay, write_key, read_vector, value, query, output);
        return;
    }
    const int key_channels = settings_.key_channels;
    std::vector<float> rotated_write(write_key, write_key + key_channels);
    std::vector<float> rotated_read(read_vector, read_vector + key_channels);
    std::vector<float> rotated_query(query, query + key_channels);
    rotate_by_hadamard(rotated_write.data(), key_channels);
    rotate_by_hadamard(rotated_read.data(), key_channels);
    rotate_by_hadamard(rotated_query.data(), key_channels);
    process_stored_token(decay, rotated_write.data(), rotated_read.data(), value, rotated_query.data(), output);
}

void Head::process_stored_token(const float* decay,
                                const float* write_key,
                                const float* read_vector,
                                const float* value,
                                const float* query,
                                float* output) {
    const int key_channels = settings_.key_channels;
    const int value_channels = settings_.value_channels;

    // Step 1. Ask the state what it already holds along the read vector:
    //             held = state_before_this_token^T * read_vector
    //         (This is the "delta rule": only the difference between the new
    //          value and what is already stored will be written.)
    std::vector<float> held(static_cast<std::size_t>(value_channels));
    read_stored_state(read_vector, held.data());

    // Step 2. The correction (equation 3 of the paper):
    //             correction = value - held
    std::vector<float> correction(static_cast<std::size_t>(value_channels));
    for (int value_index = 0; value_index < value_channels; ++value_index) {
        correction[value_index] = value[value_index] - held[value_index];
    }

    // Step 3. Append one record (decay, write_key, correction) to the window
    //         buffer, in fp16. The int8 matrix is NOT touched.
    const int new_record = tokens_in_window_;

    std::uint16_t* stored_key = window_keys_.data() + static_cast<std::size_t>(new_record) * key_channels;
    for (int key_index = 0; key_index < key_channels; ++key_index) {
        stored_key[key_index] = convert_float_to_half(write_key[key_index]);
    }

    std::uint16_t* stored_correction =
        window_corrections_.data() + static_cast<std::size_t>(new_record) * value_channels;
    for (int value_index = 0; value_index < value_channels; ++value_index) {
        stored_correction[value_index] = convert_float_to_half(correction[value_index]);
    }
    if (settings_.records_fp32) {
        for (int key_index = 0; key_index < key_channels; ++key_index) {
            window_keys_fp32_[static_cast<std::size_t>(new_record) * key_channels + key_index] = write_key[key_index];
        }
        for (int value_index = 0; value_index < value_channels; ++value_index) {
            window_corrections_fp32_[static_cast<std::size_t>(new_record) * value_channels + value_index] =
                correction[value_index];
        }
    }

    std::uint16_t* stored_log_decay =
        window_log_decays_.data() + static_cast<std::size_t>(new_record) * decay_values_per_token_;
    for (int decay_index = 0; decay_index < decay_values_per_token_; ++decay_index) {
        // The logarithm is stored; see the comment on window_log_decays_ in the header.
        stored_log_decay[decay_index] = convert_float_to_half(std::log(decay[decay_index]));
        if (settings_.log_decay_fp32) {
            window_log_decays_fp32_[static_cast<std::size_t>(new_record) * decay_values_per_token_ + decay_index] =
                std::log(decay[decay_index]);
        }
    }

    tokens_in_window_ = tokens_in_window_ + 1;

    // Step 4. The output is the new state read with the query:
    //             output = state_after_this_token^T * query
    //         read_state() already includes the record appended in step 3.
    //
    //         Note for a GPU port: steps 1 and 4 both walk over the whole int8
    //         matrix. A kernel does both in ONE pass, using the identity
    //             state_after^T * query = state_before^T * (decay * query) + correction * (write_key . query)
    //         Here they are two separate calls to keep the code plain.
    read_stored_state(query, output);

    // Step 5. If the window is full, fold it into a new boundary state.
    if (tokens_in_window_ == settings_.window_length) {
        fold_window_into_boundary_state();
    }
}

// -----------------------------------------------------------------------------
// read_state: equation 10 of the paper.
//
//   result = state^T * input_vector
//
// The state is never built. Its three parts are read one after the other:
//   (a) the window records, newest first
//   (b) the compensators
//   (c) the int8 matrix
//
// The trick that keeps this cheap: a record written t tokens ago has since
// been faded by the decays of the t tokens after it. Instead of storing a
// faded copy of every record, we fade the INPUT vector step by step while
// walking from the newest record to the oldest. When the walk reaches the
// boundary state, the input vector carries the decay of the whole window.
// -----------------------------------------------------------------------------
void Head::read_state(const float* input_vector, float* result) const {
    if (!settings_.rotate_keys) {
        read_stored_state(input_vector, result);
        return;
    }
    std::vector<float> rotated(input_vector, input_vector + settings_.key_channels);
    rotate_by_hadamard(rotated.data(), settings_.key_channels);
    read_stored_state(rotated.data(), result);
}

void Head::read_stored_state(const float* input_vector, float* result) const {
    const int key_channels = settings_.key_channels;
    const int value_channels = settings_.value_channels;

    // The input vector, faded as the walk goes back in time.
    std::vector<float> faded_input(input_vector, input_vector + key_channels);

    // One record converted from fp16 to float.
    std::vector<float> record_key(static_cast<std::size_t>(key_channels));
    std::vector<float> record_correction(static_cast<std::size_t>(value_channels));

    for (int value_index = 0; value_index < value_channels; ++value_index) {
        result[value_index] = 0.0f;
    }

    // (a) Window records, newest first.
    //     A record contributes  correction * (key . faded_input).
    for (int record = tokens_in_window_ - 1; record >= 0; --record) {
        load_record(record, record_key.data(), record_correction.data());

        const float key_match = dot_product(record_key.data(), faded_input.data(), key_channels);
        for (int value_index = 0; value_index < value_channels; ++value_index) {
            result[value_index] += key_match * record_correction[value_index];
        }

        // Everything older than this record was faded by this record's decay.
        if (settings_.one_decay_per_token) {
            const float decay_value = std::exp(stored_log_decay(record, 0));
            for (int key_index = 0; key_index < key_channels; ++key_index) {
                faded_input[key_index] = faded_input[key_index] * decay_value;
            }
        } else {
            for (int key_index = 0; key_index < key_channels; ++key_index) {
                const float decay_value = std::exp(stored_log_decay(record, key_index));
                faded_input[key_index] = faded_input[key_index] * decay_value;
            }
        }
    }
    // Here faded_input = (decay of the whole window so far) * input_vector.

    // (b) Compensators. Same form as a record, but they belong to the boundary
    //     state, so they are faded by the whole window and have no decay of their own.
    for (int compensator = 0; compensator < settings_.compensator_count; ++compensator) {
        convert_half_array_to_float(compensator_keys_.data() + static_cast<std::size_t>(compensator) * key_channels,
                                    key_channels, record_key.data());
        convert_half_array_to_float(
            compensator_corrections_.data() + static_cast<std::size_t>(compensator) * value_channels,
            value_channels, record_correction.data());

        const float key_match = dot_product(record_key.data(), faded_input.data(), key_channels);
        for (int value_index = 0; value_index < value_channels; ++value_index) {
            result[value_index] += key_match * record_correction[value_index];
        }
    }

    // (c) The int8 matrix (appendix B.3 of the paper).
    //     The stored matrix is the residual with every row divided by its
    //     key_row_scale and every column divided by its value_channel_step.
    //     So:  result += value_channel_step * ( rounded_residual^T * (key_row_scale * faded_input) )
    //
    // One step per column: the sum over keys is finished, then multiplied by
    // the step. That order is part of the 8-bit result. Several steps per
    // column do the same sum inside each group of key rows, then multiply.
    std::vector<float> column_sums(static_cast<std::size_t>(value_channels), 0.0f);
    const int group_count = scale_group_count_of(settings_);
    const int group_length = scale_group_length_of(settings_);
    for (int group = 0; group < group_count; ++group) {
        for (int value_index = 0; value_index < value_channels; ++value_index) {
            column_sums[value_index] = 0.0f;
        }
        const int begin = group * group_length;
        int end = begin + group_length;
        if (end > key_channels) {
            end = key_channels;
        }
        for (int key_index = begin; key_index < end; ++key_index) {
            const float row_weight = faded_input[key_index] * key_row_scale_[key_index];
            const std::int8_t* residual_row =
                rounded_residual_.data() + static_cast<std::size_t>(key_index) * value_channels;
            for (int value_index = 0; value_index < value_channels; ++value_index) {
                column_sums[value_index] += row_weight * static_cast<float>(residual_row[value_index]);
            }
        }
        for (int value_index = 0; value_index < value_channels; ++value_index) {
            const std::size_t step_index =
                static_cast<std::size_t>(value_index) * static_cast<std::size_t>(group_count) + static_cast<std::size_t>(group);
            result[value_index] += column_sums[value_index] * value_channel_step_[step_index];
        }
    }
}

// -----------------------------------------------------------------------------
// copy_state_to_full_precision: equation 7 of the paper.
// Builds the full matrix from the boundary state and the window records.
// -----------------------------------------------------------------------------
void Head::copy_state_to_full_precision(float* state_matrix) const {
    copy_stored_state(state_matrix);
    if (settings_.rotate_keys) {
        // H is its own inverse: back to the original basis.
        rotate_rows_by_hadamard(state_matrix, settings_.key_channels, settings_.value_channels);
    }
}

void Head::copy_stored_state(float* state_matrix) const {
    const int key_channels = settings_.key_channels;
    const int value_channels = settings_.value_channels;

    std::vector<float> record_key(static_cast<std::size_t>(key_channels));
    std::vector<float> record_correction(static_cast<std::size_t>(value_channels));

    // Step 1. Undo the rounding of the int8 matrix:
    //             entry = key_row_scale[row] * integer_level * value_channel_step[column]
    // With several steps per column, the step is the one for this key row's group.
    const int group_count = scale_group_count_of(settings_);
    const int group_length = scale_group_length_of(settings_);
    for (int key_index = 0; key_index < key_channels; ++key_index) {
        const std::int8_t* residual_row = rounded_residual_.data() + static_cast<std::size_t>(key_index) * value_channels;
        float* state_row = state_matrix + static_cast<std::size_t>(key_index) * value_channels;
        const int group = key_index / group_length;
        for (int value_index = 0; value_index < value_channels; ++value_index) {
            const std::size_t step_index =
                static_cast<std::size_t>(value_index) * static_cast<std::size_t>(group_count) + static_cast<std::size_t>(group);
            state_row[value_index] = key_row_scale_[key_index] *
                                     static_cast<float>(residual_row[value_index]) *
                                     value_channel_step_[step_index];
        }
    }

    // Step 2. Add the compensators back.
    for (int compensator = 0; compensator < settings_.compensator_count; ++compensator) {
        convert_half_array_to_float(compensator_keys_.data() + static_cast<std::size_t>(compensator) * key_channels,
                                    key_channels, record_key.data());
        convert_half_array_to_float(
            compensator_corrections_.data() + static_cast<std::size_t>(compensator) * value_channels,
            value_channels, record_correction.data());
        add_outer_product(state_matrix, key_channels, value_channels, record_key.data(), record_correction.data(), 1.0f);
    }

    // Step 3. Replay the window, oldest token first (equation 3 of the paper):
    //             state <- Diag(decay) * state + key * correction^T
    for (int record = 0; record < tokens_in_window_; ++record) {
        for (int key_index = 0; key_index < key_channels; ++key_index) {
            float decay_value = 0.0f;
            if (settings_.one_decay_per_token) {
                decay_value = std::exp(stored_log_decay(record, 0));
            } else {
                decay_value = std::exp(stored_log_decay(record, key_index));
            }
            float* state_row = state_matrix + static_cast<std::size_t>(key_index) * value_channels;
            for (int value_index = 0; value_index < value_channels; ++value_index) {
                state_row[value_index] = state_row[value_index] * decay_value;
            }
        }

        load_record(record, record_key.data(), record_correction.data());
        add_outer_product(state_matrix, key_channels, value_channels, record_key.data(), record_correction.data(), 1.0f);
    }
}

void Head::load_record(int record, float* key, float* correction) const {
    const int key_channels = settings_.key_channels;
    const int value_channels = settings_.value_channels;
    if (settings_.records_fp32) {
        for (int key_index = 0; key_index < key_channels; ++key_index) {
            key[key_index] = window_keys_fp32_[static_cast<std::size_t>(record) * key_channels + key_index];
        }
        for (int value_index = 0; value_index < value_channels; ++value_index) {
            correction[value_index] = window_corrections_fp32_[static_cast<std::size_t>(record) * value_channels + value_index];
        }
        return;
    }
    convert_half_array_to_float(window_keys_.data() + static_cast<std::size_t>(record) * key_channels, key_channels, key);
    convert_half_array_to_float(window_corrections_.data() + static_cast<std::size_t>(record) * value_channels,
                                value_channels, correction);
}

float Head::stored_log_decay(int record, int index) const {
    const std::size_t at = static_cast<std::size_t>(record) * decay_values_per_token_ + static_cast<std::size_t>(index);
    if (settings_.log_decay_fp32) {
        return window_log_decays_fp32_[at];
    }
    return convert_half_to_float(window_log_decays_[at]);
}

void Head::fold_window_into_boundary_state() {
    const std::size_t entry_count =
        static_cast<std::size_t>(settings_.key_channels) * static_cast<std::size_t>(settings_.value_channels);

    // Step 1. Build the full matrix (boundary state + window records), in the
    //         basis the state is stored in.
    std::vector<float> full_state(entry_count);
    copy_stored_state(full_state.data());

    // Step 2. Turn it into a new boundary state and empty the window.
    store_boundary_state(full_state, true);
}

// -----------------------------------------------------------------------------
// store_boundary_state: sections 3.3 and 3.4 and appendix B.3 of the paper.
//
// Input : a full-precision matrix.
// Output: compensators + key_row_scale + value_channel_step + rounded_residual.
// -----------------------------------------------------------------------------
void Head::store_boundary_state(std::vector<float>& full_state, bool continue_from_previous) {
    const int key_channels = settings_.key_channels;
    const int value_channels = settings_.value_channels;

    // -------------------------------------------------------------------------
    // Step A. Compensators (equation 6).
    //
    // Peel off the strongest pattern, then the strongest pattern of what is
    // left, and so on. Each pattern is stored in fp16. The matrix is reduced
    // by the pattern AS STORED (after fp16 rounding), so the small fp16
    // rounding error stays in the matrix and is not lost.
    // -------------------------------------------------------------------------
    std::vector<float> key_side;
    std::vector<float> value_side;
    std::vector<float> stored_key_side(static_cast<std::size_t>(key_channels));
    std::vector<float> stored_value_side(static_cast<std::size_t>(value_channels));

    // All compensators together (see Settings::fit_compensators_together).
    const int compensator_count = settings_.compensator_count;
    const bool together = settings_.fit_compensators_together && compensator_count > 0;
    std::vector<std::vector<float> > all_key_sides(static_cast<std::size_t>(compensator_count));
    std::vector<std::vector<float> > all_value_sides(static_cast<std::size_t>(compensator_count));
    if (together) {
        // The guesses: the value sides of the compensators of the previous fold,
        // exactly as they are stored. None when a new state is loaded.
        std::vector<std::vector<float> > guesses(static_cast<std::size_t>(compensator_count),
                                                 std::vector<float>(static_cast<std::size_t>(value_channels), 0.0f));
        if (continue_from_previous) {
            for (int compensator = 0; compensator < compensator_count; ++compensator) {
                convert_half_array_to_float(
                    compensator_corrections_.data() + static_cast<std::size_t>(compensator) * value_channels,
                    value_channels, guesses[compensator].data());
            }
        }
        find_strongest_patterns_together(full_state, key_channels, value_channels, settings_.power_iteration_sweeps,
                                         compensator_count, guesses, all_key_sides, all_value_sides);
    }

    for (int compensator = 0; compensator < settings_.compensator_count; ++compensator) {
        if (together) {
            key_side = all_key_sides[compensator];
            value_side = all_value_sides[compensator];
        } else {
            find_strongest_pattern(full_state, key_channels, value_channels, settings_.power_iteration_sweeps,
                                   key_side, value_side);
        }

        std::uint16_t* key_slot = compensator_keys_.data() + static_cast<std::size_t>(compensator) * key_channels;
        for (int key_index = 0; key_index < key_channels; ++key_index) {
            key_slot[key_index] = convert_float_to_half(key_side[key_index]);
            stored_key_side[key_index] = convert_half_to_float(key_slot[key_index]);
        }

        std::uint16_t* correction_slot =
            compensator_corrections_.data() + static_cast<std::size_t>(compensator) * value_channels;
        for (int value_index = 0; value_index < value_channels; ++value_index) {
            correction_slot[value_index] = convert_float_to_half(value_side[value_index]);
            stored_value_side[value_index] = convert_half_to_float(correction_slot[value_index]);
        }

        // full_state -= stored_key_side * stored_value_side^T
        add_outer_product(full_state.data(), key_channels, value_channels, stored_key_side.data(),
                          stored_value_side.data(), -1.0f);
    }
    // From here on, full_state holds the RESIDUAL: the state minus its compensators.

    // -------------------------------------------------------------------------
    // Step B. Smoothing (section 3.4).
    //
    // Some rows of the residual are much larger than others. One step size per
    // column would then be dictated by the large rows and waste precision on
    // the small ones. So every row is divided by
    //         key_row_scale = sqrt( average absolute value of the row )
    // before rounding, and multiplied back when the matrix is read.
    // -------------------------------------------------------------------------
    for (int key_index = 0; key_index < key_channels; ++key_index) {
        float* residual_row = full_state.data() + static_cast<std::size_t>(key_index) * value_channels;

        float scale = 1.0f;
        if (settings_.use_smoothing) {
            float sum_of_absolute_values = 0.0f;
            for (int value_index = 0; value_index < value_channels; ++value_index) {
                sum_of_absolute_values += std::fabs(residual_row[value_index]);
            }
            const float average_absolute_value = sum_of_absolute_values / static_cast<float>(value_channels);
            scale = std::sqrt(average_absolute_value);
            if (scale < kSmallestKeyRowScale) {
                scale = kSmallestKeyRowScale;
            }
        }
        key_row_scale_[key_index] = scale;

        for (int value_index = 0; value_index < value_channels; ++value_index) {
            residual_row[value_index] = residual_row[value_index] / scale;
        }
    }

    // -------------------------------------------------------------------------
    // Step C. One step size per value column (appendix B.3).
    //
    // The largest absolute entry of a column (or of one group of its key rows)
    // is mapped to the largest integer level:
    //     step = largest_absolute_entry / largest_integer_level
    // One group is the whole column, and that is the only form an 8-bit state uses.
    // -------------------------------------------------------------------------
    const int group_count = scale_group_count_of(settings_);
    const int group_length = scale_group_length_of(settings_);
    for (int value_index = 0; value_index < value_channels; ++value_index) {
        for (int group = 0; group < group_count; ++group) {
            const int begin = group * group_length;
            int end = begin + group_length;
            if (end > key_channels) {
                end = key_channels;
            }
            float largest_absolute_entry = 0.0f;
            for (int key_index = begin; key_index < end; ++key_index) {
                const float entry = full_state[static_cast<std::size_t>(key_index) * value_channels + value_index];
                if (std::fabs(entry) > largest_absolute_entry) {
                    largest_absolute_entry = std::fabs(entry);
                }
            }
            const std::size_t step_index =
                static_cast<std::size_t>(value_index) * static_cast<std::size_t>(group_count) + static_cast<std::size_t>(group);
            value_channel_step_[step_index] = largest_absolute_entry / static_cast<float>(largest_integer_level_);
        }
    }

    // -------------------------------------------------------------------------
    // Step D. Round every entry to the nearest integer level.
    // -------------------------------------------------------------------------
    for (int key_index = 0; key_index < key_channels; ++key_index) {
        for (int value_index = 0; value_index < value_channels; ++value_index) {
            const std::size_t position = static_cast<std::size_t>(key_index) * value_channels + value_index;
            const int group = key_index / group_length;
            const std::size_t step_index =
                static_cast<std::size_t>(value_index) * static_cast<std::size_t>(group_count) + static_cast<std::size_t>(group);
            const float step = value_channel_step_[step_index];

            float integer_level = 0.0f;
            if (step > 0.0f) {
                integer_level = std::nearbyint(full_state[position] / step);
            }
            // Guard against a level of 128 caused by floating-point rounding.
            if (integer_level > static_cast<float>(largest_integer_level_)) {
                integer_level = static_cast<float>(largest_integer_level_);
            }
            if (integer_level < -static_cast<float>(largest_integer_level_)) {
                integer_level = -static_cast<float>(largest_integer_level_);
            }
            rounded_residual_[position] = static_cast<std::int8_t>(integer_level);
        }
    }

    // -------------------------------------------------------------------------
    // Step E. The window buffer is now part of the boundary state. Empty it.
    // -------------------------------------------------------------------------
    tokens_in_window_ = 0;
}

std::size_t Head::boundary_state_bytes() const {
    const std::size_t bytes_of_rounded_residual = rounded_residual_.size() * sizeof(std::int8_t);
    const std::size_t bytes_of_value_channel_steps = value_channel_step_.size() * sizeof(float);
    const std::size_t bytes_of_key_row_scales = key_row_scale_.size() * sizeof(float);
    const std::size_t bytes_of_compensators =
        (compensator_keys_.size() + compensator_corrections_.size()) * sizeof(std::uint16_t);
    return bytes_of_rounded_residual + bytes_of_value_channel_steps + bytes_of_key_row_scales + bytes_of_compensators;
}

std::size_t Head::window_buffer_bytes() const {
    // With log_decay_fp32 the decays take 4 bytes each instead of 2 (the fp16 copy is kept only for simplicity).
    const std::size_t decay_bytes = settings_.log_decay_fp32 ? window_log_decays_fp32_.size() * sizeof(float)
                                                            : window_log_decays_.size() * sizeof(std::uint16_t);
    const std::size_t record_bytes = settings_.records_fp32
                                         ? (window_keys_fp32_.size() + window_corrections_fp32_.size()) * sizeof(float)
                                         : (window_keys_.size() + window_corrections_.size()) * sizeof(std::uint16_t);
    return record_bytes + decay_bytes;
}

const Settings& Head::settings() const {
    return settings_;
}

int Head::tokens_in_window() const {
    return tokens_in_window_;
}

int Head::rounded_residual(int key_index, int value_index) const {
    const std::size_t position =
        static_cast<std::size_t>(key_index) * settings_.value_channels + static_cast<std::size_t>(value_index);
    return static_cast<int>(rounded_residual_[position]);
}

float Head::value_channel_step(int value_index) const {
    return value_channel_step(value_index, 0);
}

float Head::value_channel_step(int value_index, int group) const {
    const int group_count = scale_group_count_of(settings_);
    const std::size_t step_index =
        static_cast<std::size_t>(value_index) * static_cast<std::size_t>(group_count) + static_cast<std::size_t>(group);
    return value_channel_step_[step_index];
}

float Head::key_row_scale(int key_index) const {
    return key_row_scale_[static_cast<std::size_t>(key_index)];
}

float Head::compensator_key(int compensator_index, int key_index) const {
    const std::size_t position =
        static_cast<std::size_t>(compensator_index) * settings_.key_channels + static_cast<std::size_t>(key_index);
    return convert_half_to_float(compensator_keys_[position]);
}

float Head::compensator_correction(int compensator_index, int value_index) const {
    const std::size_t position =
        static_cast<std::size_t>(compensator_index) * settings_.value_channels + static_cast<std::size_t>(value_index);
    return convert_half_to_float(compensator_corrections_[position]);
}

// =============================================================================
// Part 4. Model-specific wrappers (Table 4 of the paper)
// =============================================================================

void process_one_token_gated_delta_net(Head& head,
                                       float alpha,
                                       float beta,
                                       const float* key,
                                       const float* value,
                                       const float* query,
                                       float* output) {
    const int key_channels = head.settings().key_channels;

    // Gated DeltaNet in the form of equation 1:
    //     decay       = alpha        (one number for the whole state)
    //     write_key   = beta  * key
    //     read_vector = alpha * key
    std::vector<float> write_key(static_cast<std::size_t>(key_channels));
    std::vector<float> read_vector(static_cast<std::size_t>(key_channels));
    for (int key_index = 0; key_index < key_channels; ++key_index) {
        write_key[key_index] = beta * key[key_index];
        read_vector[key_index] = alpha * key[key_index];
    }

    head.process_one_token(&alpha, write_key.data(), read_vector.data(), value, query, output);
}

void process_one_token_kda(Head& head,
                           const float* alpha,
                           float beta,
                           const float* key,
                           const float* value,
                           const float* query,
                           float* output) {
    const int key_channels = head.settings().key_channels;

    // KDA in the form of equation 1:
    //     decay       = alpha        (one number per key channel)
    //     write_key   = beta * key
    //     read_vector = alpha * key  (element by element)
    std::vector<float> write_key(static_cast<std::size_t>(key_channels));
    std::vector<float> read_vector(static_cast<std::size_t>(key_channels));
    for (int key_index = 0; key_index < key_channels; ++key_index) {
        write_key[key_index] = beta * key[key_index];
        read_vector[key_index] = alpha[key_index] * key[key_index];
    }

    head.process_one_token(alpha, write_key.data(), read_vector.data(), value, query, output);
}

}  // namespace leapquant
