// =============================================================================
// leapquant_small_matrix.h
//
// Arithmetic on very small matrices (at most 9 x 9), used by the joint
// compensator fit. It is shared, word for word, by the reference core
// (leapquant.cpp) and by the GPU kernel (kernel/leapquant_kernel_blocks.h), so
// that both do the same operations in the same order and can agree bit for bit.
//
// Everything is fp32, serial, and written with plain loops. A GPU runs these
// functions on ONE thread of a block; they are tiny.
//
// A matrix of size n is stored in a float array of LQ_SMALL * LQ_SMALL entries,
// entry (row, column) at row * LQ_SMALL + column.
// =============================================================================

#ifndef LEAPQUANT_SMALL_MATRIX_H
#define LEAPQUANT_SMALL_MATRIX_H

#include <math.h>

#ifndef LQ_SMALL_FUNCTION
#if defined(__CUDACC__)
#define LQ_SMALL_FUNCTION __device__ __forceinline__
#else
#define LQ_SMALL_FUNCTION inline
#endif
#endif

// Largest number of vectors fitted together: 8 compensators plus 1 rescue vector.
#define LQ_SMALL 9

// -----------------------------------------------------------------------------
// Cholesky factor of a symmetric matrix G (n x n):  G = R^T * R,  R upper triangular.
//
// G is the table of dot products of n vectors. If vector j is (almost) a
// combination of the vectors before it, its pivot is (almost) zero. Such a
// vector is DROPPED: alive[j] becomes 0, and row and column j of R are zero.
// The caller then treats that vector as the zero vector.
// -----------------------------------------------------------------------------
LQ_SMALL_FUNCTION void lq_small_cholesky(const float* gram, int n, float* factor, int* alive) {
    for (int index = 0; index < LQ_SMALL * LQ_SMALL; ++index) {
        factor[index] = 0.0f;
    }
    for (int column = 0; column < n; ++column) {
        // Entries above the diagonal of this column.
        for (int row = 0; row < column; ++row) {
            float value = gram[row * LQ_SMALL + column];
            for (int earlier = 0; earlier < row; ++earlier) {
                value -= factor[earlier * LQ_SMALL + row] * factor[earlier * LQ_SMALL + column];
            }
            if (alive[row] != 0) {
                factor[row * LQ_SMALL + column] = value / factor[row * LQ_SMALL + row];
            }
        }
        // The pivot.
        const float diagonal = gram[column * LQ_SMALL + column];
        float pivot = diagonal;
        for (int earlier = 0; earlier < column; ++earlier) {
            pivot -= factor[earlier * LQ_SMALL + column] * factor[earlier * LQ_SMALL + column];
        }
        // Drop the vector if less than one millionth of its squared length is new.
        if (diagonal > 0.0f && pivot > 1e-6f * diagonal) {
            alive[column] = 1;
            factor[column * LQ_SMALL + column] = sqrtf(pivot);
        } else {
            alive[column] = 0;
            for (int row = 0; row < column; ++row) {
                factor[row * LQ_SMALL + column] = 0.0f;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// One element of every vector, made orthonormal:   new = old * R^-1.
// `values` holds the same element (the same coordinate) of the n vectors.
// -----------------------------------------------------------------------------
LQ_SMALL_FUNCTION void lq_small_apply_inverse(const float* factor, const int* alive, int n, float* values) {
    for (int column = 0; column < n; ++column) {
        if (alive[column] == 0) {
            values[column] = 0.0f;
            continue;
        }
        float value = values[column];
        for (int earlier = 0; earlier < column; ++earlier) {
            value -= values[earlier] * factor[earlier * LQ_SMALL + column];
        }
        values[column] = value / factor[column * LQ_SMALL + column];
    }
}

// -----------------------------------------------------------------------------
// Eigenvectors of a symmetric matrix G (n x n), strongest first.
//
// Method: cyclic Jacobi rotations, a fixed number of rounds (no data-dependent
// stopping, so every run does the same operations). On return,
// vectors[row * LQ_SMALL + i] is entry `row` of the i-th strongest eigenvector,
// and strengths[i] is its eigenvalue. G is destroyed.
// -----------------------------------------------------------------------------
LQ_SMALL_FUNCTION void lq_small_eigenvectors(float* gram, int n, float* vectors, float* strengths) {
    float rotation[LQ_SMALL * LQ_SMALL];
    for (int row = 0; row < n; ++row) {
        for (int column = 0; column < n; ++column) {
            rotation[row * LQ_SMALL + column] = (row == column) ? 1.0f : 0.0f;
        }
    }
    const int rounds = 8;
    for (int round = 0; round < rounds; ++round) {
        for (int p = 0; p < n - 1; ++p) {
            for (int q = p + 1; q < n; ++q) {
                const float off = gram[p * LQ_SMALL + q];
                if (off == 0.0f) {
                    continue;
                }
                // The rotation that makes entry (p, q) zero.
                const float theta = (gram[q * LQ_SMALL + q] - gram[p * LQ_SMALL + p]) / (2.0f * off);
                const float t = (theta >= 0.0f ? 1.0f : -1.0f) / (fabsf(theta) + sqrtf(theta * theta + 1.0f));
                const float c = 1.0f / sqrtf(t * t + 1.0f);
                const float s = t * c;
                for (int k = 0; k < n; ++k) {           // columns p and q
                    const float gkp = gram[k * LQ_SMALL + p];
                    const float gkq = gram[k * LQ_SMALL + q];
                    gram[k * LQ_SMALL + p] = c * gkp - s * gkq;
                    gram[k * LQ_SMALL + q] = s * gkp + c * gkq;
                }
                for (int k = 0; k < n; ++k) {           // rows p and q
                    const float gpk = gram[p * LQ_SMALL + k];
                    const float gqk = gram[q * LQ_SMALL + k];
                    gram[p * LQ_SMALL + k] = c * gpk - s * gqk;
                    gram[q * LQ_SMALL + k] = s * gpk + c * gqk;
                }
                for (int k = 0; k < n; ++k) {           // accumulate the rotation
                    const float rkp = rotation[k * LQ_SMALL + p];
                    const float rkq = rotation[k * LQ_SMALL + q];
                    rotation[k * LQ_SMALL + p] = c * rkp - s * rkq;
                    rotation[k * LQ_SMALL + q] = s * rkp + c * rkq;
                }
            }
        }
    }
    // Order: strongest eigenvalue first. Equal values keep their order.
    int order[LQ_SMALL];
    for (int index = 0; index < n; ++index) {
        order[index] = index;
    }
    for (int first = 0; first < n - 1; ++first) {
        int best = first;
        for (int candidate = first + 1; candidate < n; ++candidate) {
            if (gram[order[candidate] * LQ_SMALL + order[candidate]] > gram[order[best] * LQ_SMALL + order[best]]) {
                best = candidate;
            }
        }
        const int chosen = order[best];
        for (int shift = best; shift > first; --shift) {
            order[shift] = order[shift - 1];
        }
        order[first] = chosen;
    }
    for (int index = 0; index < n; ++index) {
        strengths[index] = gram[order[index] * LQ_SMALL + order[index]];
        for (int row = 0; row < n; ++row) {
            vectors[row * LQ_SMALL + index] = rotation[row * LQ_SMALL + order[index]];
        }
    }
}

#endif  // LEAPQUANT_SMALL_MATRIX_H
