/*
 * q/k row math shared by qk_prologue.cu (the training backward) and decode.cu (decoding), so the decoder's q and k
 * equal the training forward's bitwise: the forward runs in the q/k/v GEMM's epilogue (model/ops/qk_prologue.py),
 * whose rstd this file's qk_rstd reproduces term for term. On one head row x: n = x / sqrt(mean(x^2) + eps)
 * (F.rms_norm's fp32 epsilon), m = n * g, y = m rotated by -theta over the full head (m1, m2 the halves;
 * y1 = m1 cos + m2 sin, y2 = -m1 sin + m2 cos; karpathy/nanochat 92d63d4 gpt.py:57-65). A row is QK_LANES threads,
 * each holding dims o..o+VEC-1 and their rotation partners o+HALF..; row sums reduce by shuffles within those lanes.
 */
#pragma once

#include "common.cuh"

/* HEAD_DIM: model/config.py HEAD_DIM, defined by native.py's nvcc command */
constexpr int HALF = HEAD_DIM / 2;    /* rotation pairs per head: dims i and i + HALF */
constexpr int QK_LANES = HALF / VEC;  /* threads per head row, one rotation pair of vectors each */

/* Shuffle mask of this thread's row; rows are whole QK_LANES-thread groups, so a group is live together. */
static __device__ __forceinline__ unsigned qk_row_mask() {
    return ((1u << QK_LANES) - 1) << ((threadIdx.x & (WARP - 1)) & ~(QK_LANES - 1));
}

static __device__ __forceinline__ float qk_row_sum(float v) {
    const unsigned mask = qk_row_mask();
    for (int o = QK_LANES / 2; o > 0; o >>= 1) v += __shfl_xor_sync(mask, v, o);
    return v;
}

/* 1 / sqrt(mean(x^2) + FLT_EPSILON) of this lane's row, the square sum added as the GEMM epilogue adds it (QuACK's
 * HeadRstd on the WGMMA fragment, epilogue subtiles 32 columns wide): lane q of a fragment quad holds columns
 * 32 s + 8 j + 2 q + e of the head (subtile s < 4, j < 4, e < 2), folds a subtile's eight from 0 in (j, e) order, adds
 * its subtile folds in s order (S_q), and the quad's butterfly (offsets 2, 1) gives (S_0 + S_2) + (S_1 + S_3); every
 * product and sum rounded on its own. Here column 8 j' + i of a half is lane j' element i, so the (s, q) fold runs
 * over lanes 4 (s % 2) + j, elements 2 q + e of half s / 2: a running sum passed up the 4 lanes, after which lanes 3
 * and 7 of the row hold the 16 folds. */
static __device__ __forceinline__ float qk_rstd(const float *x1, const float *x2) {
    static_assert(QK_LANES == 8 && VEC == 8, "the fold layout above: 8 lanes of 8 dims per half head");
    const unsigned mask = qk_row_mask();
    const int j = threadIdx.x % 4;
    float fold[2][4];  /* [half][q], complete in lanes j = 3 */
    for (int h = 0; h < 2; h++)
        for (int q = 0; q < 4; q++) {
            const float *x = h ? x2 : x1;
            float run = 0.f;
            for (int step = 0; step < 4; step++) {
                const float prev = __shfl_up_sync(mask, run, 1, 4);
                run = __fadd_rn(__fadd_rn(j ? prev : 0.f, __fmul_rn(x[2 * q], x[2 * q])),
                                __fmul_rn(x[2 * q + 1], x[2 * q + 1]));
            }
            fold[h][q] = run;
        }
    float s[4];
    for (int q = 0; q < 4; q++)
        s[q] = __fadd_rn(__fadd_rn(__fadd_rn(__shfl_sync(mask, fold[0][q], 3, QK_LANES),
                                             __shfl_sync(mask, fold[0][q], 7, QK_LANES)),
                                   __shfl_sync(mask, fold[1][q], 3, QK_LANES)),
                         __shfl_sync(mask, fold[1][q], 7, QK_LANES));
    return rsqrtf(__fadd_rn(__fadd_rn(s[0], s[2]), __fadd_rn(s[1], s[3])) / HEAD_DIM + FLT_EPSILON);
}

/* Forward math of this lane's vector pair x1, x2 (partners HALF apart) with its gains g1, g2 and cos/sin entries. */
static __device__ __forceinline__ void qk_rotate(const float *x1, const float *x2, const float *g1, const float *g2,
                                                 const float *cs, const float *sn, float *y1, float *y2) {
    const float m = qk_rstd(x1, x2);
    for (int i = 0; i < VEC; i++) {
        const float n1 = x1[i] * m * g1[i], n2 = x2[i] * m * g2[i];
        y1[i] = n1 * cs[i] + n2 * sn[i];
        y2[i] = n2 * cs[i] - n1 * sn[i];
    }
}

/* Forward of this lane's vector pair: x, g at its first vector (partners HALF further), c, s at its cos/sin entries. */
static __device__ __forceinline__ void qk_row(const bf16 *x, const float *g, const float *c, const float *s, float *y1,
                                              float *y2) {
    float x1[VEC], x2[VEC], g1[VEC], g2[VEC], cs[VEC], sn[VEC];
    load(x, x1); load(x + HALF, x2);
    load(g, g1); load(g + HALF, g2);
    load(c, cs); load(s, sn);
    qk_rotate(x1, x2, g1, g2, cs, sn, y1, y2);
}
