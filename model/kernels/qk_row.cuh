/*
 * q/k row math shared by qk_prologue.cu (training) and decode.cu (decoding), so the decoder's q and k equal
 * the training forward's bitwise. On one head row x: n = x / sqrt(mean(x^2) + eps) (F.rms_norm's fp32 epsilon),
 * m = n * g, y = m rotated by -theta over the full head (m1, m2 the halves; y1 = m1 cos + m2 sin,
 * y2 = -m1 sin + m2 cos; karpathy/nanochat 92d63d4 gpt.py:57-65). A row is QK_LANES threads, each holding dims
 * o..o+VEC-1 and their rotation partners o+HALF..; row sums reduce by shuffles within those lanes.
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

/* 1 / sqrt(mean(x^2) + FLT_EPSILON) of this lane's row. */
static __device__ __forceinline__ float qk_rstd(const float *x1, const float *x2) {
    float ss = 0.f;
    for (int i = 0; i < VEC; i++) ss += x1[i] * x1[i] + x2[i] * x2[i];
    return rsqrtf(qk_row_sum(ss) / HEAD_DIM + FLT_EPSILON);
}

/* Forward of this lane's vector pair: x, g at its first vector (partners HALF further), c, s at its cos/sin entries. */
static __device__ __forceinline__ void qk_row(const bf16 *x, const float *g, const float *c, const float *s, float *y1,
                                              float *y2) {
    float x1[VEC], x2[VEC], g1[VEC], g2[VEC], cs[VEC], sn[VEC];
    load(x, x1); load(x + HALF, x2);
    load(g, g1); load(g + HALF, g2);
    load(c, cs); load(s, sn);
    const float m = qk_rstd(x1, x2);
    for (int i = 0; i < VEC; i++) {
        const float n1 = x1[i] * m * g1[i], n2 = x2[i] * m * g2[i];
        y1[i] = n1 * cs[i] + n2 * sn[i];
        y2[i] = n2 * cs[i] - n1 * sn[i];
    }
}
