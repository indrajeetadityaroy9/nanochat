/*
 * Device code shared by ops/qk_prologue.cu (training) and inference/decode.cu (decoding), so that given equal inputs
 * the decoder's q and k equal the training forward's bit for bit: the 16-byte load/store helpers and the forward row
 * math of gpt.py's qk_norm_rope on one head row x of HEAD_DIM bf16, n = x / sqrt(mean(x^2) + eps) with F.rms_norm's
 * fp32 epsilon, m = n * g with the fp32 per-dim gain g, y = m rotated by -theta over the full head (half split: m1 =
 * dims 0..63, m2 = dims 64..127, y1 = m1 cos + m2 sin, y2 = -m1 sin + m2 cos; karpathy/nanochat 92d63d4
 * gpt.py:57-65). A row is QK_LANES consecutive threads, each owning a rotation pair of VEC-dim vectors (offsets o =
 * lane * VEC and o + HALF); the row's sums are reduced by shuffles within its lanes.
 */
#pragma once

#include <float.h>
#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

typedef __nv_bfloat16 bf16;

constexpr int WARP = 32;                           /* threads per warp */
constexpr int HEAD_DIM = 128;                      /* model/gpt.py gpt_config head_dim */
constexpr int HALF = HEAD_DIM / 2;                 /* rotation pairs per head: dims i and i + HALF */
constexpr int VEC = sizeof(uint4) / sizeof(bf16);  /* bf16 per 16-byte access */
constexpr int QK_LANES = HALF / VEC;               /* threads per head row, one rotation pair of vectors each */

/* VEC bf16 (one 16-byte load) as fp32. */
static __device__ __forceinline__ void load(const bf16 *p, float *v) {
    const uint4 u = *reinterpret_cast<const uint4 *>(p);
    const __nv_bfloat162 *h = reinterpret_cast<const __nv_bfloat162 *>(&u);
    for (int i = 0; i < VEC / 2; i++) {
        const float2 f = __bfloat1622float2(h[i]);
        v[2 * i] = f.x; v[2 * i + 1] = f.y;
    }
}

/* VEC fp32 entries (offsets are multiples of VEC, so each float4 is aligned). */
static __device__ __forceinline__ void load(const float *p, float *v) {
    for (int i = 0; i < VEC; i += 4) *reinterpret_cast<float4 *>(v + i) = *reinterpret_cast<const float4 *>(p + i);
}

/* VEC fp32 rounded to bf16, one 16-byte store. */
static __device__ __forceinline__ void store(bf16 *p, const float *v) {
    uint4 u;
    __nv_bfloat162 *h = reinterpret_cast<__nv_bfloat162 *>(&u);
    for (int i = 0; i < VEC / 2; i++) h[i] = __floats2bfloat162_rn(v[2 * i], v[2 * i + 1]);
    *reinterpret_cast<uint4 *>(p) = u;
}

/* Lanes of this thread's row, the shuffle mask: rows are whole groups of QK_LANES threads, so a group is live
 * together. */
static __device__ __forceinline__ unsigned qk_row_mask() {
    return ((1u << QK_LANES) - 1) << ((threadIdx.x & (WARP - 1)) & ~(QK_LANES - 1));
}

static __device__ __forceinline__ float qk_row_sum(float v) {
    const unsigned mask = qk_row_mask();
    for (int o = QK_LANES / 2; o > 0; o >>= 1) v += __shfl_xor_sync(mask, v, o);
    return v;
}

/* 1 / sqrt(mean(x^2) + FLT_EPSILON) of the row whose pair of vectors x1, x2 this lane holds. */
static __device__ __forceinline__ float qk_rstd(const float *x1, const float *x2) {
    float ss = 0.f;
    for (int i = 0; i < VEC; i++) ss += x1[i] * x1[i] + x2[i] * x2[i];
    return rsqrtf(qk_row_sum(ss) / HEAD_DIM + FLT_EPSILON);
}

/* The forward of this lane's pair of vectors: x, g, c, s point at its first vector of the head row, the gain and the
 * position's cos/sin table row (the pair's second vector of x and g is HALF further); y1, y2 the fp32 outputs. */
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
