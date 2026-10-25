/*
 * q/k prologue of model/gpt.py, forward and backward, one pass over memory each: qk_row.cuh's row math in Qwen3's
 * (arXiv 2505.09388) norm -> gain -> RoPE order (transformers modeling_qwen3.py Qwen3Attention.forward).
 *
 * Backward, recomputing n from x: dm = R^T dy (dm1 = dy1 cos - dy2 sin, dm2 = dy1 sin + dy2 cos), dg = sum over rows of
 * dm * n, dn = dm * g, dx = rstd * (dn - n * mean(dn * n)).
 *
 * Layout: q, k, outputs, gradients contiguous (B, T, H, HEAD_DIM); position of row r = (r / H) % T. The first half of
 * the blocks takes q, the second k. Block size: the occupancy calculator's (a multiple of WARP, so rows never straddle
 * warps).
 *
 * Deterministic gain gradient without atomics: persistent blocks (half the device's resident blocks per tensor) stride
 * over rows accumulating dm * n in registers, reduce their warps in order into a per-block partial, and k_gain_sum adds
 * the partials in block order.
 */
#include "qk_row.cuh"

__global__ void k_forward(const bf16 *q, const bf16 *k, const float *gq, const float *gk, bf16 *qo, bf16 *ko,
                          const float *cos, const float *sin, int64_t rows, unsigned nb, int H, int T) {
    const bool isq = blockIdx.x < nb;
    const int64_t g = (int64_t)(isq ? blockIdx.x : blockIdx.x - nb) * blockDim.x + threadIdx.x, i = g / QK_LANES;
    if (i >= rows) return;
    const int o = (int)(g % QK_LANES) * VEC;
    const int64_t at = i * HEAD_DIM + o, tab = (i / H % T) * HALF + o;
    float y1[VEC], y2[VEC];
    qk_row((isq ? q : k) + at, (isq ? gq : gk) + o, cos + tab, sin + tab, y1, y2);
    bf16 *y = (isq ? qo : ko) + at;
    store(y, y1); store(y + HALF, y2);
}

constexpr int MAX_WARPS = 1024 / WARP;  /* CUDA max threads per block (Programming Guide, Compute Capabilities) */

__global__ void k_backward(const bf16 *q, const bf16 *k, const float *gq, const float *gk, const bf16 *dqo,
                           const bf16 *dko, bf16 *dq, bf16 *dk, const float *cos, const float *sin, float *part,
                           int64_t rows, unsigned nb, int H, int T) {
    __shared__ float warp_part[MAX_WARPS][HEAD_DIM];  /* each warp's gain gradient */
    const bool isq = blockIdx.x < nb;
    const int64_t first = isq ? blockIdx.x : blockIdx.x - nb, per_pass = blockDim.x / QK_LANES;
    const int o = (int)(threadIdx.x % QK_LANES) * VEC;
    const bf16 *x_in = isq ? q : k, *dy_in = isq ? dqo : dko;
    bf16 *dx_out = isq ? dq : dk;
    float g1[VEC], g2[VEC];
    load((isq ? gq : gk) + o, g1); load((isq ? gq : gk) + o + HALF, g2);
    float a1[VEC] = {}, a2[VEC] = {};  /* this lane's sum of dm * n over its rows */
    for (int64_t i = first * per_pass + threadIdx.x / QK_LANES; i < rows; i += (int64_t)nb * per_pass) {
        const int64_t at = i * HEAD_DIM + o, tab = (i / H % T) * HALF + o;
        float x1[VEC], x2[VEC], d1[VEC], d2[VEC], c[VEC], s[VEC];
        load(x_in + at, x1); load(x_in + at + HALF, x2);
        load(dy_in + at, d1); load(dy_in + at + HALF, d2);
        load(cos + tab, c); load(sin + tab, s);
        const float m = qk_rstd(x1, x2);
        float sd = 0.f;  /* sum dn * n */
        for (int j = 0; j < VEC; j++) {
            const float n1 = x1[j] * m, n2 = x2[j] * m;
            const float dm1 = d1[j] * c[j] - d2[j] * s[j], dm2 = d1[j] * s[j] + d2[j] * c[j];
            a1[j] += dm1 * n1; a2[j] += dm2 * n2;
            d1[j] = dm1 * g1[j]; d2[j] = dm2 * g2[j];
            x1[j] = n1; x2[j] = n2;
            sd += d1[j] * n1 + d2[j] * n2;
        }
        const float b = qk_row_sum(sd) / HEAD_DIM;
        for (int j = 0; j < VEC; j++) {
            d1[j] = m * (d1[j] - x1[j] * b);
            d2[j] = m * (d2[j] - x2[j] * b);
        }
        store(dx_out + at, d1); store(dx_out + at + HALF, d2);
    }
    /* lanes QK_LANES apart hold the same dims; lanes 0..QK_LANES-1 keep the warp's sums */
    for (int w = QK_LANES; w < WARP; w <<= 1)
        for (int j = 0; j < VEC; j++) {
            a1[j] += __shfl_xor_sync(0xffffffffu, a1[j], w);
            a2[j] += __shfl_xor_sync(0xffffffffu, a2[j], w);
        }
    if ((threadIdx.x & (WARP - 1)) < QK_LANES)
        for (int j = 0; j < VEC; j++) {
            warp_part[threadIdx.x / WARP][o + j] = a1[j];
            warp_part[threadIdx.x / WARP][o + HALF + j] = a2[j];
        }
    __syncthreads();
    for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) {
        float sum = 0.f;
        for (int w = 0; w < (int)(blockDim.x / WARP); w++) sum += warp_part[w][d];
        part[(int64_t)blockIdx.x * HEAD_DIM + d] = sum;
    }
}

/* Block 0: q's gain gradient, block 1: k's; thread d sums dim d of the nb partials in block order. */
__global__ void k_gain_sum(const float *part, unsigned nb, float *dgq, float *dgk) {
    const float *p = part + (int64_t)blockIdx.x * nb * HEAD_DIM + threadIdx.x;
    float sum = 0.f;
    for (unsigned j = 0; j < nb; j++) sum += p[(int64_t)j * HEAD_DIM];
    (blockIdx.x ? dgk : dgq)[threadIdx.x] = sum;
}

/* The occupancy calculator's block size for f. */
template <typename F> static int block_size(F f) {
    int min_grid, threads;
    cudaOccupancyMaxPotentialBlockSize(&min_grid, &threads, f);
    return threads;
}

static int backward_threads() {
    static const int threads = block_size(k_backward);
    return threads;
}

extern "C" void qkp_forward(const void *q, const void *k, const void *gq, const void *gk, void *qo, void *ko,
                            const void *cos, const void *sin, long long rows, int H, int T, void *stream) {
    static const int threads = block_size(k_forward);
    const unsigned nb = (rows * QK_LANES + threads - 1) / threads;
    k_forward<<<2 * nb, threads, 0, (cudaStream_t)stream>>>(
        (const bf16 *)q, (const bf16 *)k, (const float *)gq, (const float *)gk, (bf16 *)qo, (bf16 *)ko,
        (const float *)cos, (const float *)sin, rows, nb, H, T);
}

/* Blocks per tensor: half the device's resident k_backward blocks, fixed per device, so the summation order is fixed;
 * blocks past the last row contribute exact zeros. */
extern "C" int qkp_backward_blocks() {
    static const int half = resident_blocks(k_backward, backward_threads()) / 2;
    return half;
}

/* part: fp32 (2 * qkp_backward_blocks(), HEAD_DIM). */
extern "C" void qkp_backward(const void *q, const void *k, const void *gq, const void *gk, const void *dqo,
                             const void *dko, void *dq, void *dk, void *dgq, void *dgk, void *part, const void *cos,
                             const void *sin, long long rows, int H, int T, void *stream) {
    const unsigned nb = qkp_backward_blocks();
    k_backward<<<2 * nb, backward_threads(), 0, (cudaStream_t)stream>>>(
        (const bf16 *)q, (const bf16 *)k, (const float *)gq, (const float *)gk, (const bf16 *)dqo, (const bf16 *)dko,
        (bf16 *)dq, (bf16 *)dk, (const float *)cos, (const float *)sin, (float *)part, rows, nb, H, T);
    k_gain_sum<<<2, HEAD_DIM, 0, (cudaStream_t)stream>>>((const float *)part, nb, (float *)dgq, (float *)dgk);
}
