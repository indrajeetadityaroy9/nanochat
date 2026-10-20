/*
 * Block Attention Residuals' depth attention (ablations/depth_attention.py; Chen et al., arXiv 2603.15031 Eqs. 2-6) of
 * one pseudo-query q (C,) over n sources v_0..v_{n-1} (each (rows, C) bf16, read through a pointer table, no stacked
 * copy), one pass over memory per direction. Per row:
 *   forward   r_k = 1 / sqrt(mean(v_k^2) + eps), l_k = r_k q . v_k, lse = logsumexp(l), h = sum_k exp(l_k - lse) v_k
 *             (online softmax over the sources)
 *   backward  D = dh . h, w_k = exp(l_k - lse), dl_k = w_k (dh . v_k - D),
 *             dv_k = w_k dh + dl_k r_k (q - r_k l_k v_k / C), dq = sum over rows of dl_k r_k v_k
 * (r_k v_k is RMSNorm(v_k), gain-free with F.rms_norm's fp32 epsilon, as gpt.norm). fp32 math, one bf16 rounding per
 * output. A warp per row, VEC bf16 per lane per 16-byte access, CH accesses per lane: C = CH * WARP * VEC.
 * dq without atomics: each warp of a persistent grid sums its rows' dq in registers into its own partial, and
 * k_dq_sum adds the partials in warp order (bitwise reproducible).
 */
#include <stdio.h>
#include <stdlib.h>

#include "../nanochat/csrc/qk_row.cuh"

/* MAX_SOURCES (nvcc -D, ablations/depth_attention.py): the most sources an attention reads */
struct Sources { const bf16 *p[MAX_SOURCES]; };
struct Grads { bf16 *p[MAX_SOURCES]; };

static __device__ __forceinline__ float warp_sum(float v) {
    for (int o = WARP / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

template <int CH>
__global__ void k_forward(Sources src, int n, const float *q, bf16 *h, float *lse, int64_t rows) {
    constexpr int C = CH * WARP * VEC;
    const int col = (int)(threadIdx.x % WARP) * VEC;
    const int64_t warps = (int64_t)gridDim.x * blockDim.x / WARP;
    float qv[CH][VEC];
    for (int c = 0; c < CH; c++) load(q + c * WARP * VEC + col, qv[c]);
    for (int64_t i = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / WARP; i < rows; i += warps) {
        float m = -INFINITY, s = 0.f, acc[CH][VEC] = {};
        for (int k = 0; k < n; k++) {
            float v[CH][VEC], ss = 0.f, d = 0.f;
            for (int c = 0; c < CH; c++) {
                load(src.p[k] + i * C + c * WARP * VEC + col, v[c]);
                for (int j = 0; j < VEC; j++) { ss += v[c][j] * v[c][j]; d += qv[c][j] * v[c][j]; }
            }
            const float l = rsqrtf(warp_sum(ss) / C + FLT_EPSILON) * warp_sum(d);
            const float mn = fmaxf(m, l), a = expf(m - mn), b = expf(l - mn);
            s = s * a + b;
            for (int c = 0; c < CH; c++)
                for (int j = 0; j < VEC; j++) acc[c][j] = acc[c][j] * a + b * v[c][j];
            m = mn;
        }
        for (int c = 0; c < CH; c++) {
            for (int j = 0; j < VEC; j++) acc[c][j] /= s;
            store(h + i * C + c * WARP * VEC + col, acc[c]);
        }
        if (col == 0) lse[i] = m + logf(s);
    }
}

/* part: (warps of the grid, C) fp32, each warp's dq */
template <int CH>
__global__ void k_backward(Sources src, Grads grad, int n, const float *q, const bf16 *h, const float *lse,
                           const bf16 *dh, float *part, int64_t rows) {
    constexpr int C = CH * WARP * VEC;
    const int col = (int)(threadIdx.x % WARP) * VEC;
    const int64_t warp = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / WARP;
    const int64_t warps = (int64_t)gridDim.x * blockDim.x / WARP;
    float qv[CH][VEC], dq[CH][VEC] = {};
    for (int c = 0; c < CH; c++) load(q + c * WARP * VEC + col, qv[c]);
    for (int64_t i = warp; i < rows; i += warps) {
        float g[CH][VEC], D = 0.f;
        for (int c = 0; c < CH; c++) {
            float y[VEC];
            load(dh + i * C + c * WARP * VEC + col, g[c]);
            load(h + i * C + c * WARP * VEC + col, y);
            for (int j = 0; j < VEC; j++) D += g[c][j] * y[j];
        }
        D = warp_sum(D);
        const float L = lse[i];
        for (int k = 0; k < n; k++) {
            float v[CH][VEC], ss = 0.f, d = 0.f, a = 0.f;
            for (int c = 0; c < CH; c++) {
                load(src.p[k] + i * C + c * WARP * VEC + col, v[c]);
                for (int j = 0; j < VEC; j++) {
                    ss += v[c][j] * v[c][j]; d += qv[c][j] * v[c][j]; a += g[c][j] * v[c][j];
                }
            }
            const float r = rsqrtf(warp_sum(ss) / C + FLT_EPSILON), l = r * warp_sum(d);
            const float w = expf(l - L), dl = w * (warp_sum(a) - D), cq = dl * r, cv = cq * r * l / C;
            for (int c = 0; c < CH; c++) {
                float dv[VEC];
                for (int j = 0; j < VEC; j++) {
                    dv[j] = w * g[c][j] + cq * qv[c][j] - cv * v[c][j];
                    dq[c][j] += cq * v[c][j];
                }
                store(grad.p[k] + i * C + c * WARP * VEC + col, dv);
            }
        }
    }
    for (int c = 0; c < CH; c++)
        for (int j = 0; j < VEC; j++) part[warp * C + c * WARP * VEC + col + j] = dq[c][j];
}

/* A block per WARP columns, WARP threads per column: thread (g, c) sums the partials g, g + WARP, ... of its column in
 * order, then thread (0, c) the WARP sums in order (bitwise reproducible). */
__global__ void k_dq_sum(const float *part, int64_t warps, int C, float *dq) {
    __shared__ float sums[WARP][WARP];
    const int c = threadIdx.x % WARP, g = threadIdx.x / WARP, d = blockIdx.x * WARP + c;
    float sum = 0.f;
    for (int64_t w = g; w < warps; w += WARP) sum += part[w * C + d];
    sums[g][c] = sum;
    __syncthreads();
    if (g == 0) {
        for (int j = 1; j < WARP; j++) sum += sums[j][c];
        dq[d] = sum;
    }
}

struct Grid { int blocks, threads; };

/* The occupancy calculator's block size for f, and every block the device holds at once at that size. */
template <typename F> static Grid grid_of(F f) {
    int min_grid, device, sms, per_sm;
    Grid g;
    cudaOccupancyMaxPotentialBlockSize(&min_grid, &g.threads, f);
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, f, g.threads, 0);
    g.blocks = sms * per_sm;
    return g;
}

template <int CH> static Grid forward_grid() { static const Grid g = grid_of(k_forward<CH>); return g; }
template <int CH> static Grid backward_grid() { static const Grid g = grid_of(k_backward<CH>); return g; }

static Sources sources_of(const void *const *p, int n) {
    Sources s;
    for (int k = 0; k < n; k++) s.p[k] = (const bf16 *)p[k];
    return s;
}

/* The widths the experiments use: C = 256 (d8) and 512 (d16) */
#define DISPATCH(C, CALL)                                                                      \
    switch ((C) / (WARP * VEC)) {                                                               \
        case 1: CALL(1); break;                                                                 \
        case 2: CALL(2); break;                                                                 \
        default: fprintf(stderr, "depth_attention: unsupported width %d\n", (int)(C)); abort(); \
    }

extern "C" void da_forward(const void *const *sources, int n, const void *q, void *h, void *lse, long long rows, int C,
                           void *stream) {
#define FORWARD(CH)                                                                                     \
    k_forward<CH><<<forward_grid<CH>().blocks, forward_grid<CH>().threads, 0, (cudaStream_t)stream>>>( \
        sources_of(sources, n), n, (const float *)q, (bf16 *)h, (float *)lse, rows)
    DISPATCH(C, FORWARD)
}

/* Warps of the backward's grid: the rows of its partials. */
extern "C" long long da_backward_warps(int C) {
    Grid g = {};
#define WARPS(CH) g = backward_grid<CH>()
    DISPATCH(C, WARPS)
    return (long long)g.blocks * g.threads / WARP;
}

/* part: fp32 (da_backward_warps(C), C) */
extern "C" void da_backward(const void *const *sources, void *const *grads, int n, const void *q, const void *h,
                            const void *lse, const void *dh, void *part, void *dq, long long rows, int C,
                            void *stream) {
    Grads g;
    for (int k = 0; k < n; k++) g.p[k] = (bf16 *)grads[k];
#define BACKWARD(CH)                                                                                         \
    k_backward<CH><<<backward_grid<CH>().blocks, backward_grid<CH>().threads, 0, (cudaStream_t)stream>>>(   \
        sources_of(sources, n), g, n, (const float *)q, (const bf16 *)h, (const float *)lse, (const bf16 *)dh, \
        (float *)part, rows)
    DISPATCH(C, BACKWARD)
    k_dq_sum<<<C / WARP, WARP * WARP, 0, (cudaStream_t)stream>>>((const float *)part, da_backward_warps(C), C,
                                                                 (float *)dq);
}
