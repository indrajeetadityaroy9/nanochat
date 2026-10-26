/*
 * FA4's backward row statistics at its padded rows, from a precomputed dpsum (model/ops/flash_attn.py): the part of
 * FA4's backward preprocess (flash_attn/cute/flash_bwd_preprocess.py) that remains once the o_proj dX GEMM's epilogue
 * has summed dpsum = rowsum(dO O) per (row, head) (model/ops/linear.py dx_dpsum), so O and dO are not read again:
 *   dpsum_pad[h, p] = dpsum[r, h], lse_log2[h, p] = lse[h, r] log2(e) (0 where lse = -inf)
 * at FA4's padded row p = start_b + r - cu[b] of row r in segment b, start_b = (cu[b] + b m) / m * m
 * (seqlen_info.py offset_batch, padded=True), m FA4's backward m-block. Each segment's rows past its length up to a
 * multiple of m, which the backward's last m-block loads, get dpsum 0 and lse_log2 +inf (the preprocess's lse = +inf
 * there), so their P is 0; the rows between start_b + round_up(len_b, m) and start_{b+1} are never read.
 *
 * Layout: dpsum (rows, H) fp32 (the epilogue's per-(row, 128-column tile) partials), lse (H, lse_stride) fp32 (FA4's
 * forward lse over the micro-batch's rows), dpsum_pad and lse_log2 (H, padded) fp32, cu (segments + 1) int32. One
 * thread per padded row: the segment from a binary search over start_b, then H loads and stores.
 */
#include "common.cuh"

constexpr int THREADS = 256;
/* log2(e) rounded to fp32: the preprocess's LOG2_E = math.log2(math.e) as a Float32 constant */
constexpr float LOG2_E = 1.4426950408889634f;

static __device__ __forceinline__ long long start(const int *cu, int b, int m) {
    return ((long long)cu[b] + (long long)b * m) / m * m;
}

__global__ void __launch_bounds__(THREADS)
k_rows(const float *__restrict__ dpsum, const float *__restrict__ lse, long long lse_stride, const int *__restrict__ cu,
       int segments, int H, int m, long long padded, float *__restrict__ dpsum_pad, float *__restrict__ lse_log2) {
    const long long p = (long long)blockIdx.x * THREADS + threadIdx.x;
    if (p >= padded) return;
    int lo = 0, hi = segments - 1;  /* the last segment b with start_b <= p */
    while (lo < hi) {
        const int mid = (lo + hi + 1) / 2;
        if (start(cu, mid, m) <= p) lo = mid; else hi = mid - 1;
    }
    const int b = lo, len = cu[b + 1] - cu[b];
    const long long i = p - start(cu, b, m);
    if (i < len) {
        const long long r = cu[b] + i;
        for (int h = 0; h < H; h++) {
            const float l = lse[h * lse_stride + r];
            dpsum_pad[h * padded + p] = dpsum[r * H + h];
            lse_log2[h * padded + p] = l == -INFINITY ? 0.f : l * LOG2_E;
        }
    } else if (i < (len + m - 1) / m * m) {
        for (int h = 0; h < H; h++) {
            dpsum_pad[h * padded + p] = 0.f;
            lse_log2[h * padded + p] = INFINITY;
        }
    }
}

extern "C" void fa4_rows(const void *dpsum, const void *lse, long long lse_stride, const void *cu, int segments, int H,
                         int m, long long padded, void *dpsum_pad, void *lse_log2, void *stream) {
    k_rows<<<(unsigned)((padded + THREADS - 1) / THREADS), THREADS, 0, (cudaStream_t)stream>>>(
        (const float *)dpsum, (const float *)lse, lse_stride, (const int *)cu, segments, H, m, padded,
        (float *)dpsum_pad, (float *)lse_log2);
}
