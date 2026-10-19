/*
 * The attention's q/k prologue of model/gpt.py, forward and backward, each one pass over memory: the forward row math
 * of qk_row.cuh (per head RMS norm with F.rms_norm's fp32 epsilon, the fp32 per-dim gain of q or k shared across heads,
 * rotary by -theta over the full head); Qwen3's norm -> gain -> RoPE order (transformers modeling_qwen3.py
 * Qwen3Attention.forward). fp32 math, one bf16 rounding per output.
 *
 * Backward, recomputing n from x: dm = R^T dy (dm1 = dy1 cos - dy2 sin, dm2 = dy1 sin + dy2 cos), dg = sum over rows of
 * dm * n, dn = dm * g, dx = rstd * (dn - n * mean(dn * n)).
 *
 * Layout: q, k and every output and gradient contiguous (B, T, H, 128) of one shape; gains fp32 (128,); cos/sin fp32
 * (T_table >= T, 64) contiguous, position t of a row = (row / H) % T. A row is QK_LANES threads (qk_row.cuh). One
 * launch per direction: the first half of the blocks takes the q rows, the second half the k rows. Block size: the
 * occupancy calculator's for each kernel.
 *
 * The backward's gain gradient, without atomics and in a fixed order (bitwise reproducible): the backward's blocks are
 * persistent (as many as the device holds at once, half per tensor), each looping over a fixed stride of rows and
 * accumulating its lanes' dm * n in registers; each block then sums its warps (shuffles across a warp's rows, then the
 * warps in order through shared memory) into its partial, and k_gain_sum adds the partials in block order.
 */
#include "qk_row.cuh"

/* This thread's place: q or k (by its block), whether its row exists, its first vector in the row-major (rows, 128)
 * tensors, its cos/sin vector, its dims o..o+VEC-1 (the gain's); the pair's second vector is HALF further. */
struct Row {
    bool isq, live;
    int64_t at, tab;
    int o;
};

static __device__ __forceinline__ Row locate(int64_t rows, unsigned blocks_q, int H, int T) {
    Row r;
    r.isq = blockIdx.x < blocks_q;
    const int64_t g = (int64_t)(r.isq ? blockIdx.x : blockIdx.x - blocks_q) * blockDim.x + threadIdx.x, i = g / QK_LANES;
    r.live = i < rows;
    r.o = (int)(g % QK_LANES) * VEC;
    r.at = i * HEAD_DIM + r.o;
    r.tab = (i / H % T) * HALF + r.o;
    return r;
}

__global__ void k_forward(const bf16 *q, const bf16 *k, const float *gq, const float *gk, bf16 *qo, bf16 *ko,
                          const float *cos, const float *sin, int64_t rows, unsigned blocks_q, int H, int T) {
    const Row r = locate(rows, blocks_q, H, T);
    if (!r.live) return;
    float y1[VEC], y2[VEC];
    qk_row((r.isq ? q : k) + r.at, (r.isq ? gq : gk) + r.o, cos + r.tab, sin + r.tab, y1, y2);
    bf16 *y = (r.isq ? qo : ko) + r.at;
    store(y, y1); store(y + HALF, y2);
}

constexpr int MAX_WARPS = 1024 / WARP;  /* warps per block at CUDA's 1024-thread limit */

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
    /* the warp's rows: lanes QK_LANES apart hold the same dims; lanes 0..QK_LANES-1 keep the warp's sums */
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

/* The gain gradients: block 0 q's, block 1 k's; thread d adds dim d of its tensor's nb partials in block order. */
__global__ void k_gain_sum(const float *part, unsigned nb, float *dgq, float *dgk) {
    const float *p = part + (int64_t)blockIdx.x * nb * HEAD_DIM + threadIdx.x;
    float sum = 0.f;
    for (unsigned j = 0; j < nb; j++) sum += p[(int64_t)j * HEAD_DIM];
    (blockIdx.x ? dgk : dgq)[threadIdx.x] = sum;
}

/* The occupancy calculator's block size for kernel f (a multiple of the warp size, so rows of QK_LANES threads never
 * straddle warps), computed once per kernel. */
template <typename F> static int block_size(F f, int &cached) {
    if (!cached) {
        int min_grid;
        cudaOccupancyMaxPotentialBlockSize(&min_grid, &cached, f);
    }
    return cached;
}

static unsigned blocks(int64_t rows, int threads) { return (unsigned)((rows * QK_LANES + threads - 1) / threads); }

extern "C" void qkp_forward(const void *q, const void *k, const void *gq, const void *gk, void *qo, void *ko,
                            const void *cos, const void *sin, long long rows, int H, int T, void *stream) {
    static int cached = 0;
    const int threads = block_size(k_forward, cached);
    const unsigned nb = blocks(rows, threads);
    k_forward<<<2 * nb, threads, 0, (cudaStream_t)stream>>>(
        (const bf16 *)q, (const bf16 *)k, (const float *)gq, (const float *)gk, (bf16 *)qo, (bf16 *)ko,
        (const float *)cos, (const float *)sin, rows, nb, H, T);
}

/* The backward's blocks per tensor for `rows` rows: half the k_backward blocks the device holds at once (SMs x the
 * occupancy calculator's blocks per SM at its block size), at most one per pass of rows. Fixed for a device and rows,
 * and so is the gain gradient's summation order. */
extern "C" unsigned qkp_backward_blocks(long long rows) {
    static int cached = 0, resident = 0;
    const int threads = block_size(k_backward, cached);
    if (!resident) {
        int device, sms, per_sm;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k_backward, threads, 0);
        resident = sms * per_sm;
    }
    const unsigned half = resident / 2 > 1 ? resident / 2 : 1, need = blocks(rows, threads);
    return half < need ? half : need;
}

/* part: fp32 (2 * qkp_backward_blocks(rows), 128), the blocks' partial gain gradients. */
extern "C" void qkp_backward(const void *q, const void *k, const void *gq, const void *gk, const void *dqo,
                             const void *dko, void *dq, void *dk, void *dgq, void *dgk, void *part, const void *cos,
                             const void *sin, long long rows, int H, int T, void *stream) {
    static int cached = 0;
    const int threads = block_size(k_backward, cached);
    const unsigned nb = qkp_backward_blocks(rows);
    k_backward<<<2 * nb, threads, 0, (cudaStream_t)stream>>>(
        (const bf16 *)q, (const bf16 *)k, (const float *)gq, (const float *)gk, (const bf16 *)dqo, (const bf16 *)dko,
        (bf16 *)dq, (bf16 *)dk, (const float *)cos, (const float *)sin, (float *)part, rows, nb, H, T);
    k_gain_sum<<<2, HEAD_DIM, 0, (cudaStream_t)stream>>>((const float *)part, nb, (float *)dgq, (float *)dgk);
}
