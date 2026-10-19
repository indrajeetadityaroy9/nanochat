/*
 * The attention's q/k prologue of model/gpt.py, forward and backward, each one pass over memory: per head row x of 128
 * bf16, n = x / sqrt(mean(x^2) + eps) with F.rms_norm's fp32 epsilon, m = n * g with the fp32 per-dim gain g of q or k
 * (shared across heads), y = m rotated by -theta over the full head (half split: m1 = dims 0..63, m2 = dims 64..127,
 * y1 = m1 cos + m2 sin, y2 = -m1 sin + m2 cos; karpathy/nanochat 92d63d4 gpt.py:57-65); Qwen3's norm -> gain -> RoPE
 * order (transformers modeling_qwen3.py Qwen3Attention.forward). fp32 math, one bf16 rounding per output.
 *
 * Backward, recomputing n from x: dm = R^T dy (dm1 = dy1 cos - dy2 sin, dm2 = dy1 sin + dy2 cos), dg = sum over rows of
 * dm * n, dn = dm * g, dx = rstd * (dn - n * mean(dn * n)).
 *
 * Layout: q, k (B, T, H, 128) of one shape with dense heads (stride(3) = 1, stride(2) = 128, stride(0) = T * stride(1))
 * at any token stride stride(1); outputs and gradients contiguous (B, T, H, 128); gains fp32 (128,); cos/sin fp32
 * (T_table >= T, 64) contiguous, position t of a row = (row / H) % T. A row is LANES = 8 consecutive threads, each
 * owning a rotation pair of 16-byte vectors (offsets o = lane * VEC and o + 64) with the matching VEC entries of cos,
 * sin and the gain. The row's sums are reduced by shuffles within its lanes. One launch: the first half of the blocks
 * takes the q rows, the second half the k rows, so every block holds rows of one tensor and reduces its gain gradient
 * alone (shuffles across the warp's rows, shared-memory atomics across warps, one global fp32 atomicAdd per dim and
 * block). Block size: the occupancy calculator's for each kernel.
 */
#include <cfloat>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

typedef __nv_bfloat16 bf16;

constexpr int HEAD_DIM = 128;                      /* model/gpt.py gpt_config head_dim */
constexpr int HALF = HEAD_DIM / 2;                 /* rotation pairs per head: dims i and i + HALF */
constexpr int VEC = sizeof(uint4) / sizeof(bf16);  /* bf16 per 16-byte access */
constexpr int LANES = HALF / VEC;                  /* threads per head row, one rotation pair of vectors each */
constexpr int WARP = 32;

static __device__ __forceinline__ void load(const bf16 *p, float *v) {
    const uint4 u = *reinterpret_cast<const uint4 *>(p);
    const __nv_bfloat162 *h = reinterpret_cast<const __nv_bfloat162 *>(&u);
    for (int i = 0; i < VEC / 2; i++) {
        const float2 f = __bfloat1622float2(h[i]);
        v[2 * i] = f.x; v[2 * i + 1] = f.y;
    }
}

/* VEC fp32 entries (offsets are multiples of VEC, so each float4 is aligned) */
static __device__ __forceinline__ void load(const float *p, float *v) {
    for (int i = 0; i < VEC; i += 4) *reinterpret_cast<float4 *>(v + i) = *reinterpret_cast<const float4 *>(p + i);
}

static __device__ __forceinline__ void store(bf16 *p, const float *v) {
    uint4 u;
    __nv_bfloat162 *h = reinterpret_cast<__nv_bfloat162 *>(&u);
    for (int i = 0; i < VEC / 2; i++) h[i] = __floats2bfloat162_rn(v[2 * i], v[2 * i + 1]);
    *reinterpret_cast<uint4 *>(p) = u;
}

/* This thread's place: q or k (by its block), whether its row exists, its first vector in the input (read at token
 * stride sq or sk elements) and in the contiguous outputs, its cos/sin vector, its dims o..o+VEC-1 (the gain's); the
 * pair's second vector is HALF further. */
struct Row {
    bool isq, live;
    int64_t x, y, tab;
    int o;
};

static __device__ __forceinline__ Row locate(int64_t rows, unsigned blocks_q, int H, int64_t sq, int64_t sk, int T) {
    Row r;
    r.isq = blockIdx.x < blocks_q;
    const int64_t g = (int64_t)(r.isq ? blockIdx.x : blockIdx.x - blocks_q) * blockDim.x + threadIdx.x, i = g / LANES;
    const int64_t token = i / H;
    r.live = i < rows;
    r.o = (int)(g % LANES) * VEC;
    r.x = token * (r.isq ? sq : sk) + (i % H) * HEAD_DIM + r.o;
    r.y = i * HEAD_DIM + r.o;
    r.tab = (token % T) * HALF + r.o;
    return r;
}

/* Lanes of this thread's row, the shuffle mask: rows are whole groups of LANES threads, so a group is live together. */
static __device__ __forceinline__ unsigned row_mask() {
    return ((1u << LANES) - 1) << ((threadIdx.x & (WARP - 1)) & ~(LANES - 1));
}

static __device__ __forceinline__ float row_sum(float v, unsigned mask) {
    for (int o = LANES / 2; o > 0; o >>= 1) v += __shfl_xor_sync(mask, v, o);
    return v;
}

/* 1 / sqrt(mean(x^2) + FLT_EPSILON) of the row whose pair of vectors x1, x2 this lane holds. */
static __device__ __forceinline__ float rstd(const float *x1, const float *x2) {
    float ss = 0.f;
    for (int i = 0; i < VEC; i++) ss += x1[i] * x1[i] + x2[i] * x2[i];
    return rsqrtf(row_sum(ss, row_mask()) / HEAD_DIM + FLT_EPSILON);
}

__global__ void k_forward(const bf16 *q, const bf16 *k, const float *gq, const float *gk, bf16 *qo, bf16 *ko,
                          const float *cos, const float *sin, int64_t rows, unsigned blocks_q, int H, int64_t sq,
                          int64_t sk, int T) {
    const Row r = locate(rows, blocks_q, H, sq, sk, T);
    if (!r.live) return;
    const bf16 *x = (r.isq ? q : k) + r.x;
    const float *g = r.isq ? gq : gk;
    float x1[VEC], x2[VEC], g1[VEC], g2[VEC], c[VEC], s[VEC];
    load(x, x1); load(x + HALF, x2);
    load(g + r.o, g1); load(g + r.o + HALF, g2);
    load(cos + r.tab, c); load(sin + r.tab, s);
    const float m = rstd(x1, x2);
    for (int i = 0; i < VEC; i++) {
        const float n1 = x1[i] * m * g1[i], n2 = x2[i] * m * g2[i];
        x1[i] = n1 * c[i] + n2 * s[i];
        x2[i] = n2 * c[i] - n1 * s[i];
    }
    bf16 *y = (r.isq ? qo : ko) + r.y;
    store(y, x1); store(y + HALF, x2);
}

__global__ void k_backward(const bf16 *q, const bf16 *k, const float *gq, const float *gk, const bf16 *dqo,
                           const bf16 *dko, bf16 *dq, bf16 *dk, float *dgq, float *dgk, const float *cos,
                           const float *sin, int64_t rows, unsigned blocks_q, int H, int64_t sq, int64_t sk, int T) {
    __shared__ float part[HEAD_DIM];  /* the block's gain gradient */
    for (int i = threadIdx.x; i < HEAD_DIM; i += blockDim.x) part[i] = 0.f;
    __syncthreads();
    const Row r = locate(rows, blocks_q, H, sq, sk, T);
    float a1[VEC] = {}, a2[VEC] = {};  /* this row's dm * n at the lane's dims */
    if (r.live) {
        const bf16 *x = (r.isq ? q : k) + r.x, *dy = (r.isq ? dqo : dko) + r.y;
        const float *g = r.isq ? gq : gk;
        float x1[VEC], x2[VEC], g1[VEC], g2[VEC], d1[VEC], d2[VEC], c[VEC], s[VEC];
        load(x, x1); load(x + HALF, x2);
        load(g + r.o, g1); load(g + r.o + HALF, g2);
        load(dy, d1); load(dy + HALF, d2);
        load(cos + r.tab, c); load(sin + r.tab, s);
        const float m = rstd(x1, x2);
        float sd = 0.f;  /* sum dn * n */
        for (int i = 0; i < VEC; i++) {
            const float n1 = x1[i] * m, n2 = x2[i] * m;
            const float dm1 = d1[i] * c[i] - d2[i] * s[i], dm2 = d1[i] * s[i] + d2[i] * c[i];
            a1[i] = dm1 * n1; a2[i] = dm2 * n2;
            d1[i] = dm1 * g1[i]; d2[i] = dm2 * g2[i];
            x1[i] = n1; x2[i] = n2;
            sd += d1[i] * n1 + d2[i] * n2;
        }
        const float b = row_sum(sd, row_mask()) / HEAD_DIM;
        for (int i = 0; i < VEC; i++) {
            d1[i] = m * (d1[i] - x1[i] * b);
            d2[i] = m * (d2[i] - x2[i] * b);
        }
        bf16 *dx = (r.isq ? dq : dk) + r.y;
        store(dx, d1); store(dx + HALF, d2);
    }
    /* the warp's rows: lanes LANES apart hold the same dims */
    for (int o = LANES; o < WARP; o <<= 1)
        for (int i = 0; i < VEC; i++) {
            a1[i] += __shfl_xor_sync(0xffffffffu, a1[i], o);
            a2[i] += __shfl_xor_sync(0xffffffffu, a2[i], o);
        }
    if ((threadIdx.x & (WARP - 1)) < LANES)
        for (int i = 0; i < VEC; i++) {
            atomicAdd(&part[r.o + i], a1[i]);
            atomicAdd(&part[r.o + HALF + i], a2[i]);
        }
    __syncthreads();
    float *dg = r.isq ? dgq : dgk;
    for (int i = threadIdx.x; i < HEAD_DIM; i += blockDim.x) atomicAdd(&dg[i], part[i]);
}

/* The occupancy calculator's block size for kernel f (a multiple of the warp size, so rows of LANES threads never
 * straddle warps), computed once per kernel. */
template <typename F> static int block_size(F f, int &cached) {
    if (!cached) {
        int min_grid;
        cudaOccupancyMaxPotentialBlockSize(&min_grid, &cached, f);
    }
    return cached;
}

static unsigned blocks(int64_t rows, int threads) { return (unsigned)((rows * LANES + threads - 1) / threads); }

extern "C" void qkp_forward(const void *q, const void *k, const void *gq, const void *gk, void *qo, void *ko,
                            const void *cos, const void *sin, long long rows, int H, long long sq, long long sk, int T,
                            void *stream) {
    static int cached = 0;
    const int threads = block_size(k_forward, cached);
    const unsigned nb = blocks(rows, threads);
    k_forward<<<2 * nb, threads, 0, (cudaStream_t)stream>>>(
        (const bf16 *)q, (const bf16 *)k, (const float *)gq, (const float *)gk, (bf16 *)qo, (bf16 *)ko,
        (const float *)cos, (const float *)sin, rows, nb, H, sq, sk, T);
}

extern "C" void qkp_backward(const void *q, const void *k, const void *gq, const void *gk, const void *dqo,
                             const void *dko, void *dq, void *dk, void *dgq, void *dgk, const void *cos,
                             const void *sin, long long rows, int H, long long sq, long long sk, int T, void *stream) {
    static int cached = 0;
    const int threads = block_size(k_backward, cached);
    const unsigned nb = blocks(rows, threads);
    k_backward<<<2 * nb, threads, 0, (cudaStream_t)stream>>>(
        (const bf16 *)q, (const bf16 *)k, (const float *)gq, (const float *)gk, (const bf16 *)dqo, (const bf16 *)dko,
        (bf16 *)dq, (bf16 *)dk, (float *)dgq, (float *)dgk, (const float *)cos, (const float *)sin, rows, nb, H, sq, sk,
        T);
}
