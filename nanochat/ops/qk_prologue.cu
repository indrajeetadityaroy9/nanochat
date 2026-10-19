/*
 * The attention's q/k prologue of model/attention.py, forward and backward, each one pass over memory: per head row x of 128 bf16,
 * r = x with its first R dims rotated by -theta (half split of those R: r1 = x1 cos + x2 sin, r2 = -x1 sin + x2 cos;
 * dims R..127 stationary: R = 64, modded-nanogpt's half-truncated RoPE, track_1_short
 * model/attention.py:53-55), n = r / sqrt(mean(r^2) + eps) with F.rms_norm's fp32 epsilon, y = scale * n; fp32 math,
 * one bf16 rounding per output.
 *
 * Key offset (modded-nanogpt track_1_short perf/kernels/qkv_rope.py:76-85, record #49): a key's stationary dims come
 * from the previous position's normalized key, y_t[R:] = scale * n_{t-1}[R:] (each row normalized by its own norm, the
 * shift applied after the norm, as modded's kernel does); a position that starts its row (t = 0) or a segment (starts,
 * the document mask's segment starts as one flag per flattened position, if given) keeps its own. Its backward: the
 * stationary part of dn_t is dy_{t+1}[R:] unless t + 1 starts a row or segment, plus dy_t[R:] if t starts one.
 *
 * Backward, recomputing r from x: dr = rstd * (dn - n * mean(dn * n)) with dn = scale * dy (shifted as above),
 * dx = R^T dr on the rotated dims (dx1 = dr1 cos - dr2 sin, dx2 = dr1 sin + dr2 cos), dx = dr on the stationary ones.
 *
 * Layout: q, k (B, T, H, 128) with dense heads (stride(3) = 1, stride(2) = 128, stride(0) = T * stride(1)) at any
 * token stride stride(1) (column slices of a fused qkv are read in place); outputs and gradients contiguous (B, T, H,
 * 128); cos/sin fp32 (T_table >= T, R/2) contiguous, position t of a row = (row / H) % T; starts (B * T) bool or null.
 * A row is LANES = 8 consecutive threads, each owning two 16-byte vectors: the first R/16 lanes a rotation pair
 * (offsets o and o + R/2, o < R/2) with the matching VEC entries of cos and sin, the others two stationary vectors
 * (o and o + (128 - R)/2, o >= R). The row's sums are reduced by shuffles within its lanes. q rows come first, then k
 * rows, in one launch. Instantiated with and without the key offset. Block size: the
 * occupancy calculator's for each kernel (rows are independent, so it changes no result).
 */
#include <cfloat>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

typedef __nv_bfloat16 bf16;

constexpr int HEAD_DIM = 128;                      /* nanochat's head_dim */
constexpr int VEC = sizeof(uint4) / sizeof(bf16);  /* bf16 per 16-byte access */
constexpr int LANES = HEAD_DIM / 2 / VEC;          /* threads per head row, two vectors each */
constexpr int R = HEAD_DIM / 2;                    /* rotated dims: half-truncated RoPE */

static __device__ __forceinline__ void load(const bf16 *p, float *v) {
    const uint4 u = *reinterpret_cast<const uint4 *>(p);
    const __nv_bfloat162 *h = reinterpret_cast<const __nv_bfloat162 *>(&u);
    for (int i = 0; i < VEC / 2; i++) {
        const float2 f = __bfloat1622float2(h[i]);
        v[2 * i] = f.x; v[2 * i + 1] = f.y;
    }
}

/* VEC fp32 table entries (offsets are multiples of VEC, so each float4 is aligned) */
static __device__ __forceinline__ void load(const float *p, float *v) {
    for (int i = 0; i < VEC; i += 4) *reinterpret_cast<float4 *>(v + i) = *reinterpret_cast<const float4 *>(p + i);
}

static __device__ __forceinline__ void store(bf16 *p, const float *v) {
    uint4 u;
    __nv_bfloat162 *h = reinterpret_cast<__nv_bfloat162 *>(&u);
    for (int i = 0; i < VEC / 2; i++) h[i] = __floats2bfloat162_rn(v[2 * i], v[2 * i + 1]);
    *reinterpret_cast<uint4 *>(p) = u;
}

/* This thread's place: its row (q or k, position t), its vectors and table entries. Inputs q, k are read at token
 * stride sq, sk (elements; e.g. column slices of a fused qkv); outputs and gradients are contiguous. */
struct Row {
    bool isq, rot;      /* rot: this lane's vectors rotate */
    int t;
    int64_t token;      /* flattened position b * T + t */
    int64_t x, y, tab;  /* this lane's first vector in q or k, in the contiguous outputs; its cos/sin vector */
    int pair;           /* offset of the second vector from the first */
    int64_t xstep, ystep;  /* from a row to the same head's row of the next position, in the input and the outputs */
};

static __device__ __forceinline__ bool locate(int64_t rows_q, int64_t rows_k, int hq, int hk, int64_t sq, int64_t sk,
                                              int T, Row &r) {
    constexpr int ROT_LANES = R / (2 * VEC);
    const int64_t g = (int64_t)blockIdx.x * blockDim.x + threadIdx.x, row = g / LANES;
    const int lane = (int)(g % LANES);
    if (row >= rows_q + rows_k) return false;
    r.isq = row < rows_q;
    const int64_t i = r.isq ? row : row - rows_q;
    const int h = r.isq ? hq : hk;
    r.token = i / h;
    r.t = (int)(r.token % T);
    r.rot = lane < ROT_LANES;
    const int o = r.rot ? lane * VEC : R + (lane - ROT_LANES) * VEC;
    r.pair = r.rot ? R / 2 : (HEAD_DIM - R) / 2;
    r.xstep = r.isq ? sq : sk;
    r.ystep = (int64_t)h * HEAD_DIM;
    r.x = r.token * r.xstep + (i % h) * HEAD_DIM + o;
    r.y = i * HEAD_DIM + o;
    r.tab = (int64_t)r.t * (R / 2) + o;
    return true;
}

/* Lanes of this thread's row, the shuffle mask: rows are whole groups of LANES threads, so a group exits together. */
static __device__ __forceinline__ unsigned row_mask() {
    return ((1u << LANES) - 1) << ((threadIdx.x & 31) & ~(LANES - 1));
}

static __device__ __forceinline__ float row_sum(float v, unsigned mask) {
    for (int o = LANES / 2; o > 0; o >>= 1) v += __shfl_xor_sync(mask, v, o);
    return v;
}

/* This lane's two vectors of the row at x into v1, v2, rotated if r.rot (by the table vectors at c, s, kept in cs,
 * sn); returns their sum of squares. */
static __device__ __forceinline__ float rotated(const bf16 *x, const Row &r, const float *c, const float *s, float *v1,
                                                float *v2, float *cs, float *sn) {
    load(x, v1); load(x + r.pair, v2);
    float ss = 0.f;
    if (r.rot) {
        load(c, cs); load(s, sn);
        for (int i = 0; i < VEC; i++) {
            const float a = v1[i] * cs[i] + v2[i] * sn[i], b = v2[i] * cs[i] - v1[i] * sn[i];
            v1[i] = a; v2[i] = b;
            ss += a * a + b * b;
        }
    } else {
        for (int i = 0; i < VEC; i++) ss += v1[i] * v1[i] + v2[i] * v2[i];
    }
    return ss;
}

/* Whether flattened position `token` (position t of its row) starts its row or a segment of the document mask. */
static __device__ __forceinline__ bool starts_at(const bool *starts, int64_t token, int t) {
    return t == 0 || (starts != nullptr && starts[token]);
}

template <bool KOFF>
__global__ void k_forward(const bf16 *q, const bf16 *k, bf16 *qo, bf16 *ko, const float *cos, const float *sin,
                          const bool *starts, int64_t rows_q, int64_t rows_k, int hq, int hk, int64_t sq, int64_t sk,
                          int T, float scale) {
    Row r;
    if (!locate(rows_q, rows_k, hq, hk, sq, sk, T, r)) return;
    const unsigned mask = row_mask();
    const bf16 *x = (r.isq ? q : k) + r.x;
    float x1[VEC], x2[VEC], c[VEC], s[VEC];
    float m = rsqrtf(row_sum(rotated(x, r, cos + r.tab, sin + r.tab, x1, x2, c, s), mask) / HEAD_DIM + FLT_EPSILON) * scale;
    if (KOFF && !r.isq && !starts_at(starts, r.token, r.t)) {  /* every lane of the row: the previous position's norm */
        float p1[VEC], p2[VEC];
        const int64_t back = r.tab - R / 2;
        const float mp = rsqrtf(row_sum(rotated(x - r.xstep, r, cos + back, sin + back, p1, p2, c, s), mask) / HEAD_DIM +
                                FLT_EPSILON) * scale;
        if (!r.rot) {
            for (int i = 0; i < VEC; i++) { x1[i] = p1[i]; x2[i] = p2[i]; }
            m = mp;
        }
    }
    for (int i = 0; i < VEC; i++) { x1[i] *= m; x2[i] *= m; }
    bf16 *y = (r.isq ? qo : ko) + r.y;
    store(y, x1); store(y + r.pair, x2);
}

template <bool KOFF>
__global__ void k_backward(const bf16 *q, const bf16 *k, const bf16 *dqo, const bf16 *dko, bf16 *dq, bf16 *dk,
                           const float *cos, const float *sin, const bool *starts, int64_t rows_q, int64_t rows_k, int hq,
                           int hk, int64_t sq, int64_t sk, int T, float scale) {
    Row r;
    if (!locate(rows_q, rows_k, hq, hk, sq, sk, T, r)) return;
    const bf16 *dy = (r.isq ? dqo : dko) + r.y;
    float x1[VEC], x2[VEC], g1[VEC], g2[VEC], c[VEC], s[VEC];
    float ss = rotated((r.isq ? q : k) + r.x, r, cos + r.tab, sin + r.tab, x1, x2, c, s);  /* sum r^2 */
    if (KOFF && !r.isq && !r.rot) {  /* stationary dims of a key: the next position's output holds them unless it starts
                                        a row or segment; a start's own output too */
        for (int i = 0; i < VEC; i++) g1[i] = g2[i] = 0.f;
        if (starts_at(starts, r.token, r.t)) { load(dy, g1); load(dy + r.pair, g2); }
        if (r.t + 1 < T && !starts_at(starts, r.token + 1, r.t + 1)) {
            float n1[VEC], n2[VEC];
            load(dy + r.ystep, n1); load(dy + r.ystep + r.pair, n2);
            for (int i = 0; i < VEC; i++) { g1[i] += n1[i]; g2[i] += n2[i]; }
        }
    } else {
        load(dy, g1); load(dy + r.pair, g2);
    }
    float sd = 0.f;  /* sum dy * r */
    for (int i = 0; i < VEC; i++) sd += g1[i] * x1[i] + g2[i] * x2[i];
    const unsigned mask = row_mask();
    for (int o = LANES / 2; o > 0; o >>= 1) {
        ss += __shfl_xor_sync(mask, ss, o);
        sd += __shfl_xor_sync(mask, sd, o);
    }
    const float rstd = rsqrtf(ss / HEAD_DIM + FLT_EPSILON);
    /* dr = rstd * scale * (dy - r * rstd^2 * mean(dy * r)) */
    const float a = rstd * scale, b = rstd * rstd * sd / HEAD_DIM;
    for (int i = 0; i < VEC; i++) {
        const float d1 = a * (g1[i] - x1[i] * b), d2 = a * (g2[i] - x2[i] * b);
        g1[i] = r.rot ? d1 * c[i] - d2 * s[i] : d1;
        g2[i] = r.rot ? d1 * s[i] + d2 * c[i] : d2;
    }
    bf16 *dx = (r.isq ? dq : dk) + r.y;
    store(dx, g1); store(dx + r.pair, g2);
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

template <bool KOFF>
static void forward(const void *q, const void *k, void *qo, void *ko, const void *cos, const void *sin,
                   const void *starts, long long rows_q, long long rows_k, int hq, int hk, long long sq, long long sk,
                   int T, float scale, cudaStream_t stream) {
    static int cached = 0;
    const int threads = block_size(k_forward<KOFF>, cached);
    k_forward<KOFF><<<blocks(rows_q + rows_k, threads), threads, 0, stream>>>(
        (const bf16 *)q, (const bf16 *)k, (bf16 *)qo, (bf16 *)ko, (const float *)cos, (const float *)sin,
        (const bool *)starts, rows_q, rows_k, hq, hk, sq, sk, T, scale);
}

template <bool KOFF>
static void backward(const void *q, const void *k, const void *dqo, const void *dko, void *dq, void *dk, const void *cos,
                    const void *sin, const void *starts, long long rows_q, long long rows_k, int hq, int hk,
                    long long sq, long long sk, int T, float scale, cudaStream_t stream) {
    static int cached = 0;
    const int threads = block_size(k_backward<KOFF>, cached);
    k_backward<KOFF><<<blocks(rows_q + rows_k, threads), threads, 0, stream>>>(
        (const bf16 *)q, (const bf16 *)k, (const bf16 *)dqo, (const bf16 *)dko, (bf16 *)dq, (bf16 *)dk,
        (const float *)cos, (const float *)sin, (const bool *)starts, rows_q, rows_k, hq, hk, sq, sk, T, scale);
}

extern "C" void qkp_forward(const void *q, const void *k, void *qo, void *ko, const void *cos, const void *sin,
                            const void *starts, long long rows_q, long long rows_k, int hq, int hk, long long sq,
                            long long sk, int T, bool key_offset, float scale, void *stream) {
    (key_offset ? forward<true> : forward<false>)(q, k, qo, ko, cos, sin, starts, rows_q, rows_k, hq, hk, sq, sk, T,
                                                  scale, (cudaStream_t)stream);
}

extern "C" void qkp_backward(const void *q, const void *k, const void *dqo, const void *dko, void *dq, void *dk,
                             const void *cos, const void *sin, const void *starts, long long rows_q, long long rows_k,
                             int hq, int hk, long long sq, long long sk, int T, bool key_offset, float scale,
                             void *stream) {
    (key_offset ? backward<true> : backward<false>)(q, k, dqo, dko, dq, dk, cos, sin, starts, rows_q, rows_k, hq, hk, sq,
                                                    sk, T, scale, (cudaStream_t)stream);
}
