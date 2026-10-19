/*
 * Block Attention Residuals' depth attention of model/gpt.py GPT.hidden (Chen et al., arXiv 2603.15031 §3.1-3.2,
 * Eqs. 2-6) in the paper's two-phase computation (§4.2, Algorithm 1), forward and backward, each one pass over memory.
 * A sublayer's input is norm(h), h = sum_i softmax_i(l_i) x_i over its sources x_i (the embedding, the completed blocks
 * and its block's partial sum), l_i = r_i (q . x_i), r_i = 1 / sqrt(mean(x_i^2) + eps) (q . RMSNorm(x_i)), q its
 * pseudo-query; F.rms_norm's fp32 epsilon throughout; fp32 math, one bf16 rounding of h, then the norm of the rounded h
 * (F.rms_norm on bf16), one rounding of its output.
 *
 * Phase 1 (dai_*), once per block: the block's S queries against its K completed blocks (the queries are parameters, so
 * every logit of the block is known before its first sublayer), each block read once (§4.2: "reducing reads from S
 * times to just once per block"). Its outputs: the first sublayer's input y0 = norm(h0) (no partial sum yet), and for
 * each later sublayer s the inter-block average o_s = sum_i w_si x_i (fp32, softmax over the completed blocks) and its
 * log-sum-exp lse_s; the per-row statistics (logits, the blocks' r_i, y0's rstd) for the backward.
 * Phase 2 (dam_*), per later sublayer: the partial sum p's logit l_p, the online-softmax merge h = a o + b p
 * (a = e^lse / (e^lse + e^l_p), b = 1 - a; Algorithm 1 line 12) and the norm; statistics a, b, r_p, l_p, rstd.
 *
 * Backward (rounding as identity): g = dL/dh = rstd (dy - y mean(dy y)) (the norm's backward; y, the bf16 output, stands
 * for the normed h, which is not kept). For logit l = r (q . x): dl/dx = r q - l r^2 x / C, dl/dq = r x.
 * Phase 2: dL/do = a g; dl_p = b (g.p - g.h) and dL/dlse = -dl_p (both enter through a alone); dL/dp = b g + dl_p dl/dp;
 * dq += dl_p r_p p. g.h = eps rstd^2 (dy . y) exactly for the h the norm saw (its rounding as identity); o is not kept.
 * Phase 1: D_si = dL/do_s . x_i (dL/dh0 = g for s = 0), dl_si = w_si (D_si - sum_k w_sk D_sk + dL/dlse_s) (none for
 * s = 0); dL/dx_i = sum_s [w_si dL/do_s + dl_si dl_si/dx_i]; dq_s = sum over rows and i of dl_si r_i x_i.
 *
 * Layout: every (rows, C) tensor contiguous, bf16 the sources, partial sums, inputs y and their gradients, fp32 o and
 * its gradient; queries fp32 (S, C). C is whole chunks of CHUNK = head_dim 128 dims (gpt.py gpt_config). k_inter_fwd,
 * k_merge_*: a row is one warp (one block), lane l owning dims l E .. l E + E - 1 of each chunk (coalesced), the row in
 * registers, its sums by xor shuffles (no barriers). k_inter_bwd: a row is one block (below). The forwards run one
 * block per row; the backwards are persistent (as many blocks as the device holds at once, each looping over a fixed
 * stride of rows), each block's query gradient a partial in registers, k_sum adding the partials in a fixed order: no
 * atomics, bitwise reproducible.
 */
#include <type_traits>

#include "qk_row.cuh"

constexpr int E = 4;                         /* dims per lane and chunk: one 8-byte bf16 access */
constexpr int CHUNK = WARP * E;              /* dims per chunk: 128 = gpt.py gpt_config head_dim, C a multiple */
constexpr int MAX_CHUNKS = 1024 / CHUNK;     /* C up to 1024, gpt_config's width at MobileLLM's 32 layers */
constexpr int MAX_SOURCES = 8 + 1;           /* gpt.py ATTN_RES_BLOCKS completed blocks and the embedding */
constexpr int MAX_QUERIES = 2 * 32 / 8;      /* sublayers per block 2 n_layer / ATTN_RES_BLOCKS up to 32 layers */
constexpr int MERGE_STATS = 5;               /* depth_merge's statistics per row: a, b, r_p, l_p, rstd */

/* Up to M device pointers, by value in the kernel's parameters. */
template <typename T, int M> struct Ptrs {
    typedef T *elem;
    elem p[M];
};
typedef Ptrs<const bf16, MAX_SOURCES> Sources;

/* 4 bf16 as fp32. The empty asm marks u as rewritten, so the compiler converts at each use instead of hoisting fp32
 * copies of a register-resident row out of the loops (twice its registers). */
static __device__ __forceinline__ void unpack(uint2 u, float *v) {
    asm volatile("" : "+r"(u.x), "+r"(u.y));
    const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&u.x));
    const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&u.y));
    v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
}

static __device__ __forceinline__ uint2 raw(const bf16 *p) { return *reinterpret_cast<const uint2 *>(p); }

static __device__ __forceinline__ void ld(const bf16 *p, float *v) { unpack(raw(p), v); }

static __device__ __forceinline__ void ld(const float *p, float *v) {
    *reinterpret_cast<float4 *>(v) = *reinterpret_cast<const float4 *>(p);
}

static __device__ __forceinline__ void st(bf16 *p, const float *v) {
    uint2 u;
    *reinterpret_cast<__nv_bfloat162 *>(&u.x) = __floats2bfloat162_rn(v[0], v[1]);
    *reinterpret_cast<__nv_bfloat162 *>(&u.y) = __floats2bfloat162_rn(v[2], v[3]);
    *reinterpret_cast<uint2 *>(p) = u;
}

static __device__ __forceinline__ void st(float *p, const float *v) {
    *reinterpret_cast<float4 *>(p) = *reinterpret_cast<const float4 *>(v);
}

static __device__ __forceinline__ float dot(const float *a, const float *b) {
    float s = 0.f;
    for (int e = 0; e < E; e++) s += a[e] * b[e];
    return s;
}

/* The row's sum of v, in every lane. */
static __device__ __forceinline__ float row_sum(float v) {
    for (int o = WARP / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

static __device__ __forceinline__ float bf16_round(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

/* The softmax over K logits l: weights into w, the log-sum-exp returned. */
static __device__ __forceinline__ float softmax(const float *l, int K, float *w) {
    float m = -INFINITY, z = 0.f;
#pragma unroll
    for (int i = 0; i < MAX_SOURCES; i++)
        if (i < K) m = fmaxf(m, l[i]);
#pragma unroll
    for (int i = 0; i < MAX_SOURCES; i++)
        if (i < K) z += (w[i] = expf(l[i] - m));
#pragma unroll
    for (int i = 0; i < MAX_SOURCES; i++)
        if (i < K) w[i] /= z;
    return m + logf(z);
}

/* Phase 1 forward, block = row. stats row: logits (S, K), r (K), y0's rstd. */
template <int CH>
__global__ void __launch_bounds__(WARP) k_inter_fwd(Sources x, int K, const float *q, int S, bf16 *y0,
                                                    Ptrs<float, MAX_QUERIES> o, float *lse, float *stats,
                                                    int64_t rows) {
    constexpr int C = CH * CHUNK;
    const int64_t n = blockIdx.x, at = n * C + threadIdx.x * E;
    float *sn = stats + n * (S * K + K + 1);
    uint2 v[MAX_SOURCES][CH];
    float r[MAX_SOURCES];
#pragma unroll
    for (int i = 0; i < MAX_SOURCES; i++)
        if (i < K)
#pragma unroll
            for (int j = 0; j < CH; j++) v[i][j] = raw(x.p[i] + at + j * CHUNK);
#pragma unroll
    for (int i = 0; i < MAX_SOURCES; i++)
        if (i < K) {
            float ss = 0.f, f[E];
#pragma unroll
            for (int j = 0; j < CH; j++) {
                unpack(v[i][j], f);
                ss += dot(f, f);
            }
            r[i] = rsqrtf(row_sum(ss) / C + FLT_EPSILON);
            if (threadIdx.x == 0) sn[S * K + i] = r[i];
        }
    for (int s = 0; s < S; s++) {
        float d[MAX_SOURCES] = {}, l[MAX_SOURCES], w[MAX_SOURCES], qs[E], f[E];
#pragma unroll
        for (int j = 0; j < CH; j++) {
            ld(q + s * C + threadIdx.x * E + j * CHUNK, qs);
#pragma unroll
            for (int i = 0; i < MAX_SOURCES; i++)
                if (i < K) {
                    unpack(v[i][j], f);
                    d[i] += dot(qs, f);
                }
        }
#pragma unroll
        for (int i = 0; i < MAX_SOURCES; i++)
            if (i < K) {
                l[i] = r[i] * row_sum(d[i]);
                if (threadIdx.x == 0) sn[s * K + i] = l[i];
            }
        const float lse_s = softmax(l, K, w);
        if (s && threadIdx.x == 0) lse[(s - 1) * rows + n] = lse_s;
        /* h by chunk; for s = 0 twice (its norm needs the row's sum of squares first) */
        float ss = 0.f, rstd = 0.f;
        for (int pass = 0; pass < (s ? 1 : 2); pass++) {
#pragma unroll
            for (int j = 0; j < CH; j++) {
                float h[E] = {};
#pragma unroll
                for (int i = 0; i < MAX_SOURCES; i++)
                    if (i < K) {
                        unpack(v[i][j], f);
                        for (int e = 0; e < E; e++) h[e] += w[i] * f[e];
                    }
                if (s) {
                    st(o.p[s - 1] + at + j * CHUNK, h);
                    continue;
                }
                for (int e = 0; e < E; e++) h[e] = bf16_round(h[e]);
                if (pass) {
                    for (int e = 0; e < E; e++) h[e] *= rstd;
                    st(y0 + at + j * CHUNK, h);
                } else {
                    ss += dot(h, h);
                }
            }
            if (!s && !pass) {
                rstd = rsqrtf(row_sum(ss) / C + FLT_EPSILON);
                if (threadIdx.x == 0) sn[S * K + K] = rstd;
            }
        }
    }
}

/* Phase 2 forward, block = row. */
template <int CH>
__global__ void __launch_bounds__(WARP) k_merge_fwd(const float *o, const float *lse, const bf16 *p, const float *q,
                                                    bf16 *y, float *stats) {
    constexpr int C = CH * CHUNK;
    const int64_t n = blockIdx.x, at = n * C + threadIdx.x * E;
    float ov[CH][E], pv[CH][E], qv[CH][E], ss = 0.f, d = 0.f;
#pragma unroll
    for (int j = 0; j < CH; j++) {
        ld(o + at + j * CHUNK, ov[j]); ld(p + at + j * CHUNK, pv[j]); ld(q + threadIdx.x * E + j * CHUNK, qv[j]);
        ss += dot(pv[j], pv[j]);
        d += dot(qv[j], pv[j]);
    }
    const float rp = rsqrtf(row_sum(ss) / C + FLT_EPSILON), lp = rp * row_sum(d), L = lse[n];
    const float m = fmaxf(L, lp), ea = expf(L - m), eb = expf(lp - m), a = ea / (ea + eb), b = eb / (ea + eb);
    float hh = 0.f;
#pragma unroll
    for (int j = 0; j < CH; j++) {
        for (int e = 0; e < E; e++) ov[j][e] = bf16_round(a * ov[j][e] + b * pv[j][e]);
        hh += dot(ov[j], ov[j]);
    }
    const float rstd = rsqrtf(row_sum(hh) / C + FLT_EPSILON);
#pragma unroll
    for (int j = 0; j < CH; j++) {
        for (int e = 0; e < E; e++) ov[j][e] *= rstd;
        st(y + at + j * CHUNK, ov[j]);
    }
    if (threadIdx.x == 0) {
        float *sn = stats + n * MERGE_STATS;
        sn[0] = a; sn[1] = b; sn[2] = rp; sn[3] = lp; sn[4] = rstd;
    }
}

/* Phase 2 backward, persistent; part: this block's dq partial (C). */
template <int CH>
__global__ void __launch_bounds__(WARP) k_merge_bwd(const bf16 *dy, const bf16 *y, const float *stats, const bf16 *p,
                                                    const float *q, float *d_o, float *dlse, bf16 *dp, float *part,
                                                    int64_t rows) {
    constexpr int C = CH * CHUNK;
    const int c = threadIdx.x * E;
    float qv[CH][E], acc[CH][E] = {};
#pragma unroll
    for (int j = 0; j < CH; j++) ld(q + c + j * CHUNK, qv[j]);
    for (int64_t n = blockIdx.x; n < rows; n += gridDim.x) {
        const int64_t at = n * C + c;
        const float *sn = stats + n * MERGE_STATS;
        const float a = sn[0], b = sn[1], rp = sn[2], lp = sn[3], rstd = sn[4];
        float dv[CH][E], yv[CH][E], pv[CH][E], dyy = 0.f, dyp = 0.f, yp = 0.f;
#pragma unroll
        for (int j = 0; j < CH; j++) {
            ld(dy + at + j * CHUNK, dv[j]); ld(y + at + j * CHUNK, yv[j]); ld(p + at + j * CHUNK, pv[j]);
            dyy += dot(dv[j], yv[j]);
            dyp += dot(dv[j], pv[j]);
            yp += dot(yv[j], pv[j]);
        }
        dyy = row_sum(dyy); dyp = row_sum(dyp); yp = row_sum(yp);
        const float gp = rstd * (dyp - yp * dyy / C), gh = FLT_EPSILON * rstd * rstd * dyy, dlp = b * (gp - gh);
#pragma unroll
        for (int j = 0; j < CH; j++) {
            float go[E], gpv[E];
            for (int e = 0; e < E; e++) {
                const float g = rstd * (dv[j][e] - yv[j][e] * dyy / C);
                go[e] = a * g;
                gpv[e] = b * g + dlp * (rp * qv[j][e] - lp * rp * rp * pv[j][e] / C);
                acc[j][e] += dlp * rp * pv[j][e];
            }
            st(d_o + at + j * CHUNK, go);
            st(dp + at + j * CHUNK, gpv);
        }
        if (threadIdx.x == 0) dlse[n] = -dlp;
    }
#pragma unroll
    for (int j = 0; j < CH; j++) st(part + (int64_t)blockIdx.x * C + c + j * CHUNK, acc[j]);
}

/* The warp's sums of 32 per-lane values: lane l returns the sum of v[l] over the lanes (a transposed butterfly, 31
 * shuffles where 32 separate reductions take 160). Step O: lanes with bit O keep the upper O live values, the others
 * the lower, each adding its partner's; O a template parameter so every index is static (v stays in registers). */
template <int O = WARP / 2> static __device__ __forceinline__ float warp_sums(float (&v)[WARP]) {
    const bool up = threadIdx.x & O;
#pragma unroll
    for (int k = 0; k < O; k++) {
        const float send = up ? v[k] : v[k + O], keep = up ? v[k + O] : v[k];
        v[k] = keep + __shfl_xor_sync(0xffffffffu, send, O);
    }
    if constexpr (O > 1) return warp_sums<O / 2>(v);
    return v[0];
}

constexpr int PER_FRAME = WARP / MAX_SOURCES;  /* queries per frame of 32 row sums in k_inter_bwd */
constexpr int MAX_FRAMES = 1 + (MAX_QUERIES - 1 + PER_FRAME - 1) / PER_FRAME;
constexpr int INTER_BWD_BLOCKS = 2;  /* k_inter_bwd's blocks per SM within the register budget without spills */

/* Phase 1 backward, persistent, block = row (C / E threads, thread t owning dims t E .. t E + E - 1): its dq partial
 * in registers. The row sums in frames of 32 (warp_sums, then the warps in order through shared memory): frame 0
 * dy0 . y0, dy0 . x_i, y0 . x_i; frame f > 0 dL/do_s . x_i of PER_FRAME queries. Three barriers per row: sums, totals,
 * the logit gradients (thread s K + i: query s, source i). Shared memory (dynamic, floats): red (frames, warps, 32),
 * tot (frames, 32), wu (S, K) pairs (w, dl r), dll (S, K) = dl l, and two buffers (alternate rows: one is prefetched
 * while the other is read) of the row's stats and dL/dlse (S). */
template <int CH>
__global__ void __launch_bounds__(CH * WARP, INTER_BWD_BLOCKS)
k_inter_bwd(Sources x, int K, const float *q, int S, const bf16 *y0, const bf16 *dy0, Ptrs<const float, MAX_QUERIES> d_o,
            const float *dlse, const float *stats, Ptrs<bf16, MAX_SOURCES> dx, float *part, int64_t rows) {
    constexpr int C = CH * CHUNK;
    extern __shared__ float smem[];
    const int W = S * K + K + 1, NF = 1 + (S - 1 + PER_FRAME - 1) / PER_FRAME, warp = threadIdx.x / WARP,
              lane = threadIdx.x & (WARP - 1), c = threadIdx.x * E;
    float *red = smem, *tot = red + NF * CH * WARP;
    float2 *wu = reinterpret_cast<float2 *>(tot + NF * WARP);
    float *dll = reinterpret_cast<float *>(wu + S * K), *bufs = dll + S * K;
    float dq[MAX_QUERIES][E] = {};
    for (int64_t n = blockIdx.x, it = 0; n < rows; n += gridDim.x, it++) {
        const int64_t at = n * C + c;
        float *srow = bufs + (it & 1) * (W + S), *sdl = srow + W;
        for (int t = threadIdx.x; t < W; t += blockDim.x) srow[t] = stats[n * W + t];
        if (threadIdx.x > 0 && threadIdx.x < S) sdl[threadIdx.x] = dlse[(threadIdx.x - 1) * rows + n];
        uint2 v[MAX_SOURCES];
        float g[MAX_QUERIES - 1][E], dv[E], yv[E], f[E];
        ld(dy0 + at, dv); ld(y0 + at, yv);
#pragma unroll
        for (int i = 0; i < MAX_SOURCES; i++)
            if (i < K) v[i] = raw(x.p[i] + at);
#pragma unroll
        for (int s = 1; s < MAX_QUERIES; s++)
            if (s < S) ld(d_o.p[s - 1] + at, g[s - 1]);
        {
            float p[WARP] = {};
            p[0] = dot(dv, yv);
#pragma unroll
            for (int i = 0; i < MAX_SOURCES; i++)
                if (i < K) {
                    unpack(v[i], f);
                    p[1 + i] = dot(dv, f);
                    p[1 + MAX_SOURCES + i] = dot(yv, f);
                }
            red[warp * WARP + lane] = warp_sums(p);
        }
#pragma unroll
        for (int fr = 1; fr < MAX_FRAMES; fr++)
            if (fr < NF) {
                float p[WARP] = {};
#pragma unroll
                for (int k = 0; k < PER_FRAME; k++) {
                    const int s = 1 + (fr - 1) * PER_FRAME + k;
                    if (s < S)
#pragma unroll
                        for (int i = 0; i < MAX_SOURCES; i++)
                            if (i < K) {
                                unpack(v[i], f);
                                p[k * MAX_SOURCES + i] = dot(g[s - 1], f);
                            }
                }
                red[(fr * CH + warp) * WARP + lane] = warp_sums(p);
            }
        __syncthreads();
        for (int t = threadIdx.x; t < NF * WARP; t += blockDim.x) {
            float s = 0.f;
            for (int w = 0; w < CH; w++) s += red[((t / WARP) * CH + w) * WARP + t % WARP];
            tot[t] = s;
        }
        __syncthreads();
        const float rstd = srow[W - 1], dyy = tot[0];
        if (threadIdx.x < S * K) {
            const int s = threadIdx.x / K, i = threadIdx.x % K;
            /* D_sk = alpha (A_k - beta B_k): dL/do_s . x_k, for s = 0 with dL/dh0 = rstd (dy0 - y0 dyy / C) */
            const float *A = s ? tot + (1 + (s - 1) / PER_FRAME) * WARP + (s - 1) % PER_FRAME * MAX_SOURCES : tot + 1;
            const float *B = tot + 1 + MAX_SOURCES, alpha = s ? 1.f : rstd, beta = s ? 0.f : dyy / C;
            const float *l = srow + s * K;
            float m = -INFINITY, z = 0.f, mean = 0.f;
            for (int k = 0; k < K; k++) m = fmaxf(m, l[k]);
            for (int k = 0; k < K; k++) {
                const float e = expf(l[k] - m);
                z += e;
                mean += e * alpha * (A[k] - beta * B[k]);
            }
            const float w = expf(l[i] - m) / z,
                        dl = w * (alpha * (A[i] - beta * B[i]) - mean / z + (s ? sdl[s] : 0.f));
            wu[threadIdx.x] = make_float2(w, dl * srow[S * K + i]);
            dll[threadIdx.x] = dl * l[i];
        }
        __syncthreads();
        float acc[MAX_SOURCES][E], g0[E];
        for (int e = 0; e < E; e++) g0[e] = rstd * (dv[e] - yv[e] * dyy / C);
#pragma unroll
        for (int i = 0; i < MAX_SOURCES; i++)
            if (i < K) {
                float t = 0.f;
                for (int s = 0; s < S; s++) t += dll[s * K + i];
                const float r = srow[S * K + i], xi = r * r * t / C;
                unpack(v[i], f);
                for (int e = 0; e < E; e++) acc[i][e] = -xi * f[e];
            }
#pragma unroll
        for (int s = 0; s < MAX_QUERIES; s++)
            if (s < S) {
                float qs[E], gs[E];
                ld(q + s * C + c, qs);
                for (int e = 0; e < E; e++) gs[e] = s ? g[s ? s - 1 : 0][e] : g0[e];
#pragma unroll
                for (int i = 0; i < MAX_SOURCES; i++)
                    if (i < K) {
                        const float2 p = wu[s * K + i];
                        unpack(v[i], f);
                        for (int e = 0; e < E; e++) {
                            acc[i][e] += p.x * gs[e] + p.y * qs[e];
                            dq[s][e] += p.y * f[e];
                        }
                    }
            }
#pragma unroll
        for (int i = 0; i < MAX_SOURCES; i++)
            if (i < K) st(dx.p[i] + at, acc[i]);
    }
#pragma unroll
    for (int s = 0; s < MAX_QUERIES; s++)
        if (s < S) st(part + ((int64_t)blockIdx.x * S + s) * C + c, dq[s]);
}

constexpr int SUM_WARPS = 8;  /* k_sum's block: 256 threads */

/* out[j] = the sum of the nb partials part[b][j]: warp w of a block adds b = w, w + SUM_WARPS, ... in order for 32
 * consecutive j (coalesced), then warp 0 adds the warps' sums in order: a fixed order. */
__global__ void __launch_bounds__(SUM_WARPS * WARP) k_sum(const float *part, unsigned nb, int64_t len, float *out) {
    __shared__ float sums[SUM_WARPS][WARP];
    const int lane = threadIdx.x & (WARP - 1), warp = threadIdx.x / WARP;
    const int64_t j = (int64_t)blockIdx.x * WARP + lane;
    float s = 0.f;
    if (j < len)
        for (unsigned b = warp; b < nb; b += SUM_WARPS) s += part[b * len + j];
    sums[warp][lane] = s;
    __syncthreads();
    if (warp == 0 && j < len) {
        float t = 0.f;
        for (int w = 0; w < SUM_WARPS; w++) t += sums[w][lane];
        out[j] = t;
    }
}

/* f(std::integral_constant<int, C / CHUNK>): the kernels' instance for width C. */
template <typename F> static void chunks(int C, F f) {
    switch (C / CHUNK) {
    case 1: f(std::integral_constant<int, 1>()); break;
    case 2: f(std::integral_constant<int, 2>()); break;
    case 3: f(std::integral_constant<int, 3>()); break;
    case 4: f(std::integral_constant<int, 4>()); break;
    case 5: f(std::integral_constant<int, 5>()); break;
    case 6: f(std::integral_constant<int, 6>()); break;
    case 7: f(std::integral_constant<int, 7>()); break;
    case MAX_CHUNKS: f(std::integral_constant<int, MAX_CHUNKS>()); break;
    }
}

/* Blocks the device holds at once of kernel k at `threads` threads and `smem` dynamic shared bytes, at most `rows`:
 * fixed for a device and shape, and so is every partial's summation order. */
template <typename F> static unsigned resident(F k, int threads, size_t smem, int64_t rows) {
    int device, sms, per_sm;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k, threads, smem);
    const int64_t all = (int64_t)sms * per_sm;
    return (unsigned)(all < rows ? all : rows);
}

/* k_inter_bwd's shared memory (its layout there). */
static size_t inter_bwd_smem(int K, int S, int C) {
    const int frames = 1 + (S - 1 + PER_FRAME - 1) / PER_FRAME, W = S * K + K + 1;
    return sizeof(float) * ((size_t)frames * (C / CHUNK + 1) * WARP + 3 * S * K + 2 * (W + S));
}

static void sum_parts(const float *part, unsigned nb, int64_t len, float *out, cudaStream_t stream) {
    k_sum<<<(unsigned)((len + WARP - 1) / WARP), SUM_WARPS * WARP, 0, stream>>>(part, nb, len, out);
}

template <typename P, typename T> static P pointers(T *const *src, int count) {
    P ptrs = {};
    for (int i = 0; i < count; i++) ptrs.p[i] = (typename P::elem)src[i];
    return ptrs;
}

extern "C" void dai_forward(const void *const *x, int K, const void *q, int S, void *y0, void *const *o, void *lse,
                            void *stats, long long rows, int C, void *stream) {
    chunks(C, [&](auto ch) {
        k_inter_fwd<decltype(ch)::value><<<(unsigned)rows, WARP, 0, (cudaStream_t)stream>>>(
            pointers<Sources>(x, K), K, (const float *)q, S, (bf16 *)y0, pointers<Ptrs<float, MAX_QUERIES>>(o, S - 1),
            (float *)lse, (float *)stats, rows);
    });
}

extern "C" unsigned dai_backward_blocks(long long rows, int K, int S, int C) {
    unsigned nb = 0;
    chunks(C, [&](auto ch) {
        nb = resident(k_inter_bwd<decltype(ch)::value>, C / E, inter_bwd_smem(K, S, C), rows);
    });
    return nb;
}

/* part: fp32 (dai_backward_blocks(rows, K, S, C), S, C). */
extern "C" void dai_backward(const void *const *x, int K, const void *q, int S, const void *y0, const void *dy0,
                             const void *const *d_o, const void *dlse, const void *stats, void *const *dx, void *part,
                             void *dq, long long rows, int C, void *stream) {
    const unsigned nb = dai_backward_blocks(rows, K, S, C);
    chunks(C, [&](auto ch) {
        k_inter_bwd<decltype(ch)::value><<<nb, C / E, inter_bwd_smem(K, S, C), (cudaStream_t)stream>>>(
            pointers<Sources>(x, K), K, (const float *)q, S, (const bf16 *)y0, (const bf16 *)dy0,
            pointers<Ptrs<const float, MAX_QUERIES>>(d_o, S - 1), (const float *)dlse, (const float *)stats,
            pointers<Ptrs<bf16, MAX_SOURCES>>(dx, K), (float *)part, rows);
    });
    sum_parts((const float *)part, nb, (int64_t)S * C, (float *)dq, (cudaStream_t)stream);
}

extern "C" void dam_forward(const void *o, const void *lse, const void *p, const void *q, void *y, void *stats,
                            long long rows, int C, void *stream) {
    chunks(C, [&](auto ch) {
        k_merge_fwd<decltype(ch)::value><<<(unsigned)rows, WARP, 0, (cudaStream_t)stream>>>(
            (const float *)o, (const float *)lse, (const bf16 *)p, (const float *)q, (bf16 *)y, (float *)stats);
    });
}

extern "C" unsigned dam_backward_blocks(long long rows, int C) {
    unsigned nb = 0;
    chunks(C, [&](auto ch) { nb = resident(k_merge_bwd<decltype(ch)::value>, WARP, 0, rows); });
    return nb;
}

/* part: fp32 (dam_backward_blocks(rows, C), C). */
extern "C" void dam_backward(const void *dy, const void *y, const void *stats, const void *p, const void *q, void *d_o,
                             void *dlse, void *dp, void *part, void *dq, long long rows, int C, void *stream) {
    const unsigned nb = dam_backward_blocks(rows, C);
    chunks(C, [&](auto ch) {
        k_merge_bwd<decltype(ch)::value><<<nb, WARP, 0, (cudaStream_t)stream>>>(
            (const bf16 *)dy, (const bf16 *)y, (const float *)stats, (const bf16 *)p, (const float *)q, (float *)d_o,
            (float *)dlse, (bf16 *)dp, (float *)part, rows);
    });
    sum_parts((const float *)part, nb, C, (float *)dq, (cudaStream_t)stream);
}
