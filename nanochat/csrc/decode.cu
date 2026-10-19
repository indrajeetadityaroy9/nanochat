/*
 * Greedy decoding of a nanochat GPT (nanochat/model/gpt.py) for B independent sequences (slots); one step of all slots
 * is captured as a CUDA graph (nanochat/inference/decoder.py).
 *
 * A small-batch step is bound by weight reads and kernel launches: each weight is read once for all slots, and the
 * step's 5 kernels per layer plus 3 replay as one graph. Positions, input tokens, KV caches and the active-slot mask
 * live in device memory. A slot starts from the model's batch-1 prefill of its prompt but the last token
 * (dec_set_prefilled); inactive slots are skipped by every kernel.
 *
 * Each step (gpt.py's forward at one position; C width, H heads of HEAD_DIM, x0 the normed embedding):
 *   k_embed   x0 = norm(wte[token])
 *   per layer i:
 *     k_gemv  xm = resid_lambdas[i] x + x0_lambdas[i] x0 (x = x0 before layer 0), h = norm(xm),
 *             [q k v] = h [Wq Wk Wv]^T
 *     k_attn  VE layers: v += 2 sigmoid(W_g[head] . h) value_embeds[i][token] (per head, fp32);
 *             q, k = rope(norm_D(q) q_gain), rope(norm_D(k) k_gain) (qk_row.cuh, as qk_prologue.cu);
 *             y = softmax(q K^T / sqrt(HEAD_DIM)) V over the layer's window (FlashAttention's window_size left)
 *     k_gemv  xo = xm + y Wo^T
 *     k_gemv  u = relu(norm(xo) Wfc^T)^2
 *     k_gemv  x = xo + u Wproj^T
 *   k_gemv    argmax of norm(x) W_lm^T (atomicMax on a packed key)
 *   k_commit  the argmax becomes each active slot's next token
 *
 * Numerics follow the eager forward op by op: bf16 copies of the fp32 matrices and tables; fp32 rotary table, q/k
 * gains and value-embedding gate (no contraction), the gate's mix rounded once to bf16; resid/x0 lambdas as bf16 values
 * on bf16 tensors (PyTorch's 0-dim fp32 x bf16 promotion), each product and sum rounded; ReLU^2 from the fp32
 * accumulators as QuACK's gemm_act relu_sq (quack/activation.py), one bf16 rounding; F.rms_norm's fp32 epsilon; argmax
 * over fp32 logits.
 *
 * No sum mixes slots and every sum's order is fixed by the shapes and the slot's position (GEMV: MMAs over K in fixed
 * order; attention: split count from the position), so a sequence's tokens do not depend on the batch or on timing.
 *
 * Launch shapes: a GEMV runs a warp per MMA_M-row tile in blocks sized by cudaOccupancyMaxPotentialBlockSize, fewer
 * warps when that would leave SMs idle, at most one resident wave; attention runs a thread per head dim in (head,
 * split, slot) blocks, at most a wave of splits per head and slot (k_attn).
 */
#include <initializer_list>
#include <limits.h>
#include <math.h>
#include <stdlib.h>

#include "qk_row.cuh"

constexpr int LANE_DIMS = HEAD_DIM / WARP;         /* a lane's dims of a query, key or value */
constexpr int ATTN_WARPS = HEAD_DIM / WARP;        /* an attention block: a thread per head dim */
constexpr int MAX_BATCH = CHAR_BIT * sizeof(unsigned long long);  /* slots: the bits of the active mask */
/* mma.sync m16n8k16 (PTX ISA "Matrix Fragments for mma.m16n8k16"): lanes form MMA_N groups (groupID = lane / GROUP) of
 * GROUP threads (threadID_in_group = lane % GROUP), each holding ACC fp32 accumulators */
constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 16;
constexpr int GROUP = WARP / MMA_N, ACC = MMA_M * MMA_N / WARP;
constexpr int K_BLOCK = GROUP * VEC;               /* k per group step: VEC per lane */
static_assert(K_BLOCK == 2 * MMA_K, "a K_BLOCK block is two k16 MMAs (k_gemv)");
constexpr int NTILES = MAX_BATCH / MMA_N;          /* the MMA's n tiles of the slots */

enum { PRO_COPY, PRO_RESID_NORM, PRO_NORM };
enum { EPI_STORE, EPI_RESID, EPI_RELU2, EPI_ARGMAX };

static __device__ __forceinline__ float bfr(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

static __device__ __forceinline__ float sigmoidf(float x) { return __fdiv_rn(1.f, __fadd_rn(1.f, expf(-x))); }

static __device__ __forceinline__ bool live(unsigned long long mask, int b) { return (mask >> b) & 1ull; }

static __device__ __forceinline__ float warp_sum(float v) {
    for (int o = WARP / 2; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

/* Maximum over the warp, to every lane. */
static __device__ __forceinline__ float warp_max(float v) {
    for (int o = WARP / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* GEMV of B slots on the tensor cores: out[b][r] = W[r] . x[b] by mma.sync m16n8k16 (bf16 operands, fp32 accumulators),
 * a warp per MMA_M-row tile of W (tiles strided over the grid), each weight read once. A slot is one column of an n
 * tile, and each output's sum runs over K in a fixed order (K_BLOCK blocks, two k16 MMAs each), so a slot's outputs do
 * not depend on B. In a K_BLOCK block lane (g, t) = (groupID, threadID_in_group) loads k t*VEC.. of rows g, g + 8 and
 * of its column's input (one 16-byte load each); A and B map MMA k to physical k alike, so the product is W . x.
 * The prologue writes the slots' bf16 inputs a K chunk at a time to shared memory, [n tiles of slots][chunk + VEC] (the
 * VEC pad staggers banks; inactive and padding slots zero; chunk: gemv_chunk). (RESID_)NORM first computes each slot's
 * norm factor, a warp per slot. RELU2: out = bf16(relu(s) s) of the fp32 dot product s (QuACK gemm_act relu_sq). */

typedef struct {
    const bf16 *w;               /* [rows][cols] */
    int rows, cols, chunk;       /* chunk: the k of the slots' inputs held in shared memory at once (gemv_chunk) */
    int pro;                     /* prologue: how a slot's input vector is made */
    const bf16 *in, *in2;        /* [B][cols]: COPY/NORM: in; RESID_NORM: in = x, in2 = x0 */
    float l1, l2;                /* RESID_NORM: the lambdas of x and x0 (bf16 values) */
    bf16 *pro_out, *pro_out2;    /* RESID_NORM, [B][cols], block 0 writes them: the mixed residual, its norm */
    int epi;                     /* epilogue: what is done with each output */
    bf16 *out;                   /* [B][rows]: STORE/RESID/RELU2 output */
    const bf16 *res;             /* [B][rows]: RESID: residual added */
    unsigned long long *argmax;  /* [B] ARGMAX: packed (ordered logit, ~row) maximum */
    const unsigned long long *active;  /* the active slots' bit mask */
    int B;
} Gemv;

static __device__ __forceinline__ void mma16816(float *d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                                uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

/* The prologue's value of slot b at k (a bf16 value): RESID_NORM's mix; norms times the slot's factor r. */
static __device__ __forceinline__ float pro_value(const Gemv &a, size_t at, float r, float *mixed) {
    float v = __bfloat162float(a.in[at]);
    if (a.pro == PRO_RESID_NORM) v = bfr(bfr(a.l1 * v) + bfr(a.l2 * __bfloat162float(a.in2[at])));
    *mixed = v;
    return a.pro == PRO_COPY ? v : bfr(v * r);
}

__global__ void k_gemv(Gemv a) {
    extern __shared__ __align__(16) unsigned char gemv_buf[];
    const int K = a.cols, kc = a.chunk, ks = kc + VEC, tid = threadIdx.x, nthreads = blockDim.x, warp = tid / WARP,
              lane = tid % WARP, warps = nthreads / WARP, g = lane / GROUP, t = lane % GROUP;
    const int tiles = (a.B + MMA_N - 1) / MMA_N, slots = tiles * MMA_N;
    bf16 *xs = reinterpret_cast<bf16 *>(gemv_buf);                    /* [slots][ks] */
    float *rs = reinterpret_cast<float *>(xs + (size_t)slots * ks);    /* [slots]: each slot's norm factor */
    const unsigned long long mask = *a.active;
    if (a.pro != PRO_COPY) {
        for (int b = warp; b < a.B; b += warps)
            if (live(mask, b)) {
                float s = 0.f, m;
                for (int i = lane; i < K; i += WARP) {
                    const float v = (pro_value(a, (size_t)b * K + i, 1.f, &m), m);
                    s = __fmaf_rn(v, v, s);
                }
                s = warp_sum(s);
                if (lane == 0) rs[b] = rsqrtf(s / (float)K + FLT_EPSILON);
            }
    }
    __syncthreads();

    const int row_tiles = (a.rows + MMA_M - 1) / MMA_M;
    for (int base = blockIdx.x * warps; base < row_tiles; base += gridDim.x * warps) {
        const bool first = base == (int)blockIdx.x * warps;  /* block 0's first round writes pro_out, pro_out2 */
        const int tile = base + warp, row0 = tile * MMA_M;
        const int r0 = min(row0 + g, a.rows - 1), r1 = min(row0 + g + MMA_M / 2, a.rows - 1);  /* past the end: loads only */
        const uint4 *w0 = reinterpret_cast<const uint4 *>(a.w + (size_t)r0 * K);
        const uint4 *w1 = reinterpret_cast<const uint4 *>(a.w + (size_t)r1 * K);
        float acc[NTILES][ACC];
#pragma unroll
        for (int n = 0; n < NTILES; n++)
            for (int i = 0; i < ACC; i++) acc[n][i] = 0.f;
        for (int k0 = 0; k0 < K; k0 += kc) {
            const int kn = min(kc, K - k0), vecs = kn / VEC;
            __syncthreads();  /* the previous chunk is consumed */
            for (int i = tid; i < slots * vecs; i += nthreads) {
                const int b = i / vecs, c = i % vecs;
                float v[VEC] = {}, m[VEC];
                if (b < a.B && live(mask, b)) {
                    const size_t at = (size_t)b * K + k0 + c * VEC;
                    for (int e = 0; e < VEC; e++) v[e] = pro_value(a, at + e, a.pro == PRO_COPY ? 1.f : rs[b], m + e);
                    if (a.pro == PRO_RESID_NORM && blockIdx.x == 0 && first) {
                        store(a.pro_out + at, m);
                        store(a.pro_out2 + at, v);
                    }
                }
                store(xs + (size_t)b * ks + c * VEC, v);
            }
            __syncthreads();
            if (tile >= row_tiles) continue;
            for (int kk = 0; kk < kn; kk += K_BLOCK) {
                const int q = (k0 + kk) / VEC + t;  /* this lane's 16-byte vector of the rows */
                const uint4 a0 = __ldcs(w0 + q), a1 = __ldcs(w1 + q);
#pragma unroll
                for (int n = 0; n < NTILES; n++) {
                    if (n >= tiles) break;
                    const uint4 x = *reinterpret_cast<const uint4 *>(xs + (size_t)(n * MMA_N + g) * ks + kk + t * VEC);
                    mma16816(acc[n], a0.x, a1.x, a0.y, a1.y, x.x, x.y);
                    mma16816(acc[n], a0.z, a1.z, a0.w, a1.w, x.z, x.w);
                }
            }
        }
        if (tile >= row_tiles) continue;
#pragma unroll
        for (int n = 0; n < NTILES; n++) {
            if (n >= tiles) break;
            for (int i = 0; i < ACC; i++) {  /* the m16n8 accumulator fragment: rows g, g + 8; columns 2t, 2t + 1 */
                const int row = row0 + g + i / 2 * (MMA_M / 2), b = n * MMA_N + t * 2 + i % 2;
                if (row >= a.rows || b >= a.B || !live(mask, b)) continue;
                const float s = acc[n][i];
                const size_t at = (size_t)b * a.rows + row;
                if (a.epi == EPI_RELU2) {
                    a.out[at] = __float2bfloat16_rn(__fmul_rn(fmaxf(s, 0.f), s));
                    continue;
                }
                const float y = bfr(s);  /* the GEMV's bf16 output */
                if (a.epi == EPI_STORE) {
                    a.out[at] = __float2bfloat16_rn(y);
                } else if (a.epi == EPI_RESID) {
                    a.out[at] = __float2bfloat16_rn(__bfloat162float(a.res[at]) + y);
                } else {  /* EPI_ARGMAX: largest fp32 logit, lowest row among exact ties; the IEEE bits mapped to an
                           * unsigned order (negatives inverted, positives above them) */
                    const uint32_t bits = __float_as_uint(s);
                    const uint32_t ordered = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
                    atomicMax(a.argmax + b, ((unsigned long long)ordered << 32) | (UINT32_MAX - (uint32_t)row));
                }
            }
        }
    }
}

/* k_gemv's shared memory: the inputs [n tiles of slots][kc + VEC] bf16, then a norm factor per slot. */
static size_t gemv_smem(int B, int kc) {
    const size_t slots = (size_t)(B + MMA_N - 1) / MMA_N * MMA_N;
    return slots * (kc + VEC) * sizeof(bf16) + slots * sizeof(float);
}

/* The largest chunk of whole K_BLOCK blocks, at most K, whose inputs fit half the opt-in shared memory per block: two
 * co-resident blocks, the fewest that overlap one block's chunk fill with the other's MMAs. */
static int gemv_chunk(int B, int K, int optin) {
    int kc = K / K_BLOCK * K_BLOCK;
    while (kc > K_BLOCK && gemv_smem(B, kc) > (size_t)optin / 2) kc -= K_BLOCK;
    return kc;
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* Embedding: x0 = rms_norm(wte[token]), one bf16 rounding. A block per slot, a thread per channel. */

__global__ void k_embed(const int32_t *tokens, const unsigned long long *active, const bf16 *wte, bf16 *x0, int C) {
    __shared__ float r;
    const int b = blockIdx.x, tid = threadIdx.x;
    if (!live(*active, b)) return;
    const bf16 *e = wte + (size_t)tokens[b] * C;
    if (tid < WARP) {
        float s = 0.f;
        for (int i = tid; i < C; i += WARP) s = __fmaf_rn(__bfloat162float(e[i]), __bfloat162float(e[i]), s);
        s = warp_sum(s);
        if (tid == 0) r = rsqrtf(s / (float)C + FLT_EPSILON);
    }
    __syncthreads();
    for (int i = tid; i < C; i += blockDim.x) x0[(size_t)b * C + i] = __float2bfloat16_rn(__bfloat162float(e[i]) * r);
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* Attention split over the KV cache (flash decoding): (head, split, slot) blocks of HEAD_DIM threads.
 *
 * Each block runs its head's prologue (split 0 writes the position's k, v to the cache), then an online softmax over
 * its run of attn_run cached keys from the window lo..p-1, so the split count depends on the position alone. Warps take
 * the run's keys in turn, each keeping (m, l, acc); the warps merge in order into the split's partial in global memory.
 * The head's last block to finish (an atomic counter) weighs each split by exp(m - max m), sums the partials in split
 * order, adds the current key and value from its own prologue, and writes acc / l rounded to bf16 (FlashAttention's
 * output). Every sum's order is fixed by the window and the position. Cache loads stream (__ldcs). */

typedef struct {
    const bf16 *qkv, *h, *ve;  /* [B][3C], [B][C] (the normed input), the layer's value-embedding table or NULL */
    const float *ve_gate;      /* [H][C] fp32: the value-embedding gate's weights */
    const float *q_gain, *k_gain;  /* [HEAD_DIM] fp32, shared by the heads */
    const float *cos, *sin;    /* the rotary table [max_len][HALF], fp32 */
    bf16 *kc, *vc;             /* the layer's cache of slot 0, [max_len][C]; slot b's at + b kv_slot */
    size_t kv_slot;
    bf16 *out;                 /* [B][C] */
    const int32_t *tokens, *pos;  /* [B]: each slot's input token and its position */
    const unsigned long long *active;
    float *part;               /* slot b, head h, split s: [(b H + h) part_slot + s][HEAD_DIM], each split's value sum */
    float2 *stat;              /* same index: each split's maximum score m (x) and exponential sum l (y) */
    unsigned *done;            /* [B][H]: the head's splits finished in this launch; its last resets it */
    int part_slot;             /* splits per head and slot the partial buffers hold */
    int C, H, window, run;
} Attn;

/* Slot b's q, k, v of the head at position p (fp32 holding bf16 values), as gpt.py: VE layers
 * v += 2 sigmoid(ve_gate[head] . h) value_embedding[token] in fp32, one bf16 rounding; q and k by qk_row (qk_row.cuh;
 * threads 0..QK_LANES-1 the query row, the next QK_LANES the key row). Ends with the block synchronized. */
static __device__ void attn_prologue(const Attn &a, int b, int head, int p, float *q, float *k, float *v) {
    const int tid = threadIdx.x, lane = tid % WARP;
    const bf16 *qkv = a.qkv + (size_t)b * 3 * a.C + head * HEAD_DIM;
    float t = __bfloat162float(qkv[2 * a.C + tid]);
    if (a.ve) {
        const bf16 *h = a.h + (size_t)b * a.C, *ve = a.ve + (size_t)a.tokens[b] * a.C;
        const float *w = a.ve_gate + (size_t)head * a.C;
        float d = 0.f;
        for (int i = lane; i < a.C; i += WARP) d = __fmaf_rn(__bfloat162float(h[i]), w[i], d);
        const float g = __fmul_rn(2.f, sigmoidf(warp_sum(d)));
        t = __fadd_rn(t, __fmul_rn(g, __bfloat162float(ve[head * HEAD_DIM + tid])));
    }
    v[tid] = bfr(t);
    if (tid < 2 * QK_LANES) {
        const bool isq = tid < QK_LANES;
        const int o = tid % QK_LANES * VEC;
        const size_t tab = (size_t)p * HALF + o;
        float y1[VEC], y2[VEC], *y = isq ? q : k;
        qk_row(qkv + (isq ? 0 : a.C) + o, (isq ? a.q_gain : a.k_gain) + o, a.cos + tab, a.sin + tab, y1, y2);
        for (int i = 0; i < VEC; i++) {
            y[o + i] = bfr(y1[i]);
            y[o + HALF + i] = bfr(y2[i]);
        }
    }
    __syncthreads();
}

/* A lane's LANE_DIMS dims of a cached bf16 key or value row, as fp32: one 8-byte streaming load. */
static_assert(LANE_DIMS == 4, "a lane reads its dims of a cached row as one uint2, of a partial as one float4");
static __device__ __forceinline__ void load_dims(const bf16 *row, float *x) {
    const uint2 u = __ldcs((const uint2 *)row);
    const float2 lo = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&u.x));
    const float2 hi = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&u.y));
    x[0] = lo.x; x[1] = lo.y; x[2] = hi.x; x[3] = hi.y;
}

/* q . k / sqrt(HEAD_DIM) (FlashAttention's default softmax scale) to every lane: lane products in order, warp_sum. */
static __device__ __forceinline__ float key_score(const float *q, const float *k) {
    float d = __fmul_rn(q[0], k[0]);
    for (int e = 1; e < LANE_DIMS; e++) d = __fmaf_rn(q[e], k[e], d);
    return __fmul_rn(warp_sum(d), rsqrtf((float)HEAD_DIM));
}

/* exp(m - top); 0 for an empty state (m = top = -inf would give NaN) */
static __device__ __forceinline__ float rescale(float m, float top) { return m == -INFINITY ? 0.f : expf(m - top); }

__global__ void __launch_bounds__(HEAD_DIM) k_attn(Attn a) {
    extern __shared__ float weight[];  /* [splits]: the last block's weight of each split's partial */
    __shared__ float q[HEAD_DIM], k[HEAD_DIM], v[HEAD_DIM];
    __shared__ float wm[ATTN_WARPS], wl[ATTN_WARPS], wacc[ATTN_WARPS][HEAD_DIM], total, current;
    __shared__ bool last;
    const int head = blockIdx.x, split = blockIdx.y, b = blockIdx.z, tid = threadIdx.x, warp = tid / WARP,
              lane = tid % WARP;
    if (!live(*a.active, b)) return;
    const int p = a.pos[b];
    /* each warp's first key is requested before the prologue, so the cache streams while the prologue runs */
    const int lo = p > a.window ? p - a.window : 0, splits = p > lo ? (p - lo + a.run - 1) / a.run : 1;
    if (split >= splits) return;
    const int begin = lo + split * a.run, end = begin + a.run < p ? begin + a.run : p;
    const bf16 *kc = a.kc + (size_t)b * a.kv_slot, *vc = a.vc + (size_t)b * a.kv_slot;
    const size_t col = (size_t)head * HEAD_DIM + lane * LANE_DIMS;
    int j = begin + warp;
    float kd[LANE_DIMS], vd[LANE_DIMS];
    if (j < end) load_dims(kc + (size_t)j * a.C + col, kd), load_dims(vc + (size_t)j * a.C + col, vd);
    attn_prologue(a, b, head, p, q, k, v);
    if (split == 0) {
        const size_t at = (size_t)b * a.kv_slot + (size_t)p * a.C + head * HEAD_DIM + tid;
        a.kc[at] = __float2bfloat16_rn(k[tid]);
        a.vc[at] = __float2bfloat16_rn(v[tid]);
    }

    float qd[LANE_DIMS], acc[LANE_DIMS], m = -INFINITY, l = 0.f;
    for (int e = 0; e < LANE_DIMS; e++) qd[e] = q[lane * LANE_DIMS + e], acc[e] = 0.f;
    for (; j < end; j += ATTN_WARPS) {
        const float s = key_score(qd, kd);
        if (s > m) {
            const float c = rescale(m, s);
            l = __fmul_rn(l, c);
            for (int e = 0; e < LANE_DIMS; e++) acc[e] = __fmul_rn(acc[e], c);
            m = s;
        }
        const float w = expf(s - m);
        l = __fadd_rn(l, w);
        for (int e = 0; e < LANE_DIMS; e++) acc[e] = __fmaf_rn(w, vd[e], acc[e]);
        const int next = j + ATTN_WARPS;
        if (next < end) load_dims(kc + (size_t)next * a.C + col, kd), load_dims(vc + (size_t)next * a.C + col, vd);
    }

    /* the warps' states merged in warp order into the split's partial */
    if (lane == 0) wm[warp] = m, wl[warp] = l;
    for (int e = 0; e < LANE_DIMS; e++) wacc[warp][lane * LANE_DIMS + e] = acc[e];
    __syncthreads();
    float bm = -INFINITY, bl = 0.f, bacc = 0.f;
    for (int w = 0; w < ATTN_WARPS; w++) bm = fmaxf(bm, wm[w]);
    for (int w = 0; w < ATTN_WARPS; w++) {
        const float c = rescale(wm[w], bm);
        bl = __fmaf_rn(c, wl[w], bl);
        bacc = __fmaf_rn(c, wacc[w][tid], bacc);
    }
    const size_t base = ((size_t)b * a.H + head) * a.part_slot;
    float *part = a.part + base * HEAD_DIM;
    float2 *stat = a.stat + base;
    part[(size_t)split * HEAD_DIM + tid] = bacc;
    if (tid == 0) stat[split] = make_float2(bm, bl);
    __threadfence();  /* the partial is visible to every block before this block counts as done */
    __syncthreads();
    unsigned *done = a.done + (size_t)b * a.H + head;
    if (tid == 0) last = atomicAdd(done, 1u) == (unsigned)splits - 1;
    __syncthreads();
    if (!last) return;

    /* the head's last block: the partials (from L2, written by other SMs) in split order, then the current key */
    if (tid == 0) *done = 0;
    if (warp == 0) {
        const float cur = key_score(q + lane * LANE_DIMS, k + lane * LANE_DIMS);
        float top = cur;
        for (int s = lane; s < splits; s += WARP) top = fmaxf(top, __ldcg(stat + s).x);
        top = warp_max(top);
        float sum = 0.f;
        for (int s = lane; s < splits; s += WARP) {
            const float2 ml = __ldcg(stat + s);
            const float c = rescale(ml.x, top);
            weight[s] = c;
            sum = __fmaf_rn(c, ml.y, sum);
        }
        sum = warp_sum(sum);
        if (lane == 0) current = expf(cur - top), total = __fadd_rn(sum, expf(cur - top));
    }
    __syncthreads();
    /* warp w adds the splits w, w + ATTN_WARPS, ... (a lane's dims, one 16-byte load each), then the warps in order */
    float o[LANE_DIMS] = {};
    for (int s = warp; s < splits; s += ATTN_WARPS) {
        const float4 x = __ldcg((const float4 *)(part + (size_t)s * HEAD_DIM) + lane);
        const float c = weight[s];
        o[0] = __fmaf_rn(c, x.x, o[0]); o[1] = __fmaf_rn(c, x.y, o[1]);
        o[2] = __fmaf_rn(c, x.z, o[2]); o[3] = __fmaf_rn(c, x.w, o[3]);
    }
    for (int e = 0; e < LANE_DIMS; e++) wacc[warp][lane * LANE_DIMS + e] = o[e];
    __syncthreads();
    float y = 0.f;
    for (int w = 0; w < ATTN_WARPS; w++) y = __fadd_rn(y, wacc[w][tid]);
    y = __fmaf_rn(current, v[tid], y);
    a.out[(size_t)b * a.C + head * HEAD_DIM + tid] = __float2bfloat16_rn(__fdiv_rn(y, total));
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* Each active slot's argmax becomes its next token and its position advances; a thread per slot. */
__global__ void k_commit(int32_t *tokens, int32_t *pos, const unsigned long long *active, unsigned long long *argmax,
                         int B) {
    const int b = threadIdx.x;
    if (b >= B || !live(*active, b)) return;
    tokens[b] = (int32_t)(UINT32_MAX - (uint32_t)argmax[b]);
    pos[b] += 1;
    argmax[b] = 0ull;
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* Host side: one decoder per model, its step captured as a CUDA graph. */

typedef struct {
    const bf16 *qkv, *o, *fc, *proj, *ve;  /* fc: [F][C]; ve: NULL in a layer without value embeddings */
    const float *ve_gate;                 /* the value-embedding gate [H][C], fp32; NULL without */
    const float *q_gain, *k_gain;         /* [HEAD_DIM] fp32 */
    float resid_lambda, x0_lambda;
    int window;
} Layer;

typedef struct {
    int B, L, H, C, F, vocab, max_len, sms, optin, attn_wave, attn_run;  /* optin: the opt-in shared memory per block */
    const bf16 *wte, *lm_head;
    const float *cos, *sin;   /* the fp32 rotary table [max_len][HALF] */
    Layer *layer;
    int32_t *tokens, *pos;    /* [B]: each slot's input token and position */
    unsigned long long *active;  /* device: the active slots' bit mask */
    int32_t *host_tokens;     /* pinned host [B]: the last step's tokens */
    unsigned long long *argmax;  /* [B] */
    bf16 *x, *x0, *xm, *h, *qkv_out, *att, *xo, *u;  /* [B][...] */
    bf16 *kc, *vc;            /* [B][L][max_len][C] */
    float *attn_part;         /* k_attn's partials, reused by every layer: [B][H][attn_wave][HEAD_DIM] value sums */
    float2 *attn_stat;        /* [B][H][attn_wave]: their (m, l) */
    unsigned *attn_done;      /* [B][H]: k_attn's counters, zero between launches */
    cudaStream_t stream;      /* the caller's */
    cudaGraph_t graph;
    cudaGraphExec_t exec;
} Decoder;

/* f(buffer, bytes per slot) for each per-slot device buffer. */
template <class F> static void slot_buffers(Decoder *d, F f) {
    const size_t C = d->C, kv = (size_t)d->L * d->max_len * C * sizeof(bf16), part = (size_t)d->H * d->attn_wave;
    f((void **)&d->tokens, sizeof(int32_t));
    f((void **)&d->pos, sizeof(int32_t));
    f((void **)&d->argmax, sizeof(unsigned long long));
    for (bf16 **b : {&d->x, &d->x0, &d->xm, &d->h, &d->att, &d->xo}) f((void **)b, C * sizeof(bf16));
    f((void **)&d->qkv_out, 3 * C * sizeof(bf16));
    f((void **)&d->u, d->F * sizeof(bf16));
    f((void **)&d->kc, kv);
    f((void **)&d->vc, kv);
    f((void **)&d->attn_part, part * HEAD_DIM * sizeof(float));
    f((void **)&d->attn_stat, part * sizeof(float2));
    f((void **)&d->attn_done, d->H * sizeof(unsigned));
}

static Gemv gemv(const Decoder *d, const bf16 *w, int rows, int cols, int pro, const bf16 *in, int epi, bf16 *out) {
    Gemv g = {};
    g.w = w; g.rows = rows; g.cols = cols; g.pro = pro; g.in = in; g.epi = epi; g.out = out;
    g.active = d->active; g.B = d->B;
    return g;
}

/* Block size from cudaOccupancyMaxPotentialBlockSize, fewer warps when that would leave SMs without a block; grid at
 * most one resident wave (warps loop over the tiles). */
static void launch_gemv(Decoder *d, Gemv g) {
    g.chunk = gemv_chunk(g.B, g.cols, d->optin);
    const size_t smem = gemv_smem(g.B, g.chunk);
    const int tiles = (g.rows + MMA_M - 1) / MMA_M;
    int grid, block, resident;
    cudaOccupancyMaxPotentialBlockSize(&grid, &block, k_gemv, smem);
    const int spread = (tiles + d->sms - 1) / d->sms, warps = spread < block / WARP ? spread : block / WARP;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, k_gemv, WARP * warps, smem);
    const int needed = (tiles + warps - 1) / warps, wave = d->sms * resident;
    k_gemv<<<needed < wave ? needed : wave, WARP * warps, smem, d->stream>>>(g);
}

/* One step of every active slot on d->stream (captured once into the graph). */
static void enqueue_step(Decoder *d) {
    const int C = d->C, F = d->F, L = d->L, B = d->B;
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, k_embed);
    k_embed<<<B, C < fa.maxThreadsPerBlock ? C : fa.maxThreadsPerBlock, 0, d->stream>>>(d->tokens, d->active, d->wte,
                                                                                       d->x0, C);
    for (int l = 0; l < L; l++) {
        const Layer *ly = &d->layer[l];
        /* input: x0 at layer 0, else the previous layer's output */
        Gemv g = gemv(d, ly->qkv, 3 * C, C, PRO_RESID_NORM, l == 0 ? d->x0 : d->x, EPI_STORE, d->qkv_out);
        g.in2 = d->x0; g.l1 = ly->resid_lambda; g.l2 = ly->x0_lambda; g.pro_out = d->xm; g.pro_out2 = d->h;
        launch_gemv(d, g);
        /* splits per head and slot: runs of attn_run keys over at most the window */
        const int keys = ly->window < d->max_len - 1 ? ly->window : d->max_len - 1;
        const int splits = (keys + d->attn_run - 1) / d->attn_run;
        const Attn a = {d->qkv_out, d->h, ly->ve, ly->ve_gate, ly->q_gain, ly->k_gain, d->cos, d->sin,
                        d->kc + (size_t)l * d->max_len * C, d->vc + (size_t)l * d->max_len * C,
                        (size_t)L * d->max_len * C, d->att, d->tokens, d->pos, d->active, d->attn_part, d->attn_stat,
                        d->attn_done, d->attn_wave, C, d->H,
                        ly->window, d->attn_run};
        k_attn<<<dim3(d->H, splits, B), HEAD_DIM, splits * sizeof(float), d->stream>>>(a);
        g = gemv(d, ly->o, C, C, PRO_COPY, d->att, EPI_RESID, d->xo);
        g.res = d->xm;
        launch_gemv(d, g);
        launch_gemv(d, gemv(d, ly->fc, F, C, PRO_NORM, d->xo, EPI_RELU2, d->u));
        g = gemv(d, ly->proj, C, F, PRO_COPY, d->u, EPI_RESID, d->x);
        g.res = d->xo;
        launch_gemv(d, g);
    }
    /* the final norm in lm_head's prologue, the argmax in its epilogue */
    Gemv g = gemv(d, d->lm_head, d->vocab, C, PRO_NORM, d->x, EPI_ARGMAX, NULL);
    g.argmax = d->argmax;
    launch_gemv(d, g);
    k_commit<<<1, MAX_BATCH, 0, d->stream>>>(d->tokens, d->pos, d->active, d->argmax, B);
    cudaMemcpyAsync(d->host_tokens, d->tokens, (size_t)B * sizeof(int32_t), cudaMemcpyDeviceToHost, d->stream);
}

extern "C" void dec_destroy(void *handle) {
    Decoder *d = (Decoder *)handle;
    cudaGraphExecDestroy(d->exec);
    cudaGraphDestroy(d->graph);
    slot_buffers(d, [](void **p, size_t) { cudaFree(*p); });
    cudaFree(d->active);
    cudaFreeHost(d->host_tokens);
    free(d->layer);
    free(d);
}

/* The slots of a decoder (dec_create). */
extern "C" int dec_batch(void *handle) { return ((Decoder *)handle)->B; }

/*
 * cfg: L, C, MLP hidden width F (fc: [F][C]; proj: [C][F]), vocab (lm_head's rows), max_len.
 * ptrs: GLOBAL_PTRS pointers wte, cos, sin (the fp32 rotary table [max_len][HALF] of every layer), lm_head; then
 *       LAYER_PTRS per layer: qkv, o, fc, proj, value-embedding table (0 if none), value-embedding gate (0 if none),
 *       q gain, k gain.
 * scal: per layer: resid lambda, x0 lambda (bf16 values).
 * window: per layer, the keys before each query that it attends (FlashAttention's window_size left).
 * stream: the caller's stream (not the legacy default stream: it is captured), alive past dec_destroy.
 */
constexpr int GLOBAL_PTRS = 4, LAYER_PTRS = 8;

extern "C" void *dec_create(const int *cfg, const uint64_t *ptrs, const float *scal, const int *window, void *stream) {
    Decoder *d = (Decoder *)calloc(1, sizeof(Decoder));
    d->L = cfg[0]; d->C = cfg[1]; d->F = cfg[2]; d->vocab = cfg[3]; d->max_len = cfg[4];
    d->H = d->C / HEAD_DIM;
    d->wte = (const bf16 *)ptrs[0]; d->cos = (const float *)ptrs[1]; d->sin = (const float *)ptrs[2];
    d->lm_head = (const bf16 *)ptrs[3]; d->stream = (cudaStream_t)stream;
    const int L = d->L;
    d->layer = (Layer *)calloc(L, sizeof(Layer));
    for (int l = 0; l < L; l++) {
        const uint64_t *p = ptrs + GLOBAL_PTRS + LAYER_PTRS * l;
        d->layer[l] = {(const bf16 *)p[0], (const bf16 *)p[1], (const bf16 *)p[2], (const bf16 *)p[3], (const bf16 *)p[4],
                       (const float *)p[5], (const float *)p[6], (const float *)p[7], scal[2 * l], scal[2 * l + 1],
                       window[l]};
    }
    int device;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&d->sms, cudaDevAttrMultiProcessorCount, device);
    cudaDeviceGetAttribute(&d->optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    cudaFuncSetAttribute(k_gemv, cudaFuncAttributeMaxDynamicSharedMemorySize, d->optin);
    /* attn_wave: (head, split) blocks the SMs hold at once, per head; residency is retaken with the per-split shared
     * memory of that wave (fewer splits fit). attn_run: keys per split, so max_len - 1 keys take one wave */
    int resident;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, k_attn, HEAD_DIM, 0);
    d->attn_wave = d->sms * resident / d->H;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, k_attn, HEAD_DIM, d->attn_wave * sizeof(float));
    d->attn_wave = d->sms * resident / d->H;
    d->attn_run = (d->max_len - 1 + d->attn_wave - 1) / d->attn_wave;
    /* B: the most slots (at most MAX_BATCH) that fit free memory, leaving one slot's bytes for the batch-1 prefill
     * (its KV cache of at most max_len tokens) */
    size_t slot = 0, free_bytes, total_bytes;
    slot_buffers(d, [&](void **, size_t bytes) { slot += bytes; });
    cudaMemGetInfo(&free_bytes, &total_bytes);
    d->B = (int)(free_bytes / slot - 1 < (size_t)MAX_BATCH ? free_bytes / slot - 1 : MAX_BATCH);
    slot_buffers(d, [&](void **p, size_t bytes) {
        cudaMalloc(p, d->B * bytes);
        cudaMemset(*p, 0, d->B * bytes);
    });
    cudaMalloc((void **)&d->active, sizeof(unsigned long long));
    cudaMemset(d->active, 0, sizeof(unsigned long long));
    cudaHostAlloc((void **)&d->host_tokens, d->B * sizeof(int32_t), cudaHostAllocDefault);
    cudaStreamBeginCapture(d->stream, cudaStreamCaptureModeThreadLocal);
    enqueue_step(d);
    cudaStreamEndCapture(d->stream, &d->graph);
    cudaGraphInstantiate(&d->exec, d->graph, 0);
    return d;
}

/* Start slot b from PyTorch's batch-1 KVCache of its first pos tokens ((layer, position, head, dim), layer stride
 * layer_stride elements); its steps continue from token at position pos once active. Stream-ordered. */
extern "C" void dec_set_prefilled(void *handle, int b, int32_t token, int32_t pos, const bf16 *k, const bf16 *v,
                                  long long layer_stride) {
    Decoder *d = (Decoder *)handle;
    const size_t row = (size_t)d->C * sizeof(bf16), dst_pitch = (size_t)d->max_len * row,
                 src_pitch = (size_t)layer_stride * sizeof(bf16), slot = (size_t)d->L * d->max_len * d->C;
    cudaMemcpyAsync(d->tokens + b, &token, sizeof(int32_t), cudaMemcpyHostToDevice, d->stream);
    cudaMemcpyAsync(d->pos + b, &pos, sizeof(int32_t), cudaMemcpyHostToDevice, d->stream);
    cudaMemcpy2DAsync(d->kc + b * slot, dst_pitch, k, src_pitch, (size_t)pos * row, d->L, cudaMemcpyDeviceToDevice,
                      d->stream);
    cudaMemcpy2DAsync(d->vc + b * slot, dst_pitch, v, src_pitch, (size_t)pos * row, d->L, cudaMemcpyDeviceToDevice,
                      d->stream);
}

/* The active slots (bit b: slot b) of the next steps. Stream-ordered. */
extern "C" void dec_set_active(void *handle, unsigned long long mask) {
    Decoder *d = (Decoder *)handle;
    cudaMemcpyAsync(d->active, &mask, sizeof(mask), cudaMemcpyHostToDevice, d->stream);
}

/* Launch one step (asynchronous); dec_tokens returns its tokens. */
extern "C" void dec_launch(void *handle) {
    Decoder *d = (Decoder *)handle;
    cudaGraphLaunch(d->exec, d->stream);
}

/* Wait for the launched step; out[b]: the token slot b took (stale for inactive slots). */
extern "C" void dec_tokens(void *handle, int32_t *out) {
    Decoder *d = (Decoder *)handle;
    cudaStreamSynchronize(d->stream);
    for (int b = 0; b < d->B; b++) out[b] = d->host_tokens[b];
}
