/*
 * Greedy decoding of a nanochat GPT (nanochat/model/gpt.py) in CUDA C for a batch of B independent sequences, one
 * decoding step of all of them captured as a CUDA graph (nanochat/inference/decoder.py).
 *
 * At small batch a step reads every weight once and does little else, so it is bound by memory bandwidth and, in an
 * eager framework, by kernel launches. Here one step advances each of the B sequences (slots) by one token: every
 * weight is read once for all of them, and the step is 5 kernels per layer plus 5 (2 more per Engram layer), captured
 * once as two CUDA graphs and replayed: the front (the embedding and the blocks before the first Engram layer) and the
 * back (the rest). Every value a step depends on (each slot's position, input token, previous embedding, KV cache,
 * key-offset slots, Engram conv ring; which slots are active) lives in device memory, and so do the Engram memory rows
 * of the slots' input tokens: the front writes each active slot's last n tokens (k_window, n = the longest n-gram) into
 * the caller's window buffer, from which the caller gathers the rows from the model's tables on the GPU into the rows
 * buffer, on the decoder's stream (the caller's, dec_create) between the front and the back (decoder.py): the front
 * runs while the host enqueues the gather (DeepSeek Engram §6.4: the blocks before the first Engram layer overlap the
 * lookup). A sequence starts in a
 * slot from the model's batch-1 prefill of all its prompt tokens but the last, whose KV cache, last embedding, Engram
 * conv inputs and last key are copied in (dec_set_prefilled); the steps continue it greedily. Inactive slots (a bit
 * mask) are skipped by every kernel.
 *
 * Each step: embed (rms norm, smear) -> per layer [Engram layers: W_kv GEMV of the layer's memory rows (k and v), then
 * the gate and short convolution with the residual mix (h = resid_lambda x + x0_lambda x0) in its prologue, fusing them
 * into h (k_engram: alpha = sigmoid(norm_h(h) . norm_k(k) / sqrt(C)), Vt = alpha v, h + silu(conv(norm_v(Vt))) + Vt,
 * the conv's earlier rows kept per Engram layer); before the last block MUDD's first site (k_mudd: the coefficient MLP
 * of the mixed input, its residual stream and the value stream over x0 and earlier blocks' outputs, which the step
 * keeps); qkv GEMV with the residual mix (or, after Engram or MUDD, only) and rms norm in its prologue; attention split
 * over the KV cache (split-KV), with value-embedding gate and MUDD's value stream, the q/k prologue of qk_prologue.cu
 * (rotary over the first ROTARY_DIM dims, QK norm and scale, key offset on its layers), the KV-cache write, a causal
 * sliding-window online softmax, XSA and the attention output gate; output GEMV + residual; MLP-in GEMV with rms norm
 * prologue and SwiGLU; MLP-out GEMV + residual] -> MUDD's second site -> lm_head GEMV with the final norm in its
 * prologue and an argmax (atomicMax on a packed key) in its epilogue -> commit (each active slot's argmax becomes its
 * next input unless its prompt still supplies it).
 *
 * Numerics follow the PyTorch model's eager forward op by op: the matrices are bf16 copies of the fp32 parameters
 * (rounded once, as the training forward rounds them at use); the Engram rows as the model's gather gives them (bf16,
 * the caller gathers them with the model's own function); fp32 rotary tables and Engram norm/conv weights; the smear
 * and value-embedding gates in fp32 on fp32
 * weights and lambda (one operation per PyTorch op, no contraction), each mix rounded once to bf16; the attention
 * output gate's and MUDD's first-layer logits fp32 from bf16 operands (nanochat::narrow_linear), MUDD's coefficients
 * and mixes in fp32 (mudd.py), each mixed state rounded once; the resid/x0 lambdas applied as bf16 values to bf16
 * tensors (PyTorch's promotion of a 0-dim fp32 tensor with a bf16 tensor); fp32 accumulation; values rounded to bf16
 * wherever PyTorch stores a bf16 tensor; F.rms_norm's fp32 epsilon; the greedy argmax over the fp32 logits (softcap
 * tanh is monotone, so it is the argmax of the model's distribution; bf16 rounding first would create ties).
 *
 * Every slot is computed alone: no sum mixes slots, and every sum of a slot is taken in an order fixed by the model's
 * shapes and the slot's own position (each GEMV output by the tensor cores' MMAs over K in a fixed order, the slot one
 * column of them, whichever warp or block computes the row; attention's split count from the position, attn_run). So a
 * sequence's tokens and logits do not depend on the batch size, on the other sequences in the batch or on timing.
 *
 * Launch shapes follow from the kernels' limits and the problem size: a GEMV runs a warp per 16-row tile (looping over
 * tiles), in blocks of the size that maximizes its occupancy (cudaOccupancyMaxPotentialBlockSize) or of fewer warps when that
 * grid would leave SMs without a block, at most as many blocks as are resident at once; attention runs a thread per
 * head dim in each of (head, split, slot) blocks, at most a wave of splits per head and slot (see k_attn).
 */
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

typedef __nv_bfloat16 bf16;

#define HEAD_DIM 128                /* gpt_config's head dim; attention keeps a lane's HEAD_DIM / 32 dims in registers */
#define ROTARY_DIM (HEAD_DIM / 2)   /* the rotated dims of each q/k head: its first half (GPT.init_rotary) */
#define LANE_DIMS (HEAD_DIM / 32)   /* a lane's dims of a query, key or value */
#define ATTN_WARPS (HEAD_DIM / 32)  /* an attention block: a thread per head dim */
#define XSA_LANES 8                     /* xsa.cu's LANES: threads per head row */
#define XSA_DIMS (HEAD_DIM / XSA_LANES) /* xsa.cu's DIMS: consecutive dims per lane */
#define MAX_BATCH 64                /* slots: the active mask's bits (one unsigned long long); k_gemv's SwiGLU
                                     * accumulators of 64 slots are 64 fp32 registers per thread (two 16-row tiles) */
#define VEC 8                       /* bf16 per 16-byte access */

enum { PRO_COPY, PRO_RESID_NORM, PRO_NORM };
enum { EPI_STORE, EPI_RESID, EPI_SWIGLU, EPI_ARGMAX };

static __device__ __forceinline__ float bfr(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

static __device__ __forceinline__ float sigmoidf(float x) { return __fdiv_rn(1.f, __fadd_rn(1.f, expf(-x))); }

static __device__ __forceinline__ bool live(unsigned long long mask, int b) { return (mask >> b) & 1ull; }

static __device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

/* Maximum over the warp, to every lane. */
static __device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

/* 8 bf16 (a 16-byte vector) as floats. */
static __device__ __forceinline__ void unpack(const uint4 u, float *v) {
    v[0] = __uint_as_float(u.x << 16); v[1] = __uint_as_float(u.x & 0xffff0000u);
    v[2] = __uint_as_float(u.y << 16); v[3] = __uint_as_float(u.y & 0xffff0000u);
    v[4] = __uint_as_float(u.z << 16); v[5] = __uint_as_float(u.z & 0xffff0000u);
    v[6] = __uint_as_float(u.w << 16); v[7] = __uint_as_float(u.w & 0xffff0000u);
}

static __device__ __forceinline__ void load8(const bf16 *p, float *v) { unpack(*reinterpret_cast<const uint4 *>(p), v); }

/* VEC fp32 values: two 16-byte loads. */
static __device__ __forceinline__ void load8(const float *p, float *v) {
    const float4 a = *reinterpret_cast<const float4 *>(p), b = *reinterpret_cast<const float4 *>(p + 4);
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}

static __device__ __forceinline__ void store8(bf16 *p, const float *v) {
    uint4 u;
    __nv_bfloat162 *h = reinterpret_cast<__nv_bfloat162 *>(&u);
    for (int i = 0; i < VEC / 2; i++) h[i] = __floats2bfloat162_rn(v[2 * i], v[2 * i + 1]);
    *reinterpret_cast<uint4 *>(p) = u;
}

/* w . x of 8 values: the products and their sum in index order, each operation rounded on its own. */
static __device__ __forceinline__ float dot8(const float *w, const float *x) {
    float d = __fmul_rn(w[0], x[0]);
    for (int i = 1; i < VEC; i++) d = __fmaf_rn(w[i], x[i], d);
    return d;
}

/* Sum of x[i] * y[i] over i < n of bf16 vectors x and y, to every lane of the calling warp: lane l adds the terms l,
 * l + 32, ... in turn, then warp_sum adds the lanes. The order depends on n alone. */
static __device__ float warp_dot(const bf16 *x, const bf16 *y, int n) {
    float s = 0.f;
    for (int i = threadIdx.x & 31; i < n; i += 32) s = __fmaf_rn(__bfloat162float(x[i]), __bfloat162float(y[i]), s);
    return warp_sum(s);
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* GEMV of B slots on the tensor cores: out[b][r] = W[r] . x[b] by mma.sync m16n8k16 (bf16 operands, fp32 accumulators),
 * a warp per 16-row tile of W (tiles strided over the grid), every weight read once for all slots. A slot is a column
 * of the MMA's 8-wide n tiles; the MMA computes every output element from its own row and column only, and each
 * element's sum runs over K in the same order (k32 blocks in order, two k16 MMAs each) whatever the batch, so a slot's
 * outputs do not depend on B or on the other slots. Within a k32 block lane (g, t) = (lane / 4, lane % 4) loads the 8
 * consecutive k t*8..t*8+7 of rows g and g + 8 and of its column's input (one 16-byte load each): A and B map the
 * MMA's k index to these physical k the same way, so the product is W . x.
 * The slots' inputs are made by the prologue (each value a bf16 value) a K chunk at a time into shared memory, [8 n
 * tiles of slots][chunk + VEC] (the VEC pad staggers the slot rows over the banks; inactive and padding slots zero), so
 * any batch fits: the chunk follows from the shared memory (gemv_chunk). (RESID_)NORM: each slot's norm factor first,
 * a warp per slot summing the squares lane-strided in fp32. SwiGLU: out[b][r] = silu(W[r] . x) * (W[rows + r] . x),
 * both dot products fp32 as from QuACK's GEMM accumulators, only the product rounded to bf16. */

typedef struct {
    const bf16 *w;               /* [rows][cols]; SWIGLU: [2 rows][cols], the gate rows, then the up rows */
    int rows, cols, chunk;       /* chunk: the k of the slots' inputs held in shared memory at once (gemv_chunk) */
    int pro;                     /* prologue: how a slot's input vector is made */
    const bf16 *in, *in2;        /* [B][cols]: COPY/NORM: in; RESID_NORM: in = x, in2 = x0 */
    float l1, l2;                /* RESID_NORM: resid and x0 lambdas (bf16 values) */
    bf16 *pro_out, *pro_out2;    /* [B][cols], block 0 writes them: RESID_NORM the mixed residual; (RESID_)NORM its norm */
    int epi;                     /* epilogue: what is done with each output */
    bf16 *out;                   /* [B][rows]: STORE/RESID/SWIGLU output */
    const bf16 *res;             /* [B][rows]: RESID: residual added */
    unsigned long long *argmax;  /* [B] ARGMAX: packed (ordered logit, ~row) maximum */
    const unsigned long long *active;  /* the active slots' bit mask */
    int B;
} Gemv;

#define NTILES (MAX_BATCH / 8)  /* the MMA's n tiles of 8 slots */

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
    const int K = a.cols, kc = a.chunk, ks = kc + VEC, tid = threadIdx.x, nthreads = blockDim.x, warp = tid >> 5,
              lane = tid & 31, warps = nthreads >> 5, g = lane >> 2, t = lane & 3;
    const int tiles = (a.B + 7) / 8, slots = tiles * 8;
    bf16 *xs = reinterpret_cast<bf16 *>(gemv_buf);                    /* [slots][ks] */
    float *rs = reinterpret_cast<float *>(xs + (size_t)slots * ks);    /* [slots]: each slot's norm factor */
    const unsigned long long mask = *a.active;
    if (a.pro != PRO_COPY) {
        for (int b = warp; b < a.B; b += warps)
            if (live(mask, b)) {
                float s = 0.f, m;
                for (int i = lane; i < K; i += 32) {
                    const float v = (pro_value(a, (size_t)b * K + i, 1.f, &m), m);
                    s = __fmaf_rn(v, v, s);
                }
                s = warp_sum(s);
                if (lane == 0) rs[b] = rsqrtf(s / (float)K + FLT_EPSILON);
            }
    }
    __syncthreads();

    const bool swiglu = a.epi == EPI_SWIGLU;
    const int row_tiles = (a.rows + 15) / 16;
    for (int base = blockIdx.x * warps; base < row_tiles; base += gridDim.x * warps) {
        const bool first = base == (int)blockIdx.x * warps;  /* block 0's first round writes pro_out, pro_out2 */
        const int tile = base + warp, row0 = tile * 16;
        const int r0 = min(row0 + g, a.rows - 1), r1 = min(row0 + g + 8, a.rows - 1);  /* rows past the end: loads only */
        const uint4 *w0 = reinterpret_cast<const uint4 *>(a.w + (size_t)r0 * K);
        const uint4 *w1 = reinterpret_cast<const uint4 *>(a.w + (size_t)r1 * K);
        const uint4 *u0 = reinterpret_cast<const uint4 *>(a.w + (size_t)(a.rows + r0) * K);  /* SWIGLU: up rows */
        const uint4 *u1 = reinterpret_cast<const uint4 *>(a.w + (size_t)(a.rows + r1) * K);
        float acc[NTILES][4], acc2[NTILES][4];
#pragma unroll
        for (int n = 0; n < NTILES; n++)
            for (int i = 0; i < 4; i++) acc[n][i] = acc2[n][i] = 0.f;
        for (int k0 = 0; k0 < K; k0 += kc) {
            const int kn = min(kc, K - k0), vecs = kn / VEC;
            __syncthreads();  /* the previous chunk is consumed */
            for (int i = tid; i < slots * vecs; i += nthreads) {
                const int b = i / vecs, c = i % vecs;
                float v[VEC] = {}, m[VEC];
                if (b < a.B && live(mask, b)) {
                    const size_t at = (size_t)b * K + k0 + c * VEC;
                    for (int e = 0; e < VEC; e++) v[e] = pro_value(a, at + e, a.pro == PRO_COPY ? 1.f : rs[b], m + e);
                    if (blockIdx.x == 0 && first) {
                        if (a.pro == PRO_RESID_NORM) store8(a.pro_out + at, m);
                        if (a.pro != PRO_COPY && a.pro_out2) store8(a.pro_out2 + at, v);
                    }
                }
                store8(xs + (size_t)b * ks + c * VEC, v);
            }
            __syncthreads();
            if (tile >= row_tiles) continue;
            for (int kk = 0; kk < kn; kk += 4 * VEC) {
                const int q = (k0 + kk) / VEC + t;  /* this lane's 16-byte vector of the rows */
                const uint4 a0 = __ldcs(w0 + q), a1 = __ldcs(w1 + q);
                uint4 c0 = {}, c1 = {};
                if (swiglu) c0 = __ldcs(u0 + q), c1 = __ldcs(u1 + q);
#pragma unroll
                for (int n = 0; n < NTILES; n++) {
                    if (n >= tiles) break;
                    const uint4 x = *reinterpret_cast<const uint4 *>(xs + (size_t)(n * 8 + g) * ks + kk + t * VEC);
                    mma16816(acc[n], a0.x, a1.x, a0.y, a1.y, x.x, x.y);
                    mma16816(acc[n], a0.z, a1.z, a0.w, a1.w, x.z, x.w);
                    if (swiglu) {
                        mma16816(acc2[n], c0.x, c1.x, c0.y, c1.y, x.x, x.y);
                        mma16816(acc2[n], c0.z, c1.z, c0.w, c1.w, x.z, x.w);
                    }
                }
            }
        }
        if (tile >= row_tiles) continue;
#pragma unroll
        for (int n = 0; n < NTILES; n++) {
            if (n >= tiles) break;
            for (int i = 0; i < 4; i++) {
                const int row = row0 + g + (i >> 1) * 8, b = n * 8 + t * 2 + (i & 1);
                if (row >= a.rows || b >= a.B || !live(mask, b)) continue;
                const float s = acc[n][i];
                const size_t at = (size_t)b * a.rows + row;
                if (swiglu) {
                    a.out[at] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(s, sigmoidf(s)), acc2[n][i]));
                    continue;
                }
                const float y = bfr(s);  /* the GEMV's bf16 output */
                if (a.epi == EPI_STORE) {
                    a.out[at] = __float2bfloat16_rn(y);
                } else if (a.epi == EPI_RESID) {
                    a.out[at] = __float2bfloat16_rn(__bfloat162float(a.res[at]) + y);
                } else {  /* EPI_ARGMAX: largest fp32 logit, lowest row among exact ties */
                    const uint32_t bits = __float_as_uint(s);
                    const uint32_t ordered = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
                    atomicMax(a.argmax + b, ((unsigned long long)ordered << 32) | (0xffffffffu - (uint32_t)row));
                }
            }
        }
    }
}

/* Shared memory of k_gemv at B slots and chunk kc: the inputs [8-slot tiles][kc + VEC] bf16, then a norm factor per slot. */
static size_t gemv_smem(int B, int kc) {
    const size_t slots = (size_t)(B + 7) / 8 * 8;
    return slots * (kc + VEC) * sizeof(bf16) + slots * sizeof(float);
}

/* The chunk of a GEMV of K columns: the largest whole number of k32 blocks, at most K, whose inputs fit half the
 * device's opt-in shared memory per block, so that two blocks share an SM (one fills its chunk while the other
 * multiplies). */
static int gemv_chunk(int B, int K, int optin) {
    int kc = K / (4 * VEC) * (4 * VEC);
    while (kc > 4 * VEC && gemv_smem(B, kc) > (size_t)optin / 2) kc -= 4 * VEC;
    return kc;
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* Embedding: x = rms_norm(wte[token]); smear (from the second position on): x += smear_lambda * sigmoid(smear_gate .
 * x[:smear_width]) * previous embedding, the gate and the mix in fp32 (fp32 smear_gate and lambda, as gpt.py), one bf16
 * rounding. A block per slot, a thread per channel. */

__global__ void k_embed(const int32_t *seq, int seq_slot, const int32_t *pos, const unsigned long long *active, const bf16 *wte,
                        const float *smear_gate, int smear_width, float smear_lambda, bf16 *prev, bf16 *x0, int C) {
    extern __shared__ float es[];  /* C floats, then the norm's slot */
    const int b = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    if (!live(*active, b)) return;
    const int p = pos[b];
    const bf16 *e = wte + (size_t)seq[(size_t)b * seq_slot + p] * C;
    prev += (size_t)b * C; x0 += (size_t)b * C;
    for (int i = tid; i < C; i += nt) es[i] = __bfloat162float(e[i]);
    __syncthreads();
    if (tid < 32) {
        float s = 0.f;
        for (int i = tid; i < C; i += 32) s = __fmaf_rn(es[i], es[i], s);
        s = warp_sum(s);
        if (tid == 0) es[C] = rsqrtf(s / (float)C + FLT_EPSILON);
    }
    __syncthreads();
    const float r = es[C];
    __syncthreads();
    for (int i = tid; i < C; i += nt) es[i] = bfr(es[i] * r);
    __syncthreads();
    float gate = 0.f;
    if (p > 0) {
        float d = 0.f;
        for (int i = 0; i < smear_width; i++) d = __fmaf_rn(es[i], smear_gate[i], d);
        gate = __fmul_rn(smear_lambda, sigmoidf(d));
    }
    for (int i = tid; i < C; i += nt) {
        const float n = es[i];
        x0[i] = __float2bfloat16_rn(__fadd_rn(n, __fmul_rn(gate, __bfloat162float(prev[i]))));
        prev[i] = __float2bfloat16_rn(n);
    }
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* Attention, split over the KV cache (flash decoding): a grid of (head, split, slot) blocks of HEAD_DIM threads.
 *
 * Each block runs its head's prologue (attn_prologue; split 0 writes the position's key and value to the cache), then
 * an online softmax over its split of the cached keys: the keys before this position within the window, lo..p-1, cut
 * into runs of attn_run keys (the last shorter), one split per run, so the split count follows from the position alone.
 * The block's warps take the run's keys in turn (each warp's first key requested before the prologue); a warp holds a
 * running maximum score m, exponential sum l and weighted value sum acc (a lane's LANE_DIMS dims), each new maximum
 * rescaling l and acc by exp(m_old - m_new). The warps' states are merged in warp order into the split's partial
 * (acc; m, l) in global memory. The head's last block to finish (a counter orders the blocks, it carries no values)
 * weighs each split by exp(m - max m), adds the partials (warp w the splits w, w + ATTN_WARPS, ... in turn, then the
 * warps in order) and the current key and value (from its own prologue: the cache write may not be visible yet), and
 * writes the head's output acc / l, less its XSA component, times its output gate (xsa.cu's op, bitwise). Blocks whose
 * split lies past the position's runs return at once. Every sum's order is fixed by the window and the position.
 *
 * Shared memory: three head-dim vectors, the warps' states and a weight per split; the context only lengthens the runs.
 * Each step reads the cache once, so its loads stream (__ldcs). */

typedef struct {
    const bf16 *qkv, *h, *ve;  /* [B][3C], [B][C] (the normed input), the layer's value-embedding table or NULL */
    const float *v_add;        /* MUDD: [B][C] the value stream (fp32, unrounded) added to v with the value embedding; or NULL */
    const float *ve_gate;      /* [H][gate_width] fp32: the gate's weights over cat(h[:gate_width / 2], ve[:gate_width / 2]) */
    const float *cos, *sin;    /* rotary tables [max_len][ROTARY_DIM / 2], fp32 */
    bf16 *kc, *vc;             /* the layer's cache of slot 0, [max_len][C]; slot b's at + b kv_slot */
    size_t kv_slot;
    bf16 *out;                 /* [B][C] */
    bf16 *shift;               /* key offset: the layer's [2][C] of slot 0 (slot b's at + b shift_slot), the stationary key
                                * dims (normed, scaled, bf16) of positions of each parity; NULL: no key offset */
    size_t shift_slot;
    const bf16 *out_gate;      /* the attention output gate [H][C] (bf16) */
    const int32_t *seq, *pos;
    int seq_slot;
    const unsigned long long *active;
    float *part;               /* slot b, head h, split s: [(b H + h) part_slot + s][HEAD_DIM], each split's value sum */
    float *stat;               /* same index [2]: each split's maximum score m and exponential sum l */
    unsigned *done;            /* [B][H]: the head's splits finished in this launch; its last resets it */
    int part_slot;             /* splits per head and slot the partial buffers hold */
    int C, H, window, run, gate_width;
    float gate_scale, qk_scale;
} Attn;

/* qk_prologue.cu's forward arithmetic (gpt.py's q/k prologue), kept expression for expression so that given equal
 * inputs the decoder's q and k equal the PyTorch prefill's bit for bit: a head row is QKP_LANES threads, each with two
 * vectors of VEC dims; a rotation lane (the first ROTARY_DIM / (2 VEC) of a row) holds dims o and o + ROTARY_DIM / 2 and
 * rotates them by -theta with the fp32 table entries, any other lane two stationary vectors o and
 * o + (HEAD_DIM - ROTARY_DIM) / 2; the row's sum of squares is reduced by shuffles among its lanes; fp32 inside, one
 * bf16 rounding per output. */
#define QKP_LANES (HEAD_DIM / 2 / VEC)  /* threads per head row, two vectors each */

static __device__ __forceinline__ unsigned qkp_row_mask() {
    return ((1u << QKP_LANES) - 1) << ((threadIdx.x & 31) & ~(QKP_LANES - 1));
}

static __device__ __forceinline__ float qkp_row_sum(float v, unsigned mask) {
    for (int o = QKP_LANES / 2; o > 0; o >>= 1) v += __shfl_xor_sync(mask, v, o);
    return v;
}

/* This lane's two vectors of the row at x into v1, v2 (`pair` apart), rotated if rot (by the table vectors at c, s,
 * kept in cs, sn); returns their sum of squares. */
static __device__ __forceinline__ float qkp_rotated(const bf16 *x, bool rot, int pair, const float *c, const float *s,
                                                    float *v1, float *v2, float *cs, float *sn) {
    load8(x, v1); load8(x + pair, v2);
    float ss = 0.f;
    if (rot) {
        load8(c, cs); load8(s, sn);
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

/* Slot b's query, key and value of the head at position p into q, k, v (fp32 holding bf16 values), as gpt.py: v +=
 * gate_scale * sigmoid(ve_gate . cat(h[:gate_width / 2], ve[:gate_width / 2])) * value_embedding[token] + MUDD's value
 * stream in fp32 (modded record #82's gate; ve the token's whole value-embedding row), one bf16 rounding (a thread per
 * dim); q and k by qk_prologue.cu's forward (threads 0.. QKP_LANES - 1 the query row, the next QKP_LANES the key row).
 * Key offset (a.shift): from the second position on, the key's stationary dims are the previous position's, normed and
 * scaled as its own key, which the previous step (or the prefill) left in the slot of this position's parity; this
 * position's go to the other slot (split 0 writes them). Ends with the block synchronized. */
static __device__ void attn_prologue(const Attn &a, int b, int head, int p, float *q, float *k, float *v) {
    const int tid = threadIdx.x;
    const bf16 *qkv = a.qkv + (size_t)b * 3 * a.C, *h = a.h + (size_t)b * a.C;
    float t = __bfloat162float(qkv[2 * a.C + head * HEAD_DIM + tid]);
    if (a.ve) {
        const bf16 *ve = a.ve + (size_t)a.seq[(size_t)b * a.seq_slot + p] * a.C;
        const float *w = a.ve_gate + head * a.gate_width;
        const int half = a.gate_width / 2;
        float d = 0.f;
        for (int i = 0; i < half; i++) d = __fmaf_rn(__bfloat162float(h[i]), w[i], d);
        for (int i = 0; i < half; i++) d = __fmaf_rn(__bfloat162float(ve[i]), w[half + i], d);
        const float g = __fmul_rn(a.gate_scale, sigmoidf(d));
        t = __fadd_rn(t, __fmul_rn(g, __bfloat162float(ve[head * HEAD_DIM + tid])));
    }
    if (a.v_add) t = __fadd_rn(t, a.v_add[(size_t)b * a.C + head * HEAD_DIM + tid]);
    v[tid] = bfr(t);
    if (tid < 2 * QKP_LANES) {
        constexpr int rot_lanes = ROTARY_DIM / (2 * VEC);
        const int lane = tid % QKP_LANES;
        const bool isq = tid < QKP_LANES, rot = lane < rot_lanes;
        const int o = rot ? lane * VEC : ROTARY_DIM + (lane - rot_lanes) * VEC;
        const int pair = rot ? ROTARY_DIM / 2 : (HEAD_DIM - ROTARY_DIM) / 2;
        const size_t tab = (size_t)p * (ROTARY_DIM / 2) + lane * VEC;
        float x1[VEC], x2[VEC], c[VEC], s[VEC];
        const float m = rsqrtf(qkp_row_sum(qkp_rotated(qkv + (isq ? 0 : a.C) + head * HEAD_DIM + o, rot, pair,
                                                       a.cos + tab, a.sin + tab, x1, x2, c, s), qkp_row_mask()) /
                               HEAD_DIM + FLT_EPSILON) * a.qk_scale;
        for (int i = 0; i < VEC; i++) { x1[i] = bfr(x1[i] * m); x2[i] = bfr(x2[i] * m); }
        if (!isq && a.shift && !rot) {
            bf16 *shift = a.shift + (size_t)b * a.shift_slot;
            const size_t at = (size_t)head * HEAD_DIM + o;
            if (blockIdx.y == 0) {
                bf16 *next = shift + (size_t)((p + 1) & 1) * a.C + at;
                store8(next, x1); store8(next + pair, x2);
            }
            if (p > 0) {
                const bf16 *prev = shift + (size_t)(p & 1) * a.C + at;
                load8(prev, x1); load8(prev + pair, x2);
            }
        }
        float *y = isq ? q : k;
        for (int i = 0; i < VEC; i++) { y[o + i] = x1[i]; y[o + pair + i] = x2[i]; }
    }
    __syncthreads();
}

/* A lane's LANE_DIMS dims of a cached bf16 key or value row, as fp32: one 8-byte streaming load. */
static_assert(LANE_DIMS == 4, "a lane reads its dims of a cached row as one uint2, of a partial as one float4");
static __device__ __forceinline__ void load_dims(const bf16 *row, float *x) {
    const uint2 u = __ldcs((const uint2 *)row);
    x[0] = __uint_as_float(u.x << 16); x[1] = __uint_as_float(u.x & 0xffff0000u);
    x[2] = __uint_as_float(u.y << 16); x[3] = __uint_as_float(u.y & 0xffff0000u);
}

/* A key's score q . k / sqrt(HEAD_DIM), to every lane: each lane's LANE_DIMS products in order, then warp_sum. */
static __device__ __forceinline__ float key_score(const float *q, const float *k) {
    float d = __fmul_rn(q[0], k[0]);
    for (int e = 1; e < LANE_DIMS; e++) d = __fmaf_rn(q[e], k[e], d);
    return __fmul_rn(warp_sum(d), rsqrtf((float)HEAD_DIM));
}

/* exp(m - top), 0 for an empty state (m = -inf, which top = -inf would turn into NaN) */
static __device__ __forceinline__ float rescale(float m, float top) { return m == -INFINITY ? 0.f : expf(m - top); }

__global__ void __launch_bounds__(HEAD_DIM) k_attn(Attn a) {
    extern __shared__ float weight[];  /* [splits]: the last block's weight of each split's partial */
    __shared__ float q[HEAD_DIM], k[HEAD_DIM], v[HEAD_DIM];
    __shared__ float wm[ATTN_WARPS], wl[ATTN_WARPS], wacc[ATTN_WARPS][HEAD_DIM], ys[HEAD_DIM], total, current;
    __shared__ bool last;
    const int head = blockIdx.x, split = blockIdx.y, b = blockIdx.z, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    if (!live(*a.active, b)) return;
    const int p = a.pos[b];
    /* this split's run of cached keys, its warps taking them in turn; each warp's first key is requested before the
     * prologue, so the cache streams while the prologue runs */
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
    float *part = a.part + base * HEAD_DIM, *stat = a.stat + base * 2;
    part[(size_t)split * HEAD_DIM + tid] = bacc;
    if (tid == 0) stat[2 * split] = bm, stat[2 * split + 1] = bl;
    __threadfence();  /* the partial is visible to every block before this block counts as done */
    __syncthreads();
    unsigned *done = a.done + (size_t)b * a.H + head;
    if (tid == 0) last = atomicAdd(done, 1u) == (unsigned)splits - 1;
    __syncthreads();
    if (!last) return;

    /* the head's last block: the partials (read from L2, written by other SMs) in split order, then the current key */
    if (tid == 0) *done = 0;
    if (warp == 0) {
        const float cur = key_score(q + lane * LANE_DIMS, k + lane * LANE_DIMS);
        float top = cur;
        for (int s = lane; s < splits; s += 32) top = fmaxf(top, __ldcg(stat + 2 * s));
        top = warp_max(top);
        float sum = 0.f;
        for (int s = lane; s < splits; s += 32) {
            const float c = rescale(__ldcg(stat + 2 * s), top);
            weight[s] = c;
            sum = __fmaf_rn(c, __ldcg(stat + 2 * s + 1), sum);
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
    y = bfr(__fdiv_rn(y, total));  /* the attention's bf16 output */
    /* XSA, xsa.cu's op at s = 1: p = y - (sum(y v) / max(sum(v v), 1e-8)) v (the clamp: modded-nanogpt track_1_short
     * attention.py:224-228), the sums in xsa.cu's order (the row's XSA_LANES lanes take XSA_DIMS consecutive dims each,
     * fmaf in order, then xor-shuffles XSA_LANES / 2, ..., 1), times the output gate g = 1 / (1 + exp(-z)), z = h .
     * out_gate[head] in fp32 (warp 1 meanwhile); one bf16 rounding */
    ys[tid] = y;
    __syncthreads();
    if (warp == 0) {
        const int o = lane % XSA_LANES * XSA_DIMS;  /* every XSA_LANES-lane group computes the row's sums */
        float d = 0.f, n = 0.f;
        for (int i = 0; i < XSA_DIMS; i++) d = fmaf(ys[o + i], v[o + i], d), n = fmaf(v[o + i], v[o + i], n);
        for (int s = XSA_LANES / 2; s > 0; s >>= 1) d += __shfl_xor_sync(0xffffffffu, d, s), n += __shfl_xor_sync(0xffffffffu, n, s);
        if (lane == 0) wm[0] = __fdiv_rn(d, fmaxf(n, 1e-8f));
    } else if (warp == 1) {
        const float z = warp_dot(a.h + (size_t)b * a.C, a.out_gate + (size_t)head * a.C, a.C);
        if (lane == 0) wl[0] = __fdiv_rn(1.f, 1.f + expf(-z));
    }
    __syncthreads();
    const float xo = fmaf(-wm[0], v[tid], y);
    a.out[(size_t)b * a.C + head * HEAD_DIM + tid] = __float2bfloat16_rn(__fmul_rn(wl[0], xo));
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* Engram (nanochat/model/engram.py, DeepSeek Engram Eq. 3-5): the residual mix h = resid_lambda x + x0_lambda x0 (as the GEMV
 * prologue, rounded to bf16), then with k = kv[:C], v = kv[C:] (the W_kv GEMV of the layer's memory rows) and the
 * fp32 norm weights w_h, w_k, w_v and conv weights w (C x ENGRAM_TAPS):
 *     Eq. 4: alpha = sigmoid((rms_norm(h) w_h) . (rms_norm(k) w_k) / sqrt(C)),   Vt = bf16(alpha v)
 *     Eq. 5: u = bf16(rms_norm(Vt) w_v),   c_i = sum over taps j of w[i][j] u[p - dil (K - 1 - j)][i] (taps before
 *            position 0 skipped),   hm = h + (silu(c) + Vt)
 * The u rows of the layer's last dil (K - 1) positions stay in a ring of R = dil (K - 1) + 1 slots (position q in slot
 * q % R; the prefill fills it, dec_set_prefilled). nanochat/ops/engram.cu's gate and conv kernels for one token, expression for
 * expression, so given equal h, kv and ring the fused h is PyTorch's bit for bit: the three sums are its gate warp's
 * (lane l the 8-dim vectors l, l + 32, ... in turn, fma sums, xor-butterfly lane sums), taken by warp 0 from shared
 * memory; everything per channel (the mix, the products, Vt, u, the taps in order j = 0..K-1, the output, one bf16
 * rounding per stored value) runs a thread per vector, which loads all its inputs at once at the start. A block per
 * slot. */
#define ENGRAM_TAPS 4         /* nanochat.model.engram.ENGRAM_CONV_KERNEL: a channel's conv weights are one float4 */

static __device__ __forceinline__ float engram_lane_sum(float s) {
    for (int o = 16; o > 0; o >>= 1) s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, o));
    return s;
}

typedef struct {
    const bf16 *x, *x0, *kv;                        /* [B][C], [B][C], [B][2C] */
    float l1, l2;
    const float *norm_h, *norm_k, *norm_v, *conv;  /* conv: [C][ENGRAM_TAPS] fp32 */
    bf16 *ring;                                     /* slot 0's [R][C] (slot b's at + b ring_slot): the u rows of the
                                                     * layer's positions q, slot q % R */
    size_t ring_slot;
    const int32_t *pos;
    const unsigned long long *active;
    bf16 *hm;                                       /* [B][C] */
    int C, dilation, R;
} Engram;

/* A block per slot of a thread per vector (a whole number of warps); dynamic shared memory: 4 C floats. */
__global__ void k_engram(Engram e) {
    extern __shared__ float smem[];
    float *sh = smem, *sk = sh + e.C, *shw = sk + e.C, *skw = shw + e.C;  /* h, k (then Vt in sh); h w_h, k w_k */
    __shared__ float s_alpha, s_rv;
    const int bs = blockIdx.x;
    if (!live(*e.active, bs)) return;
    const int c = threadIdx.x, C = e.C, vecs = C / VEC, p = e.pos[bs], o = c * VEC;
    const bf16 *x = e.x + (size_t)bs * C, *x0 = e.x0 + (size_t)bs * C, *kv = e.kv + (size_t)bs * 2 * C;
    bf16 *ring = e.ring + (size_t)bs * e.ring_slot, *hm = e.hm + (size_t)bs * C;
    const bool mine = c < vecs;
    const float fc = (float)C;
    float h[VEC], v[VEC], wv[VEC], t[ENGRAM_TAPS - 1][VEC];
    float4 w[VEC];
    if (mine) {
        float a[VEC], b[VEC], wh[VEC], wk[VEC];
        load8(x + o, a);
        load8(x0 + o, b);
        for (int i = 0; i < VEC; i++) h[i] = bfr(bfr(e.l1 * a[i]) + bfr(e.l2 * b[i]));
        load8(kv + o, b);
        load8(kv + C + o, v);
        load8(e.norm_h + o, wh);
        load8(e.norm_k + o, wk);
        load8(e.norm_v + o, wv);
        for (int i = 0; i < VEC; i++) w[i] = reinterpret_cast<const float4 *>(e.conv)[o + i];
#pragma unroll
        for (int k = 0; k < ENGRAM_TAPS - 1; k++) {
            const int lag = e.dilation * (ENGRAM_TAPS - 1 - k);
            if (p >= lag) load8(ring + (size_t)((p - lag) % e.R) * C + o, t[k]);
        }
        for (int i = 0; i < VEC; i++) {
            sh[o + i] = h[i]; sk[o + i] = b[i];
            shw[o + i] = __fmul_rn(h[i], wh[i]); skw[o + i] = __fmul_rn(b[i], wk[i]);
        }
    }
    __syncthreads();
    if (c < 32) {
        float ssh = 0.f, ssk = 0.f, dot = 0.f;
        for (int q = c * VEC; q < C; q += 32 * VEC)
            for (int i = q; i < q + VEC; i++) {
                ssh = __fmaf_rn(sh[i], sh[i], ssh);
                ssk = __fmaf_rn(sk[i], sk[i], ssk);
                dot = __fmaf_rn(shw[i], skw[i], dot);
            }
        ssh = engram_lane_sum(ssh); ssk = engram_lane_sum(ssk); dot = engram_lane_sum(dot);
        const float rh = rsqrtf(__fadd_rn(__fdiv_rn(ssh, fc), FLT_EPSILON));
        const float rk = rsqrtf(__fadd_rn(__fdiv_rn(ssk, fc), FLT_EPSILON));
        const float s = __fmul_rn(__fmul_rn(__fmul_rn(dot, rh), rk), rsqrtf(fc));
        if (c == 0) s_alpha = __fdiv_rn(1.f, __fadd_rn(1.f, expf(-s)));
    }
    __syncthreads();
    const float alpha = s_alpha;
    if (mine)
        for (int i = 0; i < VEC; i++) sh[o + i] = v[i] = bfr(__fmul_rn(alpha, v[i]));  /* v becomes Vt */
    __syncthreads();
    if (c < 32) {
        float ssv = 0.f;
        for (int q = c * VEC; q < C; q += 32 * VEC)
            for (int i = q; i < q + VEC; i++) ssv = __fmaf_rn(sh[i], sh[i], ssv);
        ssv = engram_lane_sum(ssv);
        if (c == 0) s_rv = rsqrtf(__fadd_rn(__fdiv_rn(ssv, fc), FLT_EPSILON));
    }
    __syncthreads();
    if (!mine) return;
    const float rv = s_rv;
    float u[VEC], acc[VEC] = {};
    for (int i = 0; i < VEC; i++) u[i] = bfr(__fmul_rn(__fmul_rn(v[i], rv), wv[i]));
    store8(ring + (size_t)(p % e.R) * C + o, u);
#pragma unroll
    for (int k = 0; k < ENGRAM_TAPS; k++) {
        const int lag = e.dilation * (ENGRAM_TAPS - 1 - k);
        if (p < lag) continue;
        const float *xk = k < ENGRAM_TAPS - 1 ? t[k] : u;
        for (int i = 0; i < VEC; i++) {
            const float wk = k == 0 ? w[i].x : k == 1 ? w[i].y : k == 2 ? w[i].z : w[i].w;
            acc[i] = __fmaf_rn(wk, xk[i], acc[i]);
        }
    }
    for (int i = 0; i < VEC; i++)
        h[i] = __fadd_rn(h[i], __fadd_rn(__fdiv_rn(acc[i], __fadd_rn(1.f, expf(-acc[i]))), v[i]));
    store8(hm + o, h);
}

/* Shared memory of k_engram at width C. */
static size_t engram_smem(int C) { return (size_t)4 * C * sizeof(float); }

/* k_engram, a whole number of warps covering the vectors (dec_create checks the block and its memory fit). */
static int engram_threads(int C) { return (C / VEC + 31) / 32 * 32; }

/* ---------------------------------------------------------------------------------------------------------------- */
/* MUDD (dynamic dense connections, Xiao et al. arXiv 2502.12170; gpt.py's mudd_in / mudd_out): at one site, from the
 * site's state X (bf16 values), the coefficient MLP c = (gelu(n(X) . bf16(W1)^T) . W2^T + b) * scale (fp32 from the
 * fp32-accumulated first layer on, exact-erf gelu; scale: mudd.MUDD_SCALE), then per channel the mixes over the sources
 * src_s, s ascending:
 *     out = bf16(X + sum_s cR_s src_s),   vmix = sum_s cV_s src_s (fp32, not rounded; site 1 only, k_attn adds it to
 *     the last layer's value inside the value embedding's rounding)
 * Source s is coded src[s]: -1 x0, l in [0, L) the output of block l, L + l the value-embedding row of the step's token
 * of layer l (gpt.py's mudd_sources: "x0", ("h", l), ("ve", l)). X: the residual mix bf16(bf16(l1 x) + bf16(l2 x0)) as
 * the GEMV prologue (mix_x0 set), else x as is. A block per slot; dynamic shared memory: 2 C + 1 + M + K floats (M: the
 * coefficient MLP's hidden width, mudd.MUDD_DIM). */

typedef struct {
    const bf16 *x, *mix_x0;   /* [B][C]: X = x, or the residual mix of x and mix_x0 (if set) with lambdas l1, l2 */
    float l1, l2;
    const bf16 *w1;           /* [M][C] bf16 */
    const float *w2, *b;      /* [K][M], [K] fp32 */
    float scale;              /* the coefficients' output scale */
    const int *src;           /* [S] source codes (device) */
    const bf16 *x0, *hs;      /* [B][C] x0; [L][B][C] the block outputs */
    const bf16 *const *ve;    /* [L] per layer its value-embedding table [vocab][C] or NULL (device array of pointers) */
    const int32_t *seq, *pos;
    int seq_slot;
    const unsigned long long *active;
    bf16 *out;                /* [B][C] */
    float *vmix;              /* site 1: [B][C], c[0, S) is the value stream, c[S, 2S) the residual's; NULL: c is the residual's */
    int C, S, K, L, B, M;
} Mudd;

__global__ void k_mudd(Mudd m) {
    extern __shared__ float sm[];
    const int bs = blockIdx.x;
    if (!live(*m.active, bs)) return;
    const int C = m.C, S = m.S, tid = threadIdx.x, nt = blockDim.x, lane = tid & 31, warp = tid >> 5;
    const size_t o = (size_t)bs * C;
    float *X = sm, *N = X + C, *slot = N + C, *act = slot + 1, *c = act + m.M;
    for (int i = tid; i < C; i += nt) {
        float v = __bfloat162float(m.x[o + i]);
        if (m.mix_x0) v = bfr(bfr(m.l1 * v) + bfr(m.l2 * __bfloat162float(m.mix_x0[o + i])));
        X[i] = v;
    }
    __syncthreads();
    if (tid < 32) {
        float s = 0.f;
        for (int i = tid; i < C; i += 32) s = __fmaf_rn(X[i], X[i], s);
        s = warp_sum(s);
        if (tid == 0) *slot = rsqrtf(s / (float)C + FLT_EPSILON);
    }
    __syncthreads();
    for (int i = tid; i < C; i += nt) N[i] = bfr(X[i] * *slot);
    __syncthreads();
    for (int j = warp; j < m.M; j += nt >> 5) {  /* first layer: a warp per hidden unit, then gelu */
        const uint4 *w = reinterpret_cast<const uint4 *>(m.w1) + (size_t)j * (C / VEC);
        float d = 0.f;
        for (int q = lane; q < C / VEC; q += 32) {
            float wf[VEC];
            unpack(w[q], wf);
            d = __fadd_rn(d, dot8(wf, N + q * VEC));
        }
        const float z = warp_sum(d);
        if (lane == 0) act[j] = 0.5f * z * (1.f + erff(z * (float)M_SQRT1_2));
    }
    __syncthreads();
    for (int k = tid; k < m.K; k += nt) {
        float acc = 0.f;
        for (int j = 0; j < m.M; j++) acc = __fmaf_rn(act[j], m.w2[(size_t)k * m.M + j], acc);
        c[k] = __fmul_rn(__fadd_rn(acc, m.b[k]), m.scale);
    }
    __syncthreads();
    const float *cR = m.vmix ? c + S : c;
    const size_t tok = (size_t)m.seq[(size_t)bs * m.seq_slot + m.pos[bs]];
    for (int i = tid; i < C; i += nt) {
        float acc = X[i], vacc = 0.f;
        for (int s = 0; s < S; s++) {  /* mudd.mix: each term rounded, then added (no contraction) */
            const int code = m.src[s];
            const bf16 *src = code < 0 ? m.x0 + o : code < m.L ? m.hs + ((size_t)code * m.B + bs) * C
                                                                : m.ve[code - m.L] + tok * C;
            const float v = __bfloat162float(src[i]);
            acc = __fadd_rn(acc, __fmul_rn(cR[s], v));
            if (m.vmix) vacc = s == 0 ? __fmul_rn(c[s], v) : __fadd_rn(vacc, __fmul_rn(c[s], v));
        }
        m.out[o + i] = __float2bfloat16_rn(acc);
        if (m.vmix) m.vmix[o + i] = vacc;
    }
}

static size_t mudd_smem(int C, int M, int K) { return (size_t)(2 * C + 1 + M + K) * sizeof(float); }

/* ---------------------------------------------------------------------------------------------------------------- */
/* Each active slot's Engram window: its tokens at positions pos - n + 1 .. pos (the step's input last), positions
 * before 0 read position 0, the sequence's BOS (as BOS padding: the n-grams' history ends at the most recent BOS). A
 * thread per slot. */
__global__ void k_window(const int32_t *seq, int seq_slot, const int32_t *pos, const unsigned long long *active,
                         int64_t *window, int n, int B) {
    const int b = threadIdx.x;
    if (b >= B || !live(*active, b)) return;
    const int32_t *s = seq + (size_t)b * seq_slot;
    for (int j = 0; j < n; j++) {
        const int q = pos[b] - (n - 1) + j;
        window[(size_t)b * n + j] = s[q > 0 ? q : 0];
    }
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* Each active slot's argmax becomes its next input, unless its prompt still supplies it; its position advances, and the
 * token at it is reported in tokens[b]. A thread per slot. */
__global__ void k_commit(int32_t *seq, int seq_slot, int32_t *pos, const int32_t *prompt_len, const unsigned long long *active,
                         unsigned long long *argmax, int32_t *tokens, int B) {
    const int b = threadIdx.x;
    if (b >= B || !live(*active, b)) return;
    int32_t *s = seq + (size_t)b * seq_slot;
    const int p = pos[b];
    if (p + 1 >= prompt_len[b]) s[p + 1] = (int32_t)(0xffffffffu - (uint32_t)(argmax[b] & 0xffffffffu));
    pos[b] = p + 1;
    argmax[b] = 0ull;
    tokens[b] = s[p + 1];
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* Host side: one decoder per model and batch size, its step captured as two CUDA graphs (front, back). */

typedef struct {
    const bf16 *qkv, *o, *fc, *proj, *ve;  /* ve: NULL in a layer without value embeddings */
    const float *ve_gate;                 /* the value-embedding gate [H][gate_width], fp32; NULL without */
    const bf16 *w_kv;                    /* Engram's W_kv [2C][engram_cols] (W_k rows, then W_v); NULL: not an Engram layer */
    const float *norm_h, *norm_k, *norm_v;  /* Engram's RMSNorm weights of h, k and Vt [C], fp32 as the model's */
    const float *conv;                   /* Engram's depthwise conv weights [C][ENGRAM_TAPS], fp32 */
    const bf16 *out_gate;                /* the attention output gate [H][C] */
    int engram;                          /* the Engram layer's index j among the Engram layers (its rows: slice j) */
    float resid_lambda, x0_lambda, qk_scale;
    const float *cos, *sin;              /* the layer's fp32 rotary tables [max_len][ROTARY_DIM / 2] */
    int window, key_offset;
} Layer;

typedef struct {
    const bf16 *w1;          /* [M][C] */
    const float *w2, *b;     /* [K][M], [K] fp32 */
    int *src;                /* [S] the sources' codes (k_mudd), device */
    int S, K, M;
} MuddSite;

typedef struct {
    int B, L, H, C, F, vocab, max_len, smear_width, gate_width, engram_cols, conv_dilation,
        engram_layers, first_engram, ring, sms, optin, attn_wave, attn_run;  /* first_engram: the first Engram layer,
                                                                               * where the back graph starts; ring:
                                                                               * conv_dilation (ENGRAM_TAPS - 1) + 1
                                                                               * slots; optin: the opt-in shared memory
                                                                               * per block */
    float smear_lambda, gate_scale, mudd_scale;
    MuddSite site[2];         /* site[s]: MUDD site s + 1 (the last block's input; before the final norm) */
    const bf16 *wte, *lm_head;
    const float *smear_gate;  /* [smear_width] fp32 */
    const float *cos, *sin;   /* fp32 rotary tables [L][max_len][ROTARY_DIM / 2] */
    Layer *layer;
    int seq_slot;             /* max_len + 1: a slot's tokens (the step at max_len - 1 writes max_len) */
    int32_t *seq, *pos, *prompt_len, *tokens;  /* [B][seq_slot], [B], [B], [B] */
    unsigned long long *active;  /* device: the active slots' bit mask */
    int32_t *host_tokens;     /* pinned host [B]: the last step's tokens */
    unsigned long long *argmax;  /* [B] */
    bf16 *hs, *x0, *xm, *xf, *h, *qkv_out, *att, *xo, *u, *prev;  /* [B][...]; hs: [L][B][C] the blocks' outputs */
    bf16 *kc, *vc;            /* [B][L][max_len][C] */
    float *vmix;              /* [B][C] MUDD site 1's value stream */
    const bf16 **ve_tables;   /* device: [L] per layer its value-embedding table or NULL (MUDD's sources) */
    const bf16 *engram_x;        /* the caller's [engram_layers][B][engram_cols]: the rows of the slots' input tokens */
    int64_t *engram_window;      /* the caller's [B][window]: the slots' last window tokens (k_window) */
    int window;                  /* the longest n-gram */
    bf16 *engram_kv;             /* [B][2C] a layer's k and v */
    bf16 *engram_ring;           /* [B][engram_layers][ring][C] (k_engram) */
    bf16 *shift;                 /* [B][L][2][C]: k_attn's key-offset slots (used by key-offset layers) */
    float *attn_part;            /* k_attn's partials, reused by every layer: [B][H][attn_wave][HEAD_DIM] sums, then [..][2] */
    unsigned *attn_done;         /* [B][H]: k_attn's counters, zero between launches */
    cudaStream_t stream;         /* the caller's */
    cudaGraph_t graph[2];        /* the step's front and back */
    cudaGraphExec_t exec[2];
} Decoder;

static Gemv gemv(const Decoder *d, const bf16 *w, int rows, int cols, int pro, const bf16 *in, int epi, bf16 *out) {
    Gemv g = {};
    g.w = w; g.rows = rows; g.cols = cols; g.pro = pro; g.in = in; g.epi = epi; g.out = out;
    g.active = d->active; g.B = d->B;
    return g;
}

/* A warp per 16-row tile, in blocks of the size that maximizes k_gemv's occupancy, or of fewer warps when that grid
 * would leave SMs without a block; at most as many blocks as the SMs hold at once (the warps loop over the tiles). */
static void launch_gemv(Decoder *d, Gemv g) {
    g.chunk = gemv_chunk(g.B, g.cols, d->optin);
    const size_t smem = gemv_smem(g.B, g.chunk);
    const int tiles = (g.rows + 15) / 16;
    int grid, block, resident;
    cudaOccupancyMaxPotentialBlockSize(&grid, &block, k_gemv, smem);
    const int spread = (tiles + d->sms - 1) / d->sms, warps = spread < block / 32 ? spread : block / 32;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, k_gemv, 32 * warps, smem);
    const int needed = (tiles + warps - 1) / warps, wave = d->sms * resident;
    k_gemv<<<needed < wave ? needed : wave, 32 * warps, smem, d->stream>>>(g);
}

/* k_mudd: a block per slot, a thread per channel up to the block limit (the kernel loops over the rest). */
static void launch_mudd(Decoder *d, const Mudd &m) {
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, k_mudd);
    const int threads = d->C < fa.maxThreadsPerBlock ? d->C : fa.maxThreadsPerBlock;
    k_mudd<<<d->B, threads, mudd_smem(m.C, m.M, m.K), d->stream>>>(m);
}

/* MUDD site s + 1 (s 0: the last block's input, from x as is or its residual mix with x0; s 1: before the final norm)
 * into out, with the value stream into d->vmix at site 1. */
static Mudd mudd(const Decoder *d, int s, const bf16 *x, const bf16 *mix_x0, float l1, float l2, bf16 *out) {
    const MuddSite &w = d->site[s];
    return {x, mix_x0, l1, l2, w.w1, w.w2, w.b, d->mudd_scale, w.src, d->x0, d->hs, d->ve_tables, d->seq, d->pos, d->seq_slot,
            d->active, out, s == 0 ? d->vmix : NULL, d->C, w.S, w.K, d->L, d->B, w.M};
}

/* Part `back` of one decoding step of every active slot, enqueued on d->stream (captured once into graph[back]): the
 * front (0) embeds and runs the blocks before the first Engram layer, the back (1) the rest, which reads the rows. */
static void enqueue_step(Decoder *d, int back) {
    const int C = d->C, F = d->F, L = d->L, B = d->B;
    if (!back) {
        cudaFuncAttributes fa;
        cudaFuncGetAttributes(&fa, k_embed);
        k_embed<<<B, C < fa.maxThreadsPerBlock ? C : fa.maxThreadsPerBlock, (size_t)(C + 1) * sizeof(float), d->stream>>>(
            d->seq, d->seq_slot, d->pos, d->active, d->wte, d->smear_gate, d->smear_width, d->smear_lambda, d->prev, d->x0, C);
        k_window<<<1, MAX_BATCH, 0, d->stream>>>(d->seq, d->seq_slot, d->pos, d->active, d->engram_window, d->window, B);
    }
    for (int l = back ? d->first_engram : 0; l < (back ? L : d->first_engram); l++) {
        const Layer *ly = &d->layer[l];
        /* the layer's input: x0 before the first block, else the previous block's output; [B][C] at stride L C */
        const bf16 *xin = l == 0 ? d->x0 : d->hs + (size_t)(l - 1) * B * C;
        bf16 *xout = d->hs + (size_t)l * B * C;
        const bool site = l == L - 1;  /* MUDD site 1: xm = the mixed input plus its residual stream */
        Gemv g;
        if (ly->w_kv) {  /* Engram, as gpt.py: xm = the gate's fusion of the mixed residual, then the block */
            /* W_kv GEMV of the slots' rows of this layer (the rows are [layers][B][cols]) */
            Gemv kv = gemv(d, ly->w_kv, 2 * C, d->engram_cols, PRO_COPY, d->engram_x + (size_t)ly->engram * B * d->engram_cols,
                           EPI_STORE, d->engram_kv);
            launch_gemv(d, kv);
            const Engram e = {xin, d->x0, d->engram_kv, ly->resid_lambda, ly->x0_lambda, ly->norm_h, ly->norm_k,
                              ly->norm_v, ly->conv, d->engram_ring + (size_t)ly->engram * d->ring * C,
                              (size_t)d->engram_layers * d->ring * C, d->pos, d->active, d->xm, C, d->conv_dilation,
                              d->ring};
            k_engram<<<B, engram_threads(C), engram_smem(C), d->stream>>>(e);
            if (site) launch_mudd(d, mudd(d, 0, d->xm, NULL, 0.f, 0.f, d->xm));
            g = gemv(d, ly->qkv, 3 * C, C, PRO_NORM, d->xm, EPI_STORE, d->qkv_out);
            g.pro_out2 = d->h;
        } else if (site) {
            launch_mudd(d, mudd(d, 0, xin, d->x0, ly->resid_lambda, ly->x0_lambda, d->xm));
            g = gemv(d, ly->qkv, 3 * C, C, PRO_NORM, d->xm, EPI_STORE, d->qkv_out);
            g.pro_out2 = d->h;
        } else {
            g = gemv(d, ly->qkv, 3 * C, C, PRO_RESID_NORM, xin, EPI_STORE, d->qkv_out);
            g.in2 = d->x0; g.l1 = ly->resid_lambda; g.l2 = ly->x0_lambda; g.pro_out = d->xm; g.pro_out2 = d->h;
        }
        launch_gemv(d, g);
        /* attention splits per head and slot: the position's runs of attn_run keys, at most the window's */
        const int keys = ly->window < d->max_len - 1 ? ly->window : d->max_len - 1;
        const int splits = (keys + d->attn_run - 1) / d->attn_run;
        const Attn a = {d->qkv_out, d->h, ly->ve, site ? d->vmix : NULL, ly->ve_gate, ly->cos, ly->sin,
                        d->kc + (size_t)l * d->max_len * C, d->vc + (size_t)l * d->max_len * C,
                        (size_t)L * d->max_len * C, d->att,
                        ly->key_offset ? d->shift + (size_t)l * 2 * C : NULL, (size_t)L * 2 * C, ly->out_gate,
                        d->seq, d->pos, d->seq_slot, d->active, d->attn_part,
                        d->attn_part + (size_t)B * d->H * d->attn_wave * HEAD_DIM, d->attn_done, d->attn_wave,
                        C, d->H, ly->window, d->attn_run, d->gate_width, d->gate_scale, ly->qk_scale};
        k_attn<<<dim3(d->H, splits, B), HEAD_DIM, splits * sizeof(float), d->stream>>>(a);
        g = gemv(d, ly->o, C, C, PRO_COPY, d->att, EPI_RESID, d->xo);
        g.res = d->xm;
        launch_gemv(d, g);
        launch_gemv(d, gemv(d, ly->fc, F, C, PRO_NORM, d->xo, EPI_SWIGLU, d->u));  /* F outputs from the 2F rows of fc */
        g = gemv(d, ly->proj, C, F, PRO_COPY, d->u, EPI_RESID, xout);
        g.res = d->xo;
        launch_gemv(d, g);
    }
    if (!back) return;
    launch_mudd(d, mudd(d, 1, d->hs + (size_t)(L - 1) * B * C, NULL, 0.f, 0.f, d->xf));  /* MUDD site 2 */
    Gemv g = gemv(d, d->lm_head, d->vocab, C, PRO_NORM, d->xf, EPI_ARGMAX, NULL);
    g.argmax = d->argmax;
    launch_gemv(d, g);
    k_commit<<<1, MAX_BATCH, 0, d->stream>>>(d->seq, d->seq_slot, d->pos, d->prompt_len, d->active, d->argmax, d->tokens, B);
    cudaMemcpyAsync(d->host_tokens, d->tokens, (size_t)B * sizeof(int32_t), cudaMemcpyDeviceToHost, d->stream);
}

extern "C" void dec_destroy(void *handle) {
    Decoder *d = (Decoder *)handle;
    for (int i = 0; i < 2; i++) {
        cudaGraphExecDestroy(d->exec[i]);
        cudaGraphDestroy(d->graph[i]);
    }
    void *bufs[] = {d->seq, d->pos, d->prompt_len, d->tokens, d->active, d->argmax, d->hs, d->x0, d->xm,
                    d->xf, d->h, d->qkv_out, d->att, d->xo, d->u, d->prev, d->kc, d->vc, d->vmix, d->ve_tables,
                    d->engram_kv, d->engram_ring, d->shift, d->attn_part, d->attn_done,
                    d->site[0].src, d->site[1].src};
    for (void *b : bufs) cudaFree(b);
    cudaFreeHost(d->host_tokens);
    free(d->layer);
    free(d);
}

/* The most slots a decoder can run (MAX_BATCH). */
extern "C" int dec_max_batch() { return MAX_BATCH; }

/*
 * cfg: L, H, C, MLP hidden width F (fc: [2F][C], gate rows then up rows; proj: [C][F]), vocab (the real vocabulary:
 *      the argmax covers lm_head's first vocab rows, never its padding rows, gpt.padded_vocab), max_len, B (slots),
 *      smear gate width, value-embedding gate width, Engram row width per token and Engram layer (slots x row dim),
 *      Engram conv dilation, longest n-gram (the window), then per MUDD site (1: the last block's input, 2: before the
 *      final norm) its source count S, its coefficient MLP's hidden width M and S source codes (-1 x0, l < L the output
 *      of block l, L + l the value embedding of layer l).
 * ptrs: wte, smear_gate, cos, sin (fp32 rotary tables [L][max_len][ROTARY_DIM / 2], layer l's at l), lm_head, Engram
 *       rows (device memory,
 *       [Engram layers][B][Engram row width] bf16, which the caller fills with the rows of each step's input tokens on
 *       the stream between the step's front and back), Engram window (device memory, [B][window] int64, k_window);
 *       per layer: qkv, o, fc, proj, value-embedding table (0 if none),
 *       value-embedding gate (0 if none), Engram W_kv, norm_h, norm_k and norm_v weights and conv weights (0 if not an
 *       Engram layer; the Engram layers in increasing order take the row slices 0, 1, ...), attention output gate
 *       ([H][C]); then per MUDD site its W1 [M][C] bf16, W2 [K][M] and b [K] fp32 (K = 2S at site 1: value stream,
 *       then residual stream; K = S at site 2).
 * scal: smear lambda, value-embedding gate scale, MUDD coefficient scale; per layer: resid lambda, x0 lambda, QK scale.
 * layer: per layer, the keys before each query that it attends (FlashAttention's window_size left) and whether its
 *        keys take the key offset (1) or not (0).
 * stream: the caller's CUDA stream (not the legacy default stream: captured), on which every call here enqueues its
 *         work; the caller owns it and keeps it alive past dec_destroy.
 */
extern "C" void *dec_create(const int *cfg, const uint64_t *ptrs, const float *scal, const int *layer, void *stream) {
    Decoder *d = (Decoder *)calloc(1, sizeof(Decoder));
    d->L = cfg[0]; d->H = cfg[1]; d->C = cfg[2]; d->F = cfg[3]; d->vocab = cfg[4]; d->max_len = cfg[5]; d->B = cfg[6];
    d->smear_width = cfg[7]; d->gate_width = cfg[8]; d->engram_cols = cfg[9]; d->conv_dilation = cfg[10]; d->window = cfg[11];
    d->ring = d->conv_dilation * (ENGRAM_TAPS - 1) + 1;
    d->seq_slot = d->max_len + 1;
    d->wte = (const bf16 *)ptrs[0]; d->smear_gate = (const float *)ptrs[1]; d->cos = (const float *)ptrs[2];
    d->sin = (const float *)ptrs[3]; d->lm_head = (const bf16 *)ptrs[4]; d->engram_x = (const bf16 *)ptrs[5];
    d->engram_window = (int64_t *)ptrs[6]; d->stream = (cudaStream_t)stream;
    d->smear_lambda = scal[0]; d->gate_scale = scal[1]; d->mudd_scale = scal[2];
    const int L = d->L, C = d->C, B = d->B;
    d->layer = (Layer *)calloc(L, sizeof(Layer));
    const bf16 **ve = (const bf16 **)calloc(L, sizeof(bf16 *));  /* per layer its value-embedding table */
    for (int l = 0; l < L; l++) {
        const uint64_t *p = ptrs + 7 + 12 * l;
        d->layer[l] = {(const bf16 *)p[0], (const bf16 *)p[1], (const bf16 *)p[2], (const bf16 *)p[3], (const bf16 *)p[4],
                       (const float *)p[5], (const bf16 *)p[6], (const float *)p[7], (const float *)p[8], (const float *)p[9],
                       (const float *)p[10], (const bf16 *)p[11], d->engram_layers, scal[3 + 3 * l], scal[4 + 3 * l],
                       scal[5 + 3 * l], d->cos + (size_t)l * d->max_len * (ROTARY_DIM / 2),
                       d->sin + (size_t)l * d->max_len * (ROTARY_DIM / 2), layer[2 * l], layer[2 * l + 1]};
        if (p[6] && !d->engram_layers) d->first_engram = l;
        d->engram_layers += p[6] != 0;
        ve[l] = (const bf16 *)p[4];
    }
    const int *codes[2];
    for (int s = 0, at = 12; s < 2; s++) {
        const uint64_t *p = ptrs + 7 + 12 * L + 3 * s;
        const int S = cfg[at], M = cfg[at + 1];
        codes[s] = cfg + at + 2;
        at += 2 + S;
        d->site[s] = {(const bf16 *)p[0], (const float *)p[1], (const float *)p[2], NULL, S, s == 0 ? 2 * S : S, M};
    }
    int device;
    cudaFuncSetAttribute(k_engram, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)engram_smem(C));
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&d->sms, cudaDevAttrMultiProcessorCount, device);
    cudaDeviceGetAttribute(&d->optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    cudaFuncSetAttribute(k_gemv, cudaFuncAttributeMaxDynamicSharedMemorySize, d->optin);
    cudaFuncSetAttribute(k_embed, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)((C + 1) * sizeof(float)));
    /* attention's wave: as many (head, split) blocks as the SMs hold at once, split evenly over the heads. Its shared
     * memory grows by a float per split, so the residency is taken again at the wave found without that array (fewer
     * splits, less memory: the second wave fits). attn_run: the keys of a split, so that the longest attended range
     * (max_len - 1 keys) takes one wave per head and slot; shorter ranges take fewer splits */
    int resident;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, k_attn, HEAD_DIM, 0);
    d->attn_wave = d->sms * resident / d->H;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, k_attn, HEAD_DIM, d->attn_wave * sizeof(float));
    d->attn_wave = d->sms * resident / d->H;
    d->attn_run = (d->max_len - 1 + d->attn_wave - 1) / d->attn_wave;
    cudaMalloc((void **)&d->seq, (size_t)B * d->seq_slot * sizeof(int32_t));
    int32_t **ints[] = {&d->pos, &d->prompt_len, &d->tokens};
    for (int32_t **p : ints) cudaMalloc((void **)p, (size_t)B * sizeof(int32_t));
    cudaMalloc((void **)&d->active, sizeof(unsigned long long));
    cudaMemset(d->active, 0, sizeof(unsigned long long));
    cudaHostAlloc((void **)&d->host_tokens, (size_t)B * sizeof(int32_t), cudaHostAllocDefault);
    cudaMalloc((void **)&d->argmax, (size_t)B * sizeof(unsigned long long));
    cudaMemset(d->argmax, 0, (size_t)B * sizeof(unsigned long long));
    bf16 **bufs[] = {&d->x0, &d->xm, &d->xf, &d->h, &d->att, &d->xo, &d->prev};
    for (bf16 **b : bufs) cudaMalloc((void **)b, (size_t)B * C * sizeof(bf16));
    cudaMalloc((void **)&d->hs, (size_t)L * B * C * sizeof(bf16));
    cudaMalloc((void **)&d->vmix, (size_t)B * C * sizeof(float));
    cudaMalloc((void **)&d->ve_tables, (size_t)L * sizeof(bf16 *));
    cudaMemcpy(d->ve_tables, ve, (size_t)L * sizeof(bf16 *), cudaMemcpyHostToDevice);
    free(ve);
    for (int s = 0; s < 2; s++) {
        cudaMalloc((void **)&d->site[s].src, (size_t)d->site[s].S * sizeof(int));
        cudaMemcpy(d->site[s].src, codes[s], (size_t)d->site[s].S * sizeof(int), cudaMemcpyHostToDevice);
    }
    cudaMalloc((void **)&d->qkv_out, (size_t)B * 3 * C * sizeof(bf16));
    cudaMalloc((void **)&d->u, (size_t)B * d->F * sizeof(bf16));
    cudaMalloc((void **)&d->kc, (size_t)B * L * d->max_len * C * sizeof(bf16));
    cudaMalloc((void **)&d->vc, (size_t)B * L * d->max_len * C * sizeof(bf16));
    cudaMalloc((void **)&d->engram_kv, (size_t)B * 2 * C * sizeof(bf16));
    cudaMalloc((void **)&d->engram_ring, (size_t)B * d->engram_layers * d->ring * C * sizeof(bf16));
    cudaMalloc((void **)&d->shift, (size_t)B * L * 2 * C * sizeof(bf16));
    cudaMalloc((void **)&d->attn_part, (size_t)B * d->H * d->attn_wave * (HEAD_DIM + 2) * sizeof(float));
    cudaMalloc((void **)&d->attn_done, (size_t)B * d->H * sizeof(unsigned));
    cudaMemset(d->attn_done, 0, (size_t)B * d->H * sizeof(unsigned));
    for (int i = 0; i < 2; i++) {
        cudaStreamBeginCapture(d->stream, cudaStreamCaptureModeThreadLocal);
        enqueue_step(d, i);
        cudaStreamEndCapture(d->stream, &d->graph[i]);
        cudaGraphInstantiate(&d->exec[i], d->graph[i], 0);
    }
    return d;
}

/* Start a sequence of n tokens in slot b from PyTorch's batch-1 KVCache of its first n_cached tokens, laid out (layer,
 * position, head, dim) with a layer stride of layer_stride elements; prev: the normalized embedding of position
 * n_cached - 1; ring: the Engram conv rings [Engram layers][ring][C] holding the prefill's u rows of the positions q
 * before n_cached in slot q % ring (the steps read only positions q >= 0); tail: [L][C], per layer the key of position
 * n_cached - 1 as computed before the key offset (its stationary dims are what the step at n_cached shifts into its
 * key; read for key-offset layers only). The slot's steps continue at position n_cached once it is active
 * (dec_set_active). Enqueued after the steps launched so far. */
extern "C" void dec_set_prefilled(void *handle, int b, const int32_t *tokens, int n, int n_cached, const bf16 *k,
                                  const bf16 *v, long long layer_stride, const bf16 *prev, const bf16 *ring,
                                  const bf16 *tail) {
    Decoder *d = (Decoder *)handle;
    const size_t row = (size_t)d->C * sizeof(bf16), dst_pitch = (size_t)d->max_len * row,
                 src_pitch = (size_t)layer_stride * sizeof(bf16), slot = (size_t)d->L * d->max_len * d->C;
    cudaMemcpyAsync(d->seq + (size_t)b * d->seq_slot, tokens, (size_t)n * sizeof(int32_t), cudaMemcpyHostToDevice,
                    d->stream);
    cudaMemcpyAsync(d->prompt_len + b, &n, sizeof(int32_t), cudaMemcpyHostToDevice, d->stream);
    cudaMemcpyAsync(d->pos + b, &n_cached, sizeof(int32_t), cudaMemcpyHostToDevice, d->stream);
    cudaMemcpy2DAsync(d->kc + b * slot, dst_pitch, k, src_pitch, (size_t)n_cached * row, d->L, cudaMemcpyDeviceToDevice,
                      d->stream);
    cudaMemcpy2DAsync(d->vc + b * slot, dst_pitch, v, src_pitch, (size_t)n_cached * row, d->L, cudaMemcpyDeviceToDevice,
                      d->stream);
    cudaMemcpy2DAsync(d->shift + (size_t)b * d->L * 2 * d->C + (size_t)(n_cached & 1) * d->C, 2 * row, tail, row, row,
                      d->L, cudaMemcpyDeviceToDevice, d->stream);
    cudaMemcpyAsync(d->prev + (size_t)b * d->C, prev, row, cudaMemcpyDeviceToDevice, d->stream);
    cudaMemcpyAsync(d->engram_ring + (size_t)b * d->engram_layers * d->ring * d->C, ring,
                    (size_t)d->engram_layers * d->ring * row, cudaMemcpyDeviceToDevice, d->stream);
}

/* The active slots (bit b: slot b) of the next steps. Enqueued after the steps launched so far. */
extern "C" void dec_set_active(void *handle, unsigned long long mask) {
    Decoder *d = (Decoder *)handle;
    cudaMemcpyAsync(d->active, &mask, sizeof(mask), cudaMemcpyHostToDevice, d->stream);
}

/* Start part `back` of one step of the active slots (asynchronous): the front (0) embeds and runs the blocks before
 * the first Engram layer; the back (1) reads the Engram rows and writes each slot's token at its next position (the
 * argmax, or its prompt's while the prompt lasts), which dec_tokens returns. */
extern "C" void dec_launch(void *handle, int back) {
    Decoder *d = (Decoder *)handle;
    cudaGraphLaunch(d->exec[back], d->stream);
}

/* Wait for the launched step; out[b]: the token slot b wrote (stale for inactive slots). */
extern "C" void dec_tokens(void *handle, int32_t *out) {
    Decoder *d = (Decoder *)handle;
    cudaStreamSynchronize(d->stream);
    for (int b = 0; b < d->B; b++) out[b] = d->host_tokens[b];
}
