/*
 * Decoding of a nanochat CausalLM (model/modeling.py) for B independent sequences (slots): each step advances every
 * active slot by one token, chosen greedily or sampled (optionally from its nucleus), optionally restricted to a
 * per-slot set of allowed tokens (inference/decoder.py).
 *
 * A small-batch step is bound by weight reads and kernel launches: each weight is read from DRAM about once (each slot
 * group's GEMV block rereads it from L2), and the step's 7 kernels per layer plus 4 replay as one CUDA graph.
 * Positions, input tokens, prompts, KV caches, the active-slot mask and each slot's sampling parameters and allowed
 * tokens live in device memory. A slot starts at position 0 of its prompt (dec_set_prompt): until the prompt's last
 * token, a step's next input is the prompt's next token (k_commit), so prompts and generations run through the same
 * kernels, every slot's prompt advancing in the same steps as the other slots' generations; inactive slots are skipped
 * by every kernel. A prompt's K and V depend on its tokens alone, so the prompt store keeps them for a later request
 * with the same prompt, which starts at its last token (dec_save_prompt, dec_load_prompt; SGLang's
 * RadixAttention, arXiv 2312.07104, reuses prompt prefixes likewise).
 *
 * The step is captured whole (dec_launch), the greedy argmax for every slot, and as its body, everything before the
 * lm_head GEMV, and a head that chooses by each slot's allowed tokens, temperature and top_p (dec_launch_body,
 * dec_launch_head): the host supplies the allowed tokens, which follow from each slot's previous token, while the
 * body runs (decoder.py). The whole step's GEMVs compile without the choice (k_gemv's CHOICE).
 *
 * Each step (modeling.py's forward at one position; C width, H heads of HEAD_DIM, x0 the normed embedding):
 *   k_norm    x0 = norm(wte[token])
 *   per layer i:
 *     k_norm  xm = resid_lambdas[i] x + x0_lambdas[i] x0 (x = x0 before layer 0), h = norm(xm)
 *     k_gemv  [q k v] = h [Wq Wk Wv]^T
 *     k_attn  VE layers: v += 2 sigmoid(W_g[head] . h) value_embeds[i][token] (per head, fp32);
 *             q, k = rope(norm_D(q) q_gain), rope(norm_D(k) k_gain) (qk_row.cuh, as qk_prologue.cu);
 *             y = softmax(q K^T / sqrt(HEAD_DIM)) V over the layer's window (FlashAttention's window_size left)
 *     k_gemv  xo = xm + y Wo^T
 *     k_norm  hn = norm(xo)
 *     k_gemv  u = relu(hn Wfc^T)^2
 *     k_gemv  x = xo + u Wproj^T
 *   k_norm    hn = norm(x)
 *   k_gemv    the token choice from s = hn W_lm^T over the slot's allowed tokens (all if it has none): argmax s at
 *             temperature 0, else argmax s / T + Gumbel noise (sampling softmax(s / T), gumbel), by atomicMax on a
 *             packed key; a nucleus-sampled slot's s are stored instead (head only)
 *   k_nucleus a nucleus-sampled slot's choice: that argmax over the fewest most likely allowed tokens holding top_p of
 *             softmax(s / T) (head only)
 *   k_commit  the prompt's next token, or once past the prompt the choice, becomes each active slot's next input
 *
 * Numerics follow model/modeling.py's forward op by op: bf16 copies of the fp32 matrices and tables; fp32 rotary
 * table, q/k norm weights and value-embedding gate (no contraction), the gate's mix rounded once to bf16; resid/x0
 * lambdas as bf16 values on bf16 tensors (PyTorch's 0-dim fp32 x bf16 promotion), each product and sum rounded; ReLU^2
 * from the fp32 accumulators as QuACK's gemm_act relu_sq (quack/activation.py), one bf16 rounding; F.rms_norm's fp32
 * epsilon; the choice over fp32 logits.
 *
 * No sum mixes slots and every sum's order is fixed by the shapes and the slot's position (GEMV: MMAs over K in fixed
 * order; attention: split count from the position; nucleus: fixed-order block sums), and a sample's noise depends only
 * on its request's seed and the generated position, so a sequence's tokens do not depend on the batch or on timing.
 *
 * Launch shapes: a GEMV runs a one-warp block per (row tile, slot group); attention runs a thread per head dim in
 * (head, split, slot) blocks, at most a wave of splits per head and slot (k_attn).
 */
#include <initializer_list>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <utility>

#include <curand_kernel.h>

#include "qk_row.cuh"

constexpr int LANE_DIMS = HEAD_DIM / WARP;         /* a lane's dims of a query, key or value */
constexpr int ATTN_WARPS = HEAD_DIM / WARP;        /* an attention block: a thread per head dim */
/* slots per decoder: k_commit runs them as one block of MAX_BATCH threads */
constexpr int MAX_BATCH = 256;
constexpr int MASK_BITS = CHAR_BIT * sizeof(unsigned long long);  /* slots per word of a slot mask */
constexpr int MASK_WORDS = MAX_BATCH / MASK_BITS;                 /* words of a slot mask */
static_assert(MAX_BATCH % MASK_BITS == 0, "slot masks are whole words");
/* mma.sync m16n8k16 (PTX ISA "Matrix Fragments for mma.m16n8k16"): lanes form MMA_N groups (groupID = lane / GROUP) of
 * GROUP threads (threadID_in_group = lane % GROUP), each holding ACC fp32 accumulators */
constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 16;
constexpr int GROUP = WARP / MMA_N, ACC = MMA_M * MMA_N / WARP;
constexpr int K_BLOCK = GROUP * VEC;               /* k per group step: VEC per lane */
static_assert(K_BLOCK == 2 * MMA_K, "a K_BLOCK block is two k16 MMAs (k_gemv)");
/* k_gemv: a block's slots (GROUP_TILES n tiles, their accumulators in registers: 16 slots, so a row tile has B / 16
 * blocks to spread over the SMs) and the K_BLOCK blocks of weights its warp keeps in flight. DEPTH by Little's law:
 * the bytes an SM must keep in flight are its share of DRAM bandwidth times the memory latency; a warp holds 32
 * lanes x 2 x 16 bytes per block, 8 KB at 8 */
constexpr int GROUP_TILES = 2, GROUP_SLOTS = GROUP_TILES * MMA_N, DEPTH = 8;
static_assert(MASK_BITS % GROUP_SLOTS == 0, "a slot group lies in one word of the active mask (k_gemv)");
constexpr int NORM_WARPS = 4;                      /* k_norm: slots (a warp each) per block */
constexpr int WORD_BITS = CHAR_BIT * sizeof(uint32_t);  /* tokens per word of an allowed-token mask */
/* k_nucleus: threads per slot, a block each: the most a block holds (CUDA C++ Programming Guide, "Thread Hierarchy") */
constexpr int NUCLEUS_THREADS = 1024;

enum { NORM, RESID_NORM, EMBED };
enum { EPI_STORE, EPI_RESID, EPI_RELU2, EPI_ARGMAX };

/* A failed CUDA runtime call (an allocation lost to another process between cudaMemGetInfo and cudaMalloc, an
 * asynchronous kernel or copy error surfacing at the step's synchronization) aborts the process: never silently wrong
 * tokens from a missing buffer or a sticky error. */
#define CUDA_CHECK(call) cuda_check((call), #call, __LINE__)
static void cuda_check(cudaError_t error, const char *call, int line) {
    if (error == cudaSuccess) return;
    fprintf(stderr, "decode.cu:%d: %s: %s\n", line, call, cudaGetErrorString(error));
    abort();
}

static __device__ __forceinline__ float bfr(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

static __device__ __forceinline__ float sigmoidf(float x) { return __fdiv_rn(1.f, __fadd_rn(1.f, expf(-x))); }

/* Bit b of a slot mask (MASK_WORDS words, slot b at bit b % MASK_BITS of word b / MASK_BITS). */
static __device__ __forceinline__ bool live(const unsigned long long *mask, int b) {
    return (mask[b / MASK_BITS] >> (b % MASK_BITS)) & 1ull;
}

/* Maximum over the warp, to every lane. */
static __device__ __forceinline__ float warp_max(float v) {
    for (int o = WARP / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

/* The block's sum of v to every thread, in a fixed order: each warp's butterfly, then the warps' sums in warp order.
 * scratch: a warp_count floats of shared memory. */
static __device__ float block_sum(float v, float *scratch) {
    v = warp_sum(v);
    __syncthreads();  /* the previous call's reads of scratch are done */
    if (threadIdx.x % WARP == 0) scratch[threadIdx.x / WARP] = v;
    __syncthreads();
    float sum = 0.f;
    for (int w = 0; w < (int)blockDim.x / WARP; w++) sum = __fadd_rn(sum, scratch[w]);
    return sum;
}

/* The block's maximum of v to every thread (an exact maximum, so any order). */
static __device__ float block_max(float v, float *scratch) {
    v = warp_max(v);
    __syncthreads();
    if (threadIdx.x % WARP == 0) scratch[threadIdx.x / WARP] = v;
    __syncthreads();
    float m = -INFINITY;
    for (int w = 0; w < (int)blockDim.x / WARP; w++) m = fmaxf(m, scratch[w]);
    return m;
}

/* A token choice's packed key: the score's IEEE bits mapped to an unsigned order (negatives inverted, positives above
 * them), then ~row, so the largest key holds the largest score and, among exact ties, the lowest row. */
static __device__ __forceinline__ unsigned long long choice_key(float score, int row) {
    const uint32_t bits = __float_as_uint(score);
    const uint32_t ordered = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
    return ((unsigned long long)ordered << 32) | (UINT32_MAX - (uint32_t)row);
}

/* Gumbel noise -log(-log(u)) of token row at generated position i under a request's seed, u uniform on (0, 1) from the
 * counter-based Philox4x32-10 ("Parallel Random Numbers: As Easy as 1, 2, 3", SC '11; cuRAND's
 * curand_Philox4x32_10) at counter (row, i), key seed: so the draw depends on (seed, i, row) alone. u = (2r + 1)
 * 2^-DBL_MANT_DIG of DBL_MANT_DIG - 1 random bits r is exact in double and never 0 or 1. argmax over tokens of s / T +
 * noise samples softmax(s / T) (the Gumbel-max trick; "A* Sampling", arXiv 1411.0030,
 * §2). */
static __device__ __forceinline__ float gumbel(unsigned long long seed, int i, int row) {
    const uint4 x = curand_Philox4x32_10(make_uint4((uint32_t)row, (uint32_t)i, 0u, 0u),
                                         make_uint2((uint32_t)seed, (uint32_t)(seed >> 32)));
    const unsigned long long r = ((unsigned long long)x.x << 32 | x.y) >> (64 - (DBL_MANT_DIG - 1));
    return (float)-log(-log(ldexp(2.0 * (double)r + 1.0, -DBL_MANT_DIG)));
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* GEMV of B slots on the tensor cores: out[b][r] = W[r] . x[b] by mma.sync m16n8k16 (bf16 operands, fp32 accumulators).
 * A block is one warp: one MMA_M-row tile of W and one group of GROUP_SLOTS slots, the grid's x the group (the
 * fastest-varying index, so a row tile's groups run together and read its weights from L2 after the first), y the row
 * tile; a group without an active slot exits. Many small blocks keep every SM loading (flat GEMMs with few rows are
 * parallelism-bound: FlashDecoding++, arXiv 2311.01282, §4). A slot is one column of an n tile, and each output's sum
 * runs over K in a fixed order (K_BLOCK blocks, two k16 MMAs each), so a slot's outputs do not depend on B. In a
 * K_BLOCK block lane (g, t) = (groupID, threadID_in_group) loads k t*VEC.. of rows g, g + 8 and of its column's input
 * (one 16-byte load each); A and B map MMA k to physical k alike, so the product is W . x. The warp keeps DEPTH blocks
 * of weights in flight (a register ring) and copies its group's bf16 inputs to shared memory by cp.async a K chunk at a
 * time, [slots][chunk + VEC] (the VEC pad staggers banks; inactive slots zero; chunk: gemv_chunk), the copy overlapping
 * the ring's first loads. RELU2: out = bf16(relu(s) s) of the fp32 dot product s (QuACK gemm_act relu_sq). ARGMAX: the
 * warp's maximum key per slot over its 16 rows (an exact integer maximum, so any order), then one atomicMax. */

typedef struct {
    const bf16 *w;               /* [rows][cols] */
    int rows, cols, chunk;       /* chunk: the k of the slots' inputs held in shared memory at once (gemv_chunk) */
    const bf16 *in;              /* [B][cols] */
    int epi;                     /* epilogue: what is done with each output */
    bf16 *out;                   /* [B][rows]: STORE/RESID/RELU2 output */
    const bf16 *res;             /* [B][rows]: RESID: residual added */
    unsigned long long *argmax;  /* [B] ARGMAX: packed (ordered score, ~row) maximum */
    const unsigned long long *active;  /* [2][MASK_WORDS]: the active slots' mask, then the masked slots' (CHOICE) */
    int B;
} Gemv;

/* The per-slot inputs of a CHOICE ARGMAX and of k_nucleus, a kernel argument of their own: growing Gemv changes the
 * compiled GEMVs. */
typedef struct {
    const uint32_t *allow;           /* [B][rows / WORD_BITS]: a masked slot's allowed rows, bit row % WORD_BITS of
                                      * word row / WORD_BITS */
    const float *temperature;        /* [B]: 0 the logits' argmax, else Gumbel-max sampling at that temperature */
    const float *top_p;              /* [B]: at temperature > 0, below 1: from the nucleus holding that much
                                      * (k_nucleus) */
    const unsigned long long *seed;  /* [B] */
    const int32_t *pos, *first;      /* [B]: the input's position and the first generated token's (i = pos - first) */
    float *logits;                   /* [B][rows]: a nucleus-sampled slot's logits, -inf outside its allowed rows */
    float *probs;                    /* [B][rows]: k_nucleus's exp(s / T - max) */
    float softcap;                   /* the model's output-logit cap c (ModelConfig.final_logit_softcapping): a sampled
                                      * slot's logits are c tanh(s / c); 0 none. A greedy choice needs none: tanh is
                                      * increasing, so the argmax is the uncapped logits' (but for fp32 ties of
                                      * saturated caps) */
} Choice;

static __device__ __forceinline__ void mma16816(float *d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

/* 16 bytes global -> shared without a register round trip (cp.async, PTX ISA "Data Movement: cp.async"): all of a
 * thread's copies are in flight at once; valid false writes zeros (src-size 0). */
static __device__ __forceinline__ void copy16(void *dst, const void *src, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src), "r"(valid ? 16 : 0));
}

static __device__ __forceinline__ void copies_done() { asm volatile("cp.async.wait_all;\n" ::: "memory"); }

/* CHOICE: ARGMAX honours each slot's allowed tokens and temperature (c; the head graph); without it, the logits' argmax
 * (the whole-step graph). A compile-time argument: code added to the kernel changes the compiled loops of every GEMV,
 * so greedy steps run the kernel compiled without it. */
template <bool CHOICE> __global__ void k_gemv(Gemv a, Choice c) {
    extern __shared__ __align__(16) unsigned char gemv_buf[];
    bf16 *xs = reinterpret_cast<bf16 *>(gemv_buf);  /* [GROUP_SLOTS][ks] */
    const int s0 = blockIdx.x * GROUP_SLOTS, row0 = blockIdx.y * MMA_M;
    /* the group's active slots, GROUP_SLOTS bits of one mask word; its n tiles up to the last active */
    const unsigned long long live_bits =
        (a.active[s0 / MASK_BITS] >> (s0 % MASK_BITS)) & (~0ull >> (MASK_BITS - GROUP_SLOTS));
    if (!live_bits) return;
    const int tiles = (MASK_BITS - __clzll(live_bits) + MMA_N - 1) / MMA_N;
    const int K = a.cols, kc = a.chunk, ks = kc + VEC, lane = threadIdx.x, g = lane / GROUP, t = lane % GROUP;
    /* past the end: loads only */
    const int r0 = min(row0 + g, a.rows - 1), r1 = min(row0 + g + MMA_M / 2, a.rows - 1);
    const uint4 *w0 = reinterpret_cast<const uint4 *>(a.w + (size_t)r0 * K);
    const uint4 *w1 = reinterpret_cast<const uint4 *>(a.w + (size_t)r1 * K);
    float acc[GROUP_TILES][ACC];
#pragma unroll
    for (int n = 0; n < GROUP_TILES; n++)
        for (int i = 0; i < ACC; i++) acc[n][i] = 0.f;
    for (int k0 = 0; k0 < K; k0 += kc) {
        const int kn = min(kc, K - k0), vecs = kn / VEC;
        __syncwarp();  /* the previous chunk is consumed */
        for (int i = lane; i < tiles * MMA_N * vecs; i += WARP) {
            const int j = i / vecs, v = i % vecs;
            const bool valid = (live_bits >> j) & 1ull;
            copy16(xs + (size_t)j * ks + v * VEC, valid ? a.in + (size_t)(s0 + j) * K + k0 + v * VEC : a.in, valid);
        }
        /* the weights' first DEPTH blocks load while the inputs copy */
        const int blocks = kn / K_BLOCK, q0 = k0 / VEC + t;  /* q0: this lane's first 16-byte vector */
        uint4 wa[DEPTH], wb[DEPTH];
#pragma unroll
        for (int u = 0; u < DEPTH; u++)
            if (u < blocks) {
                wa[u] = __ldg(w0 + q0 + u * GROUP);
                wb[u] = __ldg(w1 + q0 + u * GROUP);
            }
        copies_done();
        __syncwarp();
        for (int j0 = 0; j0 < blocks; j0 += DEPTH) {
#pragma unroll
            for (int u = 0; u < DEPTH; u++) {
                const int j = j0 + u;
                if (j >= blocks) break;
                const uint4 a0 = wa[u], a1 = wb[u];
                if (j + DEPTH < blocks) {
                    wa[u] = __ldg(w0 + q0 + (j + DEPTH) * GROUP);
                    wb[u] = __ldg(w1 + q0 + (j + DEPTH) * GROUP);
                }
#pragma unroll
                for (int n = 0; n < GROUP_TILES; n++) {
                    if (n >= tiles) break;
                    const uint4 x = *reinterpret_cast<const uint4 *>(xs + (size_t)(n * MMA_N + g) * ks + j * K_BLOCK +
                                                                     t * VEC);
                    mma16816(acc[n], a0.x, a1.x, a0.y, a1.y, x.x, x.y);
                    mma16816(acc[n], a0.z, a1.z, a0.w, a1.w, x.z, x.w);
                }
            }
        }
    }
#pragma unroll
    for (int n = 0; n < GROUP_TILES; n++) {
        if (n >= tiles) break;
        /* ARGMAX: the lane's best key per column (2t, 2t + 1) over its rows g, g + 8; 0 below every key */
        unsigned long long best[2] = {0ull, 0ull};
        for (int i = 0; i < ACC; i++) {  /* the m16n8 accumulator fragment: rows g, g + 8; columns 2t, 2t + 1 */
            const int row = row0 + g + i / 2 * (MMA_M / 2), j = n * MMA_N + t * 2 + i % 2, b = s0 + j;
            if (row >= a.rows || !((live_bits >> j) & 1ull)) continue;
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
            } else {  /* EPI_ARGMAX: largest score, lowest row among exact ties (choice_key): the fp32 logit, or with
                       * CHOICE over a masked slot's allowed rows, at temperature T > 0 logit / T + gumbel; a
                       * nucleus-sampled slot's logits go to k_nucleus instead */
                float score = s;
                if constexpr (CHOICE) {
                    const bool allowed = !live(a.active + MASK_WORDS, b) ||
                        ((c.allow[(size_t)b * (a.rows / WORD_BITS) + row / WORD_BITS] >> (row % WORD_BITS)) & 1u);
                    const float z = c.softcap > 0.f && c.temperature[b] > 0.f
                        ? __fmul_rn(c.softcap, tanhf(__fdiv_rn(s, c.softcap))) : s;
                    if (c.temperature[b] > 0.f && c.top_p[b] < 1.f) {
                        c.logits[at] = allowed ? z : -INFINITY;
                        continue;
                    }
                    if (!allowed) continue;
                    if (c.temperature[b] > 0.f)
                        score = __fadd_rn(__fdiv_rn(z, c.temperature[b]),
                                          gumbel(c.seed[b], c.pos[b] - c.first[b], row));
                }
                const unsigned long long key = choice_key(score, row);
                best[i % 2] = key > best[i % 2] ? key : best[i % 2];
            }
        }
        if (a.epi == EPI_ARGMAX) {
            for (int o = GROUP; o < WARP; o <<= 1)  /* over g: the lanes sharing t */
                for (int p = 0; p < 2; p++) {
                    const unsigned long long other = __shfl_xor_sync(0xffffffffu, best[p], o);
                    best[p] = other > best[p] ? other : best[p];
                }
            if (g == 0)
                for (int p = 0; p < 2; p++)
                    if (best[p]) atomicMax(a.argmax + s0 + n * MMA_N + t * 2 + p, best[p]);
        }
    }
}

/* A nucleus-sampled slot's choice (top-p sampling: arXiv 1904.09751, "The Curious Case of Neural Text Degeneration",
 * §3.1): Gumbel-max at temperature T, as k_gemv's epilogue, over the fewest most likely allowed tokens whose
 * probability under softmax(s / T) reaches top_p; its key to argmax for k_commit. The nucleus is Hugging Face
 * transformers' TopPLogitsWarper's (the generator of MultiPL-E automodel.py and bigcode-evaluation-harness), which
 * sorts the probabilities ascending and drops every token whose cumulative probability is at most 1 - top_p; here,
 * with e = exp(s / T - max s / T) (0 outside the allowed tokens) and C(x) the sum of e over the tokens with e <= x,
 * the tokens kept are those with e >= e*, the least e with C(e) > (1 - top_p) sum(e), found by bisection on the IEEE
 * bits of x in [0, 1] (C is a step function rising at each e, so the least such bits are an e's). Tokens of exactly
 * equal probability are kept or dropped together. The sums run in a fixed order, so the choice depends on the slot
 * alone. A block per slot; inactive, greedy and top_p 1 slots exit. */
__global__ void __launch_bounds__(NUCLEUS_THREADS) k_nucleus(Choice c, const unsigned long long *active,
                                                              unsigned long long *argmax, int vocab) {
    __shared__ float scratch[NUCLEUS_THREADS / WARP];
    __shared__ unsigned long long keys[NUCLEUS_THREADS / WARP];
    const int b = blockIdx.x;
    if (!live(active, b) || !(c.temperature[b] > 0.f && c.top_p[b] < 1.f)) return;
    const float T = c.temperature[b];
    const float *s = c.logits + (size_t)b * vocab;
    float *e = c.probs + (size_t)b * vocab;  /* a thread reads back only the rows it wrote */
    float m = -INFINITY;
    for (int r = threadIdx.x; r < vocab; r += NUCLEUS_THREADS) m = fmaxf(m, __fdiv_rn(s[r], T));
    m = block_max(m, scratch);
    float sum = 0.f;
    for (int r = threadIdx.x; r < vocab; r += NUCLEUS_THREADS) {
        e[r] = expf(__fsub_rn(__fdiv_rn(s[r], T), m));  /* exp(-inf) = 0 outside the allowed tokens */
        sum = __fadd_rn(sum, e[r]);
    }
    const float drop = __fmul_rn(__fsub_rn(1.f, c.top_p[b]), block_sum(sum, scratch));
    /* the least bits u with C(u) > drop: C(1) is the whole sum, so u ends at most at 1's bits */
    uint32_t lo = 0u, hi = __float_as_uint(1.f);
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2;
        const float x = __uint_as_float(mid);
        float below = 0.f;
        for (int r = threadIdx.x; r < vocab; r += NUCLEUS_THREADS)
            if (e[r] <= x) below = __fadd_rn(below, e[r]);
        if (block_sum(below, scratch) > drop) hi = mid; else lo = mid + 1;
    }
    const float least = __uint_as_float(lo);
    const int i = c.pos[b] - c.first[b];
    unsigned long long best = 0ull;
    for (int r = threadIdx.x; r < vocab; r += NUCLEUS_THREADS) {
        if (e[r] < least) continue;
        const unsigned long long key = choice_key(__fadd_rn(__fdiv_rn(s[r], T), gumbel(c.seed[b], i, r)), r);
        best = key > best ? key : best;
    }
    for (int o = WARP / 2; o > 0; o >>= 1) {
        const unsigned long long other = __shfl_xor_sync(0xffffffffu, best, o);
        best = other > best ? other : best;
    }
    if (threadIdx.x % WARP == 0) keys[threadIdx.x / WARP] = best;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < NUCLEUS_THREADS / WARP; w++) best = keys[w] > best ? keys[w] : best;
        argmax[b] = best;
    }
}

/* k_gemv's shared memory: a group's inputs [GROUP_SLOTS][kc + VEC] bf16. */
static size_t gemv_smem(int kc) { return (size_t)GROUP_SLOTS * (kc + VEC) * sizeof(bf16); }

/* The largest chunk of whole K_BLOCK blocks, at most K, whose inputs fit half the opt-in shared memory per block (two
 * resident blocks per SM at least). */
static int gemv_chunk(int K, int optin) {
    int kc = K / K_BLOCK * K_BLOCK;
    while (kc > K_BLOCK && gemv_smem(kc) > (size_t)optin / 2) kc -= K_BLOCK;
    return kc;
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* RMS norm of each active slot's input row, a warp per slot, the sum of squares over K in lane-strided order then
 * warp_sum, r = rsqrt(mean + FLT_EPSILON): NORM out = bf16(in r) of the slot's row of in; EMBED the same of in's row
 * tokens[b] (x0 = rms_norm(wte[token])); RESID_NORM mixed = bf16(bf16(l1 in) + bf16(l2 in2)) (the resid and x0
 * lambdas, bf16 values), out = bf16(mixed r). */
__global__ void k_norm(int kind, const bf16 *in, const bf16 *in2, const int32_t *tokens, float l1, float l2,
                       bf16 *mixed, bf16 *out, const unsigned long long *active, int B, int K) {
    const int b = blockIdx.x * NORM_WARPS + threadIdx.x / WARP, lane = threadIdx.x % WARP;
    if (b >= B || !live(active, b)) return;
    const size_t row = (size_t)b * K, src = kind == EMBED ? (size_t)tokens[b] * K : row;
    auto value = [&](int i) {
        const float v = __bfloat162float(in[src + i]);
        return kind == RESID_NORM ? bfr(bfr(l1 * v) + bfr(l2 * __bfloat162float(in2[row + i]))) : v;
    };
    float s = 0.f;
    for (int i = lane; i < K; i += WARP) {
        const float v = value(i);
        s = __fmaf_rn(v, v, s);
    }
    const float r = rsqrtf(warp_sum(s) / (float)K + FLT_EPSILON);
    for (int i = lane; i < K; i += WARP) {
        const float v = value(i);
        if (kind == RESID_NORM) mixed[row + i] = __float2bfloat16_rn(v);
        out[row + i] = __float2bfloat16_rn(bfr(v * r));
    }
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
    const float *cos, *sin;    /* the rotary table [context][HALF], fp32 */
    bf16 *kc, *vc;             /* the layer's cache of slot 0, [max_len][C]; slot b's at + b kv_slot */
    size_t kv_slot;
    bf16 *out;                 /* [B][C] */
    const int32_t *tokens, *pos;  /* [B]: each slot's input token and its position */
    const unsigned long long *active;
    float *part;               /* slot b, head h, split s: [(b H + h) part_slot + s][HEAD_DIM], each split's value
                                * sum */
    float2 *stat;              /* same index: each split's maximum score m (x) and exponential sum l (y) */
    unsigned *done;            /* [B][H]: the head's splits finished in this launch; its last resets it */
    int part_slot;             /* splits per head and slot the partial buffers hold */
    int C, H, window, run;
} Attn;

/* Slot b's q, k, v of the head at position p (fp32 holding bf16 values), as modeling.py: VE layers
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
    if (!live(a.active, b)) return;
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

/* Each active slot's next input token and position: the prompt's next token while the slot is in its prompt, else the
 * chosen token (ARGMAX's maximum); a thread per slot. */
__global__ void k_commit(int32_t *tokens, int32_t *pos, const unsigned long long *active, unsigned long long *argmax,
                         const int32_t *prompt, const int32_t *plen, int max_len, int B) {
    const int b = threadIdx.x;
    if (b >= B || !live(active, b)) return;
    const int next = pos[b] + 1;
    tokens[b] = next < plen[b] ? prompt[(size_t)b * max_len + next] : (int32_t)(UINT32_MAX - (uint32_t)argmax[b]);
    pos[b] = next;
    argmax[b] = 0ull;
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* Host side: one decoder per model, its step captured as CUDA graphs. */

typedef struct {
    const bf16 *qkv, *o, *fc, *proj, *ve;  /* fc: [F][C]; ve: NULL in a layer without value embeddings */
    const float *ve_gate;                 /* the value-embedding gate [H][C], fp32; NULL without */
    const float *q_gain, *k_gain;         /* [HEAD_DIM] fp32 */
    float resid_lambda, x0_lambda;
    int window;
} Layer;

typedef struct {
    /* max_len: a slot's cache positions (the caller's longest sequence). optin: the opt-in shared memory per block.
     * store: the prompt store's positions */
    int B, L, H, C, F, vocab, max_len, optin, attn_wave, attn_run, store;
    const bf16 *wte, *lm_head;
    const float *cos, *sin;   /* the fp32 rotary table [context][HALF] */
    Layer *layer;
    int32_t *tokens, *pos;    /* [B]: each slot's input token and position */
    int32_t *prompt, *plen;   /* [B][max_len], [B]: each slot's prompt and its length */
    int32_t *first;           /* [B]: the position of each slot's first generated token's input */
    float *temperature;       /* [B] */
    float *top_p;             /* [B] */
    unsigned long long *seed; /* [B] */
    float *logits, *probs;    /* [B][vocab]: k_nucleus's inputs (Choice) */
    float softcap;            /* the output-logit cap (Choice), 0 none */
    uint32_t *allow;          /* [B][vocab / WORD_BITS]: each masked slot's allowed tokens */
    uint32_t *host_allow;     /* pinned host, as allow: written by the caller (dec_masks), copied by dec_set_active */
    unsigned long long *active;  /* device [2][MASK_WORDS]: the active slots' mask, then the masked slots' */
    int32_t *host_tokens;     /* pinned host [B]: the last step's tokens */
    unsigned long long *argmax;  /* [B] */
    bf16 *x, *x0, *xm, *h, *hn, *qkv_out, *att, *xo, *u;  /* [B][...]; hn: the normed input of fc and lm_head */
    bf16 *kc, *vc;            /* [B][L][max_len][C] */
    bf16 *store_k, *store_v;  /* [L][store][C] each, one allocation (V after K): prompts' K and V kept for reuse
                               * (dec_save_prompt, dec_load_prompt) */
    float *attn_part;         /* k_attn's partials, reused by every layer: [B][H][attn_wave][HEAD_DIM] value sums */
    float2 *attn_stat;        /* [B][H][attn_wave]: their (m, l) */
    unsigned *attn_done;      /* [B][H]: k_attn's counters, zero between launches */
    cudaStream_t stream;      /* the caller's */
    cudaGraphExec_t step, body, head;  /* the whole step; before the lm_head GEMV; from it on */
} Decoder;

/* f(buffer, bytes per slot) for each per-slot device buffer. */
template <class F> static void slot_buffers(Decoder *d, F f) {
    const size_t C = d->C, kv = (size_t)d->L * d->max_len * C * sizeof(bf16), part = (size_t)d->H * d->attn_wave;
    f((void **)&d->tokens, sizeof(int32_t));
    f((void **)&d->pos, sizeof(int32_t));
    f((void **)&d->prompt, d->max_len * sizeof(int32_t));
    f((void **)&d->plen, sizeof(int32_t));
    f((void **)&d->argmax, sizeof(unsigned long long));
    f((void **)&d->first, sizeof(int32_t));
    f((void **)&d->temperature, sizeof(float));
    f((void **)&d->top_p, sizeof(float));
    f((void **)&d->seed, sizeof(unsigned long long));
    f((void **)&d->logits, d->vocab * sizeof(float));
    f((void **)&d->probs, d->vocab * sizeof(float));
    f((void **)&d->allow, d->vocab / WORD_BITS * sizeof(uint32_t));
    for (bf16 **b : {&d->x, &d->x0, &d->xm, &d->h, &d->hn, &d->att, &d->xo}) f((void **)b, C * sizeof(bf16));
    f((void **)&d->qkv_out, 3 * C * sizeof(bf16));
    f((void **)&d->u, d->F * sizeof(bf16));
    f((void **)&d->kc, kv);
    f((void **)&d->vc, kv);
    f((void **)&d->attn_part, part * HEAD_DIM * sizeof(float));
    f((void **)&d->attn_stat, part * sizeof(float2));
    f((void **)&d->attn_done, d->H * sizeof(unsigned));
}

static Gemv gemv(const Decoder *d, const bf16 *w, int rows, int cols, const bf16 *in, int epi, bf16 *out) {
    Gemv g = {};
    g.w = w; g.rows = rows; g.cols = cols; g.in = in; g.epi = epi; g.out = out;
    g.active = d->active; g.B = d->B;
    return g;
}

static void launch_norm(Decoder *d, int kind, const bf16 *in, const bf16 *in2, float l1, float l2, bf16 *mixed,
                        bf16 *out) {
    k_norm<<<(d->B + NORM_WARPS - 1) / NORM_WARPS, NORM_WARPS * WARP, 0, d->stream>>>(
        kind, in, in2, d->tokens, l1, l2, mixed, out, d->active, d->B, d->C);
}

static Choice choice(const Decoder *d) {
    return {d->allow, d->temperature, d->top_p, d->seed, d->pos, d->first, d->logits, d->probs, d->softcap};
}

/* A one-warp block per (slot group, row tile); groups without an active slot exit at once. */
template <bool CHOICE> static void launch_gemv(Decoder *d, Gemv g) {
    g.chunk = gemv_chunk(g.cols, d->optin);
    const dim3 grid((g.B + GROUP_SLOTS - 1) / GROUP_SLOTS, (g.rows + MMA_M - 1) / MMA_M);
    k_gemv<CHOICE><<<grid, WARP, gemv_smem(g.chunk), d->stream>>>(g, choice(d));
}

/* The step's body on d->stream: every layer of every active slot, up to x before the final norm. */
static void enqueue_body(Decoder *d) {
    const int C = d->C, F = d->F, L = d->L, B = d->B;
    launch_norm(d, EMBED, d->wte, NULL, 0.f, 0.f, NULL, d->x0);
    for (int l = 0; l < L; l++) {
        const Layer *ly = &d->layer[l];
        /* input: x0 at layer 0, else the previous layer's output */
        launch_norm(d, RESID_NORM, l == 0 ? d->x0 : d->x, d->x0, ly->resid_lambda, ly->x0_lambda, d->xm, d->h);
        launch_gemv<false>(d, gemv(d, ly->qkv, 3 * C, C, d->h, EPI_STORE, d->qkv_out));
        /* splits per head and slot: runs of attn_run keys over at most the window and the cache */
        const int keys = ly->window < d->max_len - 1 ? ly->window : d->max_len - 1;
        const int splits = (keys + d->attn_run - 1) / d->attn_run;
        const Attn a = {d->qkv_out, d->h, ly->ve, ly->ve_gate, ly->q_gain, ly->k_gain, d->cos, d->sin,
                        d->kc + (size_t)l * d->max_len * C, d->vc + (size_t)l * d->max_len * C,
                        (size_t)L * d->max_len * C, d->att, d->tokens, d->pos, d->active, d->attn_part, d->attn_stat,
                        d->attn_done, d->attn_wave, C, d->H,
                        ly->window, d->attn_run};
        k_attn<<<dim3(d->H, splits, B), HEAD_DIM, splits * sizeof(float), d->stream>>>(a);
        Gemv g = gemv(d, ly->o, C, C, d->att, EPI_RESID, d->xo);
        g.res = d->xm;
        launch_gemv<false>(d, g);
        launch_norm(d, NORM, d->xo, NULL, 0.f, 0.f, NULL, d->hn);
        launch_gemv<false>(d, gemv(d, ly->fc, F, C, d->hn, EPI_RELU2, d->u));
        g = gemv(d, ly->proj, C, F, d->u, EPI_RESID, d->x);
        g.res = d->xo;
        launch_gemv<false>(d, g);
    }
}

/* The step's head on d->stream: the final norm, the token choice in lm_head's epilogue (k_gemv's CHOICE) or, for a
 * nucleus-sampled slot, by k_nucleus, the commit and the tokens' copy to the host. */
template <bool CHOICE> static void enqueue_head(Decoder *d) {
    launch_norm(d, NORM, d->x, NULL, 0.f, 0.f, NULL, d->hn);
    Gemv g = gemv(d, d->lm_head, d->vocab, d->C, d->hn, EPI_ARGMAX, NULL);
    g.argmax = d->argmax;
    launch_gemv<CHOICE>(d, g);
    if constexpr (CHOICE) k_nucleus<<<d->B, NUCLEUS_THREADS, 0, d->stream>>>(choice(d), d->active, d->argmax, d->vocab);
    k_commit<<<1, MAX_BATCH, 0, d->stream>>>(d->tokens, d->pos, d->active, d->argmax, d->prompt, d->plen, d->max_len,
                                             d->B);
    cudaMemcpyAsync(d->host_tokens, d->tokens, (size_t)d->B * sizeof(int32_t), cudaMemcpyDeviceToHost, d->stream);
}

/* The executable graph of the work enqueue() puts on d->stream. */
template <class F> static cudaGraphExec_t capture(Decoder *d, F enqueue) {
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    cudaStreamBeginCapture(d->stream, cudaStreamCaptureModeThreadLocal);
    enqueue();
    CUDA_CHECK(cudaStreamEndCapture(d->stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
    cudaGraphDestroy(graph);  /* an executable graph does not reference its source graph */
    return exec;
}

extern "C" void dec_destroy(void *handle) {
    Decoder *d = (Decoder *)handle;
    for (cudaGraphExec_t exec : {d->step, d->body, d->head}) cudaGraphExecDestroy(exec);
    slot_buffers(d, [](void **p, size_t) { cudaFree(*p); });
    cudaFree(d->active);
    cudaFree(d->store_k);
    cudaFreeHost(d->host_tokens);
    cudaFreeHost(d->host_allow);
    free(d->layer);
    free(d);
}

/* The slots of a decoder (dec_create). */
extern "C" int dec_batch(void *handle) { return ((Decoder *)handle)->B; }

/*
 * cfg: L, C, MLP hidden width F (fc: [F][C]; proj: [C][F]), vocab (lm_head's rows), max_len (a slot's cache positions,
 *      at most context), context (the model's max_position_embeddings), store (the prompt store's positions
 *      requested: it takes at most the device memory left after the slots and graphs).
 * ptrs: GLOBAL_PTRS pointers wte, cos, sin (the fp32 rotary table [context][HALF] of every layer), lm_head; then
 *       LAYER_PTRS per layer: qkv, o, fc, proj, value-embedding table (0 if none), value-embedding gate (0 if none),
 *       q norm weight, k norm weight.
 * scal: per layer: resid lambda, x0 lambda (bf16 values); then the output-logit cap
 *       (ModelConfig.final_logit_softcapping, 0 none).
 * window: per layer, the keys before each query that it attends (FlashAttention's window_size left).
 * stream: the caller's stream (not the legacy default stream: it is captured), alive past dec_destroy.
 */
constexpr int GLOBAL_PTRS = 4, LAYER_PTRS = 8;

extern "C" void *dec_create(const int *cfg, const uint64_t *ptrs, const float *scal, const int *window, void *stream) {
    Decoder *d = (Decoder *)calloc(1, sizeof(Decoder));
    d->L = cfg[0]; d->C = cfg[1]; d->F = cfg[2]; d->vocab = cfg[3]; d->max_len = cfg[4]; d->store = cfg[6];
    const int context = cfg[5];  /* the model's max_position_embeddings: the rotary table's rows, and attn_run's */
    d->H = d->C / HEAD_DIM;
    d->wte = (const bf16 *)ptrs[0]; d->cos = (const float *)ptrs[1]; d->sin = (const float *)ptrs[2];
    d->lm_head = (const bf16 *)ptrs[3]; d->stream = (cudaStream_t)stream;
    const int L = d->L;
    d->softcap = scal[2 * L];
    d->layer = (Layer *)calloc(L, sizeof(Layer));
    for (int l = 0; l < L; l++) {
        const uint64_t *p = ptrs + GLOBAL_PTRS + LAYER_PTRS * l;
        d->layer[l] = {(const bf16 *)p[0], (const bf16 *)p[1], (const bf16 *)p[2], (const bf16 *)p[3],
                       (const bf16 *)p[4], (const float *)p[5], (const float *)p[6], (const float *)p[7], scal[2 * l],
                       scal[2 * l + 1], window[l]};
    }
    int device;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&d->optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    for (const void *kernel : {(const void *)k_gemv<false>, (const void *)k_gemv<true>})
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, d->optin);
    /* attn_wave: (head, split) blocks the SMs hold at once, per head; residency is retaken with the per-split shared
     * memory of that wave (fewer splits fit). attn_run: keys per split, so context - 1 keys take one wave; from the
     * context, not max_len, so a position's splits, and the order its keys are summed in, do not depend on max_len */
    d->attn_wave = resident_blocks(k_attn, HEAD_DIM) / d->H;
    d->attn_wave = resident_blocks(k_attn, HEAD_DIM, d->attn_wave * sizeof(float)) / d->H;
    d->attn_run = (context - 1 + d->attn_wave - 1) / d->attn_wave;
    /* B: the most slots (at most MAX_BATCH) that fit free memory */
    size_t slot = 0, free_bytes, total_bytes;
    slot_buffers(d, [&](void **, size_t bytes) { slot += bytes; });
    cudaMemGetInfo(&free_bytes, &total_bytes);
    d->B = (int)(free_bytes / slot < (size_t)MAX_BATCH ? free_bytes / slot : MAX_BATCH);
    slot_buffers(d, [&](void **p, size_t bytes) {
        CUDA_CHECK(cudaMalloc(p, d->B * bytes));
        CUDA_CHECK(cudaMemset(*p, 0, d->B * bytes));
    });
    CUDA_CHECK(cudaMalloc((void **)&d->active, 2 * MASK_WORDS * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d->active, 0, 2 * MASK_WORDS * sizeof(unsigned long long)));
    CUDA_CHECK(cudaHostAlloc((void **)&d->host_tokens, d->B * sizeof(int32_t), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void **)&d->host_allow, (size_t)d->B * (d->vocab / WORD_BITS) * sizeof(uint32_t),
                             cudaHostAllocDefault));
    d->step = capture(d, [d] { enqueue_body(d); enqueue_head<false>(d); });
    d->body = capture(d, [d] { enqueue_body(d); });
    d->head = capture(d, [d] { enqueue_head<true>(d); });
    /* the prompt store: the requested positions, at most what free memory holds after the slots and graphs; K and V
     * in one allocation, shrunk 2 MiB at a time until it succeeds (cudaMemGetInfo's free bytes do not all fit one
     * allocation) */
    const size_t position = 2 * (size_t)L * d->C * sizeof(bf16);
    cudaMemGetInfo(&free_bytes, &total_bytes);
    if ((size_t)d->store > free_bytes / position) d->store = (int)(free_bytes / position);
    while (d->store > 0 && cudaMalloc((void **)&d->store_k, position * d->store) != cudaSuccess) {
        cudaGetLastError();  /* clear the failed allocation's error */
        const int step = (int)(((2 << 20) + position - 1) / position);
        d->store = d->store > step ? d->store - step : 0;
    }
    d->store_v = d->store_k + (size_t)L * d->store * d->C;
    return d;
}

/* Start slot b at position start of its prompt tokens[0..n) (n >= 2: BOS and at least one more; positions before start
 * already cached: 0, or n - 1 after dec_load_prompt): its steps run the rest of the prompt, then from the step whose
 * input is the prompt's last token choose tokens at temperature (0: argmax) with noise from seed (gumbel), from the
 * nucleus holding top_p of the probability when top_p < 1 (k_nucleus), the first choice at generated position 0.
 * Stream-ordered. */
extern "C" void dec_set_prompt(void *handle, int b, const int32_t *tokens, int32_t n, int32_t start, float temperature,
                               float top_p, unsigned long long seed) {
    Decoder *d = (Decoder *)handle;
    const int32_t last = n - 1;
    cudaMemcpyAsync(d->prompt + (size_t)b * d->max_len, tokens, (size_t)n * sizeof(int32_t), cudaMemcpyHostToDevice,
                    d->stream);
    cudaMemcpyAsync(d->plen + b, &n, sizeof(int32_t), cudaMemcpyHostToDevice, d->stream);
    cudaMemcpyAsync(d->tokens + b, tokens + start, sizeof(int32_t), cudaMemcpyHostToDevice, d->stream);
    cudaMemcpyAsync(d->pos + b, &start, sizeof(int32_t), cudaMemcpyHostToDevice, d->stream);
    cudaMemcpyAsync(d->first + b, &last, sizeof(int32_t), cudaMemcpyHostToDevice, d->stream);
    cudaMemcpyAsync(d->temperature + b, &temperature, sizeof(float), cudaMemcpyHostToDevice, d->stream);
    cudaMemcpyAsync(d->top_p + b, &top_p, sizeof(float), cudaMemcpyHostToDevice, d->stream);
    cudaMemcpyAsync(d->seed + b, &seed, sizeof(seed), cudaMemcpyHostToDevice, d->stream);
}

/* Slot b's cached K and V at positions [at, at + n) to the prompt store at [store_at, store_at + n) (to_store), or
 * back: one 2D copy of L rows each, the store's rows store positions apart, the slot's max_len. */
static void prompt_copy(Decoder *d, int b, int at, int store_at, int n, bool to_store) {
    const size_t row = (size_t)n * d->C * sizeof(bf16), slot_pitch = (size_t)d->max_len * d->C * sizeof(bf16),
                 store_pitch = (size_t)d->store * d->C * sizeof(bf16);
    const size_t in_slot = ((size_t)b * d->L * d->max_len + at) * d->C, in_store = (size_t)store_at * d->C;
    for (auto [cache, store] : {std::pair{d->kc, d->store_k}, std::pair{d->vc, d->store_v}}) {
        if (to_store)
            cudaMemcpy2DAsync(store + in_store, store_pitch, cache + in_slot, slot_pitch, row, d->L,
                              cudaMemcpyDeviceToDevice, d->stream);
        else
            cudaMemcpy2DAsync(cache + in_slot, slot_pitch, store + in_store, store_pitch, row, d->L,
                              cudaMemcpyDeviceToDevice, d->stream);
    }
}

/* Keep slot b's K and V at positions [at, at + n), written by the launched steps, in the prompt store at
 * [store_at, store_at + n). Stream-ordered. */
extern "C" void dec_save_prompt(void *handle, int b, int at, int store_at, int n) {
    prompt_copy((Decoder *)handle, b, at, store_at, n, true);
}

/* Fill slot b's cache at positions [at, at + n) from the prompt store at [store_at, store_at + n). Stream-ordered. */
extern "C" void dec_load_prompt(void *handle, int b, int at, int store_at, int n) {
    prompt_copy((Decoder *)handle, b, at, store_at, n, false);
}

/* The prompt store's positions (dec_create: the requested, at most what fit after the slots). */
extern "C" int dec_store(void *handle) { return ((Decoder *)handle)->store; }

/* The pinned host masks [B][vocab / WORD_BITS], bit t % WORD_BITS of word t / WORD_BITS of row b set iff slot b may
 * choose token t; the caller writes a masked slot's row before dec_set_active, after the previous step completed. */
extern "C" uint32_t *dec_masks(void *handle) { return ((Decoder *)handle)->host_allow; }

/* The active slots and the masked slots, whose rows of dec_masks restrict their next token choices (the rows of the
 * lowest to the highest masked slot are copied): slots[2][words], words = ceil(dec_batch / MASK_BITS), slot b at bit
 * b % MASK_BITS of word b / MASK_BITS. Stream-ordered. */
extern "C" void dec_set_active(void *handle, const unsigned long long *slots) {
    Decoder *d = (Decoder *)handle;
    const int words = (d->B + MASK_BITS - 1) / MASK_BITS;
    const size_t row = words * sizeof(unsigned long long);
    cudaMemcpy2DAsync(d->active, MASK_WORDS * sizeof(unsigned long long), slots, row, row, 2, cudaMemcpyHostToDevice,
                      d->stream);
    const unsigned long long *masked = slots + words;
    int lo = -1, hi = -1;
    for (int w = 0; w < words; w++)
        if (masked[w]) {
            if (lo < 0) lo = w * MASK_BITS + __builtin_ctzll(masked[w]);
            hi = w * MASK_BITS + MASK_BITS - 1 - __builtin_clzll(masked[w]);
        }
    if (lo >= 0) {
        const size_t n = d->vocab / WORD_BITS;
        cudaMemcpyAsync(d->allow + lo * n, d->host_allow + lo * n, (hi + 1 - lo) * n * sizeof(uint32_t),
                        cudaMemcpyHostToDevice, d->stream);
    }
}

/* Launch one greedy step (asynchronous): every slot takes its logits' argmax; dec_tokens returns its tokens. */
extern "C" void dec_launch(void *handle) {
    Decoder *d = (Decoder *)handle;
    cudaGraphLaunch(d->step, d->stream);
}

/* Launch one step's body (asynchronous): everything before the token choice, which dec_launch_head then launches. */
extern "C" void dec_launch_body(void *handle) {
    Decoder *d = (Decoder *)handle;
    cudaGraphLaunch(d->body, d->stream);
}

/* Launch one step's head (asynchronous) after its body: every slot chooses by its mask (if dec_set_active marks it
 * masked) and temperature; dec_tokens returns its tokens. */
extern "C" void dec_launch_head(void *handle) {
    Decoder *d = (Decoder *)handle;
    cudaGraphLaunch(d->head, d->stream);
}

/* Wait for the launched step; out[b]: the token slot b took (stale for inactive slots). */
extern "C" void dec_tokens(void *handle, int32_t *out) {
    Decoder *d = (Decoder *)handle;
    CUDA_CHECK(cudaStreamSynchronize(d->stream));
    for (int b = 0; b < d->B; b++) out[b] = d->host_tokens[b];
}
