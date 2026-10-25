/*
 * Decoding of a nanochat CausalLM (model/modeling.py) for B independent sequences (slots): each step advances every
 * active slot by one token, chosen greedily or sampled (optionally from its nucleus), optionally restricted to a
 * per-slot set of allowed tokens (inference/decoder.py).
 *
 * A many-slot step is bound by its KV reads (each active slot reads its window's keys and values in every layer) and
 * its GEMMs; a few-slot step by weight reads and kernel launches. The step's kernels replay as one CUDA graph, each a
 * programmatic dependent of the one before (launch).
 * Positions, input tokens, KV caches, the active-slot mask and each slot's sampling parameters and allowed tokens live
 * in device memory. A slot starts at its prompt's last token (dec_set_prompt), the positions before it cached: run as
 * rows of a prefill (dec_prefill: up to MAX_BATCH (slot, position, token) rows of any prompts through the step's body
 * at once, each row computed as a step would, the rows' K and V cached before any row's attention), or copied
 * from the prompt store, which keeps a prompt's K and V (they depend on its tokens alone) for a later request with
 * the same prompt (dec_save_prompt, dec_load_prompt; SGLang's RadixAttention, arXiv 2312.07104, reuses prompt
 * prefixes likewise); inactive slots are skipped by every kernel.
 *
 * The step is captured whole (dec_launch), the greedy argmax for every slot, and as its body, everything before the
 * lm_head GEMM, and a head that chooses by each slot's allowed tokens, temperature and top_p (dec_launch_body,
 * dec_launch_head): the host supplies the allowed tokens, which follow from each slot's previous token, while the
 * body runs (decoder.py). The head also chooses each slot's best allowed token but one lazy row (BOS), so the host
 * confirms a chosen lazy row only then and otherwise takes that alternative (dec_take_alt): the same choice as
 * excluding the row beforehand (greedy, or Gumbel-max sampling, whose noise is fixed per row). The whole step's
 * GEMMs compile without the choice (the GEMM kernels' CHOICE).
 *
 * Each step (modeling.py's forward at one position; C width, H heads of HEAD_DIM, x0 the normed embedding):
 *   k_norm    x0 = norm(wte[token])
 *   per layer i:
 *     k_norm  xm = resid_lambdas[i] x + x0_lambdas[i] x0 (x = x0 before layer 0), h = norm(xm)
 *     k_wgmma [q k v] = h [Wq Wk Wv]^T
 *     k_prologue VE layers: v += 2 sigmoid(W_g[head] . h) value_embeds[i][token] (per head, fp32);
 *             q, k = rope(norm_D(q) q_gain), rope(norm_D(k) k_gain) (qk_row.cuh, as qk_prologue.cu); k, v cached as
 *             e4m3 with a scale per position and head
 *     k_attn  y = softmax(q K^T / sqrt(HEAD_DIM)) V over the layer's window (FlashAttention's window_size left)
 *     k_wgmma xo = xm + y Wo^T
 *     k_norm  hn = norm(xo)
 *     k_wgmma u = relu(hn Wfc^T)^2
 *     k_wgmma x = xo + u Wproj^T
 *   k_norm    hn = norm(x)
 *   k_wgmma   the token choice from s = hn W_lm^T over the slot's allowed tokens (all if it has none): argmax s at
 *             temperature 0, else argmax s / T + Gumbel noise (sampling softmax(s / T), gumbel), by atomicMax on a
 *             packed key; a nucleus-sampled slot's s are stored instead (head only)
 *   k_nucleus a nucleus-sampled slot's choice: that argmax over the fewest most likely allowed tokens holding top_p of
 *             softmax(s / T) (head only)
 *   k_commit  the choice (and its alternative) becomes each active slot's next input
 *
 * Numerics follow model/modeling.py's forward op by op: bf16 copies of the fp32 matrices and tables; fp32 rotary
 * table, q/k norm weights and value-embedding gate (no contraction), the gate's mix rounded once to bf16; resid/x0
 * lambdas as bf16 values on bf16 tensors (PyTorch's 0-dim fp32 x bf16 promotion), each product and sum rounded; ReLU^2
 * from the fp32 accumulators as QuACK's gemm_act relu_sq (quack/activation.py), one bf16 rounding; F.rms_norm's fp32
 * epsilon; the choice over fp32 logits. The KV cache departs from it: keys and values are e4m3 (1 byte, a scale per
 * position and head: amax / 448), half the bytes a step reads; scores and the softmax stay fp32. A slot's arithmetic
 * does not depend on the other slots (each is its own GEMM column and attention block, its cache scales its own), so
 * its tokens do not depend on the batch. A sample's noise depends only on its request's seed and the generated
 * position.
 *
 * Launch shapes: a GEMM runs a (slot tile, row tile) block of a TMA producer and a wgmma warpgroup (k_wgmma);
 * attention a (head, slot) block of HEAD_DIM threads streaming the slot's window (k_attn).
 */
#include <initializer_list>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <utility>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <curand_kernel.h>

#include "qk_row.cuh"

constexpr int LANE_DIMS = HEAD_DIM / WARP;         /* a lane's dims of a query, key or value */
constexpr int ATTN_WARPS = HEAD_DIM / WARP;        /* an attention block: a thread per head dim */
/* slots per decoder and rows per prefill launch: k_commit runs the slots as one block of MAX_BATCH threads, the most a
 * block holds (CUDA C++ Programming Guide, "Thread Hierarchy") */
constexpr int MAX_BATCH = 1024;
constexpr int MASK_BITS = CHAR_BIT * sizeof(unsigned long long);  /* slots per word of a slot mask */
constexpr int MASK_WORDS = MAX_BATCH / MASK_BITS;                 /* words of a slot mask */
/* the narrow graphs' slots (Decoder::narrow); measured at d24 (prof steps, 256-token prompts) the fastest or within
 * 0.04 ms of it at 1, 64, 128 and 256 active slots against 64, 128, 512 and no narrow graphs (1 slot: 1.29 ms, none
 * 1.46 ms; 256 slots: 2.81 ms, none 2.97 ms) */
constexpr int SMALL_BATCH = 256;
static_assert(MAX_BATCH % MASK_BITS == 0, "slot masks are whole words");
/* mma.sync m16n8k16 (PTX ISA "Matrix Fragments for mma.m16n8k16"): lanes form MMA_N groups (groupID = lane / GROUP) of
 * GROUP threads (threadID_in_group = lane % GROUP), each holding ACC fp32 accumulators */
constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 16;
constexpr int GROUP = WARP / MMA_N, ACC = MMA_M * MMA_N / WARP;
/* k_norm: slots (a warp each) per block, the fewest that let an SM fill its threads within its resident-block limit
 * (compute capability 9.0: 2,048 threads, 32 blocks; CUDA C++ Programming Guide, "Technical Specifications") */
constexpr int NORM_WARPS = 2048 / 32 / WARP;
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
/* GEMM of B slots on the tensor cores: out[b][r] = W[r] . x[b] (bf16 operands, fp32 accumulators; k_wgmma). RELU2:
 * out = bf16(relu(s) s) of the fp32 dot product s (QuACK gemm_act relu_sq). ARGMAX: each warp's maximum key per slot
 * over its rows (an exact integer maximum, so any order), then one atomicMax. */

typedef struct {
    const bf16 *w;               /* [rows][cols] */
    int rows, cols;
    const bf16 *in;              /* [B][cols] */
    int epi;                     /* epilogue: what is done with each output */
    bf16 *out;                   /* [B][rows]: STORE/RESID/RELU2 output */
    const bf16 *res;             /* [B][rows]: RESID: residual added */
    unsigned long long *argmax;  /* [B] ARGMAX: packed (ordered score, ~row) maximum */
    const unsigned long long *active;  /* [2][MASK_WORDS]: the active slots' mask, then the masked slots' (CHOICE) */
    int B;
} Gemm;

/* The per-slot inputs of a CHOICE ARGMAX and of k_nucleus, a kernel argument of their own: growing Gemm changes the
 * compiled GEMMs. */
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
    unsigned long long *alt;         /* [B]: the choice's key over the allowed rows but lazy (a slot not nucleus-
                                      * sampled; 0 none): the host's choice when it rejects the lazy row (dec_tokens) */
    int lazy;                        /* the row whose choice the host confirms after the step (BOS: an end the host
                                      * checks only once chosen) */
} Choice;

/* 16 bytes global -> shared without a register round trip (cp.async, PTX ISA "Data Movement: cp.async"): all of a
 * thread's copies are in flight at once. */
static __device__ __forceinline__ void copy16(void *dst, const void *src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src));
}

static __device__ __forceinline__ void copies_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }

/* Wait until at most n of this thread's committed copy groups are pending. */
template <int N> static __device__ __forceinline__ void copies_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

/* Programmatic dependent launch (CUDA C++ Programming Guide, "Programmatic Dependent Launch and Synchronization"): the
 * step's kernels launch while their predecessor runs (launch); each first waits until its predecessor has completed
 * and flushed its writes (griddepcontrol.wait; every kernel waits before anything else, early exits included, so a
 * completed kernel implies every earlier one completed), then lets its successor launch (launch_dependents: its
 * blocks are scheduled once every block of this grid has started). */
static __device__ __forceinline__ void pdl_start() {
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;\n" :::);
}

/* kernel<<<grid, block, smem, stream>>>(args...) as a programmatic dependent of the stream's previous kernel. */
template <typename... P, typename... A>
static void launch(void (*kernel)(P...), dim3 grid, dim3 block, size_t smem, cudaStream_t stream, A... args) {
    cudaLaunchAttribute attr;
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid; cfg.blockDim = block; cfg.dynamicSmemBytes = smem; cfg.stream = stream;
    cfg.attrs = &attr; cfg.numAttrs = 1;
    CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, (P)args...));
}

/* One output of a GEMM, slot b's row `row` from its fp32 dot product s: STORE/RESID/RELU2 write it; ARGMAX
 * folds its key into the lane's best (and, with CHOICE, alt) for the slot. */
template <bool CHOICE> static __device__ __forceinline__ void epilogue(const Gemm &a, const Choice &c, int row, int b,
                                                                      float s, unsigned long long &best,
                                                                      unsigned long long &alt) {
    const size_t at = (size_t)b * a.rows + row;
    if (a.epi == EPI_RELU2) {
        a.out[at] = __float2bfloat16_rn(__fmul_rn(fmaxf(s, 0.f), s));
        return;
    }
    const float y = bfr(s);  /* the GEMM's bf16 output */
    if (a.epi == EPI_STORE) {
        a.out[at] = __float2bfloat16_rn(y);
        return;
    }
    if (a.epi == EPI_RESID) {
        a.out[at] = __float2bfloat16_rn(__bfloat162float(a.res[at]) + y);
        return;
    }
    /* EPI_ARGMAX: largest score, lowest row among exact ties (choice_key): the fp32 logit, or with CHOICE over a masked
     * slot's allowed rows, at temperature T > 0 logit / T + gumbel; a nucleus-sampled slot's logits go to k_nucleus */
    float score = s;
    if constexpr (CHOICE) {
        const bool allowed = !live(a.active + MASK_WORDS, b) ||
            ((c.allow[(size_t)b * (a.rows / WORD_BITS) + row / WORD_BITS] >> (row % WORD_BITS)) & 1u);
        const float z = c.softcap > 0.f && c.temperature[b] > 0.f
            ? __fmul_rn(c.softcap, tanhf(__fdiv_rn(s, c.softcap))) : s;
        if (c.temperature[b] > 0.f && c.top_p[b] < 1.f) {
            c.logits[at] = allowed ? z : -INFINITY;
            return;
        }
        if (!allowed) return;
        if (c.temperature[b] > 0.f)
            score = __fadd_rn(__fdiv_rn(z, c.temperature[b]), gumbel(c.seed[b], c.pos[b] - c.first[b], row));
    }
    const unsigned long long key = choice_key(score, row);
    best = key > best ? key : best;
    if constexpr (CHOICE)
        if (row != c.lazy) alt = key > alt ? key : alt;
}

/* ARGMAX: the lanes' best keys (and alt) for the slots of their accumulator columns, b0 + 2t + p, maximized over the
 * lanes sharing t, then one atomicMax per slot. */
template <bool CHOICE> static __device__ __forceinline__ void argmax_fold(const Gemm &a, const Choice &c, int b0,
                                                                         unsigned long long *best,
                                                                         unsigned long long *alt) {
    const int lane = threadIdx.x % WARP, g = lane / GROUP, t = lane % GROUP;
    for (int o = GROUP; o < WARP; o <<= 1)  /* over g: the lanes sharing t */
        for (int p = 0; p < 2; p++) {
            const unsigned long long other = __shfl_xor_sync(0xffffffffu, best[p], o);
            best[p] = other > best[p] ? other : best[p];
            if constexpr (CHOICE) {
                const unsigned long long other_alt = __shfl_xor_sync(0xffffffffu, alt[p], o);
                alt[p] = other_alt > alt[p] ? other_alt : alt[p];
            }
        }
    if (g == 0)
        for (int p = 0; p < 2; p++) {
            if (best[p]) atomicMax(a.argmax + b0 + t * 2 + p, best[p]);
            if constexpr (CHOICE)
                if (alt[p]) atomicMax(c.alt + b0 + t * 2 + p, alt[p]);
        }
}

/* GEMM of many slots by wgmma (sm_90a; PTX ISA "Asynchronous Warpgroup Level Matrix Multiply-Accumulate"), the
 * slot GEMM's operands and epilogues: a block computes a WG_M-row tile of W (wgmma's M, one warpgroup's rows) against a
 * WG_N-slot tile of inputs (one word of the active mask). A producer warp streams WG_BK-wide K slices of both tiles
 * into a WG_STAGES-deep shared-memory ring by TMA (cp.async.bulk.tensor, 128-byte swizzle; full/empty mbarriers per
 * stage), and a consumer warpgroup multiplies them (m64n64k16 from the swizzled tiles' descriptors, fp32
 * accumulators), one slice's MMAs in flight while the next issue, releasing a stage once its MMAs completed. The
 * accumulators' layout is mma m16n8's per warp (16 rows) and n8 chunk (PTX ISA "Register Fragments: wgmma
 * .m64nNk16"), whose lanes epilogue and argmax_fold address. Slot tiles without an active slot exit. CHOICE: ARGMAX
 * honours each slot's allowed tokens and temperature (c; the head graph); without it, the logits' argmax (the
 * whole-step graph), so greedy steps run the GEMM compiled without the choice. Measured at d24 (1,024 slots,
 * 256-token prompts, 512 steps): 8.27 ms per step, against 8.33 ms dispatching 128-row x 128-slot tiles where they
 * cover every SM; WG_STAGES 4 against 8.82, 8.44, 8.33 and 9.22 ms at 2, 3, 6 and 8 stages. */
constexpr int WG_THREADS = 128, WG_M = 64, WG_N = MASK_BITS, WG_BK = 64, WG_STAGES = 4;
static_assert(WG_BK * sizeof(bf16) == 128, "a K slice row is one 128-byte swizzle span");
constexpr uint32_t WG_A_BYTES = WG_M * WG_BK * sizeof(bf16), WG_B_BYTES = WG_N * WG_BK * sizeof(bf16);
/* k_wgmma's dynamic shared memory: the ring's stages, its full and empty barriers, and 1024 bytes to align it (a
 * 128-byte swizzle atom's 8 rows: PTX ISA "Shared Memory Matrix Layout") */
constexpr size_t WG_SMEM = 1024 + (size_t)WG_STAGES * (WG_A_BYTES + WG_B_BYTES) + 2 * WG_STAGES * sizeof(uint64_t);

static __device__ __forceinline__ uint32_t smem_addr(const void *p) { return (uint32_t)__cvta_generic_to_shared(p); }

static __device__ __forceinline__ void mbar_init(uint64_t *bar, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(smem_addr(bar)), "r"(count));
}

static __device__ __forceinline__ void mbar_expect(uint64_t *bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(smem_addr(bar)), "r"(bytes)
                 : "memory");
}

static __device__ __forceinline__ void mbar_arrive(uint64_t *bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(smem_addr(bar)) : "memory");
}

/* Wait for the barrier's phase of the given parity to complete. */
static __device__ __forceinline__ void mbar_wait(uint64_t *bar, uint32_t parity) {
    uint32_t done = 0;
    while (!done)
        asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                     "selp.u32 %0, 1, 0, p;\n}\n"
                     : "=r"(done) : "r"(smem_addr(bar)), "r"(parity) : "memory");
}

/* The tensor map's box at (c0 innermost, c1) to shared memory, completing bytes on bar. */
static __device__ __forceinline__ void tma_load(void *dst, const CUtensorMap *map, uint64_t *bar, int c0, int c1) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%3, %4}], "
                 "[%2];\n"
                 :: "r"(smem_addr(dst)), "l"(map), "r"(smem_addr(bar)), "r"(c0), "r"(c1) : "memory");
}

/* wgmma's descriptor of a K-major tile in shared memory, 128-byte swizzled rows of WG_BK (8-row atoms 1024 bytes
 * apart; the tile 1024-byte aligned, a k16 step 32 bytes further). */
static __device__ __forceinline__ uint64_t wgmma_desc(const void *p) {
    return (uint64_t)((smem_addr(p) & 0x3FFFF) >> 4) | (uint64_t)1 << 16 | (uint64_t)(1024 >> 4) << 32 |
           (uint64_t)1 << 62;
}

/* d (m64n64, fp32) = A B^T + (accumulate ? d : 0) over k16 from the descriptors' tiles. */
static __device__ __forceinline__ void wgmma_k16(float *d, uint64_t da, uint64_t db, int accumulate) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, p, 1, 1, 0, 0;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
          "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "l"(da), "l"(db), "r"(accumulate));
}
static_assert(WG_M == 64 && WG_N == 64, "wgmma_k16 is m64n64k16");

template <bool CHOICE>
__global__ void __launch_bounds__(2 * WG_THREADS) k_wgmma(const __grid_constant__ CUtensorMap tw,
                                                          const __grid_constant__ CUtensorMap tx, Gemm a, Choice c) {
    pdl_start();
    constexpr int ACCS = WG_M * WG_N / WG_THREADS;
    extern __shared__ __align__(1024) unsigned char wg_buf[];
    unsigned char *base = reinterpret_cast<unsigned char *>(((uintptr_t)wg_buf + 1023) & ~(uintptr_t)1023);
    bf16 *sa = reinterpret_cast<bf16 *>(base), *sb = reinterpret_cast<bf16 *>(base + WG_STAGES * WG_A_BYTES);
    uint64_t *full = reinterpret_cast<uint64_t *>(base + WG_STAGES * (WG_A_BYTES + WG_B_BYTES));
    uint64_t *empty = full + WG_STAGES;
    const int n0 = blockIdx.x * WG_N, row0 = blockIdx.y * WG_M;
    const unsigned long long live_slots = a.active[blockIdx.x];
    if (!live_slots) return;
    const int kt = a.cols / WG_BK;
    if (threadIdx.x == 0) {
        for (int s = 0; s < WG_STAGES; s++) {
            mbar_init(&full[s], 1);
            mbar_init(&empty[s], 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();
    if (threadIdx.x >= WG_THREADS) {  /* the producer: one thread issues every slice's loads */
        if (threadIdx.x == WG_THREADS)
            for (int kb = 0; kb < kt; kb++) {
                const int s = kb % WG_STAGES;
                mbar_wait(&empty[s], ((kb / WG_STAGES) & 1) ^ 1);  /* the first round passes */
                mbar_expect(&full[s], WG_A_BYTES + WG_B_BYTES);
                tma_load(sa + s * WG_M * WG_BK, &tw, &full[s], kb * WG_BK, row0);
                tma_load(sb + s * WG_N * WG_BK, &tx, &full[s], kb * WG_BK, n0);
            }
        return;
    }
    float acc[ACCS];  /* defined by the first slice's MMAs (accumulate 0) */
    for (int kb = 0; kb < kt; kb++) {
        const int s = kb % WG_STAGES;
        mbar_wait(&full[s], (kb / WG_STAGES) & 1);
        const bf16 *ta = sa + s * WG_M * WG_BK, *tb = sb + s * WG_N * WG_BK;
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
        for (int k = 0; k < WG_BK / MMA_K; k++)
            wgmma_k16(acc, wgmma_desc(ta + k * MMA_K), wgmma_desc(tb + k * MMA_K), kb | k);
        asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");  /* the previous slice's MMAs completed */
        if (kb > 0 && threadIdx.x == 0) mbar_arrive(&empty[(kb - 1) % WG_STAGES]);
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    /* the accumulators read only after the wait (CUTLASS's warpgroup_fence_operand) */
#pragma unroll
    for (int i = 0; i < ACCS; i++) asm volatile("" : "+f"(acc[i]) :: "memory");
    const int lane = threadIdx.x % WARP, g = lane / GROUP, t = lane % GROUP, w = threadIdx.x / WARP;
    const int r0 = row0 + w * MMA_M + g;
#pragma unroll
    for (int n = 0; n < WG_N / MMA_N; n++) {
        const unsigned long long live_bits = live_slots >> (n * MMA_N);
        unsigned long long best[2] = {0ull, 0ull}, alt[2] = {0ull, 0ull};
#pragma unroll
        for (int i = 0; i < ACC; i++) {
            const int row = r0 + i / 2 * (MMA_M / 2), j = t * 2 + i % 2;
            if (row >= a.rows || !((live_bits >> j) & 1ull)) continue;
            epilogue<CHOICE>(a, c, row, n0 + n * MMA_N + j, acc[n * ACC + i], best[i % 2], alt[i % 2]);
        }
        if (a.epi == EPI_ARGMAX) argmax_fold<CHOICE>(a, c, n0 + n * MMA_N, best, alt);
    }
}

/* A nucleus-sampled slot's choice (top-p sampling: arXiv 1904.09751, "The Curious Case of Neural Text Degeneration",
 * §3.1): Gumbel-max at temperature T, as the GEMM's epilogue, over the fewest most likely allowed tokens whose
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
    pdl_start();
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

/* ---------------------------------------------------------------------------------------------------------------- */
/* RMS norm of each active slot's input row, a warp per slot, the sum of squares over K in lane-strided order then
 * warp_sum, r = rsqrt(mean + FLT_EPSILON): NORM out = bf16(in r) of the slot's row of in; EMBED the same of in's row
 * tokens[b] (x0 = rms_norm(wte[token])); RESID_NORM mixed = bf16(bf16(l1 in) + bf16(l2 in2)) (the resid and x0
 * lambdas, bf16 values), out = bf16(mixed r). */
__global__ void k_norm(int kind, const bf16 *in, const bf16 *in2, const int32_t *tokens, float l1, float l2,
                       bf16 *mixed, bf16 *out, const unsigned long long *active, int B, int K) {
    pdl_start();
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
/* Attention: a (head, slot) block of HEAD_DIM threads per row.
 *
 * k_prologue computes each row's q, k, v and caches k, v (e4m3, a scale per position and head); then each k_attn
 * block streams its row's window begin..p-1 through shared memory in ATTN_RUN-key runs (cp.async, double-buffered).
 * Each warp scores MMA_M keys of a run by f16 MMAs (e4m3 keys and bf16-valued q are exact in f16; fp32
 * accumulators), updates its online softmax (m, l) once per run and adds the scaled values on the
 * CUDA cores, a lane's LANE_DIMS dims; the warps merge, add the current key and value, and write acc / l rounded to
 * bf16 (FlashAttention's output). One block per window: at d24 a 1,024-slot step's 6,144 (head, slot) blocks fill the
 * SMs, and a 1-slot step took 1.29 ms against 3.16 ms for the per-slot-group GEMV decoder it replaced. */

typedef struct {
    const bf16 *qkv, *h, *ve;  /* [B][3C], [B][C] (the normed input), the layer's value-embedding table or NULL */
    const float *ve_gate;      /* [H][C] fp32: the value-embedding gate's weights */
    const float *q_gain, *k_gain;  /* [HEAD_DIM] fp32, shared by the heads */
    const float *cos, *sin;    /* the rotary table [context][HALF], fp32 */
    uint8_t *kc, *vc;          /* the layer's FP8 (e4m3) cache of slot 0, [max_len][C]; slot b's at + b kv_slot */
    size_t kv_slot;
    float *ksc, *vsc;          /* the layer's cache scales of slot 0, [H][max_len] (a row's value: e4m3 x scale);
                                * slot b's at + b sc_slot */
    size_t sc_slot;
    int max_len;
    bf16 *out;                 /* [B][C] */
    const int32_t *tokens, *pos;  /* [B]: each slot's input token and its position */
    const unsigned long long *active;
    int C, H, window;
    const int32_t *cache_slot; /* [B]: a prefill row's slot, whose cache row b reads and writes; NULL: slot b */
    float *qkvf;               /* [B][3][C]: each row's q, k, v at its position (k_prologue), fp32 holding bf16 */
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

/* The low two e4m3 bytes of x as an f16x2 (exact). */
static __device__ __forceinline__ uint32_t fp8x2_f16x2(uint32_t x) {
    const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(x & 0xffffu), __NV_E4M3);
    return (uint32_t)h.x | (uint32_t)h.y << 16;
}

/* d += A B by mma.sync m16n8k16 with f16 operands, fp32 accumulators (fragments as mma16816's). */
static __device__ __forceinline__ void mma16816_f16(float *d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                    uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

/* A lane's LANE_DIMS dims of an e4m3 value row in shared memory, as fp32 (unscaled): one 4-byte load. */
static_assert(LANE_DIMS == 4, "a lane reads its dims of a cached row as one uint32");
static __device__ __forceinline__ void load_dims(const uint8_t *row, float *x) {
    const uint32_t u = *reinterpret_cast<const uint32_t *>(row), lo = fp8x2_f16x2(u), hi = fp8x2_f16x2(u >> 16);
    const float2 a = __half22float2(*reinterpret_cast<const __half2 *>(&lo));
    const float2 c = __half22float2(*reinterpret_cast<const __half2 *>(&hi));
    x[0] = a.x; x[1] = a.y; x[2] = c.x; x[3] = c.y;
}

/* q . k / sqrt(HEAD_DIM) (FlashAttention's default softmax scale) to every lane: lane products in order, warp_sum. */
static __device__ __forceinline__ float key_score(const float *q, const float *k) {
    float d = __fmul_rn(q[0], k[0]);
    for (int e = 1; e < LANE_DIMS; e++) d = __fmaf_rn(q[e], k[e], d);
    return __fmul_rn(warp_sum(d), rsqrtf((float)HEAD_DIM));
}

/* k_attn's keys per cp.async run (a commit group, double-buffered): one MMA_M tile per warp, the least a warp's MMA
 * scores; measured at d24 (1,024 slots, 256-token prompts, 512 steps) 8.31 ms per step against 8.80 ms for two tiles
 * per warp and 15.45 ms for four (under the earlier tile dispatch) */
constexpr int ATTN_RUN = ATTN_WARPS * MMA_M;
constexpr int ROW_VECS = HEAD_DIM / (int)sizeof(uint4);  /* 16-byte vectors per e4m3 key row */
constexpr float FP8_MAX = 448.f;  /* the largest finite e4m3 */
/* one run's buffer: its keys and values (e4m3), then their scales */
constexpr size_t ATTN_BUF = 2 * (size_t)ATTN_RUN * HEAD_DIM + 2 * ATTN_RUN * sizeof(float);

constexpr size_t ATTN_SMEM = 2 * ATTN_BUF;  /* k_attn's dynamic shared memory: two runs' buffers */

/* 4 bytes global -> shared (cp.async.ca: 4-byte copies take the L1 path). */
static __device__ __forceinline__ void copy4(void *dst, const void *src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                 :: "r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src));
}

/* x / scale as e4m3 (saturating). */
static __device__ __forceinline__ uint8_t to_fp8(float x, float inv) {
    return (uint8_t)__nv_cvt_float_to_fp8(x * inv, __NV_SATFINITE, __NV_E4M3);
}

/* exp(m - top); 0 for an empty state (m = top = -inf would give NaN) */
static __device__ __forceinline__ float rescale(float m, float top) { return m == -INFINITY ? 0.f : expf(m - top); }

/* The attention's prologue, once per (head, row) before k_attn: the row's q, k, v at its position
 * (attn_prologue) to qkvf, its k and v to its slot's cache as e4m3 with a scale per row and head (amax / FP8_MAX).
 * (head, row) blocks of HEAD_DIM threads. */
__global__ void __launch_bounds__(HEAD_DIM) k_prologue(Attn a) {
    pdl_start();
    __shared__ float q[HEAD_DIM], k[HEAD_DIM], v[HEAD_DIM], scratch[ATTN_WARPS];
    const int head = blockIdx.x, b = blockIdx.y, tid = threadIdx.x;
    if (!live(a.active, b)) return;
    const int p = a.pos[b], cs = a.cache_slot ? a.cache_slot[b] : b;
    attn_prologue(a, b, head, p, q, k, v);
    float *out = a.qkvf + (size_t)b * 3 * a.C + head * HEAD_DIM + tid;
    out[0] = q[tid];
    out[a.C] = k[tid];
    out[2 * a.C] = v[tid];
    const float ka = block_max(fabsf(k[tid]), scratch), va = block_max(fabsf(v[tid]), scratch);
    const float ks = ka > 0.f ? ka / FP8_MAX : 1.f, vs = va > 0.f ? va / FP8_MAX : 1.f;
    const size_t at = (size_t)cs * a.kv_slot + (size_t)p * a.C + head * HEAD_DIM + tid;
    a.kc[at] = to_fp8(k[tid], 1.f / ks);
    a.vc[at] = to_fp8(v[tid], 1.f / vs);
    if (tid == 0) {
        const size_t s = (size_t)cs * a.sc_slot + (size_t)head * a.max_len + p;
        a.ksc[s] = ks;
        a.vsc[s] = vs;
    }
}

__global__ void __launch_bounds__(HEAD_DIM) k_attn(Attn a) {
    pdl_start();
    extern __shared__ __align__(16) unsigned char attn_buf[];
    __shared__ float q[HEAD_DIM], k[HEAD_DIM], v[HEAD_DIM];
    __shared__ float wm[ATTN_WARPS], wl[ATTN_WARPS], wacc[ATTN_WARPS][HEAD_DIM];
    const int head = blockIdx.x, b = blockIdx.y, tid = threadIdx.x, warp = tid / WARP, lane = tid % WARP;
    if (!live(a.active, b)) return;
    const int p = a.pos[b];
    const int begin = p > a.window ? p - a.window : 0, n = p - begin;  /* the window's cached keys */
    const int runs = (n + ATTN_RUN - 1) / ATTN_RUN;
    const int cs = a.cache_slot ? a.cache_slot[b] : b;  /* the row's cache */
    const uint8_t *kc = a.kc + (size_t)cs * a.kv_slot + (size_t)begin * a.C + head * HEAD_DIM;
    const uint8_t *vc = a.vc + (size_t)cs * a.kv_slot + (size_t)begin * a.C + head * HEAD_DIM;
    const float *ksc = a.ksc + (size_t)cs * a.sc_slot + (size_t)head * a.max_len + begin;
    const float *vsc = a.vsc + (size_t)cs * a.sc_slot + (size_t)head * a.max_len + begin;
    /* run r's keys (16-byte chunks swizzled by the row's low 3 bits, so the QK fragments' loads of 8 rows hit distinct
     * banks), values and scales to buffer r % 2, one commit group */
    auto load_run = [&](int r) {
        uint8_t *ks = attn_buf + (r % 2) * ATTN_BUF, *vs = ks + ATTN_RUN * HEAD_DIM;
        float *kss = reinterpret_cast<float *>(vs + ATTN_RUN * HEAD_DIM), *vss = kss + ATTN_RUN;
        const int j0 = r * ATTN_RUN, m = min(ATTN_RUN, n - j0);
        for (int i = tid; i < m * ROW_VECS; i += HEAD_DIM) {
            const int j = i / ROW_VECS, c = i % ROW_VECS;
            copy16(ks + j * HEAD_DIM + (c ^ (j & 7)) * (int)sizeof(uint4), kc + (size_t)(j0 + j) * a.C + c * 16);
            copy16(vs + j * HEAD_DIM + c * (int)sizeof(uint4), vc + (size_t)(j0 + j) * a.C + c * 16);
        }
        for (int i = tid; i < m; i += HEAD_DIM) {
            copy4(kss + i, ksc + j0 + i);
            copy4(vss + i, vsc + j0 + i);
        }
        copies_commit();
    };
    if (runs > 0) load_run(0);
    const float *in = a.qkvf + (size_t)b * 3 * a.C + head * HEAD_DIM + tid;
    q[tid] = in[0];
    k[tid] = in[a.C];
    v[tid] = in[2 * a.C];
    __syncthreads();

    /* q as the QK MMAs' B fragments (column 0, lanes g = 0; f16 holds bf16 values exactly): chunk ch of 32 dims,
     * lane (g, t) the dims 32 ch + 8 t .. + 7 as (0, 1), (2, 3) for the first k16 MMA and (4, 5), (6, 7) for the
     * second (A and B map MMA k to physical dims alike, so the product is q . k) */
    const int g = lane / GROUP, t = lane % GROUP;
    uint32_t qb[HEAD_DIM / 32][4];
    for (int ch = 0; ch < HEAD_DIM / 32; ch++)
        for (int i = 0; i < 4; i++) {
            const __half2 h = __floats2half2_rn(q[32 * ch + 8 * t + 2 * i], q[32 * ch + 8 * t + 2 * i + 1]);
            qb[ch][i] = g == 0 ? *reinterpret_cast<const uint32_t *>(&h) : 0u;
        }
    float acc[LANE_DIMS], m = -INFINITY, l = 0.f;
    for (int e = 0; e < LANE_DIMS; e++) acc[e] = 0.f;
    const float scale = rsqrtf((float)HEAD_DIM);
    for (int r = 0; r < runs; r++) {
        if (r + 1 < runs) {
            load_run(r + 1);
            copies_wait<1>();
        } else {
            copies_wait<0>();
        }
        __syncthreads();  /* run r landed for every thread */
        const uint8_t *ks = attn_buf + (r % 2) * ATTN_BUF, *vs = ks + ATTN_RUN * HEAD_DIM;
        const float *kss = reinterpret_cast<const float *>(vs + ATTN_RUN * HEAD_DIM), *vss = kss + ATTN_RUN;
        const int m_run = min(ATTN_RUN, n - r * ATTN_RUN);
        /* the warp's MMA_M keys of the run: scores by MMAs (key rows g, g + 8 of the tile), then the values on the
         * CUDA cores, a lane's LANE_DIMS dims */
        const int j0 = warp * MMA_M, count = m_run - j0;
        if (count > 0) {
            float c4[ACC] = {0.f, 0.f, 0.f, 0.f};
            const int j = j0 + g;  /* rows j and j + 8 share the swizzle (j & 7) */
#pragma unroll
            for (int ch = 0; ch < HEAD_DIM / 32; ch++) {
                const int at = ((2 * ch + t / 2) ^ (j & 7)) * (int)sizeof(uint4) + 8 * (t % 2);
                const uint2 r0 = *reinterpret_cast<const uint2 *>(ks + j * HEAD_DIM + at);
                const uint2 r1 = *reinterpret_cast<const uint2 *>(ks + (j + MMA_M / 2) * HEAD_DIM + at);
                mma16816_f16(c4, fp8x2_f16x2(r0.x), fp8x2_f16x2(r1.x), fp8x2_f16x2(r0.x >> 16),
                             fp8x2_f16x2(r1.x >> 16), qb[ch][0], qb[ch][1]);
                mma16816_f16(c4, fp8x2_f16x2(r0.y), fp8x2_f16x2(r1.y), fp8x2_f16x2(r0.y >> 16),
                             fp8x2_f16x2(r1.y >> 16), qb[ch][2], qb[ch][3]);
            }
            /* column 0 (lanes t = 0) holds the scores of keys j, j + 8; to every lane of the group */
            float s0 = __shfl_sync(0xffffffffu, c4[0], lane & ~(GROUP - 1));
            float s1 = __shfl_sync(0xffffffffu, c4[2], lane & ~(GROUP - 1));
            s0 = g < count ? s0 * scale * kss[j] : -INFINITY;  /* keys past the run: none */
            s1 = g + MMA_M / 2 < count ? s1 * scale * kss[j + MMA_M / 2] : -INFINITY;
            float top = fmaxf(s0, s1);
            for (int o = GROUP; o < WARP; o <<= 1) top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, o));
            top = fmaxf(top, m);
            const float c = rescale(m, top);
            for (int e = 0; e < LANE_DIMS; e++) acc[e] *= c;
            m = top;
            const float p0 = expf(s0 - m), p1 = expf(s1 - m);
            float ps = p0 + p1;
            for (int o = GROUP; o < WARP; o <<= 1) ps += __shfl_xor_sync(0xffffffffu, ps, o);
            l = l * c + ps;
            const float w0 = p0 * vss[j], w1 = p1 * vss[j + MMA_M / 2];  /* the values' weights */
#pragma unroll
            for (int i = 0; i < MMA_M; i++) {
                if (i >= count) break;
                const float w = __shfl_sync(0xffffffffu, i < MMA_M / 2 ? w0 : w1, i % (MMA_M / 2) * GROUP);
                float vd[LANE_DIMS];
                load_dims(vs + (j0 + i) * HEAD_DIM + lane * LANE_DIMS, vd);
                for (int e = 0; e < LANE_DIMS; e++) acc[e] = __fmaf_rn(w, vd[e], acc[e]);
            }
        }
        __syncthreads();  /* buffer r % 2 consumed before run r + 2 loads into it */
    }

    /* the warps' states merged in warp order */
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
    /* the current key and value, then acc / l */
    const float cur = key_score(q + lane * LANE_DIMS, k + lane * LANE_DIMS), top = fmaxf(bm, cur);
    const float c = rescale(bm, top), e = expf(cur - top);
    const float y = __fmaf_rn(e, v[tid], __fmul_rn(c, bacc));
    a.out[(size_t)b * a.C + head * HEAD_DIM + tid] = __float2bfloat16_rn(__fdiv_rn(y, __fmaf_rn(c, bl, e)));
}

/* Each active slot's next input token and position: the chosen token (ARGMAX's maximum), and its alternative (the
 * choice but Choice::lazy, -1 if none); a thread per slot. */
__global__ void k_commit(int32_t *tokens, int32_t *alt_tokens, int32_t *pos, const unsigned long long *active,
                         unsigned long long *argmax, unsigned long long *alt, int B) {
    pdl_start();
    const int b = threadIdx.x;
    if (b >= B || !live(active, b)) return;
    tokens[b] = (int32_t)(UINT32_MAX - (uint32_t)argmax[b]);
    alt_tokens[b] = (int32_t)(UINT32_MAX - (uint32_t)alt[b]);  /* a key's row is ~(its low word): 0 gives -1 */
    pos[b] += 1;
    argmax[b] = 0ull;
    alt[b] = 0ull;
}

/* Slot b's next input becomes the alternative its last step chose (k_commit): the host rejected the lazy row. */
__global__ void k_take_alt(int32_t *tokens, const int32_t *alt_tokens, int b) { tokens[b] = alt_tokens[b]; }

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
    /* B: the slots a graph covers; capacity: the decoder's slots (B but in a narrow graph's view, and the host tokens'
     * layout). max_len: a slot's cache positions (the caller's longest sequence). store: the prompt store's
     * positions. lazy: the row whose choice the host confirms (Choice) */
    int B, capacity, L, H, C, F, vocab, max_len, store, lazy;
    const bf16 *wte, *lm_head;
    const float *cos, *sin;   /* the fp32 rotary table [context][HALF] */
    Layer *layer;
    int32_t *tokens, *pos;    /* [B]: each slot's input token and position */
    int32_t *alt_tokens;      /* [B]: each slot's last alternative choice (k_commit) */
    int32_t *first;           /* [B]: the position of each slot's first generated token's input */
    float *temperature;       /* [B] */
    float *top_p;             /* [B] */
    unsigned long long *seed; /* [B] */
    float *logits, *probs;    /* [B][vocab]: k_nucleus's inputs (Choice) */
    float softcap;            /* the output-logit cap (Choice), 0 none */
    uint32_t *allow;          /* [B][vocab / WORD_BITS]: each masked slot's allowed tokens */
    uint32_t *host_allow;     /* pinned host, as allow: written by the caller (dec_masks), copied by dec_set_active */
    unsigned long long *active;  /* device [2][MASK_WORDS]: the active slots' mask, then the masked slots' */
    int32_t *host_tokens;     /* pinned host [2][B]: the last step's tokens, then their alternatives */
    unsigned long long *argmax, *alt;  /* [B]: the choice's key, the alternative's (Choice::alt) */
    bf16 *x, *x0, *xm, *h, *hn, *qkv_out, *att, *xo, *u;  /* [B][...]; hn: the normed input of fc and lm_head */
    float *qkvf;              /* [B][3][C]: k_prologue's q, k, v */
    uint8_t *kc, *vc;         /* [B][L][max_len][C] e4m3 */
    float *ksc, *vsc;         /* [B][L][H][max_len]: their scales */
    /* the prompt store, one allocation: K and V [L][store][C] e4m3 each, then their scales [L][H][store]: prompts'
     * K and V kept for reuse (dec_save_prompt, dec_load_prompt) */
    uint8_t *store_k, *store_v;
    float *store_ksc, *store_vsc;
    cudaStream_t stream;      /* the caller's */
    /* the whole step; before the lm_head GEMM; from it on: over all slots, [1] over the first SMALL_BATCH (narrow: no
     * higher slot is active, dec_set_active) */
    cudaGraphExec_t step[2], body[2], head[2];
    bool narrow;
    /* prefill: rows (slot, position, token) of prompts run through the body at once (dec_prefill): a Decoder of
     * MAX_BATCH rows sharing this one's weights and caches, its row b at slot cache_slot[b] (NULL in a decoder) */
    int32_t *cache_slot;
    void *prefill_rows;       /* the prefill decoder (a Decoder *), NULL in it */
    cudaGraphExec_t prefill;  /* the prefill decoder's body */
} Decoder;

/* f(buffer, bytes per slot) for each per-slot device buffer. */
template <class F> static void slot_buffers(Decoder *d, F f) {
    const size_t C = d->C, kv = (size_t)d->L * d->max_len * C;
    const size_t scales = (size_t)d->L * d->H * d->max_len * sizeof(float);
    f((void **)&d->tokens, sizeof(int32_t));
    f((void **)&d->alt_tokens, sizeof(int32_t));
    f((void **)&d->pos, sizeof(int32_t));
    f((void **)&d->argmax, sizeof(unsigned long long));
    f((void **)&d->alt, sizeof(unsigned long long));
    f((void **)&d->first, sizeof(int32_t));
    f((void **)&d->temperature, sizeof(float));
    f((void **)&d->top_p, sizeof(float));
    f((void **)&d->seed, sizeof(unsigned long long));
    f((void **)&d->logits, d->vocab * sizeof(float));
    f((void **)&d->probs, d->vocab * sizeof(float));
    f((void **)&d->allow, d->vocab / WORD_BITS * sizeof(uint32_t));
    for (bf16 **b : {&d->x, &d->x0, &d->xm, &d->h, &d->hn, &d->att, &d->xo}) f((void **)b, C * sizeof(bf16));
    f((void **)&d->qkv_out, 3 * C * sizeof(bf16));
    f((void **)&d->qkvf, 3 * C * sizeof(float));
    f((void **)&d->u, d->F * sizeof(bf16));
    f((void **)&d->kc, kv);
    f((void **)&d->vc, kv);
    f((void **)&d->ksc, scales);
    f((void **)&d->vsc, scales);
}

/* f(buffer, bytes per row) for each per-row device buffer of a prefill decoder: its inputs (token, position, cache
 * slot: one [3][rows] buffer at tokens) and the body's activations. */
template <class F> static void row_buffers(Decoder *d, F f) {
    const size_t C = d->C;
    f((void **)&d->tokens, 3 * sizeof(int32_t));
    for (bf16 **b : {&d->x, &d->x0, &d->xm, &d->h, &d->hn, &d->att, &d->xo}) f((void **)b, C * sizeof(bf16));
    f((void **)&d->qkv_out, 3 * C * sizeof(bf16));
    f((void **)&d->qkvf, 3 * C * sizeof(float));
    f((void **)&d->u, d->F * sizeof(bf16));
}

static Gemm gemm(const Decoder *d, const bf16 *w, int rows, int cols, const bf16 *in, int epi, bf16 *out) {
    Gemm g = {};
    g.w = w; g.rows = rows; g.cols = cols; g.in = in; g.epi = epi; g.out = out;
    g.active = d->active; g.B = d->B;
    return g;
}

static void launch_norm(Decoder *d, int kind, const bf16 *in, const bf16 *in2, float l1, float l2, bf16 *mixed,
                        bf16 *out) {
    launch(k_norm, (d->B + NORM_WARPS - 1) / NORM_WARPS, NORM_WARPS * WARP, 0, d->stream,
           kind, in, in2, d->tokens, l1, l2, mixed, out, d->active, d->B, d->C);
}

static Choice choice(const Decoder *d) {
    return {d->allow, d->temperature, d->top_p, d->seed, d->pos, d->first, d->logits, d->probs, d->softcap, d->alt,
            d->lazy};
}

/* The driver's cuTensorMapEncodeTiled (by the runtime's entry-point query: no libcuda link). */
typedef CUresult (*EncodeTiled)(CUtensorMap *, CUtensorMapDataType, cuuint32_t, void *, const cuuint64_t *,
                                const cuuint64_t *, const cuuint32_t *, const cuuint32_t *, CUtensorMapInterleave,
                                CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

/* A TMA map of a [rows][cols] bf16 matrix in boxes of WG_BK columns by box_rows rows, 128-byte swizzled (k_wgmma),
 * out-of-bounds rows zero; L2 promotion as DeepGEMM's TMA maps (csrc/jit_kernels/impls/runtime_utils.hpp). */
static CUtensorMap tile_map(const bf16 *p, int rows, int cols, int box_rows) {
    static const EncodeTiled encode = [] {
        void *fn = NULL;
        cudaDriverEntryPointQueryResult found;
        CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &found));
        return (EncodeTiled)fn;
    }();
    CUtensorMap map;
    const cuuint64_t dims[2] = {(cuuint64_t)cols, (cuuint64_t)rows}, strides[1] = {(cuuint64_t)cols * sizeof(bf16)};
    const cuuint32_t box[2] = {WG_BK, (cuuint32_t)box_rows}, step[2] = {1, 1};
    const CUresult r = encode(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void *)p, dims, strides, box, step,
                              CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                              CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) {
        fprintf(stderr, "decode.cu: cuTensorMapEncodeTiled failed (%d)\n", (int)r);
        abort();
    }
    return map;
}

/* k_wgmma, a block per (slot tile, row tile); tiles without an active slot exit at once. */
template <bool CHOICE> static void launch_gemm(Decoder *d, const Gemm &g) {
    const dim3 grid((g.B + WG_N - 1) / WG_N, (g.rows + WG_M - 1) / WG_M);
    launch(k_wgmma<CHOICE>, grid, 2 * WG_THREADS, WG_SMEM, d->stream, tile_map(g.w, g.rows, g.cols, WG_M),
           tile_map(g.in, g.B, g.cols, WG_N), g, choice(d));
}

/* The step's body on d->stream: every layer of every active slot, up to x before the final norm. */
static void enqueue_body(Decoder *d) {
    const int C = d->C, F = d->F, L = d->L, B = d->B;
    launch_norm(d, EMBED, d->wte, NULL, 0.f, 0.f, NULL, d->x0);
    for (int l = 0; l < L; l++) {
        const Layer *ly = &d->layer[l];
        /* input: x0 at layer 0, else the previous layer's output */
        launch_norm(d, RESID_NORM, l == 0 ? d->x0 : d->x, d->x0, ly->resid_lambda, ly->x0_lambda, d->xm, d->h);
        launch_gemm<false>(d, gemm(d, ly->qkv, 3 * C, C, d->h, EPI_STORE, d->qkv_out));
        const Attn a = {d->qkv_out, d->h, ly->ve, ly->ve_gate, ly->q_gain, ly->k_gain, d->cos, d->sin,
                        d->kc + (size_t)l * d->max_len * C, d->vc + (size_t)l * d->max_len * C,
                        (size_t)L * d->max_len * C, d->ksc + (size_t)l * d->H * d->max_len,
                        d->vsc + (size_t)l * d->H * d->max_len, (size_t)L * d->H * d->max_len, d->max_len,
                        d->att, d->tokens, d->pos, d->active, C, d->H, ly->window, d->cache_slot, d->qkvf};
        /* every row's k, v cached before any row's attention (prefill: a slot's earlier positions may be rows of
         * this launch); prefill's last layer needs nothing more (no row chooses a token) */
        launch(k_prologue, dim3(d->H, B), HEAD_DIM, 0, d->stream, a);
        if (d->cache_slot && l == L - 1) break;
        launch(k_attn, dim3(d->H, B), HEAD_DIM, ATTN_SMEM, d->stream, a);
        Gemm g = gemm(d, ly->o, C, C, d->att, EPI_RESID, d->xo);
        g.res = d->xm;
        launch_gemm<false>(d, g);
        launch_norm(d, NORM, d->xo, NULL, 0.f, 0.f, NULL, d->hn);
        launch_gemm<false>(d, gemm(d, ly->fc, F, C, d->hn, EPI_RELU2, d->u));
        g = gemm(d, ly->proj, C, F, d->u, EPI_RESID, d->x);
        g.res = d->xo;
        launch_gemm<false>(d, g);
    }
}

/* The step's head on d->stream: the final norm, the token choice in lm_head's epilogue (k_wgmma's CHOICE) or, for a
 * nucleus-sampled slot, by k_nucleus, the commit and the tokens' copy to the host. */
template <bool CHOICE> static void enqueue_head(Decoder *d) {
    launch_norm(d, NORM, d->x, NULL, 0.f, 0.f, NULL, d->hn);
    Gemm g = gemm(d, d->lm_head, d->vocab, d->C, d->hn, EPI_ARGMAX, NULL);
    g.argmax = d->argmax;
    launch_gemm<CHOICE>(d, g);
    if constexpr (CHOICE)
        launch(k_nucleus, d->B, NUCLEUS_THREADS, 0, d->stream, choice(d), d->active, d->argmax, d->vocab);
    launch(k_commit, 1, MAX_BATCH, 0, d->stream, d->tokens, d->alt_tokens, d->pos, d->active, d->argmax, d->alt, d->B);
    cudaMemcpyAsync(d->host_tokens, d->tokens, (size_t)d->capacity * sizeof(int32_t), cudaMemcpyDeviceToHost,
                    d->stream);
    cudaMemcpyAsync(d->host_tokens + d->capacity, d->alt_tokens, (size_t)d->capacity * sizeof(int32_t),
                    cudaMemcpyDeviceToHost, d->stream);
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
    Decoder *pf = (Decoder *)d->prefill_rows;
    cudaGraphExecDestroy(d->prefill);
    row_buffers(pf, [](void **p, size_t) { cudaFree(*p); });
    cudaFree(pf->active);
    free(pf);
    for (int narrow = 0; narrow < 2; narrow++)
        for (cudaGraphExec_t exec : {d->step[narrow], d->body[narrow], d->head[narrow]})
            cudaGraphExecDestroy(exec);
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
 *      requested: it takes at most the device memory left after the slots and graphs), lazy (the row whose choice
 *      the host confirms after a step, Choice::lazy; -1 none), budget (MiB the decoder takes at most: slots, prefill
 *      rows, graphs and prompt store; 0: all free memory).
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
    d->lazy = cfg[7];
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
    for (const void *kernel : {(const void *)k_wgmma<false>, (const void *)k_wgmma<true>})
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, WG_SMEM));
    CUDA_CHECK(cudaFuncSetAttribute(k_attn, cudaFuncAttributeMaxDynamicSharedMemorySize, ATTN_SMEM));
    /* the decoder's memory: free memory, at most the budget (cfg[8] MiB; 0: none); B: the most slots (at most
     * MAX_BATCH) that fit it after the prefill decoder's MAX_BATCH rows */
    size_t slot = 0, row = 0, free_bytes, total_bytes;
    slot_buffers(d, [&](void **, size_t bytes) { slot += bytes; });
    row_buffers(d, [&](void **, size_t bytes) { row += bytes; });
    cudaMemGetInfo(&free_bytes, &total_bytes);
    const size_t free_start = free_bytes, budget = (size_t)cfg[8] << 20;
    if (budget > 0 && budget < free_bytes) free_bytes = budget;
    const size_t limit = free_bytes;  /* the bytes this decoder may take */
    free_bytes -= MAX_BATCH * row < free_bytes ? MAX_BATCH * row : free_bytes;
    d->B = (int)(free_bytes / slot < (size_t)MAX_BATCH ? free_bytes / slot : MAX_BATCH);
    d->capacity = d->B;
    slot_buffers(d, [&](void **p, size_t bytes) {
        CUDA_CHECK(cudaMalloc(p, d->B * bytes));
        CUDA_CHECK(cudaMemset(*p, 0, d->B * bytes));
    });
    CUDA_CHECK(cudaMalloc((void **)&d->active, 2 * MASK_WORDS * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d->active, 0, 2 * MASK_WORDS * sizeof(unsigned long long)));
    CUDA_CHECK(cudaHostAlloc((void **)&d->host_tokens, 2 * d->B * sizeof(int32_t), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void **)&d->host_allow, (size_t)d->B * (d->vocab / WORD_BITS) * sizeof(uint32_t),
                             cudaHostAllocDefault));
    /* each graph twice: over all B slots, and over the first SMALL_BATCH (launched while no higher slot is active:
     * the blocks of inactive slots exit at once, but a mostly idle batch of MAX_BATCH slots still pays their launch) */
    for (int narrow = 0; narrow < 2; narrow++) {
        Decoder view = *d;
        if (narrow) view.B = d->B < SMALL_BATCH ? d->B : SMALL_BATCH;
        Decoder *v = &view;
        d->step[narrow] = capture(v, [v] { enqueue_body(v); enqueue_head<false>(v); });
        d->body[narrow] = capture(v, [v] { enqueue_body(v); });
        d->head[narrow] = capture(v, [v] { enqueue_head<true>(v); });
    }
    /* the prefill decoder: this one's weights, shapes and caches, MAX_BATCH rows of its own activations */
    Decoder *pf = (Decoder *)malloc(sizeof(Decoder));
    *pf = *d;
    pf->B = MAX_BATCH;
    pf->prefill_rows = NULL;
    row_buffers(pf, [&](void **p, size_t bytes) {
        CUDA_CHECK(cudaMalloc(p, MAX_BATCH * bytes));
        CUDA_CHECK(cudaMemset(*p, 0, MAX_BATCH * bytes));
    });
    pf->pos = pf->tokens + MAX_BATCH;
    pf->cache_slot = pf->tokens + 2 * MAX_BATCH;
    CUDA_CHECK(cudaMalloc((void **)&pf->active, 2 * MASK_WORDS * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(pf->active, 0, 2 * MASK_WORDS * sizeof(unsigned long long)));
    d->prefill_rows = pf;
    d->prefill = capture(pf, [pf] { enqueue_body(pf); });
    /* the prompt store: the requested positions, at most what the decoder's memory holds after the slots and graphs;
     * K and V in one allocation, shrunk 2 MiB at a time until it succeeds (cudaMemGetInfo's free bytes do not all fit
     * one allocation; 2 MiB: the device allocation granularity, cuMemGetAllocationGranularity's on Hopper) */
    const size_t position = 2 * (size_t)L * (d->C + d->H * sizeof(float));
    cudaMemGetInfo(&free_bytes, &total_bytes);
    const size_t taken = free_start > free_bytes ? free_start - free_bytes : 0;
    if (limit - (taken < limit ? taken : limit) < free_bytes) free_bytes = limit - (taken < limit ? taken : limit);
    if ((size_t)d->store > free_bytes / position) d->store = (int)(free_bytes / position);
    while (d->store > 0 && cudaMalloc((void **)&d->store_k, position * d->store) != cudaSuccess) {
        cudaGetLastError();  /* clear the failed allocation's error */
        const int step = (int)(((2 << 20) + position - 1) / position);
        d->store = d->store > step ? d->store - step : 0;
    }
    d->store_v = d->store_k + (size_t)L * d->store * d->C;
    d->store_ksc = reinterpret_cast<float *>(d->store_v + (size_t)L * d->store * d->C);
    d->store_vsc = d->store_ksc + (size_t)L * d->H * d->store;
    return d;
}

/* Start slot b at its prompt's last token `token`, at position `last` (the positions before it cached: dec_prefill or
 * dec_load_prompt): from that step on it chooses tokens at temperature (0: argmax) with noise from seed (gumbel),
 * from the nucleus holding top_p of the probability when top_p < 1 (k_nucleus), the first choice at generated
 * position 0. Stream-ordered. */
extern "C" void dec_set_prompt(void *handle, int b, int32_t token, int32_t last, float temperature, float top_p,
                               unsigned long long seed) {
    Decoder *d = (Decoder *)handle;
    cudaMemcpyAsync(d->tokens + b, &token, sizeof(int32_t), cudaMemcpyHostToDevice, d->stream);
    cudaMemcpyAsync(d->pos + b, &last, sizeof(int32_t), cudaMemcpyHostToDevice, d->stream);
    cudaMemcpyAsync(d->first + b, &last, sizeof(int32_t), cudaMemcpyHostToDevice, d->stream);
    cudaMemcpyAsync(d->temperature + b, &temperature, sizeof(float), cudaMemcpyHostToDevice, d->stream);
    cudaMemcpyAsync(d->top_p + b, &top_p, sizeof(float), cudaMemcpyHostToDevice, d->stream);
    cudaMemcpyAsync(d->seed + b, &seed, sizeof(seed), cudaMemcpyHostToDevice, d->stream);
}

/* The rows a dec_prefill call takes at most. */
extern "C" int dec_prefill_rows(void *) { return MAX_BATCH; }

/* Run n prompt rows through the body at once, caching each row's K and V at its position in its slot (the K and V a
 * step with that slot's input at that position caches: every row is computed alone, in the step's order): rows[0, n)
 * the tokens, rows[n, 2n) their positions, rows[2n, 3n) their slots. A slot's earlier positions are cached already
 * or are earlier rows (the rows' K and V are cached before any row's attention). n <= dec_prefill_rows. A prompt's
 * positions before its last token, so prefilled, let its slot start at its last token (dec_set_prompt).
 * Stream-ordered. */
extern "C" void dec_prefill(void *handle, int n, const int32_t *rows) {
    Decoder *d = (Decoder *)handle;
    Decoder *pf = (Decoder *)d->prefill_rows;
    const size_t width = (size_t)n * sizeof(int32_t);
    cudaMemcpy2DAsync(pf->tokens, MAX_BATCH * sizeof(int32_t), rows, width, width, 3, cudaMemcpyHostToDevice,
                      d->stream);
    unsigned long long mask[2 * MASK_WORDS] = {};
    for (int r = 0; r < n; r++) mask[r / MASK_BITS] |= 1ull << (r % MASK_BITS);
    cudaMemcpyAsync(pf->active, mask, sizeof mask, cudaMemcpyHostToDevice, d->stream);
    cudaGraphLaunch(d->prefill, d->stream);
}

/* Slot b's cached K and V at positions [at, at + n) to the prompt store at [store_at, store_at + n) (to_store), or
 * back: per cache one 2D copy of L rows (the store's rows store positions apart, the slot's max_len), per scale array
 * one of L H rows. */
static void prompt_copy(Decoder *d, int b, int at, int store_at, int n, bool to_store) {
    auto copy = [&](void *cache, size_t cache_pitch, void *store, size_t store_pitch, size_t width, size_t height) {
        if (to_store)
            cudaMemcpy2DAsync(store, store_pitch, cache, cache_pitch, width, height, cudaMemcpyDeviceToDevice,
                              d->stream);
        else
            cudaMemcpy2DAsync(cache, cache_pitch, store, store_pitch, width, height, cudaMemcpyDeviceToDevice,
                              d->stream);
    };
    const size_t C = d->C, L = d->L, H = d->H;
    const size_t in_slot = ((size_t)b * L * d->max_len + at) * C, in_store = (size_t)store_at * C;
    for (auto [cache, store] : {std::pair{d->kc, d->store_k}, std::pair{d->vc, d->store_v}})
        copy(cache + in_slot, d->max_len * C, store + in_store, d->store * C, n * C, L);
    const size_t sc_slot = (size_t)b * L * H * d->max_len + at;
    for (auto [cache, store] : {std::pair{d->ksc, d->store_ksc}, std::pair{d->vsc, d->store_vsc}})
        copy(cache + sc_slot, d->max_len * sizeof(float), store + store_at, d->store * sizeof(float),
             n * sizeof(float), L * H);
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
    int high = -1;  /* the highest active slot */
    for (int w = 0; w < words; w++)
        if (slots[w]) high = w * MASK_BITS + MASK_BITS - 1 - __builtin_clzll(slots[w]);
    d->narrow = high < SMALL_BATCH;
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
    cudaGraphLaunch(d->step[d->narrow], d->stream);
}

/* Launch one step's body (asynchronous): everything before the token choice, which dec_launch_head then launches. */
extern "C" void dec_launch_body(void *handle) {
    Decoder *d = (Decoder *)handle;
    cudaGraphLaunch(d->body[d->narrow], d->stream);
}

/* Launch one step's head (asynchronous) after its body: every slot chooses by its mask (if dec_set_active marks it
 * masked) and temperature; dec_tokens returns its tokens. */
extern "C" void dec_launch_head(void *handle) {
    Decoder *d = (Decoder *)handle;
    cudaGraphLaunch(d->head[d->narrow], d->stream);
}

/* Wait for the launched step; out[b]: the token slot b took, out[B + b]: its alternative, the choice but the lazy row
 * (Choice::lazy; -1 if none: a greedy whole step, a nucleus-sampled slot) (stale for inactive slots). */
extern "C" void dec_tokens(void *handle, int32_t *out) {
    Decoder *d = (Decoder *)handle;
    CUDA_CHECK(cudaStreamSynchronize(d->stream));
    for (int b = 0; b < 2 * d->B; b++) out[b] = d->host_tokens[b];
}

/* Slot b's next input becomes its last step's alternative (dec_tokens), not its choice: the host rejected the lazy
 * row. Stream-ordered, after dec_tokens and before the next launch. */
extern "C" void dec_take_alt(void *handle, int b) {
    Decoder *d = (Decoder *)handle;
    k_take_alt<<<1, 1, 0, d->stream>>>(d->tokens, d->alt_tokens, b);
}
