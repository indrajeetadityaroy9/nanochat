/* Decoding of a nanochat CausalLM (model/modeling.py) in MAX_BATCH slots (inference/decoder.py): a step runs one graph
 * (greedy) or two (body, head) of dependent kernels, advancing each active slot by one greedy, sampled or nucleus token
 * within its allowed set. Batched prefill; an exact-prompt store, never evicted, reuses prompts' K and V. Numerics
 * follow the compiled training forward op by op, except the e4m3 KV cache (a scale per position and head). */
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
/* k_commit runs the slots as one block: at most 1,024 threads (CUDA C++ Programming Guide, "Thread Hierarchy") */
constexpr int MAX_BATCH = 1024;
constexpr int MASK_BITS = CHAR_BIT * sizeof(unsigned long long);  /* slots per word of a slot mask */
constexpr int MASK_WORDS = MAX_BATCH / MASK_BITS;                 /* words of a slot mask */
/* the narrow graphs' slots (Decoder::narrow), tuned at d24 */
constexpr int SMALL_BATCH = 256;
static_assert(MAX_BATCH % MASK_BITS == 0, "slot masks are whole words");
/* mma.sync m16n8k16 (PTX ISA "Matrix Fragments for mma.m16n8k16"): groupID = lane / GROUP, ACC accumulators a lane */
constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 16;
constexpr int GROUP = WARP / MMA_N, ACC = MMA_M * MMA_N / WARP;
/* k_norm: the fewest warps a block filling an SM within its resident-block limit (cc 9.0: 2,048 threads, 32 blocks) */
constexpr int NORM_WARPS = 2048 / 32 / WARP;
constexpr int WORD_BITS = CHAR_BIT * sizeof(uint32_t);  /* tokens per word of an allowed-token mask */
/* k_nucleus: threads per slot, a block each: the most a block holds (CUDA C++ Programming Guide, "Thread Hierarchy") */
constexpr int NUCLEUS_THREADS = 1024;

enum { NORM, RESID_NORM, EMBED };
enum { EPI_STORE, EPI_RESID, EPI_RELU2, EPI_ARGMAX };

/* A failed CUDA runtime call aborts: never silently wrong tokens from a missing buffer or a sticky error. */
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

static __device__ __forceinline__ float warp_max(float v) {
    for (int o = WARP / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

/* The block's sum of v to every thread, fixed order (warp butterflies, then warps in order); scratch: a float a warp */
static __device__ float block_sum(float v, float *scratch) {
    v = warp_sum(v);
    __syncthreads();  /* the previous call's reads of scratch are done */
    if (threadIdx.x % WARP == 0) scratch[threadIdx.x / WARP] = v;
    __syncthreads();
    float sum = 0.f;
    for (int w = 0; w < (int)blockDim.x / WARP; w++) sum = __fadd_rn(sum, scratch[w]);
    return sum;
}

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

/* Gumbel noise -log(-log(u)), u in (0, 1) exact in double, from Philox4x32-10 (SC '11; cuRAND) at counter (row, i), key
 * seed; argmax of s / T + noise samples softmax(s / T) (Gumbel-max, arXiv 1411.0030 §2). */
static __device__ __forceinline__ float gumbel(unsigned long long seed, int i, int row) {
    const uint4 x = curand_Philox4x32_10(make_uint4((uint32_t)row, (uint32_t)i, 0u, 0u),
                                         make_uint2((uint32_t)seed, (uint32_t)(seed >> 32)));
    const unsigned long long r = ((unsigned long long)x.x << 32 | x.y) >> (64 - (DBL_MANT_DIG - 1));
    return (float)-log(-log(ldexp(2.0 * (double)r + 1.0, -DBL_MANT_DIG)));
}

/* Slot GEMM: out[b][r] = W[r] . x[b], bf16 operands, fp32 accumulators; RELU2 out = bf16(relu(s) s) (QuACK
 * gemm_norm_act relu_sq); ARGMAX: each warp's maximum key per slot, then one atomicMax (an exact maximum, so any
 * order). */

typedef struct {
    const bf16 *w;               /* [rows][cols] */
    int rows, cols;
    const bf16 *in;              /* [B][cols] */
    int epi;
    bf16 *out;                   /* [B][rows]: STORE/RESID/RELU2 output */
    const bf16 *res;             /* [B][rows]: RESID: residual added */
    unsigned long long *argmax;  /* [B] ARGMAX: packed (ordered score, ~row) maximum */
    const unsigned long long *active;  /* [2][MASK_WORDS]: the active slots' mask, then the masked slots' (CHOICE) */
    int B;
} Gemm;

/* CHOICE ARGMAX's and k_nucleus's per-slot inputs, a separate argument: growing Gemm changes the compiled GEMMs. */
typedef struct {
    const uint32_t *allow;           /* [B][rows / WORD_BITS]: a masked slot's allowed rows, bit row % WORD_BITS */
    const float *temperature;        /* [B]: 0 the logits' argmax, else Gumbel-max sampling at that temperature */
    const float *top_p;              /* [B]: below 1 at temperature > 0: the nucleus (k_nucleus) */
    const unsigned long long *seed;
    const int32_t *pos, *first;      /* [B]: the input's position and the first generated token's (i = pos - first) */
    float *logits;                   /* [B][rows]: a nucleus-sampled slot's logits, -inf outside its allowed rows */
    float *probs;                    /* [B][rows]: k_nucleus's exp(s / T - max) */
    float softcap;                   /* logit cap c: sampled logits c tanh(s / c); argmax needs none */
    unsigned long long *alt;         /* [B]: the best allowed key but lazy's (0 none), if the host rejects lazy */
    int lazy;                        /* the row the host confirms after the step (BOS) */
} Choice;

/* 16 bytes global -> shared without a register round trip (PTX ISA "Data Movement: cp.async"). */
static __device__ __forceinline__ void copy16(void *dst, const void *src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src));
}

static __device__ __forceinline__ void copies_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }

template <int N> static __device__ __forceinline__ void copies_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

/* PDL (CUDA C++ Programming Guide, "Programmatic Dependent Launch and Synchronization"): every kernel waits for its
 * predecessor's completed, flushed writes before anything else, early exits included, then lets its successor start. */
static __device__ __forceinline__ void pdl_start() {
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;\n" :::);
}

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

/* One GEMM output, slot b's row from its fp32 dot product s; ARGMAX folds its key into the lane's best (and alt). */
template <bool CHOICE> static __device__ __forceinline__ void epilogue(const Gemm &a, const Choice &c, int row, int b,
                                                                      float s, unsigned long long &best,
                                                                      unsigned long long &alt) {
    const size_t at = (size_t)b * a.rows + row;
    if (a.epi == EPI_RELU2) {
        a.out[at] = __float2bfloat16_rn(__fmul_rn(fmaxf(s, 0.f), s));
        return;
    }
    const float y = bfr(s);
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
        const float z = c.temperature[b] > 0.f ? __fmul_rn(c.softcap, tanhf(__fdiv_rn(s, c.softcap))) : s;
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

/* ARGMAX: the lanes' keys for slots b0 + 2t + p, maximized over the lanes sharing t, then one atomicMax per slot. */
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

/* Slot GEMM by wgmma (sm_90a): a block multiplies a WG_M-row tile of W by a WG_N-slot tile (one active-mask word); a
 * producer warp TMA-loads WG_BK slices into a WG_STAGES ring (128-byte swizzle, full/empty mbarriers); a consumer
 * warpgroup runs m64n64k16, one slice's MMAs in flight while the next issue; accumulators as mma m16n8 (PTX ISA). */
constexpr int WG_THREADS = 128, WG_M = 64, WG_N = MASK_BITS, WG_BK = 64, WG_STAGES = 4;
static_assert(WG_BK * sizeof(bf16) == 128, "a K slice row is one 128-byte swizzle span");
constexpr uint32_t WG_A_BYTES = WG_M * WG_BK * sizeof(bf16), WG_B_BYTES = WG_N * WG_BK * sizeof(bf16);
/* k_wgmma's smem: ring, barriers, 1024 bytes to align it (a 128-byte swizzle atom's 8 rows: PTX ISA "Shared Memory
 * Matrix Layout") */
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

/* Nucleus sampling (arXiv 1904.09751 §3.1) as transformers' TopPLogitsWarper: keeps e >= e*, the least e with C(e) >
 * (1 - top_p) sum(e), C(x) the sum of e <= x, e = exp(s / T - max s / T), by bisection on x's IEEE bits in [0, 1];
 * then Gumbel-max as the epilogue. Equal probabilities are kept or dropped together; fixed-order sums. */
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

/* RMS norm of each active slot's row, a warp per slot: r = rsqrt(mean v^2 + FLT_EPSILON). EMBED: x0 = v r in fp32,
 * unrounded (Inductor's x0); RESID_NORM: v = fma(l1, h, x0 l2) as Triton contracts the mix, mixed = bf16(v). */
__global__ void k_norm(int kind, const bf16 *in, const float *wte, float *x0, const int32_t *tokens, float l1,
                       float l2, bf16 *mixed, bf16 *out, const unsigned long long *active, int B, int K) {
    pdl_start();
    extern __shared__ float rows[];  /* each warp's v, computed once: [NORM_WARPS][K] */
    const int b = blockIdx.x * NORM_WARPS + threadIdx.x / WARP, lane = threadIdx.x % WARP;
    if (b >= B || !live(active, b)) return;
    const size_t row = (size_t)b * K;
    float *v = rows + (size_t)(threadIdx.x / WARP) * K;  /* a lane reads back only the elements it wrote */
    float s = 0.f;
    for (int i = lane; i < K; i += WARP) {
        if (kind == EMBED) {
            v[i] = wte[(size_t)tokens[b] * K + i];
        } else if (kind == NORM) {
            v[i] = __bfloat162float(in[row + i]);
        } else {
            const float h = in ? __bfloat162float(in[row + i]) : x0[row + i];
            v[i] = __fmaf_rn(l1, h, __fmul_rn(x0[row + i], l2));
            mixed[row + i] = __float2bfloat16_rn(v[i]);
        }
        s = __fmaf_rn(v[i], v[i], s);
    }
    const float r = rsqrtf(warp_sum(s) / (float)K + FLT_EPSILON);
    for (int i = lane; i < K; i += WARP) {
        if (kind == EMBED)
            x0[row + i] = __fmul_rn(v[i], r);
        else
            out[row + i] = __float2bfloat16_rn(__fmul_rn(v[i], r));
    }
}

/* Attention, (head, slot) blocks of HEAD_DIM threads: k_prologue caches the row's k, v as e4m3; k_attn streams the
 * window in ATTN_RUN-key double-buffered runs, f16 MMA scores (exact for e4m3 k and bf16 q), fp32 online softmax. */

typedef struct {
    const bf16 *qkv, *h, *ve;  /* [B][3C], [B][C] (the normed input), the layer's value-embedding table or NULL */
    const float *ve_gate;      /* [H][C] fp32: the value-embedding gate's weights */
    const float *q_gain, *k_gain;  /* [HEAD_DIM] fp32, shared by the heads */
    const float *cos, *sin;    /* the rotary table [context][HALF], fp32 */
    uint8_t *kc, *vc;          /* the layer's FP8 (e4m3) cache of slot 0, [max_len][C]; slot b's at + b kv_slot */
    size_t kv_slot;
    float *ksc, *vsc;          /* scales of slot 0, [H][max_len] (value: e4m3 x scale); slot b's at + b sc_slot */
    size_t sc_slot;
    int max_len;
    bf16 *out;
    const int32_t *tokens, *pos;  /* [B]: each slot's input token and its position */
    const unsigned long long *active;
    int C, H, window;
    const int32_t *cache_slot; /* [B]: a prefill row's slot, whose cache row b reads and writes; NULL: slot b */
    float *qkvf;               /* [B][3][C]: each row's q, k, v at its position (k_prologue), fp32 holding bf16 */
} Attn;

/* Slot b's q, k, v of the head at position p (fp32 holding bf16); VE v += 2 sigmoid(ve_gate[head] . h) ve[token] in
 * fp32, one bf16 rounding; q, k by qk_row (threads 0..QK_LANES-1 q, the next QK_LANES k). Ends synchronized. */
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

/* k_attn's keys per cp.async run: one MMA_M tile per warp (tuned at d24) */
constexpr int ATTN_RUN = ATTN_WARPS * MMA_M;
constexpr int ROW_VECS = HEAD_DIM / (int)sizeof(uint4);  /* 16-byte vectors per e4m3 key row */
constexpr float FP8_MAX = 448.f;  /* the largest finite e4m3 */
/* one run's buffer: its keys and values (e4m3), then their scales */
constexpr size_t ATTN_BUF = 2 * (size_t)ATTN_RUN * HEAD_DIM + 2 * ATTN_RUN * sizeof(float);

constexpr size_t ATTN_SMEM = 2 * ATTN_BUF;

/* 4 bytes global -> shared (cp.async.ca: 4-byte copies take the L1 path). */
static __device__ __forceinline__ void copy4(void *dst, const void *src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                 :: "r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src));
}

static __device__ __forceinline__ uint8_t to_fp8(float x, float inv) {
    return (uint8_t)__nv_cvt_float_to_fp8(x * inv, __NV_SATFINITE, __NV_E4M3);
}

/* exp(m - top); 0 for an empty state (m = top = -inf would give NaN) */
static __device__ __forceinline__ float rescale(float m, float top) { return m == -INFINITY ? 0.f : expf(m - top); }

/* Before k_attn, per (head, row): q, k, v to qkvf; k, v to the slot's cache as e4m3, a scale amax / FP8_MAX each. */
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
    const int begin = p > a.window ? p - a.window : 0, n = p - begin;
    const int runs = (n + ATTN_RUN - 1) / ATTN_RUN;
    const int cs = a.cache_slot ? a.cache_slot[b] : b;
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

    /* q as the QK MMAs' B fragments (column 0, lanes g = 0): lane (g, t) of chunk ch holds dims 32 ch + 8 t .. + 7;
     * A and B map MMA k to the same physical dims, so the product is q . k */
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
        /* the warp's MMA_M keys: scores by MMAs (key rows g, g + 8), then the values on the CUDA cores */
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
            const float w0 = p0 * vss[j], w1 = p1 * vss[j + MMA_M / 2];
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
    const float cur = key_score(q + lane * LANE_DIMS, k + lane * LANE_DIMS), top = fmaxf(bm, cur);
    const float c = rescale(bm, top), e = expf(cur - top);
    const float y = __fmaf_rn(e, v[tid], __fmul_rn(c, bacc));
    a.out[(size_t)b * a.C + head * HEAD_DIM + tid] = __float2bfloat16_rn(__fdiv_rn(y, __fmaf_rn(c, bl, e)));
}

/* Each active slot's next input and position: the choice and its alternative (-1 if none); a thread per slot. */
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

__global__ void k_take_alt(int32_t *tokens, const int32_t *alt_tokens, int b) { tokens[b] = alt_tokens[b]; }

typedef struct {
    const bf16 *qkv, *o, *fc, *proj, *ve;  /* fc: [F][C]; ve: NULL in a layer without value embeddings */
    const float *ve_gate;                 /* the value-embedding gate [H][C], fp32; NULL without */
    const float *q_gain, *k_gain;         /* [HEAD_DIM] fp32 */
    float resid_lambda, x0_lambda;
    int window;
} Layer;

typedef struct {
    /* B: the slots a graph covers; capacity: the decoder's slots (the host tokens' layout); max_len: a slot's cache
     * positions; store: the prompt store's positions; lazy: the row the host confirms (Choice) */
    int B, capacity, L, H, C, F, vocab, max_len, store, lazy;
    const float *wte;         /* the fp32 embedding table (k_norm EMBED) */
    const bf16 *lm_head;
    const float *cos, *sin;   /* the fp32 rotary table [context][HALF] */
    Layer *layer;
    int32_t *tokens, *pos;    /* [B]: each slot's input token and position */
    int32_t *alt_tokens;      /* [B]: each slot's last alternative choice (k_commit) */
    int32_t *first;           /* [B]: the position of each slot's first generated token's input */
    float *temperature;
    float *top_p;
    unsigned long long *seed;
    float *logits, *probs;    /* [B][vocab]: k_nucleus's inputs (Choice) */
    float softcap;            /* the output-logit cap (Choice) */
    uint32_t *allow;          /* [B][vocab / WORD_BITS]: each masked slot's allowed tokens */
    uint32_t *host_allow;     /* pinned host, as allow: written by the caller (dec_masks), copied by dec_set_active */
    unsigned long long *active;  /* device [2][MASK_WORDS]: the active slots' mask, then the masked slots' */
    int32_t *host_tokens;     /* pinned host [2][B]: the last step's tokens, then their alternatives */
    unsigned long long *argmax, *alt;  /* [B]: the choice's key, the alternative's (Choice::alt) */
    bf16 *x, *xm, *h, *hn, *qkv_out, *att, *xo, *u;  /* [B][...]; hn: the normed input of fc and lm_head */
    float *x0;                /* [B][C]: the normed embedding, fp32 (k_norm EMBED) */
    float *qkvf;              /* [B][3][C]: k_prologue's q, k, v */
    uint8_t *kc, *vc;         /* [B][L][max_len][C] e4m3 */
    float *ksc, *vsc;         /* [B][L][H][max_len]: their scales */
    /* the prompt store, one allocation: K and V [L][store][C] e4m3 each, then their scales [L][H][store] */
    uint8_t *store_k, *store_v;
    float *store_ksc, *store_vsc;
    cudaStream_t stream;
    /* the whole step, its body (before lm_head), its head: [0] over all slots, [1] over the first SMALL_BATCH */
    cudaGraphExec_t step[2], body[2], head[2];
    bool narrow;
    /* prefill: a MAX_BATCH-row Decoder sharing weights and caches, row b at slot cache_slot[b] (NULL in a decoder) */
    int32_t *cache_slot;
    void *prefill_rows;       /* the prefill decoder (a Decoder *), NULL in it */
    cudaGraphExec_t prefill;
} Decoder;

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
    for (bf16 **b : {&d->x, &d->xm, &d->h, &d->hn, &d->att, &d->xo}) f((void **)b, C * sizeof(bf16));
    f((void **)&d->x0, C * sizeof(float));
    f((void **)&d->qkv_out, 3 * C * sizeof(bf16));
    f((void **)&d->qkvf, 3 * C * sizeof(float));
    f((void **)&d->u, d->F * sizeof(bf16));
    f((void **)&d->kc, kv);
    f((void **)&d->vc, kv);
    f((void **)&d->ksc, scales);
    f((void **)&d->vsc, scales);
}

/* f(buffer, bytes per row) for a prefill decoder's buffers; tokens holds [3][rows]: token, position, slot. */
template <class F> static void row_buffers(Decoder *d, F f) {
    const size_t C = d->C;
    f((void **)&d->tokens, 3 * sizeof(int32_t));
    for (bf16 **b : {&d->x, &d->xm, &d->h, &d->hn, &d->att, &d->xo}) f((void **)b, C * sizeof(bf16));
    f((void **)&d->x0, C * sizeof(float));
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

static void launch_norm(Decoder *d, int kind, const bf16 *in, float l1, float l2, bf16 *mixed, bf16 *out) {
    launch(k_norm, (d->B + NORM_WARPS - 1) / NORM_WARPS, NORM_WARPS * WARP, NORM_WARPS * d->C * sizeof(float),
           d->stream, kind, in, d->wte, d->x0, d->tokens, l1, l2, mixed, out, d->active, d->B, d->C);
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

template <bool CHOICE> static void launch_gemm(Decoder *d, const Gemm &g) {
    const dim3 grid((g.B + WG_N - 1) / WG_N, (g.rows + WG_M - 1) / WG_M);
    launch(k_wgmma<CHOICE>, grid, 2 * WG_THREADS, WG_SMEM, d->stream, tile_map(g.w, g.rows, g.cols, WG_M),
           tile_map(g.in, g.B, g.cols, WG_N), g, choice(d));
}

static void enqueue_body(Decoder *d) {
    const int C = d->C, F = d->F, L = d->L, B = d->B;
    launch_norm(d, EMBED, NULL, 0.f, 0.f, NULL, NULL);
    for (int l = 0; l < L; l++) {
        const Layer *ly = &d->layer[l];
        launch_norm(d, RESID_NORM, l == 0 ? NULL : d->x, ly->resid_lambda, ly->x0_lambda, d->xm, d->h);
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
        launch_norm(d, NORM, d->xo, 0.f, 0.f, NULL, d->hn);
        launch_gemm<false>(d, gemm(d, ly->fc, F, C, d->hn, EPI_RELU2, d->u));
        g = gemm(d, ly->proj, C, F, d->u, EPI_RESID, d->x);
        g.res = d->xo;
        launch_gemm<false>(d, g);
    }
}

/* The step's head: the final norm, the choice (lm_head's epilogue or k_nucleus), the commit, the tokens to the host. */
template <bool CHOICE> static void enqueue_head(Decoder *d) {
    launch_norm(d, NORM, d->x, 0.f, 0.f, NULL, d->hn);
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

extern "C" int dec_batch(void *handle) { return ((Decoder *)handle)->B; }

/* cfg: L, C, F, vocab, max_len (<= context), store (positions requested), lazy. ptrs: wte
 * (fp32), cos, sin ([context][HALF] fp32), lm_head; per layer qkv, o, fc, proj, ve table, ve gate (0 if none),
 * q norm, k norm. scal: per layer resid, x0 lambda; then the logit cap. window: per layer, keys before a query.
 * stream: the caller's (not the legacy default stream: it is captured), alive past dec_destroy. */
constexpr int GLOBAL_PTRS = 4, LAYER_PTRS = 8;

extern "C" void *dec_create(const int *cfg, const uint64_t *ptrs, const float *scal, const int *window, void *stream) {
    Decoder *d = (Decoder *)calloc(1, sizeof(Decoder));
    d->L = cfg[0]; d->C = cfg[1]; d->F = cfg[2]; d->vocab = cfg[3]; d->max_len = cfg[4]; d->store = cfg[5];
    d->lazy = cfg[6];
    d->H = d->C / HEAD_DIM;
    d->wte = (const float *)ptrs[0]; d->cos = (const float *)ptrs[1]; d->sin = (const float *)ptrs[2];
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
    /* B: the most slots (<= MAX_BATCH) that fit in free memory after the prefill's MAX_BATCH rows */
    size_t slot = 0, row = 0, free_bytes, total_bytes;
    slot_buffers(d, [&](void **, size_t bytes) { slot += bytes; });
    row_buffers(d, [&](void **, size_t bytes) { row += bytes; });
    cudaMemGetInfo(&free_bytes, &total_bytes);
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
    /* each graph over all B slots and over the first SMALL_BATCH: idle slots' blocks exit but still cost a launch */
    for (int narrow = 0; narrow < 2; narrow++) {
        Decoder view = *d;
        if (narrow) view.B = d->B < SMALL_BATCH ? d->B : SMALL_BATCH;
        Decoder *v = &view;
        d->step[narrow] = capture(v, [v] { enqueue_body(v); enqueue_head<false>(v); });
        d->body[narrow] = capture(v, [v] { enqueue_body(v); });
        d->head[narrow] = capture(v, [v] { enqueue_head<true>(v); });
    }
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
    /* the prompt store: at most what is left after slots and graphs, shrunk 2 MiB (Hopper's allocation granularity)
     * at a time until one allocation succeeds (cudaMemGetInfo's free bytes need not fit one allocation) */
    const size_t position = 2 * (size_t)L * (d->C + d->H * sizeof(float));
    cudaMemGetInfo(&free_bytes, &total_bytes);
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

/* Start slot b at its prompt's last token, position `last` (earlier positions cached), with its sampling parameters. */
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

extern "C" int dec_prefill_rows(void *) { return MAX_BATCH; }

/* Run n <= dec_prefill_rows prompt rows (rows: n tokens, n positions, n slots) through the body, caching each row's K
 * and V as a step would; a slot's earlier positions are cached or earlier rows. Stream-ordered. */
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

/* Slot b's K, V and scales at [at, at + n) to the prompt store at store_at (to_store) or back: one 2D copy per cache
 * (L rows) and per scale array (L H rows). */
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

/* Keep slot b's K and V at [at, at + n), written by the launched steps, in the store at store_at. Stream-ordered. */
extern "C" void dec_save_prompt(void *handle, int b, int at, int store_at, int n) {
    prompt_copy((Decoder *)handle, b, at, store_at, n, true);
}

extern "C" void dec_load_prompt(void *handle, int b, int at, int store_at, int n) {
    prompt_copy((Decoder *)handle, b, at, store_at, n, false);
}

extern "C" int dec_store(void *handle) { return ((Decoder *)handle)->store; }

/* The pinned host masks [B][vocab / WORD_BITS], bit t % WORD_BITS of word t / WORD_BITS of row b set iff slot b may
 * choose token t; the caller writes a masked slot's row before dec_set_active, after the previous step completed. */
extern "C" uint32_t *dec_masks(void *handle) { return ((Decoder *)handle)->host_allow; }

/* The active and masked slots, slots[2][ceil(dec_batch / MASK_BITS)], slot b at bit b % MASK_BITS of word
 * b / MASK_BITS; dec_masks' rows from the lowest to the highest masked slot are copied. Stream-ordered. */
extern "C" void dec_set_active(void *handle, const unsigned long long *slots) {
    Decoder *d = (Decoder *)handle;
    const int words = (d->B + MASK_BITS - 1) / MASK_BITS;
    const size_t row = words * sizeof(unsigned long long);
    cudaMemcpy2DAsync(d->active, MASK_WORDS * sizeof(unsigned long long), slots, row, row, 2, cudaMemcpyHostToDevice,
                      d->stream);
    const unsigned long long *masked = slots + words;
    int high = -1;
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

extern "C" void dec_launch(void *handle) {
    Decoder *d = (Decoder *)handle;
    cudaGraphLaunch(d->step[d->narrow], d->stream);
}

extern "C" void dec_launch_body(void *handle) {
    Decoder *d = (Decoder *)handle;
    cudaGraphLaunch(d->body[d->narrow], d->stream);
}

/* Launch one step's head (asynchronous) after its body: each slot chooses by its mask, if masked, and temperature. */
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
