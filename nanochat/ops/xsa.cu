/*
 * XSA (Zhai, arXiv 2603.09078 Eq. 2; modded-nanogpt track_1_short attention.py:224-228) at strength 1, the model's
 * (model/attention.py), as one op per direction, each one pass over memory, followed by the head-wise
 * attention-output gate (Qiu et al., arXiv 2505.06708, G1 head-wise; the gate's logits z come in). Per (token, head)
 * row of 128 dims, with y the attention's output and v its value input:
 *     p = y - r v,   r = (y . v) / max(v . v, 1e-8),   out = g p,   g = 1 / (1 + exp(-z))
 * Backward, with G the gradient of out, g' = g G, gv = g' . v, den = max(v . v, 1e-8), c1 = gv / den:
 *     dy = g' - c1 v,   dv = -(r g' + c1 y) + c2 v  (c2 = 2 c1 r where v . v >= 1e-8, the clamp's pass-through, else 0),
 *     dz = (G . p) g (1 - g)
 * fp32 math from the bf16 inputs (every product of two bf16 values is exact in fp32), one bf16 rounding per output (the
 * output: of g p, p unrounded).
 *
 * Layout: y, v, g, out, dy, dv (rows, 128) bf16 contiguous, row = token * H + head; z, dz (rows) fp32.
 * A row is LANES = 8 consecutive threads, lane j owning dims [16 j, 16 j + 16) (two 16-byte vectors). Each sum: a lane's
 * 16 terms in order (fmaf), then xor-shuffles 4, 2, 1 within the row's lanes. decode.cu's XSA reproduces this order and
 * these operations, so the decoder's XSA is bitwise this op's. Block size: the occupancy calculator's for each kernel
 * (a multiple of the warp size; rows are independent, so it changes no result).
 */
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

typedef __nv_bfloat16 bf16;

#define HEAD_DIM 128                                  /* nanochat's head_dim */
#define DIMS (2 * (int)(sizeof(uint4) / sizeof(bf16)))  /* dims per lane: two 16-byte vectors */
#define LANES (HEAD_DIM / DIMS)                       /* threads per row */
#define FULL_WARP 0xffffffffu       /* every thread of a launched warp reaches the shuffles (dead rows load zeros) */

static __device__ __forceinline__ float row_sum(float s) {
    for (int o = LANES / 2; o > 0; o >>= 1) s += __shfl_xor_sync(FULL_WARP, s, o);
    return s;
}

/* the lane's DIMS values of row `row` of x, as floats (zeros past the last row) */
static __device__ __forceinline__ void load(const bf16 *x, long long row, int lane, bool live, float *out) {
    uint4 raw[2] = {};
    if (live) {
        const uint4 *p = (const uint4 *)(x + row * HEAD_DIM + lane * DIMS);
        raw[0] = p[0], raw[1] = p[1];
    }
    const bf16 *b = (const bf16 *)raw;
    for (int i = 0; i < DIMS; i++) out[i] = __bfloat162float(b[i]);
}

static __device__ __forceinline__ void store(bf16 *x, long long row, int lane, const float *in) {
    uint4 raw[2];
    bf16 *b = (bf16 *)raw;
    for (int i = 0; i < DIMS; i++) b[i] = __float2bfloat16_rn(in[i]);
    uint4 *p = (uint4 *)(x + row * HEAD_DIM + lane * DIMS);
    p[0] = raw[0], p[1] = raw[1];
}

/* the gate of row `row` (1 on dead rows, which store nothing) */
static __device__ __forceinline__ float gate(const float *z, long long row, bool live) {
    return live ? __fdiv_rn(1.f, 1.f + expf(-z[row])) : 1.f;
}

__global__ void k_forward(const bf16 *y, const bf16 *v, const float *z, bf16 *out, long long rows) {
    const long long row = ((long long)blockIdx.x * blockDim.x + threadIdx.x) / LANES;
    const int lane = threadIdx.x % LANES;
    const bool live = row < rows;
    float yd[DIMS], vd[DIMS], o[DIMS];
    load(y, row, lane, live, yd);
    load(v, row, lane, live, vd);
    float dot = 0.f, sq = 0.f;
    for (int i = 0; i < DIMS; i++) dot = fmaf(yd[i], vd[i], dot), sq = fmaf(vd[i], vd[i], sq);
    dot = row_sum(dot), sq = row_sum(sq);
    if (!live) return;
    const float r = __fdiv_rn(dot, fmaxf(sq, 1e-8f));
    const float g = gate(z, row, live);
    for (int i = 0; i < DIMS; i++) o[i] = __fmul_rn(g, fmaf(-r, vd[i], yd[i]));
    store(out, row, lane, o);
}

__global__ void k_backward(const bf16 *g, const bf16 *y, const bf16 *v, const float *z, bf16 *dy, bf16 *dv, float *dz,
                           long long rows) {
    const long long row = ((long long)blockIdx.x * blockDim.x + threadIdx.x) / LANES;
    const int lane = threadIdx.x % LANES;
    const bool live = row < rows;
    float gd[DIMS], yd[DIMS], vd[DIMS], o[DIMS], G[DIMS];
    load(g, row, lane, live, gd);
    load(y, row, lane, live, yd);
    load(v, row, lane, live, vd);
    const float gate_value = gate(z, row, live);
    for (int i = 0; i < DIMS; i++) G[i] = gd[i], gd[i] = __fmul_rn(gate_value, gd[i]);  /* g' = g G; G kept for dz */
    float gv = 0.f, dot = 0.f, sq = 0.f;
    for (int i = 0; i < DIMS; i++) gv = fmaf(gd[i], vd[i], gv), dot = fmaf(yd[i], vd[i], dot), sq = fmaf(vd[i], vd[i], sq);
    gv = row_sum(gv), dot = row_sum(dot), sq = row_sum(sq);
    const float den = fmaxf(sq, 1e-8f), r = __fdiv_rn(dot, den);
    float gp = 0.f;  /* G . p: every lane of the row reaches the shuffles (dead rows: zeros) */
    for (int i = 0; i < DIMS; i++) gp = fmaf(G[i], fmaf(-r, vd[i], yd[i]), gp);
    gp = row_sum(gp);
    if (!live) return;
    const float c1 = __fdiv_rn(gv, den);
    const float c2 = sq >= 1e-8f ? __fmul_rn(2.f * c1, r) : 0.f;
    for (int i = 0; i < DIMS; i++) o[i] = fmaf(-c1, vd[i], gd[i]);
    store(dy, row, lane, o);
    for (int i = 0; i < DIMS; i++) o[i] = fmaf(c2, vd[i], -fmaf(r, gd[i], __fmul_rn(c1, yd[i])));
    store(dv, row, lane, o);
    if (lane == 0) dz[row] = __fmul_rn(__fmul_rn(gp, gate_value), 1.f - gate_value);
}

/* The occupancy calculator's block size for kernel f, computed once per kernel. */
template <typename F> static int block_size(F f, int &cached) {
    if (!cached) {
        int min_grid;
        cudaOccupancyMaxPotentialBlockSize(&min_grid, &cached, f);
    }
    return cached;
}

static unsigned blocks(long long rows, int threads) { return (unsigned)((rows * LANES + threads - 1) / threads); }

extern "C" void xsa_forward(const void *y, const void *v, const void *z, void *out, long long rows, void *stream) {
    static int cached = 0;
    const int threads = block_size(k_forward, cached);
    k_forward<<<blocks(rows, threads), threads, 0, (cudaStream_t)stream>>>((const bf16 *)y, (const bf16 *)v,
                                                                           (const float *)z, (bf16 *)out, rows);
}

extern "C" void xsa_backward(const void *g, const void *y, const void *v, const void *z, void *dy, void *dv, void *dz,
                            long long rows, void *stream) {
    static int cached = 0;
    const int threads = block_size(k_backward, cached);
    k_backward<<<blocks(rows, threads), threads, 0, (cudaStream_t)stream>>>(
        (const bf16 *)g, (const bf16 *)y, (const bf16 *)v, (const float *)z, (bf16 *)dy, (bf16 *)dv, (float *)dz, rows);
}
