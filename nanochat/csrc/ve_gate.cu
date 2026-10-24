/*
 * Value-embedding gate and mix of model/gpt.py Attention (per-layer token-indexed value embeddings behind a per-head
 * gate), forward and backward, one pass over each row's memory:
 *   z = n W^T (fp32; W the gate's (H, C) fp32 weight), gate = 2 sigmoid(z),
 *   ve = bf16(table[idx]), out = bf16(v + gate[head] ve).
 * Backward from dout (dv = dout, passed through by the op):
 *   dgate[h] = sum over head h's dims of dout ve, dz = 2 dgate (1 - s) s with s = gate / 2 (sigmoid's derivative as
 *   torch's sigmoid_backward writes it), dn = bf16(dz W), dve = bf16(gate[head] dout) added into the table's fp32
 *   gradient at row idx (atomics, as the embedding backward), dW = dz^T n.
 * Inductor's fusion of the same expressions ran its z and dW reductions far below memory bandwidth.
 *
 * Layout: n, v, out, dout, dn contiguous (rows, C), C = H * HEAD_DIM; idx (rows,) int32; table, dtable (V, C) fp32;
 * gate, dz (rows, H) fp32. A row is one warp; lane l holds the VEC-wide chunks l, l + WARP, ... of the row, each inside
 * one head (HEAD_DIM is a multiple of VEC). H is a template parameter, so per-head arrays stay in registers: the
 * depths run, d8, d16, d24 (model/gpt.py gpt_config; conf/experiment), have H = 2, 4, 6.
 *
 * Deterministic dW without atomics: ve_dw's blocks each sum dz x n over a fixed range of rows per column (one thread
 * per column) into a per-block partial, and ve_dw_sum adds the partials in block order.
 */
#include "qk_row.cuh"

/* VEC fp32 of the table, each rounded to bf16 and back: ve = F.embedding(idx, table).to(bfloat16). */
static __device__ __forceinline__ void load_ve(const float *p, float *v) {
    load(p, v);
    for (int i = 0; i < VEC; i++) v[i] = __bfloat162float(__float2bfloat16(v[i]));
}

static __device__ __forceinline__ float warp_sum(float v) {
    for (int o = WARP / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

/* sum += x with Kahan's compensation err (Higham, "Accuracy and Stability of Numerical Algorithms", §4.3). */
static __device__ __forceinline__ void kahan_add(float &sum, float &err, float x) {
    const float y = x - err, t = sum + y;
    err = (t - sum) - y;
    sum = t;
}

constexpr int THREADS = 256;  /* rows (warps) per block: THREADS / WARP */

template <int H> __global__ void __launch_bounds__(THREADS)
ve_forward_kernel(const bf16 *n, const bf16 *v, const int *idx, const float *table, const float *w, bf16 *out,
                  float *gate, int64_t rows) {
    constexpr int C = H * HEAD_DIM, CHUNKS = C / (WARP * VEC);
    static_assert(C % (WARP * VEC) == 0, "a row is whole warp-wide chunks");
    __shared__ __align__(16) float ws[H * C];
    for (int i = threadIdx.x; i < H * C; i += blockDim.x) ws[i] = w[i];
    __syncthreads();
    const int lane = threadIdx.x % WARP;
    const int64_t warps = (int64_t)gridDim.x * (THREADS / WARP);
    for (int64_t r = (int64_t)blockIdx.x * (THREADS / WARP) + threadIdx.x / WARP; r < rows; r += warps) {
        float z[H] = {};
#pragma unroll
        for (int k = 0; k < CHUNKS; k++) {
            const int c = (k * WARP + lane) * VEC;
            float x[VEC];
            load(n + r * C + c, x);
#pragma unroll
            for (int h = 0; h < H; h++) {
                float wv[VEC];
                load(ws + h * C + c, wv);
                for (int j = 0; j < VEC; j++) z[h] += x[j] * wv[j];
            }
        }
        float g[H];
#pragma unroll
        for (int h = 0; h < H; h++) g[h] = 2.f * (1.f / (1.f + expf(-warp_sum(z[h]))));
#pragma unroll
        for (int h = 0; h < H; h++)
            if (lane == h) gate[r * H + h] = g[h];
        const float *row = table + (int64_t)idx[r] * C;
#pragma unroll
        for (int k = 0; k < CHUNKS; k++) {
            const int c = (k * WARP + lane) * VEC;
            float x[VEC], e[VEC], gh = 0.f;
            load(v + r * C + c, x);
            load_ve(row + c, e);
#pragma unroll
            for (int h = 0; h < H; h++)
                if (c / HEAD_DIM == h) gh = g[h];
            for (int j = 0; j < VEC; j++) x[j] = __fadd_rn(x[j], __fmul_rn(gh, e[j]));
            store(out + r * C + c, x);
        }
    }
}

template <int H> __global__ void __launch_bounds__(THREADS)
ve_backward_kernel(const bf16 *dout, const int *idx, const float *table, const float *w, const float *gate, bf16 *dn,
                   float *dtable, float *dz_out, int64_t rows) {
    constexpr int C = H * HEAD_DIM, CHUNKS = C / (WARP * VEC);
    __shared__ __align__(16) float ws[H * C];
    for (int i = threadIdx.x; i < H * C; i += blockDim.x) ws[i] = w[i];
    __syncthreads();
    const int lane = threadIdx.x % WARP;
    const int64_t warps = (int64_t)gridDim.x * (THREADS / WARP);
    for (int64_t r = (int64_t)blockIdx.x * (THREADS / WARP) + threadIdx.x / WARP; r < rows; r += warps) {
        float g[H], dg[H] = {};
#pragma unroll
        for (int h = 0; h < H; h++) g[h] = gate[r * H + h];
        const int64_t at = (int64_t)idx[r] * C;
#pragma unroll
        for (int k = 0; k < CHUNKS; k++) {
            const int c = (k * WARP + lane) * VEC;
            float d[VEC], e[VEC], gh = 0.f, s = 0.f;
            load(dout + r * C + c, d);
            load_ve(table + at + c, e);
#pragma unroll
            for (int h = 0; h < H; h++)
                if (c / HEAD_DIM == h) gh = g[h];
            for (int j = 0; j < VEC; j++) {
                s += d[j] * e[j];
                d[j] = __bfloat162float(__float2bfloat16(gh * d[j]));  /* dve */
            }
#pragma unroll
            for (int h = 0; h < H; h++)
                if (c / HEAD_DIM == h) dg[h] += s;
            float4 *t = reinterpret_cast<float4 *>(dtable + at + c);
            for (int j = 0; j < VEC; j += 4) atomicAdd(t + j / 4, make_float4(d[j], d[j + 1], d[j + 2], d[j + 3]));
        }
        float dz[H];
#pragma unroll
        for (int h = 0; h < H; h++) {
            const float sg = g[h] * 0.5f;
            dz[h] = 2.f * warp_sum(dg[h]) * (1.f - sg) * sg;
            if (lane == h) dz_out[r * H + h] = dz[h];
        }
#pragma unroll
        for (int k = 0; k < CHUNKS; k++) {
            const int c = (k * WARP + lane) * VEC;
            float x[VEC] = {};
#pragma unroll
            for (int h = 0; h < H; h++) {
                float wv[VEC];
                load(ws + h * C + c, wv);
                for (int j = 0; j < VEC; j++) x[j] += dz[h] * wv[j];
            }
            store(dn + r * C + c, x);
        }
    }
}

/* One thread per column c: dz[h] n[c] summed over this block's rows, per head, into part (gridDim.x, H, C). */
template <int H> __global__ void ve_dw_kernel(const bf16 *n, const float *dz, float *part, int64_t rows) {
    constexpr int C = H * HEAD_DIM;
    const int c = threadIdx.x;
    const int64_t per = (rows + gridDim.x - 1) / gridDim.x, first = (int64_t)blockIdx.x * per;
    const int64_t last = first + per < rows ? first + per : rows;
    float acc[H] = {}, err[H] = {};  /* compensated (Kahan) sums: rows run long, the products' magnitudes vary */
    for (int64_t r = first; r < last; r++) {
        const float x = __bfloat162float(n[r * C + c]);
#pragma unroll
        for (int h = 0; h < H; h++) kahan_add(acc[h], err[h], dz[r * H + h] * x);
    }
#pragma unroll
    for (int h = 0; h < H; h++) part[((int64_t)blockIdx.x * H + h) * C + c] = acc[h];
}

/* dW[i]: entry i of the partials summed in block order, compensated. */
__global__ void ve_dw_sum_kernel(const float *part, int nb, int HC, float *dw) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= HC) return;
    float sum = 0.f, err = 0.f;
    for (int b = 0; b < nb; b++) kahan_add(sum, err, part[(int64_t)b * HC + i]);
    dw[i] = sum;
}

static int sms() {
    static const int count = [] {
        int device, c;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&c, cudaDevAttrMultiProcessorCount, device);
        return c;
    }();
    return count;
}

/* Persistent grids: resident blocks of a kernel, all SMs. */
template <typename F> static int grid(F f) {
    int per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, f, THREADS, 0);
    return per_sm * sms();
}

/* Blocks of ve_dw, fixed per device, so the summation order is fixed; part: fp32 (ve_dw_blocks(), H, C). */
extern "C" int ve_dw_blocks() { return 2 * sms(); }

template <int H> static void forward(const void *n, const void *v, const void *idx, const void *table, const void *w,
                                     void *out, void *gate, long long rows, cudaStream_t s) {
    static const int blocks = grid(ve_forward_kernel<H>);
    ve_forward_kernel<H><<<blocks, THREADS, 0, s>>>((const bf16 *)n, (const bf16 *)v, (const int *)idx,
                                                    (const float *)table, (const float *)w, (bf16 *)out,
                                                    (float *)gate, rows);
}

template <int H> static void backward(const void *dout, const void *n, const void *idx, const void *table,
                                      const void *w, const void *gate, void *dn, void *dtable, long long table_rows,
                                      void *dw, void *dz, void *part, long long rows, cudaStream_t s) {
    constexpr int C = H * HEAD_DIM;
    static const int blocks = grid(ve_backward_kernel<H>);
    const int nb = ve_dw_blocks();
    cudaMemsetAsync(dtable, 0, sizeof(float) * table_rows * C, s);
    ve_backward_kernel<H><<<blocks, THREADS, 0, s>>>((const bf16 *)dout, (const int *)idx, (const float *)table,
                                                     (const float *)w, (const float *)gate, (bf16 *)dn,
                                                     (float *)dtable, (float *)dz, rows);
    ve_dw_kernel<H><<<nb, C, 0, s>>>((const bf16 *)n, (const float *)dz, (float *)part, rows);
    ve_dw_sum_kernel<<<(H * C + THREADS - 1) / THREADS, THREADS, 0, s>>>((const float *)part, nb, H * C, (float *)dw);
}

/* H: heads per row (C = H * HEAD_DIM); 0 when H is not compiled (2, 4, 6). */
extern "C" int ve_forward(const void *n, const void *v, const void *idx, const void *table, const void *w, void *out,
                          void *gate, long long rows, int H, void *stream) {
    const cudaStream_t s = (cudaStream_t)stream;
    switch (H) {
        case 2: forward<2>(n, v, idx, table, w, out, gate, rows, s); return 1;
        case 4: forward<4>(n, v, idx, table, w, out, gate, rows, s); return 1;
        case 6: forward<6>(n, v, idx, table, w, out, gate, rows, s); return 1;
        default: return 0;
    }
}

/* dtable: fp32 (table_rows, C), zeroed here; dz: fp32 (rows, H) scratch; part: fp32 (ve_dw_blocks(), H, C). */
extern "C" int ve_backward(const void *dout, const void *n, const void *idx, const void *table, const void *w,
                           const void *gate, void *dn, void *dtable, long long table_rows, void *dw, void *dz,
                           void *part, long long rows, int H, void *stream) {
    const cudaStream_t s = (cudaStream_t)stream;
    switch (H) {
        case 2: backward<2>(dout, n, idx, table, w, gate, dn, dtable, table_rows, dw, dz, part, rows, s); return 1;
        case 4: backward<4>(dout, n, idx, table, w, gate, dn, dtable, table_rows, dw, dz, part, rows, s); return 1;
        case 6: backward<6>(dout, n, idx, table, w, gate, dn, dtable, table_rows, dw, dz, part, rows, s); return 1;
        default: return 0;
    }
}
