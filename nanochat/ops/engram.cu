// The Engram fusion of DeepSeek's paper (arXiv 2601.07372 v2: Eq. 4 context-aware gating, Eq. 5 short convolution),
// forward and backward (nanochat/ops/engram.py).
//
// Fusion, per token t (h: residual stream, C; k, v: the two halves of kv = W_kv x, C each; w_h, w_k, w_v: RMSNorm
// weights; W: the depthwise causal convolution, C x TAPS, dilation dil):
//   gate (Eq. 4):  r_h = rsqrt(sum(h^2) / C + eps), r_k = rsqrt(sum(k^2) / C + eps)
//                  s = (sum_i (h_i w_h,i) (k_i w_k,i)) r_h r_k / sqrt(C) = RMSNorm_h(h) . RMSNorm_k(k) / sqrt(C)
//                  alpha = sigmoid(s),  vt = bf16(alpha v)                                      (V~ = alpha v)
//                  r_v = rsqrt(sum(vt^2) / C + eps),  u = bf16(vt r_v w_v)                      (RMSNorm(V~))
//   conv (Eq. 5):  c_i = sum_k W[i, k] u[t - (TAPS - 1 - k) dil]_i over the taps inside t's row and segment,
//                  out = bf16(h + (silu(c) + vt))                                            (H + Y, Y = SiLU(conv) + V~)
// Kernels: gate_fwd (one warp per token: the three reductions) writes vt, u and r_v; conv_fwd (one thread per 8-channel
// vector of a token) reads the taps of other tokens. Rounding points: h, kv bf16 in; every sum and product in fp32 with
// explicit _rn intrinsics (no contraction choices left to the compiler); vt, u and out rounded once to bf16. Gate sums:
// lane l of a token's warp takes the 16-byte vectors c = l, l + 32, ... in order, element 0..7 of each, accumulating with
// fmaf; then one xor-butterfly over the warp (offsets 16, 8, 4, 2, 1). Conv sum: taps k = 0..TAPS-1 in order (k =
// TAPS-1 the token itself), fmaf from 0, skipped taps (before the row's or the segment's first token) contribute nothing.
// nanochat/decode.cu mirrors these orders exactly.
// Backward (a = RMSNorm_h(h), b = RMSNorm_k(k); dout the gradient of out):
//   conv_bwd:  dc_t = dout_t silu'(c_t); du_t = sum_k W[., k] dc_{t + lag_k} (the taps that read u_t);
//              dW[., k] = sum_t dc_t u_{t - lag_k};  dw_v = sum_t du_t vt_t r_v,t
//   gate_bwd:  g = du w_v;  dvt = dout + r_v g - r_v^3 vt (g . vt) / C;  dv = alpha dvt;  ds = (dvt . v) alpha (1 - alpha)
//              dh = dout + ds r_h / sqrt(C) (b w_h - h r_h s / sqrt(C)),  dk = ds r_k / sqrt(C) (a w_k - k r_k s / sqrt(C))
//              dw_h = P w_k, dw_k = P w_h with P_i = sum over tokens of ds r_h r_k / sqrt(C) h_i k_i
// (sum(a b) = s sqrt(C) removes the RMSNorm backward's own reductions). The weight gradients are reduced
// deterministically: per block in a fixed order into one partial row, the partial rows by the caller (a torch sum).
//
// Layouts: activations (N, C) and (N, 2C) bf16 row-major move as 16-byte vectors (8 bf16), C a multiple of 128 (gpt.py);
// the convolution's weights (C, TAPS) fp32 row-major, one float4 per channel; segment starts (N,) bytes (1 = the token
// starts an attention segment) or NULL (every row one segment). One warp per row for the row reductions (gating), one
// thread per vector for the convolution. Launch shapes come from the CUDA occupancy calculator per
// kernel (cudaOccupancyMaxPotentialBlockSize*: the block size of the highest occupancy and the grid that fills the
// device once, from its SM count, registers and shared memory); the backward's tokens per warp are those that cover the
// tokens in that one wave, so its partial-sum rows (and their summation order) are fixed per device and token count.
//
// Built by nanochat.native.cuda_library (nvcc -O3 -arch=sm_<cc>), called through ctypes; every launcher takes the
// caller's stream.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>

#include <map>
#include <mutex>

typedef __nv_bfloat16 bf16;
static constexpr int WARP = 32;
static constexpr int VEC = 8;                  // bf16 per 16-byte vector
static constexpr int TAPS = 4;                 // convolution kernel size (paper Appendix A; nanochat.ops.engram.ENGRAM_CONV_KERNEL)
static constexpr int CONV_SUMS = TAPS + 1;     // conv backward's per-channel sums: dW's taps, then dw_v

// A kernel's launch shape: block size and the grid of one full wave (cudaOccupancyMaxPotentialBlockSizeVariableSMem
// with smem_per_thread bytes of dynamic shared memory per thread, up to the device's opt-in limit per block), computed
// once per kernel and smem_per_thread.
struct Launch {
    int block, grid;
};

struct SmemOfBlock {   // the occupancy calculator's block size -> dynamic shared memory functor
    size_t per_thread;
    __host__ __device__ size_t operator()(int block) const { return (size_t)block * per_thread; }
};

template <auto Kernel>
static Launch launch_of(size_t smem_per_thread) {
    static std::mutex mu;
    static std::map<size_t, Launch> shapes;
    std::lock_guard<std::mutex> lock(mu);
    auto it = shapes.find(smem_per_thread);
    if (it != shapes.end()) return it->second;
    Launch r;
    int device, optin;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    cudaFuncSetAttribute(Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, optin);
    cudaOccupancyMaxPotentialBlockSizeVariableSMem(&r.grid, &r.block, Kernel, SmemOfBlock{smem_per_thread});
    return shapes[smem_per_thread] = r;
}

static constexpr float RMS_EPS = FLT_EPSILON;  // nanochat's norms: torch.finfo(torch.float32).eps

static __device__ __forceinline__ void unpack(uint4 v, float f[VEC]) {
    const __nv_bfloat162 *p = reinterpret_cast<const __nv_bfloat162 *>(&v);
#pragma unroll
    for (int i = 0; i < VEC / 2; i++) {
        float2 t = __bfloat1622float2(p[i]);
        f[2 * i] = t.x;
        f[2 * i + 1] = t.y;
    }
}

static __device__ __forceinline__ uint4 pack(const float f[VEC]) {
    uint4 v;
    __nv_bfloat162 *p = reinterpret_cast<__nv_bfloat162 *>(&v);
#pragma unroll
    for (int i = 0; i < VEC / 2; i++) p[i] = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
    return v;
}

// The 8 fp32 values of vector c of a (C,) fp32 array.
static __device__ __forceinline__ void load_weights(const float4 *w, int c, float f[VEC]) {
    float4 a = __ldg(w + 2 * c), b = __ldg(w + 2 * c + 1);
    f[0] = a.x; f[1] = a.y; f[2] = a.z; f[3] = a.w;
    f[4] = b.x; f[5] = b.y; f[6] = b.z; f[7] = b.w;
}

// The 8 fp32 values of vector c of a (C,) fp32 array in shared memory.
static __device__ __forceinline__ void load_shared(const float4 *w, int c, float f[VEC]) {
    float4 a = w[2 * c], b = w[2 * c + 1];
    f[0] = a.x; f[1] = a.y; f[2] = a.z; f[3] = a.w;
    f[4] = b.x; f[5] = b.y; f[6] = b.z; f[7] = b.w;
}

static __device__ __forceinline__ float tap(float4 w, int k) { return k == 0 ? w.x : k == 1 ? w.y : k == 2 ? w.z : w.w; }

static __device__ __forceinline__ float warp_sum(float s) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, o));
    return s;
}

static __device__ __forceinline__ float sigmoid(float x) { return __fdiv_rn(1.f, __fadd_rn(1.f, expf(-x))); }

static unsigned blocks_for(int64_t threads, int block) { return (unsigned)((threads + block - 1) / block); }

// Kernel over `threads` threads (a thread or a warp per item: results independent of the block size) at its occupancy
// calculator's block size.
template <auto Kernel, typename... Args>
static void launch(int64_t threads, cudaStream_t stream, Args... args) {
    const Launch l = launch_of<Kernel>(0);
    Kernel<<<blocks_for(threads, l.block), l.block, 0, stream>>>(args...);
}

// The gate of one token from its warp-summed sum(h^2), sum(k^2) and dot: r_h, r_k, s and alpha = sigmoid(s) (Eq. 4).
struct Gate {
    float rh, rk, s, g;
};

static __device__ __forceinline__ Gate gate_of(float ssh, float ssk, float dot, int c) {
    Gate t;
    float fc = (float)c;
    t.rh = rsqrtf(__fadd_rn(__fdiv_rn(ssh, fc), RMS_EPS));
    t.rk = rsqrtf(__fadd_rn(__fdiv_rn(ssk, fc), RMS_EPS));
    t.s = __fmul_rn(__fmul_rn(__fmul_rn(dot, t.rh), t.rk), rsqrtf(fc));
    t.g = sigmoid(t.s);
    return t;
}

// The tokens before t (position pos in its row) that its taps may read, at most max_lag: back to the row's first token
// or to t's segment's first token (starts[t - d] = 1), whichever is nearer.
static __device__ __forceinline__ int reach_of(const uint8_t *starts, int64_t t, int pos, int max_lag) {
    int r = min(pos, max_lag);
    if (starts)
        for (int d = 0; d < r; d++)
            if (starts[t - d]) return d;
    return r;
}

// The convolution's pre-activation c of vector c of token t (8 channels; w: their TAPS weights), taps up to `reach`
// tokens back.
static __device__ __forceinline__ void conv_at(const uint4 *u, const float4 w[VEC], int64_t t, int reach, int nv, int c,
                                               int dil, float acc[VEC]) {
    float f[VEC];
#pragma unroll
    for (int i = 0; i < VEC; i++) acc[i] = 0.f;
#pragma unroll
    for (int k = 0; k < TAPS; k++) {
        int lag = (TAPS - 1 - k) * dil;
        if (lag > reach) continue;
        unpack(u[(t - lag) * nv + c], f);
#pragma unroll
        for (int i = 0; i < VEC; i++) acc[i] = __fmaf_rn(tap(w[i], k), f[i], acc[i]);
    }
}

// One warp per token; V = vectors per lane (ceil(C / 8 / 32)). vt stays in registers between its sum and u.
template <int V>
__global__ void gate_fwd_kernel(const uint4 *h, const uint4 *kv, const float4 *wh, const float4 *wk, const float4 *wv,
                                uint4 *vt, uint4 *u, float *rv, float *alpha, int64_t n, int nv) {
    int64_t t = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / WARP;
    int lane = threadIdx.x % WARP;
    if (t >= n) return;
    const uint4 *ht = h + t * nv, *kt = kv + t * 2 * nv, *vp = kt + nv;
    float ssh = 0.f, ssk = 0.f, dot = 0.f, ssv = 0.f, a[VEC], b[VEC], x[VEC], y[VEC];
#pragma unroll
    for (int j = 0; j < V; j++) {
        int c = lane + j * WARP;
        if (c >= nv) break;
        unpack(ht[c], a);
        unpack(kt[c], b);
        load_weights(wh, c, x);
        load_weights(wk, c, y);
#pragma unroll
        for (int i = 0; i < VEC; i++) {
            ssh = __fmaf_rn(a[i], a[i], ssh);
            ssk = __fmaf_rn(b[i], b[i], ssk);
            dot = __fmaf_rn(__fmul_rn(a[i], x[i]), __fmul_rn(b[i], y[i]), dot);
        }
    }
    Gate gt = gate_of(warp_sum(ssh), warp_sum(ssk), warp_sum(dot), nv * VEC);
    uint4 vv[V];
#pragma unroll
    for (int j = 0; j < V; j++) {
        int c = lane + j * WARP;
        if (c >= nv) break;
        unpack(vp[c], b);
#pragma unroll
        for (int i = 0; i < VEC; i++) a[i] = __fmul_rn(gt.g, b[i]);
        vv[j] = pack(a);
        unpack(vv[j], a);
#pragma unroll
        for (int i = 0; i < VEC; i++) ssv = __fmaf_rn(a[i], a[i], ssv);
    }
    float r = rsqrtf(__fadd_rn(__fdiv_rn(warp_sum(ssv), (float)(nv * VEC)), RMS_EPS));
#pragma unroll
    for (int j = 0; j < V; j++) {
        int c = lane + j * WARP;
        if (c >= nv) break;
        unpack(vv[j], a);
        load_weights(wv, c, x);
#pragma unroll
        for (int i = 0; i < VEC; i++) a[i] = __fmul_rn(__fmul_rn(a[i], r), x[i]);
        vt[t * nv + c] = vv[j];
        u[t * nv + c] = pack(a);
    }
    if (lane == 0) {
        rv[t] = r;
        if (alpha) alpha[t] = gt.g;
    }
}

// One thread per 8-channel vector c of token t (vectors = N * nv); row length T.
__global__ void conv_fwd_kernel(const uint4 *h, const uint4 *vt, const uint4 *u, const float4 *w, const uint8_t *starts,
                                uint4 *out, int64_t vectors, int nv, int T, int dil) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= vectors) return;
    int64_t t = i / nv;
    int c = (int)(i % nv);
    float4 wc[VEC];
#pragma unroll
    for (int e = 0; e < VEC; e++) wc[e] = __ldg(w + c * VEC + e);
    float acc[VEC], a[VEC], b[VEC];
    conv_at(u, wc, t, reach_of(starts, t, (int)(t % T), (TAPS - 1) * dil), nv, c, dil, acc);
    unpack(h[i], a);
    unpack(vt[i], b);
#pragma unroll
    for (int e = 0; e < VEC; e++) a[e] = __fadd_rn(a[e], __fadd_rn(__fdiv_rn(acc[e], __fadd_rn(1.f, expf(-acc[e]))), b[e]));
    out[i] = pack(a);
}

// Conv backward. Block (x, y) of W warps: lane l takes vector c = 32 x + l, warp w the tpw consecutive tokens from
// (y * W + w) * tpw; per token du (fp32) and the running sums dW, dw_v (in token order); the block's sums (warps in
// order) are its row of `partial` (C x CONV_SUMS floats: channel major, then dW's taps, then dw_v). Dynamic shared
// memory: W x WARP x VEC x CONV_SUMS floats.
__global__ void conv_bwd_kernel(const uint4 *dout, const uint4 *vt, const uint4 *u, const float *rv, const float4 *w,
                                const uint8_t *starts, float4 *du, float *partial, int64_t n, int nv, int T, int dil,
                                int tpw) {
    extern __shared__ float smem[];   // [warp][lane][VEC * CONV_SUMS]
    const int warps = blockDim.x / WARP, row = VEC * CONV_SUMS;
    int warp = threadIdx.x / WARP, lane = threadIdx.x % WARP, c = blockIdx.x * WARP + lane;
    int max_lag = (TAPS - 1) * dil;
    float sums[VEC][CONV_SUMS];
#pragma unroll
    for (int e = 0; e < VEC; e++)
#pragma unroll
        for (int q = 0; q < CONV_SUMS; q++) sums[e][q] = 0.f;
    if (c < nv) {
        float4 wc[VEC];
#pragma unroll
        for (int e = 0; e < VEC; e++) wc[e] = __ldg(w + c * VEC + e);
        int64_t first = ((int64_t)blockIdx.y * warps + warp) * tpw;
        for (int64_t t = first; t < first + tpw && t < n; t++) {
            int pos = (int)(t % T);
            float g[VEC], dct[VEC], cp[VEC], f[VEC];
#pragma unroll
            for (int e = 0; e < VEC; e++) g[e] = 0.f;
            // du_t: tap k of token p = t + lag_k reads u_t, if p is in t's row and reaches back to t
#pragma unroll
            for (int k = 0; k < TAPS; k++) {
                int lag = (TAPS - 1 - k) * dil;
                if (pos + lag >= T) continue;
                int64_t p = t + lag;
                int reach = reach_of(starts, p, pos + lag, max_lag);
                if (lag > reach) continue;
                conv_at(u, wc, p, reach, nv, c, dil, cp);
                unpack(dout[p * nv + c], f);
#pragma unroll
                for (int e = 0; e < VEC; e++) {
                    float sg = sigmoid(cp[e]);
                    float dc = __fmul_rn(f[e], __fmul_rn(sg, __fadd_rn(1.f, __fmul_rn(cp[e], __fsub_rn(1.f, sg)))));
                    g[e] = __fmaf_rn(tap(wc[e], k), dc, g[e]);
                    if (k == TAPS - 1) dct[e] = dc;  // lag 0: dc of t itself
                }
            }
            // dW: dc_t times the taps t read
            int reach = reach_of(starts, t, pos, max_lag);
#pragma unroll
            for (int k = 0; k < TAPS; k++) {
                int lag = (TAPS - 1 - k) * dil;
                if (lag > reach) continue;
                unpack(u[(t - lag) * nv + c], f);
#pragma unroll
                for (int e = 0; e < VEC; e++) sums[e][k] = __fmaf_rn(dct[e], f[e], sums[e][k]);
            }
            // dw_v: du_t vt_t r_v,t
            float r = rv[t];
            unpack(vt[t * nv + c], f);
#pragma unroll
            for (int e = 0; e < VEC; e++) sums[e][TAPS] = __fmaf_rn(g[e], __fmul_rn(f[e], r), sums[e][TAPS]);
            du[(t * nv + c) * 2] = make_float4(g[0], g[1], g[2], g[3]);
            du[(t * nv + c) * 2 + 1] = make_float4(g[4], g[5], g[6], g[7]);
        }
    }
#pragma unroll
    for (int e = 0; e < VEC; e++)
#pragma unroll
        for (int q = 0; q < CONV_SUMS; q++) smem[(warp * WARP + lane) * row + e * CONV_SUMS + q] = sums[e][q];
    __syncthreads();
    for (int o = threadIdx.x; o < WARP * row; o += blockDim.x) {
        int l = o / row, q = o % row, col = blockIdx.x * WARP + l;
        if (col >= nv) continue;
        float s = smem[l * row + q];
        for (int k = 1; k < warps; k++) s = __fadd_rn(s, smem[(k * WARP + l) * row + q]);
        partial[((int64_t)blockIdx.y * nv + col) * row + q] = s;
    }
}

// Gate backward: a block's W warps take tpw consecutive tokens each (block b: tokens from b * W * tpw).
// Per token three passes over its vectors: the forward's gate sums (same order as gate_fwd_kernel) with dout . v and
// g . v; then vt (recomputed as in the forward) for g . vt and vt . v; then dh, dk, dv and the warp's running P. The
// first pass reads the token's h, k, v, dout and du from global memory once and keeps them in the warp's shared-memory
// stage (each lane its own vectors), which the later passes read, so the kernel's DRAM traffic is its compulsory 18 C
// bytes per token (it is bound by that traffic; re-reading the 12 C bytes per token through L1/L2 measured 7.4 vs
// 6.8-7.0 ms per call at C 1280, M 65,536 on GB10). The block's P (warps summed in order) is its row of `partial`
// (C floats). Dynamic shared memory: W x 12 C bytes (h, k, v, dout bf16, du fp32), its start reused for the P sums
// (W x WARP x VEC floats).
template <int V>
__global__ void gate_bwd_kernel(const uint4 *dout, const float4 *du, const uint4 *h, const uint4 *kv, const float *rv,
                                const float4 *wh, const float4 *wk, const float4 *wv, uint4 *dh, uint4 *dkv,
                                float *partial, int64_t n, int nv, int tpw) {
    extern __shared__ uint4 stage[];
    const int warps = blockDim.x / WARP;
    int warp = threadIdx.x / WARP, lane = threadIdx.x % WARP, c_dim = nv * VEC;
    uint4 *sh = stage + (size_t)warp * 6 * nv, *sk = sh + nv, *sv = sk + nv, *sd = sv + nv;
    float4 *sg = reinterpret_cast<float4 *>(sd + nv);
    float p[V][VEC];
#pragma unroll
    for (int j = 0; j < V; j++)
#pragma unroll
        for (int i = 0; i < VEC; i++) p[j][i] = 0.f;
    float fc = (float)c_dim, inv_sqrt_c = rsqrtf(fc);
    int64_t first = ((int64_t)blockIdx.x * warps + warp) * tpw;
    for (int64_t t = first; t < first + tpw && t < n; t++) {
        const uint4 *ht = h + t * nv, *kt = kv + t * 2 * nv, *vp = kt + nv, *dt = dout + t * nv;
        const float4 *gt4 = du + t * nv * 2;
        float ssh = 0.f, ssk = 0.f, dot = 0.f, sdv = 0.f, sgv = 0.f, sgvt = 0.f, svtv = 0.f;
        float a[VEC], b[VEC], x[VEC], y[VEC], z[VEC], vf[VEC], df[VEC];
#pragma unroll
        for (int j = 0; j < V; j++) {
            int c = lane + j * WARP;
            if (c >= nv) break;
            uint4 hq = ht[c], kq = kt[c], vq = vp[c], dq = dt[c];
            float4 g0 = gt4[2 * c], g1 = gt4[2 * c + 1];
            sh[c] = hq, sk[c] = kq, sv[c] = vq, sd[c] = dq, sg[2 * c] = g0, sg[2 * c + 1] = g1;
            unpack(hq, a);
            unpack(kq, b);
            load_weights(wh, c, x);
            load_weights(wk, c, y);
#pragma unroll
            for (int i = 0; i < VEC; i++) {
                ssh = __fmaf_rn(a[i], a[i], ssh);
                ssk = __fmaf_rn(b[i], b[i], ssk);
                dot = __fmaf_rn(__fmul_rn(a[i], x[i]), __fmul_rn(b[i], y[i]), dot);
            }
            unpack(vq, vf);
            unpack(dq, df);
            load_shared(sg, c, a);
            load_weights(wv, c, z);
#pragma unroll
            for (int i = 0; i < VEC; i++) {
                sdv = __fmaf_rn(df[i], vf[i], sdv);
                sgv = __fmaf_rn(__fmul_rn(a[i], z[i]), vf[i], sgv);
            }
        }
        Gate gt = gate_of(warp_sum(ssh), warp_sum(ssk), warp_sum(dot), c_dim);
#pragma unroll
        for (int j = 0; j < V; j++) {
            int c = lane + j * WARP;
            if (c >= nv) break;
            unpack(sv[c], vf);
            load_shared(sg, c, a);
            load_weights(wv, c, z);
#pragma unroll
            for (int i = 0; i < VEC; i++) b[i] = __fmul_rn(gt.g, vf[i]);
            unpack(pack(b), b);  // vt as the forward rounds it
#pragma unroll
            for (int i = 0; i < VEC; i++) {
                sgvt = __fmaf_rn(__fmul_rn(a[i], z[i]), b[i], sgvt);
                svtv = __fmaf_rn(b[i], vf[i], svtv);
            }
        }
        sdv = warp_sum(sdv), sgv = warp_sum(sgv), sgvt = warp_sum(sgvt), svtv = warp_sum(svtv);
        float r = rv[t];
        float coef = __fdiv_rn(__fmul_rn(__fmul_rn(__fmul_rn(r, r), r), sgvt), fc);    // r_v^3 (g . vt) / C
        float dalpha = __fsub_rn(__fmaf_rn(r, sgv, sdv), __fmul_rn(coef, svtv));     // dvt . v
        float ds = __fmul_rn(dalpha, __fmul_rn(gt.g, __fsub_rn(1.f, gt.g)));
        float ch = __fmul_rn(__fmul_rn(ds, gt.rh), inv_sqrt_c), ck = __fmul_rn(__fmul_rn(ds, gt.rk), inv_sqrt_c);
        float sc = __fmul_rn(gt.s, inv_sqrt_c);
        float hs = __fmul_rn(gt.rh, sc), ks = __fmul_rn(gt.rk, sc);   // r_h s / sqrt(C), r_k s / sqrt(C)
        float m = __fmul_rn(ch, gt.rk);                                // ds r_h r_k / sqrt(C)
#pragma unroll
        for (int j = 0; j < V; j++) {
            int c = lane + j * WARP;
            if (c >= nv) break;
            float hf[VEC], kf[VEC], gf[VEC], o[VEC];
            unpack(sh[c], hf);
            unpack(sk[c], kf);
            unpack(sv[c], vf);
            unpack(sd[c], df);
            load_shared(sg, c, gf);
            load_weights(wh, c, x);
            load_weights(wk, c, y);
            load_weights(wv, c, z);
#pragma unroll
            for (int i = 0; i < VEC; i++) b[i] = __fmul_rn(gt.g, vf[i]);
            unpack(pack(b), b);
#pragma unroll
            for (int i = 0; i < VEC; i++) {
                float an = __fmul_rn(__fmul_rn(hf[i], gt.rh), x[i]);   // a_i
                float bn = __fmul_rn(__fmul_rn(kf[i], gt.rk), y[i]);   // b_i
                o[i] = __fmaf_rn(ch, __fsub_rn(__fmul_rn(bn, x[i]), __fmul_rn(hf[i], hs)), df[i]);
                p[j][i] = __fmaf_rn(m, __fmul_rn(hf[i], kf[i]), p[j][i]);
                a[i] = __fmul_rn(ck, __fsub_rn(__fmul_rn(an, y[i]), __fmul_rn(kf[i], ks)));
                float dvt = __fsub_rn(__fmaf_rn(r, __fmul_rn(gf[i], z[i]), df[i]), __fmul_rn(coef, b[i]));
                vf[i] = __fmul_rn(gt.g, dvt);
            }
            dh[t * nv + c] = pack(o);
            dkv[t * 2 * nv + c] = pack(a);
            dkv[t * 2 * nv + nv + c] = pack(vf);
        }
    }
    // Block sum of the warps' P, WARP * VEC columns (one vector per lane) at a time, in the stage's first bytes.
    float *sums = reinterpret_cast<float *>(stage);
    __syncthreads();
#pragma unroll
    for (int j = 0; j < V; j++) {
        if (j * WARP * VEC >= c_dim) break;
#pragma unroll
        for (int i = 0; i < VEC; i++) sums[warp * WARP * VEC + lane * VEC + i] = p[j][i];
        __syncthreads();
        for (int o = threadIdx.x; o < WARP * VEC; o += blockDim.x) {
            int col = j * WARP * VEC + o;
            if (col >= c_dim) break;
            float s = sums[o];
            for (int w = 1; w < warps; w++) s = __fadd_rn(s, sums[w * WARP * VEC + o]);
            partial[(int64_t)blockIdx.x * c_dim + col] = s;
        }
        __syncthreads();
    }
}

// The backward's shape for n tokens over gx block columns: blocks of the kernel's occupancy block size; the block rows
// of one wave, floor(grid / gx), so that every block is resident at once (a row more would start a second, almost
// empty wave: at C 1280, 5 columns x 20 rows = 100 blocks on 96 slots ran 1.04 waves, 9.4 vs 7.0 ms); tokens per warp
// such that those rows cover the n tokens, and the block rows that then cover them. Slots a row of gx blocks leaves
// unused stay idle (C 1280: 45 of 48); the block size of equal occupancy that fills them best (1 warp: 575 of 576) ran
// slower, conv_bwd 5.38 vs 5.12 ms per call at M 65,536.
struct BwdShape {
    int block, tpw;
    int64_t rows;
};

static BwdShape bwd_shape(Launch l, int64_t n, int64_t gx) {
    int64_t warps = l.block / WARP, wave_rows = l.grid / gx;
    int64_t tpw = (n + wave_rows * warps - 1) / (wave_rows * warps);
    return BwdShape{l.block, (int)tpw, (n + warps * tpw - 1) / (warps * tpw)};
}

static BwdShape conv_bwd_shape(int64_t n, int c) {
    return bwd_shape(launch_of<conv_bwd_kernel>(WARP * VEC * CONV_SUMS * sizeof(float) / WARP), n,
                     (c / VEC + WARP - 1) / WARP);
}

// gate_bwd_kernel's stage: 6 16-byte vectors per channel vector (h, k, v, dout bf16; du fp32) per warp.
static size_t gate_bwd_smem_per_thread(int nv) { return (size_t)6 * nv * sizeof(uint4) / WARP; }

template <int V>
static BwdShape gate_bwd_shape(int64_t n, int nv) {
    return bwd_shape(launch_of<gate_bwd_kernel<V>>(gate_bwd_smem_per_thread(nv)), n, 1);
}

template <int V>
static void gate_fwd(const void *h, const void *kv, const float *wh, const float *wk, const float *wv, void *vt, void *u,
                     float *rv, float *alpha, int64_t n, int nv, cudaStream_t stream) {
    launch<gate_fwd_kernel<V>>(n * WARP, stream, (const uint4 *)h, (const uint4 *)kv, (const float4 *)wh,
                               (const float4 *)wk, (const float4 *)wv, (uint4 *)vt, (uint4 *)u, rv, alpha, n, nv);
}

template <int V>
static void gate_bwd(const void *dout, const float *du, const void *h, const void *kv, const float *rv, const float *wh,
                     const float *wk, const float *wv, void *dh, void *dkv, float *partial, int64_t n, int nv,
                     cudaStream_t stream) {
    BwdShape s = gate_bwd_shape<V>(n, nv);
    gate_bwd_kernel<V><<<(unsigned)s.rows, s.block, s.block * gate_bwd_smem_per_thread(nv), stream>>>(
        (const uint4 *)dout, (const float4 *)du, (const uint4 *)h, (const uint4 *)kv, rv, (const float4 *)wh,
        (const float4 *)wk, (const float4 *)wv, (uint4 *)dh, (uint4 *)dkv, partial, n, nv, s.tpw);
}

// The gating kernels for V = 1..8 vectors per lane: C <= 8 * 32 * 8 = 2048 (up to d32).
#define DISPATCH(V_EXPR, CALL)                                                                                         \
    switch (V_EXPR) {                                                                                                  \
        case 1: return CALL(1);                                                                                        \
        case 2: return CALL(2);                                                                                        \
        case 3: return CALL(3);                                                                                        \
        case 4: return CALL(4);                                                                                        \
        case 5: return CALL(5);                                                                                        \
        case 6: return CALL(6);                                                                                        \
        case 7: return CALL(7);                                                                                        \
        default: return CALL(8);                                                                                       \
    }

static int lane_vectors(int c) { return (c / VEC + WARP - 1) / WARP; }

extern "C" {

// vt, u (n, c) bf16, rv (n,) fp32 and, if alpha is not null, the gate alpha (n,) fp32 of h (n, c), kv (n, 2c) bf16 and
// the RMSNorm weights wh, wk, wv (c,) fp32.
void ec_gate_fwd(const void *h, const void *kv, const float *wh, const float *wk, const float *wv, void *vt, void *u,
                 float *rv, float *alpha, int64_t n, int c, void *stream) {
#define FWD(V) gate_fwd<V>(h, kv, wh, wk, wv, vt, u, rv, alpha, n, c / VEC, (cudaStream_t)stream)
    DISPATCH(lane_vectors(c), FWD)
#undef FWD
}

// out (n, c) bf16 of h, vt, u (n, c) bf16 in rows of t tokens, the convolution's weights w (c, TAPS) fp32, dilation
// dil and the segment starts (n bytes, or NULL).
void ec_conv_fwd(const void *h, const void *vt, const void *u, const float *w, const uint8_t *starts, void *out,
                 int64_t n, int t, int c, int dil, void *stream) {
    int64_t vectors = n * (c / VEC);
    launch<conv_fwd_kernel>(vectors, (cudaStream_t)stream, (const uint4 *)h, (const uint4 *)vt, (const uint4 *)u,
                            (const float4 *)w, starts, (uint4 *)out, vectors, c / VEC, t, dil);
}

// Rows of the conv backward's partial sums for n tokens of width c (its workspace `partial` holds rows * c * (TAPS + 1)
// floats).
int64_t ec_conv_partials(int64_t n, int c) { return conv_bwd_shape(n, c).rows; }

// du (n, c) fp32 and the partial sums of dW and dw_v for the gradient dout (n, c) of ec_conv_fwd's out.
void ec_conv_bwd(const void *dout, const void *vt, const void *u, const float *rv, const float *w, const uint8_t *starts,
                 float *du, float *partial, int64_t n, int t, int c, int dil, void *stream) {
    int nv = c / VEC;
    BwdShape s = conv_bwd_shape(n, c);
    dim3 grid((unsigned)((nv + WARP - 1) / WARP), (unsigned)s.rows);
    conv_bwd_kernel<<<grid, s.block, s.block * VEC * CONV_SUMS * sizeof(float), (cudaStream_t)stream>>>(
        (const uint4 *)dout, (const uint4 *)vt, (const uint4 *)u, rv, (const float4 *)w, starts, (float4 *)du, partial, n,
        nv, t, dil, s.tpw);
}

// Rows of the gate backward's partial sums of P for n tokens of width c (its workspace `partial` holds rows * c floats).
int64_t ec_gate_partials(int64_t n, int c) {
#define ROWS(V) gate_bwd_shape<V>(n, c / VEC).rows
    DISPATCH(lane_vectors(c), ROWS)
#undef ROWS
}

// dh (n, c), dkv (n, 2c) bf16 and the partial sums of P from dout (n, c) bf16 and du (n, c) fp32.
void ec_gate_bwd(const void *dout, const float *du, const void *h, const void *kv, const float *rv, const float *wh,
                 const float *wk, const float *wv, void *dh, void *dkv, float *partial, int64_t n, int c, void *stream) {
#define BWD(V) gate_bwd<V>(dout, du, h, kv, rv, wh, wk, wv, dh, dkv, partial, n, c / VEC, (cudaStream_t)stream)
    DISPATCH(lane_vectors(c), BWD)
#undef BWD
}

}  // extern "C"
