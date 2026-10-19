/*
 * The token and value embeddings of model/gpt.py: the lookup of fp32 table rows as bf16, and the table's gradient
 * without atomics, in a fixed summation order (bitwise reproducible; Inductor's embedding backward adds the rows with
 * fp32 atomics, in arrival order).
 *
 * Gradient: the positions sorted by token (stable, ops/embedding.py) form one run per token; each run is cut into
 * partials of at most PARTIAL_ROWS positions (k_partials: one block per partial, the rows added in position order),
 * then each table row sums its token's partials in order (k_rows; 0 for tokens absent from the batch). The partials
 * bound the longest serial sum (a frequent token's run) at PARTIAL_ROWS rows in k_partials.
 */
#include "qk_row.cuh"

/* out (n, C) bf16 = table rows idx (n,): block i its row, one thread per VEC dims (two float4 loads, one 16-byte store). */
__global__ void k_gather(const float *table, const int *idx, bf16 *out, int C) {
    const int64_t i = blockIdx.x;
    const int o = threadIdx.x * VEC;
    float v[VEC];
    load(table + (int64_t)idx[i] * C + o, v);
    store(out + i * C + o, v);
}

/* Partial p of each sorted position j that starts one (start[j]): the sum, in position order, of the gradient rows
 * order[j], order[j + 1], ... while the token is tokens[j], at most PARTIAL_ROWS rows; written to row slot[j]. */
__global__ void k_partials(const bf16 *grad, const int64_t *order, const int *tokens, const bool *start,
                           const int *slot, float *partial, int64_t n, int C, int partial_rows) {
    const int64_t j = blockIdx.x;
    if (!start[j]) return;
    const int64_t end = j + partial_rows < n ? j + partial_rows : n;
    for (int o = threadIdx.x * VEC; o < C; o += blockDim.x * VEC) {
        float acc[VEC] = {}, row[VEC];
        for (int64_t r = j; r < end && tokens[r] == tokens[j]; r++) {
            load(grad + order[r] * C + o, row);
            for (int i = 0; i < VEC; i++) acc[i] += row[i];
        }
        for (int i = 0; i < VEC; i += 4)
            *reinterpret_cast<float4 *>(partial + (int64_t)slot[j] * C + o + i) = *reinterpret_cast<float4 *>(acc + i);
    }
}

/* Row v of the gradient: the partials first[v] .. first[v + 1] - 1 (its token's, in position order) summed in order. */
__global__ void k_rows(const float *partial, const int *first, float *out, int C) {
    const int v = blockIdx.x;
    for (int o = threadIdx.x * VEC; o < C; o += blockDim.x * VEC) {
        float acc[VEC] = {}, row[VEC];
        for (int p = first[v]; p < first[v + 1]; p++) {
            load(partial + (int64_t)p * C + o, row);
            for (int i = 0; i < VEC; i++) acc[i] += row[i];
        }
        for (int i = 0; i < VEC; i += 4)
            *reinterpret_cast<float4 *>(out + (int64_t)v * C + o + i) = *reinterpret_cast<float4 *>(acc + i);
    }
}

extern "C" void emb_forward(const void *table, const void *idx, void *out, long long n, int C, void *stream) {
    k_gather<<<(unsigned)n, C / VEC, 0, (cudaStream_t)stream>>>((const float *)table, (const int *)idx, (bf16 *)out, C);
}

extern "C" void emb_backward(const void *grad, const void *order, const void *tokens, const void *start,
                             const void *slot, const void *first, void *partial, void *out, long long n, int rows,
                             int C, int partial_rows, void *stream) {
    const int threads = C / VEC;  /* one thread per VEC dims of a row */
    k_partials<<<(unsigned)n, threads, 0, (cudaStream_t)stream>>>(
        (const bf16 *)grad, (const int64_t *)order, (const int *)tokens, (const bool *)start, (const int *)slot,
        (float *)partial, n, C, partial_rows);
    k_rows<<<rows, threads, 0, (cudaStream_t)stream>>>((const float *)partial, (const int *)first, (float *)out, C);
}
