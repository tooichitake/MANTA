/*
 * MANTA CUDA Kernels — Warp-parallel, single-pass online softmax.
 *
 *   1. MDNA  — Multi-scale Dilated Neighborhood Attention (self-attention)
 *   2. TANCA — Temporally-Aligned Neighborhood Cross-Attention
 *
 * Each warp (32 threads) cooperates on one query position:
 *   - Each lane handles ceil(E/32) elements
 *   - Dot products via __shfl_down_sync (5 shuffles, no shared memory)
 *   - Single-pass online softmax with per-lane output accumulation
 *   - __ldg() for read-only data via texture cache
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>
#include <cstdint>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff
#define MAX_LOCAL 4  // max ceil(128/32)

// Read-only load via texture/L1 cache
__device__ __forceinline__ float ldg(const float* p) { return __ldg(p); }
__device__ __forceinline__ int32_t ldg(const int32_t* p) { return __ldg(p); }
__device__ __forceinline__ int8_t ldg(const int8_t* p) { return __ldg(p); }

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(FULL_MASK, val, offset);
    return __shfl_sync(FULL_MASK, val, 0);
}

// ================================================================
//  MDNA — Self-Attention Forward
// ================================================================
__global__ void mdna_fwd_kernel(
    const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V,
    const int32_t* __restrict__ nb_idx, const int8_t* __restrict__ nb_valid,
    float* __restrict__ Out, float* __restrict__ Lse,
    const int B, const int H, const int N,
    const int E, const int max_K, const float scale
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int total = B * H * N;
    if (warp_id >= total) return;

    int tmp = warp_id;
    const int n = tmp % N; tmp /= N;
    const int h = tmp % H; tmp /= H;
    const int b = tmp;
    const int64_t q_base = ((int64_t)b * H + h) * N * E + n * E;

    float q_local[MAX_LOCAL];
    int n_local = 0;
    for (int e = lane; e < E; e += WARP_SIZE)
        q_local[n_local++] = ldg(&Q[q_base + e]);

    float run_max = -FLT_MAX, run_sum = 0.0f;
    float out_local[MAX_LOCAL] = {0};

    for (int ki = 0; ki < max_K; ki++) {
        const int mask_off = h * N * max_K + n * max_K + ki;
        if (!ldg(&nb_valid[mask_off])) continue;
        const int m = ldg(&nb_idx[mask_off]);
        const int64_t kv_base = ((int64_t)b * H + h) * N * E + m * E;

        float dot_partial = 0.0f;
        int idx = 0;
        for (int e = lane; e < E; e += WARP_SIZE)
            dot_partial += q_local[idx++] * ldg(&K[kv_base + e]);
        float score = warp_reduce_sum(dot_partial) * scale;

        const float new_max = fmaxf(run_max, score);
        const float exp_old = expf(run_max - new_max);
        const float exp_new = expf(score - new_max);
        idx = 0;
        for (int e = lane; e < E; e += WARP_SIZE) {
            out_local[idx] = out_local[idx] * exp_old + exp_new * ldg(&V[kv_base + e]);
            idx++;
        }
        run_sum = run_sum * exp_old + exp_new;
        run_max = new_max;
    }

    if (run_sum > 0.0f) {
        const float inv = 1.0f / run_sum;
        int idx = 0;
        for (int e = lane; e < E; e += WARP_SIZE)
            Out[q_base + e] = out_local[idx++] * inv;
    }
    if (lane == 0)
        Lse[((int64_t)b * H + h) * N + n] = run_max + logf(run_sum + 1e-10f);
}

// ================================================================
//  MDNA — Self-Attention Backward
// ================================================================
__global__ void mdna_bwd_kernel(
    const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V, const float* __restrict__ dO,
    const float* __restrict__ Lse, const float* __restrict__ D,
    const int32_t* __restrict__ nb_idx, const int8_t* __restrict__ nb_valid,
    float* __restrict__ dQ, float* __restrict__ dK, float* __restrict__ dV,
    const int B, const int H, const int N,
    const int E, const int max_K, const float scale
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int total = B * H * N;
    if (warp_id >= total) return;

    int tmp = warp_id;
    const int n = tmp % N; tmp /= N;
    const int h = tmp % H; tmp /= H;
    const int b = tmp;
    const int64_t q_base = ((int64_t)b * H + h) * N * E + n * E;
    const int64_t l_off = ((int64_t)b * H + h) * N + n;
    const float lse_val = ldg(&Lse[l_off]), d_val = ldg(&D[l_off]);

    float q_local[MAX_LOCAL], do_local[MAX_LOCAL], dq_local[MAX_LOCAL];
    int n_local = 0;
    for (int e = lane; e < E; e += WARP_SIZE) {
        q_local[n_local] = ldg(&Q[q_base + e]);
        do_local[n_local] = ldg(&dO[q_base + e]);
        dq_local[n_local] = 0.0f;
        n_local++;
    }

    for (int ki = 0; ki < max_K; ki++) {
        const int mask_off = h * N * max_K + n * max_K + ki;
        if (!ldg(&nb_valid[mask_off])) continue;
        const int m = ldg(&nb_idx[mask_off]);
        const int64_t kv_base = ((int64_t)b * H + h) * N * E + m * E;

        float dot_partial = 0.0f;
        int idx = 0;
        for (int e = lane; e < E; e += WARP_SIZE)
            dot_partial += q_local[idx++] * ldg(&K[kv_base + e]);
        float score = warp_reduce_sum(dot_partial) * scale;
        float attn = expf(score - lse_val);

        float dattn_partial = 0.0f;
        idx = 0;
        for (int e = lane; e < E; e += WARP_SIZE)
            dattn_partial += do_local[idx++] * ldg(&V[kv_base + e]);
        float d_attn = warp_reduce_sum(dattn_partial);
        float d_score = attn * (d_attn - d_val);

        idx = 0;
        for (int e = lane; e < E; e += WARP_SIZE) {
            dq_local[idx] += d_score * ldg(&K[kv_base + e]) * scale;
            atomicAdd(&dK[kv_base + e], d_score * q_local[idx] * scale);
            atomicAdd(&dV[kv_base + e], attn * do_local[idx]);
            idx++;
        }
    }

    int idx = 0;
    for (int e = lane; e < E; e += WARP_SIZE)
        dQ[q_base + e] = dq_local[idx++];
}

// ================================================================
//  TANCA — Cross-Attention Forward
// ================================================================
__global__ void tanca_fwd_kernel(
    const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V,
    const int32_t* __restrict__ nb_idx, const int8_t* __restrict__ nb_valid,
    float* __restrict__ Out, float* __restrict__ Lse,
    const int B, const int Vq, const int Vkv,
    const int H, const int N, const int E, const int max_K, const float scale
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int total = B * Vq * H * N;
    if (warp_id >= total) return;

    int tmp = warp_id;
    const int n = tmp % N; tmp /= N;
    const int h = tmp % H; tmp /= H;
    const int vq = tmp % Vq; tmp /= Vq;
    const int b = tmp;
    const int64_t q_base = ((int64_t)(b * Vq + vq) * H + h) * N * E + n * E;

    float q_local[MAX_LOCAL];
    int n_local = 0;
    for (int e = lane; e < E; e += WARP_SIZE)
        q_local[n_local++] = ldg(&Q[q_base + e]);

    float run_max = -FLT_MAX, run_sum = 0.0f;
    float out_local[MAX_LOCAL] = {0};

    for (int vkv = 0; vkv < Vkv; vkv++) {
        for (int ki = 0; ki < max_K; ki++) {
            const int mask_off = h * N * max_K + n * max_K + ki;
            if (!ldg(&nb_valid[mask_off])) continue;
            const int m = ldg(&nb_idx[mask_off]);
            const int64_t kv_base = ((int64_t)(b * Vkv + vkv) * H + h) * N * E + m * E;

            float dot_partial = 0.0f;
            int idx = 0;
            for (int e = lane; e < E; e += WARP_SIZE)
                dot_partial += q_local[idx++] * ldg(&K[kv_base + e]);
            float score = warp_reduce_sum(dot_partial) * scale;

            const float new_max = fmaxf(run_max, score);
            const float exp_old = expf(run_max - new_max);
            const float exp_new = expf(score - new_max);
            idx = 0;
            for (int e = lane; e < E; e += WARP_SIZE) {
                out_local[idx] = out_local[idx] * exp_old + exp_new * ldg(&V[kv_base + e]);
                idx++;
            }
            run_sum = run_sum * exp_old + exp_new;
            run_max = new_max;
        }
    }

    if (run_sum > 0.0f) {
        const float inv = 1.0f / run_sum;
        int idx = 0;
        for (int e = lane; e < E; e += WARP_SIZE)
            Out[q_base + e] = out_local[idx++] * inv;
    }
    if (lane == 0)
        Lse[((int64_t)(b * Vq + vq) * H + h) * N + n] = run_max + logf(run_sum + 1e-10f);
}

// ================================================================
//  Shared: compute D_i = dot(dO, O)
// ================================================================
__global__ void compute_D_kernel(
    const float* __restrict__ dO, const float* __restrict__ O,
    float* __restrict__ D, const int total, const int E
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    if (warp_id >= total) return;
    float partial = 0.0f;
    const int64_t base = (int64_t)warp_id * E;
    for (int e = lane; e < E; e += WARP_SIZE)
        partial += ldg(&dO[base + e]) * ldg(&O[base + e]);
    float result = warp_reduce_sum(partial);
    if (lane == 0) D[warp_id] = result;
}

// ================================================================
//  TANCA — Cross-Attention Backward
// ================================================================
__global__ void tanca_bwd_kernel(
    const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V, const float* __restrict__ dO,
    const float* __restrict__ Lse, const float* __restrict__ D,
    const int32_t* __restrict__ nb_idx, const int8_t* __restrict__ nb_valid,
    float* __restrict__ dQ, float* __restrict__ dK, float* __restrict__ dV,
    const int B, const int Vq, const int Vkv,
    const int H, const int N, const int E, const int max_K, const float scale
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int total = B * Vq * H * N;
    if (warp_id >= total) return;

    int tmp = warp_id;
    const int n = tmp % N; tmp /= N;
    const int h = tmp % H; tmp /= H;
    const int vq = tmp % Vq; tmp /= Vq;
    const int b = tmp;
    const int64_t q_base = ((int64_t)(b * Vq + vq) * H + h) * N * E + n * E;
    const int64_t l_off = ((int64_t)(b * Vq + vq) * H + h) * N + n;
    const float lse_val = ldg(&Lse[l_off]), d_val = ldg(&D[l_off]);

    float q_local[MAX_LOCAL], do_local[MAX_LOCAL], dq_local[MAX_LOCAL];
    int n_local = 0;
    for (int e = lane; e < E; e += WARP_SIZE) {
        q_local[n_local] = ldg(&Q[q_base + e]);
        do_local[n_local] = ldg(&dO[q_base + e]);
        dq_local[n_local] = 0.0f;
        n_local++;
    }

    for (int vkv = 0; vkv < Vkv; vkv++) {
        for (int ki = 0; ki < max_K; ki++) {
            const int mask_off = h * N * max_K + n * max_K + ki;
            if (!ldg(&nb_valid[mask_off])) continue;
            const int m = ldg(&nb_idx[mask_off]);
            const int64_t kv_base = ((int64_t)(b * Vkv + vkv) * H + h) * N * E + m * E;

            float dot_partial = 0.0f;
            int idx = 0;
            for (int e = lane; e < E; e += WARP_SIZE)
                dot_partial += q_local[idx++] * ldg(&K[kv_base + e]);
            float score = warp_reduce_sum(dot_partial) * scale;
            float attn = expf(score - lse_val);

            float dattn_partial = 0.0f;
            idx = 0;
            for (int e = lane; e < E; e += WARP_SIZE)
                dattn_partial += do_local[idx++] * ldg(&V[kv_base + e]);
            float d_attn = warp_reduce_sum(dattn_partial);
            float d_score = attn * (d_attn - d_val);

            idx = 0;
            for (int e = lane; e < E; e += WARP_SIZE) {
                dq_local[idx] += d_score * ldg(&K[kv_base + e]) * scale;
                atomicAdd(&dK[kv_base + e], d_score * q_local[idx] * scale);
                atomicAdd(&dV[kv_base + e], attn * do_local[idx]);
                idx++;
            }
        }
    }

    int idx = 0;
    for (int e = lane; e < E; e += WARP_SIZE)
        dQ[q_base + e] = dq_local[idx++];
}

// ================================================================
//  C-linkage launch functions
// ================================================================
extern "C" {

void mdna_forward_launch(
    float* Q, float* K, float* V,
    int32_t* nb_idx, int8_t* nb_valid,
    float* Out, float* Lse,
    int B, int H, int N, int E, int max_K, float scale
) {
    const int total_warps = B * H * N;
    const int warps_per_block = 8;
    const int thr = warps_per_block * WARP_SIZE;
    const int blk = (total_warps + warps_per_block - 1) / warps_per_block;
    mdna_fwd_kernel<<<blk, thr>>>(
        Q, K, V, nb_idx, nb_valid, Out, Lse, B, H, N, E, max_K, scale);
}

void mdna_backward_launch(
    float* Q, float* K, float* V, float* dO,
    float* Lse, float* D_buf,
    int32_t* nb_idx, int8_t* nb_valid,
    float* dQ, float* dK, float* dV, float* Out,
    int B, int H, int N, int E, int max_K, float scale
) {
    const int total_warps = B * H * N;
    const int warps_per_block = 8;
    const int thr = warps_per_block * WARP_SIZE;
    const int blk = (total_warps + warps_per_block - 1) / warps_per_block;
    compute_D_kernel<<<blk, thr>>>(dO, Out, D_buf, total_warps, E);
    mdna_bwd_kernel<<<blk, thr>>>(
        Q, K, V, dO, Lse, D_buf, nb_idx, nb_valid, dQ, dK, dV,
        B, H, N, E, max_K, scale);
}

void tanca_forward_launch(
    float* Q, float* K, float* V,
    int32_t* nb_idx, int8_t* nb_valid,
    float* Out, float* Lse,
    int B, int Vq, int Vkv, int H, int N, int E, int max_K, float scale
) {
    const int total_warps = B * Vq * H * N;
    const int warps_per_block = 8;
    const int thr = warps_per_block * WARP_SIZE;
    const int blk = (total_warps + warps_per_block - 1) / warps_per_block;
    tanca_fwd_kernel<<<blk, thr>>>(
        Q, K, V, nb_idx, nb_valid, Out, Lse, B, Vq, Vkv, H, N, E, max_K, scale);
}

void tanca_backward_launch(
    float* Q, float* K, float* V, float* dO,
    float* Lse, float* D_buf,
    int32_t* nb_idx, int8_t* nb_valid,
    float* dQ, float* dK, float* dV, float* Out,
    int B, int Vq, int Vkv, int H, int N, int E, int max_K, float scale
) {
    const int total_warps = B * Vq * H * N;
    const int warps_per_block = 8;
    const int thr = warps_per_block * WARP_SIZE;
    const int blk = (total_warps + warps_per_block - 1) / warps_per_block;
    compute_D_kernel<<<blk, thr>>>(dO, Out, D_buf, total_warps, E);
    tanca_bwd_kernel<<<blk, thr>>>(
        Q, K, V, dO, Lse, D_buf, nb_idx, nb_valid, dQ, dK, dV,
        B, Vq, Vkv, H, N, E, max_K, scale);
}

}  // extern "C"
