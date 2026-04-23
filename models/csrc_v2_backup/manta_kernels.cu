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
 *
 * Optimizations applied:
 *   - __expf/__logf fast math intrinsics
 *   - __launch_bounds__ for register allocation control
 *   - Ballot-based validity bitmask (eliminates per-iteration nb_valid loads)
 *   - K-value caching in backward (eliminates redundant K loads)
 *   - Atomic-free backward: split into dQ (query-centric) + dKV (key-centric)
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>
#include <cstdint>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff
#define MAX_LOCAL 9  // max ceil(264/32) — supports E up to 288

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

// Load validity bitmask: lane 0 reads all nb_valid entries, broadcasts as uint32_t
__device__ __forceinline__ uint32_t load_valid_bitmask(
    const int8_t* __restrict__ nb_valid, int mask_base, int max_K, int lane
) {
    uint32_t bits = 0;
    if (lane == 0) {
        for (int ki = 0; ki < max_K && ki < 32; ki++) {
            if (ldg(&nb_valid[mask_base + ki]))
                bits |= (1u << ki);
        }
    }
    return __shfl_sync(FULL_MASK, bits, 0);
}

// ================================================================
//  MDNA — Self-Attention Forward
// ================================================================
__global__ void __launch_bounds__(256, 4) mdna_fwd_kernel(
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

    const int mask_base = h * N * max_K + n * max_K;
    const uint32_t valid_bits = load_valid_bitmask(nb_valid, mask_base, max_K, lane);

    for (int ki = 0; ki < max_K; ki++) {
        if (!(valid_bits & (1u << ki))) continue;
        const int m = ldg(&nb_idx[mask_base + ki]);
        const int64_t kv_base = ((int64_t)b * H + h) * N * E + m * E;

        float dot_partial = 0.0f;
        int idx = 0;
        for (int e = lane; e < E; e += WARP_SIZE)
            dot_partial += q_local[idx++] * ldg(&K[kv_base + e]);
        float score = warp_reduce_sum(dot_partial) * scale;

        const float new_max = fmaxf(run_max, score);
        const float exp_old = __expf(run_max - new_max);
        const float exp_new = __expf(score - new_max);
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
        Lse[((int64_t)b * H + h) * N + n] = run_max + __logf(run_sum + 1e-10f);
}

// ================================================================
//  MDNA — Self-Attention Backward: dQ only (query-centric, no atomics)
// ================================================================
__global__ void __launch_bounds__(256, 4) mdna_bwd_dq_kernel(
    const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V, const float* __restrict__ dO,
    const float* __restrict__ Lse, const float* __restrict__ D,
    const int32_t* __restrict__ nb_idx, const int8_t* __restrict__ nb_valid,
    float* __restrict__ dQ,
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

    const int mask_base = h * N * max_K + n * max_K;
    const uint32_t valid_bits = load_valid_bitmask(nb_valid, mask_base, max_K, lane);

    for (int ki = 0; ki < max_K; ki++) {
        if (!(valid_bits & (1u << ki))) continue;
        const int m = ldg(&nb_idx[mask_base + ki]);
        const int64_t kv_base = ((int64_t)b * H + h) * N * E + m * E;

        // Load K and compute dot product
        float k_local[MAX_LOCAL];
        float dot_partial = 0.0f;
        int idx = 0;
        for (int e = lane; e < E; e += WARP_SIZE) {
            float k_val = ldg(&K[kv_base + e]);
            k_local[idx] = k_val;
            dot_partial += q_local[idx] * k_val;
            idx++;
        }
        float score = warp_reduce_sum(dot_partial) * scale;
        float attn = __expf(score - lse_val);

        float dattn_partial = 0.0f;
        idx = 0;
        for (int e = lane; e < E; e += WARP_SIZE)
            dattn_partial += do_local[idx++] * ldg(&V[kv_base + e]);
        float d_attn = warp_reduce_sum(dattn_partial);
        float d_score = attn * (d_attn - d_val);

        // Accumulate dQ only (no atomics)
        idx = 0;
        for (int e = lane; e < E; e += WARP_SIZE) {
            dq_local[idx] += d_score * k_local[idx] * scale;
            idx++;
        }
    }

    int idx = 0;
    for (int e = lane; e < E; e += WARP_SIZE)
        dQ[q_base + e] = dq_local[idx++];
}

// ================================================================
//  MDNA — Self-Attention Backward: dK,dV only (key-centric, no atomics)
// ================================================================
__global__ void __launch_bounds__(256, 4) mdna_bwd_dkv_kernel(
    const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V, const float* __restrict__ dO,
    const float* __restrict__ Lse, const float* __restrict__ D,
    const int32_t* __restrict__ rev_nb_idx, const int8_t* __restrict__ rev_nb_valid,
    float* __restrict__ dK, float* __restrict__ dV,
    const int B, const int H, const int N,
    const int E, const int max_rev_K, const float scale
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int total = B * H * N;
    if (warp_id >= total) return;

    // Each warp handles one KEY position (b, h, m)
    int tmp = warp_id;
    const int m = tmp % N; tmp /= N;
    const int h = tmp % H; tmp /= H;
    const int b = tmp;
    const int64_t kv_base = ((int64_t)b * H + h) * N * E + m * E;

    // Load K_m and V_m into registers
    float k_local[MAX_LOCAL], v_local[MAX_LOCAL];
    float dk_local[MAX_LOCAL] = {0}, dv_local[MAX_LOCAL] = {0};
    int n_local = 0;
    for (int e = lane; e < E; e += WARP_SIZE) {
        k_local[n_local] = ldg(&K[kv_base + e]);
        v_local[n_local] = ldg(&V[kv_base + e]);
        n_local++;
    }

    // Preload reverse validity bitmask
    const int rev_mask_base = h * N * max_rev_K + m * max_rev_K;
    const uint32_t rev_valid_bits = load_valid_bitmask(rev_nb_valid, rev_mask_base, max_rev_K, lane);

    // For each query position that attends to this key
    for (int qi = 0; qi < max_rev_K; qi++) {
        if (!(rev_valid_bits & (1u << qi))) continue;
        const int n = ldg(&rev_nb_idx[rev_mask_base + qi]);
        const int64_t q_base = ((int64_t)b * H + h) * N * E + n * E;
        const int64_t l_off = ((int64_t)b * H + h) * N + n;
        const float lse_n = ldg(&Lse[l_off]);
        const float d_n = ldg(&D[l_off]);

        // Load Q_n and dO_n once, cache for reuse
        float q_cache[MAX_LOCAL], do_cache[MAX_LOCAL];
        float dot_partial = 0.0f, dattn_partial = 0.0f;
        int idx = 0;
        for (int e = lane; e < E; e += WARP_SIZE) {
            float q_val = ldg(&Q[q_base + e]);
            float do_val = ldg(&dO[q_base + e]);
            q_cache[idx] = q_val;
            do_cache[idx] = do_val;
            dot_partial += q_val * k_local[idx];
            dattn_partial += do_val * v_local[idx];
            idx++;
        }
        float score = warp_reduce_sum(dot_partial) * scale;
        float attn = __expf(score - lse_n);
        float d_attn = warp_reduce_sum(dattn_partial);
        float d_score = attn * (d_attn - d_n);

        // Accumulate dK_m and dV_m using cached Q/dO (no atomics!)
        idx = 0;
        for (int e = lane; e < E; e += WARP_SIZE) {
            dk_local[idx] += d_score * q_cache[idx] * scale;
            dv_local[idx] += attn * do_cache[idx];
            idx++;
        }
    }

    // Direct write — no atomics
    int idx = 0;
    for (int e = lane; e < E; e += WARP_SIZE) {
        dK[kv_base + e] = dk_local[idx];
        dV[kv_base + e] = dv_local[idx];
        idx++;
    }
}

// ================================================================
//  TANCA — Cross-Attention Forward
// ================================================================
__global__ void __launch_bounds__(256, 4) tanca_fwd_kernel(
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

    const int mask_base = h * N * max_K + n * max_K;
    const uint32_t valid_bits = load_valid_bitmask(nb_valid, mask_base, max_K, lane);

    for (int vkv = 0; vkv < Vkv; vkv++) {
        for (int ki = 0; ki < max_K; ki++) {
            if (!(valid_bits & (1u << ki))) continue;
            const int m = ldg(&nb_idx[mask_base + ki]);
            const int64_t kv_base = ((int64_t)(b * Vkv + vkv) * H + h) * N * E + m * E;

            float dot_partial = 0.0f;
            int idx = 0;
            for (int e = lane; e < E; e += WARP_SIZE)
                dot_partial += q_local[idx++] * ldg(&K[kv_base + e]);
            float score = warp_reduce_sum(dot_partial) * scale;

            const float new_max = fmaxf(run_max, score);
            const float exp_old = __expf(run_max - new_max);
            const float exp_new = __expf(score - new_max);
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
        Lse[((int64_t)(b * Vq + vq) * H + h) * N + n] = run_max + __logf(run_sum + 1e-10f);
}

// ================================================================
//  Shared: compute D_i = dot(dO, O)
// ================================================================
__global__ void __launch_bounds__(256, 4) compute_D_kernel(
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
//  TANCA — Cross-Attention Backward: dQ only (query-centric, no atomics)
// ================================================================
__global__ void __launch_bounds__(256, 4) tanca_bwd_dq_kernel(
    const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V, const float* __restrict__ dO,
    const float* __restrict__ Lse, const float* __restrict__ D,
    const int32_t* __restrict__ nb_idx, const int8_t* __restrict__ nb_valid,
    float* __restrict__ dQ,
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

    const int mask_base = h * N * max_K + n * max_K;
    const uint32_t valid_bits = load_valid_bitmask(nb_valid, mask_base, max_K, lane);

    for (int vkv = 0; vkv < Vkv; vkv++) {
        for (int ki = 0; ki < max_K; ki++) {
            if (!(valid_bits & (1u << ki))) continue;
            const int m = ldg(&nb_idx[mask_base + ki]);
            const int64_t kv_base = ((int64_t)(b * Vkv + vkv) * H + h) * N * E + m * E;

            float k_local[MAX_LOCAL];
            float dot_partial = 0.0f;
            int idx = 0;
            for (int e = lane; e < E; e += WARP_SIZE) {
                float k_val = ldg(&K[kv_base + e]);
                k_local[idx] = k_val;
                dot_partial += q_local[idx] * k_val;
                idx++;
            }
            float score = warp_reduce_sum(dot_partial) * scale;
            float attn = __expf(score - lse_val);

            float dattn_partial = 0.0f;
            idx = 0;
            for (int e = lane; e < E; e += WARP_SIZE)
                dattn_partial += do_local[idx++] * ldg(&V[kv_base + e]);
            float d_attn = warp_reduce_sum(dattn_partial);
            float d_score = attn * (d_attn - d_val);

            idx = 0;
            for (int e = lane; e < E; e += WARP_SIZE) {
                dq_local[idx] += d_score * k_local[idx] * scale;
                idx++;
            }
        }
    }

    int idx = 0;
    for (int e = lane; e < E; e += WARP_SIZE)
        dQ[q_base + e] = dq_local[idx++];
}

// ================================================================
//  TANCA — Cross-Attention Backward: dK,dV only (key-centric, no atomics)
// ================================================================
__global__ void __launch_bounds__(256, 4) tanca_bwd_dkv_kernel(
    const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V, const float* __restrict__ dO,
    const float* __restrict__ Lse, const float* __restrict__ D,
    const int32_t* __restrict__ rev_nb_idx, const int8_t* __restrict__ rev_nb_valid,
    float* __restrict__ dK, float* __restrict__ dV,
    const int B, const int Vq, const int Vkv,
    const int H, const int N, const int E, const int max_rev_K, const float scale
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    // Each warp handles one KV position (b, vkv, h, m)
    const int total = B * Vkv * H * N;
    if (warp_id >= total) return;

    int tmp = warp_id;
    const int m = tmp % N; tmp /= N;
    const int h = tmp % H; tmp /= H;
    const int vkv = tmp % Vkv; tmp /= Vkv;
    const int b = tmp;
    const int64_t kv_base = ((int64_t)(b * Vkv + vkv) * H + h) * N * E + m * E;

    // Load K_m and V_m into registers
    float k_local[MAX_LOCAL], v_local[MAX_LOCAL];
    float dk_local[MAX_LOCAL] = {0}, dv_local[MAX_LOCAL] = {0};
    int n_local = 0;
    for (int e = lane; e < E; e += WARP_SIZE) {
        k_local[n_local] = ldg(&K[kv_base + e]);
        v_local[n_local] = ldg(&V[kv_base + e]);
        n_local++;
    }

    // Preload reverse validity bitmask
    const int rev_mask_base = h * N * max_rev_K + m * max_rev_K;
    const uint32_t rev_valid_bits = load_valid_bitmask(rev_nb_valid, rev_mask_base, max_rev_K, lane);

    // Iterate over all query variables and positions that attend to this key
    for (int vq = 0; vq < Vq; vq++) {
        for (int qi = 0; qi < max_rev_K; qi++) {
            if (!(rev_valid_bits & (1u << qi))) continue;
            const int n = ldg(&rev_nb_idx[rev_mask_base + qi]);
            const int64_t q_base = ((int64_t)(b * Vq + vq) * H + h) * N * E + n * E;
            const int64_t l_off = ((int64_t)(b * Vq + vq) * H + h) * N + n;
            const float lse_n = ldg(&Lse[l_off]);
            const float d_n = ldg(&D[l_off]);

            // Load Q_n and dO_n once, cache for reuse
            float q_cache[MAX_LOCAL], do_cache[MAX_LOCAL];
            float dot_partial = 0.0f, dattn_partial = 0.0f;
            int idx = 0;
            for (int e = lane; e < E; e += WARP_SIZE) {
                float q_val = ldg(&Q[q_base + e]);
                float do_val = ldg(&dO[q_base + e]);
                q_cache[idx] = q_val;
                do_cache[idx] = do_val;
                dot_partial += q_val * k_local[idx];
                dattn_partial += do_val * v_local[idx];
                idx++;
            }
            float score = warp_reduce_sum(dot_partial) * scale;
            float attn = __expf(score - lse_n);
            float d_attn = warp_reduce_sum(dattn_partial);
            float d_score = attn * (d_attn - d_n);

            // Accumulate dK_m and dV_m using cached Q/dO
            idx = 0;
            for (int e = lane; e < E; e += WARP_SIZE) {
                dk_local[idx] += d_score * q_cache[idx] * scale;
                dv_local[idx] += attn * do_cache[idx];
                idx++;
            }
        }
    }

    // Direct write — no atomics
    int idx = 0;
    for (int e = lane; e < E; e += WARP_SIZE) {
        dK[kv_base + e] = dk_local[idx];
        dV[kv_base + e] = dv_local[idx];
        idx++;
    }
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
    int32_t* rev_nb_idx, int8_t* rev_nb_valid,
    float* dQ, float* dK, float* dV, float* Out,
    int B, int H, int N, int E, int max_K, int max_rev_K, float scale
) {
    const int warps_per_block = 8;
    const int thr = warps_per_block * WARP_SIZE;

    const int total_q = B * H * N;
    const int blk_q = (total_q + warps_per_block - 1) / warps_per_block;

    // Step 1: compute D_i = dot(dO, O)
    compute_D_kernel<<<blk_q, thr>>>(dO, Out, D_buf, total_q, E);

    // Step 2: dQ (query-centric, no atomics)
    mdna_bwd_dq_kernel<<<blk_q, thr>>>(
        Q, K, V, dO, Lse, D_buf, nb_idx, nb_valid, dQ,
        B, H, N, E, max_K, scale);

    // Step 3: dK, dV (key-centric, no atomics)
    // Same grid dimensions since total key positions = B*H*N
    mdna_bwd_dkv_kernel<<<blk_q, thr>>>(
        Q, K, V, dO, Lse, D_buf, rev_nb_idx, rev_nb_valid, dK, dV,
        B, H, N, E, max_rev_K, scale);
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
    int32_t* rev_nb_idx, int8_t* rev_nb_valid,
    float* dQ, float* dK, float* dV, float* Out,
    int B, int Vq, int Vkv, int H, int N, int E, int max_K, int max_rev_K, float scale
) {
    const int warps_per_block = 8;
    const int thr = warps_per_block * WARP_SIZE;

    // Step 1: compute D_i = dot(dO, O)
    const int total_q = B * Vq * H * N;
    const int blk_q = (total_q + warps_per_block - 1) / warps_per_block;
    compute_D_kernel<<<blk_q, thr>>>(dO, Out, D_buf, total_q, E);

    // Step 2: dQ (query-centric, no atomics)
    tanca_bwd_dq_kernel<<<blk_q, thr>>>(
        Q, K, V, dO, Lse, D_buf, nb_idx, nb_valid, dQ,
        B, Vq, Vkv, H, N, E, max_K, scale);

    // Step 3: dK, dV (key-centric, no atomics)
    const int total_kv = B * Vkv * H * N;
    const int blk_kv = (total_kv + warps_per_block - 1) / warps_per_block;
    tanca_bwd_dkv_kernel<<<blk_kv, thr>>>(
        Q, K, V, dO, Lse, D_buf, rev_nb_idx, rev_nb_valid, dK, dV,
        B, Vq, Vkv, H, N, E, max_rev_K, scale);
}

}  // extern "C"
