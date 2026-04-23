/*
 * MANTA CUDA Kernels v3 — Shared Memory Tiling + Online Softmax.
 *
 * Optimizations inspired by FlashAttention 1/2/3 and NATTEN/FNA:
 *   - Shared memory tiling: K/V (or Q/dO) loaded into SMEM once per block,
 *     reused by all N warps. Eliminates ~7x redundant HBM loads (MDNA)
 *     and N-fold reduction for TANCA.
 *   - Online softmax: single-pass, O(1) per-position memory (unchanged).
 *   - Atomic-free backward: dQ (query-centric) + dKV (key-centric) split.
 *   - Ballot-based validity bitmask.
 *   - __expf/__logf fast math intrinsics.
 *
 * Grid mapping:
 *   MDNA:  1 block per (b,h),     N warps per block
 *   TANCA: 1 block per (b,v,h),   N warps per block
 *
 * Compatible with SM80+ (Ampere and later).
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>  // cp.async (SM80+)
#include <cmath>
#include <cfloat>
#include <cstdint>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

// Template dispatch: CALL is a macro that takes MAX_E as argument
#define DISPATCH_E(E_val, CALL) \
    do { \
        if      ((E_val) <= 32)  { CALL(32);  } \
        else if ((E_val) <= 64)  { CALL(64);  } \
        else if ((E_val) <= 128) { CALL(128); } \
        else if ((E_val) <= 192) { CALL(192); } \
        else                     { CALL(288); } \
    } while (0)

// ================================================================
//  Device helpers
// ================================================================

__device__ __forceinline__ float ldg(const float* p) { return __ldg(p); }
__device__ __forceinline__ float safe_expf(float x) { return __expf(fminf(fmaxf(x, -50.0f), 50.0f)); }
__device__ __forceinline__ int32_t ldg(const int32_t* p) { return __ldg(p); }
__device__ __forceinline__ int8_t ldg(const int8_t* p) { return __ldg(p); }

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(FULL_MASK, val, offset);
    return __shfl_sync(FULL_MASK, val, 0);
}

// Load validity bitmask: lane 0 reads, broadcasts via __shfl
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

// Cooperative load: all threads in block load count floats from global to SMEM
// Uses float4 (16-byte) vectorized loads when both pointers are aligned
__device__ __forceinline__ void coop_load(
    float* __restrict__ dst, const float* __restrict__ src, int count
) {
    const bool aligned = ((reinterpret_cast<uintptr_t>(dst) | reinterpret_cast<uintptr_t>(src)) & 15) == 0;
    if (aligned) {
        const int count4 = count / 4;
        float4* dst4 = reinterpret_cast<float4*>(dst);
        const float4* src4 = reinterpret_cast<const float4*>(src);
        for (int i = threadIdx.x; i < count4; i += blockDim.x)
            dst4[i] = __ldg(&src4[i]);
        const int base = count4 * 4;
        for (int i = base + threadIdx.x; i < count; i += blockDim.x)
            dst[i] = ldg(&src[i]);
    } else {
        for (int i = threadIdx.x; i < count; i += blockDim.x)
            dst[i] = ldg(&src[i]);
    }
}

// Asynchronous cooperative load: cp.async global→shared (SM80+)
// Uses 16-byte cp.async when aligned, falls back to 4-byte otherwise
__device__ __forceinline__ void coop_load_async(
    float* __restrict__ dst, const float* __restrict__ src, int count
) {
    const bool aligned = ((reinterpret_cast<uintptr_t>(dst) | reinterpret_cast<uintptr_t>(src)) & 15) == 0;
    if (aligned) {
        const int count4 = count / 4;
        for (int i = threadIdx.x; i < count4; i += blockDim.x)
            __pipeline_memcpy_async(
                reinterpret_cast<float4*>(dst) + i,
                reinterpret_cast<const float4*>(src) + i,
                sizeof(float4));
        const int base = count4 * 4;
        for (int i = base + threadIdx.x; i < count; i += blockDim.x)
            __pipeline_memcpy_async(&dst[i], &src[i], sizeof(float));
    } else {
        for (int i = threadIdx.x; i < count; i += blockDim.x)
            __pipeline_memcpy_async(&dst[i], &src[i], sizeof(float));
    }
}

// ================================================================
//  MDNA — Self-Attention Forward (Shared Memory)
// ================================================================
//  1 block per (b,h). N warps in block, one per query position.
//  SMEM holds K[N,E] + V[N,E] — loaded once, read by all warps.
// ================================================================
template <int MAX_E>
__global__ void __launch_bounds__(384, 4) mdna_fwd_kernel(
    const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V,
    const int32_t* __restrict__ nb_idx, const int8_t* __restrict__ nb_valid,
    float* __restrict__ Out, float* __restrict__ Lse,
    const int B, const int H, const int N,
    const int E, const int max_K, const float scale
) {
    constexpr int LOCAL = (MAX_E + WARP_SIZE - 1) / WARP_SIZE;
    extern __shared__ float smem[];
    float* smem_K = smem;           // [N * E]
    float* smem_V = smem + N * E;   // [N * E]

    const int block_id = blockIdx.x;  // = b * H + h
    if (block_id >= B * H) return;
    const int h = block_id % H;
    const int b = block_id / H;

    const int warp_in_block = threadIdx.x / WARP_SIZE;  // = query position n
    const int lane = threadIdx.x % WARP_SIZE;
    const int n = warp_in_block;  // each warp handles one query position

    // Base offset for this (b,h) slice in K,V tensors: [B, H, N, E]
    const int64_t bh_base = ((int64_t)b * H + h) * N * E;

    // Step 1: Cooperative load K and V into shared memory
    coop_load(smem_K, &K[bh_base], N * E);
    coop_load(smem_V, &V[bh_base], N * E);
    __syncthreads();

    // Warps beyond N are idle (block has N warps, but N may not be power of 2)
    if (n >= N) return;

    // Step 2: Load Q into registers
    const int64_t q_base = bh_base + n * E;
    float q_local[LOCAL];
    int n_local = 0;
    #pragma unroll
        for (int e = lane; e < E; e += WARP_SIZE)
        q_local[n_local++] = ldg(&Q[q_base + e]);

    // Step 3: Validity bitmask
    const int mask_base = h * N * max_K + n * max_K;
    const uint32_t valid_bits = load_valid_bitmask(nb_valid, mask_base, max_K, lane);

    // Step 4: Online softmax over neighbors — read K/V from SMEM
    float run_max = -FLT_MAX, run_sum = 0.0f;
    float out_local[LOCAL] = {0};

    for (int ki = 0; ki < max_K; ki++) {
        if (!(valid_bits & (1u << ki))) continue;
        const int m = ldg(&nb_idx[mask_base + ki]);
        const int smem_off = m * E;  // offset into smem_K / smem_V

        float dot_partial = 0.0f;
        int idx = 0;
        #pragma unroll
        for (int e = lane; e < E; e += WARP_SIZE)
            dot_partial += q_local[idx++] * smem_K[smem_off + e];
        float score = warp_reduce_sum(dot_partial) * scale;

        const float new_max = fmaxf(run_max, score);
        const float exp_old = __expf(run_max - new_max);
        const float exp_new = __expf(score - new_max);
        idx = 0;
        #pragma unroll
        for (int e = lane; e < E; e += WARP_SIZE) {
            out_local[idx] = out_local[idx] * exp_old + exp_new * smem_V[smem_off + e];
            idx++;
        }
        run_sum = run_sum * exp_old + exp_new;
        run_max = new_max;
    }

    // Step 5: Write output
    if (run_sum > 0.0f) {
        const float inv = 1.0f / run_sum;
        int idx = 0;
        #pragma unroll
        for (int e = lane; e < E; e += WARP_SIZE)
            Out[q_base + e] = out_local[idx++] * inv;
    }
    if (lane == 0)
        Lse[((int64_t)b * H + h) * N + n] = run_max + __logf(run_sum + 1e-6f);
}

// ================================================================
//  MDNA — Backward dQ (query-centric, SMEM for K+V)
// ================================================================
//  1 block per (b,h). SMEM holds K[N,E] + V[N,E].
// ================================================================
template <int MAX_E>
__global__ void __launch_bounds__(384, 4) mdna_bwd_dq_kernel(
    const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V, const float* __restrict__ dO,
    const float* __restrict__ Lse, const float* __restrict__ D,
    const int32_t* __restrict__ nb_idx, const int8_t* __restrict__ nb_valid,
    float* __restrict__ dQ,
    const int B, const int H, const int N,
    const int E, const int max_K, const float scale
) {
    constexpr int LOCAL = (MAX_E + WARP_SIZE - 1) / WARP_SIZE;
    extern __shared__ float smem[];
    float* smem_K = smem;
    float* smem_V = smem + N * E;

    const int block_id = blockIdx.x;
    if (block_id >= B * H) return;
    const int h = block_id % H;
    const int b = block_id / H;
    const int n = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int64_t bh_base = ((int64_t)b * H + h) * N * E;

    // Cooperative load K, V
    coop_load(smem_K, &K[bh_base], N * E);
    coop_load(smem_V, &V[bh_base], N * E);
    __syncthreads();

    if (n >= N) return;

    const int64_t q_base = bh_base + n * E;
    const int64_t l_off = ((int64_t)b * H + h) * N + n;
    const float lse_val = ldg(&Lse[l_off]), d_val = ldg(&D[l_off]);

    float q_local[LOCAL], do_local[LOCAL], dq_local[LOCAL];
    int n_local = 0;
    #pragma unroll
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
        const int smem_off = m * E;

        float k_local[LOCAL];
        float dot_partial = 0.0f;
        int idx = 0;
        #pragma unroll
        for (int e = lane; e < E; e += WARP_SIZE) {
            float k_val = smem_K[smem_off + e];
            k_local[idx] = k_val;
            dot_partial += q_local[idx] * k_val;
            idx++;
        }
        float score = warp_reduce_sum(dot_partial) * scale;
        float attn = safe_expf(score - lse_val);

        float dattn_partial = 0.0f;
        idx = 0;
        #pragma unroll
        for (int e = lane; e < E; e += WARP_SIZE)
            dattn_partial += do_local[idx++] * smem_V[smem_off + e];
        float d_attn = warp_reduce_sum(dattn_partial);
        float d_score = attn * (d_attn - d_val);

        idx = 0;
        #pragma unroll
        for (int e = lane; e < E; e += WARP_SIZE) {
            dq_local[idx] += d_score * k_local[idx] * scale;
            idx++;
        }
    }

    int idx = 0;
    #pragma unroll
        for (int e = lane; e < E; e += WARP_SIZE)
        dQ[q_base + e] = dq_local[idx++];
}

// ================================================================
//  MDNA — Backward dK,dV (key-centric, SMEM for Q+dO)
// ================================================================
//  1 block per (b,h). SMEM holds Q[N,E] + dO[N,E].
//  Each warp handles one KEY position m.
// ================================================================
template <int MAX_E>
__global__ void mdna_bwd_dkv_kernel(
    const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V, const float* __restrict__ dO,
    const float* __restrict__ Lse, const float* __restrict__ D,
    const int32_t* __restrict__ rev_nb_idx, const int8_t* __restrict__ rev_nb_valid,
    float* __restrict__ dK, float* __restrict__ dV,
    const int B, const int H, const int N,
    const int E, const int max_rev_K, const float scale
) {
    constexpr int LOCAL = (MAX_E + WARP_SIZE - 1) / WARP_SIZE;
    extern __shared__ float smem[];
    float* smem_Q  = smem;
    float* smem_dO = smem + N * E;

    const int block_id = blockIdx.x;
    if (block_id >= B * H) return;
    const int h = block_id % H;
    const int b = block_id / H;
    const int m = threadIdx.x / WARP_SIZE;  // key position
    const int lane = threadIdx.x % WARP_SIZE;
    const int64_t bh_base = ((int64_t)b * H + h) * N * E;

    // Cooperative load Q, dO
    coop_load(smem_Q,  &Q[bh_base],  N * E);
    coop_load(smem_dO, &dO[bh_base], N * E);
    __syncthreads();

    if (m >= N) return;

    const int64_t kv_base = bh_base + m * E;

    float k_local[LOCAL], v_local[LOCAL];
    float dk_local[LOCAL] = {0}, dv_local[LOCAL] = {0};
    int n_local = 0;
    #pragma unroll
        for (int e = lane; e < E; e += WARP_SIZE) {
        k_local[n_local] = ldg(&K[kv_base + e]);
        v_local[n_local] = ldg(&V[kv_base + e]);
        n_local++;
    }

    const int rev_mask_base = h * N * max_rev_K + m * max_rev_K;
    const uint32_t rev_valid_bits = load_valid_bitmask(rev_nb_valid, rev_mask_base, max_rev_K, lane);

    for (int qi = 0; qi < max_rev_K; qi++) {
        if (!(rev_valid_bits & (1u << qi))) continue;
        const int qn = ldg(&rev_nb_idx[rev_mask_base + qi]);
        const int smem_off = qn * E;
        const int64_t l_off = ((int64_t)b * H + h) * N + qn;
        const float lse_n = ldg(&Lse[l_off]);
        const float d_n = ldg(&D[l_off]);

        float dot_partial = 0.0f, dattn_partial = 0.0f;
        int idx = 0;
        #pragma unroll
        for (int e = lane; e < E; e += WARP_SIZE) {
            float q_val = smem_Q[smem_off + e];
            float do_val = smem_dO[smem_off + e];
            dot_partial += q_val * k_local[idx];
            dattn_partial += do_val * v_local[idx];
            idx++;
        }
        float score = warp_reduce_sum(dot_partial) * scale;
        float attn = safe_expf(score - lse_n);
        float d_attn = warp_reduce_sum(dattn_partial);
        float d_score = attn * (d_attn - d_n);

        idx = 0;
        #pragma unroll
        for (int e = lane; e < E; e += WARP_SIZE) {
            dk_local[idx] += d_score * smem_Q[smem_off + e] * scale;
            dv_local[idx] += attn * smem_dO[smem_off + e];
            idx++;
        }
    }

    int idx = 0;
    #pragma unroll
        for (int e = lane; e < E; e += WARP_SIZE) {
        dK[kv_base + e] = dk_local[idx];
        dV[kv_base + e] = dv_local[idx];
        idx++;
    }
}

// ================================================================
//  TANCA — Cross-Attention Forward (Double-Buffered cp.async)
// ================================================================
//  1 block per (b,vq,h). N warps, one per query position.
//  Double-buffered SMEM: prefetch next vkv's K/V while computing current.
//  Inspired by FlashAttention-3's load-compute overlap.
// ================================================================
template <int MAX_E>
__global__ void __launch_bounds__(384, 3) tanca_fwd_kernel(
    const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V,
    const int32_t* __restrict__ nb_idx, const int8_t* __restrict__ nb_valid,
    float* __restrict__ Out, float* __restrict__ Lse,
    const int B, const int Vq, const int Vkv,
    const int H, const int N, const int E, const int max_K, const float scale
) {
    constexpr int LOCAL = (MAX_E + WARP_SIZE - 1) / WARP_SIZE;
    // Double-buffered SMEM: buf[0] and buf[1], each holds K[N,E] + V[N,E]
    extern __shared__ float smem[];
    const int tile = N * E;
    float* smem_K0 = smem;                    // buf 0 K
    float* smem_V0 = smem + tile;             // buf 0 V
    float* smem_K1 = smem + 2 * tile;         // buf 1 K
    float* smem_V1 = smem + 3 * tile;         // buf 1 V
    float* buf_K[2] = {smem_K0, smem_K1};
    float* buf_V[2] = {smem_V0, smem_V1};

    const int block_id = blockIdx.x;
    if (block_id >= B * Vq * H) return;
    const int h = block_id % H;
    const int vq = (block_id / H) % Vq;
    const int b = block_id / (Vq * H);

    const int n = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;

    // Load Q into registers
    float q_local[LOCAL];
    int n_local = 0;
    int64_t q_base = 0;
    if (n < N) {
        q_base = ((int64_t)(b * Vq + vq) * H + h) * N * E + n * E;
        #pragma unroll
        for (int e = lane; e < E; e += WARP_SIZE)
            q_local[n_local++] = ldg(&Q[q_base + e]);
    }

    uint32_t valid_bits = 0;
    int mask_base = 0;
    if (n < N) {
        mask_base = h * N * max_K + n * max_K;
        valid_bits = load_valid_bitmask(nb_valid, mask_base, max_K, lane);
    }

    float run_max = -FLT_MAX, run_sum = 0.0f;
    float out_local[LOCAL] = {0};

    // Async prefetch first tile into buf[0]
    {
        const int64_t kv_base0 = ((int64_t)(b * Vkv + 0) * H + h) * N * E;
        coop_load_async(buf_K[0], &K[kv_base0], tile);
        coop_load_async(buf_V[0], &V[kv_base0], tile);
        __pipeline_commit();
    }

    for (int vkv = 0; vkv < Vkv; vkv++) {
        const int cur = vkv & 1;
        const bool has_next = (vkv + 1 < Vkv);

        // Prefetch next tile into other buffer
        if (has_next) {
            const int nxt = 1 - cur;
            const int64_t kv_base_nxt = ((int64_t)(b * Vkv + vkv + 1) * H + h) * N * E;
            coop_load_async(buf_K[nxt], &K[kv_base_nxt], tile);
            coop_load_async(buf_V[nxt], &V[kv_base_nxt], tile);
            __pipeline_commit();
        }

        // Wait for current tile: if prefetched next, allow 1 in flight; otherwise drain all
        has_next ? __pipeline_wait_prior(1) : __pipeline_wait_prior(0);
        __syncthreads();

        if (n < N) {
            for (int ki = 0; ki < max_K; ki++) {
                if (!(valid_bits & (1u << ki))) continue;
                const int m_pos = ldg(&nb_idx[mask_base + ki]);
                const int smem_off = m_pos * E;

                float dot_partial = 0.0f;
                int idx = 0;
                #pragma unroll
        for (int e = lane; e < E; e += WARP_SIZE)
                    dot_partial += q_local[idx++] * buf_K[cur][smem_off + e];
                float score = warp_reduce_sum(dot_partial) * scale;

                const float new_max = fmaxf(run_max, score);
                const float exp_old = __expf(run_max - new_max);
                const float exp_new = __expf(score - new_max);
                idx = 0;
                #pragma unroll
        for (int e = lane; e < E; e += WARP_SIZE) {
                    out_local[idx] = out_local[idx] * exp_old + exp_new * buf_V[cur][smem_off + e];
                    idx++;
                }
                run_sum = run_sum * exp_old + exp_new;
                run_max = new_max;
            }
        }
        __syncthreads();
    }

    if (n < N) {
        if (run_sum > 0.0f) {
            const float inv = 1.0f / run_sum;
            int idx = 0;
            #pragma unroll
        for (int e = lane; e < E; e += WARP_SIZE)
                Out[q_base + e] = out_local[idx++] * inv;
        }
        if (lane == 0)
            Lse[((int64_t)(b * Vq + vq) * H + h) * N + n] = run_max + __logf(run_sum + 1e-6f);
    }
}

// ================================================================
//  Shared: compute D_i = dot(dO, O)  [unchanged from v2]
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
    #pragma unroll
        for (int e = lane; e < E; e += WARP_SIZE)
        partial += ldg(&dO[base + e]) * ldg(&O[base + e]);
    float result = warp_reduce_sum(partial);
    if (lane == 0) D[warp_id] = result;
}

// ================================================================
//  TANCA — Backward dQ (query-centric, double-buffered K+V tiles)
// ================================================================
template <int MAX_E>
__global__ void __launch_bounds__(384, 3) tanca_bwd_dq_kernel(
    const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V, const float* __restrict__ dO,
    const float* __restrict__ Lse, const float* __restrict__ D,
    const int32_t* __restrict__ nb_idx, const int8_t* __restrict__ nb_valid,
    float* __restrict__ dQ,
    const int B, const int Vq, const int Vkv,
    const int H, const int N, const int E, const int max_K, const float scale
) {
    constexpr int LOCAL = (MAX_E + WARP_SIZE - 1) / WARP_SIZE;
    extern __shared__ float smem[];
    const int tile = N * E;
    float* buf_K[2] = {smem, smem + 2 * tile};
    float* buf_V[2] = {smem + tile, smem + 3 * tile};

    const int block_id = blockIdx.x;
    if (block_id >= B * Vq * H) return;
    const int h = block_id % H;
    const int vq = (block_id / H) % Vq;
    const int b = block_id / (Vq * H);

    const int n = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;

    float q_local[LOCAL], do_local[LOCAL], dq_local[LOCAL];
    int n_local_count = 0;
    int64_t q_base = 0;
    float lse_val = 0.0f, d_val = 0.0f;
    uint32_t valid_bits = 0;
    int mask_base = 0;

    if (n < N) {
        q_base = ((int64_t)(b * Vq + vq) * H + h) * N * E + n * E;
        const int64_t l_off = ((int64_t)(b * Vq + vq) * H + h) * N + n;
        lse_val = ldg(&Lse[l_off]);
        d_val = ldg(&D[l_off]);
        #pragma unroll
        for (int e = lane; e < E; e += WARP_SIZE) {
            q_local[n_local_count] = ldg(&Q[q_base + e]);
            do_local[n_local_count] = ldg(&dO[q_base + e]);
            dq_local[n_local_count] = 0.0f;
            n_local_count++;
        }
        mask_base = h * N * max_K + n * max_K;
        valid_bits = load_valid_bitmask(nb_valid, mask_base, max_K, lane);
    }

    // Prefetch first tile
    {
        const int64_t kv_base0 = ((int64_t)(b * Vkv + 0) * H + h) * N * E;
        coop_load_async(buf_K[0], &K[kv_base0], tile);
        coop_load_async(buf_V[0], &V[kv_base0], tile);
        __pipeline_commit();
    }

    for (int vkv = 0; vkv < Vkv; vkv++) {
        const int cur = vkv & 1;
        const bool has_next = (vkv + 1 < Vkv);

        if (has_next) {
            const int nxt = 1 - cur;
            const int64_t kv_base_nxt = ((int64_t)(b * Vkv + vkv + 1) * H + h) * N * E;
            coop_load_async(buf_K[nxt], &K[kv_base_nxt], tile);
            coop_load_async(buf_V[nxt], &V[kv_base_nxt], tile);
            __pipeline_commit();
        }

        has_next ? __pipeline_wait_prior(1) : __pipeline_wait_prior(0);
        __syncthreads();

        if (n < N) {
            for (int ki = 0; ki < max_K; ki++) {
                if (!(valid_bits & (1u << ki))) continue;
                const int m_pos = ldg(&nb_idx[mask_base + ki]);
                const int smem_off = m_pos * E;

                float k_local[LOCAL];
                float dot_partial = 0.0f;
                int idx = 0;
                #pragma unroll
        for (int e = lane; e < E; e += WARP_SIZE) {
                    float k_val = buf_K[cur][smem_off + e];
                    k_local[idx] = k_val;
                    dot_partial += q_local[idx] * k_val;
                    idx++;
                }
                float score = warp_reduce_sum(dot_partial) * scale;
                float attn = safe_expf(score - lse_val);

                float dattn_partial = 0.0f;
                idx = 0;
                #pragma unroll
        for (int e = lane; e < E; e += WARP_SIZE)
                    dattn_partial += do_local[idx++] * buf_V[cur][smem_off + e];
                float d_attn = warp_reduce_sum(dattn_partial);
                float d_score = attn * (d_attn - d_val);

                idx = 0;
                #pragma unroll
        for (int e = lane; e < E; e += WARP_SIZE) {
                    dq_local[idx] += d_score * k_local[idx] * scale;
                    idx++;
                }
            }
        }
        __syncthreads();
    }

    if (n < N) {
        int idx = 0;
        #pragma unroll
        for (int e = lane; e < E; e += WARP_SIZE)
            dQ[q_base + e] = dq_local[idx++];
    }
}

// ================================================================
//  TANCA — Backward dK,dV TILED (Vq split into grid, atomicAdd)
// ================================================================
//  Grid: B * Vkv * H * n_tiles.  Each block handles vq_tile_size
//  Vq iterations, then atomicAdd partial dK/dV into output.
// ================================================================
template <int MAX_E>
__global__ void __launch_bounds__(384, 2) tanca_bwd_dkv_tiled_kernel(
    const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V, const float* __restrict__ dO,
    const float* __restrict__ Lse, const float* __restrict__ D,
    const int32_t* __restrict__ rev_nb_idx, const int8_t* __restrict__ rev_nb_valid,
    float* __restrict__ dK, float* __restrict__ dV,
    const int B, const int Vq, const int Vkv,
    const int H, const int N, const int E, const int max_rev_K, const float scale,
    const int vq_tile_size
) {
    constexpr int LOCAL = (MAX_E + WARP_SIZE - 1) / WARP_SIZE;
    extern __shared__ float smem[];
    const int tile = N * E;
    float* buf_Q[2]  = {smem, smem + 2 * tile};
    float* buf_dO[2] = {smem + tile, smem + 3 * tile};

    // Grid decomposition: (tile_idx, b, vkv, h)
    const int n_tiles = (Vq + vq_tile_size - 1) / vq_tile_size;
    const int bvh_block = blockIdx.x / n_tiles;
    const int tile_idx  = blockIdx.x % n_tiles;

    if (bvh_block >= B * Vkv * H) return;
    const int h   = bvh_block % H;
    const int vkv = (bvh_block / H) % Vkv;
    const int b   = bvh_block / (Vkv * H);

    const int vq_start = tile_idx * vq_tile_size;
    const int vq_end   = vq_start + vq_tile_size < Vq ? vq_start + vq_tile_size : Vq;

    const int m    = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;

    float k_local[LOCAL], v_local[LOCAL];
    float dk_local[LOCAL] = {0}, dv_local[LOCAL] = {0};
    int n_local_count = 0;
    int64_t kv_base = 0;
    uint32_t rev_valid_bits = 0;
    int rev_mask_base = 0;

    if (m < N) {
        kv_base = ((int64_t)(b * Vkv + vkv) * H + h) * N * E + m * E;
        #pragma unroll
        for (int e = lane; e < E; e += WARP_SIZE) {
            k_local[n_local_count] = ldg(&K[kv_base + e]);
            v_local[n_local_count] = ldg(&V[kv_base + e]);
            n_local_count++;
        }
        rev_mask_base = h * N * max_rev_K + m * max_rev_K;
        rev_valid_bits = load_valid_bitmask(rev_nb_valid, rev_mask_base, max_rev_K, lane);
    }

    // Prefetch first Q/dO tile for vq_start
    {
        const int64_t q_base0 = ((int64_t)(b * Vq + vq_start) * H + h) * N * E;
        coop_load_async(buf_Q[0],  &Q[q_base0],  tile);
        coop_load_async(buf_dO[0], &dO[q_base0], tile);
        __pipeline_commit();
    }

    for (int vq = vq_start; vq < vq_end; vq++) {
        const int cur = (vq - vq_start) & 1;
        const bool has_next = (vq + 1 < vq_end);

        // Prefetch next tile
        if (has_next) {
            const int nxt = 1 - cur;
            const int64_t q_base_nxt = ((int64_t)(b * Vq + vq + 1) * H + h) * N * E;
            coop_load_async(buf_Q[nxt],  &Q[q_base_nxt],  tile);
            coop_load_async(buf_dO[nxt], &dO[q_base_nxt], tile);
            __pipeline_commit();
        }

        has_next ? __pipeline_wait_prior(1) : __pipeline_wait_prior(0);
        __syncthreads();

        if (m < N) {
            for (int qi = 0; qi < max_rev_K; qi++) {
                if (!(rev_valid_bits & (1u << qi))) continue;
                const int qn = ldg(&rev_nb_idx[rev_mask_base + qi]);
                const int smem_off = qn * E;
                const int64_t l_off = ((int64_t)(b * Vq + vq) * H + h) * N + qn;
                const float lse_n = ldg(&Lse[l_off]);
                const float d_n = ldg(&D[l_off]);

                float dot_partial = 0.0f, dattn_partial = 0.0f;
                int idx = 0;
                #pragma unroll 1
                for (int e = lane; e < E; e += WARP_SIZE) {
                    float q_val = buf_Q[cur][smem_off + e];
                    float do_val = buf_dO[cur][smem_off + e];
                    dot_partial += q_val * k_local[idx];
                    dattn_partial += do_val * v_local[idx];
                    idx++;
                }
                float score = warp_reduce_sum(dot_partial) * scale;
                float attn = safe_expf(score - lse_n);
                float d_attn = warp_reduce_sum(dattn_partial);
                float d_score = attn * (d_attn - d_n);

                idx = 0;
                #pragma unroll 1
                for (int e = lane; e < E; e += WARP_SIZE) {
                    dk_local[idx] += d_score * buf_Q[cur][smem_off + e] * scale;
                    dv_local[idx] += attn * buf_dO[cur][smem_off + e];
                    idx++;
                }
            }
        }
        __syncthreads();
    }

    // Atomic writeback — multiple tiles accumulate into same dK/dV
    if (m < N) {
        int idx = 0;
        #pragma unroll
        for (int e = lane; e < E; e += WARP_SIZE) {
            atomicAdd(&dK[kv_base + e], dk_local[idx]);
            atomicAdd(&dV[kv_base + e], dv_local[idx]);
            idx++;
        }
    }
}

// ================================================================
//  C-linkage launch functions
// ================================================================

// Helper: set max dynamic SMEM if needed (>48KB requires opt-in)
#define SET_SMEM_IF_NEEDED(kernel, smem_bytes) \
    do { \
        if ((smem_bytes) > 48 * 1024) { \
            cudaFuncSetAttribute( \
                (kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(smem_bytes)); \
        } \
    } while (0)

extern "C" {

void mdna_forward_launch(
    float* Q, float* K, float* V,
    int32_t* nb_idx, int8_t* nb_valid,
    float* Out, float* Lse,
    int B, int H, int N, int E, int max_K, float scale
) {
    const int grid = B * H;
    const int block = N * WARP_SIZE;
    const size_t smem = 2 * N * E * sizeof(float);
    #define LAUNCH_FWD(ME) do { \
        SET_SMEM_IF_NEEDED(mdna_fwd_kernel<ME>, smem); \
        mdna_fwd_kernel<ME><<<grid, block, smem>>>( \
            Q, K, V, nb_idx, nb_valid, Out, Lse, B, H, N, E, max_K, scale); \
    } while(0)
    DISPATCH_E(E, LAUNCH_FWD);
    #undef LAUNCH_FWD
}

void mdna_backward_launch(
    float* Q, float* K, float* V, float* dO,
    float* Lse, float* D_buf,
    int32_t* nb_idx, int8_t* nb_valid,
    int32_t* rev_nb_idx, int8_t* rev_nb_valid,
    float* dQ, float* dK, float* dV, float* Out,
    int B, int H, int N, int E, int max_K, int max_rev_K, float scale
) {
    // compute_D uses old warp-level parallelism (no SMEM needed)
    const int total_q = B * H * N;
    const int d_warps_per_block = 8;
    const int d_thr = d_warps_per_block * WARP_SIZE;
    const int d_blk = (total_q + d_warps_per_block - 1) / d_warps_per_block;
    compute_D_kernel<<<d_blk, d_thr>>>(dO, Out, D_buf, total_q, E);

    // dQ and dKV use SMEM tiling
    const int grid_bh = B * H;
    const int block_n = N * WARP_SIZE;
    const size_t smem = 2 * N * E * sizeof(float);

    #define LAUNCH_DQ(ME) do { \
        SET_SMEM_IF_NEEDED(mdna_bwd_dq_kernel<ME>, smem); \
        mdna_bwd_dq_kernel<ME><<<grid_bh, block_n, smem>>>( \
            Q, K, V, dO, Lse, D_buf, nb_idx, nb_valid, dQ, B, H, N, E, max_K, scale); \
    } while(0)
    DISPATCH_E(E, LAUNCH_DQ);
    #undef LAUNCH_DQ

    #define LAUNCH_DKV(ME) do { \
        SET_SMEM_IF_NEEDED(mdna_bwd_dkv_kernel<ME>, smem); \
        mdna_bwd_dkv_kernel<ME><<<grid_bh, block_n, smem>>>( \
            Q, K, V, dO, Lse, D_buf, rev_nb_idx, rev_nb_valid, dK, dV, B, H, N, E, max_rev_K, scale); \
    } while(0)
    DISPATCH_E(E, LAUNCH_DKV);
    #undef LAUNCH_DKV
}

void tanca_forward_launch(
    float* Q, float* K, float* V,
    int32_t* nb_idx, int8_t* nb_valid,
    float* Out, float* Lse,
    int B, int Vq, int Vkv, int H, int N, int E, int max_K, float scale
) {
    const int grid = B * Vq * H;
    const int block = N * WARP_SIZE;
    const size_t smem = 4 * N * E * sizeof(float);  // double-buffered K+V
    #define LAUNCH_TFWD(ME) do { \
        SET_SMEM_IF_NEEDED(tanca_fwd_kernel<ME>, smem); \
        tanca_fwd_kernel<ME><<<grid, block, smem>>>( \
            Q, K, V, nb_idx, nb_valid, Out, Lse, B, Vq, Vkv, H, N, E, max_K, scale); \
    } while(0)
    DISPATCH_E(E, LAUNCH_TFWD);
    #undef LAUNCH_TFWD
}

void tanca_backward_launch(
    float* Q, float* K, float* V, float* dO,
    float* Lse, float* D_buf,
    int32_t* nb_idx, int8_t* nb_valid,
    int32_t* rev_nb_idx, int8_t* rev_nb_valid,
    float* dQ, float* dK, float* dV, float* Out,
    int B, int Vq, int Vkv, int H, int N, int E, int max_K, int max_rev_K, float scale
) {
    // compute_D
    const int total_q = B * Vq * H * N;
    const int d_warps_per_block = 8;
    const int d_thr = d_warps_per_block * WARP_SIZE;
    const int d_blk = (total_q + d_warps_per_block - 1) / d_warps_per_block;
    compute_D_kernel<<<d_blk, d_thr>>>(dO, Out, D_buf, total_q, E);

    const int block_n = N * WARP_SIZE;
    const size_t smem_dq = 4 * N * E * sizeof(float);  // double-buffered K+V

    // dQ: 1 block per (b,vq,h)
    const int grid_dq = B * Vq * H;
    #define LAUNCH_TDQ(ME) do { \
        SET_SMEM_IF_NEEDED(tanca_bwd_dq_kernel<ME>, smem_dq); \
        tanca_bwd_dq_kernel<ME><<<grid_dq, block_n, smem_dq>>>( \
            Q, K, V, dO, Lse, D_buf, nb_idx, nb_valid, dQ, B, Vq, Vkv, H, N, E, max_K, scale); \
    } while(0)
    DISPATCH_E(E, LAUNCH_TDQ);
    #undef LAUNCH_TDQ

    // dKV: tiled over Vq dimension with atomicAdd accumulation
    const int vq_tile_size = 32;
    const int n_tiles = (Vq + vq_tile_size - 1) / vq_tile_size;
    const int grid_dkv = B * Vkv * H * n_tiles;
    #define LAUNCH_TDKV(ME) do { \
        SET_SMEM_IF_NEEDED(tanca_bwd_dkv_tiled_kernel<ME>, smem_dq); \
        tanca_bwd_dkv_tiled_kernel<ME><<<grid_dkv, block_n, smem_dq>>>( \
            Q, K, V, dO, Lse, D_buf, rev_nb_idx, rev_nb_valid, dK, dV, B, Vq, Vkv, H, N, E, max_rev_K, scale, vq_tile_size); \
    } while(0)
    DISPATCH_E(E, LAUNCH_TDKV);
    #undef LAUNCH_TDKV
}

}  // extern "C"
