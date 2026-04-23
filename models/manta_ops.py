"""
MANTA CUDA ops: Python autograd wrappers for MDNA + TANCA kernels.

MDNA:  Q,K,V [B, H, N, E] -> O [B, H, N, E]           (self-attention)
TANCA: Q [B,Vq,H,N,E]  K,V [B,Vkv,H,N,E] -> O [B,Vq,H,N,E]  (cross-attention)

Both use online softmax — no full attention matrix, O(1) per-position memory.
TANCA avoids the O(n_vars²) KV expansion entirely.
Backward uses atomic-free key-centric kernels for dK/dV.
"""

import os
import sys

import torch
from torch.utils.cpp_extension import load

# Windows: ensure CUDA and PyTorch DLLs are findable at load time
if sys.platform == "win32":
    _dll_dirs = [
        os.path.join(os.path.dirname(torch.__file__), "lib"),
    ]
    # Add CUDA toolkit bin directory
    _cuda_path = os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2")
    _cuda_bin = os.path.join(_cuda_path, "bin")
    if os.path.isdir(_cuda_bin):
        _dll_dirs.append(_cuda_bin)
    for _d in _dll_dirs:
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(_d)
        if _d not in os.environ.get("PATH", ""):
            os.environ["PATH"] = _d + ";" + os.environ.get("PATH", "")

# JIT compile CUDA extension (cached after first build)
_dir = os.path.dirname(os.path.abspath(__file__))
_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = load(
            name="manta_cuda",
            sources=[
                os.path.join(_dir, "csrc", "manta_kernels_binding.cpp"),
                os.path.join(_dir, "csrc", "manta_kernels.cu"),
            ],
            extra_cuda_cflags=["-Xcompiler", "/Zc:preprocessor", "--use_fast_math"],
            verbose=True,
        )
    return _ext


# ================================================================
#  MDNA — Multi-scale Dilated Neighborhood Attention (self-attention)
# ================================================================
class MDNAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, nb_idx, nb_valid, rev_nb_idx, rev_nb_valid, scale):
        E = Q.shape[-1]
        N = Q.shape[2]
        max_K = nb_idx.shape[2]
        assert E <= 288, f"MDNA: E must be <= 288 (got {E}). Reduce d_model or increase n_heads."
        assert N <= 32, f"MDNA: N must be <= 32 (got {N}). Block size = N*32 = {N*32} exceeds CUDA 1024 limit. Increase patch_len."
        assert max_K <= 32, f"MDNA: max_K must be <= 32 (got {max_K}). Validity bitmask is uint32."
        smem_bytes = 2 * N * E * 4
        assert smem_bytes <= 227 * 1024, (
            f"MDNA SMEM overflow: need {smem_bytes}B for N={N}, E={E} (max ~99KB). "
            f"Reduce seq_len/patch_len ratio or increase n_heads."
        )
        ext = _get_ext()
        Out, Lse = ext.mdna_forward(Q, K, V, nb_idx, nb_valid, scale)
        ctx.save_for_backward(Q, K, V, Out, Lse, nb_idx, nb_valid, rev_nb_idx, rev_nb_valid)
        ctx.scale = scale
        return Out

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, Out, Lse, nb_idx, nb_valid, rev_nb_idx, rev_nb_valid = ctx.saved_tensors
        ext = _get_ext()
        dQ, dK, dV = ext.mdna_backward(
            Q, K, V, Out, Lse, dO.contiguous(),
            nb_idx, nb_valid, rev_nb_idx, rev_nb_valid, ctx.scale
        )
        return dQ, dK, dV, None, None, None, None, None


def mdna_attention(Q, K, V, nb_idx, nb_valid, rev_nb_idx, rev_nb_valid, scale):
    """Sparse 1D multi-scale dilated neighborhood self-attention via CUDA."""
    return MDNAFunction.apply(Q, K, V, nb_idx, nb_valid, rev_nb_idx, rev_nb_valid, scale)


# ================================================================
#  TANCA — Temporally-Aligned Neighborhood Cross-Attention
# ================================================================
class TANCAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, nb_idx, nb_valid, rev_nb_idx, rev_nb_valid, scale):
        E = Q.shape[-1]
        N = Q.shape[3]
        max_K = nb_idx.shape[2]
        assert E <= 288, f"TANCA: E must be <= 288 (got {E}). Reduce d_model or increase n_heads."
        assert N <= 32, f"TANCA: N must be <= 32 (got {N}). Block size = N*32 = {N*32} exceeds CUDA 1024 limit. Increase patch_len."
        assert max_K <= 32, f"TANCA: max_K must be <= 32 (got {max_K}). Validity bitmask is uint32."
        smem_bytes = 4 * N * E * 4  # double-buffered K+V
        assert smem_bytes <= 227 * 1024, (
            f"TANCA SMEM overflow: need {smem_bytes}B for N={N}, E={E} (max ~99KB). "
            f"Reduce seq_len/patch_len ratio or increase n_heads."
        )
        ext = _get_ext()
        Out, Lse = ext.tanca_forward(Q, K, V, nb_idx, nb_valid, scale)
        ctx.save_for_backward(Q, K, V, Out, Lse, nb_idx, nb_valid, rev_nb_idx, rev_nb_valid)
        ctx.scale = scale
        return Out

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, Out, Lse, nb_idx, nb_valid, rev_nb_idx, rev_nb_valid = ctx.saved_tensors
        ext = _get_ext()
        dQ, dK, dV = ext.tanca_backward(
            Q, K, V, Out, Lse, dO.contiguous(),
            nb_idx, nb_valid, rev_nb_idx, rev_nb_valid, ctx.scale
        )
        return dQ, dK, dV, None, None, None, None, None


def tanca_attention(Q, K, V, nb_idx, nb_valid, rev_nb_idx, rev_nb_valid, scale):
    """Sparse cross-attention with O(n_vars) memory via CUDA."""
    return TANCAFunction.apply(Q, K, V, nb_idx, nb_valid, rev_nb_idx, rev_nb_valid, scale)
