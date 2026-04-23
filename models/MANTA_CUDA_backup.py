"""
MANTA-CUDA: Multi-scale Aligned Neighborhood Temporal Attention.

Uses custom CUDA kernels for both MDNA (self-attn) and TANCA (cross-attn).
- MDNA:  Multi-scale Dilated Neighborhood Attention — sparse, no [B,H,N,N] mask
- TANCA: Temporally-Aligned Neighborhood Cross-Attention — O(n_vars) memory, no KV expand

CUDA version eliminates the need for variable sampling or chunking.
Training and inference both use ALL variables with full precision.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.manta_ops import mdna_attention, tanca_attention


# ================================================================
#  Reusable components
# ================================================================

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            return self._normalize(x)
        return self._denormalize(x)

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class FlattenHead(nn.Module):
    def __init__(self, nf, target_window, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        return self.dropout(self.linear(self.flatten(x)))


class EnEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, dropout, n_vars,
                 use_time_features=False, freq='h'):
        super().__init__()
        self.patch_len = patch_len
        self.use_time_features = use_time_features
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_embedding = PositionalEmbedding(d_model)
        self.variable_embedding = nn.Embedding(n_vars, d_model)
        if use_time_features:
            freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1,
                        'a': 1, 'w': 2, 'd': 3, 'b': 3}
            self.time_embedding = nn.Linear(
                freq_map.get(freq, 4), d_model, bias=False
            )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark=None):
        B, n_vars, L = x.shape
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = x.reshape(B * n_vars, x.shape[2], x.shape[3])
        patch_num = x.shape[1]
        x_embed = self.value_embedding(x) + self.position_embedding(x)
        var_ids = torch.arange(n_vars, device=x.device)
        var_embed = self.variable_embedding(var_ids).unsqueeze(0).expand(B, -1, -1)
        var_embed = var_embed.reshape(B * n_vars, 1, -1)
        x_embed = x_embed + var_embed
        if self.use_time_features and x_mark is not None:
            x_mark_patched = x_mark.unfold(
                dimension=1, size=self.patch_len, step=self.patch_len
            )
            x_mark_patched = x_mark_patched[:, :patch_num, :, self.patch_len // 2]
            x_mark_patched = x_mark_patched.unsqueeze(1).expand(-1, n_vars, -1, -1)
            x_mark_patched = x_mark_patched.reshape(B * n_vars, patch_num, -1)
            x_embed = x_embed + self.time_embedding(x_mark_patched)
        return self.dropout(x_embed), n_vars, patch_num


# ================================================================
#  Helpers
# ================================================================

def build_dilated_na_indices(kernel_size, dilation, L, device):
    K = kernel_size
    half_K = K // 2
    pos = torch.arange(L, device=device)
    window_offset = (torch.arange(K, device=device) - half_K) * dilation
    ideal = pos.unsqueeze(1) + window_offset
    nb_idx = ideal.clamp(0, L - 1)
    valid = (ideal >= 0) & (ideal < L)
    return nb_idx, valid


def auto_head_configs(n_heads, patch_num):
    candidates = set()
    for d in range(1, patch_num + 1):
        for K in [3, 5, 7]:
            eff = (K - 1) * d + 1
            if eff <= patch_num and K <= patch_num and K % 2 == 1:
                candidates.add((K, d, eff))
    gk = patch_num if patch_num % 2 == 1 else patch_num - 1
    gk = max(1, gk)
    candidates.add((gk, 1, gk))
    candidates = sorted(candidates, key=lambda x: (x[2], -x[0], x[1]))
    if not candidates:
        return [(1, 1)] * n_heads
    configs = []
    for h in range(n_heads):
        idx = round(h * (len(candidates) - 1) / max(1, n_heads - 1))
        idx = min(idx, len(candidates) - 1)
        configs.append((candidates[idx][0], candidates[idx][1]))
    return configs


def build_na_buffers(n_heads, patch_num):
    """Build nb_idx [H, N, max_K] and nb_valid [H, N, max_K] for CUDA kernels."""
    head_cfgs = auto_head_configs(n_heads, patch_num)
    max_K = max(K for K, _ in head_cfgs)
    nb_idx_all = torch.zeros(n_heads, patch_num, max_K, dtype=torch.int32)
    nb_valid_all = torch.zeros(n_heads, patch_num, max_K, dtype=torch.int8)
    for h, (K, d) in enumerate(head_cfgs):
        idx, valid = build_dilated_na_indices(K, d, patch_num, torch.device('cpu'))
        nb_idx_all[h, :, :K] = idx.int()
        nb_valid_all[h, :, :K] = valid.to(torch.int8)
    return nb_idx_all, nb_valid_all, head_cfgs


# ================================================================
#  MDNA — CUDA-backed self-attention
# ================================================================

class MDNA_CUDA(nn.Module):
    """Multi-scale Dilated Neighborhood Attention via CUDA kernel."""

    def __init__(self, n_heads, patch_num, d_model):
        super().__init__()
        self.n_heads = n_heads
        self.patch_num = patch_num
        self.d_k = d_model // n_heads

        nb_idx, nb_valid, self.head_configs = build_na_buffers(n_heads, patch_num)
        self.register_buffer('_nb_idx', nb_idx)
        self.register_buffer('_nb_valid', nb_valid)

        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.key_projection = nn.Linear(d_model, d_model, bias=False)
        self.value_projection = nn.Linear(d_model, d_model, bias=False)
        self.out_projection = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, N, D = x.shape
        H, E = self.n_heads, self.d_k
        scale = E ** -0.5

        q = self.query_projection(x).view(B, N, H, E).permute(0, 2, 1, 3).contiguous()
        k = self.key_projection(x).view(B, N, H, E).permute(0, 2, 1, 3).contiguous()
        v = self.value_projection(x).view(B, N, H, E).permute(0, 2, 1, 3).contiguous()

        out = mdna_attention(q, k, v, self._nb_idx, self._nb_valid, scale)

        out = out.permute(0, 2, 1, 3).reshape(B, N, D)
        return self.out_projection(out)


# ================================================================
#  TANCA — CUDA-backed cross-attention
# ================================================================

class TANCA_CUDA(nn.Module):
    """Temporally-Aligned Neighborhood Cross-Attention via CUDA kernel.
    O(n_vars) memory — no KV expansion."""

    def __init__(self, n_heads, patch_num, d_model):
        super().__init__()
        self.n_heads = n_heads
        self.patch_num = patch_num
        self.d_k = d_model // n_heads
        self.d_model = d_model

        nb_idx, nb_valid, self.head_configs = build_na_buffers(n_heads, patch_num)
        self.register_buffer('_nb_idx', nb_idx)
        self.register_buffer('_nb_valid', nb_valid)

        self.var_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.key_projection = nn.Linear(d_model, d_model, bias=False)
        self.value_projection = nn.Linear(d_model, d_model, bias=False)
        self.out_projection = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x_q, x_kv):
        B, Vq, N, D = x_q.shape
        Vkv = x_kv.shape[1]
        H, E = self.n_heads, self.d_k
        scale = E ** -0.5

        q = self.query_projection(x_q.reshape(B * Vq, N, D))
        q = q.view(B, Vq, N, H, E).permute(0, 1, 3, 2, 4).contiguous()

        k = self.key_projection(x_kv.reshape(B * Vkv, N, D))
        k = k.view(B, Vkv, N, H, E).permute(0, 1, 3, 2, 4).contiguous()

        v = self.value_projection(x_kv.reshape(B * Vkv, N, D))
        v = v.view(B, Vkv, N, H, E).permute(0, 1, 3, 2, 4).contiguous()

        # var_gate on projected V (matches MANTA.py)
        v_pool = v.mean(dim=3)                     # [B, Vkv, H, E]
        v_pool = v_pool.reshape(B, Vkv, H * E)     # [B, Vkv, D]
        gate = self.var_gate(v_pool)                # [B, Vkv, 1]
        v = v * gate.view(B, Vkv, 1, 1, 1)

        out = tanca_attention(q, k, v, self._nb_idx, self._nb_valid, scale)

        out = out.permute(0, 1, 3, 2, 4).reshape(B * Vq, N, D)
        return self.out_projection(out).view(B, Vq, N, D)


# ================================================================
#  Encoder layers
# ================================================================

class SelfEncoderLayer(nn.Module):
    def __init__(self, mdna, d_model, d_ff=None, dropout=0.1, activation="gelu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = mdna
        self.use_swiglu = activation == "swiglu"
        if self.use_swiglu:
            self.ffn = SwiGLU(d_model, d_ff, dropout=dropout)
        else:
            self.linear1 = nn.Linear(d_model, d_ff)
            self.linear2 = nn.Linear(d_ff, d_model)
            self.activation = F.relu if activation == "relu" else F.gelu
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)
        if self.use_swiglu:
            x = x + self.ffn(self.norm2(x))
        else:
            y = self.dropout(self.linear2(
                self.dropout(self.activation(self.linear1(self.norm2(x))))
            ))
            x = x + y
        return x


class CrossEncoderLayer(nn.Module):
    def __init__(self, tanca, d_model, d_ff=None, dropout=0.1, activation="gelu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = tanca
        self.use_swiglu = activation == "swiglu"
        if self.use_swiglu:
            self.ffn = SwiGLU(d_model, d_ff, dropout=dropout)
        else:
            self.linear1 = nn.Linear(d_model, d_ff)
            self.linear2 = nn.Linear(d_ff, d_model)
            self.activation = F.relu if activation == "relu" else F.gelu
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cross):
        attn_out = self.attention(self.norm1(x), cross)
        x = x + self.dropout(attn_out)
        if self.use_swiglu:
            x = x + self.ffn(self.norm2(x))
        else:
            y = self.dropout(self.linear2(
                self.dropout(self.activation(self.linear1(self.norm2(x))))
            ))
            x = x + y
        return x


# ================================================================
#  Encoder stacks
# ================================================================

class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


class CrossEncoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross):
        for layer in self.layers:
            x = layer(x, cross)
        if self.norm is not None:
            x = self.norm(x)
        return x


# ================================================================
#  Model
# ================================================================

class Model(nn.Module):
    """MANTA-CUDA: Multi-scale Aligned Neighborhood Temporal Attention.

    Uses CUDA kernels for MDNA and TANCA.
    Training: random variable sampling (k_sample) for speed + regularization.
    Inference: all variables via CUDA kernel (exact, O(n_vars) memory, no chunking).
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.features = configs.features
        self.pred_len = configs.pred_len
        self.patch_len = configs.patch_len
        self.patch_num = configs.seq_len // configs.patch_len
        self.output_attention = getattr(configs, "output_attention", False)
        self.k_sample = getattr(configs, "k_sample", 8)

        self.use_revin = getattr(configs, "use_revin", True)
        if self.use_revin:
            self.revin_layer = RevIN(
                configs.enc_in,
                affine=getattr(configs, "use_revin_affine", True),
            )

        activation = getattr(configs, "activation", "gelu")

        head_cfgs = auto_head_configs(configs.n_heads, self.patch_num)
        config_summary = {}
        for K, d in head_cfgs:
            eff = (K - 1) * d + 1
            key = f"K={K},d={d},eff={eff}"
            config_summary[key] = config_summary.get(key, 0) + 1
        print(f"  [MANTA-CUDA] patch_num={self.patch_num}, head configs: {config_summary}")

        self.shared_embedding = EnEmbedding(
            configs.d_model, self.patch_len, configs.dropout,
            n_vars=configs.enc_in, use_time_features=True, freq=configs.freq,
        )

        self.shared_encoder = Encoder(
            [
                SelfEncoderLayer(
                    mdna=MDNA_CUDA(
                        n_heads=configs.n_heads,
                        patch_num=self.patch_num,
                        d_model=configs.d_model,
                    ),
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout,
                    activation=activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.RMSNorm(configs.d_model),
        )

        self.cross_encoder = CrossEncoder(
            [
                CrossEncoderLayer(
                    tanca=TANCA_CUDA(
                        n_heads=configs.n_heads,
                        patch_num=self.patch_num,
                        d_model=configs.d_model,
                    ),
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout,
                    activation=activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.RMSNorm(configs.d_model),
        )

        self.head_nf = configs.d_model * self.patch_num
        self.head = FlattenHead(
            self.head_nf, configs.pred_len, head_dropout=configs.dropout,
        )

        self._init_weights()

    def _init_weights(self):
        for _, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.RMSNorm):
                nn.init.ones_(m.weight)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_revin:
            x_enc = self.revin_layer(x_enc, "norm")
        batch_size = x_enc.shape[0]
        n_vars = x_enc.shape[2]

        all_embed, _, patch_num = self.shared_embedding(
            x_enc.permute(0, 2, 1), x_mark_enc
        )
        all_embed = self.shared_encoder(all_embed)
        all_embed = all_embed.reshape(batch_size, n_vars, patch_num, -1)

        endo = all_embed[:, -1:, :, :]
        exo = all_embed[:, :-1, :, :]
        endo_out = self.cross_encoder(endo, exo)

        endo_out = endo_out.permute(0, 1, 3, 2)
        dec_out = self.head(endo_out).permute(0, 2, 1)

        if self.use_revin:
            dec_out = self.revin_layer(dec_out, "denorm")
        return dec_out

    def forecast_multi(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """M mode: training uses random variable sampling, inference uses all vars.
        CUDA kernel handles both — no expand, no chunking needed."""
        if self.use_revin:
            x_enc = self.revin_layer(x_enc, "norm")
        batch_size = x_enc.shape[0]
        n_vars = x_enc.shape[2]

        all_embed, _, patch_num = self.shared_embedding(
            x_enc.permute(0, 2, 1), x_mark_enc
        )
        all_embed = self.shared_encoder(all_embed)
        all_embed = all_embed.reshape(batch_size, n_vars, patch_num, -1)

        if self.training and n_vars > self.k_sample:
            # Variable-dimension KV Dropout (VarKVDrop):
            # Each query variable attends to a random subset of k KV variables
            # instead of all n_vars. Q remains full (all vars need predictions).
            # Analogous to DropKey (Li et al. 2023) but operates on the variable
            # dimension rather than the token dimension. Serves dual purpose:
            #   1. Regularization: prevents over-reliance on specific variable
            #      combinations, similar to dropout on hidden units.
            #   2. Efficiency: reduces cross-attention cost from O(n_vars²) to
            #      O(n_vars * k) while preserving temporal alignment.
            # At inference, all variables are used (exact, no approximation).
            idx = torch.randperm(n_vars, device=all_embed.device)[:self.k_sample]
            idx = idx.sort().values
            kv = all_embed[:, idx]  # [B, k, N, D]
        else:
            # Inference: all variables as KV (exact)
            kv = all_embed

        endo_out = self.cross_encoder(all_embed, kv)

        endo_out = endo_out.permute(0, 1, 3, 2)
        dec_out = self.head(endo_out).permute(0, 2, 1)

        if self.use_revin:
            dec_out = self.revin_layer(dec_out, "denorm")
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            if self.features == "M":
                dec_out = self.forecast_multi(
                    x_enc, x_mark_enc, x_dec, x_mark_dec
                )
            else:
                dec_out = self.forecast(
                    x_enc, x_mark_enc, x_dec, x_mark_dec
                )
            return dec_out[:, -self.pred_len:, :]
        return None
