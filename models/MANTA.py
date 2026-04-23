"""
MANTA: Multi-scale Aligned Neighborhood Temporal Attention.

Uses mask+matmul instead of gather for memory efficiency.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def build_dilated_na_indices(kernel_size, dilation, L, device):
    """Build neighbor indices with dilation for 1D neighborhood attention.

    For dilation=1: standard neighbors [-K//2, ..., 0, ..., K//2]
    For dilation=d: sparse neighbors [-K//2*d, ..., 0, ..., K//2*d]

    This gives effective receptive field = (K-1)*d + 1 while computing
    attention over only K positions.
    """
    K = kernel_size
    half_K = K // 2
    pos = torch.arange(L, device=device)
    window_offset = (torch.arange(K, device=device) - half_K) * dilation
    ideal = pos.unsqueeze(1) + window_offset  # [L, K]
    nb_idx = ideal.clamp(0, L - 1)
    valid = (ideal >= 0) & (ideal < L)
    return nb_idx, valid



def auto_head_configs(n_heads, patch_num):
    """Generate multi-scale (kernel_size, dilation) configs for each head.

    Covers local-to-global temporal scales by varying both K and dilation.
    Inspired by DiNAT (Hassani 2022) which assigns different dilations
    to different heads within the same layer.

    Returns a list of (K, dilation) tuples, one per head, sorted from
    smallest to largest effective receptive field.
    """
    # Collect all valid (K, dilation) → effective_span
    candidates = set()
    for d in range(1, patch_num + 1):
        for K in [3, 5, 7]:
            eff = (K - 1) * d + 1
            if eff <= patch_num and K <= patch_num and K % 2 == 1:
                candidates.add((K, d, eff))

    # Add global attention (largest odd K, dilation=1)
    gk = patch_num if patch_num % 2 == 1 else patch_num - 1
    gk = max(1, gk)
    candidates.add((gk, 1, gk))

    # Sort: by effective span (ascending), then prefer dense (larger K)
    candidates = sorted(candidates, key=lambda x: (x[2], -x[0], x[1]))

    if not candidates:
        return [(1, 1)] * n_heads

    # Evenly sample n_heads configs across the scale range
    configs = []
    for h in range(n_heads):
        idx = round(h * (len(candidates) - 1) / max(1, n_heads - 1))
        idx = min(idx, len(candidates) - 1)
        configs.append((candidates[idx][0], candidates[idx][1]))

    return configs


class MultiScaleNA1D(nn.Module):
    """
    Multi-scale neighborhood self-attention.

    Memory-efficient version: uses [H, N, N] mask + standard matmul
    instead of gather. No [B, H, N, max_K, E] intermediate tensor.
    """

    def __init__(self, n_heads, patch_num, attention_dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.dropout_p = attention_dropout
        self.patch_num = patch_num

        # Auto-assign (K, dilation) per head
        self.head_configs = auto_head_configs(n_heads, patch_num)

        # Build [H, N, N] boolean mask from NA indices
        na_mask = torch.zeros(n_heads, patch_num, patch_num, dtype=torch.bool)
        for h, (K, d) in enumerate(self.head_configs):
            nb_idx, valid = build_dilated_na_indices(K, d, patch_num, torch.device('cpu'))
            for n in range(patch_num):
                for k in range(K):
                    if valid[n, k]:
                        na_mask[h, n, nb_idx[n, k]] = True

        self.register_buffer('_na_mask', na_mask)  # [H, N, N]

    def extra_repr(self):
        lines = [f"n_heads={self.n_heads}, patch_num={self.patch_num}"]
        cfg_map = {}
        for h, (K, d) in enumerate(self.head_configs):
            cfg_map.setdefault((K, d), []).append(h)
        for (K, d), heads in cfg_map.items():
            eff = (K - 1) * d + 1
            lines.append(f"  heads {heads}: K={K}, d={d}, eff={eff}")
        return "\n".join(lines)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        scale = E ** -0.5
        q = queries.permute(0, 2, 1, 3)  # [B, H, L, E]
        k = keys.permute(0, 2, 1, 3)
        v = values.permute(0, 2, 1, 3)

        # Standard matmul + mask (no gather)
        attn = torch.einsum("bhne,bhme->bhnm", q, k) * scale  # [B, H, N, N]
        attn = attn.masked_fill(~self._na_mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        # TODO: attention weights dropout is not used in modern Transformers
        # (LLaMA, PatchTST, etc.). Consider removing — residual + FFN dropout suffice.
        if self.training and self.dropout_p > 0:
            attn = F.dropout(attn, p=self.dropout_p)
        out = torch.einsum("bhnm,bhme->bhne", attn, v)  # [B, H, N, E]

        return out.permute(0, 2, 1, 3).contiguous(), None


class MultiScaleTANCA(nn.Module):
    """
    Multi-scale Temporally-Aligned Neighborhood Cross-Attention.

    Uses N×N mask + standard matmul instead of gather.
    Supports two modes:
      - forward(): concatenate all exo vars, one-shot matmul (fast, default)
      - forward_online(): loop over exo vars with online softmax (low memory)
    """

    def __init__(self, n_heads, patch_num, d_model, attention_dropout=0.1,
                 use_online=False):
        super().__init__()
        self.n_heads = n_heads
        self.dropout_p = attention_dropout
        self.use_online = use_online

        # Variable importance gate
        self.var_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

        # Auto-assign (K, dilation) per head
        self.head_configs = auto_head_configs(n_heads, patch_num)

        # Build [H, N, N] boolean mask from NA indices
        # mask[h, n, m] = True if position m is a valid neighbor of n for head h
        na_mask = torch.zeros(n_heads, patch_num, patch_num, dtype=torch.bool)
        for h, (K, d) in enumerate(self.head_configs):
            nb_idx, valid = build_dilated_na_indices(K, d, patch_num, torch.device('cpu'))
            for n in range(patch_num):
                for k in range(K):
                    if valid[n, k]:
                        na_mask[h, n, nb_idx[n, k]] = True

        self.register_buffer('_na_mask', na_mask)  # [H, N, N]

    def extra_repr(self):
        lines = [f"n_heads={self.n_heads}"]
        cfg_map = {}
        for h, (K, d) in enumerate(self.head_configs):
            cfg_map.setdefault((K, d), []).append(h)
        for (K, d), heads in cfg_map.items():
            eff = (K - 1) * d + 1
            lines.append(f"  heads {heads}: K={K}, d={d}, eff={eff}")
        return "\n".join(lines)

    def forward(self, queries, keys, values):
        if self.use_online:
            return self._forward_online(queries, keys, values)
        return self._forward_concat(queries, keys, values)

    def _forward_concat(self, queries, keys, values):
        """Concatenate all exo vars, one-shot matmul (fast, default)."""
        B, N, H, E = queries.shape
        C = keys.shape[1]
        scale = E ** -0.5

        # Variable importance gate
        v_pool = values.mean(dim=2)                # [B, C, H, E]
        v_pool = v_pool.reshape(B, C, H * E)      # [B, C, d_model]
        gate = self.var_gate(v_pool)               # [B, C, 1]

        q = queries.permute(0, 2, 1, 3)            # [B, H, N, E]

        # Rearrange: [B, C, N, H, E] → [B, C, H, N, E]
        keys_r = keys.permute(0, 1, 3, 2, 4)      # [B, C, H, N, E]
        vals_r = values.permute(0, 1, 3, 2, 4)    # [B, C, H, N, E]
        vals_r = vals_r * gate.view(B, C, 1, 1, 1)

        # Concatenate all C exo variables: [B, C, H, N, E] → [B, H, C*N, E]
        k_cat = keys_r.permute(0, 2, 1, 3, 4).reshape(B, H, C * N, E)
        v_cat = vals_r.permute(0, 2, 1, 3, 4).reshape(B, H, C * N, E)

        # Tile mask: [H, N, N] → [H, N, C*N]
        tanca_mask = self._na_mask.unsqueeze(2).expand(-1, -1, C, -1).reshape(H, N, C * N)

        # One-shot matmul
        attn = torch.einsum("bhne,bhme->bhnm", q, k_cat) * scale  # [B, H, N, C*N]
        attn = attn.masked_fill(~tanca_mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        # TODO: consider removing attention weights dropout (see MDNA comment)
        if self.training and self.dropout_p > 0:
            attn = F.dropout(attn, p=self.dropout_p)
        out = torch.einsum("bhnm,bhme->bhne", attn, v_cat)        # [B, H, N, E]

        return out.permute(0, 2, 1, 3).contiguous(), None

    def _forward_online(self, queries, keys, values):
        """Online softmax — lower memory, loops over exo vars."""
        B, N, H, E = queries.shape
        C = keys.shape[1]
        scale = E ** -0.5

        v_pool = values.mean(dim=2)
        v_pool = v_pool.reshape(B, C, H * E)
        gate = self.var_gate(v_pool)

        q = queries.permute(0, 2, 1, 3)
        keys_r = keys.permute(0, 1, 3, 2, 4)
        vals_r = values.permute(0, 1, 3, 2, 4)
        vals_r = vals_r * gate.view(B, C, 1, 1, 1)

        running_max = torch.full(
            (B, H, N, 1), float("-inf"), device=q.device, dtype=q.dtype
        )
        running_sum = torch.zeros(B, H, N, 1, device=q.device, dtype=q.dtype)
        running_out = torch.zeros(B, H, N, E, device=q.device, dtype=q.dtype)

        for c in range(C):
            k_c = keys_r[:, c]
            v_c = vals_r[:, c]
            scores_c = torch.einsum("bhne,bhme->bhnm", q, k_c) * scale
            scores_c = scores_c.masked_fill(~self._na_mask, float("-inf"))

            chunk_max = scores_c.max(dim=-1, keepdim=True).values
            new_max = torch.maximum(running_max, chunk_max)
            old_scale = torch.exp(running_max - new_max)
            running_out = running_out * old_scale
            running_sum = running_sum * old_scale

            exp_scores = torch.exp(scores_c - new_max)
            running_sum = running_sum + exp_scores.sum(dim=-1, keepdim=True)
            running_out = running_out + torch.einsum(
                "bhnm,bhme->bhne", exp_scores, v_c
            )
            running_max = new_max

        out = running_out / running_sum
        # TODO: consider removing attention dropout (see MDNA comment)
        if self.training and self.dropout_p > 0:
            out = F.dropout(out, p=self.dropout_p)

        return out.permute(0, 2, 1, 3).contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super().__init__()
        self.inner_attention = attention
        self.n_heads = n_heads
        d_k = d_model // n_heads
        self.query_projection = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.key_projection = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.value_projection = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.out_projection = nn.Linear(d_k * n_heads, d_model, bias=False)

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out, w = self.inner_attention(queries, keys, values)
        return self.out_projection(out.view(B, L, -1)), w


class CrossAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super().__init__()
        self.inner_attention = attention
        self.n_heads = n_heads
        d_k = d_model // n_heads
        self.query_projection = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.key_projection = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.value_projection = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.out_projection = nn.Linear(d_k * n_heads, d_model, bias=False)

    def forward(self, queries, keys, values):
        B, N, _ = queries.shape
        C = keys.shape[1]
        H = self.n_heads
        queries = self.query_projection(queries).view(B, N, H, -1)
        keys = self.key_projection(keys.reshape(B * C, N, -1))
        keys = keys.view(B, C, N, H, -1)
        values = self.value_projection(values.reshape(B * C, N, -1))
        values = values.view(B, C, N, H, -1)
        out, w = self.inner_attention(queries, keys, values)
        return self.out_projection(out.view(B, N, -1)), w


class SelfEncoderLayer(nn.Module):
    """self-attn → FFN, with pre-norm residual."""

    def __init__(self, attention, d_model, d_ff=None,
                 dropout=0.1, activation="gelu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
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
        x_norm = self.norm1(x)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
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
    """cross-attn → FFN, with pre-norm residual."""

    def __init__(self, attention, d_model, d_ff=None,
                 dropout=0.1, activation="gelu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
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
        x_norm = self.norm1(x)
        cross_out, _ = self.attention(x_norm, cross, cross)
        x = x + self.dropout(cross_out)
        if self.use_swiglu:
            x = x + self.ffn(self.norm2(x))
        else:
            y = self.dropout(self.linear2(
                self.dropout(self.activation(self.linear1(self.norm2(x))))
            ))
            x = x + y
        return x


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


class Model(nn.Module):
    """MANTA: Multi-scale Aligned Neighborhood Temporal Attention."""

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.features = configs.features
        self.pred_len = configs.pred_len
        self.patch_len = configs.patch_len
        self.patch_num = configs.seq_len // configs.patch_len
        self.output_attention = getattr(configs, "output_attention", False)
        self.k_sample = getattr(configs, "k_sample", 8)

        # RevIN
        self.use_revin = getattr(configs, "use_revin", True)
        if self.use_revin:
            self.revin_layer = RevIN(
                configs.enc_in,
                affine=getattr(configs, "use_revin_affine", True),
            )

        activation = getattr(configs, "activation", "gelu")

        # Print head config for logging
        head_cfgs = auto_head_configs(configs.n_heads, self.patch_num)
        config_summary = {}
        for K, d in head_cfgs:
            eff = (K - 1) * d + 1
            key = f"K={K},d={d},eff={eff}"
            config_summary[key] = config_summary.get(key, 0) + 1
        print(f"  [MANTA] patch_num={self.patch_num}, head configs: {config_summary}")

        # Shared embedding
        self.shared_embedding = EnEmbedding(
            configs.d_model, self.patch_len, configs.dropout,
            n_vars=configs.enc_in, use_time_features=True, freq=configs.freq,
        )

        # Shared encoder: self-attn + FFN for all variables
        self.shared_encoder = Encoder(
            [
                SelfEncoderLayer(
                    attention=AttentionLayer(
                        MultiScaleNA1D(
                            n_heads=configs.n_heads,
                            patch_num=self.patch_num,
                            attention_dropout=configs.dropout,
                        ),
                        configs.d_model,
                        configs.n_heads,
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

        # Cross encoder: cross-attn + FFN for endo with processed exo
        self.cross_encoder = CrossEncoder(
            [
                CrossEncoderLayer(
                    attention=CrossAttentionLayer(
                        MultiScaleTANCA(
                            n_heads=configs.n_heads,
                            patch_num=self.patch_num,
                            d_model=configs.d_model,
                            attention_dropout=configs.dropout,
                        ),
                        configs.d_model,
                        configs.n_heads,
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

        # Prediction head
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
        """MS mode: predict last variable using others as exogenous."""
        if self.use_revin:
            x_enc = self.revin_layer(x_enc, "norm")
        batch_size = x_enc.shape[0]
        n_vars = x_enc.shape[2]

        all_embed, _, patch_num = self.shared_embedding(
            x_enc.permute(0, 2, 1), x_mark_enc
        )
        # Stage 1: shared self-attention on all variables
        all_embed = self.shared_encoder(all_embed)  # [B*n_vars, patch_num, D]
        all_embed = all_embed.reshape(batch_size, n_vars, patch_num, -1)

        endo = all_embed[:, -1, :, :]     # [B, patch_num, D]
        exo = all_embed[:, :-1, :, :]     # [B, C, patch_num, D]

        # Stage 2: cross-attention
        endo_out = self.cross_encoder(endo, exo)

        endo_out = endo_out.unsqueeze(1).permute(0, 1, 3, 2)
        dec_out = self.head(endo_out).permute(0, 2, 1)

        if self.use_revin:
            dec_out = self.revin_layer(dec_out, "denorm")
        return dec_out

    def forecast_multi(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """M mode: all variables as both endogenous and exogenous.
        Random variable sampling: each query var attends to a random subset
        of k variables as KV, preserving temporal alignment.
        Training: random k vars (dropout-like regularization).
        Inference: all variables as KV (exact)."""
        if self.use_revin:
            x_enc = self.revin_layer(x_enc, "norm")
        batch_size = x_enc.shape[0]
        n_vars = x_enc.shape[2]

        all_embed, _, patch_num = self.shared_embedding(
            x_enc.permute(0, 2, 1), x_mark_enc
        )
        # Stage 1: shared self-attention on all variables
        all_embed = self.shared_encoder(all_embed)  # [B*n_vars, patch_num, D]
        all_embed = all_embed.reshape(batch_size, n_vars, patch_num, -1)

        # Stage 2: cross-attention with variable sampling
        k_sample = getattr(self, 'k_sample', 32)

        if self.training and n_vars > k_sample:
            # Variable-dimension KV Dropout (VarKVDrop):
            # Each query variable attends to a random subset of k KV variables.
            # Q remains full (all vars need predictions). Analogous to DropKey
            # (Li et al. 2023) but on the variable dimension. Provides
            # regularization + O(n_vars*k) efficiency. Exact at inference.
            idx = torch.randperm(n_vars, device=all_embed.device)[:k_sample]
            idx = idx.sort().values
            sampled_kv = all_embed[:, idx]  # [B, k, patch_num, D]
            C = k_sample

            endo = all_embed.reshape(batch_size * n_vars, patch_num, -1)
            exo = sampled_kv.unsqueeze(1).expand(
                batch_size, n_vars, C, patch_num, -1
            ).reshape(batch_size * n_vars, C, patch_num, -1)
            endo_out = self.cross_encoder(endo, exo)
            endo_out = endo_out.reshape(batch_size, n_vars, patch_num, -1)
        else:
            # Inference: all vars as KV, chunked to avoid OOM (exact, no approx)
            chunk_size = getattr(self, 'var_chunk_size', 32)
            outputs = []
            for start in range(0, n_vars, chunk_size):
                end = min(start + chunk_size, n_vars)
                c = end - start
                endo_chunk = all_embed[:, start:end].reshape(
                    batch_size * c, patch_num, -1
                )
                exo_chunk = all_embed.unsqueeze(1).expand(
                    batch_size, c, n_vars, patch_num, -1
                ).reshape(batch_size * c, n_vars, patch_num, -1)
                out_chunk = self.cross_encoder(endo_chunk, exo_chunk)
                outputs.append(out_chunk.reshape(batch_size, c, patch_num, -1))
            endo_out = torch.cat(outputs, dim=1)
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
