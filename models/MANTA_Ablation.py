"""
MANTA ablation model.

Implements the paper's ablation variants behind a single `ablation_variant`
switch while keeping the same public `Model(configs)` interface as MANTA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.MANTA import (
    AttentionLayer,
    CrossAttentionLayer,
    CrossEncoder,
    CrossEncoderLayer,
    EnEmbedding,
    Encoder,
    FlattenHead,
    MultiScaleNA1D,
    RevIN,
    SelfEncoderLayer,
    auto_head_configs,
    build_dilated_na_indices,
)


VALID_ABLATIONS = {
    "full",
    "single_scale_na",
    "full_self_attn",
    "full_cross_attn",
    "no_var_gate",
    "no_cross_encoder",
    "separate_encoders",
}


class FullSelfAttention1D(nn.Module):
    def __init__(self, attention_dropout=0.1):
        super().__init__()
        self.dropout_p = attention_dropout

    def forward(self, queries, keys, values):
        _, _, _, head_dim = queries.shape
        scale = head_dim ** -0.5
        q = queries.permute(0, 2, 1, 3)
        k = keys.permute(0, 2, 1, 3)
        v = values.permute(0, 2, 1, 3)

        attn = torch.einsum("bhne,bhme->bhnm", q, k) * scale
        attn = F.softmax(attn, dim=-1)
        if self.training and self.dropout_p > 0:
            attn = F.dropout(attn, p=self.dropout_p)
        out = torch.einsum("bhnm,bhme->bhne", attn, v)
        return out.permute(0, 2, 1, 3).contiguous(), None


class FixedScaleNA1D(nn.Module):
    def __init__(self, n_heads, patch_num, kernel_size=7, dilation=1, attention_dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.patch_num = patch_num
        self.dropout_p = attention_dropout
        effective_kernel = min(
            kernel_size,
            patch_num if patch_num % 2 == 1 else max(1, patch_num - 1),
        )
        self.head_configs = [(effective_kernel, dilation)] * n_heads

        na_mask = torch.zeros(n_heads, patch_num, patch_num, dtype=torch.bool)
        nb_idx, valid = build_dilated_na_indices(
            effective_kernel,
            dilation,
            patch_num,
            torch.device("cpu"),
        )
        for h in range(n_heads):
            for n in range(patch_num):
                for k in range(nb_idx.shape[1]):
                    if valid[n, k]:
                        na_mask[h, n, nb_idx[n, k]] = True

        self.register_buffer("_na_mask", na_mask)

    def forward(self, queries, keys, values):
        _, _, _, head_dim = queries.shape
        scale = head_dim ** -0.5
        q = queries.permute(0, 2, 1, 3)
        k = keys.permute(0, 2, 1, 3)
        v = values.permute(0, 2, 1, 3)

        attn = torch.einsum("bhne,bhme->bhnm", q, k) * scale
        attn = attn.masked_fill(~self._na_mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        if self.training and self.dropout_p > 0:
            attn = F.dropout(attn, p=self.dropout_p)
        out = torch.einsum("bhnm,bhme->bhne", attn, v)
        return out.permute(0, 2, 1, 3).contiguous(), None


class AblationCrossAttention(nn.Module):
    def __init__(
        self,
        n_heads,
        patch_num,
        d_model,
        attention_dropout=0.1,
        use_local_mask=True,
        use_var_gate=True,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.patch_num = patch_num
        self.dropout_p = attention_dropout
        self.use_local_mask = use_local_mask
        self.use_var_gate = use_var_gate
        self.head_configs = auto_head_configs(n_heads, patch_num)

        if self.use_var_gate:
            self.var_gate = nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, 1),
                nn.Sigmoid(),
            )

        na_mask = torch.zeros(n_heads, patch_num, patch_num, dtype=torch.bool)
        for h, (kernel_size, dilation) in enumerate(self.head_configs):
            nb_idx, valid = build_dilated_na_indices(
                kernel_size, dilation, patch_num, torch.device("cpu")
            )
            for n in range(patch_num):
                for k in range(kernel_size):
                    if valid[n, k]:
                        na_mask[h, n, nb_idx[n, k]] = True
        self.register_buffer("_na_mask", na_mask)

    def forward(self, queries, keys, values):
        batch_size, patch_num, n_heads, head_dim = queries.shape
        n_exo = keys.shape[1]
        scale = head_dim ** -0.5

        q = queries.permute(0, 2, 1, 3)
        keys_r = keys.permute(0, 1, 3, 2, 4)
        vals_r = values.permute(0, 1, 3, 2, 4)

        if self.use_var_gate:
            v_pool = values.mean(dim=2).reshape(batch_size, n_exo, n_heads * head_dim)
            gate = self.var_gate(v_pool)
            vals_r = vals_r * gate.view(batch_size, n_exo, 1, 1, 1)

        k_cat = keys_r.permute(0, 2, 1, 3, 4).reshape(
            batch_size, n_heads, n_exo * patch_num, head_dim
        )
        v_cat = vals_r.permute(0, 2, 1, 3, 4).reshape(
            batch_size, n_heads, n_exo * patch_num, head_dim
        )

        attn = torch.einsum("bhne,bhme->bhnm", q, k_cat) * scale
        if self.use_local_mask:
            tanca_mask = (
                self._na_mask.unsqueeze(2)
                .expand(-1, -1, n_exo, -1)
                .reshape(n_heads, patch_num, n_exo * patch_num)
            )
            attn = attn.masked_fill(~tanca_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        if self.training and self.dropout_p > 0:
            attn = F.dropout(attn, p=self.dropout_p)
        out = torch.einsum("bhnm,bhme->bhne", attn, v_cat)
        return out.permute(0, 2, 1, 3).contiguous(), None


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.features = configs.features
        self.pred_len = configs.pred_len
        self.patch_len = configs.patch_len
        self.patch_num = configs.seq_len // configs.patch_len
        self.output_attention = getattr(configs, "output_attention", False)
        self.ablation_variant = getattr(configs, "ablation_variant", "full")

        if self.ablation_variant not in VALID_ABLATIONS:
            raise ValueError(
                f"Unknown ablation_variant: {self.ablation_variant}. "
                f"Available: {sorted(VALID_ABLATIONS)}"
            )

        self.use_revin = getattr(configs, "use_revin", True)
        if self.use_revin:
            self.revin_layer = RevIN(
                configs.enc_in,
                affine=getattr(configs, "use_revin_affine", True),
            )

        activation = getattr(configs, "activation", "gelu")

        head_cfgs = auto_head_configs(configs.n_heads, self.patch_num)
        config_summary = {}
        for kernel_size, dilation in head_cfgs:
            eff = (kernel_size - 1) * dilation + 1
            key = f"K={kernel_size},d={dilation},eff={eff}"
            config_summary[key] = config_summary.get(key, 0) + 1
        print(
            f"  [MANTA_Ablation:{self.ablation_variant}] "
            f"patch_num={self.patch_num}, full head configs: {config_summary}"
        )

        self.shared_embedding = EnEmbedding(
            configs.d_model,
            self.patch_len,
            configs.dropout,
            n_vars=configs.enc_in,
            use_time_features=True,
            freq=configs.freq,
        )

        self.shared_encoder = self._build_shared_encoder(configs, activation)

        self.use_cross_encoder = self.ablation_variant != "no_cross_encoder"
        self.use_separate_encoders = self.ablation_variant == "separate_encoders"
        if self.use_separate_encoders:
            self.endo_encoder = self._build_multiscale_encoder(configs, activation)
            self.exo_encoder = self._build_multiscale_encoder(configs, activation)

        if self.use_cross_encoder:
            self.cross_encoder = self._build_cross_encoder(configs, activation)

        self.head_nf = configs.d_model * self.patch_num
        self.head = FlattenHead(self.head_nf, configs.pred_len, head_dropout=configs.dropout)

        self._init_weights()

    def _build_multiscale_encoder(self, configs, activation):
        return Encoder(
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

    def _build_shared_encoder(self, configs, activation):
        if self.ablation_variant == "full_self_attn":
            attention_factory = lambda: AttentionLayer(
                FullSelfAttention1D(attention_dropout=configs.dropout),
                configs.d_model,
                configs.n_heads,
            )
        elif self.ablation_variant == "single_scale_na":
            attention_factory = lambda: AttentionLayer(
                FixedScaleNA1D(
                    n_heads=configs.n_heads,
                    patch_num=self.patch_num,
                    kernel_size=7,
                    dilation=1,
                    attention_dropout=configs.dropout,
                ),
                configs.d_model,
                configs.n_heads,
            )
        else:
            attention_factory = lambda: AttentionLayer(
                MultiScaleNA1D(
                    n_heads=configs.n_heads,
                    patch_num=self.patch_num,
                    attention_dropout=configs.dropout,
                ),
                configs.d_model,
                configs.n_heads,
            )

        return Encoder(
            [
                SelfEncoderLayer(
                    attention=attention_factory(),
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout,
                    activation=activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.RMSNorm(configs.d_model),
        )

    def _build_cross_encoder(self, configs, activation):
        if self.ablation_variant == "full_cross_attn":
            cross_attention = lambda: CrossAttentionLayer(
                AblationCrossAttention(
                    n_heads=configs.n_heads,
                    patch_num=self.patch_num,
                    d_model=configs.d_model,
                    attention_dropout=configs.dropout,
                    use_local_mask=False,
                    use_var_gate=False,
                ),
                configs.d_model,
                configs.n_heads,
            )
        elif self.ablation_variant == "no_var_gate":
            cross_attention = lambda: CrossAttentionLayer(
                AblationCrossAttention(
                    n_heads=configs.n_heads,
                    patch_num=self.patch_num,
                    d_model=configs.d_model,
                    attention_dropout=configs.dropout,
                    use_local_mask=True,
                    use_var_gate=False,
                ),
                configs.d_model,
                configs.n_heads,
            )
        else:
            cross_attention = lambda: CrossAttentionLayer(
                AblationCrossAttention(
                    n_heads=configs.n_heads,
                    patch_num=self.patch_num,
                    d_model=configs.d_model,
                    attention_dropout=configs.dropout,
                    use_local_mask=True,
                    use_var_gate=True,
                ),
                configs.d_model,
                configs.n_heads,
            )

        return CrossEncoder(
            [
                CrossEncoderLayer(
                    attention=cross_attention(),
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout,
                    activation=activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.RMSNorm(configs.d_model),
        )

    def _init_weights(self):
        for _, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.RMSNorm):
                nn.init.ones_(module.weight)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def _apply_separate_encoders_ms(self, all_embed, batch_size, n_vars, patch_num):
        all_embed = all_embed.reshape(batch_size, n_vars, patch_num, -1)
        endo = all_embed[:, -1, :, :]
        exo = all_embed[:, :-1, :, :]

        endo = self.endo_encoder(endo)
        if exo.shape[1] > 0:
            exo = exo.reshape(batch_size * (n_vars - 1), patch_num, -1)
            exo = self.exo_encoder(exo)
            exo = exo.reshape(batch_size, n_vars - 1, patch_num, -1)
        return endo, exo

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_revin:
            x_enc = self.revin_layer(x_enc, "norm")
        batch_size = x_enc.shape[0]
        n_vars = x_enc.shape[2]

        all_embed, _, patch_num = self.shared_embedding(x_enc.permute(0, 2, 1), x_mark_enc)

        if self.use_separate_encoders:
            endo, exo = self._apply_separate_encoders_ms(all_embed, batch_size, n_vars, patch_num)
        else:
            all_embed = self.shared_encoder(all_embed)
            all_embed = all_embed.reshape(batch_size, n_vars, patch_num, -1)
            endo = all_embed[:, -1, :, :]
            exo = all_embed[:, :-1, :, :]

        if self.use_cross_encoder:
            endo = self.cross_encoder(endo, exo)

        endo = endo.unsqueeze(1).permute(0, 1, 3, 2)
        dec_out = self.head(endo).permute(0, 2, 1)

        if self.use_revin:
            dec_out = self.revin_layer(dec_out, "denorm")
        return dec_out

    def forecast_multi(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_revin:
            x_enc = self.revin_layer(x_enc, "norm")
        batch_size = x_enc.shape[0]
        n_vars = x_enc.shape[2]

        all_embed, _, patch_num = self.shared_embedding(x_enc.permute(0, 2, 1), x_mark_enc)
        all_embed = self.shared_encoder(all_embed)
        all_embed = all_embed.reshape(batch_size, n_vars, patch_num, -1)

        if not self.use_cross_encoder:
            endo_out = all_embed
        else:
            endo = all_embed.reshape(batch_size * n_vars, patch_num, -1)
            exo = all_embed.unsqueeze(1).expand(
                batch_size, n_vars, n_vars, patch_num, -1
            ).reshape(batch_size * n_vars, n_vars, patch_num, -1)
            endo_out = self.cross_encoder(endo, exo)
            endo_out = endo_out.reshape(batch_size, n_vars, patch_num, -1)

        endo_out = endo_out.permute(0, 1, 3, 2)
        dec_out = self.head(endo_out).permute(0, 2, 1)

        if self.use_revin:
            dec_out = self.revin_layer(dec_out, "denorm")
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            if self.features == "M":
                dec_out = self.forecast_multi(x_enc, x_mark_enc, x_dec, x_mark_dec)
            else:
                dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len :, :]
        return None
