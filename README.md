# MANTA: Multi-scale Aligned Neighborhood Temporal Attention for Time Series Forecasting

Official PyTorch implementation of **MANTA**, submitted to *IEEE Transactions on Knowledge and Data Engineering (TKDE)*.

> **MANTA: Multi-scale Aligned Neighborhood Temporal Attention for Time Series Forecasting**
> Zhiyuan Zhao, Ali Anaissi
> School of Computer Science, University of Technology Sydney

---

## Overview

MANTA addresses three limitations of existing Transformer-based time series forecasters:

1. **Multi-scale Dilated Neighborhood Attention (MDNA)** — assigns each attention head a distinct `(kernel size, dilation)` pair, so that a single attention layer simultaneously captures local-to-global temporal scales.
2. **Temporally-Aligned Neighborhood Cross-Attention (TANCA)** — fuses exogenous variables into the endogenous representation through temporally-aligned neighborhood constraints and a learned variable-importance gate.
3. **Variable-dimension KV Dropout (VarKVDrop)** — regularizes cross-attention in high-dimensional multivariate settings by randomly subsampling KV variables at training time; full variable set is used at inference.
4. **Custom CUDA kernels** — realize the theoretical `O(N·k)` sparse complexity for both MDNA and TANCA via shared-memory tiling and online softmax, without materializing dense score matrices.

## Model Files

| File | Description |
| --- | --- |
| [models/MANTA.py](models/MANTA.py) | Reference PyTorch implementation (mask-based fallback, portable). |
| [models/MANTA_CUDA.py](models/MANTA_CUDA.py) | CUDA-accelerated implementation using custom kernels. |
| [models/MANTA_Ablation.py](models/MANTA_Ablation.py) | Ablation variants used in Table V. |
| [models/manta_ops.py](models/manta_ops.py) | Python bindings to the CUDA kernels. |
| [models/csrc/manta_kernels.cu](models/csrc/manta_kernels.cu) | CUDA kernels (MDNA + TANCA forward/backward). |
| [models/csrc/manta_kernels_binding.cpp](models/csrc/manta_kernels_binding.cpp) | PyTorch C++ binding. |

## Requirements

- Python ≥ 3.13
- CUDA ≥ 12.x (Ampere SM80+ for custom kernels; the mask-based `MANTA.py` runs on any backend)
- PyTorch ≥ 2.10 with CUDA support

Install dependencies via the provided [pyproject.toml](pyproject.toml):

```bash
pip install -e .
# With CUDA kernel build dependencies (ninja):
pip install -e ".[cuda]"
```

## Data Preparation

All benchmark datasets are included under [datasets/](datasets/):

- **Long-term forecasting**: ECL, ETTh1, ETTh2, ETTm1, ETTm2, Weather, Traffic
- **Short-term (EPF)**: NP, PJM, BE, FR, DE

A few files exceed GitHub's 100 MB limit and must be obtained separately (see `.gitignore` — entries under `# Files exceeding GitHub 100MB limit`), e.g. `datasets/traffic/traffic.csv`. Standard sources follow the TimeXer / PatchTST protocols.

## Reproducing the Paper

### EPF short-term ablation (Table IV, Table V)

Run the full ablation sweep on all five EPF markets:

```bash
python train.py --run_epf_ablation --results_file ablation_results.csv
```

Select specific datasets / variants:

```bash
python train.py --run_epf_ablation \
    --datasets NP,PJM,BE,FR,DE \
    --variants full,single_scale_na,full_self_attn,full_cross_attn,no_var_gate,no_cross_encoder,separate_encoders
```

### Traffic VarKVDrop ablation (Table VI)

```bash
python train_traffic_varkvdrop_ablation.py
```

### Long-term forecasting (Tables II–III)

Example — ECL under the M setting with the CUDA implementation:

```bash
python train.py \
    --model MANTA_CUDA --features M \
    --data custom --root_path ./datasets/electricity/ --data_path electricity.csv \
    --enc_in 321 --dec_in 321 --c_out 321 \
    --seq_len 96 --pred_len 96 --patch_len 16 \
    --d_model 576 --d_ff 256 --n_heads 16 --e_layers 3
```

Swap `--features M` ↔ `--features MS` for multivariate-to-single forecasting, and adjust `--pred_len ∈ {96, 192, 336, 720}` for each horizon.

## Citation

If you find this work useful, please cite:

```bibtex
@article{zhao2026manta,
  title   = {MANTA: Multi-scale Aligned Neighborhood Temporal Attention for Time Series Forecasting},
  author  = {Zhao, Zhiyuan and Anaissi, Ali},
  journal = {IEEE Transactions on Knowledge and Data Engineering},
  year    = {2026},
  note    = {Under review}
}
```

## Acknowledgements

This codebase builds on the excellent [Time-Series-Library](https://github.com/thuml/Time-Series-Library) and draws on [TimeXer](https://github.com/thuml/TimeXer), [PatchTST](https://github.com/yuqinie98/PatchTST), and [iTransformer](https://github.com/thuml/iTransformer) for baseline comparisons. The neighborhood attention formulation is inspired by [NAT](https://github.com/SHI-Labs/NATTEN) / [DiNAT](https://github.com/SHI-Labs/Neighborhood-Attention-Transformer).

## License

Released under the [MIT License](LICENSE).
