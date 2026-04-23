"""
train_ablation.py -- EPF ablation runner using best hyperparameters from results.txt.
"""

import ast
import copy
from pathlib import Path

import numpy as np

from train import (
    EPF_DATASETS,
    append_result,
    apply_epf_defaults,
    build_parser,
    finalize_args,
    fix_random_seed,
    run_training,
)


BEST_CONFIG_KEYS = [
    "features",
    "seq_len",
    "label_len",
    "pred_len",
    "enc_in",
    "dec_in",
    "c_out",
    "d_model",
    "d_ff",
    "n_heads",
    "e_layers",
    "patch_len",
    "batch_size",
    "patience",
    "train_epochs",
    "learning_rate",
    "dropout",
    "lradj",
    "activation",
]


def parse_result_value(raw_value):
    raw_value = raw_value.strip()
    if not raw_value:
        return raw_value
    if raw_value[0] == raw_value[-1] == '"':
        return raw_value[1:-1]
    try:
        return ast.literal_eval(raw_value)
    except (SyntaxError, ValueError):
        return raw_value


def load_best_epf_configs(results_file, dataset_names):
    path = Path(results_file)
    if not path.exists():
        raise FileNotFoundError(f"Best-config file not found: {results_file}")

    blocks = []
    current = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            if current:
                blocks.append(current)
                current = {}
            continue
        if "=" not in stripped:
            continue
        key, value = [item.strip() for item in stripped.split("=", 1)]
        current[key] = parse_result_value(value)
    if current:
        blocks.append(current)

    best_configs = {}
    for block in blocks:
        dataset_name = block.get("dataset")
        if dataset_name in dataset_names:
            best_configs[dataset_name] = block

    missing = [name for name in dataset_names if name not in best_configs]
    if missing:
        raise ValueError(
            "Missing EPF best configs in "
            f"{results_file} for: {', '.join(missing)}"
        )
    return best_configs


def apply_best_epf_config(args, best_config):
    for key in BEST_CONFIG_KEYS:
        if key in best_config:
            setattr(args, key, best_config[key])
    return args


def run_epf_ablation(args):
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    best_configs = load_best_epf_configs(args.best_config_file, datasets)

    all_rows = []
    for dataset_name in datasets:
        if dataset_name not in EPF_DATASETS:
            raise ValueError(f"Unknown EPF dataset: {dataset_name}")
        for variant_name in variants:
            run_args = copy.deepcopy(args)
            run_args = apply_epf_defaults(run_args, dataset_name, variant_name)
            run_args = apply_best_epf_config(run_args, best_configs[dataset_name])
            run_args.model = "MANTA_Ablation"
            run_args.ablation_variant = variant_name
            run_args.model_id = f"{dataset_name}_{variant_name}"
            run_args = finalize_args(run_args)
            fix_random_seed(run_args.seed)

            print("=" * 72)
            print(f"dataset={dataset_name} variant={variant_name} model={run_args.model}")
            print(
                "best_cfg "
                f"d_model={run_args.d_model} d_ff={run_args.d_ff} "
                f"n_heads={run_args.n_heads} e_layers={run_args.e_layers} "
                f"patch_len={run_args.patch_len} batch_size={run_args.batch_size} "
                f"lr={run_args.learning_rate} dropout={run_args.dropout}"
            )
            print("=" * 72)

            test_mse, test_mae, n_params = run_training(run_args)
            row = {
                "dataset": dataset_name,
                "model": run_args.model,
                "variant": variant_name,
                "mse": f"{test_mse:.6f}",
                "mae": f"{test_mae:.6f}",
                "params": n_params,
                "seq_len": run_args.seq_len,
                "pred_len": run_args.pred_len,
                "patch_len": run_args.patch_len,
            }
            append_result(args.results_file, row)
            all_rows.append(row)

    print("\nAverage across selected datasets:")
    for variant_name in variants:
        variant_rows = [row for row in all_rows if row["variant"] == variant_name]
        if not variant_rows:
            continue
        avg_mse = np.mean([float(row["mse"]) for row in variant_rows])
        avg_mae = np.mean([float(row["mae"]) for row in variant_rows])
        print(f"  {variant_name:18s} MSE={avg_mse:.6f} MAE={avg_mae:.6f}")


def main():
    parser = build_parser()
    parser.description = "EPF ablation runner using best configs from results.txt"
    parser.set_defaults(model="MANTA_Ablation")
    parser.add_argument("--best_config_file", type=str, default="results.txt")
    args = parser.parse_args()
    run_epf_ablation(args)


if __name__ == "__main__":
    main()
