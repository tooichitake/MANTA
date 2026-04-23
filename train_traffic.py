"""
train.py -- direct training loop with support for MANTA ablation experiments.
"""

import argparse
import copy
import csv
import gc
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.backends
import torch.nn as nn
from torch import optim

from tqdm import tqdm

from data_provider.data_factory import data_provider
from exp.exp_basic import get_model_class
from utils.metrics import metric


EPF_DATASETS = {
    "NP": {
        "data": "custom",
        "root_path": "./datasets/EPF/",
        "data_path": "NP.csv",
        "target": "OT",
        "enc_in": 3,
        "dec_in": 3,
        "c_out": 3,
        "freq": "h",
    },
    "PJM": {
        "data": "custom",
        "root_path": "./datasets/EPF/",
        "data_path": "PJM.csv",
        "target": "OT",
        "enc_in": 3,
        "dec_in": 3,
        "c_out": 3,
        "freq": "h",
    },
    "BE": {
        "data": "custom",
        "root_path": "./datasets/EPF/",
        "data_path": "BE.csv",
        "target": "OT",
        "enc_in": 3,
        "dec_in": 3,
        "c_out": 3,
        "freq": "h",
    },
    "FR": {
        "data": "custom",
        "root_path": "./datasets/EPF/",
        "data_path": "FR.csv",
        "target": "OT",
        "enc_in": 3,
        "dec_in": 3,
        "c_out": 3,
        "freq": "h",
    },
    "DE": {
        "data": "custom",
        "root_path": "./datasets/EPF/",
        "data_path": "DE.csv",
        "target": "OT",
        "enc_in": 3,
        "dec_in": 3,
        "c_out": 3,
        "freq": "h",
    },
}


DEFAULT_EPf_VARIANTS = [
    "full",
    "single_scale_na",
    "full_self_attn",
    "full_cross_attn",
    "no_var_gate",
    "no_cross_encoder",
    "separate_encoders",
]


ECL_M_FULL_BASE_CONFIG = {
    "dataset_name": "ECL",
    "task_name": "long_term_forecast",
    "model": "MANTA_CUDA",
    "features": "M",
    "target": "OT",
    "freq": "h",
    "data": "custom",
    "root_path": "./datasets/electricity/",
    "data_path": "electricity.csv",
    "seasonal_patterns": "Monthly",
    "embed": "timeF",
    "num_workers": 0,
    "augmentation_ratio": 0,
    "enc_in": 321,
    "dec_in": 321,
    "c_out": 321,
    "seq_len": 96,
    "label_len": 48,
    "patch_len": 8,
    "train_epochs": 10,
    "patience": 3,
    "use_gpu": True,
    "gpu": 0,
    "gpu_type": "cuda",
}


ECL_M_FULL_CONFIGS = [
    {
        **ECL_M_FULL_BASE_CONFIG,
        "pred_len": 96,
        "d_model": 576,
        "d_ff": 1152,
        "n_heads": 16,
        "e_layers": 3,
        "batch_size": 16,
        "learning_rate": 7e-4,
        "dropout": 0.05,
        "lradj": "cosine",
        "activation": "relu",
    },
    {
        **ECL_M_FULL_BASE_CONFIG,
        "pred_len": 192,
        "d_model": 576,
        "d_ff": 1152,
        "n_heads": 16,
        "e_layers": 3,
        "batch_size": 16,
        "learning_rate": 7e-4,
        "dropout": 0.05,
        "lradj": "type1",
        "activation": "relu",
    },
    {
        **ECL_M_FULL_BASE_CONFIG,
        "pred_len": 336,
        "d_model": 576,
        "d_ff": 256,
        "n_heads": 16,
        "e_layers": 3,
        "batch_size": 16,
        "learning_rate": 7e-4,
        "dropout": 0.05,
        "lradj": "type1",
        "activation": "gelu",
    },
    {
        **ECL_M_FULL_BASE_CONFIG,
        "pred_len": 720,
        "d_model": 576,
        "d_ff": 1152,
        "n_heads": 16,
        "e_layers": 3,
        "batch_size": 16,
        "learning_rate": 7e-4,
        "dropout": 0.05,
        "lradj": "cosine",
        "activation": "relu",
    },
]


PRINT_KEYS = [
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


def fix_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj == "type1":
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == "type2":
        lr_adjust = {
            2: 5e-5,
            4: 1e-5,
            6: 5e-6,
            8: 1e-6,
            10: 5e-7,
            15: 1e-7,
            20: 5e-8,
        }
    elif args.lradj == "type3":
        lr_adjust = {
            epoch: (
                args.learning_rate
                if epoch < 3
                else args.learning_rate * (0.9 ** ((epoch - 3) // 1))
            )
        }
    elif args.lradj == "cosine":
        lr_adjust = {
            epoch: args.learning_rate
            / 2
            * (1 + math.cos(epoch / args.train_epochs * math.pi))
        }
    else:
        lr_adjust = {}
    if epoch in lr_adjust:
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


def vali(model, vali_loader, criterion, args, device):
    model.eval()
    total_loss = []
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in tqdm(
            vali_loader, desc="    Vali", leave=False
        ):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len :, :]).float()
            dec_inp = (
                torch.cat([batch_y[:, : args.label_len, :], dec_inp], dim=1)
                .float()
                .to(device)
            )
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if args.features == "MS" else 0
            outputs = outputs[:, -args.pred_len :, f_dim:]
            batch_y = batch_y[:, -args.pred_len :, f_dim:].to(device)
            loss = criterion(outputs.detach(), batch_y.detach())
            total_loss.append(loss.item())
    model.train()
    return np.average(total_loss)


def test_evaluate(model, test_loader, args, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in tqdm(
            test_loader, desc="    Test", leave=False
        ):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len :, :]).float()
            dec_inp = (
                torch.cat([batch_y[:, : args.label_len, :], dec_inp], dim=1)
                .float()
                .to(device)
            )
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if args.features == "MS" else 0
            outputs = outputs[:, -args.pred_len :, :].detach().cpu().numpy()
            batch_y = batch_y[:, -args.pred_len :, :].detach().cpu().numpy()
            preds.append(outputs[:, :, f_dim:])
            trues.append(batch_y[:, :, f_dim:])
    preds = np.concatenate(preds, axis=0).reshape(
        -1, preds[0].shape[-2], preds[0].shape[-1]
    )
    trues = np.concatenate(trues, axis=0).reshape(
        -1, trues[0].shape[-2], trues[0].shape[-1]
    )
    mae, mse, _, _, _ = metric(preds, trues)
    model.train()
    return mse, mae


def build_parser():
    parser = argparse.ArgumentParser(description="Direct training loop for MANTA and MANTA ablations")
    parser.add_argument("--model", type=str, default="MANTA_CUDA")
    parser.add_argument("--ablation_variant", type=str, default="full")
    parser.add_argument("--run_epf_ablation", action="store_true", default=False)
    parser.add_argument("--single_run", action="store_true", default=False)
    parser.add_argument("--datasets", type=str, default="NP,PJM,BE,FR,DE")
    parser.add_argument("--variants", type=str, default=",".join(DEFAULT_EPf_VARIANTS))
    parser.add_argument("--results_file", type=str, default="ablation_results.csv")
    parser.add_argument("--seed", type=int, default=2021)

    parser.add_argument("--task_name", type=str, default="long_term_forecast")
    parser.add_argument("--model_id", type=str, default="test")
    parser.add_argument("--features", type=str, default="M")
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--data", type=str, default="custom")
    parser.add_argument("--root_path", type=str, default="./datasets/electricity/")
    parser.add_argument("--data_path", type=str, default="electricity.csv")
    parser.add_argument("--seasonal_patterns", type=str, default="Monthly")
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--augmentation_ratio", type=int, default=0)

    parser.add_argument("--enc_in", type=int, default=321)
    parser.add_argument("--dec_in", type=int, default=321)
    parser.add_argument("--c_out", type=int, default=321)
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=336)
    parser.add_argument("--patch_len", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=576)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=16)
    parser.add_argument("--e_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--activation", type=str, default="gelu")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=7e-4)
    parser.add_argument("--lradj", type=str, default="type1")

    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpu_type", type=str, default="cuda")
    return parser


def finalize_args(args):
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device("cuda:{}".format(args.gpu))
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and args.use_gpu:
        args.device = torch.device("mps")
    else:
        args.device = torch.device("cpu")
    return args


def apply_epf_defaults(args, dataset_name, variant_name):
    ds = EPF_DATASETS[dataset_name]
    args.data = ds["data"]
    args.root_path = ds["root_path"]
    args.data_path = ds["data_path"]
    args.target = ds["target"]
    args.enc_in = ds["enc_in"]
    args.dec_in = ds["dec_in"]
    args.c_out = ds["c_out"]
    args.freq = ds["freq"]
    args.features = "MS"
    args.seq_len = 168
    args.label_len = 24
    args.pred_len = 24
    args.patch_len = 24
    args.model = "MANTA_Ablation"
    args.ablation_variant = variant_name
    args.model_id = f"{dataset_name}_{variant_name}"
    return args


def apply_traffic_config(args, config):
    for key, value in config.items():
        setattr(args, key, value)
    args.model_id = f'{config["dataset_name"]}_{config["pred_len"]}'
    return args


def print_run_config(args):
    dataset_name = getattr(args, "dataset_name", args.data)
    print(f'    dataset = "{dataset_name}"')
    print(f'    model = "{args.model}"')
    print(f'    features = "{args.features}"')
    for key in PRINT_KEYS:
        print(f"    {key} = {getattr(args, key)}")


def run_training(args):
    device = args.device

    ModelClass = get_model_class(args.model)
    model = ModelClass(args).float().to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {n_params:,}")

    _, train_loader = data_provider(args, flag="train")
    _, vali_loader = data_provider(args, flag="val")
    _, test_loader = data_provider(args, flag="test")
    model_optim = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    best_vali_loss = np.inf
    best_model_state = None
    patience_counter = 0

    for epoch in range(args.train_epochs):
        model.train()
        train_losses = []
        for batch_x, batch_y, batch_x_mark, batch_y_mark in tqdm(
            train_loader, desc=f"  Epoch {epoch+1}", leave=False
        ):
            model_optim.zero_grad()
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len :, :]).float()
            dec_inp = (
                torch.cat([batch_y[:, : args.label_len, :], dec_inp], dim=1)
                .float()
                .to(device)
            )

            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if args.features == "MS" else 0
            outputs = outputs[:, -args.pred_len :, f_dim:]
            batch_y = batch_y[:, -args.pred_len :, f_dim:].to(device)
            loss = criterion(outputs, batch_y)

            loss.backward()
            model_optim.step()

            train_losses.append(loss.item())

        train_loss = np.average(train_losses)
        vali_loss = vali(model, vali_loader, criterion, args, device)

        print(f"  Epoch {epoch+1} | Train: {train_loss:.7f} " f"Vali: {vali_loss:.7f}")

        if vali_loss < best_vali_loss:
            best_vali_loss = vali_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  EarlyStopping counter: {patience_counter} out of {args.patience}")
            if patience_counter >= args.patience:
                print("  Early stopping")
                break

        adjust_learning_rate(model_optim, epoch + 1, args)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    test_mse, test_mae = test_evaluate(model, test_loader, args, device)

    del model, model_optim
    del train_loader, vali_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return test_mse, test_mae, n_params


def append_result(results_file, row):
    path = Path(results_file)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="ascii") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "model",
                "variant",
                "mse",
                "mae",
                "params",
                "seq_len",
                "pred_len",
                "patch_len",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_epf_ablation(args):
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]

    all_rows = []
    for dataset_name in datasets:
        if dataset_name not in EPF_DATASETS:
            raise ValueError(f"Unknown EPF dataset: {dataset_name}")
        for variant_name in variants:
            run_args = copy.deepcopy(args)
            run_args = apply_epf_defaults(run_args, dataset_name, variant_name)
            run_args = finalize_args(run_args)
            fix_random_seed(run_args.seed)

            print("=" * 72)
            print(f"dataset={dataset_name} variant={variant_name} model={run_args.model}")
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


def print_traffic_summary(summary_rows):
    if not summary_rows:
        return

    print("\nECL M Full Summary")
    print(f'{"pred_len":>8} {"MSE":>12} {"MAE":>12}')
    for row in summary_rows:
        print(f'{row["pred_len"]:>8} {row["mse"]:>12.6f} {row["mae"]:>12.6f}')


def run_traffic_m_full(args):
    summary_rows = []
    for config in ECL_M_FULL_CONFIGS:
        run_args = copy.deepcopy(args)
        run_args = apply_traffic_config(run_args, config)
        run_args = finalize_args(run_args)
        fix_random_seed(run_args.seed)

        print("=" * 72)
        print(f'dataset="{run_args.dataset_name}" pred_len={run_args.pred_len} model={run_args.model}')
        print("=" * 72)
        print_run_config(run_args)

        test_mse, test_mae, _ = run_training(run_args)
        print(f"MSE={test_mse:.6f}, MAE={test_mae:.6f}")
        summary_rows.append(
            {
                "pred_len": run_args.pred_len,
                "mse": test_mse,
                "mae": test_mae,
            }
        )

    print_traffic_summary(summary_rows)


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.run_epf_ablation:
        run_epf_ablation(args)
        return

    if not args.single_run:
        run_traffic_m_full(args)
        return

    args = finalize_args(args)
    fix_random_seed(args.seed)
    print_run_config(args)
    test_mse, test_mae, _ = run_training(args)
    print(f"MSE={test_mse:.6f}, MAE={test_mae:.6f}")


if __name__ == "__main__":
    main()


