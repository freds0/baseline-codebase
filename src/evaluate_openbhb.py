#!/usr/bin/env python
"""
Evaluate a trained model on the OpenBHB brain age prediction dataset.

Loads a checkpoint, runs inference on the validation (or full) split,
and reports regression metrics (MAE, MSE, R2) along with per-sample predictions.

Usage:
    # Evaluate on validation split (same split used during training)
    python src/evaluate_openbhb.py \
        --data_dir /path/to/openbhb_train_sample \
        --checkpoint_path ./models/openbhb/OpenBHB_BrainAge/unet_b/version_0/checkpoints/last.ckpt

    # Evaluate on all data (no split)
    python src/evaluate_openbhb.py \
        --data_dir /path/to/openbhb_train_sample \
        --checkpoint_path ./models/openbhb/OpenBHB_BrainAge/unet_b/version_0/checkpoints/last.ckpt \
        --split all

    # Save per-sample predictions to CSV
    python src/evaluate_openbhb.py \
        --data_dir /path/to/openbhb_train_sample \
        --checkpoint_path ./models/openbhb/OpenBHB_BrainAge/unet_b/version_0/checkpoints/last.ckpt \
        --output_csv results.csv
"""

import argparse
import csv
import json
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.supervised_base import BaseSupervisedModel
from data.openbhb_dataset import OpenBHBDataset, load_openbhb_metadata
from data.task_configs import openbhb_config
from utils.utils import load_pretrained_weights

from yucca.modules.data.augmentation.YuccaAugmentationComposer import (
    YuccaAugmentationComposer,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on OpenBHB brain age prediction"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to OpenBHB dataset root (containing train.tsv)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to model checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="unet_b",
        help="Model architecture name (unet_b, unet_xl, etc.)",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=96,
        help="3D patch size (must match training)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "train", "all"],
        help="Which split to evaluate on (val, train, or all)",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.2,
        help="Fraction used for validation (must match training to reproduce the split)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for train/val split (must match training)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to save per-sample predictions as CSV",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Path to save summary metrics as JSON",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="32",
        choices=["32", "bf16-mixed"],
        help="Inference precision",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile (must match training)",
    )
    return parser.parse_args()


def build_eval_dataset(data_dir, patch_size, split, val_fraction, seed):
    """
    Build the evaluation dataset using the same split logic as training.

    Returns:
        (dataset, sample_list) where sample_list contains the original metadata dicts.
    """
    all_samples = load_openbhb_metadata(data_dir)
    assert len(all_samples) > 0, (
        f"No samples found in {data_dir}. "
        "Check that train.tsv exists and .npy files are in train/quasiraw_3d/"
    )

    if split == "all":
        eval_samples = all_samples
    else:
        # Reproduce the exact same split used during training
        rng = random.Random(seed)
        indices = list(range(len(all_samples)))
        rng.shuffle(indices)
        val_size = max(1, int(len(all_samples) * val_fraction))

        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        if split == "val":
            eval_samples = [all_samples[i] for i in val_indices]
        else:
            eval_samples = [all_samples[i] for i in train_indices]

    # Use validation transforms (no augmentation)
    augmenter = YuccaAugmentationComposer(
        patch_size=patch_size,
        task_type_preset="classification",
        parameter_dict={},
        deep_supervision=False,
    )

    dataset = OpenBHBDataset(
        samples=eval_samples,
        patch_size=patch_size,
        data_dir=data_dir,
        composed_transforms=augmenter.val_transforms,
    )
    return dataset, eval_samples


def load_model(checkpoint_path, model_name, patch_size, compile_flag):
    """
    Instantiate the model and load weights from a checkpoint.

    Returns:
        model on the appropriate device, in eval mode.
    """
    task_cfg = openbhb_config
    patch_size_tuple = (patch_size,) * 3

    # We need a version_dir for the model config; use the checkpoint's parent dir
    ckpt_dir = os.path.dirname(os.path.dirname(os.path.abspath(checkpoint_path)))
    version_dir = ckpt_dir if os.path.isdir(ckpt_dir) else "."

    config = {
        "task_name": task_cfg["task_name"],
        "task_type": task_cfg["task_type"],
        "model_name": model_name,
        "num_classes": task_cfg["num_classes"],
        "num_modalities": len(task_cfg["modalities"]),
        "patch_size": patch_size_tuple,
        "version_dir": version_dir,
    }

    model = BaseSupervisedModel.create(
        task_type=task_cfg["task_type"],
        config=config,
        learning_rate=1e-4,  # Not used during eval
        do_compile=compile_flag,
        compile_mode="default",
    )

    # Load checkpoint weights
    state_dict = load_pretrained_weights(checkpoint_path, compile_flag)
    model.load_state_dict(state_dict=state_dict, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, device


@torch.no_grad()
def run_evaluation(model, dataloader, device, use_amp):
    """
    Run inference on the entire dataloader and collect predictions.

    Returns:
        (all_predictions, all_targets, all_file_paths)
    """
    all_preds = []
    all_targets = []
    all_file_paths = []

    amp_dtype = torch.bfloat16 if use_amp else None

    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        labels = batch["label"].float()
        file_paths = batch["file_path"]

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            output = model(images)

        preds = output.cpu().squeeze(-1)  # (B,)
        targets = labels.squeeze(-1)  # (B,)

        all_preds.append(preds)
        all_targets.append(targets)
        all_file_paths.extend(file_paths)

        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {(batch_idx + 1) * dataloader.batch_size} samples...")

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    return all_preds, all_targets, all_file_paths


def compute_metrics(predictions, targets):
    """Compute regression metrics."""
    errors = predictions - targets
    abs_errors = errors.abs()

    mae = abs_errors.mean().item()
    mse = (errors ** 2).mean().item()
    rmse = mse ** 0.5

    # R2 score
    ss_res = (errors ** 2).sum().item()
    ss_tot = ((targets - targets.mean()) ** 2).sum().item()
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")

    # Median absolute error
    median_ae = abs_errors.median().item()

    # Percentiles of absolute error
    p90 = abs_errors.quantile(0.9).item()
    p95 = abs_errors.quantile(0.95).item()

    return {
        "n_samples": len(predictions),
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "median_ae": median_ae,
        "abs_error_p90": p90,
        "abs_error_p95": p95,
        "mean_error": errors.mean().item(),  # bias
        "target_mean": targets.mean().item(),
        "target_std": targets.std().item(),
        "pred_mean": predictions.mean().item(),
        "pred_std": predictions.std().item(),
    }


def save_predictions_csv(filepath, samples, predictions, targets):
    """Save per-sample predictions to a CSV file."""
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["participant_id", "true_age", "predicted_age", "error", "abs_error"])
        for sample, pred, target in zip(samples, predictions, targets):
            error = pred.item() - target.item()
            writer.writerow([
                sample["participant_id"],
                f"{target.item():.2f}",
                f"{pred.item():.2f}",
                f"{error:.2f}",
                f"{abs(error):.2f}",
            ])


def main():
    args = parse_args()

    assert args.patch_size % 8 == 0, (
        f"Patch size must be divisible by 8, got {args.patch_size}"
    )

    print("=" * 60)
    print("OpenBHB Model Evaluation")
    print("=" * 60)
    print(f"  Checkpoint:    {args.checkpoint_path}")
    print(f"  Data dir:      {args.data_dir}")
    print(f"  Model:         {args.model_name}")
    print(f"  Patch size:    {args.patch_size}")
    print(f"  Split:         {args.split}")
    print(f"  Precision:     {args.precision}")

    if not os.path.isfile(args.checkpoint_path):
        print(f"ERROR: Checkpoint not found: {args.checkpoint_path}")
        return 1
    if not os.path.isdir(args.data_dir):
        print(f"ERROR: Data directory not found: {args.data_dir}")
        return 1

    # Build dataset
    print("\n" + "-" * 60)
    print(f"Loading {args.split} dataset...")
    print("-" * 60)
    dataset, eval_samples = build_eval_dataset(
        data_dir=args.data_dir,
        patch_size=(args.patch_size,) * 3,
        split=args.split,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    print(f"Evaluation samples: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    # Load model
    print("\n" + "-" * 60)
    print("Loading model...")
    print("-" * 60)
    model, device = load_model(
        checkpoint_path=args.checkpoint_path,
        model_name=args.model_name,
        patch_size=args.patch_size,
        compile_flag=args.compile,
    )
    print(f"Model loaded on {device}")

    # Run evaluation
    print("\n" + "-" * 60)
    print("Running inference...")
    print("-" * 60)
    use_amp = args.precision == "bf16-mixed"
    predictions, targets, file_paths = run_evaluation(model, dataloader, device, use_amp)

    # Compute metrics
    print("\n" + "-" * 60)
    print("Results")
    print("-" * 60)
    metrics = compute_metrics(predictions, targets)

    print(f"\n  Samples evaluated:   {metrics['n_samples']}")
    print(f"  MAE:                 {metrics['mae']:.2f} years")
    print(f"  MSE:                 {metrics['mse']:.2f}")
    print(f"  RMSE:                {metrics['rmse']:.2f} years")
    print(f"  R2 Score:            {metrics['r2']:.4f}")
    print(f"  Median AE:           {metrics['median_ae']:.2f} years")
    print(f"  90th pct AE:         {metrics['abs_error_p90']:.2f} years")
    print(f"  95th pct AE:         {metrics['abs_error_p95']:.2f} years")
    print(f"  Mean Error (bias):   {metrics['mean_error']:.2f} years")
    print(f"\n  Target age:  {metrics['target_mean']:.1f} +/- {metrics['target_std']:.1f}")
    print(f"  Predicted:   {metrics['pred_mean']:.1f} +/- {metrics['pred_std']:.1f}")

    # Save per-sample predictions
    if args.output_csv:
        save_predictions_csv(args.output_csv, eval_samples, predictions, targets)
        print(f"\nPer-sample predictions saved to: {args.output_csv}")

    # Save summary metrics
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(
                {
                    "checkpoint": args.checkpoint_path,
                    "data_dir": args.data_dir,
                    "split": args.split,
                    "model_name": args.model_name,
                    "patch_size": args.patch_size,
                    "metrics": metrics,
                },
                f,
                indent=2,
            )
        print(f"Summary metrics saved to: {args.output_json}")

    print("\n" + "=" * 60)
    print("Evaluation complete.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
