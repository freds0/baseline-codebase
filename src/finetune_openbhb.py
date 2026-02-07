#!/usr/bin/env python
"""
Fine-tuning script for brain age prediction on the OpenBHB dataset.

Adapts the FOMO25 baseline codebase for regression on OpenBHB quasiraw 3D MRI volumes.

Usage:
    # Train from scratch
    python src/finetune_openbhb.py \
        --data_dir /path/to/openbhb_train_sample \
        --save_dir ./models/openbhb \
        --patch_size 96 \
        --batch_size 2 \
        --epochs 100

    # Fine-tune from pretrained weights
    python src/finetune_openbhb.py \
        --data_dir /path/to/openbhb_train_sample \
        --save_dir ./models/openbhb \
        --pretrained_weights_path /path/to/pretrained.ckpt \
        --patch_size 96 \
        --batch_size 2 \
        --epochs 100
"""

import argparse
import os
import logging
import torch
import lightning as L
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint

from models.supervised_base import BaseSupervisedModel
from augmentations.finetune_augmentation_presets import (
    get_finetune_augmentation_params,
)
from utils.utils import (
    setup_seed,
    find_checkpoint,
    load_pretrained_weights,
)

from batchgenerators.utilities.file_and_folder_operations import (
    maybe_mkdir_p as ensure_dir_exists,
)

from yucca.modules.data.augmentation.YuccaAugmentationComposer import (
    YuccaAugmentationComposer,
)
from yucca.modules.callbacks.loggers import YuccaLogger
from yucca.pipeline.configuration.configure_paths import detect_version

from data.openbhb_datamodule import OpenBHBDataModule
from data.openbhb_dataset import load_openbhb_metadata
from data.task_configs import openbhb_config


def main():
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(
        description="Fine-tune for brain age prediction on OpenBHB"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to OpenBHB dataset root (containing train.tsv)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./models/openbhb",
        help="Path to save models and results",
    )
    parser.add_argument(
        "--pretrained_weights_path",
        type=str,
        default=None,
        help="Path to pretrained checkpoint for finetuning",
    )
    # Model configuration
    parser.add_argument(
        "--model_name",
        type=str,
        default="unet_b",
        help="Model name (unet_b, unet_xl, etc.)",
    )
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--patch_size", type=int, default=96)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile_mode", type=str, default=None)
    # Hardware configuration
    parser.add_argument("--num_devices", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--fast_dev_run", action="store_true")
    # Experiment tracking
    parser.add_argument("--new_version", action="store_true")
    parser.add_argument(
        "--augmentation_preset",
        type=str,
        choices=["all", "basic", "none"],
        default="basic",
    )
    # Training parameters
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--train_batches_per_epoch", type=int, default=100)
    # Split configuration
    parser.add_argument(
        "--val_fraction", type=float, default=0.2, help="Fraction of data for validation"
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for train/val split")
    parser.add_argument(
        "--experiment", type=str, default="brain_age", help="Name of experiment"
    )
    parser.add_argument(
        "--disable_wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    args = parser.parse_args()

    assert (
        args.patch_size % 8 == 0
    ), f"Patch size must be divisible by 8, got {args.patch_size}"

    # Task configuration from openbhb_config
    task_cfg = openbhb_config
    task_type = task_cfg["task_type"]
    task_name = task_cfg["task_name"]
    num_classes = task_cfg["num_classes"]
    num_modalities = len(task_cfg["modalities"])
    labels = task_cfg["labels"]

    run_type = "from_scratch" if args.pretrained_weights_path is None else "finetune"
    experiment_name = f"{run_type}_{args.model_name}_{args.experiment}_openbhb"

    print(f"Using num_workers: {args.num_workers}, num_devices: {args.num_devices}")
    print(f"Task: {task_name} ({task_type})")
    print(f"Modalities: {num_modalities}, Output dim: {num_classes}")
    print("ARGS:", args)

    # Count available samples
    all_samples = load_openbhb_metadata(args.data_dir)
    val_size = max(1, int(len(all_samples) * args.val_fraction))
    train_dataset_size = len(all_samples) - val_size
    val_dataset_size = val_size

    # Set up directory structure
    save_dir = os.path.join(args.save_dir, task_name, args.model_name)

    # Handle versioning
    continue_from_most_recent = not args.new_version
    version = detect_version(save_dir, continue_from_most_recent)
    version_dir = os.path.join(save_dir, f"version_{version}")
    ensure_dir_exists(version_dir)

    # Set up seed
    seed = setup_seed(continue_from_most_recent)
    ckpt_path = find_checkpoint(version_dir, continue_from_most_recent)

    # Calculate training metrics
    effective_batch_size = args.num_devices * args.batch_size
    max_iterations = int(args.epochs * args.train_batches_per_epoch)

    config = {
        # Task information
        "task": task_name,
        "task_type": task_type,
        "experiment": experiment_name,
        "model_name": args.model_name,
        "model_dimensions": "3D",
        "run_type": run_type,
        # Directories
        "save_dir": save_dir,
        "train_data_dir": args.data_dir,
        "version_dir": version_dir,
        "version": version,
        # Checkpoint
        "ckpt_path": ckpt_path,
        "pretrained_weights_path": args.pretrained_weights_path,
        # Reproducibility
        "seed": seed,
        # Dataset properties
        "num_classes": num_classes,
        "num_modalities": num_modalities,
        "labels": labels,
        # Training parameters
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "patch_size": (args.patch_size,) * 3,
        "precision": args.precision,
        "augmentation_preset": args.augmentation_preset,
        "epochs": args.epochs,
        "train_batches_per_epoch": args.train_batches_per_epoch,
        "effective_batch_size": effective_batch_size,
        # Dataset metrics
        "train_dataset_size": train_dataset_size,
        "val_dataset_size": val_dataset_size,
        "max_iterations": max_iterations,
        # Hardware settings
        "num_devices": args.num_devices,
        "num_workers": args.num_workers,
        # Model compilation
        "compile": args.compile,
        "compile_mode": args.compile_mode,
        # Trainer
        "fast_dev_run": args.fast_dev_run,
    }

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=10,
        save_top_k=1,
        filename="last",
        enable_version_counter=False,
    )

    best_checkpoint_callback = ModelCheckpoint(
        monitor="val/mae",
        mode="min",
        save_top_k=1,
        filename="best-mae-{epoch:03d}-{val/mae:.2f}",
    )

    callbacks = [checkpoint_callback, best_checkpoint_callback]

    # Loggers
    yucca_logger = YuccaLogger(
        save_dir=save_dir,
        version=version,
        steps_per_epoch=args.train_batches_per_epoch,
    )
    loggers = [yucca_logger]

    if not args.disable_wandb:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            project="openbhb-brain-age",
            name=f"{config['experiment']}_version_{config['version']}",
            log_model=True,
        )
        loggers.append(wandb_logger)

    # Augmentations (use classification preset for regression, same as original code)
    aug_params = get_finetune_augmentation_params(args.augmentation_preset)
    augmenter = YuccaAugmentationComposer(
        patch_size=config["patch_size"],
        task_type_preset="classification",
        parameter_dict=aug_params,
        deep_supervision=False,
    )

    # Data module
    data_module = OpenBHBDataModule(
        data_dir=args.data_dir,
        patch_size=config["patch_size"],
        batch_size=config["batch_size"],
        num_workers=args.num_workers,
        val_fraction=args.val_fraction,
        seed=args.seed,
        composed_train_transforms=augmenter.train_transforms,
        composed_val_transforms=augmenter.val_transforms,
    )

    print(f"Run type: {run_type}")
    print(
        f"Starting training with {max_iterations} max iterations over {args.epochs} epochs "
        f"with train dataset of ~{train_dataset_size} datapoints and val dataset of ~{val_dataset_size} "
        f"and effective batch size of {effective_batch_size}"
    )

    # Create regression model
    model = BaseSupervisedModel.create(
        task_type=task_type,
        config=config,
        learning_rate=args.learning_rate,
        do_compile=args.compile,
        compile_mode="default" if args.compile_mode is None else args.compile_mode,
    )

    # Trainer
    trainer = L.Trainer(
        callbacks=callbacks,
        logger=loggers,
        accelerator="auto" if torch.cuda.is_available() else "cpu",
        strategy="auto",
        num_nodes=1,
        devices=args.num_devices,
        default_root_dir=save_dir,
        max_epochs=args.epochs,
        limit_train_batches=args.train_batches_per_epoch,
        precision=args.precision,
        fast_dev_run=args.fast_dev_run,
    )

    # Load pretrained weights if finetuning
    if run_type == "finetune":
        print("Transferring weights for finetuning")
        print(f"Checkpoint path: {ckpt_path}")
        assert ckpt_path is None, (
            "Error: You're attempting to load pretrained weights while "
            "simultaneously continuing from a checkpoint. Use either "
            "--pretrained_weights_path for finetuning OR continue training "
            "without the --new_version flag, but not both."
        )
        state_dict = load_pretrained_weights(args.pretrained_weights_path, args.compile)
        num_successful = model.load_state_dict(state_dict=state_dict, strict=False)
        assert num_successful > 0, "No weights were successfully transferred"
    else:
        print("Training from scratch, no weights will be transferred")

    # Start training
    trainer.fit(model=model, datamodule=data_module, ckpt_path="last")

    if not args.disable_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
