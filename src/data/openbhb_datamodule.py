import lightning as pl
import logging
import torch
import random
from typing import Optional, Tuple
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from data.openbhb_dataset import OpenBHBDataset, load_openbhb_metadata


class OpenBHBDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for the OpenBHB brain age prediction dataset.

    Handles loading metadata from train.tsv, splitting into train/val sets,
    and creating DataLoaders with appropriate augmentations.
    """

    def __init__(
        self,
        data_dir: str,
        patch_size: Tuple[int, int, int],
        batch_size: int,
        num_workers: int = 8,
        val_fraction: float = 0.2,
        seed: int = 42,
        composed_train_transforms: Optional[Compose] = None,
        composed_val_transforms: Optional[Compose] = None,
    ):
        """
        Args:
            data_dir: Root directory of the OpenBHB dataset.
            patch_size: 3D patch size for cropping/padding.
            batch_size: Batch size for DataLoaders.
            num_workers: Number of workers for DataLoaders.
            val_fraction: Fraction of data to use for validation.
            seed: Random seed for reproducible train/val split.
            composed_train_transforms: Augmentation transforms for training.
            composed_val_transforms: Augmentation transforms for validation.
        """
        super().__init__()
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.seed = seed
        self.composed_train_transforms = composed_train_transforms
        self.composed_val_transforms = composed_val_transforms

        self.num_workers = (
            max(0, int(torch.get_num_threads() - 1))
            if num_workers is None
            else num_workers
        )

        logging.info(f"Using {self.num_workers} workers")

    def setup(self, stage: str = "fit"):
        assert stage == "fit"

        all_samples = load_openbhb_metadata(self.data_dir)
        assert len(all_samples) > 0, (
            f"No samples found in {self.data_dir}. "
            "Check that train.tsv exists and .npy files are in train/quasiraw_3d/"
        )

        # Reproducible train/val split
        rng = random.Random(self.seed)
        indices = list(range(len(all_samples)))
        rng.shuffle(indices)
        val_size = max(1, int(len(all_samples) * self.val_fraction))

        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        self.train_samples = [all_samples[i] for i in train_indices]
        self.val_samples = [all_samples[i] for i in val_indices]

        print(
            f"OpenBHB split: {len(self.train_samples)} train, "
            f"{len(self.val_samples)} val"
        )

        self.train_dataset = OpenBHBDataset(
            samples=self.train_samples,
            patch_size=self.patch_size,
            data_dir=self.data_dir,
            composed_transforms=self.composed_train_transforms,
        )

        self.val_dataset = OpenBHBDataset(
            samples=self.val_samples,
            patch_size=self.patch_size,
            data_dir=self.data_dir,
            composed_transforms=self.composed_val_transforms,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=False,
        )
