import numpy as np
import torch
import os
import csv
from torch.utils.data import Dataset
from typing import Tuple, Optional
from yucca.modules.data.augmentation.transforms.cropping_and_padding import CropPad
from yucca.modules.data.augmentation.transforms.formatting import NumpyToTorch


class OpenBHBDataset(Dataset):
    """
    Dataset class for OpenBHB brain age prediction.

    Reads participant metadata from train.tsv and loads quasiraw 3D .npy volumes.
    Each sample returns an image volume and the participant's age as a regression target.

    Expected directory structure:
        data_dir/
        ├── train.tsv
        └── train/
            └── quasiraw_3d/
                ├── {participant_id}_quasiraw_3d.npy
                └── ...
    """

    def __init__(
        self,
        samples: list,
        patch_size: Tuple[int, int, int],
        data_dir: str,
        composed_transforms=None,
        allow_missing_modalities: Optional[bool] = False,
        p_oversample_foreground: Optional[float] = None,
        task_type: str = "regression",
    ):
        """
        Args:
            samples: List of dicts with keys 'participant_id' and 'age'.
            patch_size: 3D patch size for cropping/padding.
            data_dir: Root directory of the OpenBHB dataset.
            composed_transforms: Optional torchvision transforms to apply.
        """
        super().__init__()
        self.samples = samples
        self.patch_size = patch_size
        self.data_dir = data_dir
        self.composed_transforms = composed_transforms
        self.image_dir = os.path.join(data_dir, "train", "quasiraw_3d")

        self.croppad = CropPad(patch_size=self.patch_size)
        self.to_torch = NumpyToTorch()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        participant_id = sample["participant_id"]
        age = sample["age"]

        # Load the 3D volume
        npy_path = os.path.join(
            self.image_dir, f"{participant_id}_quasiraw_3d.npy"
        )
        vol = np.load(npy_path, mmap_mode="r")

        # Add channel dimension if needed: (H, W, D) -> (1, H, W, D)
        if vol.ndim == 3:
            vol = vol[np.newaxis, ...]

        # Volume-wise z-normalization
        vol = vol.astype(np.float32, copy=True)
        mask = vol > 0
        if mask.any():
            mean = vol[mask].mean()
            std = vol[mask].std()
            if std > 0:
                vol = (vol - mean) / std

        # Build data dict compatible with the augmentation pipeline
        label = np.atleast_1d(np.float64(age))

        data_dict = {
            "file_path": npy_path,
            "image": vol,
            "label": label,
        }
        metadata = {"foreground_locations": []}

        return self._transform(data_dict, metadata)

    def _transform(self, data_dict, metadata=None):
        label = data_dict["label"]
        data_dict["label"] = None
        data_dict = self.croppad(data_dict, metadata)
        if self.composed_transforms is not None:
            data_dict = self.composed_transforms(data_dict)
        data_dict["label"] = label
        return self.to_torch(data_dict)


def load_openbhb_metadata(data_dir):
    """
    Load participant metadata from train.tsv.

    Args:
        data_dir: Root directory of the OpenBHB dataset containing train.tsv.

    Returns:
        List of dicts with keys 'participant_id' and 'age'.
    """
    tsv_path = os.path.join(data_dir, "train.tsv")
    image_dir = os.path.join(data_dir, "train", "quasiraw_3d")

    samples = []
    with open(tsv_path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            participant_id = row["participant_id"]
            npy_file = f"{participant_id}_quasiraw_3d.npy"
            # Only include samples whose image file exists
            if os.path.exists(os.path.join(image_dir, npy_file)):
                samples.append(
                    {
                        "participant_id": participant_id,
                        "age": float(row["age"]),
                    }
                )

    print(f"Loaded {len(samples)} samples with available images from {tsv_path}")
    return samples
