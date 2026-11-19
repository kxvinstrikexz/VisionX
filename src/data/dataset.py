"""
PyTorch Dataset and DataLoader for Satellite IR to RGB Image Translation.
Supports both single-band NIR and multispectral IR inputs with data augmentations.
"""

import os
import glob
import random
from typing import Tuple, Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SatelliteIRDataset(Dataset):
    """
    Dataset for paired Satellite Infrared and RGB imagery.
    
    Args:
        data_dir: Path to split folder containing .npz files.
        input_mode: 'single' for 1-channel NIR or 'multispectral' for 3-channel IR composite.
        augment: Whether to apply spatial/radiometric data augmentations.
    """
    def __init__(self, data_dir: str, input_mode: str = "single", augment: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.input_mode = input_mode.lower()
        self.augment = augment
        
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No .npz dataset files found in {data_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def _apply_augmentations(self, ir: np.ndarray, rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Applies spatial and radiometric augmentations consistently to IR and RGB."""
        # Random horizontal flip
        if random.random() > 0.5:
            ir = np.flip(ir, axis=-1)
            rgb = np.flip(rgb, axis=-1)

        # Random vertical flip
        if random.random() > 0.5:
            ir = np.flip(ir, axis=-2)
            rgb = np.flip(rgb, axis=-2)

        # Random 90-degree rotations
        k = random.choice([0, 1, 2, 3])
        if k > 0:
            ir = np.rot90(ir, k, axes=(-2, -1))
            rgb = np.rot90(rgb, k, axes=(-2, -1))

        # Radiometric jitter (scale reflectance by 0.95 to 1.05)
        if random.random() > 0.5:
            scale = random.uniform(0.95, 1.05)
            ir = np.clip(ir * scale, 0.0, 1.0)
            rgb = np.clip(rgb * scale, 0.0, 1.0)

        return np.ascontiguousarray(ir), np.ascontiguousarray(rgb)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        file_path = self.files[idx]
        data = np.load(file_path, allow_pickle=True)

        # Choose input modality
        if self.input_mode == "single":
            ir = data['ir_single'].astype(np.float32)  # (1, H, W)
        else:
            ir = data['ir_multi'].astype(np.float32)   # (3, H, W)

        rgb = data['rgb'].astype(np.float32)           # (3, H, W)

        if self.augment:
            ir, rgb = self._apply_augmentations(ir, rgb)

        meta = {
            'class_name': str(data.get('class_name', 'unknown')),
            'source_file': str(data.get('source_file', os.path.basename(file_path))),
            'path': file_path
        }

        ir_tensor = torch.from_numpy(ir)
        rgb_tensor = torch.from_numpy(rgb)

        return ir_tensor, rgb_tensor, meta


def create_dataloaders(
    processed_dir: str = "data/processed",
    input_mode: str = "single",
    batch_size: int = 16,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Creates train, val, and test DataLoaders."""
    train_dir = os.path.join(processed_dir, "train")
    val_dir = os.path.join(processed_dir, "val")
    test_dir = os.path.join(processed_dir, "test")

    train_dataset = SatelliteIRDataset(train_dir, input_mode=input_mode, augment=True)
    val_dataset = SatelliteIRDataset(val_dir, input_mode=input_mode, augment=False)
    test_dataset = SatelliteIRDataset(test_dir, input_mode=input_mode, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_ld, val_ld, test_ld = create_dataloaders(input_mode="single", batch_size=4)
    ir, rgb, meta = next(iter(train_ld))
    print(f"Batch IR shape: {ir.shape}, RGB shape: {rgb.shape}")
    print(f"Classes in batch: {meta['class_name']}")
    assert ir.shape == (4, 1, 64, 64)
    assert rgb.shape == (4, 3, 64, 64)
    print("DataLoader smoke test passed!")
