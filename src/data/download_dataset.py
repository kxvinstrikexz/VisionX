"""
Automated Dataset Downloader and Multispectral Preprocessing Pipeline.
Downloads Sentinel-2 satellite imagery (EuroSAT), extracts Infrared and RGB bands,
applies radiometric calibration, and produces stratified train/val/test splits.
"""

import os
import glob
import io
import json
import random
import shutil
import urllib.request
import zipfile
import argparse
from typing import Dict, List, Tuple

import numpy as np
import tifffile
from PIL import Image

# Sentinel-2 Multispectral Band Indices in EuroSAT (0-indexed)
BAND_INDICES = {
    'B01_AEROSOL': 0,
    'B02_BLUE': 1,
    'B03_GREEN': 2,
    'B04_RED': 3,
    'B05_RE1': 4,
    'B06_RE2': 5,
    'B07_RE3': 6,
    'B08_NIR': 7,       # Near-Infrared (~842nm)
    'B8A_NARROW_NIR': 8,
    'B09_WATER_VAPOR': 9,
    'B10_CIRRUS': 10,
    'B11_SWIR1': 11,    # Short-Wave Infrared 1 (~1610nm)
    'B12_SWIR2': 12     # Short-Wave Infrared 2 (~2190nm)
}

DEFAULT_NORM_RGB = 3000.0
DEFAULT_NORM_IR = 4000.0


def download_eurosat100(dest_dir: str = "data/raw/eurosat100") -> str:
    """Downloads EuroSAT100 multispectral benchmark dataset if not present."""
    os.makedirs(dest_dir, exist_ok=True)
    existing_tifs = glob.glob(os.path.join(dest_dir, "**/*.tif"), recursive=True)
    if len(existing_tifs) >= 100:
        print(f"[Dataset] Found {len(existing_tifs)} existing Sentinel-2 TIFF files in {dest_dir}")
        return dest_dir

    url = "https://huggingface.co/datasets/torchgeo/eurosat/resolve/main/EuroSAT100.zip"
    print(f"[Dataset] Downloading Sentinel-2 EuroSAT100 dataset from {url}...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp:
        data = resp.read()

    print("[Dataset] Extracting archive...")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(dest_dir)

    tif_count = len(glob.glob(os.path.join(dest_dir, "**/*.tif"), recursive=True))
    print(f"[Dataset] Extracted {tif_count} multispectral TIFF images successfully.")
    return dest_dir


def process_sentinel2_tif(file_path: str, norm_rgb: float = DEFAULT_NORM_RGB, norm_ir: float = DEFAULT_NORM_IR) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads a 13-band Sentinel-2 TIFF, extracts IR & RGB channels,
    and returns radiometrically calibrated arrays in range [0, 1].
    
    Returns:
        ir_single: (1, H, W) float32 (NIR band)
        ir_multi: (3, H, W) float32 (NIR, SWIR-1, SWIR-2 composite)
        rgb: (3, H, W) float32 (Red, Green, Blue true color)
    """
    arr = tifffile.imread(file_path).astype(np.float32)  # (H, W, 13)
    
    # Extract True Color RGB: B04 (Red), B03 (Green), B02 (Blue)
    r = arr[:, :, BAND_INDICES['B04_RED']]
    g = arr[:, :, BAND_INDICES['B03_GREEN']]
    b = arr[:, :, BAND_INDICES['B02_BLUE']]
    rgb = np.stack([r, g, b], axis=0)  # (3, H, W)
    rgb = np.clip(rgb / norm_rgb, 0.0, 1.0)

    # Extract Near-Infrared (NIR) Band 8
    nir = arr[:, :, BAND_INDICES['B08_NIR']]
    ir_single = np.expand_dims(nir, axis=0)  # (1, H, W)
    ir_single = np.clip(ir_single / norm_ir, 0.0, 1.0)

    # Extract Multi-Infrared Composite: NIR (B08), SWIR-1 (B11), SWIR-2 (B12)
    swir1 = arr[:, :, BAND_INDICES['B11_SWIR1']]
    swir2 = arr[:, :, BAND_INDICES['B12_SWIR2']]
    ir_multi = np.stack([nir, swir1, swir2], axis=0)  # (3, H, W)
    ir_multi = np.clip(ir_multi / norm_ir, 0.0, 1.0)

    return ir_single, ir_multi, rgb


def prepare_dataset(
    raw_dir: str = "data/raw/eurosat100",
    processed_dir: str = "data/processed",
    train_split: float = 0.75,
    val_split: float = 0.15,
    seed: int = 42
) -> Dict[str, int]:
    """
    Splits data into train, val, test, converts to processed .npz format,
    and generates companion preview images.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Find all tif files grouped by class directory
    tif_files = glob.glob(os.path.join(raw_dir, "**/*.tif"), recursive=True)
    if not tif_files:
        raise FileNotFoundError(f"No .tif files found in {raw_dir}")

    class_to_files: Dict[str, List[str]] = {}
    for f in tif_files:
        cls_name = os.path.basename(os.path.dirname(f))
        if cls_name not in class_to_files:
            class_to_files[cls_name] = []
        class_to_files[cls_name].append(f)

    splits = {'train': [], 'val': [], 'test': []}
    for cls_name, files in class_to_files.items():
        files_shuffled = files.copy()
        random.shuffle(files_shuffled)
        n = len(files_shuffled)
        n_train = max(1, int(n * train_split))
        n_val = max(1, int(n * val_split))
        
        splits['train'].extend([(cls_name, f) for f in files_shuffled[:n_train]])
        splits['val'].extend([(cls_name, f) for f in files_shuffled[n_train:n_train + n_val]])
        splits['test'].extend([(cls_name, f) for f in files_shuffled[n_train + n_val:]])

    # Create destination directories
    previews_dir = os.path.join(processed_dir, "previews")
    os.makedirs(previews_dir, exist_ok=True)

    counts = {}
    for split_name, items in splits.items():
        split_dir = os.path.join(processed_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        counts[split_name] = len(items)

        for idx, (cls_name, file_path) in enumerate(items):
            base_name = f"{cls_name}_{os.path.splitext(os.path.basename(file_path))[0]}"
            out_file = os.path.join(split_dir, f"{base_name}.npz")

            ir_single, ir_multi, rgb = process_sentinel2_tif(file_path)

            np.savez_compressed(
                out_file,
                ir_single=ir_single,
                ir_multi=ir_multi,
                rgb=rgb,
                class_name=cls_name,
                source_file=os.path.basename(file_path)
            )

            # Export first 10 previews for visual inspection
            if split_name == 'train' and idx < 10:
                # Save side-by-side preview: IR (NIR), Multi-IR, True Color RGB
                nir_img = (ir_single[0] * 255).astype(np.uint8)
                multi_img = (np.transpose(ir_multi, (1, 2, 0)) * 255).astype(np.uint8)
                rgb_img = (np.transpose(rgb, (1, 2, 0)) * 255).astype(np.uint8)
                
                # Combine horizontally: NIR (grayscale converted to RGB), Multi-IR, RGB
                nir_rgb = np.stack([nir_img]*3, axis=-1)
                combined = np.concatenate([nir_rgb, multi_img, rgb_img], axis=1)
                preview_path = os.path.join(previews_dir, f"{base_name}_preview.png")
                Image.fromarray(combined).resize((192 * 2, 64 * 2), Image.NEAREST).save(preview_path)

    summary_file = os.path.join(processed_dir, "dataset_summary.json")
    with open(summary_file, 'w') as f:
        json.dump({
            'classes': list(class_to_files.keys()),
            'counts': counts,
            'total_samples': sum(counts.values()),
            'bands': BAND_INDICES
        }, f, indent=2)

    print(f"[Dataset] Processed dataset summary: {counts} (Total: {sum(counts.values())})")
    print(f"[Dataset] Summary saved to {summary_file}")
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and preprocess Sentinel-2 satellite dataset")
    parser.add_argument("--raw_dir", type=str, default="data/raw/eurosat100")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    args = parser.parse_args()

    download_eurosat100(args.raw_dir)
    prepare_dataset(args.raw_dir, args.processed_dir)
