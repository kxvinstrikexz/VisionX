"""
Inference CLI and Module for Satellite Infrared to RGB Image Translation.
Translates individual IR satellite images (TIFF, PNG, JPG) or entire directories into True Color RGB.
"""

import os
import sys
import glob
import argparse
from typing import Union, Tuple, Optional
import cv2

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from PIL import Image
import tifffile

from src.models.unet import SatelliteUNet
from src.data.download_dataset import BAND_INDICES, DEFAULT_NORM_IR


class SatelliteIRPredictor:
    """Predictor class for IR to RGB satellite image translation."""
    def __init__(self, checkpoint_path: str = "runs/checkpoints/best_model_single.pth", device: str = "auto"):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.input_mode = ckpt.get("input_mode", "single")
        self.in_channels = ckpt.get("in_channels", 1 if self.input_mode == "single" else 3)

        self.model = SatelliteUNet(in_channels=self.in_channels, out_channels=3)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def preprocess_image(self, img_path: str) -> Tuple[torch.Tensor, np.ndarray]:
        """Loads and pre-processes an input IR image into model tensor format."""
        ext = os.path.splitext(img_path)[1].lower()

        if ext in ['.tif', '.tiff']:
            arr = tifffile.imread(img_path).astype(np.float32)
            if arr.ndim == 3 and arr.shape[2] >= 13:
                # 13-band Sentinel-2 TIFF
                if self.in_channels == 1:
                    raw_ir = arr[:, :, BAND_INDICES['B08_NIR']]
                    norm_ir = np.clip(raw_ir / DEFAULT_NORM_IR, 0.0, 1.0)
                    tensor = torch.from_numpy(norm_ir).unsqueeze(0).unsqueeze(0)
                    vis_ir = norm_ir
                else:
                    nir = arr[:, :, BAND_INDICES['B08_NIR']]
                    swir1 = arr[:, :, BAND_INDICES['B11_SWIR1']]
                    swir2 = arr[:, :, BAND_INDICES['B12_SWIR2']]
                    raw_ir = np.stack([nir, swir1, swir2], axis=-1)
                    norm_ir = np.clip(raw_ir / DEFAULT_NORM_IR, 0.0, 1.0)
                    tensor = torch.from_numpy(np.transpose(norm_ir, (2, 0, 1))).unsqueeze(0)
                    vis_ir = norm_ir
            elif arr.ndim == 2:
                # Single band grayscale TIFF
                norm_ir = np.clip(arr / (arr.max() + 1e-6 if arr.max() > 1.0 else 1.0), 0.0, 1.0)
                if self.in_channels == 1:
                    tensor = torch.from_numpy(norm_ir).unsqueeze(0).unsqueeze(0)
                else:
                    tensor = torch.from_numpy(np.stack([norm_ir]*3, axis=0)).unsqueeze(0)
                vis_ir = norm_ir
            else:
                norm_ir = np.clip(arr / (arr.max() + 1e-6 if arr.max() > 1.0 else 1.0), 0.0, 1.0)
                if self.in_channels == 1 and norm_ir.shape[-1] >= 1:
                    tensor = torch.from_numpy(norm_ir[:, :, 0]).unsqueeze(0).unsqueeze(0)
                    vis_ir = norm_ir[:, :, 0]
                else:
                    tensor = torch.from_numpy(np.transpose(norm_ir[:, :, :3], (2, 0, 1))).unsqueeze(0)
                    vis_ir = norm_ir[:, :, :3]
        elif ext == '.npz':
            data = np.load(img_path)
            key = 'ir_single' if self.in_channels == 1 else 'ir_multi'
            raw = data[key]
            tensor = torch.from_numpy(raw).unsqueeze(0)
            vis_ir = np.transpose(raw, (1, 2, 0)) if raw.shape[0] == 3 else raw[0]
        else:
            # Standard PNG/JPG image
            pil_img = Image.open(img_path)
            if self.in_channels == 1:
                pil_gray = pil_img.convert("L")
                arr = np.array(pil_gray, dtype=np.float32) / 255.0
                tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
                vis_ir = arr
            else:
                pil_rgb = pil_img.convert("RGB")
                arr = np.array(pil_rgb, dtype=np.float32) / 255.0
                tensor = torch.from_numpy(np.transpose(arr, (2, 0, 1))).unsqueeze(0)
                vis_ir = arr

        return tensor.to(self.device, dtype=torch.float32), vis_ir

    def predict(
        self,
        img_path: str,
        photographic: bool = False,
        reference_path: Optional[str] = None,
        saturation: float = 1.25,
        sharpen: float = 1.3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Translates an IR image into True Color RGB numpy array [0, 1]."""
        tensor, vis_ir = self.preprocess_image(img_path)
        with torch.no_grad():
            pred = self.model(tensor)
            pred = torch.clamp(pred, 0.0, 1.0)
            rgb_arr = pred.squeeze(0).cpu().numpy()
            rgb_arr = np.transpose(rgb_arr, (1, 2, 0))  # (H, W, 3)

        if photographic:
            from src.models.enhancement import reconstruct_photographic_rgb
            ref_rgb = None
            if reference_path and os.path.exists(reference_path):
                ref_img = Image.open(reference_path).convert("RGB")
                ref_rgb = np.array(ref_img, dtype=np.float32) / 255.0

            norm_ir = vis_ir if vis_ir.ndim == 2 else vis_ir[:, :, 0]
            rgb_arr = reconstruct_photographic_rgb(
                input_ir=norm_ir,
                reference_rgb=ref_rgb,
                unet_pred_rgb=rgb_arr,
                saturation=saturation,
                detail_strength=sharpen
            )

        return rgb_arr, vis_ir

    def translate_and_save(
        self,
        input_path: str,
        output_path: str,
        save_comparison: bool = True,
        photographic: bool = False,
        reference_path: Optional[str] = None,
        saturation: float = 1.25,
        sharpen: float = 1.3
    ):
        """Translates IR image and saves the resulting RGB and optional comparison."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        pred_rgb, vis_ir = self.predict(
            input_path,
            photographic=photographic,
            reference_path=reference_path,
            saturation=saturation,
            sharpen=sharpen
        )

        # Save output RGB
        rgb_uint8 = (np.clip(pred_rgb, 0.0, 1.0) * 255).astype(np.uint8)
        Image.fromarray(rgb_uint8).save(output_path)
        print(f"Saved True-Color RGB: {output_path}")

        if save_comparison:
            base, ext = os.path.splitext(output_path)
            comp_path = f"{base}_comparison.png"
            if vis_ir.ndim == 2:
                vis_ir_disp = np.stack([(np.clip(vis_ir, 0, 1) * 255).astype(np.uint8)] * 3, axis=-1)
            else:
                vis_ir_disp = (np.clip(vis_ir, 0, 1) * 255).astype(np.uint8)

            if vis_ir_disp.shape[:2] != rgb_uint8.shape[:2]:
                vis_ir_disp = cv2.resize(vis_ir_disp, (rgb_uint8.shape[1], rgb_uint8.shape[0]))

            side_by_side = np.concatenate([vis_ir_disp, rgb_uint8], axis=1)
            Image.fromarray(side_by_side).save(comp_path)
            print(f"Saved Side-by-Side Comparison: {comp_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict RGB True Color from Infrared Satellite Image")
    parser.add_argument("--input", type=str, required=True, help="Path to input IR image (.tif, .png, .jpg, .npz)")
    parser.add_argument("--output", type=str, default="runs/predictions/output_rgb.png", help="Path to output RGB image")
    parser.add_argument("--checkpoint", type=str, default="runs/checkpoints/best_model_single.pth")
    parser.add_argument("--photographic", action="store_true", help="Apply high-fidelity photographic natural-color enhancement")
    parser.add_argument("--reference", type=str, default="runs/reference_lakeside.png", help="Path to reference natural-color image")
    parser.add_argument("--saturation", type=float, default=1.25, help="Color saturation multiplier")
    parser.add_argument("--sharpen", type=float, default=1.3, help="Detail sharpening strength")
    parser.add_argument("--no_comparison", action="store_true", help="Do not save side-by-side comparison")
    args = parser.parse_args()

    predictor = SatelliteIRPredictor(checkpoint_path=args.checkpoint)
    predictor.translate_and_save(
        args.input,
        args.output,
        save_comparison=not args.no_comparison,
        photographic=args.photographic,
        reference_path=args.reference if args.photographic else None,
        saturation=args.saturation,
        sharpen=args.sharpen
    )

