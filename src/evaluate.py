"""
Evaluation Pipeline for Satellite IR to RGB Image Translation.
Computes PSNR, SSIM, MAE, RMSE, and Remote Sensing Spectral Angle Mapper (SAM).
Generates class-by-class benchmark reports and comparative visual mosaics.
"""

import os
import sys
import json
import argparse
import yaml
from typing import Dict, List, Tuple

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from src.models.unet import SatelliteUNet
from src.models.losses import SSIM, compute_psnr
from src.data.dataset import create_dataloaders


def compute_spectral_angle_mapper(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Computes Spectral Angle Mapper (SAM) in degrees.
    SAM measures the spectral angle between predicted and target pixel vectors,
    independent of illumination intensity. Lower is better.
    """
    # pred, target: (B, C, H, W)
    dot = torch.sum(pred * target, dim=1)  # (B, H, W)
    norm_p = torch.norm(pred, dim=1) + eps
    norm_t = torch.norm(target, dim=1) + eps
    cos_theta = torch.clamp(dot / (norm_p * norm_t), -1.0, 1.0)
    angle_rad = torch.acos(cos_theta)
    angle_deg = torch.rad2deg(angle_rad)
    return angle_deg.mean().item()


def evaluate(args):
    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    input_mode = ckpt.get('input_mode', 'single')
    in_channels = ckpt.get('in_channels', 1 if input_mode == 'single' else 3)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Input Mode: {input_mode} ({in_channels} channels)")

    model = SatelliteUNet(in_channels=in_channels, out_channels=3)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Load test dataset
    _, _, test_loader = create_dataloaders(
        processed_dir=args.data_dir,
        input_mode=input_mode,
        batch_size=1
    )
    print(f"Test samples count: {len(test_loader)}")

    ssim_metric = SSIM(channels=3).to(device)

    all_psnr = []
    all_ssim = []
    all_mae = []
    all_rmse = []
    all_sam = []
    class_metrics: Dict[str, Dict[str, List[float]]] = {}

    samples_for_viz = []

    with torch.no_grad():
        for ir, rgb, meta in test_loader:
            ir_dev = ir.to(device)
            rgb_dev = rgb.to(device)

            pred = model(ir_dev)
            pred = torch.clamp(pred, 0.0, 1.0)

            cls_name = meta['class_name'][0]
            if cls_name not in class_metrics:
                class_metrics[cls_name] = {'psnr': [], 'ssim': [], 'mae': [], 'rmse': [], 'sam': []}

            # Metrics
            psnr_val = compute_psnr(pred, rgb_dev)
            ssim_val = ssim_metric(pred, rgb_dev).item()
            mae_val = F.l1_loss(pred, rgb_dev).item()
            rmse_val = torch.sqrt(F.mse_loss(pred, rgb_dev)).item()
            sam_val = compute_spectral_angle_mapper(pred, rgb_dev)

            all_psnr.append(psnr_val)
            all_ssim.append(ssim_val)
            all_mae.append(mae_val)
            all_rmse.append(rmse_val)
            all_sam.append(sam_val)

            class_metrics[cls_name]['psnr'].append(psnr_val)
            class_metrics[cls_name]['ssim'].append(ssim_val)
            class_metrics[cls_name]['mae'].append(mae_val)
            class_metrics[cls_name]['rmse'].append(rmse_val)
            class_metrics[cls_name]['sam'].append(sam_val)

            samples_for_viz.append({
                'ir': ir[0].cpu().numpy(),
                'rgb': rgb[0].cpu().numpy(),
                'pred': pred[0].cpu().numpy(),
                'class': cls_name,
                'psnr': psnr_val,
                'ssim': ssim_val
            })

    # Summary Statistics
    summary = {
        'total_samples': len(all_psnr),
        'mean_psnr_db': float(np.mean(all_psnr)),
        'std_psnr_db': float(np.std(all_psnr)),
        'mean_ssim': float(np.mean(all_ssim)),
        'std_ssim': float(np.std(all_ssim)),
        'mean_mae': float(np.mean(all_mae)),
        'mean_rmse': float(np.mean(all_rmse)),
        'mean_sam_deg': float(np.mean(all_sam)),
        'by_class': {}
    }

    print("\n" + "=" * 65)
    print(" SATELLITE IR-TO-RGB MODEL EVALUATION REPORT")
    print("=" * 65)
    print(f" Mean PSNR : {summary['mean_psnr_db']:.2f} ± {summary['std_psnr_db']:.2f} dB")
    print(f" Mean SSIM : {summary['mean_ssim']:.4f} ± {summary['std_ssim']:.4f}")
    print(f" Mean MAE  : {summary['mean_mae']:.4f}")
    print(f" Mean RMSE : {summary['mean_rmse']:.4f}")
    print(f" Mean SAM  : {summary['mean_sam_deg']:.2f}° (Spectral Angle Mapper)")
    print("-" * 65)
    print(f" {'Terrain Class':<22} | {'PSNR (dB)':<10} | {'SSIM':<8} | {'SAM (°)':<8}")
    print("-" * 65)

    for cls_name, vals in sorted(class_metrics.items()):
        cls_psnr = np.mean(vals['psnr'])
        cls_ssim = np.mean(vals['ssim'])
        cls_sam = np.mean(vals['sam'])
        summary['by_class'][cls_name] = {
            'psnr': float(cls_psnr),
            'ssim': float(cls_ssim),
            'mae': float(np.mean(vals['mae'])),
            'rmse': float(np.mean(vals['rmse'])),
            'sam': float(cls_sam),
            'count': len(vals['psnr'])
        }
        print(f" {cls_name:<22} | {cls_psnr:<10.2f} | {cls_ssim:<8.4f} | {cls_sam:<8.2f}")
    print("=" * 65)

    # Save summary json
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Detailed metrics report saved to: {args.output_json}")

    # Generate High-Resolution Visual Mosaic
    num_display = min(8, len(samples_for_viz))
    fig, axes = plt.subplots(num_display, 4, figsize=(14, 3.2 * num_display))
    if num_display == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(num_display):
        s = samples_for_viz[i]
        ir_arr = s['ir']
        rgb_arr = np.transpose(s['rgb'], (1, 2, 0))
        pred_arr = np.transpose(s['pred'], (1, 2, 0))

        if ir_arr.shape[0] == 1:
            ir_disp = np.repeat(np.transpose(ir_arr, (1, 2, 0)), 3, axis=-1)
        else:
            ir_disp = np.transpose(ir_arr, (1, 2, 0))

        err = np.abs(pred_arr - rgb_arr).mean(axis=-1)

        axes[i, 0].imshow(np.clip(ir_disp, 0, 1))
        axes[i, 0].set_title(f"Input IR ({s['class']})", fontsize=11, fontweight='bold')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(np.clip(rgb_arr, 0, 1))
        axes[i, 1].set_title("Ground Truth RGB", fontsize=11, fontweight='bold')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(np.clip(pred_arr, 0, 1))
        axes[i, 2].set_title(f"AI Predicted RGB\nPSNR: {s['psnr']:.1f}dB, SSIM: {s['ssim']:.3f}", fontsize=11, fontweight='bold', color='#1b5e20')
        axes[i, 2].axis('off')

        im_err = axes[i, 3].imshow(err, cmap='magma', vmin=0, vmax=0.3)
        axes[i, 3].set_title("Pixel Absolute Error", fontsize=11, fontweight='bold', color='#b71c1c')
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.savefig(args.output_img, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Visual test evaluation mosaic saved to: {args.output_img}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Satellite IR to RGB Model")
    parser.add_argument("--checkpoint", type=str, default="runs/checkpoints/best_model_single.pth")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_json", type=str, default="runs/test_metrics.json")
    parser.add_argument("--output_img", type=str, default="runs/test_evaluation_mosaic.png")
    args = parser.parse_args()

    evaluate(args)
