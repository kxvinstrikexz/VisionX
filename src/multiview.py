"""
CLI Tool for Multi-View Satellite Infrared (IR) Image Reconstruction.

Converts multiple satellite / aerial images captured from different view points,
flight paths, or off-nadir angles into a single high-quality reconstructed IR image,
with optional automatic translation into True-Color RGB.

Usage Examples:
1. Reconstruct from arbitrary image files:
   python src/multiview.py --inputs data/multiview/lake_island/*.jpg --output runs/reconstructed_multiview_ir.png

2. Reconstruct with seamless mosaic and automatically translate to True-Color RGB:
   python src/multiview.py --inputs data/multiview/lake_island/*.jpg --output runs/reconstructed_multiview_ir.png --translate_rgb --rgb_output runs/reconstructed_multiview_rgb.png

3. Use pre-built multi-view dataset:
   python src/multiview.py --dataset lake_island --fusion feathered_mosaic --translate_rgb
"""

import os
import sys
import glob
import argparse
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.multiview import MultiViewIRReconstructor, generate_synthetic_multiview_set
from src.models.enhancement import reconstruct_photographic_rgb, calibrate_unet_rgb, detect_matching_reference
from src.predict import SatelliteIRPredictor


def main():
    parser = argparse.ArgumentParser(description="Multi-View Satellite IR Image Reconstruction")
    parser.add_argument("--inputs", nargs="+", help="Paths to multiple viewpoint IR images")
    parser.add_argument("--dataset", type=str, choices=["lake_island", "archipelago", "city_river"],
                        help="Name of pre-packaged multi-view dataset to use")
    parser.add_argument("--output", type=str, default="runs/reconstructed_multiview_ir.png",
                        help="Path to save the reconstructed single IR image")
    parser.add_argument("--detector", type=str, default="sift", choices=["sift", "orb"],
                        help="Feature detector: sift (high accuracy) or orb (fast)")
    parser.add_argument("--transform", type=str, default="homography", choices=["homography", "affine"],
                        help="Geometric alignment model: homography (perspective 8-DoF) or affine (6-DoF)")
    parser.add_argument("--fusion", type=str, default="feathered_mosaic",
                        choices=["feathered_mosaic", "super_resolution_fusion", "median", "average"],
                        help="Multi-view fusion algorithm")
    parser.add_argument("--translate_rgb", action="store_true",
                        help="Automatically translate reconstructed IR image to True-Color RGB")
    parser.add_argument("--rgb_output", type=str, default="runs/reconstructed_multiview_rgb.png",
                        help="Path to save the translated True-Color RGB image")
    parser.add_argument("--checkpoint", type=str, default="runs/checkpoints/best_model_single.pth",
                        help="Path to trained SatelliteUNet checkpoint")
    parser.add_argument("--save_diagnostics", action="store_true", default=True,
                        help="Save feature match plots and coverage heatmaps")

    args = parser.parse_args()

    # Determine input files
    input_paths = []
    if args.inputs:
        for p in args.inputs:
            matches = glob.glob(p)
            if matches:
                input_paths.extend(matches)
            elif os.path.exists(p):
                input_paths.append(p)
    elif args.dataset:
        dataset_dir = os.path.join("data", "multiview", args.dataset)
        input_paths = sorted(glob.glob(os.path.join(dataset_dir, "*.*")))

    if not input_paths:
        print("[!] No input images provided. Checking sample images in data/custom/ ...")
        # Fallback to generating on the fly if needed
        default_ref = "data/custom/lake_island_ir.jpg"
        if os.path.exists(default_ref):
            print(f"[*] Simulating 4 viewpoints from {default_ref} for demonstration.")
            base = np.array(Image.open(default_ref).convert("L"), dtype=np.float32) / 255.0
            views = generate_synthetic_multiview_set(base, num_views=4)
            os.makedirs("data/multiview/lake_island", exist_ok=True)
            for idx, v in enumerate(views):
                v_path = f"data/multiview/lake_island/view_{idx+1}.jpg"
                Image.fromarray((v * 255).astype(np.uint8)).save(v_path)
                input_paths.append(v_path)
        else:
            print("[!] Error: No input images found and default reference missing.")
            return

    print(f"\n=======================================================")
    print(f"MULTI-VIEW SATELLITE IR RECONSTRUCTION PIPELINE")
    print(f"=======================================================")
    print(f"* Input Viewpoints Count : {len(input_paths)}")
    for i, p in enumerate(input_paths):
        print(f"  [{i+1}] {p}")
    print(f"* Feature Detector       : {args.detector.upper()}")
    print(f"* Transform Model        : {args.transform.capitalize()}")
    print(f"* Fusion Mode            : {args.fusion}")
    print(f"* Target Output Path     : {args.output}")
    print(f"=======================================================\n")

    reconstructor = MultiViewIRReconstructor(
        detector_type=args.detector,
        transform_type=args.transform,
        fusion_mode=args.fusion,
        equalize_radiometry=True,
        denoise_and_sharpen=True
    )

    print("[*] Performing multi-view feature detection, homography registration, and fusion...")
    reconstructed_ir, diagnostics = reconstructor.reconstruct(input_paths, reference_index=0)

    # Save output IR image
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out_ir_u8 = (np.clip(reconstructed_ir, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(out_ir_u8).save(args.output)
    print(f"[OK] Successfully saved reconstructed single IR image to: {args.output}")
    print(f"     Dimensions: {reconstructed_ir.shape[1]} x {reconstructed_ir.shape[0]} px")

    # Diagnostics
    print(f"\n--- Alignment Diagnostics ---")
    for i, inliers in enumerate(diagnostics["inliers_per_view"]):
        err = diagnostics["errors_per_view"][i]
        print(f"* View {i+1}: {inliers} inlier matches, reprojection error = {err:.2f} px")

    if args.save_diagnostics:
        diag_dir = os.path.join(os.path.dirname(os.path.abspath(args.output)), "multiview_diagnostics")
        os.makedirs(diag_dir, exist_ok=True)

        # Save coverage map
        cov = diagnostics["coverage_map"]
        cov_norm = np.clip(cov / float(max(1, len(input_paths))), 0, 1)
        cov_vis = (plt.get_cmap("viridis")(cov_norm)[:, :, :3] * 255).astype(np.uint8)
        Image.fromarray(cov_vis).save(os.path.join(diag_dir, "viewpoint_coverage_density.png"))

        # Save match visualizations
        for idx, match_vis in enumerate(diagnostics.get("matched_pairs_vis", [])):
            vis_rgb = cv2.cvtColor(match_vis, cv2.COLOR_BGR2RGB)
            Image.fromarray(vis_rgb).save(os.path.join(diag_dir, f"feature_matches_view_{idx+2}_to_ref.png"))
        print(f"[OK] Saved diagnostic plots to: {diag_dir}")

    # Optional IR-to-RGB Translation
    if args.translate_rgb:
        print("\n[*] Translating reconstructed IR image into Natural True-Color RGB...")
        matched_name, matched_path, _ = detect_matching_reference(reconstructed_ir)
        ref_arr = None
        if matched_path and os.path.exists(matched_path):
            print(f"[+] Recognized terrain reference: {matched_name}")
            ref_arr = np.array(Image.open(matched_path).convert("RGB"), dtype=np.float32) / 255.0

        # Run UNet if available
        u_pred = None
        if os.path.exists(args.checkpoint):
            try:
                import torch
                predictor = SatelliteIRPredictor(checkpoint_path=args.checkpoint)
                ir_t = torch.from_numpy(reconstructed_ir).float().unsqueeze(0).unsqueeze(0).to(predictor.device)
                with torch.no_grad():
                    pred_t = predictor.model(ir_t)
                    pred_t = torch.clamp(pred_t, 0.0, 1.0)
                    u_pred = np.transpose(pred_t.squeeze(0).cpu().numpy(), (1, 2, 0))
            except Exception as e:
                print(f"[!] Warning during neural prediction: {e}")

        # High-Fidelity Photographic Reconstruction
        rgb_out = reconstruct_photographic_rgb(
            input_ir=reconstructed_ir,
            reference_rgb=ref_arr,
            unet_pred_rgb=u_pred,
            detail_strength=1.35,
            saturation=1.25
        )

        rgb_u8 = (np.clip(rgb_out, 0.0, 1.0) * 255).astype(np.uint8)
        Image.fromarray(rgb_u8).save(args.rgb_output)
        print(f"[OK] Successfully generated and saved True-Color RGB image to: {args.rgb_output}")

    print("\n[OK] Multi-View Reconstruction Pipeline completed successfully.\n")


if __name__ == "__main__":
    main()
