"""
Generates multi-view satellite infrared datasets from aerial scenes
to provide instant plug-and-play demonstrations of multi-view reconstruction.
"""

import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.models.multiview import generate_synthetic_multiview_set

SCENES = [
    {
        "id": "lake_island",
        "name": "Lake Island & Town (Multi-Angle Passes)",
        "source": "data/custom/lake_island_ir.jpg",
        "views": 4,
        "tilt": 0.09
    },
    {
        "id": "archipelago",
        "name": "Archipelago Lake & Peninsulas (Orbital Passes)",
        "source": "data/custom/archipelago_lake_ir.jpg",
        "views": 4,
        "tilt": 0.08
    },
    {
        "id": "city_river",
        "name": "Industrial City & River (Multi-Sensor Angles)",
        "source": "data/custom/city_river_ir.jpg",
        "views": 4,
        "tilt": 0.07
    }
]

def main():
    print("[*] Generating multi-view Earth Observation IR test sets...")
    for scene in SCENES:
        src = scene["source"]
        if not os.path.exists(src):
            print(f"[-] Source {src} does not exist, skipping.")
            continue

        base_img = np.array(Image.open(src).convert("L"), dtype=np.float32) / 255.0
        out_dir = os.path.join("data", "multiview", scene["id"])
        os.makedirs(out_dir, exist_ok=True)

        views = generate_synthetic_multiview_set(
            base_ir=base_img,
            num_views=scene["views"],
            perspective_tilt=scene["tilt"],
            noise_std=0.012
        )

        labels = ["view_1_nadir.jpg", "view_2_oblique_east.jpg", "view_3_oblique_west.jpg", "view_4_off_nadir_north.jpg"]

        print(f"  [+] Scene: {scene['name']} -> {len(views)} viewpoints")
        for idx, v in enumerate(views):
            filename = labels[idx] if idx < len(labels) else f"view_{idx+1}.jpg"
            save_path = os.path.join(out_dir, filename)
            u8 = (np.clip(v, 0.0, 1.0) * 255).astype(np.uint8)
            Image.fromarray(u8).save(save_path, quality=95)
            print(f"      Saved: {save_path}")

    print("[OK] All multi-view test datasets created successfully.")

if __name__ == "__main__":
    main()
