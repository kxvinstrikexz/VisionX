# 🛰️ visionX: Satellite Infrared-to-RGB Image Translation AI

An end-to-end deep learning system for **cross-spectral satellite image translation**. Converts non-visible **Infrared (NIR / SWIR)** satellite imagery into natural **True-Color RGB** images with high spatial and radiometric fidelity, trained on real multispectral imagery from the **Copernicus Sentinel-2** satellite constellation.

---

## 📖 Scientific Background

### The Cross-Spectral Translation Challenge
Optical Earth Observation (EO) satellites like **Sentinel-2** and **Landsat-8** capture reflected solar radiation across narrow spectral bands far outside the human visible spectrum:
* **Near-Infrared (NIR, Band 8 ~842nm)**: Chlorophyll within healthy plant tissue strongly scatters and reflects NIR light (the "red edge" effect). Conversely, open water bodies strongly absorb NIR radiation.
* **Short-Wave Infrared (SWIR, Bands 11 & 12 ~1610nm & ~2190nm)**: Highly sensitive to soil moisture, mineral geology, and atmospheric penetration through haze and smoke.
* **True-Color RGB (Bands 4, 3, 2 ~665nm, ~560nm, ~490nm)**: Standard natural color imagery visible to human vision.

Converting an Infrared satellite image to True-Color RGB requires the neural network to learn **geophysical radiometric mappings**:
1. Recognizing vegetation textures from high NIR reflectance and translating them to lush natural green tones ($R \approx 0.12, G \approx 0.28, B \approx 0.14$).
2. Recognizing water bodies from near-zero NIR reflectance and translating them to deep oceanic blues.
3. Recognizing urban and industrial concrete from moderate, uniform reflectance and translating them to neutral asphalt gray and white rooftops.

---

## 🏗️ Neural Architecture

The project implements **SatelliteUNet**, a tailored U-Net architecture optimized for satellite remote sensing translation:

```
[ Input: Infrared Image (1-ch NIR or 3-ch Multi-IR) ]
                       │
             ┌─────────▼─────────┐
             │ DoubleConv (32)   ├────────────────────────┐ (Skip Connection)
             └─────────┬─────────┘                        │
                       │ MaxPool                          │
             ┌─────────▼─────────┐                        │
             │ DoubleConv (64)   ├──────────────┐         │
             └─────────┬─────────┘              │         │
                       │ MaxPool                │         │
             ┌─────────▼─────────┐              │         │
             │ DoubleConv (128)  ├────┐         │         │
             └─────────┬─────────┘    │         │         │
                       │ MaxPool      │         │         │
             ┌─────────▼─────────┐    │         │         │
             │ Bottleneck (256)  │    │         │         │
             └─────────┬─────────┘    │         │         │
                       │ ConvTranspose│         │         │
             ┌─────────▼─────────┐    │         │         │
             │ UpBlock (128)     │◄───┘         │         │
             └─────────┬─────────┘              │         │
                       │ ConvTranspose          │         │
             ┌─────────▼─────────┐              │         │
             │ UpBlock (64)      │◄─────────────┘         │
             └─────────┬─────────┘                        │
                       │ ConvTranspose                    │
             ┌─────────▼─────────┐                        │
             │ UpBlock (32)      │◄───────────────────────┘
             └─────────┬─────────┘
                       │ Conv 1x1 + Sigmoid
                       ▼
[ Output: True-Color RGB Image (3-ch [0, 1] Normalized Reflectance) ]
```

### Key Architectural Highlights:
* **High-Frequency Skip Connections**: Directly passes spatial details (rivers, highways, parcel borders, building contours) from the encoder to the decoder to prevent spatial blurring.
* **Residual Conv Blocks**: LeakyReLU activations with batch normalization and identity shortcuts ensure smooth gradient flow and training stability.
* **Composite Radiometric & Structural Loss**:
  $$\mathcal{L}_{total} = \lambda_{L1} \mathcal{L}_{L1} + \lambda_{SSIM} (1 - \text{SSIM}) + \lambda_{grad} \mathcal{L}_{grad}$$
  - $\mathcal{L}_{L1}$: Preserves absolute spectral radiance and color values.
  - $\mathcal{L}_{SSIM}$: Differentiable Structural Similarity Index ensures sharp ground textures and micro-features.
  - $\mathcal{L}_{grad}$: Sobel edge gradient difference enforces crisp geographic boundaries.

---

## 📊 Benchmark & Quantitative Results

Evaluated across test scenes encompassing **10 diverse Earth land-cover categories** from the Sentinel-2 EuroSAT benchmark:

| Metric | Measured Value | Standard Deviation |
| :--- | :--- | :--- |
| **Mean PSNR (Peak Signal-to-Noise Ratio)** | **16.27 dB** | $\pm 3.83$ dB |
| **Mean SSIM (Structural Similarity Index)** | **0.6162** | $\pm 0.1611$ |
| **Mean MAE (Mean Absolute Error)** | **0.1514** | $\pm 0.038$ |
| **Mean RMSE (Root Mean Squared Error)** | **0.1672** | $\pm 0.041$ |
| **Spectral Angle Mapper (SAM)** | **7.04°** | Spectral Vector Fidelity |

### Category-Specific Breakdown:
* **Sea & Lakes (`SeaLake`)**: **24.41 dB PSNR**, **0.9358 SSIM** (High fidelity water identification)
* **Residential Areas (`Residential`)**: **17.88 dB PSNR**, **3.88° SAM** (Accurate roof and asphalt mapping)
* **Highways & Transport (`Highway`)**: **16.58 dB PSNR**, **0.5850 SSIM** (Crisp linear road recovery)
* **Pastures & Grasslands (`Pasture`)**: **14.52 dB PSNR**, **0.7369 SSIM** (Realistic chlorophyll balance)

---

## 📁 Repository Structure

```
visionX/
├── app/
│   └── app.py                     # Interactive Streamlit Web Dashboard
├── config/
│   └── config.yaml                # Centralized hyperparameters & band configuration
├── data/
│   ├── raw/eurosat100/            # Raw 13-band Sentinel-2 multispectral TIFF files
│   └── processed/                 # Train / Val / Test splits (.npz archives + PNG previews)
├── runs/
│   ├── checkpoints/               # Trained model weights (best_model_single.pth)
│   ├── predictions/               # Sample output predictions
│   ├── samples/                   # Validation epoch comparison grids
│   ├── test_metrics.json          # Quantitative evaluation statistics
│   ├── all_images_metrics.json    # Accuracy and radiometric metrics for all 100 images
│   ├── all_images_accuracy.png    # Comprehensive accuracy breakdown across all images & classes
│   ├── training_curves.png        # Convergence plots (Loss, Accuracy %, PSNR, SSIM)
│   └── test_evaluation_mosaic.png # Test set multi-class comparative mosaic
├── src/
│   ├── data/
│   │   ├── download_dataset.py    # Automated dataset downloader & band extractor
│   │   └── dataset.py             # PyTorch Dataset, DataLoaders, and augmentations
│   ├── models/
│   │   ├── unet.py                # SatelliteUNet deep neural network architecture
│   │   ├── discriminator.py       # PatchGAN discriminator (cGAN mode)
│   │   └── losses.py              # Composite L1, SSIM, Gradient losses and PSNR metrics
│   ├── train.py                   # Complete training pipeline with dynamic scheduler
│   ├── evaluate.py                # Test set evaluation and comparative reporting
│   └── predict.py                 # CLI inference tool for custom IR images
├── requirements.txt               # Python package dependencies
└── README.md                      # Complete documentation
```

---

## 🚀 Quickstart Guide

### 1. Environment Installation
Install the required packages using pip:
```bash
pip install -r requirements.txt
```

### 2. Dataset Download & Preprocessing
The automated downloader fetches the Sentinel-2 multispectral benchmark dataset, extracts bands (B08 NIR, B04 Red, B03 Green, B02 Blue), and creates stratified splits:
```bash
python src/data/download_dataset.py
```

### 3. Training the Model
Train the `SatelliteUNet` on CPU or GPU with live metric logging and checkpointing:
```bash
python src/train.py --epochs 35 --batch_size 16 --lr 0.0003
```
* Best model saved to: `runs/checkpoints/best_model_single.pth`
* Training curves plotted to: `runs/training_curves.png`
* Validation grids saved to: `runs/samples/`

### 4. Evaluating on the Test Set
Compute PSNR, SSIM, MAE, and SAM across all 10 terrain classes:
```bash
python src/evaluate.py --checkpoint runs/checkpoints/best_model_single.pth
```
Outputs qualitative comparison mosaics to `runs/test_evaluation_mosaic.png`.

### 5. Running Inference on Any Satellite Image
Convert any Infrared satellite image (raw 13-band Sentinel-2 `.tif`, grayscale `.png`/`.jpg`, or `.npz`):
```bash
python src/predict.py --input data/processed/test/Forest_Forest_321.npz --output runs/predictions/forest_rgb.png
```
Or on a raw Sentinel-2 multispectral TIFF:
```bash
python src/predict.py --input data/raw/eurosat100/ds/images/remote_sensing/otherDatasets/sentinel_2/tif/River/River_1690.tif --output runs/predictions/river_rgb.png
```

### 6. Multi-View Satellite IR Image Reconstruction
Convert multiple satellite or aerial images captured from **different viewpoints, off-nadir angles, or orbital passes** into a **single seamless, high-resolution reconstructed IR image**, and optionally translate directly to True-Color RGB:
```bash
# Reconstruct from pre-packaged multi-view dataset:
python src/multiview.py --dataset lake_island --fusion feathered_mosaic --translate_rgb

# Reconstruct from arbitrary image files:
python src/multiview.py --inputs path/to/view1.jpg path/to/view2.jpg path/to/view3.jpg --output runs/reconstructed_multiview_ir.png --translate_rgb --rgb_output runs/reconstructed_multiview_rgb.png
```
Supported fusion modes:
* `feathered_mosaic`: Expands canvas to union of all viewpoints with distance-transform feathering to eliminate seamlines.
* `super_resolution_fusion`: Co-registers views to reference grid with sub-pixel multi-frame median stacking to boost SNR and suppress noise.
* `median`: Multi-frame median filtering to remove transient artifacts, glint, and cloud shadows.
* `average`: Exposure-weighted radiometric average.

### 7. Launching the Interactive Web Dashboard
Run the Streamlit application:
```bash
streamlit run app/app.py
```
Open **`http://localhost:8501`** in your browser to:
* **🛰️ Multi-View IR Reconstruction**: Interactively fuse multiple viewpoint passes, adjust homography/fusion algorithms, and generate True-Color RGB translations.
* **🖼️ Single-Image IR-to-RGB Translation**: Upload custom IR imagery (.tif, .png, .jpg, .npz) to synthesize photographic or calibrated AI True-Color RGB imagery.
* **📊 Dynamic Image Analytics**: Real-time radiometric metrics, spectral reflectance histograms, horizontal transect profiles, and Sobel edge frequency analysis computed dynamically from your active satellite imagery.

