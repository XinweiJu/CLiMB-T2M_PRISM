# 🧭 CLiMB-T2M_PRISM

> **Official source release for T2M-DLPE-clean** — our PRISM-conditioned
> Track2Map entry for the [CLiMB Challenge](https://lnkd.in/eDpE5wAT).

[![Release](https://img.shields.io/github/v/release/XinweiJu/CLiMB-T2M_PRISM?label=Release)](https://github.com/XinweiJu/CLiMB-T2M_PRISM/releases/latest)
![CUDA](https://img.shields.io/badge/CUDA-12.8-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-orange)
![Platform](https://img.shields.io/badge/platform-Linux-blue)

T2M-DLPE-clean replaces Track2Map's generic monocular anchor-depth predictor
with the luminance-conditioned **PRISM DLPE DepthNet**. **CoTracker3 Online**
supplies multi-frame point correspondences. Anchor inverse depth lifts these
correspondences into 3D, PnP-RANSAC estimates relative camera motion, bounded
motion propagation handles short tracking failures, and fixed-lag smoothing
regularizes the completed trajectory.

> [!NOTE]
> Only **PRISM DepthNet** and the **luminance generator** run during
> submission inference. PRISM PoseNet and DexiNed are training-only components.
> The released clean checkpoint lineage excludes the challenge-designated
> EndoMapper sequence from SegCol-derived edge supervision.

---

## 🔗 Quick links

- [Pipeline](#-pipeline)
- [Repository layout](#-repository-layout)
- [Requirements](#️-requirements)
- [Model weights](#-model-weights)
- [Build and run](#-build-and-run)
- [Training](#-training)
- [Method](#️-method-name-and-short-description)
- [Reproducibility](#-reproducibility-notes)
- [Citation](#-citation)

---

## 🔄 Pipeline

```text
BGR video ─┬─> CoTracker3 Online ─> 2D tracks ───────────────┐
           │                                                  │
           └─> IID/SHADES ─> PRISM DLPE inverse depth ─> 3D lift
                                                              │
                                      PnP-RANSAC <─────────────┘
                                           │
                                 safeguards + propagation
                                           │
                                  fixed-lag smoothing
                                           │
                                  trajectory + sparse map
```

---

## 📁 Repository layout

- **`submission/`** — CLiMB container entrypoint and T2M-DLPE-clean inference code.
- **`training/`** — PRISM depth/pose training and leakage-controlled edge generation.
- **`ablations/`** — track, optical-flow, and learned-extra-channel experiments.

Model weights, datasets, generated modalities, evaluation outputs, and the
CoTracker3 checkout are intentionally not committed.

See **`submission/README.md`** for the expected local Docker layout.

---

## ⚙️ Requirements

- Linux with an NVIDIA GPU and NVIDIA Container Toolkit
- Docker with CUDA-compatible driver support
- The released T2M-DLPE-clean PRISM weight bundle
- The official CoTracker3 source and `scaled_online.pth` checkpoint

The challenge image uses a **PyTorch 2.7.1 / CUDA 12.8** runtime.

The inference entrypoint itself receives OpenCV BGR `uint8` frames. PRISM
resizes each anchor to $288\times288$, predicts inverse depth, and bilinearly
restores the result to the original frame size before Track2Map uses it.

---

## 📦 Model weights

Our GitHub Release
**[CLiMB-T2M_PRISM v1.0.0](https://github.com/XinweiJu/CLiMB-T2M_PRISM/releases/tag/v1.0.0)**
contains only the PRISM components trained or selected for this method:

```text
T2M-DLPE-clean-v1.0.0/
├── checkpoints.json
└── weights/
    ├── depth/
    │   ├── encoder.pth
    │   └── depth.pth
    └── shading/
        ├── decompose_encoder.pth
        └── decompose.pth
```

> [!IMPORTANT]
> Verify the downloaded archive against its accompanying **SHA256** file before
> building the image. The manifest also records hashes for every individual
> checkpoint.

### CoTracker3

CoTracker3 is **not redistributed** in our Release. Obtain the code from the
[official CoTracker repository](https://github.com/facebookresearch/co-tracker)
and download its official online checkpoint:

```bash
git clone https://github.com/facebookresearch/co-tracker.git

mkdir -p co-tracker/checkpoints

wget -O co-tracker/checkpoints/scaled_online.pth \
  https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth
```

For this repository's Docker layout, copy or link the checkpoint to:

```text
submission/vendor/co-tracker/scaled_online.pth
```

---

## 🚀 Build and run

After downloading the PRISM Release archive and CoTracker3, arrange the files
as documented in `submission/README.md`, then run:

```bash
docker build -t t2m-dlpe-clean:local submission

docker run --rm --gpus all --network=none \
  -v /absolute/input:/input:ro \
  -v /absolute/output:/output \
  t2m-dlpe-clean:local
```

**📥 Input:** challenge MP4 files.

**📤 Output:** CLiMB COLMAP-style results containing five trajectory/map runs
per sequence.

---

## 🧪 Training

The leakage-controlled PRISM training lineage consists of three stages:

### Stage 1 — Edge supervision

Retrain **DexiNed** on the allowed SegCol split and precompute edge maps.

### Stage 2 — Joint PRISM training

Jointly train PRISM **DepthNet + PoseNet** on real Hyper-Kvasir video with:

- RGB + luminance depth input
- RGB + edge pose input
- edge-SSIM supervision

### Stage 3 — Pose post-finetuning

Freeze the complete depth encoder/decoder and post-fine-tune PoseNet.

> [!NOTE]
> The selected submission uses the **Stage-2 DepthNet only**.
> Stage 3 is retained to reproduce the direct-pose ablations.

Scripts and configuration are under `training/`; track/flow conditioning
experiments are isolated under `ablations/`.

### 🔒 Leakage control

No hidden CLiMB video, trajectory, COLMAP reconstruction, or
hidden-test-derived artifact is used for training, checkpoint selection, or
hyperparameter tuning.

---

## 🏷️ Method name and short description

### **T2M-DLPE-clean**

**CoTracker3 Online correspondences + leakage-controlled PRISM DLPE inverse
depth + depth-assisted PnP + bounded motion propagation + fixed-lag trajectory
smoothing.**

The complete submission-form text and measured EndoMapper short-sequence
wall-clock values are provided in `submission/README.md`.

---

## ✅ Reproducibility notes

- The PRISM weight manifest records model dimensions, auxiliary-channel
  routing, provenance, and SHA256 hashes.
- Model weights, generated modalities, datasets, and evaluation outputs are
  excluded from Git history.
- Local runtime is diagnostic only; official challenge runtime is measured by
  the server.
- The final method report will be added after the clean Table 3 / Table 5
  ablations are complete.

---

## 📚 Citation

If you use this repository or its components, please consider citing the
corresponding works:

- **Track2Map** — Tianyi Song, Sierra Bonilla, Xinwei Ju, et al.  
  [*Track2Map: Online Deformable SLAM with Motion-Aware Pose Optimization in Robotic Surgery*](https://arxiv.org/abs/2607.08408), MICCAI 2026.

- **PRISM** — Xinwei Ju, Rema Daher, Danail Stoyanov, Sophia Bano, and Francisco Vasconcelos.  
  [*Multi-Modal Monocular Endoscopic Depth and Pose Estimation with Edge-Guided Self-Supervision*](https://link.springer.com/article/10.1007/s11548-026-03669-1), International Journal of Computer Assisted Radiology and Surgery, 2026.

- **CoTracker3** — Nikita Karaev, Yuri Makarov, Jianyuan Wang, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht.  
  [*CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos*](https://openaccess.thecvf.com/content/ICCV2025/html/Karaev_CoTracker3_Simpler_and_Better_Point_Tracking_by_Pseudo-Labelling_Real_Videos_ICCV_2025_paper.html), ICCV 2025.

- **Monodepth2** — Clément Godard, Oisin Mac Aodha, Michael Firman, and Gabriel J. Brostow.  
  [*Digging Into Self-Supervised Monocular Depth Estimation*](https://openaccess.thecvf.com/content_ICCV_2019/html/Godard_Digging_Into_Self-Supervised_Monocular_Depth_Estimation_ICCV_2019_paper.html), ICCV 2019.

- **Hyper-Kvasir** — Hanna Borgli, Vajira Thambawita, Pia H. Smedsrud, et al.  
  [*HyperKvasir, a Comprehensive Multi-Class Image and Video Dataset for Gastrointestinal Endoscopy*](https://www.nature.com/articles/s41597-020-00622-y), Scientific Data, 2020.

- **EndoMapper** — Pablo Azagra, Carlos Sostres, Ángel Ferrández, et al.  
  [*EndoMapper Dataset of Complete Calibrated Endoscopy Procedures*](https://arxiv.org/abs/2204.14240), 2022.

- **SegCol** — Xinwei Ju, Rema Daher, Razvan Caramalau, Baoru Huang, Danail Stoyanov, and Francisco Vasconcelos.  
  [*SegCol Challenge: Semantic Segmentation for Tools and Fold Edges in Colonoscopy Data*](https://arxiv.org/abs/2412.16078), EndoVis / MICCAI 2024.
