# 🧭 CLiMB-T2M_PRISM

> **Official source release for T2M-DLPE-G22-clean** — our PRISM-conditioned
> Track2Map entry for the [CLiMB Challenge](https://lnkd.in/eDpE5wAT).

[![Release](https://img.shields.io/github/v/release/XinweiJu/CLiMB-T2M_PRISM?label=Release)](https://github.com/XinweiJu/CLiMB-T2M_PRISM/releases/latest)
![CUDA](https://img.shields.io/badge/CUDA-12.8-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-orange)
![Platform](https://img.shields.io/badge/platform-Linux-blue)

T2M-DLPE-G22-clean replaces Track2Map's generic monocular anchor-depth predictor
with the luminance-conditioned **PRISM DLPE DepthNet**. **CoTracker3 Online**
supplies multi-frame point correspondences. Anchor inverse depth lifts these
correspondences into 3D, PnP-RANSAC estimates relative camera motion, bounded
motion propagation handles short tracking failures, and fixed-lag smoothing
regularizes the completed trajectory. The G22 submission uses a
22-by-22 CoTracker query grid, a 6-pixel PnP-RANSAC threshold, and 11/5-frame
translation/rotation smoothing.

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
- [Results](#-results)
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

- **`submission/`** — CLiMB container entrypoint and T2M-DLPE-G22-clean inference code.
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
docker build -t t2m-dlpe-g22-clean:local submission

docker run --rm --gpus all --network=none \
  -v /absolute/input:/input:ro \
  -v /absolute/output:/output \
  t2m-dlpe-g22-clean:local
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

### **T2M-DLPE-G22-clean**

**CoTracker3 Online correspondences + leakage-controlled PRISM DLPE inverse
depth + depth-assisted PnP + bounded motion propagation + fixed-lag trajectory
smoothing.**

The complete submission-form text and measured EndoMapper short-sequence
wall-clock values are provided in `submission/README.md`.

---

## 📊 Results

ATE is reported in millimetres and rotational error is the challenge's
40-frame relative rotation error in degrees; lower is better. Official hidden
test results are kept separate from local released-data diagnostics.

### Official hidden-test leaderboard

The earlier entries below completed all 32 hidden-test sequences with 100%
tracking coverage. The supplied G22 leaderboard row did not include its TFR,
runtime, submission ID, or displayed rank.

| Submission (ID) | Mean ATE ↓ | RotErr ↓ | s/frame | Rank |
|---|---:|---:|---:|---:|
| **T2M-DLPE-G22-clean (ID to verify)** | 10.591 | **3.583** | — | — |
| Track2Map_PRISM_clean (9776768) | 10.690 | 4.075 | 0.017 | 1 |
| T2M-PRISM-Track-clean (9777547) | 10.782 | 4.174 | 0.018 | — |
| Track2Map-v2 (9773126) | **10.381** | 4.147 | 0.021 | 3 |
| PRISM-DLPL pose (9773960) | 10.723 | 4.344 | 0.017 | 4 |
| Track2Map-v3 (9774107) | 10.590 | 5.228 | 0.017 | 8 |

T2M-DLPE-G22-clean improves the earlier clean submission from 10.690 to
10.591 mm ATE and from 4.075° to 3.583° RotErr. Its official runtime,
coverage, submission ID, and displayed rank were not included in the supplied
leaderboard export and are therefore left unreported here.
The Learned Track5 submission completed all 32 sequences but did not improve
over the clean unconditioned model. Its rank was not included in the exported
leaderboard row available when this README was updated.
Under the challenge protocol, methods within 5% in weighted ATE are
tie-broken by rotational error.

### Main released-data comparison

These results compare the principal depth-to-PnP and direct-PoseNet variants
on four released CLiMB clips and eight EndoMapper clips. Each dataset cell is
`ATE / RotErr`; runtime is locally measured end-to-end processing time.

| Method | CLiMB (4) ↓ | EndoMapper (8) ↓ | ms/frame ↓ |
|---|---:|---:|---:|
| T2M | 5.814 / 4.381 | 3.128 / 7.242 | 43.98 |
| T2M-DLPE-clean | 5.810 / 3.790 | 2.887 / 6.847 | 65.59 |
| **T2M-DLPE-G22-clean** | **4.720 / 3.300** | **2.760 / 6.290** | 30.63* |
| T2M-DLPL | 6.207 / 4.356 | 2.787 / 7.119 | 37.17 |
| PRISM-E | 7.858 / 7.116 | 3.377 / 9.556 | **24.90** |
| PRISM-DLPL | 7.990 / 8.162 | 3.420 / 11.597 | 39.18 |
| PRISM-DLPE | 6.604 / 7.046 | 3.124 / 10.050 | 59.67 |

\* G22 runtime is the exact EndoMapper local run; local timings were collected
under varying machine load and are not directly comparable with official
server timing.

`T2M-*` variants use the indicated depth prediction with CoTracker3 and PnP.
The `PRISM-*` rows instead use direct PoseNet estimates. PRISM-E is the
three-channel RGB model with edge-guided stage-3 pose fine-tuning; DLPL uses
luminance for both networks, while DLPE uses luminance for DepthNet and edges
for PoseNet.

### Motion-conditioning ablations

This ablation compares direct addition of point-track/flow cues with a learned
fifth input channel. It also compares depth followed by Track2Map/PnP against
direct PRISM PoseNet output. Each cell is `CLiMB ATE / RotErr; EndoMapper ATE /
RotErr`; bold values are the best individual metric in each output branch.

| Conditioning | T2M: depth → PnP | PRISM: PoseNet |
|---|---:|---:|
| None (clean DLPE) | 5.810 / **3.790**; 2.887 / 6.847 | 7.08 / 7.36; 3.40 / 9.57 |
| Track-add | 6.28 / 4.33; 2.92 / 7.13 | **7.06** / 7.27; 3.40 / 9.52 |
| Farneback flow-add | 5.90 / 4.36; 3.06 / 7.07 | **7.06** / 7.22; 3.40 / 9.49 |
| Learned Track5 | **5.72** / 4.01; **2.77 / 6.39** | 7.75 / 7.43; 3.58 / 9.79 |
| C3VD flow-add | 7.36 / 6.00; 3.10 / 7.92 | 7.84 / **6.87**; **3.30 / 9.15** |
| C3VD flow5 | 8.19 / 6.09; 3.13 / 8.21 | 8.16 / 6.96; 3.38 / 9.26 |

Among the motion-conditioning ablations, Learned Track5 gives the strongest
released-data depth-to-PnP result. On the
official hidden test it obtained 10.782 mm ATE and 4.174° RotErr, behind both
the earlier clean DLPE result (10.690 mm/4.075°) and G22-clean
(10.591 mm/3.583°). Motion conditioning therefore did
not yield a consistent improvement across released and hidden-test data or
when used for direct pose regression.

---

## ✅ Reproducibility notes

- The PRISM weight manifest records model dimensions, auxiliary-channel
  routing, provenance, and SHA256 hashes.
- Model weights, generated modalities, datasets, and evaluation outputs are
  excluded from Git history.
- Local runtime is diagnostic only; official challenge runtime is measured by
  the server.
- Official hidden-test and local released-data results are reported separately.

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
