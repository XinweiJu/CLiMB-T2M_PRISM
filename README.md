# CLiMB-T2M_PRISM

Official source release for **T2M-DLPE-clean**, our PRISM-conditioned
Track2Map entry for the CLiMB monocular colonoscopy localization challenge.

T2M-DLPE-clean replaces Track2Map's generic monocular anchor-depth predictor
with the luminance-conditioned PRISM DLPE DepthNet. CoTracker3 Online supplies
multi-frame point correspondences. Anchor inverse depth lifts these
correspondences into 3D, PnP-RANSAC estimates relative camera motion, bounded
motion propagation handles short tracking failures, and fixed-lag smoothing
regularizes the completed trajectory.

Only PRISM DepthNet and the IID/SHADES luminance generator run during
submission inference. PRISM PoseNet and DexiNed are training-only components.
The released clean checkpoint lineage excludes the challenge-designated
EndoMapper sequence from SegCol-derived edge supervision.

## Pipeline

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

## Repository layout

- `submission/`: CLiMB container entrypoint and T2M-DLPE-clean inference code.
- `training/`: PRISM depth/pose training and leakage-controlled edge generation.
- `ablations/`: track, optical-flow, and learned-extra-channel experiments.

Model weights, datasets, generated modalities, evaluation outputs, and the
CoTracker3 checkout are intentionally not committed. See
`submission/README.md` for the expected local Docker layout.

## Requirements

- Linux with an NVIDIA GPU and NVIDIA Container Toolkit;
- Docker with CUDA-compatible driver support;
- the released T2M-DLPE-clean PRISM weight bundle;
- the official CoTracker3 source and `scaled_online.pth` checkpoint.

The challenge image uses a PyTorch 2.7.1/CUDA 12.8 runtime. The inference
entrypoint itself receives OpenCV BGR uint8 frames. PRISM resizes each anchor
to $288\times288$, predicts inverse depth, and bilinearly restores the result
to the original frame size before Track2Map uses it.

## Model weights

Our GitHub Release contains only the PRISM components trained or selected for
this method:

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

Verify the downloaded archive against its accompanying SHA256 file before
building the image. The manifest also records hashes for every individual
checkpoint.

CoTracker3 is not redistributed in our Release. Obtain the code from the
[official CoTracker repository](https://github.com/facebookresearch/co-tracker)
and download its official online checkpoint:

```bash
git clone https://github.com/facebookresearch/co-tracker.git
mkdir -p co-tracker/checkpoints
wget -O co-tracker/checkpoints/scaled_online.pth \
  https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth
```

For this repository's Docker layout, copy or link the checkpoint to
`submission/vendor/co-tracker/scaled_online.pth`.

## Build and run

After downloading the PRISM Release archive and CoTracker3, arrange the files
as documented in `submission/README.md`, then run:

```bash
docker build -t t2m-dlpe-clean:local submission
docker run --rm --gpus all --network=none \
  -v /absolute/input:/input:ro \
  -v /absolute/output:/output \
  t2m-dlpe-clean:local
```

The input directory contains challenge MP4 files. The output follows the
CLiMB COLMAP-style contract and contains five trajectory/map runs per sequence.

## Training

The leakage-controlled PRISM training lineage has three stages:

1. retrain DexiNed on the allowed SegCol split and precompute edge maps;
2. jointly train PRISM DepthNet and PoseNet on real Hyper-Kvasir video with
   RGB+luminance depth input, RGB+edge pose input, and edge-SSIM;
3. freeze the complete depth encoder/decoder and post-fine-tune PoseNet.

The selected submission uses the stage-2 DepthNet only. Stage 3 is retained to
reproduce the direct-pose ablations. Scripts and configuration are under
`training/`; track/flow conditioning experiments are isolated under
`ablations/`.

No hidden CLiMB video, trajectory, COLMAP reconstruction, or hidden-test-derived
artifact is used for training, checkpoint selection, or hyperparameter tuning.

## Method name and short description

**T2M-DLPE-clean** — CoTracker3 Online correspondences combined with
leakage-controlled PRISM DLPE inverse depth, depth-assisted PnP, bounded motion
propagation, and fixed-lag trajectory smoothing.

The complete submission-form text and measured EndoMapper short-sequence
wall-clock values are provided in `submission/README.md`.

## Reproducibility notes

- The PRISM weight manifest records model dimensions, auxiliary-channel
  routing, provenance, and SHA256 hashes.
- Model weights, generated modalities, datasets, and evaluation outputs are
  excluded from Git history.
- Local runtime is diagnostic only; official challenge runtime is measured by
  the server.
- The final method report will be added after the clean Table 3/Table 5
  ablations are complete.

## Citation

Please cite Track2Map, PRISM, CoTracker3, Monodepth2, Hyper-Kvasir, EndoMapper,
and SegCol when using the corresponding components or training procedure.
