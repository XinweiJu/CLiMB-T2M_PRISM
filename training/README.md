# EndoMapper-CLiMB PRISM fine-tuning

This directory keeps CLiMB training separate from the submission/inference code.

The required three stages are:

1. Freeze IID and DexiNed and precompute 288x288 shading and edge images.
2. Jointly fine-tune DepthNet and PoseNet with edge-SSIM reprojection.
3. Load stage 2, freeze the complete depth encoder/decoder, and fine-tune PoseNet.

The two variants are:

| Variant | Depth input | Pose input | Auxiliary loss |
| --- | --- | --- | --- |
| `dlpe` | RGB + IID shading | RGB + DexiNed edge | edge SSIM |
| `dlpl` | RGB + IID shading | RGB + IID shading | edge SSIM |

Build the pinned A100-compatible environment once:

```bash
docker build -t prism-depth-climb:train .
```

The environment fixes PyTorch 2.7.1/CUDA 12.8, OpenCV 4.11, NumPy 2.2.6,
Pillow 11, and tensorboardX 2.6.4. The host's PyTorch 1.10/CUDA 10.2 must not
be used on A100 because it has no `sm_80` kernels.
The launcher also allocates a 16 GB Docker shared-memory segment for the
multi-worker DataLoader; Docker's 64 MB default causes worker bus errors.

The default command processes `trainval.txt` at 10 sampled frames per second,
then runs DLPE first and DLPL second. GPU 1 is the default and can be changed
with `PRISM_GPU`:

```bash
./run_docker.sh
PRISM_GPU=3 ./run_docker.sh
```

Useful controls:

```bash
# Inspect every command without writing data or training
./run_docker.sh --dry-run

# Precompute only; the operation resumes by skipping existing outputs
./run_docker.sh --prepare-only

# Train after preprocessing has completed
./run_docker.sh --skip-prepare

# Run one branch
./run_docker.sh --skip-prepare --variants dlpe
```

Inputs are read from:

```text
/raid/scratch_not_backed_up/xinwei/CLiMB/EndoMapper-CLiMB/raw
/raid/scratch_not_backed_up/xinwei/CLiMB/EndoMapper-CLiMB/trainval.txt
```

Prepared data, metadata, TensorBoard logs, and checkpoints are written below:

```text
/raid/scratch_not_backed_up/xinwei/CLiMB/EndoMapper-CLiMB/prism_depth_climb
```

Validation is split by sequence using seed 2026 (54 train sequences and 6
validation sequences), rather than splitting neighboring frames from one video.
The source-frame/time mapping for every sampled frame is retained under
`metadata/`.

The preprocessor checks every sequence named by `trainval.txt` before expensive
inference. Missing videos are reported and skipped; train/validation sequence
splits are then computed only from the videos which are actually available.

The 60 videos contain about 17.68 hours of footage. The 10 FPS default produces
approximately 636,000 samples. Increase `--target-fps` only if the additional
storage and training time are intentional.

## A100 benchmark

DLPE joint-training benchmarks on one A100 selected batch size 64 with 4
DataLoader workers:

| Batch | Workers | Samples/s | Peak GPU memory |
| ---: | ---: | ---: | ---: |
| 12 | 12 | 35.45 | 5.47 GiB |
| 24 | 12 | 48.48 | 10.27 GiB |
| 48 | 12 | 60.41 | 20.06 GiB |
| 64 | 4 | **66.56** | 26.58 GiB |
| 72 | 4 | 58.81 | 29.84 GiB |
| 84 | 4 | 63.04 | 34.73 GiB |

The launcher therefore defaults to batch 64, 4 workers, and checkpoints every
5 epochs. With 577,845 training frames, one epoch is approximately 2.41 hours
and one 30-epoch stage is approximately 72 hours. DLPE and DLPL can run on two
different GPUs concurrently; each variant still runs joint training followed
by its frozen-depth pose stage.
