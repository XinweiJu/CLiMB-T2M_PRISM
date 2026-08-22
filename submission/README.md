# T2M-DLPE-clean submission

The container accepts CLiMB input under `/input` and writes five COLMAP-style
trajectory/map runs per sequence under `/output`.

Before building, populate the following untracked files:

```text
submission/vendor/co-tracker/
  scaled_online.pth
  ... CoTracker3 source ...
submission/vendor/PRISM-CLiMB/
  checkpoints.json
  prism_climb/
  weights/depth/{encoder.pth,depth.pth}
  weights/shading/{decompose_encoder.pth,decompose.pth}
```

`checkpoints.example.json` records the selected architecture, preprocessing,
checkpoint hashes, and training provenance. Copy it to
`vendor/PRISM-CLiMB/checkpoints.json` after placing weights with matching
hashes.

Build and smoke-test locally:

```bash
docker build -t t2m-dlpe-clean:local submission
docker run --rm --gpus all --network=none \
  -v /absolute/input:/input:ro \
  -v /absolute/output:/output \
  t2m-dlpe-clean:local
```

## Submission-form text

**Method name:** T2M-DLPE-clean

**One-line description:** CoTracker3 Online correspondences combined with
leakage-controlled PRISM DLPE inverse depth, depth-assisted PnP, bounded motion
propagation, and fixed-lag trajectory smoothing.

Local processing wall-clock on the eight short EndoMapper diagnostic clips
(initialization excluded):

| Sequence | Frames | Seconds |
| --- | ---: | ---: |
| Seq_001_m01 | 403 | 18.90 |
| Seq_001_m04 | 283 | 13.68 |
| Seq_001_m06 | 539 | 26.22 |
| Seq_001_m19 | 208 | 8.12 |
| Seq_002_m01 | 394 | 13.53 |
| Seq_002_m16 | 557 | 24.39 |
| Seq_002_m21 | 256 | 10.53 |
| Seq_002_m33 | 399 | 17.08 |

Concise form value: approximately **8.12--26.22 seconds per sequence** on
208--557-frame EndoMapper short clips. The first sequence additionally incurred
2.49 seconds of one-time model initialization; subsequent initialization was
approximately 0.003 seconds.

The mandatory final method write-up is intentionally not embedded in this
source tree yet. Add the finalized report to the submission bundle only after
the clean ablations have completed.
