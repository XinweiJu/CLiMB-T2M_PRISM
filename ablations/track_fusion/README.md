# PRISM TrackFusion CLiMB

An isolated experiment which conditions the original (non-EndoMapper-trained)
DLPE depth and pose checkpoints on sparse points exported by Track2Map's
CoTracker frontend.

The pretrained depth encoder has four input channels: RGB plus IID shading.
The pretrained DLPE pose encoder has eight: two RGB+DexiNed-edge frames. To
retain exact checkpoint compatibility, the default experiment rasterizes the
visible points and fuses that heat map into the existing auxiliary channel.
No learned 5th channel is silently initialized.

Two modes are provided:

* `depth`: fused DLPE depth replaces Track2Map's anchor depth, while Track2Map
  continues to estimate pose with PnP.
* `pose`: CoTracker points are rasterized for each adjacent pair and fused into
  the original DLPE pose model, which directly outputs relative pose.

```bash
python run.py --mode depth --input /path/to/input --output /tmp/depth-fusion
python run.py --mode pose  --input /path/to/input --output /tmp/pose-fusion
```

Use `--fusion add|blend` and `--track-alpha 0.2`. `add` preserves the original
auxiliary signal away from tracks and is the default. The project references
the sibling PRISM/Track2Map source trees but never changes them.

`--conditioning flow` replaces the sparse point heat map with a dense,
robustly normalized Farneback optical-flow magnitude map. It needs no extra
checkpoint. CoTracker remains active in `depth` mode because Track2Map still
needs point correspondences for PnP, but its points are not fused into DLPE.
