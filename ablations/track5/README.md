# PRISM Track5 CLiMB

Independent five-channel training experiment using the original, non-
EndoMapper-adapted DLPE `weights_19` checkpoint.

Inputs per image:

* depth: RGB + IID shading + CoTracker heat map = 5 channels;
* pose: RGB + DexiNed edge + CoTracker heat map = 5 channels, or 10 channels
  for an adjacent pair.

The original four channels are copied exactly from the source checkpoint. The
new motion channel is initialized to `0.1 * original_auxiliary_conv_weight` and
then learned. Depth-only training freezes pose; pose-only training freezes
depth.

Data preparation uses video-directory-level splitting (seed 2026), so no
adjacent frames from a held-out video enter training.

```bash
python prepare_tracks.py --help
python run_tracks_training.py --experiment depth --benchmark-batches 20
python run_tracks_training.py --experiment depth
python run_tracks_training.py --experiment pose
```

The training engine is shared with `../prism_depth_climb/training`; all new
options default to disabled, preserving its existing 3/4-channel behavior.
