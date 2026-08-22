# CLiMB-T2M_PRISM

Source release for **T2M-DLPE-clean**, the PRISM-conditioned Track2Map method
developed for the CLiMB challenge.

The method replaces Track2Map's generic monocular anchor-depth predictor with
the luminance-conditioned PRISM DLPE DepthNet. CoTracker3 supplies multi-frame
point correspondences; inverse depth lifts anchor observations into 3D, and
PnP-RANSAC estimates camera motion. The released clean checkpoint lineage
excludes the challenge-designated EndoMapper sequence from SegCol-derived edge
supervision.

## Repository layout

- `submission/`: CLiMB container entrypoint and T2M-DLPE-clean inference code.
- `training/`: PRISM depth/pose training and leakage-controlled edge generation.
- `ablations/`: track, optical-flow, and learned-extra-channel experiments.

Model weights, datasets, generated modalities, evaluation outputs, and the
CoTracker3 checkout are intentionally not committed. See
`submission/README.md` for the expected local Docker layout.

## Method name and short description

**T2M-DLPE-clean** — CoTracker3 Online correspondences combined with
leakage-controlled PRISM DLPE inverse depth, depth-assisted PnP, bounded motion
propagation, and fixed-lag trajectory smoothing.

## Citation

Please cite Track2Map, PRISM, CoTracker3, Monodepth2, Hyper-Kvasir, EndoMapper,
and SegCol when using the corresponding components or training procedure.
