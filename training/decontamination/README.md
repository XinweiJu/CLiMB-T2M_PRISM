# Seq_022-clean PRISM depth training

This pipeline rebuilds every edge-dependent training artifact while excluding
EndoMapper `Seq_022`, which is a held-out CLiMB test sequence. It never edits
the source SegCol dataset.

The dependency chain is deliberately one-way:

1. `build_segcol_manifest.py` pairs SegCol images/fold contours and rejects
   every directory whose normalized sequence ID is 22.
2. `train_dexined_clean.py` fine-tunes the public BIPED DexiNed checkpoint on
   only the clean SegCol train split and selects by the clean validation loss.
3. `generate_hk_edges.py` produces Hyper-Kvasir edge maps with that exact
   checkpoint and records its SHA256 in the output manifest.
4. PRISM stage 2 jointly trains DepthNet/PoseNet from the RGB-only Monodepth2
   checkpoint fine-tuned on Hyper-Kvasir, then expands the depth encoder's
   first convolution from RGB to RGB+luminance. It trains with SHADES/IID
   luminance and the clean edge cache.
5. Only the resulting depth encoder/decoder and SHADES/IID weights enter the
   T2M submission image. DexiNed and PoseNet are training-only dependencies.

Default artifact root:

```text
/raid/scratch_not_backed_up/xinwei/CLiMB/seq022_clean_prism
```

The scripts refuse manifests containing `Seq_022`, `Seq-22`, or equivalent
zero-padded forms. Old DLPE checkpoints must not be used to initialize the
clean stage-2 run because their pose branch was trained with an edge cache of
uncertain provenance.

## Stage-2 initialization

Canonical checkpoint:

```text
/raid/scratch_not_backed_up/xinwei/Datasets/Weights_FEAST/Weights_Feast_Dataset_Finetune_all/monodepth/hkfull_mono_finetuned/models/weights_19
```

This existing checkpoint---not `mono_640x192`---is loaded directly to start
the clean stage-2 run. It is the standard three-channel Monodepth2 DepthNet
and PoseNet fine-tuned on the Hyper-Kvasir `hk` split at 288 x 288 for 20
epochs (batch size 12, learning rate 1e-4). `mono_640x192` is only its upstream
historical initialization. Its saved options contain no DLPE, auxiliary edge
input, or edge loss. Verified hashes are:

```text
encoder.pth  2664990464c4004aa7d6a461c0b004579dd3db350c5a5aa39d31ad5dc16582a3
depth.pth    547faa278d29572480b97662fffb2e3bb0a97a2c58a0c32fa673a11e72ad93b8
pose_encoder.pth  03c5cc93f73b8f5ef19b50c594031607a8482a3e2b622f17c0c75c611f9d9313
pose.pth          70d3aa70f0a8ea5d5bcb20310dee55746b6ff23a9f8dd4f5975d7924319b4058
```

The archived duplicates under `Weights_Feast_all`, `Weights_Feast_Main`, and
`Weights_Feast_Dataset_qualitative` have the same encoder/depth hashes.
Do not initialize from an existing DLPE checkpoint, since that would preserve
the edge-dependent training history this clean rebuild is intended to remove.

## Current clean artifacts

The strict DexiNed run uses the archived `bdcn_loss2`, side weights
`[0.7, 0.7, 1.1, 1.1, 0.3, 0.3, 1.3]`, a 352-pixel crop/resize,
17 epochs, and learning-rate drops at zero-based epochs 10 and 15. Its final
checkpoint is:

```text
/raid/scratch_not_backed_up/xinwei/CLiMB/seq022_clean_prism/edge_model_prism_exact/16_model.pth
sha256 642bb779b66b91add58d2246903d930054a134fa59703870579e1a1689d2b63e
```

The resulting 19,711-image Hyper-Kvasir edge cache is:

```text
/raid/scratch_not_backed_up/xinwei/CLiMB/seq022_clean_prism/hk_edges_prism_exact
```

The 30-epoch joint stage-2 run writes checkpoints at epochs 9, 19, and 29 to:

```text
/raid/scratch_not_backed_up/xinwei/CLiMB/seq022_clean_prism/stage2/hk_mono_finetuned_dlpe_edge_ssim_seq022_clean/models
```
