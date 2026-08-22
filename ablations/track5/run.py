#!/usr/bin/env python3
"""CLiMB-format evaluation runner for Track5 depth or pose."""

from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent; PRISM=ROOT.parent
T2M=PRISM/'Track2Map-CLiMB'/'track2map-climb'
for p in (ROOT, T2M): sys.path.insert(0,str(p))
from climb_io import PoseRecord, write_points3d, write_runtime, write_trajectory
from track2map_frontend import Track2MapFrontendConfig, Track2MapMonocularFrontend, load_track2map_models
sys.path.insert(0,str(PRISM/'PRISM-TrackFusion-CLiMB'))
from track_fusion import TrackRasterizer
from inference import Track5DepthGenerator, Track5PoseGenerator

DEPTH='/raid/scratch_not_backed_up/xinwei/Datasets/Weights_FEAST/PRISM_Track5_HK/hk_dlpe_track5_depth/models/weights_9'
POSE='/raid/scratch_not_backed_up/xinwei/Datasets/Weights_FEAST/PRISM_Track5_HK/hk_dlpe_track5_pose/models/weights_9'

class DepthFrontend(Track2MapMonocularFrontend):
    def __init__(self,*a,track5,rasterizer,**kw): super().__init__(*a,**kw); self.track5=track5; self.rasterizer=rasterizer; self.track_map=None
    def _track_segment(self,frames):
        t,v=super()._track_segment(frames); self.track_map=self.rasterizer(t[0],v[0],frames[0].shape[:2]); return t,v
    def _estimate_anchor_depth(self,image,alignment_samples=None):
        old=self.depth_model; self.depth_model=lambda x:self.track5(x,self.track_map)
        try: return super()._estimate_anchor_depth(image,alignment_samples)
        finally: self.depth_model=old

def cfg(max_frames): return Track2MapFrontendConfig(max_frames=max_frames,processing_width=720,segment_length=12,grid_size=18,mixed_precision=True,motion_gate_enabled=False,propagate_on_failure=True,propagation_min_inlier_ratio=.15)
def tracker(video,args,bundle,cls=Track2MapMonocularFrontend,**extra):
    vendor=T2M/'vendor'
    return cls(video_path=video,seed=2834,config=cfg(args.max_frames),cotracker_repo=vendor/'co-tracker',cotracker_checkpoint=vendor/'co-tracker/scaled_online.pth',prism_climb_root=vendor/'PRISM-CLiMB',device=args.device,models=bundle,**extra)

def pose_run(video,args,bundle,raster):
    model=Track5PoseGenerator(POSE,args.device); tr=tracker(video,args,bundle); tr._decoded_frames=0
    records=[]; world=np.eye(4); overlap=None; offset=0
    try:
        while True:
            frames,end=tr._read_segment(overlap)
            if not frames or (overlap is not None and len(frames)==1): break
            tracks,vis=tr._track_segment(frames); maps=[raster(tracks[i],vis[i],frames[i].shape[:2]) for i in range(len(frames))]
            for i in range(0 if not records else 1,len(frames)):
                state='anchor'
                if i>0: world=world@np.linalg.inv(model(frames[i-1],frames[i],maps[i-1],maps[i]).astype(np.float64)); state='track5_pose'
                n=offset+i; records.append(PoseRecord(n+1,n/tr.fps,world.copy(),[state,'1',str(int(vis[i].sum())),'0','0']))
            if end: break
            overlap=frames[-1]; offset+=len(frames)-1
    finally: tr.close()
    return records,[]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--mode',choices=('depth','pose'),required=True); p.add_argument('--input',required=True); p.add_argument('--output',required=True); p.add_argument('--device',default='cuda'); p.add_argument('--max-frames',type=int,default=0); a=p.parse_args()
    vendor=T2M/'vendor'; bundle=load_track2map_models(vendor/'co-tracker',vendor/'co-tracker/scaled_online.pth',vendor/'PRISM-CLiMB',device=a.device); raster=TrackRasterizer(3,2.0)
    inputs=[Path(a.input)] if Path(a.input).is_file() else sorted(Path(a.input).glob('*.mp4'))
    for video in inputs:
        start=time.perf_counter()
        if a.mode=='depth':
            est=tracker(video,a,bundle,DepthFrontend,track5=Track5DepthGenerator(DEPTH,a.device),rasterizer=raster)
            try: records,points=est.process()
            finally: est.close()
        else: records,points=pose_run(video,a,bundle,raster)
        elapsed=time.perf_counter()-start; root=Path(a.output)/video.stem/'1'
        write_trajectory(root/'camera_trajectory/cam_traj_map_000.txt',records); write_points3d(root/'3D_maps/0/points3D.txt',points); write_runtime(root/'runtime.txt',0,elapsed)
        print(f'{video.stem}: frames={len(records)} points={len(points)} seconds={elapsed:.3f}',flush=True)
if __name__=='__main__': main()
