"""Run the original CountVid video program with only its detector replaced."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from video.countvid_yoloworld import patch_reference,assert_video_algorithm_unchanged,load_reference_transforms,load_replacement,replacement_count,REFERENCE_SHA256,SAM2_SHA256
from models.yoloworld_countvid import YOLOWorldCountVid
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__,add_help=False)
    p.add_argument('--upstream',type=Path,required=True);p.add_argument('--yolo-weights',type=Path,required=True)
    p.add_argument('--protocol-manifest',type=Path,required=True)
    p.add_argument('--pretrain_model_path',type=Path,required=True)
    p.add_argument('--sam_checkpoint',type=Path,required=True)
    p.add_argument('--sam_model_cfg',default='configs/sam2.1/sam2.1_hiera_l.yaml')
    args,remaining=p.parse_known_args();repo=args.upstream.resolve();weights=args.yolo_weights.resolve()
    source_path=repo/'count_in_videos.py';source=source_path.read_text()
    if hashlib.sha256(source.encode()).hexdigest()!=REFERENCE_SHA256:raise RuntimeError('Use pinned CountVid commit690ca3e48369c35f5e23a3ab5d2b04b56881a343')
    sam_path=args.sam_checkpoint.resolve()
    digest=hashlib.sha256()
    with sam_path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    if digest.hexdigest()!=SAM2_SHA256 or args.sam_model_cfg!='configs/sam2.1/sam2.1_hiera_l.yaml':
        raise ValueError('Main-paper protocol requires original SAM2.1 Hiera-L weights and configuration')
    checkpoint=args.pretrain_model_path.resolve()
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if state.get('architecture')!=YOLOWorldCountVid.ARCHITECTURE or state.get('phase')!='box' or state.get('manifest',{}).get('smoke'):
        raise ValueError('A completed multimodal box-phase checkpoint is required before loading SAM2')
    del state
    tree=patch_reference(source);assert_video_algorithm_unchanged(source,tree)
    args.protocol_manifest.parent.mkdir(parents=True,exist_ok=True)
    args.protocol_manifest.write_text(json.dumps(dict(upstream_sha256=hashlib.sha256(source.encode()).hexdigest(),
        detector_functions_replaced=['build_model_and_transforms','count'],video_algorithm_ast_unchanged=True,
        sam2_temporal_filter_and_birth_logic='original upstream statements and function bodies unchanged',
        transform='original resize/normalization retained externally, normalization inverted only inside detector adapter',
        legacy_text_only_checkpoint_allowed=False),indent=2))
    transforms=load_reference_transforms(repo/'datasets/transforms.py')
    os.chdir(repo);sys.path.insert(0,str(repo));sys.argv=[str(source_path),*remaining,'--pretrain_model_path',str(checkpoint),
        '--sam_checkpoint',str(sam_path),'--sam_model_cfg',args.sam_model_cfg]
    namespace=dict(__name__='__main__',__file__=str(source_path),T=transforms,
        _load_replacement=lambda args:load_replacement(args,weights),_replacement_count=replacement_count)
    exec(compile(tree,str(source_path),'exec'),namespace)


if __name__=='__main__':main()
