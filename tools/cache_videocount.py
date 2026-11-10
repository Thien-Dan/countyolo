"""Create a private Kaggle cache of the exact CountVid sampled JPEG frames."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tools'))
from video.eval_common import DATASETS,atomic_json,frame_paths


def build(data_root,folder):
    folder.mkdir(parents=True,exist_ok=True)
    target=folder/'videocount-benchmark.zip'
    if target.exists() and (folder/'cache_build.json').exists():
        print('Using existing cache:',target,flush=True)
        return target
    marker=dict(schema_version=1,protocol='countvid-text-v1',pre_sampled=True,videos={},files_sha256={})
    total=0
    with zipfile.ZipFile(target.with_suffix('.tmp'),'w',compression=zipfile.ZIP_STORED) as z:
        for ds,(sub,annotation,factor) in DATASETS.items():
            directory=data_root/sub
            gt=json.loads((directory/'anno'/annotation).read_text(encoding='utf-8'))
            marker['videos'][ds]={}
            for video in gt:
                frames=frame_paths(directory/'frames'/video,factor)
                marker['videos'][ds][video]=dict(frame_names=[Path(f).name for f in frames],original_downsample=factor)
                for f in frames:
                    p=Path(f)
                    relative='VideoCount/'+p.relative_to(data_root).as_posix()
                    payload=p.read_bytes()
                    marker['files_sha256'][relative]=hashlib.sha256(payload).hexdigest()
                    z.writestr(relative,payload)
                    total+=1
                    if total%200==0:
                        print('Cached',total,'original JPEG frames',flush=True)
            for p in (directory/'anno').rglob('*'):
                if p.is_file():
                    payload=p.read_bytes()
                    relative='VideoCount/'+p.relative_to(data_root).as_posix()
                    marker['files_sha256'][relative]=hashlib.sha256(payload).hexdigest()
                    z.writestr(relative,payload)
        z.writestr('VideoCount/countyolo_sampling.json',json.dumps(marker,indent=2))
    target.with_suffix('.tmp').replace(target)
    atomic_json(folder/'cache_build.json',dict(frames=total,bytes=target.stat().st_size,
                sha256=hashlib.sha256(target.read_bytes()).hexdigest(),sampling=marker['videos']))
    print('Cache ready:',total,'frames,',round(target.stat().st_size/1024**2,1),'MiB',flush=True)
    return target


def upload(folder,username):
    from kaggle_watch import accounts,api_for
    api=api_for(username,accounts()[username])
    ref=f'{username}/videocount-benchmark'
    atomic_json(folder/'dataset-metadata.json',dict(id=ref,title='VideoCount sampled benchmark cache',
        licenses=[dict(name='other')],
        description='Private research cache of original VideoCount JPEG frames and annotations from CountVid. '
        'Original authors retain all rights; no relicensing is intended. '
        'Frames are selected using the exact CountVid ceil/round/linspace protocol. '
        'MOT20 is presampled at 25/3; Crystals and Penguins retain all provided frames. '
        'No pixels are resized or recompressed. Source: https://github.com/niki-amini-naieni/CountVid'))
    result=api.dataset_create_new(str(folder),public=False,quiet=False,convert_to_csv=False)
    if result.error:
        raise RuntimeError(result.error)
    print('Created private dataset:',ref,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,default=ROOT/'data/VideoCount')
    p.add_argument('--folder',type=Path,default=ROOT/'outputs/videocount_cache')
    p.add_argument('--username',nargs='+')
    a=p.parse_args()
    build(a.data_root,a.folder)
    for user in a.username or []:
        upload(a.folder,user)
