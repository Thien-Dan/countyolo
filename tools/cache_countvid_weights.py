"""Upload the existing CountGD-Box checkpoint once to a private Kaggle cache."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tools'))
from video.eval_common import atomic_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--username',default='ttdan2006')
    a=p.parse_args()
    source=ROOT/'checkpoints/countgd_box.pth'
    folder=ROOT/'outputs/countvid_weights'
    folder.mkdir(parents=True,exist_ok=True)
    target=folder/'countgd_box.pth'
    if not target.exists():
        try:
            os.link(source,target)
        except OSError:
            shutil.copyfile(source,target)
    with source.open('rb') as f:
        digest=hashlib.file_digest(f,'sha256').hexdigest()
    with target.open('rb') as f:
        assert hashlib.file_digest(f,'sha256').hexdigest()==digest
    atomic_json(folder/'weights_manifest.json',dict(file='countgd_box.pth',bytes=source.stat().st_size,sha256=digest,
        source='CountVid official CountGD-Box checkpoint, Google Drive 1bw-YIS-Il5efGgUqGVisIZ8ekrhhf_FD'))
    atomic_json(folder/'dataset-metadata.json',dict(id=f'{a.username}/countvid-weights',
        title='CountVid CountGD-Box checkpoint cache',licenses=[dict(name='other')],
        description='Private research cache of the original CountGD-Box checkpoint distributed by CountVid. '
        'All original model licenses and rights remain applicable; no relicensing is intended. '
        'Source: https://github.com/niki-amini-naieni/CountVid'))
    from kaggle_watch import accounts,api_for
    api=api_for(a.username,accounts()[a.username])
    result=api.dataset_create_new(str(folder),public=False,quiet=True,convert_to_csv=False)
    if result.error:
        raise RuntimeError(result.error)
    print('Created private checkpoint cache:',a.username+'/countvid-weights',flush=True)


if __name__=='__main__':
    main()
