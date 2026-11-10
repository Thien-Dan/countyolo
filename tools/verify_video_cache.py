"""Verify cached and original VideoCount produce identical cases and JPEG bytes."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from video.eval_common import DATASETS,cases


def main():
    archive=ROOT/'outputs/videocount_cache/videocount-benchmark.zip'
    with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as tmp:
        with zipfile.ZipFile(archive) as z:
            z.extractall(tmp)
        root=Path(tmp)/'VideoCount'
        marker=json.loads((root/'countyolo_sampling.json').read_text())
        n=0
        for ds in DATASETS:
            original=cases(ROOT/'data/VideoCount',ds)
            cached=cases(root,ds)
            assert len(original)==len(cached)
            for a,b in zip(original,cached):
                for k in ('case_id','gt','frame_manifest','frame_sha256','downsample'):
                    assert a[k]==b[k],(ds,a['video'],k)
                assert b['inference_downsample']==1
                for f in b['frames']:
                    relative='VideoCount/'+Path(f).relative_to(root).as_posix()
                    assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==marker['files_sha256'][relative]
                n+=1
        print('Cache verified:',n,'video/prompt cases; all selected JPEG bytes and frame hashes match original')


if __name__=='__main__':
    main()
