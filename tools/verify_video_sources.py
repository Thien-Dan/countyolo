from pathlib import Path
import sys
import tempfile
import urllib.request
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from countvid_ext.patch_countvid import main as patch
url='https://raw.githubusercontent.com/niki-amini-naieni/CountVid/main/count_in_videos.py'
source=urllib.request.urlopen(url,timeout=30).read().decode()
with tempfile.TemporaryDirectory() as d:
    p=Path(d)/'count_in_videos.py'
    p.write_text(source,encoding='utf-8')
    patch(d)
    changed=p.read_text(encoding='utf-8')
    compile(changed,str(p),'exec')
    assert 'return None, data_transform' in changed
    assert 'fusion=args.countyolo_fusion' in changed
    assert (Path(d)/'proposals.py').exists()
    print('Upstream CountVid patch and syntax: OK')
