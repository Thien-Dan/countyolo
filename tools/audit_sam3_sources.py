"""Fetch public implementation sources for a versioned SAM 3 setup audit; no models loaded."""
import io,json,urllib.request,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'outputs/sam3_audit_sources'
folder.mkdir(parents=True,exist_ok=True)
def get(url):
    return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'countyolo-setup-audit'}),timeout=45).read()
repo='facebookresearch/sam3'
# Freeze the audited reference; updating this is a deliberate new experiment.
commit='f1647e8d0c1d280991f4d1b7d974c2704537fe66'
paths=['sam3/model_builder.py','sam3/model/sam3_video_predictor.py','sam3/model/sam3_base_predictor.py',
       'sam3/model/sam3_video_inference.py','sam3/model/sam3_tracking_predictor.py',
       'sam3/model/sam3_image_processor.py','sam3/model/io_utils.py','sam3/utils/video_loading.py',
       'examples/sam3_video_predictor_example.ipynb','pyproject.toml']
for name in paths:
    try:
        payload=get(f'https://raw.githubusercontent.com/{repo}/{commit}/{name}')
        p=folder/'meta'/name
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(payload)
        print('Meta',name,len(payload),flush=True)
    except Exception as e:
        print(name,type(e).__name__,flush=True)
version='8.4.204'
metadata=json.loads(get(f'https://pypi.org/pypi/ultralytics/{version}/json'))
wheel=next(f for f in metadata['urls'] if f['filename'].endswith('.whl'))
with zipfile.ZipFile(io.BytesIO(get(wheel['url']))) as z:
    for name in z.namelist():
        if name.endswith('.py') and ('/models/sam/' in name or name.endswith('/engine/results.py') or name.endswith('/engine/predictor.py')):
            p=folder/'ultralytics'/name
            p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes(z.read(name))
(folder/'versions.json').write_text(json.dumps(dict(meta_commit=commit,ultralytics=version),indent=2))
print('Audit sources ready',commit,version,flush=True)
