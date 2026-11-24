"""Image SAM3 detector only: FSC147 val and independently scored VideoCount frames."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial.distance import cdist
from video.eval_common import DATASETS,cases,atomic_json,frame_count_annotation


def class_names(root):
    result={}
    for line in (root/'ImageClasses_FSC147.txt').read_text(encoding='utf-8').splitlines():
        parts=line.split('\t')
        if len(parts)>=2:result[parts[0]]=parts[1].strip()
    return result


def point_match(boxes,points,size,radius):
    boxes=np.asarray(boxes,dtype=float).reshape(-1,4)
    points=np.asarray(points,dtype=float).reshape(-1,2)
    scale=np.array([640/size[0],640/size[1]])
    centers=(boxes[:,:2]+boxes[:,2:])/2*scale;points=points*scale
    tp=0
    if len(centers) and len(points):
        graph=csr_matrix(cdist(centers,points)<=radius)
        tp=int((maximum_bipartite_matching(graph,perm_type='column')>=0).sum())
    return dict(tp=tp,fp=len(centers)-tp,fn=len(points)-tp)


def summary(rows,expected):
    good=[r for r in rows if r['status']=='complete']
    labeled=[r for r in good if r['gt'] is not None]
    errors=np.array([r['pred']-r['gt'] for r in labeled],dtype=float)
    result=dict(complete=len(rows)==expected and len(good)==expected,expected=expected,
        processed=len(rows),successful=len(good),failed=len(rows)-len(good),scored=len(labeled),
        missing_gt=len(good)-len(labeled),mae=float(abs(errors).mean()) if len(errors) else None,
        rmse=float(np.sqrt((errors**2).mean())) if len(errors) else None,
        bias=float(errors.mean()) if len(errors) else None)
    if labeled and all('points' in r for r in labeled):
        result['points_640']={}
        for radius in ('8','16'):
            tp,fp,fn=[sum(r['points'][radius][k] for r in labeled) for k in ['tp','fp','fn']]
            result['points_640'][radius]=dict(tp=tp,fp=fp,fn=fn,precision=tp/max(1,tp+fp),
                recall=tp/max(1,tp+fn),f1=2*tp/max(1,2*tp+fp+fn))
    return result


def run_cases(records,counter,output,kind):
    rows=[];expected=len(records)
    for i,record in enumerate(records):
        row={k:v for k,v in record.items() if k not in ('path','gt_points')}
        started=time.perf_counter()
        try:
            path=Path(record['path']);row['image_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            with Image.open(path) as im:image=im.convert('RGB')
            result=counter.count_image(image,record['prompt'])
            assert result['grid']==1,'Detector-only configuration must not tile'
            row.update(status='complete',pred=result['count'],boxes=result['boxes'].tolist(),
                scores=result['scores'].tolist(),image_size=list(image.size),numerics=result['numerics'])
            if 'gt_points' in record:
                row['points']={str(r):point_match(result['boxes'],record['gt_points'],image.size,r) for r in (8,16)}
        except Exception as exc:
            row.update(status='failed',pred=None,error=f'{type(exc).__name__}: {exc}')
            if torch.cuda.is_available():torch.cuda.empty_cache()
        row['seconds']=time.perf_counter()-started;rows.append(row)
        # JSONL preserves each image if a long-running job is interrupted.
        with (output/(kind+'_rows.jsonl')).open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        if (i+1)%25==0 or i+1==expected:
            state=summary(rows,expected);atomic_json(output/(kind+'_results.json'),dict(summary=state,rows=rows))
            print(kind,i+1,expected,'MAE',state['mae'],'failed',state['failed'],flush=True)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fsc-root',type=Path,required=True);p.add_argument('--video-root',type=Path,required=True)
    p.add_argument('--weights',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if any(a.output.glob('*_rows.jsonl')):raise RuntimeError('Use a fresh output directory')
    if not torch.cuda.is_available():raise RuntimeError('Model evaluation requires Kaggle GPU')
    import ultralytics
    if ultralytics.__version__!='8.4.204':raise RuntimeError('Pinned Ultralytics8.4.204 required')
    split=json.loads((a.fsc_root/'Train_Test_Val_FSC_147.json').read_text())
    annotations=json.loads((a.fsc_root/'annotation_FSC147_384.json').read_text())
    names=class_names(a.fsc_root)
    assert not set(split['val'])&set(split['test'])
    fsc=[dict(image=name,path=str(a.fsc_root/'images_384_VarV2'/name),prompt=names[name],
              gt=len(annotations[name]['points']),gt_points=annotations[name]['points']) for name in split['val']]
    video=[]
    for dataset,(sub,_,_) in DATASETS.items():
        labels=json.loads((a.video_root/sub/'anno'/frame_count_annotation(dataset)).read_text())
        for case in cases(a.video_root,dataset):
            by_frame=labels.get(case['video'],{}).get(case['text'],{})
            for path in case['frames']:
                name=Path(path).name;gt=by_frame.get(name)
                if gt is not None and (not isinstance(gt,(int,float)) or gt<0):raise ValueError('Invalid frame GT')
                record=dict(dataset=dataset,video=case['video'],image=name,path=path,
                    canonical_prompt=case['text'],prompt=case['text'],arm='canonical',gt=gt)
                video.append(record)
                if dataset=='mot20':video.append(dict(record,prompt='person',arm='person_alias'))
    metadata=dict(engine='SAM3SemanticPredictor image detector/segmentation only, Ultralytics port',
        tracker=False,temporal_identity=False,tiling=False,visual_exemplars=False,fine_tuned=False,
        conf=.5,precision='fp16',imgsz=1008,max_grid=1,fsc_split='val',fsc_expected=len(fsc),
        video_expected=len(video),video_canonical_frames=sum(r['arm']=='canonical' for r in video),
        video_missing_gt=[{k:r[k] for k in ['dataset','video','image','arm']} for r in video if r['gt'] is None],
        note='No test split, threshold selection, random absent labels, tracking or unique-video count. '
             'VID MAE uses per-frame GT. person_alias is separate from canonical human. '
             'FSC point localization of box centers is not bbox AP. Single-pass query budget is200.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    atomic_json(a.output/'experiment.json',metadata)
    from engines.sam3_counter import SAM3Counter
    from unittest.mock import patch
    counter=SAM3Counter(a.weights,conf=.5,half=True,max_grid=1);reports=[]
    original_load=torch.nn.Module.load_state_dict
    def audit_load(module,*args,**kw):
        result=original_load(module,*args,**kw)
        missing=sorted(set(result.missing_keys)&dict(module.named_parameters()).keys())
        reports.append(dict(model=type(module).__name__,missing_parameters=missing,
                            missing_buffers=sorted(set(result.missing_keys)-set(missing)),unexpected=len(result.unexpected_keys)))
        if missing:raise RuntimeError('Incomplete checkpoint mapping')
        return result
    with patch.object(torch.nn.Module,'load_state_dict',audit_load):counter._image_predictor().setup_model()
    atomic_json(a.output/'weight_load_reports.json',reports)
    # Score each frame independently. No video container, state or track IDs.
    vid_rows=run_cases(video,counter,a.output,'video_frames')
    fsc_rows=run_cases(fsc,counter,a.output,'fsc147_val')
    groups={}
    for dataset in DATASETS:
        for arm in ['canonical','person_alias']:
            selected=[r for r in vid_rows if r['dataset']==dataset and r['arm']==arm]
            if not selected:continue
            by_video={v:summary([r for r in selected if r['video']==v],sum(r['video']==v for r in selected))
                      for v in sorted({r['video'] for r in selected})}
            aggregate=summary(selected,len(selected))
            maes=[r['mae'] for r in by_video.values() if r['mae'] is not None]
            aggregate['video_macro_mae']=sum(maes)/len(maes) if maes else None
            groups[dataset+'/'+arm]=dict(summary=aggregate,videos=by_video)
    result=dict(fsc147_val=summary(fsc_rows,len(fsc)),video_frames=groups,metadata=metadata,
                complete=all(r['status']=='complete' for r in vid_rows+fsc_rows))
    atomic_json(a.output/'results.json',result)
    if not result['complete']:raise RuntimeError('Some images failed; partial metrics are not a complete evaluation')


if __name__=='__main__':main()
