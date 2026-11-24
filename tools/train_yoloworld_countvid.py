"""Two-phase CountGD -> CountGD-Box curriculum with a YOLO-World detector."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial.distance import cdist
from models.yoloworld_counting import CountingCriterion,xywh_to_xyxy
from models.yoloworld_countvid import YOLOWorldCountVid
from datasets.countvid_protocol import CountVidData


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def logical_batch_weights(samples):
    """Undo per-image criterion normalization, then normalize over the batch."""
    counts=[len(sample['points']) for sample in samples]
    denominator=max(1,sum(counts))
    return [max(1,count)/denominator for count in counts]


def save_model(path,model,phase,epoch,mae,manifest,optimizer=None,scheduler=None):
    tmp=path.with_suffix('.tmp')
    torch.save(dict(architecture=model.ARCHITECTURE,phase=phase,epoch=epoch,mae=mae,
        model=model.export_state(),manifest=manifest,
        optimizer=optimizer.state_dict() if optimizer else None,scheduler=scheduler.state_dict() if scheduler else None,
        rng_python=random.getstate(),rng_numpy=np.random.get_state(),rng_torch=torch.get_rng_state(),
        rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),tmp)
    tmp.replace(path)


def summarize(rows):
    result={}
    for threshold in rows[0]['methods']:
        errors=np.array([r['methods'][threshold]['count']-r['gt'] for r in rows])
        tp,fp,fn=[sum(r['methods'][threshold][k] for r in rows) for k in ['tp','fp','fn']]
        result[threshold]=dict(mae=float(abs(errors).mean()),rmse=float(np.sqrt((errors**2).mean())),
            precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),f1=2*tp/max(1,2*tp+fp+fn))
    return result


@torch.no_grad()
def evaluate(model,data,device,mode,output,names=None,absent=False):
    model.set_training(False);rows=[]
    for name in (names if names is not None else data.names):
        sample=data.get(name);image=sample['image'][None].to(device)
        if absent and mode!='text':raise ValueError('Absent diagnostic must not receive positive exemplars')
        prompt='elephants' if absent else (None if mode=='exemplar' else sample['prompt'])
        ex=sample['exemplar_boxes'].to(device) if mode!='text' else None
        boxes,logits=model(image,prompt,exemplar_boxes=ex)
        pw,ph=sample['padded_size'];w,h=sample['resized_size']
        centers=boxes[:,:2].cpu().numpy()*[pw/w*640,ph/h*640]
        valid=((centers>=0)&(centers<=640)).all(-1)
        points=np.empty((0,2)) if absent else sample['points'].numpy()*[pw/w*640,ph/h*640]
        methods={}
        for threshold in [.14,.17,.20,.23,.26]:
            selected=centers[(logits.sigmoid().cpu().numpy()>threshold)&valid]
            tp=int((maximum_bipartite_matching(csr_matrix(cdist(selected,points)<=16),perm_type='column')>=0).sum()) if len(selected) and len(points) else 0
            methods[str(threshold)]=dict(count=len(selected),tp=tp,fp=len(selected)-tp,fn=len(points)-tp)
        rows.append(dict(image=name,prompt=prompt,gt_class=sample['prompt'],gt=len(points),methods=methods))
        if len(rows)%100==0:print('eval',mode,len(rows),flush=True)
    value=dict(mode=mode,n=len(rows),summary=summarize(rows),rows=rows);save(output,value);return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--protocol-root',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=30,help='Epochs PER PHASE: counting then box, 60 total by default')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--phase',choices=['point','box','both'],default='both')
    p.add_argument('--init-counting',type=Path,help='Required counting-phase checkpoint for a separate box-phase job')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'manifest.json').exists():raise RuntimeError('Use fresh output')
    random.seed(20261120);np.random.seed(20261120);torch.manual_seed(20261120);torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    train=CountVidData(a.data_root/'images_384_VarV2',a.protocol_root,'train')
    val=CountVidData(a.data_root/'images_384_VarV2',a.protocol_root,'val')
    assert not set(train.names)&set(val.names)
    model=YOLOWorldCountVid(a.weights,device)
    model.cache_text(sorted(set(train.prompts+val.prompts+['elephant','elephants'])))
    manifest=dict(architecture=model.ARCHITECTURE,seed=20261120,train_names=train.names,val_names=val.names,
        train_vocabulary=train.prompts,val_vocabulary=val.prompts,epochs_per_phase=1 if a.smoke else a.epochs,
        smoke=a.smoke,train_sample_limit=2 if a.smoke else None,val_sample_limit=2 if a.smoke else None,
        curriculum=['point','box'],phase_job=a.phase,initialization='box phase loads best counting-phase model, including prompt/exemplar modules',
        input_shots='minimum available across logical batch, up to3; effective released engine.py override',training_caption='full author training vocabulary; target text and exemplar tokens positive, other tokens negative as author protocol',
        logical_batch=4,loss_normalization='total GT across logical batch; sequential backward with frozen BN',
        box_loss='author total-GT normalization and nearest-exemplar targets',primary_selection='full validation both-modal MAE at .23',
        original_protocol_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in a.protocol_root.iterdir() if f.is_file()},
        retained=['reference ROI projection/ROIAlign logic','within-class text-exemplar self-attention','effective released-code exemplar sampling','two-stage curriculum','reference flip/resize/crop transform bodies','all train class prompts','point/focal and partial-box losses'],
        detector_replacement_differences=['YOLO CNN/CLIP embeddings and neck/head replace Swin/BERT/transformer query decoder','phrase-level CLIP text tokens instead of BERT word tokens','dense YOLO slots instead of 900 deformable queries; no borrowed pretrained query decoder','YOLO RGB/255 and stride32 padding replace detector-specific ImageNet normalization','new class-masked prompt fusion is trained from scratch; not a migrated CountGD weight'],
        legacy_checkpoint_compatible=False,test_used=False,production_promoted=False)
    save(a.output/'manifest.json',manifest)
    frozen={n:t.detach().cpu().clone() for n,t in model.named_parameters() if not t.requires_grad and 'clip_model' not in n}
    # Verify that the cold text-only path still reproduces native YOLO-World.
    with torch.no_grad():
        probe=train.get(train.names[0]);image=probe['image'][None].to(device);prompt=probe['prompt']
        boxes,logits=model(image,prompt)
        native,raw=model.detector.predict(image,txt_feats=model.text_cache[prompt][None,None].to(device))
        scale=image.new_tensor([image.shape[-1],image.shape[-2]]*2)
        torch.testing.assert_close(boxes,native[0,:4].T/scale,atol=1e-5,rtol=1e-4)
        torch.testing.assert_close(logits,raw['scores'][0,0].float(),atol=1e-4,rtol=1e-4)
        del native,raw,image,boxes,logits
    if a.smoke:
        names=train.names[:2];train.names=names;val.names=val.names[:2];a.epochs=1
    else:evaluate(model,val,device,'text',a.output/'baseline_native_text.json')
    phases=['point','box'] if a.phase=='both' else [a.phase]
    if a.phase=='box' and not a.init_counting:raise ValueError('Box phase requires --init-counting; generic pretrained initialization is forbidden')
    for phase in phases:
        folder=a.output/phase;folder.mkdir()
        if phase=='box':
            previous=torch.load(a.init_counting or a.output/'point/best.pth',map_location='cpu',weights_only=False)
            assert previous['phase']=='point' and previous['architecture']==model.ARCHITECTURE
            if previous['manifest'].get('smoke') and not a.smoke:raise ValueError('Smoke checkpoints cannot initialize a real training run')
            model.import_state(previous['model'])
        criterion=CountingCriterion(loc=1.,cls=5.,giou=0.) if phase=='point' else CountingCriterion()
        parameters=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(parameters,lr=1e-4,weight_decay=1e-4)
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer,10,.1);best=float('inf');history=[];gradient_paths=set()
        for epoch in range(1,a.epochs+1):
            model.set_training(True);optimizer.zero_grad(set_to_none=True);names=list(train.names);random.shuffle(names);losses=[]
            samples=[];weights=[]
            for index,name in enumerate(names):
                if index%4==0:
                    samples=[train.get(n,augment=True,box_phase=phase=='box') for n in names[index:index+4]]
                    weights=logical_batch_weights(samples)
                    shot=min(3,min(len(s['exemplar_boxes']) for s in samples))
                sample=samples[index%4]
                ex=sample['exemplar_boxes'][:shot].to(device)
                prompts=list(train.prompts);random.shuffle(prompts)
                target_class=prompts.index(sample['prompt'])
                boxes,logits,class_ids=model.forward_tokens(sample['image'][None].to(device),prompts,ex,
                    exemplar_class=target_class)
                target_mask=class_ids==target_class
                ids=sample['exemplar_indices'][:shot].to(device) if phase=='box' else torch.empty(0,dtype=torch.long,device=device)
                targets=sample['exemplar_targets'][:shot].to(device) if phase=='box' else torch.empty((0,4),device=device)
                loss,_=criterion(boxes,logits,sample['points'].to(device),ids,targets,target_tokens=target_mask)
                if not torch.isfinite(loss):raise FloatingPointError(name)
                (loss*weights[index%4]).backward()
                head_prefix=f'detector.model.{len(model.detector.model)-1}.'
                for key,pattern in [('exemplar_projection','feature_map_proj'),('prompt_fusion','prompt_fuser'),('box_head',head_prefix+'cv2'),('region_text_head',head_prefix+'cv3')]:
                    if key not in gradient_paths and any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if pattern in n):gradient_paths.add(key)
                if (index+1)%4==0 or index+1==len(names):
                    torch.nn.utils.clip_grad_norm_(parameters,.1,error_if_nonfinite=True);optimizer.step();optimizer.zero_grad(set_to_none=True)
                losses.append(float(loss.detach()))
                if (index+1)%100==0:
                    print('train',phase,epoch,index+1,len(names),float(np.mean(losses)),flush=True)
                    save(a.output/'progress.json',dict(phase=phase,epoch=epoch,images=index+1,total=len(names)))
            scheduler.step()
            result=evaluate(model,val,device,'both',folder/f'val_epoch{epoch:02d}.json')
            mae=result['summary']['0.23']['mae'];history.append(dict(epoch=epoch,loss=float(np.mean(losses)),mae=mae))
            save(folder/'history.json',history)
            save_model(folder/'last.pth',model,phase,epoch,mae,manifest,optimizer,scheduler)
            if mae<best:best=mae;save_model(folder/'best.pth',model,phase,epoch,mae,manifest)
            print('epoch',phase,epoch,'mae',mae,'best',best,flush=True)
        if a.smoke:assert gradient_paths=={'exemplar_projection','prompt_fusion','box_head','region_text_head'},gradient_paths
        save(folder/'gradient_paths.json',sorted(gradient_paths))
    last_phase=phases[-1]
    best=torch.load(a.output/last_phase/'best.pth',map_location='cpu',weights_only=False);model.import_state(best['model'])
    for mode in ['text','both','exemplar']:evaluate(model,val,device,mode,a.output/f'final_{mode}.json')
    evaluate(model,val,device,'text',a.output/'final_absent.json',names=['1919.jpg','3436.jpg'],absent=True)
    assert all(torch.equal(t.detach().cpu(),frozen[n]) for n,t in model.named_parameters() if n in frozen)
    save(a.output/'result.json',dict(state='complete',phase=last_phase,best_epoch=best['epoch'],frozen_encoders_unchanged=True,
        smoke=a.smoke,native_text_baseline_parity=True,video_evaluated=False,production_promoted=False))


if __name__=='__main__':main()
