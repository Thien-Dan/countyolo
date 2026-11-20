"""Train a YOLO-World text-only counting detector with CountGD-Box supervision."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torchvision.ops import nms
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from models.yoloworld_counting import YOLOWorldCounting, CountingCriterion, xywh_to_xyxy
from datasets.yoloworld_counting import CountingData

THRESHOLDS = [.01,.05,.14,.17,.20,.23,.26,.35]


def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2),encoding='utf-8');tmp.replace(path)


def checkpoint(path, model, epoch, **extra):
    tmp=path.with_suffix('.tmp')
    state={n:p for n,p in model.detector.state_dict().items() if not n.startswith('clip_model.')}
    qstate=model.query_decoder.state_dict() if model.query_decoder is not None else None
    torch.save(dict(detector=state,query_decoder=qstate,epoch=epoch,clip_encoder_omitted=True,**extra),tmp)
    tmp.replace(path)


def logical_batch_weights(samples):
    """Undo per-image criterion normalization, then normalize by total GT across the logical batch."""
    counts = [len(s['supervision_points']) for s in samples]
    denominator = max(1, sum(counts))
    return [max(1, c) / denominator for c in counts]


def sample_training_prompts(target_prompt, train_vocabulary, max_labels=80):
    """Author odvg.py multi-class prompt sampling: 1 positive + up to (max_labels-1) shuffled negatives."""
    if max_labels <= 1 or len(train_vocabulary) <= 1:
        return [target_prompt], torch.tensor([True])
    negatives = [c for c in train_vocabulary if c != target_prompt]
    k = min(len(negatives), max_labels - 1)
    chosen = [target_prompt] + random.sample(negatives, k)
    random.shuffle(chosen)
    mask = torch.tensor([c == target_prompt for c in chosen], dtype=torch.bool)
    return chosen, mask


def topk_negative_focal_loss(logits, k=32):
    """Penalize only top-K false-positive candidate logits on a negative prompt (normalized by K)."""
    k = min(k, logits.numel())
    if k <= 0:
        return logits.sum() * 0.0
    top_logits = logits.float().view(-1).topk(k).values
    p = top_logits.sigmoid()
    return (0.75 * p.square() * (-torch.nn.functional.logsigmoid(-top_logits))).mean()


def totals(rows):
    out={}
    for method in rows[0]['methods']:
        errors=np.array([r['methods'][method]['count']-r['gt'] for r in rows])
        tp,fp,fn=[sum(r['methods'][method][k] for r in rows) for k in ['tp','fp','fn']]
        out[method]=dict(mae=float(abs(errors).mean()),rmse=float(np.sqrt((errors**2).mean())),
            bias=float(errors.mean()),precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),f1=2*tp/max(2*tp+fp+fn,1))
    if 'nms_count_at023' in rows[0]:
        nms_err=np.array([r['nms_count_at023']-r['gt'] for r in rows])
        out['nms_0.23']=dict(mae=float(abs(nms_err).mean()),rmse=float(np.sqrt((nms_err**2).mean())),bias=float(nms_err.mean()))
    return out


@torch.no_grad()
def evaluate(model, dataset, names, device, output, absent=False, use_exemplars=None):
    model.set_training(False);rows=[]
    if use_exemplars is None:
        use_exemplars = (model.query_decoder is not None) and (not absent)
    for i,name in enumerate(names):
        sample=dataset.get(name)
        prompt='elephant' if absent else sample['prompt']
        ex=sample['exemplar_boxes'].to(device) if (use_exemplars and not absent) else None
        boxes,logits=model(sample['image'][None].to(device),prompt,exemplar_boxes=ex)
        boxes=boxes.float().cpu();scores=logits.sigmoid().cpu()
        pw,ph=sample['padded_size'];nw,nh=sample['resized_size']
        # Undo top-left padding, then map into the same 640x640 metric space.
        scale=boxes.new_tensor([pw/nw*640,ph/nh*640]*2)
        xyxy=xywh_to_xyxy(boxes)*scale
        centers=((xyxy[:,:2]+xyxy[:,2:])/2).numpy()
        valid=((centers>=0)&(centers<=640)).all(-1)
        points=np.empty((0,2)) if absent else sample['points'].numpy()*[pw/nw*640,ph/nh*640]
        methods={}
        for threshold in THRESHOLDS:
            mask=(scores.numpy()>threshold)&valid
            chosen_centers=centers[mask]
            tp=int((maximum_bipartite_matching(csr_matrix(cdist(chosen_centers,points)<=16),perm_type='column')>=0).sum()) if len(chosen_centers) and len(points) else 0
            methods[str(threshold)]=dict(count=len(chosen_centers),tp=tp,fp=len(chosen_centers)-tp,fn=len(points)-tp)
        keep=(scores>.23)&torch.from_numpy(valid)
        nms_count=len(nms(xyxy[keep],scores[keep],.7)) if keep.any() else 0
        rows.append(dict(image=name,prompt=prompt,gt=len(points),methods=methods,nms_count_at023=nms_count))
        if (i+1)%100==0:
            print('eval',output.name,i+1,len(names),flush=True)
    value=dict(n=len(rows),summary=totals(rows),rows=rows)
    save(output,value)
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--accumulate',type=int,default=4)
    p.add_argument('--max-labels',type=int,default=1,help='Prompts per forward (keep 1 to preserve C2fAttn MaxSigmoidAttnBlock distribution)')
    p.add_argument('--neg-weight',type=float,default=0.15,help='Weight of top-K hard-negative focal loss on 1 negative prompt per accumulation batch')
    p.add_argument('--neg-topk',type=int,default=32,help='Number of highest-scoring slots penalized on a negative prompt pass')
    p.add_argument('--neighbor-sigma',type=float,default=0.004,help='Gaussian negative relaxation sigma around GT centers in normalized coords (~3.2px at 800)')
    p.add_argument('--neck-lr',type=float,default=2e-5,help='Learning rate for neck layers (C2fAttn/ImagePoolingAttn); head uses 1e-4')
    p.add_argument('--stride-radius',type=float,default=4.0,help='Spatial anchor gating radius in multiples of FPN stride (0 disables)')
    p.add_argument('--query-decoder',action='store_true',help='Attach NMS-free Self-Attention Query Decoder + PROB Objectness Head (Direction 1)')
    p.add_argument('--num-queries',type=int,default=2200,help='Number of top language-guided queries in PROBQueryDecoder (>=1912 to cover max FSC147 train count)')
    p.add_argument('--random-crop',action='store_true',help='Enable RandomSizeCrop(384,600); off by default to preserve 100% exemplar boxes')
    p.add_argument('--raw-prompts',action='store_true',help='Use raw ImageClasses_FSC147.txt instead of FSC147-D singularized prompts')
    p.add_argument('--dev-per-class',type=int,default=8)
    p.add_argument('--pilot-images',type=Path)
    p.add_argument('--exclude-train',type=Path)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--integration-check',action='store_true',help='Two-image train/save/load/evaluate check; not a benchmark')
    p.add_argument('--resume',type=Path,help='last.pth with optimizer/scheduler; sibling history/best/baselines required')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'manifest.json').exists():raise RuntimeError('Use a fresh output directory')
    random.seed(20261119);np.random.seed(20261119);torch.manual_seed(20261119)
    torch.set_num_threads(4)
    excluded=json.loads(a.exclude_train.read_text()) if a.exclude_train else []
    train=CountingData(a.data_root,'train',excluded,use_fsc147d_prompts=not a.raw_prompts,random_crop=a.random_crop)
    val=CountingData(a.data_root,'val',use_fsc147d_prompts=not a.raw_prompts,random_crop=False)
    pilot=json.loads(a.pilot_images.read_text()) if a.pilot_images else []
    assert not set(train.names)&set(val.names)
    assert not {train.classes[n] for n in train.names}&{val.classes[n] for n in val.names}
    if a.integration_check:
        train.names=train.names[:2];val.names=val.names[:2];pilot=[]
        if a.epochs==30:a.epochs=2
    groups={}
    for n in val.names:
        if n not in pilot:groups.setdefault(val.classes[n],[]).append(n)
    rng=random.Random(5199);dev=[]
    for c in sorted(groups):
        group=sorted(groups[c]);rng.shuffle(group);dev.extend(group[:a.dev_per_class])
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=YOLOWorldCounting(a.weights,device,use_query_decoder=a.query_decoder,num_queries=a.num_queries)
    train_vocab=sorted({train.classes[n] for n in train.names})
    val_vocab=sorted({val.classes[n] for n in val.names})
    model.cache_text(sorted(set(train_vocab+val_vocab+['elephant','elephants'])))
    if any(p.requires_grad for layer in model.detector.model[:model.neck_start] for p in layer.parameters()):
        raise AssertionError('Backbone must be frozen')
    trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    assert any('cv2' in n for n,p in trainable) and any('cv3' in n for n,p in trainable)
    assert not any(p.requires_grad for p in model.head.dfl.parameters())
    frozen={n:p.detach().cpu().clone() for n,p in model.named_parameters() if not p.requires_grad and 'clip_model' not in n}
    manifest=dict(variant='YOLO-World CountGD-Box-style text-only v4 (nc=1 + Neighborhood Gaussian Relaxation + TopK Contrastive + Spatial Anchor Gating + Balanced Accum4 + FSC147-D)',epochs=a.epochs,
        integration_check=a.integration_check,
        optimizer=f'AdamW head_lr=1e-4 neck_lr={a.neck_lr} weight_decay=1e-4, StepLR10 gamma.1',accumulate=a.accumulate,
        max_labels=a.max_labels,neg_weight=a.neg_weight,neg_topk=a.neg_topk,neighbor_sigma=a.neighbor_sigma,
        stride_radius=a.stride_radius,random_crop=a.random_crop,fsc147d_prompts=not a.raw_prompts,
        train_images=train.names,excluded_train=excluded,dev_images=dev,pilot_images=pilot,
        train_classes=train_vocab,val_classes=sorted(groups),
        primary_threshold=.23,thresholds=THRESHOLDS,selection='best dev MAE at fixed .23; pilot not used for selection',
        neck_start=model.neck_start,trainable_parameters=sum(p.numel() for n,p in trainable),
        trainable_names=[n for n,p in trainable],weights_sha256=hashlib.sha256(a.weights.read_bytes()).hexdigest(),
        loss=dict(center_and_exemplar_wh=5,exemplar_giou=2,focal_classification=2,alpha=.25,gamma=2,
            neighbor_sigma=a.neighbor_sigma,neg_topk=a.neg_topk,neg_weight=a.neg_weight),
        geometry_normalizer='total number of GT targets per image; balanced 1/accumulate gradient averaging across 4 images',
        inference='One-to-one set predictions counted before NMS, as CountGD; .7 NMS diagnostic also recorded',
        differences_from_paper=['YOLO CNN/CLIP neck/head replaces Swin/BERT transformer fusion/decoder',
            'Text-only, no exemplar input tokens; true exemplar boxes supervise WH/GIoU only',
            'Preserves nc=1 in C2fAttn MaxSigmoidAttnBlock; top-K hard-negative contrastive penalty on 1 image per accumulation group',
            'CenterNet-style Gaussian negative relaxation around GT points prevents adjacent 3x3 grid anchors from suppressing the matched center',
            'Spatial anchor radius gating in Hungarian matcher to respect bounded DFL grid reach',
            'YOLO RGB/255 and top-left stride32 padding; native CountGD ImageNet normalization not appropriate for YOLO',
            'Aspect-preserving multiscale 480..800 + hflip (100% exemplar box retention)',
            'Nearest-L2 exemplar association and corrected training centers follow original odvg.py; metric GT points unchanged; image7171 boxes disabled',
            'BatchNorm statistics frozen with batch1; balanced gradient accumulation4; no auxiliary decoder losses'],
        initialization_difference='Generic YOLO-World checkpoint; author CountGD-Box starts from an already counting-adapted CountGD checkpoint. This run is a joint adaptation trial, not the same two-stage curriculum.',
        papers=['https://arxiv.org/html/2407.04619v2#S3.SS3','https://arxiv.org/html/2506.15368v2#S3.SS2'],
        training_data_only=True,test_used=False,production_defaults_changed=False)
    save(a.output/'manifest.json',manifest)
    head_prefix=f'detector.model.{len(model.detector.model)-1}.'
    neck_params=[p for n,p in trainable if not n.startswith(head_prefix) and not n.startswith('query_decoder.')]
    head_params=[p for n,p in trainable if n.startswith(head_prefix) or n.startswith('query_decoder.')]
    assert neck_params and head_params
    criterion=CountingCriterion()
    optimizer=torch.optim.AdamW([
        dict(params=neck_params,lr=a.neck_lr),
        dict(params=head_params,lr=1e-4)
    ],lr=1e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=.1)
    if a.smoke:
        names=train.names[:2]
        before={n:p.detach().clone() for n,p in trainable}
        model.set_training(True);optimizer.zero_grad()
        for idx_s,name in enumerate(names):
            sample=train.get(name,augment=True)
            img_t=sample['image'][None].to(device)
            ex_in=sample['exemplar_boxes'].to(device) if (model.query_decoder is not None and idx_s%2==0) else None
            boxes,logits=model(img_t,sample['prompt'],exemplar_boxes=ex_in)
            anc_xy,anc_st=model.anchor_priors(img_t)
            loss,parts=criterion(boxes,logits,*[sample[k].to(device) for k in ['supervision_points','exemplar_indices','exemplar_boxes']],
                anchor_xy=anc_xy,anchor_strides=anc_st,max_stride_radius=a.stride_radius,neighbor_sigma=a.neighbor_sigma)
            if a.neg_weight>0 and idx_s==0 and len(train_vocab)>1:
                neg_prompt=random.choice([c for c in train_vocab if c!=sample['prompt']])
                _,neg_logits=model(img_t,neg_prompt,exemplar_boxes=None)
                loss=loss+a.neg_weight*topk_negative_focal_loss(neg_logits,a.neg_topk)
            assert torch.isfinite(loss)
            (loss/len(names)).backward()
        finite=all(p.grad is None or torch.isfinite(p.grad).all() for n,p in trainable)
        assert finite
        assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in trainable if 'cv2' in n)
        assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in trainable if 'cv3' in n)
        if model.query_decoder is not None:
            assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in trainable if n.startswith('query_decoder.'))
        torch.nn.utils.clip_grad_norm_([p for n,p in trainable],.1);optimizer.step()
        assert any(not torch.equal(before[n],p) for n,p in trainable)
        assert all(torch.equal(frozen[n],p.cpu()) for n,p in model.named_parameters() if n in frozen)
        model.set_training(False)
        with torch.no_grad():boxes,logits=model(sample['image'][None].to(device),sample['prompt'])
        assert boxes.shape[1]==4 and torch.isfinite(logits).all()
        # Reproduce the reported eval->train cache transition at the SAME shape.
        with torch.inference_mode():
            model.head.shape=None
            model(sample['image'][None].to(device),sample['prompt'])
        assert torch.is_inference(model.head.anchors)
        model.set_training(True);optimizer.zero_grad(set_to_none=True)
        img_t=sample['image'][None].to(device)
        boxes,logits=model(img_t,sample['prompt'])
        assert not torch.is_inference(model.head.anchors)
        anc_xy,anc_st=model.anchor_priors(img_t)
        transition_loss,_=criterion(boxes,logits,*[sample[k].to(device) for k in ['supervision_points','exemplar_indices','exemplar_boxes']],
            anchor_xy=anc_xy,anchor_strides=anc_st,max_stride_radius=a.stride_radius,neighbor_sigma=a.neighbor_sigma)
        transition_loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for n,p in trainable)
        save(a.output/'smoke.json',dict(passed=True,finite_gradients=True,frozen_backbone_unchanged=True,
            box_and_text_heads_receive_gradients=True,eval_inference_to_train_same_shape_passed=True,loss=float(loss.detach()),candidates=len(boxes),nc=1,topk_negative=a.neg_topk))
        print('smoke passed',float(loss.detach()),flush=True);return
    history=[];best=float('inf');steps=0;start_epoch=0
    if a.resume:
        saved=torch.load(a.resume,map_location='cpu',weights_only=False)
        for key in ['train_images','excluded_train','dev_images','trainable_names','loss','weights_sha256','optimizer']:
            if saved['manifest'][key]!=manifest[key]:raise ValueError('Resume protocol mismatch: '+key)
        loading=model.detector.load_state_dict(saved['detector'],strict=False)
        assert not loading.unexpected_keys and all(n.startswith('clip_model.') for n in loading.missing_keys)
        if model.query_decoder is not None and saved.get('query_decoder') is not None:
            model.query_decoder.load_state_dict(saved['query_decoder'])
        optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        start_epoch=int(saved['epoch'])
        if not 0<start_epoch<a.epochs:raise ValueError('Resume epoch must precede target epoch')
        for name in ['history.json','best.pth','baseline_dev.json','baseline_fullval.json']:
            shutil.copyfile(a.resume.parent/name,a.output/name)
        for path in a.resume.parent.glob('dev_epoch*.json'):shutil.copyfile(path,a.output/path.name)
        history=json.loads((a.output/'history.json').read_text())
        assert history[-1]['epoch']==start_epoch
        best_state=torch.load(a.output/'best.pth',map_location='cpu',weights_only=False)
        best=float(best_state['dev_mae']);del best_state
        steps=start_epoch*((len(train.names)+a.accumulate-1)//a.accumulate)
        rng_restored='rng_python' in saved
        if rng_restored:
            random.setstate(saved['rng_python']);np.random.set_state(saved['rng_numpy']);torch.set_rng_state(saved['rng_torch'])
            if device=='cuda' and saved.get('rng_cuda') is not None:torch.cuda.set_rng_state_all(saved['rng_cuda'])
        save(a.output/'resume_manifest.json',dict(source=str(a.resume),checkpoint_sha256=hashlib.sha256(a.resume.read_bytes()).hexdigest(),
            start_epoch=start_epoch,optimizer_restored=True,scheduler_restored=True,lr=optimizer.param_groups[0]['lr'],
            rng_restored=rng_restored,note='Old checkpoint lacked RNG state; resumed data order is reseeded' if not rng_restored else 'RNG restored'))
        evaluate(model,val,dev,device,a.output/'resume_dev.json')
        evaluate(model,val,val.names,device,a.output/f'resume_epoch{start_epoch:02d}_fullval.json')
        if pilot:evaluate(model,val,pilot,device,a.output/f'resume_epoch{start_epoch:02d}_pilot58.json')
        del saved
    else:
        evaluate(model,val,dev,device,a.output/'baseline_dev.json')
        evaluate(model,val,val.names,device,a.output/'baseline_fullval.json')
    for epoch in range(start_epoch+1,a.epochs+1):
        model.set_training(True);optimizer.zero_grad(set_to_none=True)
        names=list(train.names);random.shuffle(names);losses=[];part_sums={k:0. for k in ['center','wh','giou','cls']}
        coverage=dict(points=0,annotated_exemplars=0,matched_exemplars=0)
        for i,name in enumerate(names):
            sample=train.get(name,augment=True)
            img_t=sample['image'][None].to(device)
            # 70% GT exemplar boxes + 30% Self-Exemplar (None) when PROBQueryDecoder is active
            ex_in=sample['exemplar_boxes'].to(device) if (model.query_decoder is not None and random.random()<0.7) else None
            boxes,logits=model(img_t,sample['prompt'],exemplar_boxes=ex_in)
            anc_xy,anc_st=model.anchor_priors(img_t)
            loss,parts=criterion(boxes,logits,*[sample[k].to(device) for k in ['supervision_points','exemplar_indices','exemplar_boxes']],
                anchor_xy=anc_xy,anchor_strides=anc_st,max_stride_radius=a.stride_radius,neighbor_sigma=a.neighbor_sigma)
            if a.neg_weight>0 and (i%a.accumulate==0) and len(train_vocab)>1:
                neg_prompt=random.choice([c for c in train_vocab if c!=sample['prompt']])
                _,neg_logits=model(img_t,neg_prompt,exemplar_boxes=None)
                loss=loss+a.neg_weight*topk_negative_focal_loss(neg_logits,a.neg_topk)
            if not torch.isfinite(loss):raise FloatingPointError(f'Non-finite loss {name}')
            group_start=(i//a.accumulate)*a.accumulate
            group_size=min(a.accumulate,len(names)-group_start)
            (loss/group_size).backward()
            if (i+1)%a.accumulate==0 or i+1==len(names):
                norm=torch.nn.utils.clip_grad_norm_([p for n,p in trainable],.1,error_if_nonfinite=True)
                optimizer.step();optimizer.zero_grad(set_to_none=True);steps+=1
            losses.append(float(loss.detach()))
            coverage['points']+=len(sample['points'])
            coverage['annotated_exemplars']+=sample['raw_exemplar_count']
            coverage['matched_exemplars']+=len(sample['exemplar_indices'])
            for k in part_sums:part_sums[k]+=float(parts[k])
            if (i+1)%100==0:
                save(a.output/'progress.json',dict(epoch=epoch,images=i+1,total=len(names),optimizer_steps=steps,mean_loss=float(np.mean(losses))))
                print('train',epoch,i+1,len(names),'loss',round(float(np.mean(losses)),4),flush=True)
        scheduler.step()
        result=evaluate(model,val,dev,device,a.output/f'dev_epoch{epoch:02d}.json')
        mae=result['summary']['0.23']['mae']
        nms_mae=result['summary'].get('nms_0.23',{}).get('mae',mae)
        best_th_mae=min(result['summary'][str(t)]['mae'] for t in THRESHOLDS)
        history.append(dict(epoch=epoch,loss=float(np.mean(losses)),parts={k:v/len(names) for k,v in part_sums.items()},coverage=coverage,
            dev_mae=mae,dev_nms_mae=nms_mae,dev_best_th_mae=best_th_mae,lr=optimizer.param_groups[-1]['lr']))
        save(a.output/'history.json',history)
        checkpoint(a.output/'last.pth',model,epoch,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),manifest=manifest,
            rng_python=random.getstate(),rng_numpy=np.random.get_state(),rng_torch=torch.get_rng_state(),
            rng_cuda=torch.cuda.get_rng_state_all() if device=='cuda' else None)
        if mae<best:
            best=mae;checkpoint(a.output/'best.pth',model,epoch,dev_mae=best,manifest=manifest)
        print('epoch',epoch,'dev_mae',round(mae,3),'nms_mae',round(nms_mae,3),'best_th_mae',round(best_th_mae,3),'best',round(best,3),flush=True)
    best_state=torch.load(a.output/'best.pth',map_location='cpu',weights_only=False)
    loading=model.detector.load_state_dict(best_state['detector'],strict=False)
    assert not loading.unexpected_keys
    assert all(n.startswith('clip_model.') for n in loading.missing_keys)
    if model.query_decoder is not None and best_state.get('query_decoder') is not None:
        model.query_decoder.load_state_dict(best_state['query_decoder'])
    final=evaluate(model,val,val.names,device,a.output/'adapted_fullval.json')
    if model.query_decoder is not None:
        evaluate(model,val,val.names,device,a.output/'adapted_fullval_selfex.json',use_exemplars=False)
    if pilot:evaluate(model,val,pilot,device,a.output/'adapted_pilot58.json')
    evaluate(model,val,['1919.jpg','3436.jpg'],device,a.output/'adapted_absent.json',absent=True,use_exemplars=False)
    assert all(torch.equal(frozen[n],p.detach().cpu()) for n,p in model.named_parameters() if n in frozen)
    save(a.output/'result.json',dict(state='complete',best_epoch=best_state['epoch'],best_dev_mae=best,
        final_fullval=final['summary'],frozen_backbone_unchanged=True,production_promoted=False,test_used=False))


if __name__=='__main__':main()
