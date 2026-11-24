"""Replace only CountVid's detector functions; execute its video algorithm unchanged."""
import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import random
import torch
from torch.nn import functional as F
from torchvision import transforms as TV
from torchvision.transforms import functional as TVF
from torchvision.ops import box_convert

REPLACED_FUNCTIONS={'build_model_and_transforms','count'}
REFERENCE_SHA256='42f940d0625f37488209f8ad306e9a261a21eb195697fc4d8ee7c307cac0efcd'
SAM2_SHA256='2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318'
REFERENCE_TRANSFORMS_SHA256='ff3340a91173bbda058cbe809f33c7b28398ffe511d49fb88829942c04c59087'


def patch_reference(source):
    tree=ast.parse(source)
    replacement=ast.parse('''
def build_model_and_transforms(args):
    normalize=T.Compose([T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    transform=T.Compose([T.RandomResize([800],max_size=1333),normalize])
    return _load_replacement(args),transform
def count(input_image,input_image_exemplars,exemplars,label,text,device='cuda'):
    return _replacement_count(model,input_image,input_image_exemplars,exemplars,label,text,device)
''')
    functions={n.name:n for n in replacement.body}
    seen=set()
    for index,node in enumerate(tree.body):
        if isinstance(node,ast.FunctionDef) and node.name in REPLACED_FUNCTIONS:
            seen.add(node.name);tree.body[index]=functions[node.name]
        elif isinstance(node,ast.Import) and any(a.name=='datasets.transforms' for a in node.names):
            if len(node.names)!=1:raise RuntimeError('Unexpected upstream transform import')
            # Avoid Python package-name collisions; T is the SAME upstream code.
            tree.body[index]=ast.Pass()
    if seen!=REPLACED_FUNCTIONS:raise RuntimeError('Reference detector API changed')
    return ast.fix_missing_locations(tree)


def assert_video_algorithm_unchanged(source,tree):
    original=ast.parse(source)
    expected=patch_reference(source)
    if ast.dump(expected,include_attributes=False)!=ast.dump(tree,include_attributes=False):
        raise AssertionError('Unexpected change outside detector boundary')
    for before,after in zip(original.body,tree.body):
        allowed=(isinstance(before,ast.FunctionDef) and before.name in REPLACED_FUNCTIONS) or (
            isinstance(before,ast.Import) and any(a.name=='datasets.transforms' for a in before.names))
        if not allowed and ast.dump(before,include_attributes=False)!=ast.dump(after,include_attributes=False):
            raise AssertionError('Upstream video algorithm changed')


def load_reference_transforms(path):
    names={'crop','hflip','resize','RandomResize','ToTensor','Normalize','Compose'}
    source=Path(path).read_text()
    if hashlib.sha256(source.encode()).hexdigest()!=REFERENCE_TRANSFORMS_SHA256:
        raise RuntimeError('Use the pinned upstream CountVid inference transforms')
    tree=ast.parse(source)
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes}!=names:raise RuntimeError('Reference transform API changed')
    scope=dict(torch=torch,np=np,random=random,T=TV,F=TVF,interpolate=F.interpolate,
        box_xyxy_to_cxcywh=lambda boxes:box_convert(boxes,'xyxy','cxcywh'))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
    return SimpleNamespace(**{name:scope[name] for name in names})


def to_yolo(tensor):
    """Invert only CountVid's detector normalization; retain its resized pixels."""
    mean=tensor.new_tensor([.485,.456,.406])[None,:,None,None]
    std=tensor.new_tensor([.229,.224,.225])[None,:,None,None]
    rgb=(tensor.float()*std+mean).clamp(0,1)
    h,w=rgb.shape[-2:];ph,pw=(h+31)//32*32,(w+31)//32*32
    return F.pad(rgb,(0,pw-w,0,ph-h),value=114/255),(w,h),(pw,ph)


@torch.no_grad()
def replacement_count(model,images,references,exemplars,labels,text,device='cuda'):
    prompt=text.strip().rstrip(' .') or None
    boxes_out=[];logits_out=[]
    with torch.autocast(torch.device(device).type,enabled=False):
        if prompt is not None:model.cache_text([prompt])
        model.set_training(False)
        for b in range(len(images)):
            if (labels[b]!=0).any():raise ValueError('CountVid adapter expects class index0 for its single requested category')
            image,size,padded=to_yolo(images[b:b+1].to(device))
            ex=exemplars[b].reshape(-1,4).to(device)
            reference=None
            if len(ex):reference,_,_=to_yolo(references[b:b+1].to(device))
            boxes,logits=model(image,prompt,exemplar_boxes=ex,exemplar_image=reference)
            # Return normalized cxcywh against VALID image dimensions, as upstream.
            boxes=boxes*boxes.new_tensor([padded[0]/size[0],padded[1]/size[1]]*2)
            valid=((boxes[:,:2]>=0)&(boxes[:,:2]<=1)).all(-1)&(boxes[:,2:]>0).all(-1)
            logits=logits.masked_fill(~valid,-torch.inf)
            boxes_out.append(boxes);logits_out.append(logits[:,None])
    return dict(pred_boxes=torch.stack(boxes_out),pred_logits=torch.stack(logits_out))


def load_replacement(args,yolo_weights):
    from models.yoloworld_countvid import YOLOWorldCountVid
    state=torch.load(args.pretrain_model_path,map_location='cpu',weights_only=False)
    if state.get('architecture')!=YOLOWorldCountVid.ARCHITECTURE or state.get('phase')!='box' or state.get('manifest',{}).get('smoke'):
        raise ValueError('Require a trained multimodal BOX-phase checkpoint; legacy text-only checkpoints are incompatible')
    model=YOLOWorldCountVid(yolo_weights,args.device);model.import_state(state['model']);return model
