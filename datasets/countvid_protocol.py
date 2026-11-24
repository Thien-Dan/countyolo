"""Read the author's CountGD-Box annotation files and augmentation functions."""
import ast
import json
from pathlib import Path
import random
import numpy as np
import PIL
from PIL import Image
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torchvision.ops import box_convert
from models.yoloworld_counting import exemplar_point_indices


def author_transforms(source):
    """Use reference transform bodies unchanged; only final normalization differs."""
    names={'crop','hflip','resize','RandomHorizontalFlip','RandomResize','RandomSizeCrop','RandomSelect','Compose'}
    tree=ast.parse(Path(source).read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes}!=names:raise RuntimeError('Unexpected reference transform version')
    namespace=dict(torch=torch,np=np,random=random,T=T,F=F,PIL=PIL,interpolate=torch.nn.functional.interpolate)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
    scales=[480,512,544,576,608,640,672,704,736,768,800]
    return namespace['Compose']([namespace['RandomHorizontalFlip'](),namespace['RandomSelect'](
        namespace['RandomResize'](scales,max_size=1333),namespace['Compose']([
            namespace['RandomResize']([400,500,600]),namespace['RandomSizeCrop'](384,600),
            namespace['RandomResize'](scales,max_size=1333)]))]),namespace['RandomResize']([800],max_size=1333)


class CountVidData:
    def __init__(self,image_root,protocol_root,split):
        self.image_root=Path(image_root);self.protocol_root=Path(protocol_root);self.split=split
        self.train_transform,self.eval_transform=author_transforms(self.protocol_root/'transforms.py')
        self.records={}
        if split=='train':
            self.vocabulary=json.loads((self.protocol_root/'train_label_map.json').read_text())
            with (self.protocol_root/'train.jsonl').open() as stream:
                for line in stream:
                    row=json.loads(line);self.records[row['filename']]=row
        elif split=='val':
            coco=json.loads((self.protocol_root/'val.json').read_text())
            self.vocabulary={str(c['id']):c['name'] for c in coco['categories']}
            annotations={}
            for row in coco['annotations']:annotations.setdefault(row['image_id'],[]).append(row)
            for row in coco['images']:
                instances=[]
                annotation=annotations[row['id']]
                # Author coco.py appends three ROI boxes after the point objects.
                # They are prompt metadata, NOT three additional counted objects.
                if len(annotation)<3 or sum(a['area']!=4 for a in annotation)!=3:
                    raise ValueError('Unexpected original COCO exemplar layout')
                for a in annotation[:-3]:
                    x,y,w,h=a['bbox'];instances.append(dict(bbox=[x,y,x+w,y+h],label=a['category_id']))
                ex=[]
                for a in annotation[-3:]:
                    x,y,w,h=a['bbox'];ex.append([x,y,x+w,y+h])
                self.records[row['file_name']]=dict(filename=row['file_name'],detection=dict(instances=instances),exemplars=ex)
        else:raise ValueError('Only train/val are supported; test is not used')
        self.names=list(self.records)
        self.prompts=[self.vocabulary[k] for k in sorted(self.vocabulary,key=int)]

    def get(self,name,augment=False,box_phase=True):
        row=self.records[name];image=Image.open(self.image_root/name).convert('RGB')
        instances=row['detection']['instances']
        ids={int(a['label']) for a in instances}
        if len(ids)!=1:raise ValueError('This FSC147 adapter expects one annotated category per image')
        category=next(iter(ids));prompt=self.vocabulary[str(category)]
        boxes=torch.tensor([a['bbox'] for a in instances],dtype=torch.float32).reshape(-1,4)
        ex=torch.tensor(row.get('exemplars',[]),dtype=torch.float32).reshape(-1,4)
        if self.split=='val':
            width,height=image.size
            boxes[:,0::2].clamp_(0,width);boxes[:,1::2].clamp_(0,height)
            boxes=boxes[((boxes[:,2:]-boxes[:,:2])>0).all(-1)]
            ex[:,0::2].clamp_(0,width);ex[:,1::2].clamp_(0,height)
        if name=='7171.jpg':ex=ex[:0]
        # Author odvg.py: replace training targets with their true exemplar boxes.
        if augment and box_phase and len(boxes) and len(ex):
            original_centers=(boxes[:,:2]+boxes[:,2:])/2
            for exemplar in ex:
                nearest=((original_centers-(exemplar[:2]+exemplar[2:])/2)**2).sum(-1).argmin()
                boxes[nearest]=exemplar
        target=dict(boxes=boxes,exemplars=ex,labels=torch.full((len(boxes),),category,dtype=torch.long),
                    area=(boxes[:,2:]-boxes[:,:2]).prod(-1),iscrowd=torch.zeros(len(boxes),dtype=torch.long))
        image,target=(self.train_transform if augment else self.eval_transform)(image,target)
        w,h=image.size;pw,ph=(w+31)//32*32,(h+31)//32*32
        canvas=np.full((ph,pw,3),114,dtype=np.uint8);canvas[:h,:w]=np.asarray(image)
        points=(target['boxes'][:,:2]+target['boxes'][:,2:])/2/torch.tensor([pw,ph])
        ex=target['exemplars'];normalized=ex/torch.tensor([pw,ph,pw,ph])
        idx,ex_targets=exemplar_point_indices(points.numpy(),normalized.numpy())
        return dict(image=torch.from_numpy(canvas.copy()).permute(2,0,1).float()/255,
            points=points,exemplar_boxes=ex,exemplar_indices=torch.from_numpy(idx),
            exemplar_targets=torch.from_numpy(ex_targets),prompt=prompt,category=category,
            resized_size=(w,h),padded_size=(pw,ph),name=name)
