"""CountGD prompt contract adapted to the YOLO-World detector.

CNN/CLIP/YOLO neck-head replace Grounding DINO's internals. Exemplar ROI
projection and class-masked text/exemplar self-attention are explicit modules.
This architecture requires its own counting and box training checkpoints.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import roi_align
from models.yoloworld_counting import YOLOWorldCounting


class PromptFusion(nn.Module):
    def __init__(self,width=512):
        super().__init__()
        layer=nn.TransformerEncoderLayer(width,8,width*4,dropout=0.,activation='gelu',batch_first=True,norm_first=True)
        self.attention=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.residual_scale=nn.Parameter(torch.zeros(()))

    def forward(self,tokens,class_ids):
        # No self-attention between unrelated class phrases and exemplars.
        blocked=class_ids[:,None]!=class_ids[None,:]
        update=self.attention(tokens[None],mask=blocked)[0]
        # Exact identity at zero residual; do not silently renormalize native PE.
        # WorldDetect performs its own contrastive normalization.
        return tokens+self.residual_scale.tanh()*update


class YOLOWorldCountVid(YOLOWorldCounting):
    ARCHITECTURE='yoloworld-countvid-multimodal-v1'

    def __init__(self,weights,device='cuda'):
        super().__init__(weights,device)
        # Obtain frozen CNN feature dimensions without changing detector state.
        with torch.no_grad():
            _,features=self._backbone(torch.zeros((1,3,64,64),device=device))
        channels=sum(features[i].shape[1] for i in [4,6,9])
        self.feature_map_proj=nn.Conv2d(channels,512,1).to(device)
        self.prompt_fuser=PromptFusion().to(device)
        self.set_training(False)

    def set_training(self,training):
        super().set_training(training)
        if hasattr(self,'prompt_fuser'):self.prompt_fuser.train(training)

    @torch.no_grad()
    def _backbone(self,image):
        x=image;features=[]
        for module in self.detector.model[:self.neck_start]:
            if module.f!=-1:
                x=features[module.f] if isinstance(module.f,int) else [x if i==-1 else features[i] for i in module.f]
            x=module(x);features.append(x)
        return x,features

    def _visual_tokens(self,features,boxes):
        # Same multiscale-concat -> 1x1 projection -> ROIAlign(1x1) logic as CountGD.
        size=features[4].shape[-2:]
        combined=torch.cat([F.interpolate(features[i],size=size,mode='bilinear',align_corners=True)
                            if features[i].shape[-2:]!=size else features[i] for i in [4,6,9]],1)
        projected=self.feature_map_proj(combined)
        return F.normalize(roi_align(projected,[boxes],output_size=(1,1),spatial_scale=1/8,aligned=True).flatten(1),dim=-1)

    def forward_tokens(self,image,prompts,exemplar_boxes=None,exemplar_image=None,exemplar_class=0):
        if image.shape[0]!=1:raise ValueError('Adapter uses one image per forward; upstream batches are split explicitly')
        x,features=self._backbone(image)
        tokens=[];ids=[]
        for idx,prompt in enumerate(prompts):
            if prompt is not None:
                if prompt not in self.text_cache:raise KeyError(prompt)
                tokens.append(self.text_cache[prompt].to(image.device));ids.append(idx)
        if exemplar_boxes is not None and len(exemplar_boxes):
            if not 0<=exemplar_class<len(prompts):raise ValueError('Invalid exemplar class')
            if exemplar_image is not None:
                _,reference_features=self._backbone(exemplar_image)
            else:reference_features=features
            visual=self._visual_tokens(reference_features,exemplar_boxes.to(image.device))
            tokens.extend(visual.unbind(0));ids.extend([exemplar_class]*len(visual))
        if not tokens:raise ValueError('Specify text or at least one visual exemplar')
        class_ids=torch.tensor(ids,device=image.device)
        guide=self.prompt_fuser(torch.stack(tokens),class_ids)[None]
        self.head.nc=guide.shape[1]
        from ultralytics.nn.modules import C2fAttn,ImagePoolingAttn,WorldDetect
        for module in self.detector.model[self.neck_start:]:
            if module.f!=-1:
                x=features[module.f] if isinstance(module.f,int) else [x if i==-1 else features[i] for i in module.f]
            if isinstance(module,C2fAttn):x=module(x,guide)
            elif isinstance(module,WorldDetect):x=module(x,guide)
            elif isinstance(module,ImagePoolingAttn):guide=module(x,guide)
            else:x=module(x)
            features.append(x)
        if isinstance(x,tuple):decoded,raw=x
        else:raw=x;decoded=self.head._inference(raw)
        scale=image.new_tensor([image.shape[-1],image.shape[-2]]*2)
        return decoded[0,:4].T/scale,raw['scores'][0].T.float(),class_ids

    def forward(self,image,prompt,exemplar_boxes=None,exemplar_image=None):
        boxes,logits,_=self.forward_tokens(image,[prompt],exemplar_boxes,exemplar_image)
        return boxes,logits.max(-1).values

    def export_state(self):
        return {n:t for n,t in self.state_dict().items() if not n.startswith('detector.clip_model.')}

    def import_state(self,state):
        report=self.load_state_dict(state,strict=False)
        if report.unexpected_keys or any(not n.startswith('detector.clip_model.') for n in report.missing_keys):
            raise RuntimeError('Wrong multimodal architecture checkpoint: '+str(report))
