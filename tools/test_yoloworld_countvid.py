import ast
import os
import json
import tempfile
from types import SimpleNamespace
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from models.yoloworld_countvid import PromptFusion
from models.yoloworld_counting import CountingCriterion
from video.countvid_yoloworld import patch_reference,assert_video_algorithm_unchanged,to_yolo,replacement_count,load_replacement,load_reference_transforms
from datasets.countvid_protocol import CountVidData

torch.set_num_threads(2)
ROOT=Path(__file__).resolve().parents[1]


class CountVidReplacementTests(unittest.TestCase):
    def test_logical_batch_total_gt_normalization_including_empty_images(self):
        from tools.train_yoloworld_countvid import logical_batch_weights
        for counts in ([1,9,0,2],[0,0],[7]):
            parameter=torch.tensor(2.,requires_grad=True)
            numerators=[parameter.square()*(i+1) for i in range(len(counts))]
            weights=logical_batch_weights([dict(points=[None]*n) for n in counts])
            accumulated=sum(loss/max(n,1)*w for loss,n,w in zip(numerators,counts,weights))
            reference=sum(numerators)/max(sum(counts),1)
            torch.testing.assert_close(accumulated,reference)
            torch.testing.assert_close(torch.autograd.grad(accumulated,parameter,retain_graph=True)[0],
                torch.autograd.grad(reference,parameter)[0])

    def test_new_fusion_is_exact_identity_before_training(self):
        fusion=PromptFusion(32);fusion.eval();tokens=torch.randn(3,32)
        self.assertTrue(torch.equal(tokens,fusion(tokens,torch.tensor([0,0,1]))))

    def test_legacy_detector_checkpoint_rejected_before_loading_yolo(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint=Path(directory)/'legacy.pth';torch.save(dict(detector={},epoch=10),checkpoint)
            with self.assertRaisesRegex(ValueError,'multimodal BOX-phase'):
                load_replacement(SimpleNamespace(pretrain_model_path=checkpoint,device='cpu'),'does-not-exist.pt')

    def test_reference_inference_transforms_load_without_package_collision(self):
        reference=ROOT/'outputs/countvid_detect/CountVid/datasets/transforms.py'
        if not reference.exists():self.skipTest('Reference transform file unavailable')
        transforms=load_reference_transforms(reference)
        transform=transforms.Compose([transforms.RandomResize([800],max_size=1333),transforms.ToTensor(),
            transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        image,target=transform(Image.new('RGB',(100,50)),dict(exemplars=torch.empty((0,4))))
        # Preserve the reference's integer rounding (666x1332, not 667x1333).
        self.assertEqual(tuple(image.shape[-2:]),(666,1332));self.assertTrue(torch.isfinite(image).all())

    def test_author_data_vocabulary_and_crop_transform(self):
        data=Path(os.environ.get('FSC147_ROOT',ROOT/'data/FSC147'))
        protocol=Path(os.environ.get('COUNTVID_PROTOCOL_ROOT',ROOT/'kaggle_adapt_cache/thiend27/countvid_protocol'))
        if not data.is_dir() or not protocol.is_dir():self.skipTest('Reference dataset unavailable')
        train=CountVidData(data/'images_384_VarV2',protocol,'train')
        val=CountVidData(data/'images_384_VarV2',protocol,'val')
        self.assertFalse(set(train.names)&set(val.names))
        self.assertIn('package of fresh cut fruit',val.prompts)
        self.assertIn('pepper',train.prompts)
        annotation=json.loads((data/'annotation_FSC147_384.json').read_text())
        self.assertTrue(all(len(row['detection']['instances'])==len(annotation[name]['points']) for name,row in val.records.items()))
        for _ in range(8):
            sample=train.get(train.names[0],augment=True)
            self.assertTrue(torch.isfinite(sample['points']).all())
            self.assertEqual(sample['image'].shape[-1]%32,0)
            self.assertEqual(sample['image'].shape[-2]%32,0)

    def test_reference_video_algorithm_is_unchanged(self):
        source=(ROOT/'outputs/countvid_detect/CountVid/count_in_videos.py').read_text()
        tree=patch_reference(source);assert_video_algorithm_unchanged(source,tree)
        compile(tree,'reference','exec')

    def test_changed_temporal_filter_rejected(self):
        source=(ROOT/'outputs/countvid_detect/CountVid/count_in_videos.py').read_text();tree=patch_reference(source)
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='temporal_filter_fast')
        function.body=[ast.Return(value=ast.Constant(None))]
        with self.assertRaises(AssertionError):assert_video_algorithm_unchanged(source,tree)

    def test_prompt_classes_do_not_cross_attend(self):
        fusion=PromptFusion(32);fusion.residual_scale.data.fill_(.4);fusion.eval()
        x=torch.randn(4,32);ids=torch.tensor([0,0,1,1]);before=fusion(x,ids)
        changed=x.clone();changed[2:]+=10;after=fusion(changed,ids)
        torch.testing.assert_close(before[:2],after[:2])

    def test_other_class_tokens_receive_negative_supervision(self):
        boxes=torch.tensor([[.2,.2,.1,.1],[.8,.8,.1,.1]],requires_grad=True)
        logits=torch.zeros((2,3),requires_grad=True)
        loss,_=CountingCriterion()(boxes,logits,torch.tensor([[.2,.2]]),torch.empty(0,dtype=torch.long),torch.empty((0,4)),torch.tensor([True,False,True]))
        loss.backward();self.assertTrue((logits.grad[:,1]>0).all());self.assertLess(float(logits.grad[0,0]),0);self.assertLess(float(logits.grad[0,2]),0)

    def test_single_token_matches_scalar_criterion(self):
        boxes=torch.tensor([[.2,.2,.1,.1],[.8,.8,.1,.1]]);logits=torch.tensor([1.,-1.]);points=torch.tensor([[.2,.2]])
        args=(points,torch.empty(0,dtype=torch.long),torch.empty((0,4)))
        scalar,_=CountingCriterion()(boxes,logits,*args)
        token,_=CountingCriterion()(boxes,logits[:,None],*args,torch.tensor([True]))
        torch.testing.assert_close(scalar,token)

    def test_normalization_and_padding_roundtrip(self):
        rgb=torch.rand((1,3,51,101));mean=torch.tensor([.485,.456,.406])[None,:,None,None];std=torch.tensor([.229,.224,.225])[None,:,None,None]
        output,size,padded=to_yolo((rgb-mean)/std)
        torch.testing.assert_close(output[:,:,:51,:101],rgb);self.assertEqual(size,(101,51));self.assertEqual(padded,(128,64))

    def test_detector_adapter_preserves_boxes_logits_and_reference(self):
        class Fake:
            def cache_text(self,names):pass
            def set_training(self,training):pass
            def __call__(self,image,prompt,exemplar_boxes,exemplar_image):
                self.reference=exemplar_image
                return torch.tensor([[.5,.5,.2,.2]]),torch.tensor([2.])
        model=Fake();rgb=torch.rand((1,3,51,101));mean=torch.tensor([.485,.456,.406])[None,:,None,None];std=torch.tensor([.229,.224,.225])[None,:,None,None]
        image=(rgb-mean)/std
        output=replacement_count(model,image,image,torch.tensor([[[10.,10.,20.,20.]]]),torch.tensor([[0]]),'bird','cpu')
        torch.testing.assert_close(output['pred_boxes'][0,0],torch.tensor([.5*128/101,.5*64/51,.2*128/101,.2*64/51]))
        self.assertEqual(float(output['pred_logits'][0,0,0]),2.);self.assertIsNotNone(model.reference)

    def test_padding_predictions_cannot_become_video_births(self):
        class Fake:
            def cache_text(self,names):pass
            def set_training(self,training):pass
            def __call__(self,*args,**kwargs):return torch.tensor([[.99,.99,.1,.1]]),torch.tensor([8.])
        output=replacement_count(Fake(),torch.zeros((1,3,51,101)),torch.zeros((1,3,51,101)),torch.empty((1,0)),torch.tensor([[0]]),'bird','cpu')
        self.assertTrue(torch.isneginf(output['pred_logits']).all())


if __name__=='__main__':unittest.main()
