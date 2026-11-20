import unittest
import numpy as np
from PIL import Image
import torch
from models.yoloworld_counting import CountingCriterion, exemplar_point_indices, PROBQueryDecoder
from datasets.yoloworld_counting import prepare, load_fsc147d_class_names
from tools.train_yoloworld_counting import logical_batch_weights, sample_training_prompts, topk_negative_focal_loss


class CountingTests(unittest.TestCase):
    def test_exemplar_association_matches_nearest_l2_even_outside(self):
        points=np.array([[.2,.2],[.8,.8]],dtype=np.float32)
        boxes=np.array([[.1,.1,.3,.3],[.7,.7,.9,.9],[.4,.4,.5,.5]],dtype=np.float32)
        idx,b=exemplar_point_indices(points,boxes)
        self.assertEqual(idx.tolist(),[0,1,0]);self.assertEqual(len(b),3)

    def test_empty_exemplars(self):
        idx,b=exemplar_point_indices(np.array([[.5,.5]]),np.empty((0,4)))
        self.assertEqual(b.shape,(0,4));self.assertEqual(idx.dtype,np.int64)

    def test_partial_box_geometry_normalized_by_all_targets(self):
        boxes=torch.tensor([[.2,.2,.1,.1],[.8,.8,.1,.1]])
        targets=torch.tensor([[.2,.2],[.8,.8]])
        exemplar=torch.tensor([[.1,.1,.3,.3]])
        _,parts=CountingCriterion()(boxes,torch.zeros(2),targets,torch.tensor([0]),exemplar)
        # WH error is .1+.1=.2, normalized by TWO objects, not one exemplar.
        self.assertAlmostEqual(float(parts['wh']),.1,places=6)
        # Predicted area .01, target area .04 -> GIoU loss .75 / two objects.
        self.assertAlmostEqual(float(parts['giou']),.375,places=6)

    def test_exemplar_changes_training_center_without_changing_metric_gt(self):
        record=dict(points=[[20,10]],box_examples_coordinates=[[[15,7],[35,7],[35,17],[15,17]]])
        sample=prepare(Image.new('RGB',(100,50)),record,short=100)
        np.testing.assert_allclose(sample['points'][0],[40/224,20/128],atol=1e-6)
        np.testing.assert_allclose(sample['supervision_points'][0],[50/224,24/128],atol=1e-6)

    def test_point_loss_does_not_invent_wh_targets(self):
        boxes=torch.tensor([[.2,.3,.1,.1],[.8,.7,.2,.2]],requires_grad=True)
        logits=torch.tensor([0.,-1.],requires_grad=True)
        loss,parts=CountingCriterion()(boxes,logits,torch.tensor([[.21,.31]]),torch.empty(0,dtype=torch.long),torch.empty((0,4)))
        loss.backward()
        self.assertTrue(torch.equal(boxes.grad[:,2:],torch.zeros((2,2))))
        self.assertTrue(torch.isfinite(logits.grad).all());self.assertEqual(parts['matched'],1)

    def test_box_loss_reaches_only_annotated_exemplar(self):
        boxes=torch.tensor([[.21,.21,.2,.2],[.8,.8,.1,.1]],requires_grad=True)
        loss,parts=CountingCriterion()(boxes,torch.zeros(2,requires_grad=True),torch.tensor([[.2,.2],[.8,.8]]),
            torch.tensor([0]),torch.tensor([[.05,.05,.35,.35]]))
        loss.backward()
        self.assertGreater(float(boxes.grad[0,2:].abs().sum()),0)
        self.assertEqual(float(boxes.grad[1,2:].abs().sum()),0)

    def test_one_to_one_matching(self):
        boxes=torch.tensor([[.2,.2,.1,.1],[.201,.201,.1,.1],[.8,.8,.1,.1]],requires_grad=True)
        logits=torch.zeros(3,requires_grad=True)
        loss,_=CountingCriterion()(boxes,logits,torch.tensor([[.2,.2],[.8,.8]]),torch.empty(0,dtype=torch.long),torch.empty((0,4)))
        loss.backward()
        self.assertEqual(int((logits.grad<0).sum()),2)
        self.assertEqual(int((logits.grad>0).sum()),1)

    def test_empty_target_is_all_negative(self):
        logits=torch.zeros(3,requires_grad=True)
        loss,_=CountingCriterion()(torch.ones((3,4),requires_grad=True),logits,torch.empty((0,2)),torch.empty(0,dtype=torch.long),torch.empty((0,4)))
        loss.backward();self.assertTrue((logits.grad>0).all())

    def test_reject_target_overflow(self):
        with self.assertRaises(ValueError):
            CountingCriterion()(torch.ones((1,4)),torch.zeros(1),torch.ones((2,2)),torch.empty(0,dtype=torch.long),torch.empty((0,4)))

    def test_matching_is_independent_of_target_order(self):
        boxes=torch.tensor([[.2,.2,.1,.1],[.8,.8,.1,.1]])
        logits=torch.tensor([1.,2.]);points=torch.tensor([[.21,.22],[.81,.82]])
        criterion=CountingCriterion();empty=torch.empty(0,dtype=torch.long);b=torch.empty((0,4))
        first,_=criterion(boxes,logits,points,empty,b)
        second,_=criterion(boxes,logits,points.flip(0),empty,b)
        self.assertAlmostEqual(float(first),float(second),places=6)

    def test_geometry_preserves_point_box_padding_and_flip(self):
        image=Image.new('RGB',(100,50))
        record=dict(points=[[20,10]],box_examples_coordinates=[[[10,5],[30,5],[30,15],[10,15]]])
        sample=prepare(image,record,short=100,flip=True)
        np.testing.assert_allclose(sample['points'][0],[160/224,20/128],atol=1e-6)
        self.assertEqual(sample['image'].shape,(3,128,224))
        self.assertEqual(sample['exemplar_indices'].tolist(),[0])
        np.testing.assert_allclose(sample['exemplar_boxes'][0],[140/224,10/128,180/224,30/128],atol=1e-6)

    def test_random_size_crop_preserves_valid_normalized_coordinates(self):
        image=Image.new('RGB',(600,400))
        pts=[[x,y] for x in range(40,560,40) for y in range(40,360,40)]
        record=dict(points=pts,box_examples_coordinates=[[[70,70],[90,70],[90,90],[70,90]]])
        sample=prepare(image,record,short=640,flip=False,crop=True)
        self.assertEqual(sample['image'].shape[1]%32,0)
        self.assertEqual(sample['image'].shape[2]%32,0)
        self.assertTrue((sample['points']>=0).all() and (sample['points']<=1).all())
        self.assertTrue((sample['supervision_points']>=0).all() and (sample['supervision_points']<=1).all())

    def test_multiclass_prompt_sampling_and_logical_batch_weights(self):
        vocab=['apple','banana','orange','grape','pear']
        prompts,mask=sample_training_prompts('banana',vocab,max_labels=4)
        self.assertEqual(len(prompts),4)
        self.assertEqual(int(mask.sum()),1)
        self.assertEqual(prompts[int(torch.where(mask)[0][0])],'banana')
        counts=[2,6,0,2]
        samples=[dict(supervision_points=torch.zeros((n,2))) for n in counts]
        weights=logical_batch_weights(samples)
        numerators=[10.,30.,5.,20.]
        accumulated=sum((num/max(1,n))*w for num,n,w in zip(numerators,counts,weights))
        self.assertAlmostEqual(accumulated,sum(numerators)/sum(counts),places=6)

    def test_spatial_anchor_gating_prefers_nearby_anchor_over_distant_high_logit(self):
        boxes=torch.tensor([[.2,.2,.1,.1],[.2,.2,.1,.1]],requires_grad=True)
        logits=torch.tensor([2.0,0.0],requires_grad=True)
        anchor_xy=torch.tensor([[.9,.9],[.2,.2]])
        anchor_strides=torch.tensor([.02,.02])
        loss,_=CountingCriterion()(boxes,logits,torch.tensor([[.2,.2]]),
            torch.empty(0,dtype=torch.long),torch.empty((0,4)),
            anchor_xy=anchor_xy,anchor_strides=anchor_strides,max_stride_radius=4.0)
        loss.backward()
        self.assertLess(float(logits.grad[1]),0.0)
        self.assertGreater(float(logits.grad[0]),0.0)

    def test_neighborhood_gaussian_relaxation_reduces_neighbor_negative_penalty(self):
        # Slot 0 is at (.2,.2) matched to GT (.2,.2). Slot 1 is an adjacent neighbor at (.202,.202).
        # Slot 2 is far background at (.8,.8). With neighbor_sigma=0.004, slot 1's negative gradient must be much smaller than slot 2's.
        boxes=torch.tensor([[.2,.2,.1,.1],[.202,.202,.1,.1],[.8,.8,.1,.1]])
        logits1=torch.tensor([0.,0.,0.],requires_grad=True)
        loss,_=CountingCriterion()(boxes,logits1,torch.tensor([[.2,.2]]),
            torch.empty(0,dtype=torch.long),torch.empty((0,4)),neighbor_sigma=0.004)
        loss.backward()
        self.assertLess(float(logits1.grad[1]),float(logits1.grad[2])*0.1)

    def test_topk_negative_focal_loss_penalizes_only_topk(self):
        logits=torch.tensor([-2.0,1.5,-3.0,2.0,-1.0],requires_grad=True)
        loss=topk_negative_focal_loss(logits,k=2)
        loss.backward()
        self.assertGreater(float(logits.grad[1]),0.0)
        self.assertGreater(float(logits.grad[3]),0.0)
        self.assertEqual(float(logits.grad[0]),0.0)
        self.assertEqual(float(logits.grad[2]),0.0)
        self.assertEqual(float(logits.grad[4]),0.0)

    def test_prob_query_decoder_self_attention_and_gradients(self):
        decoder=PROBQueryDecoder(in_channels=(16,32,64),text_dim=32,hidden_dim=32,num_queries=10,num_heads=4,num_layers=2)
        feats=[torch.randn(1,16,8,8,requires_grad=True),torch.randn(1,32,4,4,requires_grad=True),torch.randn(1,64,2,2,requires_grad=True)]
        base_boxes=torch.rand(25,4,requires_grad=True)
        base_logits=torch.randn(25,requires_grad=True)
        text_pe=torch.randn(1,1,32)
        anc_xy=torch.rand(25,2);anc_st=torch.full((25,),0.05)
        exemplars=torch.tensor([[.1,.1,.3,.3],[.6,.6,.8,.8]])
        boxes,logits,q_xy,q_st=decoder(feats,base_boxes,base_logits,text_pe,anc_xy,anc_st,exemplar_boxes=exemplars)
        self.assertEqual(boxes.shape,(10,4))
        self.assertEqual(logits.shape,(10,))
        self.assertEqual(q_xy.shape,(10,2))
        loss,_=CountingCriterion()(boxes,logits,torch.tensor([[.3,.3],[.7,.7]]),
            torch.tensor([0]),torch.tensor([[.2,.2,.4,.4]]),anchor_xy=q_xy,anchor_strides=q_st)
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in decoder.self_attn_layers.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in decoder.ex_proj_p3.parameters()))
        self.assertIsNotNone(decoder.obj_mean.grad)
        # Also verify Self-Exemplar fallback when exemplar_boxes=None
        boxes_se,logits_se,_,_=decoder(feats,base_boxes,base_logits,text_pe,anc_xy,anc_st,exemplar_boxes=None)
        self.assertEqual(boxes_se.shape,(10,4))
        self.assertEqual(logits_se.shape,(10,))


if __name__=='__main__':unittest.main()
