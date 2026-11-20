"""YOLO-World adaptation of CountGD-Box's text-only point/partial-box loss.

This is an architectural adaptation, not a reproduction of CountGD's decoder.
No density head, DINO features, pseudo boxes, or exemplar input tokens are used.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou


def xywh_to_xyxy(boxes):
    return torch.cat((boxes[..., :2]-boxes[..., 2:]/2,
                      boxes[..., :2]+boxes[..., 2:]/2), -1)


def exemplar_point_indices(points, boxes):
    """Author's nearest-L2 target association; exemplars need not be unique."""
    if not len(points) or not len(boxes):
        return np.empty(0, dtype=np.int64), np.empty((0, 4), dtype=np.float32)
    centers = (boxes[:, :2]+boxes[:, 2:])/2
    if not np.isfinite(boxes).all() or not ((boxes[:,2:]-boxes[:,:2])>0).all():
        raise ValueError('Invalid exemplar box geometry')
    distances = ((points[None]-centers[:,None])**2).sum(-1)
    return distances.argmin(axis=1).astype(np.int64), boxes.copy()


class CountingCriterion(nn.Module):
    """Hungarian points + focal classification + exemplar-only WH/GIoU.

    Loss weights (5,2,2) follow CountVid section C.1. No full-box labels are
    invented for points. Matching excludes WH/GIoU because most targets lack it.
    """
    def __init__(self, loc=5., giou=2., cls=2.):
        super().__init__()
        self.loc, self.giou, self.cls = loc, giou, cls

    def forward(self, boxes, logits, points, exemplar_indices, exemplar_xyxy, target_tokens=None,
                anchor_xy=None, anchor_strides=None, max_stride_radius=4.0, neighbor_sigma=0.0):
        assert boxes.ndim == 2 and boxes.shape[1] == 4
        assert logits.shape[0] == boxes.shape[0] and logits.ndim in (1,2)
        if logits.ndim==2:
            if target_tokens is None or target_tokens.shape!=(logits.shape[1],) or not target_tokens.any():
                raise ValueError('Multimodal loss requires a nonempty target-token mask')
        if len(points) > len(boxes):
            raise ValueError(f'More targets ({len(points)}) than candidate slots ({len(boxes)}); never silently drop GT')
        p = logits.float().sigmoid()
        positive = .25*(1-p).square()*(-F.logsigmoid(logits.float()))
        negative = .75*p.square()*(-F.logsigmoid(-logits.float()))
        with torch.no_grad():
            cost = self.loc*torch.cdist(boxes[:, :2].float(), points.float(), p=1)
            class_cost=(positive-negative)
            if logits.ndim==2:class_cost=class_cost[:,target_tokens].mean(-1)
            cost = cost + self.cls*class_cost[:, None]
            if anchor_xy is not None and anchor_strides is not None and len(points) and max_stride_radius > 0:
                # YOLO grid anchors have bounded DFL reach; penalize assignments outside max_stride_radius * stride.
                grid_dist = torch.cdist(anchor_xy.float(), points.float(), p=2)
                outside = grid_dist > (anchor_strides.float() * max_stride_radius)[:, None]
                cost = cost + outside.float() * (100.0 + 10.0 * grid_dist)
            pred_ids, target_ids = linear_sum_assignment(cost.detach().cpu().numpy())
            pred_ids = torch.as_tensor(pred_ids, device=boxes.device, dtype=torch.long)
            target_ids = torch.as_tensor(target_ids, device=boxes.device, dtype=torch.long)
            if neighbor_sigma > 0 and len(points):
                # CenterNet-style Gaussian negative relaxation around GT points so adjacent
                # 3x3 grid anchors sharing the same visual feature do not suppress the matched center.
                min_dist2 = torch.cdist(boxes[:, :2].float(), points.float(), p=2).min(dim=1).values.square()
                neg_weight = (1.0 - torch.exp(-min_dist2 / (2.0 * (neighbor_sigma ** 2)))).clamp(0.0, 1.0) ** 4
                if logits.ndim == 2:
                    neg_weight = neg_weight[:, None]
            else:
                neg_weight = 1.0
        target = torch.zeros_like(logits, dtype=torch.float32)
        if logits.ndim==1:target[pred_ids] = 1
        else:target[pred_ids[:,None],torch.where(target_tokens)[0][None,:]]=1
        normalizer = max(1, len(points))
        loss_cls = torch.where(target.bool(), positive, negative * neg_weight).sum()/normalizer
        loss_center = F.l1_loss(boxes[pred_ids, :2].float(), points[target_ids].float(), reduction='sum')/normalizer
        zero = boxes.sum()*0
        loss_wh, loss_giou = zero, zero
        if len(exemplar_indices):
            inverse = torch.full((len(points),), -1, device=boxes.device, dtype=torch.long)
            inverse[target_ids] = pred_ids
            matched = boxes[inverse[exemplar_indices]].float()
            if (inverse[exemplar_indices] < 0).any():
                raise AssertionError('Exemplar target unmatched')
            target_wh = exemplar_xyxy[:, 2:]-exemplar_xyxy[:, :2]
            # Original CountGD-Box normalizes ALL geometry terms by GT count.
            loss_wh = F.l1_loss(matched[:, 2:], target_wh.float(), reduction='sum')/normalizer
            loss_giou = (1-generalized_box_iou(xywh_to_xyxy(matched), exemplar_xyxy.float()).diag()).sum()/normalizer
        total = self.loc*(loss_center+loss_wh)+self.giou*loss_giou+self.cls*loss_cls
        return total, dict(center=loss_center.detach(), wh=loss_wh.detach(),
            giou=loss_giou.detach(), cls=loss_cls.detach(), matched=len(points),
            exemplars=len(exemplar_indices))


class PROBQueryDecoder(nn.Module):
    """NMS-free Self-Attention Query Decoder + PROB Objectness + Visual/Self-Exemplar Similarity Guidance.

    Combines three complementary signals for open-world counting:
    1. Language-Guided + Visual/Self-Exemplar Similarity Query Selection (K=2200 queries):
       Extracts exemplar embeddings from 3 provided exemplar boxes (or auto-discovers
       top-5 high-confidence local peaks as Self-Exemplars when no boxes are given),
       computes multi-scale P3/P4 cosine similarity maps [mean_sim, max_sim], and fuses
       similarity with CLIP text scores so unseen classes with weak text response are still selected.
    2. Multi-Head Self-Attention + Cross-Attention over P3/P4/P5 + Exemplar tokens (1-to-1 NMS-free suppression).
    3. PROB Mahalanobis Gaussian Objectness Head (CVPR 2023).
    """
    def __init__(self, in_channels=(128, 256, 512), text_dim=512, hidden_dim=128, num_queries=2200, num_heads=8, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.level_projs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, hidden_dim, 1, bias=False), nn.LayerNorm([hidden_dim]))
            for c in in_channels
        ])
        self.level_embed = nn.Embedding(len(in_channels), hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # Visual/Self-Exemplar similarity projections on P3 (stride 8) and P4 (stride 16)
        self.ex_proj_p3 = nn.Conv2d(in_channels[0], 64, 1)
        self.ex_proj_p4 = nn.Conv2d(in_channels[1], 64, 1)
        self.sim_prior_scale = nn.Parameter(torch.tensor(2.0))
        self.sim_prior_bias = nn.Parameter(torch.tensor(-0.5))
        self.sim_to_query = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.pos_mlp = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.self_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, dropout=0.0, batch_first=True)
            for _ in range(num_layers)
        ])
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, dropout=0.0, batch_first=True)
            for _ in range(num_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            for _ in range(num_layers)
        ])
        self.norms1 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.norms3 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        # PROB Probabilistic Objectness Head: diagonal Mahalanobis energy in query embedding space.
        self.obj_norm = nn.LayerNorm(hidden_dim)
        self.obj_mean = nn.Parameter(torch.zeros(hidden_dim))
        self.obj_log_var = nn.Parameter(torch.zeros(hidden_dim))
        self.obj_scale = nn.Parameter(torch.tensor(0.1))
        self.obj_bias = nn.Parameter(torch.tensor(0.0))

        # Direct similarity residual + decoder residual heads
        self.sim_cls_head = nn.Sequential(
            nn.Linear(4, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.sim_cls_head[-1].weight)
        nn.init.zeros_(self.sim_cls_head[-1].bias)
        self.cls_delta = nn.Linear(hidden_dim, 1)
        self.box_delta = nn.Linear(hidden_dim, 4)
        nn.init.zeros_(self.cls_delta.weight)
        nn.init.zeros_(self.cls_delta.bias)
        nn.init.zeros_(self.box_delta.weight)
        nn.init.zeros_(self.box_delta.bias)

    def _exemplar_centers(self, base_boxes, base_logits, exemplar_boxes=None, num_self_ex=5):
        """Return (E, 2) normalized exemplar centers from GT exemplar boxes or top-5 Self-Exemplar peaks."""
        if exemplar_boxes is not None and len(exemplar_boxes) > 0:
            ex = exemplar_boxes.to(base_boxes.device).float().reshape(-1, 4)
            return ((ex[:, :2] + ex[:, 2:]) * 0.5).clamp(0.0, 1.0)
        # Self-Exemplar fallback: pick up to num_self_ex spatially separated high-scoring slots
        scores = base_logits if base_logits.ndim == 1 else base_logits.max(dim=-1).values
        top_idx = scores.topk(min(64, len(scores))).indices
        cand_xy = base_boxes[top_idx, :2].detach()
        picked = [cand_xy[0]]
        for pt in cand_xy[1:]:
            if len(picked) >= num_self_ex:
                break
            if all((pt - p).square().sum() > (0.04 ** 2) for p in picked):
                picked.append(pt)
        while len(picked) < num_self_ex:
            picked.append(picked[0])
        return torch.stack(picked, dim=0).clamp(0.0, 1.0)

    def _compute_slot_similarities(self, feats, slot_centers, ex_centers):
        """Compute P3 & P4 cosine similarities [sim3_mean, sim3_max, sim4_mean, sim4_max] at every candidate slot."""
        p3 = F.normalize(self.ex_proj_p3(feats[0]), dim=1)  # (1, 64, H3, W3)
        p4 = F.normalize(self.ex_proj_p4(feats[1]), dim=1)  # (1, 64, H4, W4)
        ex_grid = (ex_centers.clamp(0.0, 1.0) * 2.0 - 1.0)[None, :, None, :]  # (1, E, 1, 2)
        slot_grid = (slot_centers.clamp(0.0, 1.0) * 2.0 - 1.0)[None, :, None, :]  # (1, N, 1, 2)

        ex3 = F.normalize(F.grid_sample(p3, ex_grid, mode='bilinear', padding_mode='border', align_corners=False).squeeze(-1).transpose(1, 2)[0], dim=-1)  # (E, 64)
        ex4 = F.normalize(F.grid_sample(p4, ex_grid, mode='bilinear', padding_mode='border', align_corners=False).squeeze(-1).transpose(1, 2)[0], dim=-1)  # (E, 64)

        s3 = F.normalize(F.grid_sample(p3, slot_grid, mode='bilinear', padding_mode='border', align_corners=False).squeeze(-1).transpose(1, 2)[0], dim=-1)  # (N, 64)
        s4 = F.normalize(F.grid_sample(p4, slot_grid, mode='bilinear', padding_mode='border', align_corners=False).squeeze(-1).transpose(1, 2)[0], dim=-1)  # (N, 64)

        sim3 = s3 @ ex3.T  # (N, E)
        sim4 = s4 @ ex4.T  # (N, E)
        return torch.stack([sim3.mean(dim=-1), sim3.max(dim=-1).values, sim4.mean(dim=-1), sim4.max(dim=-1).values], dim=-1)

    def _sample_multiscale_tokens(self, feats, centers_norm):
        """Bilinearly sample P3/P4/P5 features at each query's (cx, cy) in [0,1]^2."""
        grid = (centers_norm.clamp(0.0, 1.0) * 2.0 - 1.0)[None, :, None, :]
        level_tokens = []
        for lvl, (feat, proj) in enumerate(zip(feats, self.level_projs)):
            sampled = F.grid_sample(feat, grid, mode='bilinear', padding_mode='border', align_corners=False)
            sampled = sampled.squeeze(-1).transpose(1, 2)
            w = proj[0].weight.squeeze(-1).squeeze(-1)
            t = F.layer_norm(F.linear(sampled, w), (self.hidden_dim,), proj[1].weight, proj[1].bias)
            t = t + self.level_embed.weight[lvl][None, None, :]
            level_tokens.append(t)
        return torch.stack(level_tokens, dim=2)  # (1, K, L, D)

    def forward(self, feats, base_boxes, base_logits, text_pe, anchor_xy=None, anchor_strides=None, exemplar_boxes=None):
        """Select top-K candidates via Text + Visual/Self-Exemplar Similarity and refine via Self-Attention + PROB."""
        k = min(self.num_queries, base_boxes.shape[0])
        ex_centers = self._exemplar_centers(base_boxes, base_logits, exemplar_boxes)
        slot_sims = self._compute_slot_similarities(feats, base_boxes[:, :2].detach(), ex_centers)  # (N, 4)

        # Combine text logit with visual similarity prior for query selection
        text_scores = base_logits if base_logits.ndim == 1 else base_logits.max(dim=-1).values
        sim_prior = self.sim_prior_scale.tanh() * (slot_sims[:, 1] + slot_sims[:, 3] + self.sim_prior_bias)
        selection_scores = text_scores + sim_prior

        topk_idx = selection_scores.topk(k, sorted=True).indices
        q_boxes = base_boxes[topk_idx]
        q_logits = base_logits[topk_idx]
        q_sims = slot_sims[topk_idx]  # (K, 4)
        q_anc_xy = anchor_xy[topk_idx] if anchor_xy is not None else None
        q_anc_st = anchor_strides[topk_idx] if anchor_strides is not None else None

        # Multi-scale visual tokens at each query center and at exemplar centers
        ms_tokens = self._sample_multiscale_tokens(feats, q_boxes[:, :2].detach())
        ex_tokens = self._sample_multiscale_tokens(feats, ex_centers).mean(dim=2)  # (1, E, D)
        text_token = self.text_proj(text_pe.view(1, -1, text_pe.shape[-1]).mean(dim=1, keepdim=True))  # (1, 1, D)

        # Geometry + text score + similarity positional encoding (10D)
        t_sel = text_scores[topk_idx]
        score_feat = torch.stack([t_sel.sigmoid(), (t_sel / 10.0).tanh()], dim=-1)
        pos = self.pos_mlp(torch.cat([q_boxes.detach(), score_feat.detach(), q_sims.detach()], dim=-1))[None]  # (1, K, D)

        # Initial query state: visual tokens + text conditioning + visual exemplar similarity embedding
        q = ms_tokens.mean(dim=2) + text_token + self.sim_to_query(q_sims)[None]
        # Cross-attention memory includes multi-scale query context, exemplar visual tokens, and text token
        kv = torch.cat([ms_tokens.mean(dim=2), ex_tokens, text_token], dim=1)

        for s_attn, c_attn, ffn, n1, n2, n3 in zip(
            self.self_attn_layers, self.cross_attn_layers, self.ffn_layers,
            self.norms1, self.norms2, self.norms3
        ):
            qk = q + pos
            q = n1(q + s_attn(qk, qk, q, need_weights=False)[0])
            q = n2(q + c_attn(q + pos, kv, kv, need_weights=False)[0])
            q = n3(q + ffn(q))

        q_out = q[0]  # (K, D)
        # PROB Mahalanobis Gaussian log-objectness: -0.5 * mean((z - mu)^2 / exp(log_var))
        z = self.obj_norm(q_out)
        mahalanobis = ((z - self.obj_mean) ** 2 * torch.exp(-self.obj_log_var.clamp(-4.0, 4.0))).mean(dim=-1)
        prob_obj_logit = -0.5 * mahalanobis * self.obj_scale.tanh() + self.obj_bias

        sim_logit = self.sim_cls_head(q_sims).squeeze(-1)
        cls_offset = self.cls_delta(q_out).squeeze(-1) + prob_obj_logit + sim_logit
        if q_logits.ndim == 1:
            refined_logits = q_logits + cls_offset
        else:
            refined_logits = q_logits + cls_offset[:, None]

        delta = self.box_delta(q_out).tanh() * 0.05
        refined_xy = (q_boxes[:, :2] + delta[:, :2]).clamp(0.0, 1.0)
        refined_wh = (q_boxes[:, 2:] * torch.exp(delta[:, 2:] * 2.0)).clamp(1e-4, 1.0)
        refined_boxes = torch.cat([refined_xy, refined_wh], dim=-1)
        return refined_boxes, refined_logits, q_anc_xy, q_anc_st


class YOLOWorldCounting(nn.Module):
    def __init__(self, weights, device='cuda', use_query_decoder=False, num_queries=2200):
        super().__init__()
        from ultralytics import YOLOWorld
        official = YOLOWorld(str(weights))
        self.detector = official.model.to(device).float()
        self.neck_start = len(self.detector.yaml['backbone'])
        for i, layer in enumerate(self.detector.model):
            layer.requires_grad_(i >= self.neck_start)
        # DFL is a fixed integral over distance bins, never an adaptable layer.
        self.detector.model[-1].dfl.requires_grad_(False)
        self.text_cache = {}
        self.detector.model[-1].nc = 1
        self.query_decoder = PROBQueryDecoder(num_queries=num_queries).to(device) if use_query_decoder else None
        self.last_anchor_xy = None
        self.last_anchor_strides = None
        self.set_training(False)

    @property
    def head(self):
        return self.detector.model[-1]

    @torch.no_grad()
    def cache_text(self, names):
        missing = list(dict.fromkeys(n for n in names if n not in self.text_cache))
        if missing:
            pe = self.detector.get_text_pe(missing, cache_clip_model=True)[0]
            self.text_cache.update({name:feature.detach().float().cpu() for name,feature in zip(missing,pe)})
        # CLIP is created lazily after the initial freeze; explicitly freeze it too.
        if self.detector.clip_model is not None:
            self.detector.clip_model.requires_grad_(False).eval()

    def set_training(self, training):
        # Batch size one: preserve pretrained BatchNorm statistics everywhere.
        self.detector.eval()
        self.head.training = training
        if self.query_decoder is not None:
            self.query_decoder.train(training)
        if training:
            # Evaluation may have cached anchors created under inference_mode.
            # Recreate them under the training context before autograd uses them.
            self.head.shape = None

    def _grid_priors(self, image):
        w, h = float(image.shape[-1]), float(image.shape[-2])
        strides = self.head.strides.T.squeeze(-1)
        anchors_px = self.head.anchors.T * strides[:, None]
        anchor_xy = anchors_px / image.new_tensor([w, h])
        norm_strides = strides / max(w, h)
        return anchor_xy, norm_strides

    def anchor_priors(self, image):
        """Return normalized (N,2) grid anchor centers and (N,) normalized stride scale for spatial matching."""
        if self.last_anchor_xy is not None and self.last_anchor_strides is not None:
            return self.last_anchor_xy, self.last_anchor_strides
        return self._grid_priors(image)

    def forward_tokens(self, image, prompts, exemplar_boxes=None):
        if isinstance(prompts, str):
            prompts = [prompts]
        if not prompts:
            raise ValueError('At least one text prompt is required')
        missing = [p for p in prompts if p not in self.text_cache]
        if missing:
            raise KeyError(f'Encode prompts outside the optimization forward: {missing[:3]}')
        text = torch.stack([self.text_cache[p] for p in prompts])[None].to(image.device)
        self.head.nc = len(prompts)
        result = self.detector.predict(image, txt_feats=text)
        if isinstance(result, tuple):
            decoded, raw = result
        else:
            raw = result
            decoded = self.head._inference(raw)
        if not isinstance(raw, dict) or raw['scores'].shape[1] != len(prompts):
            raise RuntimeError('Unexpected Ultralytics WorldDetect raw output schema')
        scale = image.new_tensor([image.shape[-1], image.shape[-2]]*2)
        boxes = decoded[0, :4].T / scale
        logits = raw['scores'][0].T.float()
        anc_xy, anc_st = self._grid_priors(image)
        if self.query_decoder is not None:
            boxes, logits, anc_xy, anc_st = self.query_decoder(
                raw['feats'], boxes, logits, text, anchor_xy=anc_xy, anchor_strides=anc_st,
                exemplar_boxes=exemplar_boxes)
        self.last_anchor_xy = anc_xy
        self.last_anchor_strides = anc_st
        return boxes, logits

    def forward(self, image, prompt, exemplar_boxes=None):
        if not isinstance(prompt, str):
            return self.forward_tokens(image, prompt, exemplar_boxes=exemplar_boxes)
        boxes, logits = self.forward_tokens(image, [prompt], exemplar_boxes=exemplar_boxes)
        return boxes, logits[:, 0]
