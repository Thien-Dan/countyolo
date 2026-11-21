"""Pure-PyTorch standalone OV-DQUO core modules (EVA02-CLIP ViT + 3-scale Multi-Scale Deformable Transformer).

Zero external C++/CUDA build dependencies:
- Replaces `xformers` with PyTorch 2.x `F.scaled_dot_product_attention` (identical math, native FP16 on T4).
- Replaces compiled `MultiScaleDeformableAttention` CUDA op with pure-PyTorch FP32 `F.grid_sample`.
- Replaces `einops` and `mmcv.BaseModule` with pure `torch` operations.
- Preserves 100% exact parameter/state_dict keys of official `OVDQUO_ViT-B16_LVIS.pth` & `OVDQUO_ViT-L14_LVIS.pth`.
"""
import copy
import math
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F


def inverse_sigmoid(x: Tensor, eps: float = 1e-5) -> Tensor:
    x = x.clamp(min=0.0, max=1.0)
    x1 = x.clamp(min=eps)
    x2 = (1.0 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def _rotate_half(x: Tensor) -> Tensor:
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _broadcat(tensors: list[Tensor], dim: int = -1) -> Tensor:
    num_tensors = len(tensors)
    shape_len = len(tensors[0].shape)
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*[list(t.shape) for t in tensors]))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    max_dims = [(i, max(val)) for i, val in expandable_dims]
    expanded_dims = [(i, (val,) * num_tensors) for i, val in max_dims]
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*[t[1] for t in expanded_dims]))
    tensors = [t.expand(*s) for t, s in zip(tensors, expandable_shapes)]
    return torch.cat(tensors, dim=dim)


class VisionRotaryEmbeddingFast(nn.Module):
    """Exact 2D RoPE matching `src/open_clip/eva_clip/rope.py` without `einops`."""
    def __init__(self, dim: int, pt_seq_len: int, ft_seq_len: Optional[int] = None, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.pt_seq_len = pt_seq_len
        self.ft_seq_len = ft_seq_len if ft_seq_len is not None else pt_seq_len
        self.theta = theta
        freqs_cos, freqs_sin = self._build_freqs(self.ft_seq_len, self.ft_seq_len)
        self.register_buffer('freqs_cos', freqs_cos)
        self.register_buffer('freqs_sin', freqs_sin)
        self._cache: dict[tuple[int, int, str, torch.dtype], tuple[Tensor, Tensor]] = {}

    def _build_freqs(self, h: int, w: int) -> tuple[Tensor, Tensor]:
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim))
        th = torch.arange(h, dtype=torch.float32) / float(h) * float(self.pt_seq_len)
        tw = torch.arange(w, dtype=torch.float32) / float(w) * float(self.pt_seq_len)
        fh = torch.einsum('..., f -> ... f', th, freqs).repeat_interleave(2, dim=-1)
        fw = torch.einsum('..., f -> ... f', tw, freqs).repeat_interleave(2, dim=-1)
        f2d = _broadcat([fh[:, None, :], fw[None, :, :]], dim=-1)
        return f2d.cos().reshape(-1, f2d.shape[-1]), f2d.sin().reshape(-1, f2d.shape[-1])

    def forward(self, t: Tensor, hw: Optional[tuple[int, int]] = None) -> Tensor:
        seq_len = t.shape[2]
        if hw is None:
            side = int(round(seq_len ** 0.5))
            hw = (side, side)
        h, w = hw
        if h == self.ft_seq_len and w == self.ft_seq_len and seq_len == self.freqs_cos.shape[0]:
            fc = self.freqs_cos.to(device=t.device, dtype=t.dtype)
            fs = self.freqs_sin.to(device=t.device, dtype=t.dtype)
        else:
            key = (h, w, str(t.device), t.dtype)
            cached = self._cache.get(key)
            if cached is None:
                fc, fs = self._build_freqs(h, w)
                fc = fc.to(device=t.device, dtype=t.dtype)
                fs = fs.to(device=t.device, dtype=t.dtype)
                if len(self._cache) > 16:
                    self._cache.clear()
                self._cache[key] = (fc, fs)
            else:
                fc, fs = cached
        return t * fc + _rotate_half(t) * fs


class SwiGLU(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, subln: bool = True, drop: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.act = nn.SiLU()
        self.ffn_ln = nn.LayerNorm(hidden_features) if subln else nn.Identity()
        self.w3 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.w3(self.ffn_ln(self.act(self.w1(x)) * self.w2(x))))


class EVAAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 12, qkv_bias: bool = True,
                 rope: Optional[VisionRotaryEmbeddingFast] = None, subln: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.subln = subln
        if subln:
            self.q_proj = nn.Linear(dim, dim, bias=False)
            self.k_proj = nn.Linear(dim, dim, bias=False)
            self.v_proj = nn.Linear(dim, dim, bias=False)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.inner_attn_ln = nn.LayerNorm(dim) if subln else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.0)
        self.rope = rope

    def forward(self, x: Tensor, hw: Optional[tuple[int, int]] = None) -> Tensor:
        bsz, seq_len, _ = x.shape
        if self.subln:
            q = F.linear(x, self.q_proj.weight, self.q_bias).reshape(bsz, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k = F.linear(x, self.k_proj.weight, None).reshape(bsz, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v = F.linear(x, self.v_proj.weight, self.v_bias).reshape(bsz, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        else:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias), self.v_bias)) if self.q_bias is not None else None
            qkv = F.linear(x, self.qkv.weight, qkv_bias).reshape(bsz, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

        if self.rope is not None:
            q = torch.cat((q[:, :, :1, :], self.rope(q[:, :, 1:, :], hw=hw)), dim=-2).type_as(v)
            k = torch.cat((k[:, :, :1, :], self.rope(k[:, :, 1:, :], hw=hw)), dim=-2).type_as(v)

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, scale=self.scale)
        out = out.transpose(1, 2).reshape(bsz, seq_len, -1)
        out = self.proj_drop(self.proj(self.inner_attn_ln(out)))
        return out

    def proj_without_attn(self, x: Tensor) -> Tensor:
        """CLIPSelf dense patch projection: skips spatial attention in final ViT block."""
        if self.subln:
            x = F.linear(x, self.v_proj.weight, self.v_bias)
        else:
            dim = self.num_heads * self.head_dim
            v_w = self.qkv.weight[2 * dim:]
            x = F.linear(x, v_w, self.v_bias)
        return self.proj_drop(self.proj(self.inner_attn_ln(x)))


class EVABlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 2.6667, qkv_bias: bool = True,
                 rope: Optional[VisionRotaryEmbeddingFast] = None, subln: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EVAAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, rope=rope, subln=subln)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = SwiGLU(dim, int(dim * mlp_ratio), subln=subln)

    def forward(self, x: Tensor, hw: Optional[tuple[int, int]] = None) -> Tensor:
        x = x + self.attn(self.norm1(x), hw=hw)
        x = x + self.mlp(self.norm2(x))
        return x

    def forward_without_attn(self, x: Tensor) -> Tensor:
        x = x + self.attn.proj_without_attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class EVAPatchEmbed(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.patch_shape = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.patch_shape[0] * self.patch_shape[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class EVAVisionTransformer(nn.Module):
    """Exact state-dict compatible EVA02-CLIP Vision Transformer (B/16 or L/14-336)."""
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 2.6667,
        num_classes: int = 512,
        pt_hw_seq_len: int = 16,
        subln: bool = True,
        rope: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = EVAPatchEmbed(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(0.0)
        if rope:
            half_head_dim = embed_dim // num_heads // 2
            hw_seq_len = img_size // patch_size
            self.rope = VisionRotaryEmbeddingFast(dim=half_head_dim, pt_seq_len=pt_hw_seq_len, ft_seq_len=hw_seq_len)
        else:
            self.rope = None
        self.blocks = nn.ModuleList([
            EVABlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, rope=self.rope, subln=subln)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def rescale_positional_embedding(self, out_size: tuple[int, int]) -> Tensor:
        h, w = out_size
        if (h, w) == self.patch_embed.patch_shape:
            return self.pos_embed
        rescaled = self.pos_embed.new_zeros(1, 1 + h * w, self.pos_embed.shape[2])
        rescaled[0, 0] = self.pos_embed[0, 0]
        pe_2d = self.pos_embed[0, 1:].T.contiguous().reshape(1, -1, *self.patch_embed.patch_shape)
        pe_2d = F.interpolate(pe_2d, out_size, mode='bicubic', align_corners=False).reshape(-1, h * w)
        rescaled[0, 1:] = pe_2d.T.contiguous()
        return rescaled


class EvaCLIPViTBackbone(nn.Module):
    """Matches OV-DQUO `backbone.0` (`EvaCLIPViT`): frozen EVA02-CLIP ViT + 3-scale pyramid + dense CLIPSelf map."""
    def __init__(self, visual: EVAVisionTransformer, out_indices: tuple[int, int, int] = (5, 7, 11)):
        super().__init__()
        self.visual = visual
        self.out_indices = list(out_indices)
        self.width = visual.embed_dim
        self.patch_size = visual.patch_embed.patch_size[0]
        self.interpolate1 = nn.Sequential(nn.ConvTranspose2d(self.width, self.width, kernel_size=2, stride=2))
        self.interpolate2 = nn.Identity()
        self.interpolate3 = nn.MaxPool2d(kernel_size=2, stride=2)

    def _expand_x(self, x: Tensor, h: int, w: int) -> Tensor:
        return x[:, 1:].permute(0, 2, 1).contiguous().reshape(-1, self.width, h, w)

    def forward(self, pixel_values: Tensor, mask: Tensor) -> tuple[list[Tensor], list[Tensor], Tensor, Tensor]:
        """Return (3-scale feature maps, 3-scale padding masks, dense CLIPSelf map, dense mask)."""
        bs, _, img_h, img_w = pixel_values.shape
        h = img_h // self.patch_size
        w = img_w // self.patch_size
        visual = self.visual

        with torch.no_grad():
            x = visual.patch_embed(pixel_values)
            cls_tokens = visual.cls_token.expand(bs, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + visual.rescale_positional_embedding((h, w))
            x = visual.pos_drop(x)

            raw_outs = []
            for i, blk in enumerate(visual.blocks[:-1]):
                x = blk(x, hw=(h, w))
                if i in self.out_indices:
                    raw_outs.append(self._expand_x(x, h, w))
            x = visual.blocks[-1].forward_without_attn(x)
            if (len(visual.blocks) - 1) in self.out_indices:
                raw_outs.append(self._expand_x(x, h, w))

            dense_tokens = visual.head(visual.norm(x[:, 1:]))
            dense_tokens = F.normalize(dense_tokens.float(), dim=-1)
            dense_map = dense_tokens.reshape(bs, h, w, -1).permute(0, 3, 1, 2).contiguous()

        feats, masks = [], []
        interps = [self.interpolate1, self.interpolate2, self.interpolate3]
        for out_t, interp in zip(raw_outs, interps):
            f = interp(out_t.detach())
            m = F.interpolate(mask[None].float(), size=f.shape[-2:]).to(torch.bool)[0]
            feats.append(f)
            masks.append(m)
        dense_mask = F.interpolate(mask[None].float(), size=dense_map.shape[-2:]).to(torch.bool)[0]
        return feats, masks, dense_map, dense_mask


class PositionEmbeddingSineHW(nn.Module):
    def __init__(self, num_pos_feats: int = 128, temperatureH: float = 20.0, temperatureW: float = 20.0, normalize: bool = True):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperatureH = temperatureH
        self.temperatureW = temperatureW
        self.normalize = normalize
        self.scale = 2.0 * math.pi

    def forward(self, mask: Tensor, dtype: torch.dtype = torch.float32) -> Tensor:
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_tx = self.temperatureW ** (2 * (dim_t // 2) / self.num_pos_feats)
        dim_ty = self.temperatureH ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_tx
        pos_y = y_embed[:, :, :, None] / dim_ty
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2).to(dtype)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def gen_sineembed_for_position(pos_tensor: Tensor, d_model: int = 256) -> Tensor:
    scale = 2.0 * math.pi
    half_d = d_model // 2
    dim_t = torch.arange(half_d, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000.0 ** (2 * (dim_t // 2) / half_d)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = torch.stack(((x_embed[:, :, None] / dim_t)[:, :, 0::2].sin(), (x_embed[:, :, None] / dim_t)[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack(((y_embed[:, :, None] / dim_t)[:, :, 0::2].sin(), (y_embed[:, :, None] / dim_t)[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        return torch.cat((pos_y, pos_x), dim=2)
    w_embed = pos_tensor[:, :, 2] * scale
    h_embed = pos_tensor[:, :, 3] * scale
    pos_w = torch.stack(((w_embed[:, :, None] / dim_t)[:, :, 0::2].sin(), (w_embed[:, :, None] / dim_t)[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_h = torch.stack(((h_embed[:, :, None] / dim_t)[:, :, 0::2].sin(), (h_embed[:, :, None] / dim_t)[:, :, 1::2].cos()), dim=3).flatten(2)
    return torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)


def gen_encoder_output_proposals(memory: Tensor, memory_padding_mask: Tensor, spatial_shapes: Tensor) -> tuple[Tensor, Tensor]:
    bsz = memory.shape[0]
    proposals = []
    cur = 0
    for lvl, (h_, w_) in enumerate(spatial_shapes.tolist()):
        mask_l = memory_padding_mask[:, cur: cur + h_ * w_].reshape(bsz, h_, w_, 1)
        valid_h = torch.sum(~mask_l[:, :, 0, 0], 1)
        valid_w = torch.sum(~mask_l[:, 0, :, 0], 1)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, h_ - 1, h_, dtype=torch.float32, device=memory.device),
            torch.linspace(0, w_ - 1, w_, dtype=torch.float32, device=memory.device),
            indexing='ij',
        )
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
        scale = torch.cat([valid_w.unsqueeze(-1), valid_h.unsqueeze(-1)], 1).reshape(bsz, 1, 1, 2).clamp(min=1.0)
        grid = (grid.unsqueeze(0).expand(bsz, -1, -1, -1) + 0.5) / scale
        wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
        proposals.append(torch.cat((grid, wh), -1).reshape(bsz, -1, 4))
        cur += h_ * w_

    output_proposals = torch.cat(proposals, 1)
    valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
    output_proposals = torch.log(output_proposals.clamp(1e-4, 1.0 - 1e-4) / (1.0 - output_proposals.clamp(1e-4, 1.0 - 1e-4)))
    output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
    output_proposals = output_proposals.masked_fill(~valid, float('inf'))

    output_memory = memory.masked_fill(memory_padding_mask.unsqueeze(-1), 0.0)
    output_memory = output_memory.masked_fill(~valid, 0.0)
    return output_memory, output_proposals


def ms_deform_attn_core_pytorch(value: Tensor, spatial_shapes: Tensor, sampling_locations: Tensor, attention_weights: Tensor) -> Tensor:
    """Pure-PyTorch FP32 Multi-Scale Deformable Attention core via `F.grid_sample`."""
    bsz, _, n_heads, head_dim = value.shape
    _, lq, _, n_levels, n_points, _ = sampling_locations.shape
    value_list = value.split([int(h_ * w_) for h_, w_ in spatial_shapes.tolist()], dim=1)
    sampling_grids = 2.0 * sampling_locations - 1.0
    sampling_value_list = []
    for lid, (h_, w_) in enumerate(spatial_shapes.tolist()):
        v_l = value_list[lid].flatten(2).transpose(1, 2).reshape(bsz * n_heads, head_dim, h_, w_)
        grid_l = sampling_grids[:, :, :, lid].transpose(1, 2).flatten(0, 1)
        sampled_l = F.grid_sample(v_l, grid_l, mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampled_l)
    attn_w = attention_weights.transpose(1, 2).reshape(bsz * n_heads, 1, lq, n_levels * n_points)
    out = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attn_w).sum(-1).reshape(bsz, n_heads * head_dim, lq)
    return out.transpose(1, 2).contiguous()


class MSDeformAttn(nn.Module):
    def __init__(self, d_model: int = 256, n_levels: int = 3, n_heads: int = 8, n_points: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).reshape(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.reshape(-1))
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(self, query: Tensor, reference_points: Tensor, input_flatten: Tensor,
                input_spatial_shapes: Tensor, input_padding_mask: Optional[Tensor] = None) -> Tensor:
        bsz, lq, _ = query.shape
        _, lin, _ = input_flatten.shape
        orig_dtype = query.dtype
        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], 0.0)
        value = value.reshape(bsz, lin, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).reshape(bsz, lq, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = F.softmax(
            self.attention_weights(query).reshape(bsz, lq, self.n_heads, self.n_levels * self.n_points).float(), dim=-1
        ).reshape(bsz, lq, self.n_heads, self.n_levels, self.n_points)

        with torch.autocast('cuda', enabled=False):
            ref = reference_points.float()
            offs = sampling_offsets.float()
            if ref.shape[-1] == 2:
                offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1).float()
                sampling_locations = ref[:, :, None, :, None, :] + offs / offset_normalizer[None, None, None, :, None, :]
            else:
                sampling_locations = ref[:, :, None, :, None, :2] + offs / float(self.n_points) * ref[:, :, None, :, None, 2:] * 0.5
            out = ms_deform_attn_core_pytorch(value.float(), input_spatial_shapes, sampling_locations, attention_weights)
        return self.output_proj(out.to(orig_dtype))


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=2048, dropout=0.0, n_levels=3, n_heads=8, n_points=4):
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src: Tensor, pos: Tensor, reference_points: Tensor,
                spatial_shapes: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        src2 = self.self_attn(src + pos, reference_points, src, spatial_shapes, key_padding_mask)
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout2(F.relu(self.linear1(src))))
        return self.norm2(src + self.dropout3(src2))


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: DeformableTransformerEncoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes: Tensor, valid_ratios: Tensor, device) -> Tensor:
        refs = []
        for lvl, (h_, w_) in enumerate(spatial_shapes.tolist()):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, h_ - 0.5, h_, dtype=torch.float32, device=device),
                torch.linspace(0.5, w_ - 0.5, w_, dtype=torch.float32, device=device),
                indexing='ij',
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * h_).clamp(min=1e-4)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * w_).clamp(min=1e-4)
            refs.append(torch.stack((ref_x, ref_y), -1))
        return torch.cat(refs, 1)[:, :, None] * valid_ratios[:, None]

    def forward(self, src: Tensor, pos: Tensor, spatial_shapes: Tensor,
                valid_ratios: Tensor, key_padding_mask: Tensor, grad_checkpoint: bool = False) -> Tensor:
        out = src
        ref_pts = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for layer in self.layers:
            if grad_checkpoint and self.training and torch.is_grad_enabled():
                out = torch.utils.checkpoint.checkpoint(
                    lambda s, _l=layer: _l(s, pos, ref_pts, spatial_shapes, key_padding_mask),
                    out,
                    use_reentrant=False,
                )
            else:
                out = layer(out, pos, ref_pts, spatial_shapes, key_padding_mask)
        return out


class DeformableTransformerDecoderLayer(nn.Module):
    """Exact OV-DQUO decoder layer order: `module_seq = ['sa', 'ca', 'ffn']`."""
    def __init__(self, d_model=256, d_ffn=2048, dropout=0.0, n_levels=3, n_heads=8, n_points=4):
        super().__init__()
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt: Tensor, tgt_query_pos: Tensor, tgt_reference_points: Tensor,
                memory: Tensor, memory_spatial_shapes: Tensor, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        # 1. Self-Attention across queries (L, B, C)
        q = k = tgt + tgt_query_pos
        tgt2 = self.self_attn(q, k, tgt, need_weights=False)[0]
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        # 2. Multi-Scale Deformable Cross-Attention (expects B, L, C)
        tgt2 = self.cross_attn(
            (tgt + tgt_query_pos).transpose(0, 1),
            tgt_reference_points.transpose(0, 1).contiguous(),
            memory.transpose(0, 1),
            memory_spatial_shapes,
            memory_key_padding_mask,
        ).transpose(0, 1)
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        # 3. FFN
        tgt2 = self.linear2(self.dropout3(F.relu(self.linear1(tgt))))
        return self.norm3(tgt + self.dropout4(tgt2))


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer: DeformableTransformerDecoderLayer, num_layers: int, d_model: int = 256):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.ref_point_head = MLP(2 * d_model, d_model, d_model, 2)
        self.d_model = d_model
        self.bbox_embed: Optional[nn.ModuleList] = None

    def forward(self, tgt: Tensor, memory: Tensor, memory_key_padding_mask: Tensor,
                refpoints_unsigmoid: Tensor, spatial_shapes: Tensor, valid_ratios: Tensor):
        output = tgt
        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]  # Look-forward-twice: keeps undetached new_reference_points

        for layer_id, layer in enumerate(self.layers):
            reference_points_input = reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
            query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :], d_model=self.d_model)
            query_pos = self.ref_point_head(query_sine_embed)
            output = layer(output, query_pos, reference_points_input, memory, spatial_shapes, memory_key_padding_mask)

            if self.bbox_embed is not None:
                delta_unsig = self.bbox_embed[layer_id](output)
                new_reference_points = (delta_unsig + inverse_sigmoid(reference_points)).sigmoid()
                reference_points = new_reference_points.detach()
                ref_points.append(new_reference_points)  # UNDETACHED -> full look-forward-twice gradient!
            intermediate.append(self.norm(output))

        return (
            [itm.transpose(0, 1) for itm in intermediate],
            [ref.transpose(0, 1) for ref in ref_points],
        )


class OVDeformableTransformer(nn.Module):
    """Exact state-dict compatible `OVDeformableTransformer` matching `OVDQUO_ViT-B16_LVIS.pth`."""
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_queries: int = 1000,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        num_feature_levels: int = 3,
        enc_n_points: int = 4,
        dec_n_points: int = 4,
        text_dim: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_queries = num_queries
        self.num_feature_levels = num_feature_levels
        self.num_decoder_layers = num_decoder_layers

        enc_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward, dropout, num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(enc_layer, num_encoder_layers)

        dec_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward, dropout, num_feature_levels, nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(dec_layer, num_decoder_layers, d_model=d_model)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.tgt_embed = nn.Embedding(num_queries, d_model)
        self.enc_output = nn.Linear(d_model, d_model)
        self.enc_output_norm = nn.LayerNorm(d_model)
        self.enc_out_class_embed = nn.Linear(d_model, 1)
        self.enc_out_bbox_embed = MLP(d_model, d_model, 4, 3)
        self.text_proj = MLP(text_dim, 128, d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        nn.init.normal_(self.level_embed)
        nn.init.normal_(self.tgt_embed.weight)
        prior_prob = 0.01
        bias_val = -math.log((1.0 - prior_prob) / prior_prob)
        nn.init.constant_(self.enc_out_class_embed.bias, bias_val)
        nn.init.constant_(self.enc_out_bbox_embed.layers[-1].weight, 0.0)
        nn.init.constant_(self.enc_out_bbox_embed.layers[-1].bias, 0.0)

    @staticmethod
    def get_valid_ratio(mask: Tensor) -> Tensor:
        _, h, w = mask.shape
        valid_h = torch.sum(~mask[:, :, 0], 1).float() / float(h)
        valid_w = torch.sum(~mask[:, 0, :], 1).float() / float(w)
        return torch.stack([valid_w, valid_h], -1)

    def encode_and_propose(self, srcs: list[Tensor], masks: list[Tensor], pos_embeds: list[Tensor], grad_checkpoint: bool = False):
        src_flatten, mask_flatten, lvl_pos_flatten, spatial_shapes = [], [], [], []
        for lvl, (src, mask, pos) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shapes.append((h, w))
            src_flatten.append(src.flatten(2).transpose(1, 2))
            mask_flatten.append(mask.flatten(1))
            lvl_pos_flatten.append(pos.flatten(2).transpose(1, 2) + self.level_embed[lvl].reshape(1, 1, -1))
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_flatten = torch.cat(lvl_pos_flatten, 1)
        spatial_shapes_t = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        memory = self.encoder(src_flatten, lvl_pos_flatten, spatial_shapes_t, valid_ratios, mask_flatten, grad_checkpoint=grad_checkpoint)
        output_memory, output_proposals = gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes_t)
        output_memory = self.enc_output_norm(self.enc_output(output_memory))

        enc_class = self.enc_out_class_embed(output_memory)  # (B, S, 1)
        enc_coord_unact = self.enc_out_bbox_embed(output_memory) + output_proposals  # (B, S, 4)
        topk = min(self.num_queries, enc_class.shape[1])
        topk_inds = torch.topk(enc_class.max(-1)[0], topk, dim=1)[1]

        refpoint_undetach = torch.gather(enc_coord_unact, 1, topk_inds.unsqueeze(-1).expand(-1, -1, 4))
        tgt_undetach = torch.gather(output_memory, 1, topk_inds.unsqueeze(-1).expand(-1, -1, self.d_model))
        enc_class_topk = torch.gather(enc_class, 1, topk_inds.unsqueeze(-1))
        return dict(
            memory=memory,
            mask_flatten=mask_flatten,
            spatial_shapes=spatial_shapes_t,
            valid_ratios=valid_ratios,
            refpoint_undetach=refpoint_undetach,
            tgt_undetach=tgt_undetach,
            enc_class_topk=enc_class_topk,
        )
