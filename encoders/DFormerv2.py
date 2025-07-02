"""
DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation
Code: https://github.com/VCIP-RGBD/DFormer

Author: yinbow
Email: bowenyin@mail.nankai.edu.cn

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
from timm.models.layers import DropPath, trunc_normal_
from typing import List, Union, Optional
from mmengine.runner.checkpoint import load_state_dict
from mmengine.runner.checkpoint import load_checkpoint
from typing import Tuple
import sys
import os
from collections import OrderedDict
from torch import Tensor


class TempMoE(nn.Module):
    def __init__(self, 
                 d_model: int = 512,
                 nhead: int = 8,
                 topK: int = 5,
                 n_experts: int = 10,
                 sigma: int = 9,
                 dropout: float = 0.1,
                 vis_branch: bool = False,
    ):
        super(TempMoE, self).__init__()

        self.sigma = sigma
        self.topK = topK
        self.n_experts = n_experts
        
        if vis_branch:
            self.anorm = nn.LayerNorm(d_model)
            self.vnorm = nn.LayerNorm(d_model)
        else:
            self.norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.qst_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.1)
        self.gauss_pred = nn.Sequential(
            nn.Linear(d_model, 2 * n_experts)
        )
        self.router = nn.Sequential(
            nn.Linear(d_model, n_experts)
        )
        self.experts = nn.ModuleList([
            nn.Sequential(*[
                nn.Linear(d_model, int(d_model // 2)),
                nn.ReLU(),
                nn.Linear(int(d_model // 2), d_model)
            ])
            for _ in range(n_experts)
        ])
        self.experts.apply(self._init_weights)

        self.margin = (1 / (n_experts * 2)) # non overlapping center area
        self.center = torch.linspace(self.margin, 1-self.margin, self.n_experts)
        self.center.requires_grad_(False)
        self.router.apply(self._init_weights)
        self.gauss_pred.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def generate_gaussian(self,
                          pred: torch.Tensor, 
                          topk_inds: torch.Tensor,
                          T: int = 60
    ) -> Tensor:
        # [refernce] https://github.com/minghangz/cpl
        weights = []
        centers = self.center.unsqueeze(0).repeat(pred.size(0), 1).to(pred.device)
        centers = centers + pred[:, :, 0]
        centers = torch.gather(centers, 1, topk_inds)
        widths = torch.gather(pred[:, :, 1], 1, topk_inds)
        
        for i in range(self.topK):
            center = centers[:, i]
            width = widths[:, i]
            
            weight = torch.linspace(0, 1, T)
            weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)
            center = torch.clamp(center.unsqueeze(-1), min=0, max=1)
            width = torch.clamp(width.unsqueeze(-1), min=0.09) / self.sigma
            
            w = 0.3989422804014327
            weight = w/width*torch.exp(-(weight-center)**2/(2*width**2))
            weights.append(
                weight/weight.max(dim=-1, keepdim=True)[0]
            )
        return torch.stack(weights, dim=1)
    
    def get_output(self,
                   experts_logits: Tensor,
                   gauss_weight: Tensor,
                   topk_inds: Tensor,
                   topk_probs: Tensor,
                   shape: tuple,
    ) -> Tensor:
        B, T, C = shape
        
        experts_logits = torch.gather(
            experts_logits.permute(1, 0, 2, 3).reshape(B*T, self.n_experts, -1), 1, 
            topk_inds.repeat(T, 1).unsqueeze(-1).repeat(1, 1, C)
        )
        experts_logits = experts_logits.reshape(B, T, self.topK, -1).contiguous()
        output = [
            (gauss_weight[:, i, :].unsqueeze(1) @ experts_logits[:, :, i, :])
            for i in range(self.topK)
        ]
        output = torch.cat(output, dim=1) # [B, N_EXPERTS, C]
        output = topk_probs.unsqueeze(1) @ output
        return output
    
    def forward(self,
                qst: Tensor,                       # [B, D]
                data: Tensor                     # [B, T, D] Audio | Video,
    ):
        B, T, C = data.size()
        data = data.permute(1, 0, 2)
        
        qst = qst.unsqueeze(0)                                              # [1, B, D]
        temp_w = self.qst_attn(qst, data, data)[0]                          # [1, B, C]
        temp_w = temp_w.squeeze(0)                                          # [B, C]
        router_logits = self.router(temp_w)                                 # [B, N_EXPERTS]
        router_probs = F.softmax(router_logits, dim=-1)                     # [B, N_EXPERTS]
        topk_probs, topk_inds = torch.topk(router_probs, self.topK, dim=-1) # [B, TOPK]
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)      # [B, TOPK]
        
        gauss_cw = self.gauss_pred(temp_w)                                  # [B, 2*N_EXPERTS]
        gauss_cw = gauss_cw.view(B, self.n_experts, 2)                      # [B, N_EXPERTS, 2]
        gauss_cw[:, :, 0] = torch.tanh(gauss_cw[:, :, 0]) * self.margin     
        gauss_cw[:, :, 1] = torch.sigmoid(gauss_cw[:, :, 1])
        gauss_weight = self.generate_gaussian(gauss_cw, topk_inds=topk_inds, T=T) # [B, TOPK, T]
        
        return gauss_weight

class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: torch.Tensor):
        """
        input shape (b c h w)
        """
        x = x.permute(0, 2, 3, 1).contiguous()  # (b h w c)
        x = self.norm(x)  # (b h w c)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding
    """

    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, 3, 8, 1),
            nn.SyncBatchNorm(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, 3, 1, 1),
            nn.SyncBatchNorm(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, 3, 4, 1),
            nn.SyncBatchNorm(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.SyncBatchNorm(embed_dim),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).permute(0, 2, 3, 1)
        return x


class DWConv2d(nn.Module):
    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        """
        input (b h w c)
        """
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        return x


class PatchMerging(nn.Module):
    """
    Patch Merging Layer.
    """

    def __init__(self, dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(dim, out_dim, 3, 2, 1)
        self.norm = nn.SyncBatchNorm(out_dim)

    def forward(self, x):
        """
        x: B H W C
        """
        x = x.permute(0, 3, 1, 2).contiguous()  # (b c h w)
        x = self.reduction(x)  # (b oc oh ow)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (b oh ow oc)
        return x


def angle_transform(x, sin, cos):
    x1 = x[:, :, :, :, ::2]
    x2 = x[:, :, :, :, 1::2]
    return (x * cos) + (torch.stack([-x2, x1], dim=-1).flatten(-2) * sin)


class GeoPriorGen(nn.Module):
    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.weight = nn.Parameter(torch.ones(2, 1, 1, 1), requires_grad=True)
        decay = torch.log(
            1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads)
        )
        self.register_buffer("angle", angle)
        self.register_buffer("decay", decay)

    def generate_depth_decay(self, H: int, W: int, depth_grid):
        """
        generate 2d decay mask, the result is (HW)*(HW)
        H, W are the numbers of patches at each column and row
        """
        B, _, H, W = depth_grid.shape
        grid_d = depth_grid.reshape(B, H * W, 1)
        mask_d = grid_d[:, :, None, :] - grid_d[:, None, :, :]
        mask_d = (mask_d.abs()).sum(dim=-1)
        mask_d = mask_d.unsqueeze(1) * self.decay[None, :, None, None]
        return mask_d

    def generate_pos_decay(self, H: int, W: int):
        """
        generate 2d decay mask, the result is (HW)*(HW)
        H, W are the numbers of patches at each column and row
        """
        index_h = torch.arange(H).to(self.decay)
        index_w = torch.arange(W).to(self.decay)
        grid = torch.meshgrid([index_h, index_w])
        grid = torch.stack(grid, dim=-1).reshape(H * W, 2)
        mask = grid[:, None, :] - grid[None, :, :]
        mask = (mask.abs()).sum(dim=-1)
        mask = mask * self.decay[:, None, None]
        return mask

    def generate_1d_depth_decay(self, H, W, depth_grid):
        """
        generate 1d depth decay mask, the result is l*l
        """
        mask = depth_grid[:, :, :, :, None] - depth_grid[:, :, :, None, :]
        mask = mask.abs()
        mask = mask * self.decay[:, None, None, None]
        assert mask.shape[2:] == (W, H, H)
        return mask

    def generate_1d_decay(self, l: int):
        """
        generate 1d decay mask, the result is l*l
        """
        index = torch.arange(l).to(self.decay)
        mask = index[:, None] - index[None, :]
        mask = mask.abs()
        mask = mask * self.decay[:, None, None]
        return mask

    def forward(self, HW_tuple, depth_map, split_or_not=False):
        """
        depth_map: depth patches
        HW_tuple: (H, W)
        H * W == l
        """
        depth_map = F.interpolate(depth_map, size=HW_tuple, mode="bilinear", align_corners=False)

        if split_or_not:
            index = torch.arange(HW_tuple[0] * HW_tuple[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            sin = sin.reshape(HW_tuple[0], HW_tuple[1], -1)
            cos = torch.cos(index[:, None] * self.angle[None, :])
            cos = cos.reshape(HW_tuple[0], HW_tuple[1], -1)

            mask_d_h = self.generate_1d_depth_decay(HW_tuple[0], HW_tuple[1], depth_map.transpose(-2, -1))
            mask_d_w = self.generate_1d_depth_decay(HW_tuple[1], HW_tuple[0], depth_map)

            mask_h = self.generate_1d_decay(HW_tuple[0])
            mask_w = self.generate_1d_decay(HW_tuple[1])

            mask_h = self.weight[0] * mask_h.unsqueeze(0).unsqueeze(2) + self.weight[1] * mask_d_h
            mask_w = self.weight[0] * mask_w.unsqueeze(0).unsqueeze(2) + self.weight[1] * mask_d_w

            geo_prior = ((sin, cos), (mask_h, mask_w))

        else:
            index = torch.arange(HW_tuple[0] * HW_tuple[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            sin = sin.reshape(HW_tuple[0], HW_tuple[1], -1)
            cos = torch.cos(index[:, None] * self.angle[None, :])
            cos = cos.reshape(HW_tuple[0], HW_tuple[1], -1)
            mask = self.generate_pos_decay(HW_tuple[0], HW_tuple[1])

            mask_d = self.generate_depth_decay(HW_tuple[0], HW_tuple[1], depth_map)
            mask = self.weight[0] * mask + self.weight[1] * mask_d

            geo_prior = ((sin, cos), mask)

        return geo_prior


class Decomposed_GSA(nn.Module):
    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim**-0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)

        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def forward(self, x: torch.Tensor, rel_pos, split_or_not=False):
        bsz, h, w, _ = x.size()

        (sin, cos), (mask_h, mask_w) = rel_pos

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k = k * self.scaling
        q = q.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)  # (b n h w d1)
        k = k.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)  # (b n h w d1)
        qr = angle_transform(q, sin, cos)
        kr = angle_transform(k, sin, cos)

        qr_w = qr.transpose(1, 2)
        kr_w = kr.transpose(1, 2)
        v = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 1, 3, 2, 4)

        qk_mat_w = qr_w @ kr_w.transpose(-1, -2)
        qk_mat_w = qk_mat_w + mask_w.transpose(1, 2)
        qk_mat_w = torch.softmax(qk_mat_w, -1)
        v = torch.matmul(qk_mat_w, v)

        qr_h = qr.permute(0, 3, 1, 2, 4)
        kr_h = kr.permute(0, 3, 1, 2, 4)
        v = v.permute(0, 3, 2, 1, 4)

        qk_mat_h = qr_h @ kr_h.transpose(-1, -2)
        qk_mat_h = qk_mat_h + mask_h.transpose(1, 2)
        qk_mat_h = torch.softmax(qk_mat_h, -1)
        output = torch.matmul(qk_mat_h, v)

        output = output.permute(0, 3, 1, 2, 4).flatten(-2, -1)
        output = output + lepe
        output = self.out_proj(output)
        return output

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)


class Full_GSA(nn.Module):
    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim**-0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)
        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def forward(self, x: torch.Tensor, rel_pos, split_or_not=False):
        """
        x: (b h w c)
        rel_pos: mask: (n l l)
        """
        bsz, h, w, _ = x.size()
        (sin, cos), mask = rel_pos
        assert h * w == mask.size(3)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k = k * self.scaling
        q = q.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        k = k.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        qr = angle_transform(q, sin, cos)
        kr = angle_transform(k, sin, cos)

        qr = qr.flatten(2, 3)
        kr = kr.flatten(2, 3)
        vr = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        vr = vr.flatten(2, 3)
        qk_mat = qr @ kr.transpose(-1, -2)
        qk_mat = qk_mat + mask
        qk_mat = torch.softmax(qk_mat, -1)
        output = torch.matmul(qk_mat, vr)
        output = output.transpose(1, 2).reshape(bsz, h, w, -1)
        output = output + lepe
        output = self.out_proj(output)
        return output

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)


class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        activation_fn=F.gelu,
        dropout=0.0,
        activation_dropout=0.0,
        layernorm_eps=1e-6,
        subln=False,
        subconv=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = activation_fn
        self.activation_dropout_module = torch.nn.Dropout(activation_dropout)
        self.dropout_module = torch.nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim)
        self.ffn_layernorm = nn.LayerNorm(ffn_dim, eps=layernorm_eps) if subln else None
        self.dwconv = DWConv2d(ffn_dim, 3, 1, 1) if subconv else None

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        if self.ffn_layernorm is not None:
            self.ffn_layernorm.reset_parameters()

    def forward(self, x: torch.Tensor):
        """
        input shape: (b h w c)
        """
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout_module(x)
        residual = x
        if self.dwconv is not None:
            x = self.dwconv(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = x + residual
        x = self.fc2(x)
        x = self.dropout_module(x)
        return x


class RGBD_Block(nn.Module):
    def __init__(
        self,
        split_or_not: str,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        drop_path=0.0,
        layerscale=False,
        layer_init_values=1e-5,
        init_value=2,
        heads_range=4,
    ):
        num_heads = 1
        super().__init__()
        self.embed_dim = embed_dim
        self.cnn_pos_encode = DWConv2d(embed_dim, 3, 1, 1)
        self.decay = torch.abs(torch.log(
            1 - 2 ** (-init_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads)
        )).cuda()

        # Gauusian
        self.wid_gauusian_w = nn.Linear(self.embed_dim,self.embed_dim)
        self.hei_gauusian_w = nn.Linear(self.embed_dim,self.embed_dim)
        self.gauusian_gen = TempMoE(d_model=self.embed_dim,topK=1)

        # Axis-attention
        self.xq_axis_proj = nn.Linear(self.embed_dim,self.embed_dim)
        self.yq_axis_proj = nn.Linear(self.embed_dim,self.embed_dim)

        self.xk_axis_proj = nn.Linear(self.embed_dim,self.embed_dim)
        self.yk_axis_proj = nn.Linear(self.embed_dim,self.embed_dim)

        self.v_proj = nn.Linear(self.embed_dim,self.embed_dim)

    def geo_gen(self,wid_x,hei_x):
        b,h,w,d=wid_x.size()

        wid_x = wid_x.reshape(b,h,w*d)
        hei_x = hei_x.reshape(b,w,h*d)

        # wid_geo_prior
        wid_x_1 = wid_x.repeat(1,1,h).reshape(b,h*h,w*d)
        wid_x_2 = wid_x.repeat(1,h,1).reshape(b,h*h,w*d)
        wid_geo_prior = F.cosine_similarity(wid_x_1,wid_x_2,dim=-1).reshape(b,h,h)   #[B,H,H]

        # hei_geo_prior
        hei_x_1 = hei_x.repeat(1,1,w).reshape(b,w*w,h*d)
        hei_x_2 = hei_x.repeat(1,w,1).reshape(b,w*w,h*d)
        hei_geo_prior = F.cosine_similarity(hei_x_1,hei_x_2,dim=-1).reshape(b,w,w)   #[B,W,W]
         
        return torch.abs(wid_geo_prior), torch.abs(hei_geo_prior)


    def forward(self, x: torch.Tensor, split_or_not=False):
        x = x + self.cnn_pos_encode(x)
        b, h, w, d = x.size()

        cls_fea_1 = torch.mean(x,dim=1,keepdim=True)
        cls_fea = torch.mean(cls_fea_1,dim=2,keepdim=True)
        cls_fea = cls_fea.reshape(b,d)

        wid_cls = self.wid_gauusian_w(cls_fea)
        hei_cls = self.hei_gauusian_w(cls_fea)

        wid_gauusian = self.gauusian_gen(wid_cls,x.reshape(b,h*w,d)).reshape(b,h,w,1)
        hei_gauusian = self.gauusian_gen(hei_cls,x.reshape(b,h*w,d)).reshape(b,h,w,1)

        wid_x = wid_gauusian * x
        hei_x = hei_gauusian * x

        wid_geo_prior, hei_geo_prior = self.geo_gen(wid_x,hei_x) #[B,H,H],[B,W,W]

        # x-axis attention
        x_q = self.xq_axis_proj(x).reshape(b,h,w*d)
        x_k = self.xk_axis_proj(x).reshape(b,h,w*d)
        x_res = nn.Softmax(dim=-1)((x_q @ x_k.permute(0,2,1))/math.sqrt(w*d))  #[B,H,H]
        x_res = x_res * self.decay**(wid_geo_prior)

        # y-axis attention
        y_q = self.yq_axis_proj(x).reshape(b,w,h*d)
        y_k = self.yk_axis_proj(x).reshape(b,w,h*d)
        y_res = nn.Softmax(dim=-1)((y_q @ y_k.permute(0,2,1))/math.sqrt(h*d))  #[B,W,W]
        y_res = y_res * self.decay ** (hei_geo_prior)

        v = self.v_proj(x).reshape(b,h,w*d)

        v_res = (x_res @ v).reshape(b,w,h*d)

        res = (y_res @ v_res).reshape(b,h,w,d)

        return res


class BasicLayer(nn.Module):
    """
    A basic RGB-D layer in DFormerv2.
    """

    def __init__(
        self,
        embed_dim,
        out_dim,
        depth,
        num_heads,
        init_value: float,
        heads_range: float,
        ffn_dim=96.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        split_or_not=False,
        downsample: PatchMerging = None,
        use_checkpoint=False,
        layerscale=False,
        layer_init_values=1e-5,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.split_or_not = split_or_not

        # build blocks
        self.blocks = nn.ModuleList(
            [
                RGBD_Block(
                    split_or_not,
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    drop_path[i] if isinstance(drop_path, list) else drop_path,
                    layerscale,
                    layer_init_values,
                    init_value=init_value,
                    heads_range=heads_range,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=embed_dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        b, h, w, d = x.size()
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x=x, split_or_not=self.split_or_not)
            else:
                x = blk(x, split_or_not=self.split_or_not)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class dformerv2(nn.Module):
    def __init__(
        self,
        out_indices=(0, 1, 2),
        embed_dims=[64, 128, 256],
        depths=[2, 2, 8],
        num_heads=[4, 4, 8],
        init_values=[2, 2, 2],
        heads_ranges=[4, 4, 6],
        mlp_ratios=[4, 4, 3],
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        use_checkpoint=False,
        projection=1024,
        norm_cfg=None,
        layerscales=[False, False, False],
        layer_init_values=1e-6,
        norm_eval=True,
    ):
        super().__init__()
        self.out_indices = out_indices
        self.num_layers = len(depths)
        self.embed_dim = embed_dims[0]
        self.patch_norm = patch_norm
        self.num_features = embed_dims[-1]
        self.mlp_ratios = mlp_ratios
        self.norm_eval = norm_eval

        # patch embedding
        self.patch_embed = PatchEmbed(
            in_chans=3, embed_dim=embed_dims[0], norm_layer=norm_layer if self.patch_norm else None
        )

        # drop path rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        self.layers = nn.ModuleList()

        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                embed_dim=embed_dims[i_layer],
                out_dim=embed_dims[i_layer + 1] if (i_layer < self.num_layers - 1) else None,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                init_value=init_values[i_layer],
                heads_range=heads_ranges[i_layer],
                ffn_dim=int(mlp_ratios[i_layer] * embed_dims[i_layer]),
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                split_or_not=(i_layer != 3),
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                layerscale=layerscales[i_layer],
                layer_init_values=layer_init_values,
            )
            self.layers.append(layer)

        self.extra_norms = nn.ModuleList()
        layer_dim=[256,256]
        for i in range(2):
            self.extra_norms.append(nn.LayerNorm(layer_dim[i]))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            try:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            except:
                pass

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            # logger = get_root_logger()
            _state_dict = torch.load(pretrained)
            if "model" in _state_dict.keys():
                _state_dict = _state_dict["model"]
            if "state_dict" in _state_dict.keys():
                _state_dict = _state_dict["state_dict"]
            state_dict = OrderedDict()

            for k, v in _state_dict.items():
                if k.startswith("backbone."):
                    state_dict[k[9:]] = v
                else:
                    state_dict[k] = v
            print("load " + pretrained)
            load_state_dict(self, state_dict, strict=False)
            # load_checkpoint(self, pretrained, strict=False)
            # load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError("pretrained must be a str or None")

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward(self, x):
        # rgb input
        x = self.patch_embed(x)
        outs = []

        for i in range(self.num_layers):
            layer = self.layers[i]
            x = layer(x)
            if i in self.out_indices:
                if i != 0:
                    x = self.extra_norms[i - 1](x)
                out = x.permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()


def DFormerv2_S(pretrained=False, **kwargs):
    model = dformerv2(
        embed_dims=[64, 128, 256],
        depths=[3, 4, 18],
        num_heads=[4, 4, 8],
        heads_ranges=[4, 4, 6],
        **kwargs,
    )
    return model


def DFormerv2_B(pretrained=False, **kwargs):
    model = dformerv2(
        embed_dims=[80, 160, 320],
        depths=[4, 8, 25],
        num_heads=[5, 5, 10],
        heads_ranges=[5, 5, 6],
        layerscales=[False, False, True],
        layer_init_values=1e-6,
        **kwargs,
    )
    return model


def DFormerv2_L(pretrained=False, **kwargs):
    model = dformerv2(
        embed_dims=[112, 224, 448, 640],
        depths=[4, 8, 25, 8],
        num_heads=[7, 7, 14, 20],
        heads_ranges=[6, 6, 6, 6],
        layerscales=[False, False, True, True],
        layer_init_values=1e-6,
        **kwargs,
    )
    return model
