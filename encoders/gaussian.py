import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Union, Optional

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
                data: Tensor,                      # [B, T, D] Audio | Video,
                sub_data: Optional[Tensor] = None, # [[B, T, D], [B, T, D]] Patch(Audio | Video)
    ) -> Union[Tensor, List[Tensor]]:
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
        
        if sub_data is not None:
            a_data = sub_data[0].permute(1, 0, 2)                                             # [T, B, C]
            a_data = data + a_data
            a_outs = torch.stack([exprt(a_data) for exprt in self.experts], dim=2)            # [T, B, N_EXPERTS, C]
            a_outs = self.get_output(a_outs, gauss_weight, topk_inds, topk_probs, (B, T, C))  # [B, 1, C]
            v_data = sub_data[1].permute(1, 0, 2)                                             # [T, B, C]
            v_data = data + v_data
            v_outs = torch.stack([exprt(v_data) for exprt in self.experts], dim=2)            # [T, B, N_EXPERTS, C]
            v_outs = self.get_output(v_outs, gauss_weight, topk_inds, topk_probs, (B, T, C))  # [B, 1, C]
            return self.anorm(a_outs), self.vnorm(v_outs)
        else:
            main_outs = torch.stack([exprt(data) for exprt in self.experts], dim=2)                # [T, B, N_EXPERTS, C]
            main_outs = self.get_output(main_outs, gauss_weight, topk_inds, topk_probs, (B, T, C)) # [B, 1, C]
            return self.norm(main_outs)