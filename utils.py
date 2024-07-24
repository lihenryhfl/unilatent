import torch
import torch.nn as nn
import torch.nn.functional as F
import xformers.ops

def pad_mask(orig_mask, prefix_len=77):
    extra_zeros = torch.ones(size=(orig_mask.shape[0], prefix_len), dtype=orig_mask.dtype, device=orig_mask.device)
    return torch.cat([extra_zeros, orig_mask], axis=1)

def pad_ids(input_ids, prefix_len=77):
    extra_zeros = torch.zeros(len(input_ids), prefix_len, dtype=torch.int64, device=input_ids.device)
    return torch.cat([extra_zeros, input_ids], axis=1)

    
class ReLength(nn.Module):
    def __init__(self, target_len, d_model, num_heads, attn_drop=0., proj_drop=0.,
                 mask_for='src', **block_kwargs):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.target_len = target_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.register_buffer('q', torch.randn(size=(1, target_len, d_model)) * 0.5)
        self.kv_linear = nn.Linear(d_model, d_model*2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

        self.mask_for = mask_for

    def forward(self, src, mask=None):
        # query: qry; key/value: src; mask: if padding tokens
        B, _, C = src.shape
        N = self.target_len

        q = self.q.tile(B, 1, 1).view(1, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(src).view(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        attn_bias = None

        # use_fp32_attention = getattr(self, 'fp32_attention', False)     # necessary for NAN loss
        # if use_fp32_attention:
        #     q, k, v = q.float(), k.float(), v.float()
        if mask is not None:
            if self.mask_for == 'src':
                attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens([N] * B, mask)
            else:
                attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens(mask, [N] * B)

        q = q.type(k.dtype)
        x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)
        x = x.view(B, -1, C)
        
        # if use_fp32_attention:
        #     x = x.to(qry.dtype)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x