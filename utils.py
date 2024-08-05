import torch
import torch.nn as nn
import torch.nn.functional as F
import xformers.ops

from transformers.modeling_utils import ModuleUtilsMixin
from transformers.configuration_utils import PretrainedConfig

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import ModelMixin

def pad_mask(orig_mask, prefix_len=77):
    extra_zeros = torch.ones(size=(orig_mask.shape[0], prefix_len), dtype=orig_mask.dtype, device=orig_mask.device)
    return torch.cat([extra_zeros, orig_mask], axis=1)

def pad_ids(input_ids, prefix_len=77):
    extra_zeros = torch.zeros(len(input_ids), prefix_len, dtype=torch.int64, device=input_ids.device)
    return torch.cat([extra_zeros, input_ids], axis=1)

class ReLength(nn.Module):
    def __init__(self, target_len, d_model, num_heads, attn_drop=0., proj_drop=0.,
                 mask_for='src'):
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
        B, _, C = src.shape
        N = self.target_len

        q = self.q.tile(B, 1, 1).view(1, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(src).view(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        attn_bias = None

        if mask is not None:
            if self.mask_for == 'src':
                attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens([N] * B, mask)
            else:
                attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens(mask, [N] * B)

        q = q.type(k.dtype)
        x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)
        x = x.view(B, -1, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class ReDimension(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        return self.linear(x)

class Adapter(nn.Module):
    def __init__(self, input_dim, output_dim, input_length, output_length, n_heads=16):
        super().__init__()
        self.redimension = ReDimension(input_dim, output_dim)

        if input_length != output_length:
            self.relength = ReLength(target_len, output_dim, n_heads)

    def forward(self, x):
        x = self.redimension(x)

        if hasattr(self, 'relength'):
            x = self.relength(x)

        return x
        
class EmbedAdapter(ModelMixin, ConfigMixin, ModuleUtilsMixin):
    @register_to_config
    def __init__(self, input_dim, output_dim, output_length=-1, n_heads=16, embed_pool=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_length = output_length
        self.redimension = ReDimension(input_dim, output_dim)
        self.embed_pool = embed_pool
        if embed_pool:
            self.pooled_redimension = ReDimension(input_dim, output_dim)

        if output_length > -1:
            self.relength = ReLength(output_length, output_dim, n_heads)

    def forward(self, x, x_pooled=None):
        x = self.redimension(x)

        if hasattr(self, 'relength'):
            x = self.relength(x)

        if self.embed_pool:
            x_pooled = self.pooled_redimension(x_pooled)
            return x, x_pooled

        return x
        

class AdapterConfig(PretrainedConfig):

    # model_type = "adapter"
    # keys_to_ignore_at_inference = ["past_key_values"]
    # attribute_map = {
    #     "hidden_size": "n_embd",
    #     "max_position_embeddings": "n_positions",
    #     "num_attention_heads": "n_head",
    #     "num_hidden_layers": "n_layer",
    # }

    def __init__(
        self,
        input_dim, 
        output_dim, 
        output_length=-1, 
        n_heads=16
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_length = output_length
        self.n_heads = n_heads

        super().__init__()


class SoftPrompter(ModelMixin, ConfigMixin, ModuleUtilsMixin):
    @register_to_config
    def __init__(self, d_model, length=1, std=0.):
        super().__init__()
        self.register_buffer('soft_prompt', torch.randn(size=(1, length, d_model)) * std)
        self.register_buffer('pooled_soft_prompt', torch.randn(size=(1, 1, d_model)) * std)

    def forward(self, x, x_pooled):
        return x + self.soft_prompt, x_pooled + self.pooled_soft_prompt
        

class SoftPrompterConfig(PretrainedConfig):

    def __init__(
        self,
        d_model, 
        length=1,
        n_heads=16
    ):
        self.d_model = d_model
        self.length = length
        self.n_heads = n_heads

        super().__init__()

def generate_captions(pipe, dataloader, save_path, sampler, sampler_kwargs={}):
    json_list = []
    progbar = tqdm(dataloader)
    for i, batch in enumerate(progbar):
        with torch.no_grad():
            decoded_text = sampler(batch, pipe, **sampler_kwargs)
        
        caption = decoded_text.strip('!').replace('<|endoftext|>', '').replace('<|EOS|>', '').strip()
        image_id = batch[-1]['image_id'].item() if 'image_id' in batch[-1] else 0
        json_list.append({'image_id': image_id, 'caption': caption})

        progbar.set_description(f"Image: {i:05d} | Predicted: {caption} | True: {batch[1][0]}")

        if (i + 1) % 50 == 0:
            with open(save_path, 'w') as f:
                test = json.dump(json_list, f)

    return json_list