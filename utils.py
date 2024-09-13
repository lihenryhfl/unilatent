import torch
import torch.nn as nn
import torch.nn.functional as F
import xformers.ops

from torch.nn.parallel import DistributedDataParallel as DDP

from transformers.modeling_utils import ModuleUtilsMixin
from transformers.configuration_utils import PretrainedConfig

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import ModelMixin
from diffusers.models.attention import BasicTransformerBlock

from data.builder import build_dataset, build_dataloader
from aspect_ratio_sampler import AspectRatioBatchSampler
from torch.utils.data import RandomSampler

import os
import subprocess

def get_metrics(fname):
    MP = int(os.environ.get('MASTER_PORT', '29900'))
    command = f"MASTER_PORT={MP+1} cd /mnt/bn/us-aigc-temp/henry/clipscore/; bash run_clipscore.sh {fname}"
    print("RUNNING COMMAND", command)
    output = subprocess.run(command, shell=True, capture_output=True).stdout.decode("utf-8")
    metric_names = ['bleu-1', 'bleu-2', 'bleu-3', 'bleu-4', 'meteor', 'rouge', 'cider', 'clipscore']
    
    return_dict = {}
    for line in output.split('\n'):
        for metric_name in metric_names:
            metric_name = metric_name.lower()
            if metric_name in line.lower() and line.lower() != 'refclipscore':
                return_dict[metric_name] = float(line.split(': ')[1])

    assert len(return_dict) == len(metric_names), f"ERROR! return_dict: {return_dict} \n output: {output}"

    return return_dict

def unwrap(accelerator, pipe):
    pipe.transformer = accelerator.unwrap_model(pipe.transformer)
    pipe.text_encoder = accelerator.unwrap_model(pipe.text_encoder)
    pipe.text_encoder_2 = accelerator.unwrap_model(pipe.text_encoder_2)
    pipe.clip_image_encoder = accelerator.unwrap_model(pipe.clip_image_encoder)
    pipe.text_decoder = accelerator.unwrap_model(pipe.text_decoder)
    pipe.vae = accelerator.unwrap_model(pipe.vae)
    pipe.image_encoder_adapter = accelerator.unwrap_model(pipe.image_encoder_adapter)
    pipe.image_decoder_adapter = accelerator.unwrap_model(pipe.image_decoder_adapter)
    pipe.layer_aggregator = accelerator.unwrap_model(pipe.layer_aggregator)
    return pipe

def pad_mask(orig_mask, prefix_len=77):
    extra_zeros = torch.ones(size=(orig_mask.shape[0], prefix_len), dtype=orig_mask.dtype, device=orig_mask.device)
    return torch.cat([extra_zeros, orig_mask], axis=1)

def pad_ids(input_ids, prefix_len=77):
    extra_zeros = torch.zeros(len(input_ids), prefix_len, dtype=torch.int64, device=input_ids.device)
    return torch.cat([extra_zeros, input_ids], axis=1)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_suffix_ids(batch, tokenizer, device):
    idxs = batch[-1]['data_idx']
    # assert (idxs < len(train_config['roots'])).all()
    suffix_text = [f'<|dataset{i}|>' for i in idxs]
    suffix_ids = tokenizer(suffix_text, return_tensors='pt', max_length=1).input_ids.to(device)
    return suffix_ids

def get_dataloader(args, data_config, val=False):
    batch_size = 1 if val else args.batch_size
    kwargs = {}
    if hasattr(args, 'image_size') and args.image_size > 0:
        resolution = args.image_size
        aspect_ratio_type = f'ASPECT_RATIO_{args.image_size}' if args.image_size in [256, 512] else 'ASPECT_RATIO_512'
        data_config['type'] = 'FlexibleInternalData'
        kwargs['return_image_id'] = val
    else:
        resolution = 512
        aspect_ratio_type = 'ASPECT_RATIO_512'
        data_config['type'] = 'FlexibleInternalDataMS'
        kwargs['return_image_id'] = val
    
    dataset = build_dataset(
        data_config, resolution=resolution, aspect_ratio_type=aspect_ratio_type,
        real_prompt_ratio=1.0, max_length=77, **kwargs
    )
    if hasattr(args, 'image_size') and args.image_size > 0:
        dataloader = build_dataloader(dataset, batch_size=batch_size, num_workers=10)
    else:
        batch_sampler = AspectRatioBatchSampler(sampler=RandomSampler(dataset), dataset=dataset,
                                            batch_size=batch_size, aspect_ratios=dataset.aspect_ratio, drop_last=True,
                                            ratio_nums=dataset.ratio_nums, valid_num=0)
        dataloader = build_dataloader(dataset, batch_sampler=batch_sampler, num_workers=10)
    
    return dataloader

class GradientFixer:
    def __init__(self, decoder, tokenizer, dtype=torch.float32):
        self.decoder = decoder
        self.tokenizer = tokenizer

        weight = self.decoder.transformer.transformer.wte.weight
        
        mask = torch.zeros(size=weight.shape, dtype=dtype)
        for token, token_id in tokenizer.get_added_vocab().items():
            mask[token_id, :] = 1.
        self.gradient_mask = mask

        for n, p in self.decoder.named_parameters():
            if 'wte' in n or 'prefix' in n:
                p.requires_grad = True
                # print(f"Training {n}.")
            else:
                p.requires_grad = False
                # print(f"Not training {n}.")

    def __getattr__(self, name):
        """
        Deals with issues accessing the model when it is wrapped by a DDP class.
        """
        print(name)
        obj = self.__getattribute__(name)
        if isinstance(obj, DDP):
            obj = obj.module

        return obj

    def fix_gradients(self):
        with torch.no_grad():
            for n, p in self.decoder.named_parameters():
                if 'wte' in n:
                    if p.grad is not None:
                        self.gradient_mask = self.gradient_mask.to(p.device)
                        p.grad *= self.gradient_mask
                    else:
                        print(f"In FrozenDecoderTrainableDataTokenWrapper: {n} has no gradient!")
                else:
                    if 'prefix' not in n:
                        assert not p.requires_grad, f"{n}"
                    


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_drop=0., proj_drop=0.,
                 mask_for='src',
                **block_kwargs):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model*2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

        self.mask_for = mask_for

    def forward(self, qry, src, mask=None):
        input_shape = qry.shape
        # query: qry; key/value: src; mask: if padding tokens
        if self.mask_for == 'src':
            B, N, C = qry.shape
        else:
            B, N, C = src.shape

        q = self.q_linear(qry).view(1, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(src).view(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        attn_bias = None

        if mask is not None:
            if self.mask_for == 'src':
                attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens([N] * B, mask)
            else:
                attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens(mask, [N] * B)

        x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)
        x = x.view(input_shape[0], -1, input_shape[-1])
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv_linear = nn.Linear(d_model, d_model*3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = input_shape = x.shape

        qkv = self.qkv_linear(x).view(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        # aggregate queries into one
        # q = q.mean(axis=N)

        x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p)
        x = x.view(input_shape[0], -1, input_shape[-1])
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class LayerAggregator(ModelMixin, ConfigMixin, ModuleUtilsMixin):
    @register_to_config
    def __init__(self, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.layer_logits = torch.nn.Parameter(torch.randn(n_layers) * 0.01)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, layers):
        B, N, C_prime = layers.shape
        assert C_prime % self.n_layers == 0
        C = C_prime // self.n_layers
        # split on channel dimension to regain layers
        layers = layers.reshape(B, N, self.n_layers, C)
        layer_gates = self.softmax(self.layer_logits).reshape(self.n_layers, 1)
        x = layers * layer_gates
        x = x.sum(axis=-2)

        return x

class AttentionLayerAggregator(LayerAggregator):
    @register_to_config
    def __init__(self, n_layers, prefix_dim):
        super().__init__(n_layers)
        self.adapters = nn.ModuleList([
            EmbedAdapterV3(prefix_dim, prefix_dim, -1, embed_pool=False, use_attn=True) for _ in range(n_layers)
        ])

    def forward(self, layers):
        B, N, C_prime = layers.shape
        assert C_prime % self.n_layers == 0
        C = C_prime // self.n_layers

        # split on channel dimension to regain layers
        layers = layers.reshape(B, N, self.n_layers, C)

        new_layers = []
        for i in range(self.n_layers):
            layer = layers[..., i, :]
            layer = self.adapters[i](layer)
            new_layers.append(layer)

        layers = torch.stack(new_layers, axis=-2)
        assert layers.shape == (B, N, self.n_layers, C), f"{layers.shape}, {(B, N, self.n_layers, C)}"

        layer_gates = self.softmax(self.layer_logits).reshape(self.n_layers, 1)
        x = layers * layer_gates
        x = x.sum(axis=-2)

        return x

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
            self.relength = ReLength(output_length, output_dim, n_heads)

    def forward(self, x):
        x = self.redimension(x)

        if hasattr(self, 'relength'):
            x = self.relength(x)

        return x

class EmbedAdapter(ModelMixin, ConfigMixin, ModuleUtilsMixin):
    @register_to_config
    def __init__(self, input_dim, output_dim, output_length=-1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_length = output_length
            
    def forward(self, x, x_pooled=None):
        raise NotImplementedError
        
class EmbedAdapterV1(EmbedAdapter):
    @register_to_config
    def __init__(self, input_dim, output_dim, output_length=-1, n_heads=16, use_redimension=True,
                 embed_pool=False):
        super().__init__(input_dim=input_dim, output_dim=output_dim, output_length=output_length)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_length = output_length
        if use_redimension:
            self.redimension = ReDimension(input_dim, output_dim)
            self.embed_pool = embed_pool
            if embed_pool:
                self.pooled_redimension = ReDimension(input_dim, output_dim)

        if output_length > -1:
            self.relength = ReLength(output_length, output_dim, n_heads)
            
    def forward(self, x, x_pooled=None):
        if hasattr(self, 'redimension'):
            x = self.redimension(x)

        if hasattr(self, 'relength'):
            x = self.relength(x)

        if hasattr(self, 'pooled_redimension') and self.embed_pool:
            x_pooled = self.pooled_redimension(x_pooled)
            return x, x_pooled

        return x

class EmbedAdapterV2(EmbedAdapter):
    @register_to_config
    def __init__(self, input_dim, output_dim, output_length=-1, n_heads=16,
                 use_attn=False, embed_pool=False, dropout=0.):
        super().__init__(input_dim=input_dim, output_dim=output_dim, output_length=output_length)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_length = output_length
        self.embed_pool = embed_pool

        self.redimension = ReDimension(input_dim, output_dim)
        attn = BasicTransformerBlock(output_dim, n_heads, output_dim // n_heads, dropout=dropout)
        self.attn = attn
        self.add_module('attn', attn)

        if output_length > -1:
            self.relength = ReLength(output_length, output_dim, n_heads)

    def forward(self, x, x_pooled=None):
        if self.embed_pool:
            x = torch.cat([x, x_pooled], axis=1)

        x = self.redimension(x)

        if hasattr(self, 'attn'):
            x = x + self.attn(x)

        if self.embed_pool:
            x, x_pooled = x[:, :-1], x[:, -1:]
            
        if hasattr(self, 'relength'):
            x = self.relength(x)
        
        if self.embed_pool:
            return x, x_pooled
        else:
            return x

class EmbedAdapterV3(EmbedAdapter):
    @register_to_config
    def __init__(self, input_dim, output_dim, output_length=-1, n_heads=16,
                 use_attn=False, embed_pool=False, dropout=0.):
        super().__init__(input_dim=input_dim, output_dim=output_dim, output_length=output_length)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_length = output_length
        self.embed_pool = embed_pool

        self.redimension = ReDimension(input_dim, output_dim)
        attn = BasicTransformerBlock(output_dim, n_heads, output_dim // n_heads, dropout=dropout)
        self.attn = attn
        self.add_module('attn', attn)

        if output_length > -1:
            self.relength = ReLength(output_length, output_dim, n_heads)

    def forward(self, x, x_pooled=None):
        if self.embed_pool:
            x = torch.cat([x, x_pooled], axis=1)

        x = self.redimension(x)

        if hasattr(self, 'attn'):
            x = self.attn(x)

        if self.embed_pool:
            x, x_pooled = x[:, :-1], x[:, -1:]
            
        if hasattr(self, 'relength'):
            x = self.relength(x)
        
        if self.embed_pool:
            return x, x_pooled
        else:
            return x


class SoftPrompter(ModelMixin, ConfigMixin, ModuleUtilsMixin):
    @register_to_config
    def __init__(self, d_model, length=1, std=0.):
        super().__init__()
        self.register_buffer('soft_prompt', torch.randn(size=(1, length, d_model)) * std)
        self.register_buffer('pooled_soft_prompt', torch.randn(size=(1, 1, d_model)) * std)

    def forward(self, x, x_pooled):
        B, N, C = x.shape
        _, _, _ = x_pooled.shape
        assert C == self.soft_prompt.shape[-1], f"{C.shape}, {self.soft_prompt.shape}"
        new_x = self.soft_prompt.tile(B, 1, 1)
        new_x_pooled = self.pooled_soft_prompt.tile(B, 1, 1)
        return new_x, new_x_pooled
        

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