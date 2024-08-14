import os
import json
import argparse
import torch
from diffusers import StableDiffusion3Pipeline
from unilatent import UniLatentPipeline, retrieve_timesteps

from data.builder import build_dataset, build_dataloader
from aspect_ratio_sampler import AspectRatioBatchSampler
from torch.utils.data import RandomSampler

from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm
from diffusers import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from transformers import (
    GPT2Tokenizer,
    CLIPVisionModel,
    CLIPImageProcessor,
)

# from caption_decoder import TextDecoder
from caption_decoder_v1 import TextDecoder
from utils import FrozenDecoderTrainableDataTokenWrapper, LayerAggregator
from utils import EmbedAdapterV1 as EmbedAdapter
# from utils import EmbedAdapterV2 as EmbedAdapter

from transformer import SD3Transformer2DModel

parser = argparse.ArgumentParser(description="Training.")
parser.add_argument('--work_dir', default='/mnt/bn/us-aigc-temp/henry/data/clip2text/', help='the dir to save logs and models')
parser.add_argument('--load_from', default='', help='the dir to load logs and models')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--index', type=int, default=750)
parser.add_argument('--n_agg', type=int, default=1)

parser.add_argument('--block_num', nargs='+', type=int, default='12')
parser.add_argument('--image_size', type=int, default=-1)
parser.add_argument('--step_offset', type=int, default=0)
parser.add_argument('--n_steps', type=int, default=200_000)
parser.add_argument('--sample_and_exit', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--clip', action='store_true')
parser.add_argument('--clip_text', action='store_true')
parser.add_argument('--no_adapter', action='store_true')
parser.add_argument('--pretrain', action='store_true')
parser.add_argument('--layer_aggregator', action='store_true')
args = parser.parse_args()

assert not (args.clip_text and args.clip)

args.block_num = [args.block_num] if not isinstance(args.block_num, list) else args.block_num
print("BLOCK NUM", args.block_num)

val_config = {
    'type': 'FlexibleInternalData',
    'roots': [
        '/mnt/bn/us-aigc-temp/henry/coco_2014/val/val2014/',
    ],
    'json_lst': [
        '/mnt/bn/us-aigc-temp/henry/test.json',
    ],
    'load_vae_feat': False,
    'load_t5_feat': False,
    'transform': 'default_train'
}
train_config = {
    'roots': [
        '/mnt/bn/us-aigc-temp/henry/coco_2014/train2014/',
        '/mnt/bn/aigc-us/zjl/laion-coco-aesthetic/data_max1024/',
        '/mnt/bn/aigc-us/zjl/recap_datacom_1b_aesthetic_subset/data/',
        '/mnt/bn/aigc-us/zjl/sharegpt4v_processed_data/data/',
        '/mnt/bn/aigc-us/zjl/openimages/data/',
    ],
    'json_lst': [
        '/mnt/bn/us-aigc-temp/henry/train.json',
        '/mnt/bn/aigc-us/zjl/laion-coco-aesthetic/data_max1024/meta_data_coco_edited.json',
        '/mnt/bn/aigc-us/zjl/recap_datacom_1b_aesthetic_subset/data/aes5_meta_data_all.json',
        '/mnt/bn/aigc-us/zjl/sharegpt4v_processed_data/data/meta_data.json',
        '/mnt/bn/aigc-us/zjl/openimages/data/meta_data.json',
    ],
    'load_vae_feat': False,
    'load_t5_feat': False,
    'transform': 'default_train'
}

if args.load_from:
    pipe = UniLatentPipeline.from_pretrained(args.load_from, torch_dtype=torch.float32)
else:
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float32)
    decoder_tokenizer = GPT2Tokenizer.from_pretrained('/mnt/bn/us-aigc-temp/henry/unilatent_weights/gpt_tokenizer/')
    decoder_tokenizer.add_special_tokens({'pad_token': decoder_tokenizer.eos_token})
    decoder_tokenizer.add_tokens([f'<|dataset{i}|>' for i in range(len(train_config['roots']))])
    pipe.decoder_tokenizer = decoder_tokenizer

    if args.image_size == -1:
        prefix_length = 512
    elif args.image_size == 256:
        prefix_length = int(args.image_size ** 2 / 16 ** 2)
        assert prefix_length == 256, prefix_length
    elif args.image_size == 512:
        prefix_length = int(args.image_size ** 2 / 16 ** 2)
        assert prefix_length == 1024, prefix_length
    else:
        print("USING IMAGE SIZE", args.image_size)
        prefix_length = int(args.image_size ** 2 / 16 ** 2)

    layer_aggregator = None
    if args.layer_aggregator:
        layer_aggregator = LayerAggregator(len(args.block_num))

    if args.clip:
        prefix_length = 259
        prefix_dim = 1024
        embed_pool = True
    elif args.clip_text:
        prefix_length += 1
        prefix_dim = 1536
        embed_pool = False
    else:
        prefix_length += 1
        prefix_dim = 1536
        embed_pool = False

    text_decoder = TextDecoder(
        prefix_length=prefix_length,
        prefix_inner_dim=prefix_dim,
        vocab_size=len(decoder_tokenizer) + len(decoder_tokenizer.get_added_vocab()))
    pipe.text_decoder = text_decoder

    pipe.clip_image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float32)
    pipe.clip_image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float32)

    transformer = SD3Transformer2DModel.from_config(pipe.transformer.config)
    transformer.load_state_dict(pipe.transformer.state_dict())
    pipe.transformer = transformer

    soft_prompter = image_encoder_adapter = None
    image_encoder_adapter = EmbedAdapter(prefix_dim, prefix_dim, prefix_length - 1, embed_pool=embed_pool)
    # image_encoder_adapter = EmbedAdapter(prefix_dim * len(args.block_num), prefix_dim, -1, embed_pool=embed_pool, use_attn=True)
    if args.no_adapter:
        image_encoder_adapter = None
        
    pipe = UniLatentPipeline(
        transformer=pipe.transformer,
        scheduler=pipe.scheduler,
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        tokenizer=pipe.tokenizer,
        text_encoder_2=pipe.text_encoder_2,
        tokenizer_2=pipe.tokenizer_2,
        clip_image_encoder=pipe.clip_image_encoder,
        clip_image_processor=pipe.clip_image_processor,
        text_decoder=pipe.text_decoder,
        decoder_tokenizer=pipe.decoder_tokenizer,
        image_encoder_adapter=image_encoder_adapter,
        soft_prompter=soft_prompter,
        layer_aggregator=layer_aggregator
    )

if args.layer_aggregator:
    layer_logits = pipe.layer_aggregator.layer_logits
    print(f"Layer logits: {layer_logits}")

def get_dataloader(data_config, val=False):
    batch_size = 1 if val else args.batch_size
    kwargs = {}
    if args.image_size > 0:
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
    if args.image_size > 0:
        dataloader = build_dataloader(dataset, batch_size=batch_size, num_workers=10)
    else:
        batch_sampler = AspectRatioBatchSampler(sampler=RandomSampler(dataset), dataset=dataset,
                                            batch_size=batch_size, aspect_ratios=dataset.aspect_ratio, drop_last=True,
                                            ratio_nums=dataset.ratio_nums, valid_num=0)
        dataloader = build_dataloader(dataset, batch_sampler=batch_sampler, num_workers=10)
    
    return dataloader

val_loader = get_dataloader(val_config, val=True)

if args.debug or not args.sample_and_exit:
    dataloader = get_dataloader(train_config)
else:
    dataloader = val_loader

models = [pipe.text_decoder]
ft_models = []

if args.clip:
    ft_models.append(pipe.clip_image_encoder)

if args.clip_text:
    ft_models.extend([pipe.text_encoder, pipe.text_encoder_2])

if args.layer_aggregator:
    models.append(pipe.layer_aggregator)

num_steps = args.n_steps
optimizer = torch.optim.AdamW(lr=2e-5, params=pipe.parameters(models=models))
lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=2000 + args.step_offset,
            num_training_steps=num_steps,
        )

if ft_models:
    optimizer.add_param_group(dict(params=pipe.parameters(models=ft_models), lr=1e-6))

for p in pipe.parameters():
    p.requires_grad = False

for p in pipe.parameters(models=models + ft_models):
    p.requires_grad = True

accelerator = Accelerator(
        mixed_precision='fp16',
    )

pipe = pipe.to(accelerator.device)
(
    optimizer, 
    lr_scheduler,
    pipe.transformer,
    pipe.text_encoder, 
    pipe.text_encoder_2,
    pipe.clip_image_encoder,
    pipe.text_decoder,
    pipe.vae,
    pipe.image_encoder_adapter,
    pipe.soft_prompter,
) = accelerator.prepare(
    optimizer, 
    lr_scheduler,
    pipe.transformer,
    pipe.text_encoder, 
    pipe.text_encoder_2,
    pipe.clip_image_encoder,
    pipe.text_decoder,
    pipe.vae,
    pipe.image_encoder_adapter,
    pipe.soft_prompter,
)

if args.pretrain:
    pipe.wrapped_text_decoder = FrozenDecoderTrainableDataTokenWrapper(pipe.text_decoder)
else:
    pipe.wrapped_text_decoder = pipe.text_decoder

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

print(f"TOTAL TRANSFORMER LAYERS: {len(pipe.transformer.transformer_blocks)} | OUR CHOSEN BLOCK: {args.block_num}")

def get_suffix_ids(batch):
    idxs = batch[-1]['data_idx']
    assert (idxs < len(train_config['roots'])).all()
    suffix_text = [f'<|dataset{i}|>' for i in idxs]
    suffix_ids = pipe.decoder_tokenizer(suffix_text, return_tensors='pt', max_length=1).input_ids.to(accelerator.device)
    return suffix_ids

def sample(batch):
    with torch.no_grad():
        if len(batch[0]) > 1:
            print(f"Sample batch size is large ({len(batch[0])})! Is this really what we want?")
        image, prompt = batch[0].to(accelerator.device), batch[1]
        index = torch.zeros(size=(len(image),), dtype=torch.long) + args.index
        suffix_input_ids = get_suffix_ids(batch)
        if args.clip:
            embeds, pooled_embeds = pipe.encode_image(image, dtype=torch.float16)
        elif args.clip_text:
            embeds, pooled_embeds = pipe.encode_text(prompt)
        else:
            embeds, pooled_embeds = pipe.dift_features(
                image, index, return_layers=args.block_num, dataset_conditioning=True)
        embeds = torch.cat([embeds, pooled_embeds], axis=1)
        decoded_tokens = pipe.text_decoder.generate_captions(embeds, 
                            eos_token_id=pipe.decoder_tokenizer.eos_token_id, device=accelerator.device,
                            suffix_input_ids=suffix_input_ids
                            )[0]
        decoded_text = pipe.decoder_tokenizer.batch_decode(decoded_tokens)
    return decoded_text

iter_val_loader = iter(val_loader)

if args.sample_and_exit:
    save_path = os.path.join(args.work_dir, 'captions.json')
    print("Saving to", save_path)
    json_list = []
    progbar = tqdm(val_loader)
    # progbar = tqdm(dataloader)
    for i, batch in enumerate(progbar):
        decoded_text = sample(batch)[0]
        
        caption = decoded_text.strip('!').replace('<|endoftext|>', '').replace('<|EOS|>', '').strip()
        image_id = batch[-1]['image_id'].item() #if 'image_id' in batch[-1] else 0
        json_list.append({'image_id': image_id, 'caption': caption})
        progbar.set_description(f"Image: {i:05d} | id: {image_id} | Predicted: {caption} | True: {batch[1][0]}")

        if (i + 1) % 100 == 0:
            with open(save_path, 'w') as f:
                json.dump(json_list, f)
else:
    losses = []
    step = 0
    while step < args.step_offset:
        lr_scheduler.step()
        step += 1
    while step < num_steps:
        progbar = tqdm(dataloader, disable=not accelerator.is_main_process)
        for batch in progbar:
            if step >= num_steps:
                break
            optimizer.zero_grad()
            
            # prepare data
            image, prompt = batch[0].to(accelerator.device), batch[1]
            batch[1] = [x.replace('<|endoftext|>', '') for x in batch[1]]
            index = torch.zeros(size=(len(image),), dtype=torch.long) + args.index

            # run model
            suffix_input_ids = get_suffix_ids(batch)
            if args.clip:
                embeds, pooled_embeds = pipe.encode_image(image, dtype=torch.float16)
            else:
                if args.clip_text:
                    embeds, pooled_embeds = pipe.encode_text(prompt)
                else:
                    embeds = pooled_embeds = None

                embeds, pooled_embeds = pipe.dift_features(image, index, return_layers=args.block_num, 
                    dataset_conditioning=True, num_aggregation_steps=args.n_agg, embed=embeds, pooled_embed=pooled_embeds)

            loss = pipe.embed_to_decoder(embeds, pooled_embeds, prompt, suffix_input_ids=suffix_input_ids)
            accelerator.backward(loss)

            for n, p in pipe.named_parameters():
                if p.grad is not None:
                    torch.nan_to_num(p.grad, nan=0, posinf=1e5, neginf=-1e5, out=p.grad)

            optimizer.step()
            lr_scheduler.step()
            
            losses.append(accelerator.gather(loss).detach().cpu())
            assert losses
            progbar.set_description(f"loss: {torch.stack(losses).mean().item():.3f}, lr: {get_lr(optimizer)}")

            if accelerator.is_main_process and (step + 1) % 500 == 0:
                if (step + 1) % 5000 == 0:
                    pipe.save_pretrained(f'{args.work_dir}/step_{step}/')
                    print(f"Saved model to directory {f'{args.work_dir}/step_{step}/'}")
                elif (step + 1) % 1000 == 0:
                    pipe.save_pretrained(f'{args.work_dir}/current/')

                batch = next(iter_val_loader)
                decoded_text = sample(batch)
                print(
                    f"Recon: {decoded_text[0].strip('!').replace('<|endoftext|>', '').replace('<|EOS|>', '')} | "
                    f"True: {batch[1][0]}"
                )
                if args.layer_aggregator:
                    layer_logits = pipe.layer_aggregator.layer_logits
                    print(f"Layer logits: {layer_logits}")

                losses = []
            
            step += 1
    