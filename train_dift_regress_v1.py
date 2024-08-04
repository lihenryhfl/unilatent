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
from utils import ReLength, EmbedAdapter, SoftPrompter

from transformer import SD3Transformer2DModel

parser = argparse.ArgumentParser(description="Training.")
parser.add_argument('--work_dir', default='/mnt/bn/us-aigc-temp/henry/data/clip2text/', help='the dir to save logs and models')
parser.add_argument('--load_from', default='', help='the dir to load logs and models')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--index', type=int, default=750)

parser.add_argument('--block_num', nargs='+', type=int, default='12')
parser.add_argument('--image_size', type=int, default=-1)
parser.add_argument('--sample_and_exit', action='store_true')
parser.add_argument('--dataset_conditioning', action='store_true')
parser.add_argument('--clip', action='store_true')
args = parser.parse_args()

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
        
    ],
    'json_lst': [
        '/mnt/bn/us-aigc-temp/henry/train.json',
        '/mnt/bn/aigc-us/zjl/laion-coco-aesthetic/data_max1024/meta_data_coco_edited.json',
        
    ],
    'load_vae_feat': False,
    'load_t5_feat': False,
    'transform': 'default_train'
}

if args.dataset_conditioning:
    train_config['roots'].append('/mnt/bn/aigc-us/zjl/recap_datacom_1b_aesthetic_subset/data/')
    train_config['json_lst'].append('/mnt/bn/aigc-us/zjl/recap_datacom_1b_aesthetic_subset/data/aes5_meta_data_all.json')

if args.load_from:
    pipe = UniLatentPipeline.from_pretrained(args.load_from, torch_dtype=torch.float32)
else:
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float32)
    add_vocab_size = 1
    decoder_tokenizer = GPT2Tokenizer.from_pretrained('/mnt/bn/us-aigc-temp/henry/unilatent_weights/gpt_tokenizer/')
    decoder_tokenizer.add_special_tokens({'pad_token': decoder_tokenizer.eos_token})
    if args.dataset_conditioning:
        decoder_tokenizer.add_tokens([f'<|dataset{i}|>' for i in range(len(train_config['roots']))])
        add_vocab_size += len(train_config['roots'])
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

    if args.clip:
        prefix_length = 259 if args.dataset_conditioning else 258
        prefix_dim = 1024
    else:
        prefix_dim = 1536 * len(args.block_num)

    text_decoder = TextDecoder(
        prefix_length=prefix_length,
        prefix_inner_dim=prefix_dim,
        vocab_size=decoder_tokenizer.vocab_size + add_vocab_size)
    pipe.text_decoder = text_decoder

    pipe.clip_image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float32)
    pipe.clip_image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float32)

    transformer = SD3Transformer2DModel.from_config(pipe.transformer.config)
    transformer.load_state_dict(pipe.transformer.state_dict())
    pipe.transformer = transformer

    # image_encoder_adapter = EmbedAdapter(input_dim=1536, output_dim=1536, output_length=511, n_heads=16)
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
    )

def get_dataloader(data_config, val=False):
    batch_size = 1 if val else args.batch_size
    kwargs = {}
    if args.image_size > 0:
        resolution = args.image_size
        aspect_ratio_type = f'ASPECT_RATIO_{args.image_size}' if args.image_size in [256, 512] else 'ASPECT_RATIO_512'
        data_config['type'] = 'FlexibleInternalData'
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

if not args.sample_and_exit:
    dataloader = get_dataloader(train_config)
else:
    dataloader = val_loader

num_epochs = 2

models = [pipe.text_decoder]
if args.clip:
    ft_models = [pipe.clip_image_encoder]
else:
    ft_models = []

if args.clip:
    optimizer = torch.optim.AdamW(lr=5e-5, params=pipe.parameters(models=models))
    optimizer.add_param_group(dict(params=pipe.parameters(models=ft_models), lr=1e-6))
else:
    optimizer = torch.optim.AdamW(lr=1e-4, params=pipe.parameters(models=models))
lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=(len(dataloader) * num_epochs),
        )

for p in pipe.parameters():
    p.requires_grad = False

for p in pipe.parameters(models=models + ft_models):
    p.requires_grad = True

accelerator = Accelerator(
        mixed_precision='fp16',
    )

def unwrap(pipe):
    pipe.transformer = accelerator.unwrap_model(pipe.transformer)
    pipe.text_encoder = accelerator.unwrap_model(pipe.text_encoder)
    pipe.text_encoder_2 = accelerator.unwrap_model(pipe.text_encoder_2)
    pipe.clip_image_encoder = accelerator.unwrap_model(pipe.clip_image_encoder)
    pipe.text_decoder = accelerator.unwrap_model(pipe.text_decoder)
    pipe.vae = accelerator.unwrap_model(pipe.vae)
    pipe.image_encoder_adapter = accelerator.unwrap_model(pipe.image_encoder_adapter)
    pipe.image_decoder_adapter = accelerator.unwrap_model(pipe.image_decoder_adapter)
    return pipe

def truncate(texts):
    # texts = pipe.tokenizer.batch_decode(pipe.tokenizer(
    #     texts,
    #     padding="max_length",
    #     max_length=77 - 3,
    #     truncation=True,
    #     return_tensors="pt",
    # ).input_ids)
    # texts = [x.replace('<|endoftext|>', '').replace('<|startoftext|>', '') for x in texts]

    return texts

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
)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

print(f"TOTAL TRANSFORMER LAYERS: {len(pipe.transformer.transformer_blocks)} | OUR CHOSEN BLOCK: {args.block_num}")

def get_suffix_ids(batch):
    if args.dataset_conditioning:
        idxs = batch[-1]['data_idx']
        assert (idxs < len(train_config['roots'])).all()
        suffix_text = [f'<|dataset{i}|>' for i in idxs]
        suffix_ids = pipe.decoder_tokenizer(suffix_text, return_tensors='pt').input_ids.to(accelerator.device)
        return suffix_ids
    return None

def sample(batch):
    with torch.no_grad():
        if len(batch[0]) > 1:
            print(f"Sample batch size is large ({len(batch[0])})! Is this really what we want?")
        image = batch[0].to(accelerator.device)
        index = torch.zeros(size=(len(image),), dtype=torch.long) + args.index
        suffix_input_ids = get_suffix_ids(batch)
        print('suffix_input_ids', suffix_input_ids)
        if args.clip:
            embeds, pooled_embeds = pipe.encode_image(image, dtype=torch.float16)
        else:
            embeds, pooled_embeds = pipe.dift_features(
                image, index, return_layers=args.block_num, dataset_conditioning=args.dataset_conditioning)
        embeds = torch.cat([embeds, pooled_embeds], axis=1)
        decoded_tokens = pipe.text_decoder.generate_captions(embeds, 
                            eos_token_id=pipe.decoder_tokenizer.eos_token_id, device=accelerator.device,
                            suffix_input_ids=suffix_input_ids
                            )[0]
        decoded_text = pipe.decoder_tokenizer.batch_decode(decoded_tokens)
    return decoded_text

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
    for epoch in range(num_epochs):
        progbar = tqdm(dataloader, disable=not accelerator.is_main_process)
        for step, batch in enumerate(progbar):
            optimizer.zero_grad()
            
            # prepare data
            image, prompt = batch[0].to(accelerator.device), truncate(batch[1])
            batch[1] = [x.replace('<|endoftext|>', '') for x in batch[1]]
            index = torch.zeros(size=(len(image),), dtype=torch.long) + args.index

            # run model
            suffix_input_ids = get_suffix_ids(batch)
            if args.clip:
                embeds, pooled_embeds = pipe.encode_image(image, dtype=torch.float16)
            else:
                embeds, pooled_embeds = pipe.dift_features(image, index, return_layers=args.block_num, 
                dataset_conditioning=args.dataset_conditioning)

            loss = pipe.embed_to_decoder(embeds, pooled_embeds, prompt, suffix_input_ids=suffix_input_ids)
            accelerator.backward(loss)

            for p in pipe.parameters():
                if p.grad is not None:
                    torch.nan_to_num(p.grad, nan=0, posinf=1e5, neginf=-1e5, out=p.grad)

            optimizer.step()
            lr_scheduler.step()
            
            progbar.set_description(f"loss: {loss.item():.3f}, lr: {get_lr(optimizer)}")

            if accelerator.is_main_process and ((step + 1) % 500 == 0 or step == 10):
                if (step + 1) % 10000 == 0:
                    pipe = unwrap(pipe)
                    pipe.save_pretrained(f'{args.work_dir}/epoch_{epoch}_step_{step}/')
                    print(f"Saved model to directory {f'{args.work_dir}/epoch_{epoch}_step_{step}/'}")
                else:
                    pipe.save_pretrained(f'{args.work_dir}/current/')

                batch = next(iter(val_loader))
                decoded_text = sample(batch)
                print(
                    f"Recon: {decoded_text[0].strip('!').replace('<|endoftext|>', '').replace('<|EOS|>', '')} | "
                    f"True: {batch[1][0]}"
                )
    