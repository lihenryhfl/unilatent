import json
import os
import argparse
import torch
import numpy as np
from diffusers import StableDiffusion3Pipeline
from unilatent import UniLatentPipeline

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

from caption_decoder_v1 import TextDecoder
from transformer import SD3Transformer2DModel
from utils import EmbedAdapter

parser = argparse.ArgumentParser(description="Training.")
parser.add_argument('--work_dir', default='/mnt/bn/us-aigc-temp/henry/data/clip2text/', help='the dir to save logs and models')
parser.add_argument('--load_from', default='', help='the dir to load from')
parser.add_argument('--batch_size', type=int, default=48)
parser.add_argument('--step_offset', type=int, default=0)
parser.add_argument('--sample_and_exit', action='store_true')
args = parser.parse_args()

if not args.load_from:
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float32)

    decoder_tokenizer = GPT2Tokenizer.from_pretrained('/mnt/bn/us-aigc-temp/henry/unilatent_weights/gpt_tokenizer/')
    decoder_tokenizer.add_special_tokens({'pad_token': decoder_tokenizer.eos_token})
    pipe.decoder_tokenizer = decoder_tokenizer

    text_decoder = TextDecoder(prefix_length=78, prefix_inner_dim=2048, prefix_hidden_dim=2048, vocab_size=decoder_tokenizer.vocab_size + 1)
    pipe.text_decoder = text_decoder

    pipe.clip_image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float32)
    pipe.clip_image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float32)

    transformer = SD3Transformer2DModel.from_config(pipe.transformer.config)
    transformer.load_state_dict(pipe.transformer.state_dict())
    pipe.transformer = transformer

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
        image_encoder_adapter=EmbedAdapter(1024, 2048, 77)
    )
else:
    pipe = UniLatentPipeline.from_pretrained(args.load_from, torch_dtype=torch.float32)

pipe.register_modules(image_decoder_adapter=EmbedAdapter(2048, 2048, 77))

val_data_config = {
    'type': 'FlexibleInternalDataMS',
    'roots': [
        '/mnt/bn/us-aigc-temp/henry/coco_2014/val/val2014/',
    ],
    'json_lst': [
        '/mnt/bn/us-aigc-temp/henry/test.json',
    ],
    'load_vae_feat': False,
    'load_t5_feat': False
}
val_dataset = build_dataset(
    val_data_config, resolution=512, aspect_ratio_type='ASPECT_RATIO_512',
    real_prompt_ratio=0.0, max_length=77, return_image_id=True
)
val_batch_sampler = AspectRatioBatchSampler(sampler=RandomSampler(val_dataset), dataset=val_dataset,
                                    batch_size=1, aspect_ratios=val_dataset.aspect_ratio, drop_last=True,
                                    ratio_nums=val_dataset.ratio_nums, valid_num=0)
val_loader = build_dataloader(val_dataset, batch_sampler=val_batch_sampler, num_workers=10)

if not args.sample_and_exit:
    data_config = {
        'type': 'FlexibleInternalDataMS',
        'roots': [
            # '/mnt/bn/us-aigc-temp/henry/coco_2014/val/val2014/',
            '/mnt/bn/aigc-us/zjl/laion-coco-aesthetic/data_max1024/',
            # '/mnt/bn/aigc-us/zjl/recap_datacom_1b_aesthetic_subset/data/',
            # '/mnt/bn/aigc-us/zjl/openimages/data/',
            # '/mnt/bn/aigc-us/zjl/sharegpt4v_processed_data/data/'
        ],
        'json_lst': [
            # '/mnt/bn/us-aigc-temp/henry/test.json',
            '/mnt/bn/aigc-us/zjl/laion-coco-aesthetic/data_max1024/meta_data_coco_edited.json',
            # '/mnt/bn/aigc-us/zjl/recap_datacom_1b_aesthetic_subset/data/aes5_meta_data_all.json',
            # '/mnt/bn/aigc-us/zjl/sharegpt4v_processed_data/data/meta_data.json',
            # '/mnt/bn/aigc-us/zjl/sharegpt4v_processed_data/data/meta_data.json'
        ],
        'load_vae_feat': False,
        'load_t5_feat': False
    }
    dataset = build_dataset(
        data_config, resolution=512, aspect_ratio_type='ASPECT_RATIO_512',
        real_prompt_ratio=0.0, max_length=77,
    )
    batch_sampler = AspectRatioBatchSampler(sampler=RandomSampler(dataset), dataset=dataset,
                                        batch_size=args.batch_size, aspect_ratios=dataset.aspect_ratio, drop_last=True,
                                        ratio_nums=dataset.ratio_nums, valid_num=0)
    dataloader = build_dataloader(dataset, batch_sampler=batch_sampler, num_workers=10)
else:
    dataloader = val_loader

num_epochs = 2

# models = [pipe.text_decoder]
# models = [pipe.transformer, pipe.text_decoder, pipe.clip_image_encoder, pipe.text_encoder, pipe.text_encoder_2]
# models = [pipe.text_decoder, pipe.clip_image_encoder, pipe.text_encoder, pipe.text_encoder_2, pipe.image_encoder_adapter]
models = [pipe.text_decoder, pipe.image_encoder_adapter, pipe.image_decoder_adapter]
models2 = [pipe.text_encoder, pipe.text_encoder_2, pipe.clip_image_encoder]

optimizer = torch.optim.AdamW(lr=5e-5, params=pipe.parameters(models=models))
optimizer.add_param_group(dict(params=pipe.parameters(models=models2), lr=1e-6))
lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=1000,
            num_training_steps=(len(dataloader) * num_epochs),
        )

for p in pipe.parameters():
    p.requires_grad = False

for p in pipe.parameters(models=models + models2):
    p.requires_grad = True

accelerator = Accelerator(
        mixed_precision='fp16',
        # gradient_accumulation_steps=config.gradient_accumulation_steps
    )

def truncate(texts):
    texts = pipe.tokenizer.batch_decode(pipe.tokenizer(
        texts,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    ).input_ids)
    texts = [x.replace('<|endoftext|>', '').replace('<|startoftext|>', '').strip().strip('!').strip() for x in texts]

    texts = pipe.tokenizer_2.batch_decode(pipe.tokenizer_2(
        texts,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    ).input_ids)
    texts = [x.replace('<|endoftext|>', '').replace('<|startoftext|>', '').strip().strip('!').strip() for x in texts]

    return texts

optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)

pipe = pipe.to(accelerator.device)
def prepare(pipe):
    (
        pipe.transformer,
        pipe.text_encoder, 
        pipe.text_encoder_2,
        pipe.clip_image_encoder,
        pipe.text_decoder,
        pipe.vae,
        pipe.image_encoder_adapter,
        pipe.image_decoder_adapter
    ) = accelerator.prepare(
        pipe.transformer,
        pipe.text_encoder, 
        pipe.text_encoder_2,
        pipe.clip_image_encoder,
        pipe.text_decoder,
        pipe.vae,
        pipe.image_encoder_adapter,
        pipe.image_decoder_adapter
    )
    return pipe

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

def save(pipe, path):
    # accelerator.unwrap_model(pipe.transformer).save_pretrained(os.path.join(path, 'transformer'))
    accelerator.unwrap_model(pipe.text_encoder).save_pretrained(os.path.join(path, 'text_encoder'))
    accelerator.unwrap_model(pipe.text_encoder_2).save_pretrained(os.path.join(path, 'text_encoder_2'))
    accelerator.unwrap_model(pipe.clip_image_encoder).save_pretrained(os.path.join(path, 'clip_image_encoder'))
    accelerator.unwrap_model(pipe.text_decoder).save_pretrained(os.path.join(path, 'text_decoder'))
    accelerator.unwrap_model(pipe.vae).save_pretrained(os.path.join(path, 'vae'))
    accelerator.unwrap_model(pipe.image_encoder_adapter).save_pretrained(os.path.join(path, 'image_encoder_adapter'))

pipe = prepare(pipe)

def sample(batch):
    image, prompt = batch[0].to('cuda'), truncate(batch[1])
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = pipe.encode_text(prompt[:1])
        image_embeds, pooled_image_embeds = pipe.encode_image(image[:1], dtype=torch.float16)
        embeds = torch.cat([prompt_embeds, image_embeds])
        pooled_embeds = torch.cat([pooled_prompt_embeds, pooled_image_embeds])
        embeds = torch.cat([embeds, pooled_embeds], axis=1)
        decoded_tokens = pipe.text_decoder.generate_captions(embeds, 
                            eos_token_id=pipe.decoder_tokenizer.eos_token_id, device=accelerator.device)[0]
        decoded_text = pipe.decoder_tokenizer.batch_decode(decoded_tokens)
    return decoded_text

d_losses = []
c_losses = []

iter_val_loader = iter(val_loader)
if args.sample_and_exit:
    save_path = os.path.join(args.work_dir, 'captions.json')
    print("Saving to", save_path)
    json_list = []
    progbar = tqdm(val_loader)
    for i, batch in enumerate(progbar):
        decoded_text = sample(batch)[1]
        
        caption = decoded_text.strip('!').replace('<|endoftext|>', '').replace('<|EOS|>', '').strip()
        image_id = batch[-1]['image_id'].item() if 'image_id' in batch[-1] else 0
        json_list.append({'image_id': image_id, 'caption': caption})
        progbar.set_description(f"Image: {i:05d} | Predicted: {caption} | True: {batch[1][0]}")

        if (i + 1) % 100 == 0:
            with open(save_path, 'w') as f:
                json.dump(json_list, f)
else:
    for epoch in range(num_epochs):
        # progbar = tqdm(dataloader, mininterval=30, disable=not accelerator.is_main_process)
        progbar = tqdm(dataloader, disable=not accelerator.is_main_process)
        for step, batch in enumerate(progbar):
            step = args.step_offset + step
            optimizer.zero_grad()
            
            # prepare data
            image, prompt = batch[0].to('cuda'), truncate(batch[1])
            index = torch.randint(0, 1000, size=(len(image),))

            # run model
            image_embeds, pooled_image_embeds = pipe.encode_image(image, dtype=torch.float16)

            embeds = image_embeds
            pooled_embeds = pooled_image_embeds

            model_output, target = pipe.embed_to_denoiser(image, embeds, pooled_embeds, index)
            d_loss = torch.nan_to_num(((model_output - target) ** 2).mean())
            c_loss = torch.tensor(0., dtype=torch.float16, device=accelerator.device)

            # c_loss = pipe.embed_to_decoder(embeds, pooled_embeds, prompt)
            if not c_loss.isfinite().all():
                print(f"c_loss has nan.")
            c_loss = torch.nan_to_num(c_loss)

            loss = d_loss + c_loss
            accelerator.backward(loss)

            d_losses.append(d_loss.item())
            c_losses.append(c_loss.item())

            num_params, num_nans = 0, 0
            for p in pipe.parameters():
                if p.grad is not None:
                    num_params += np.prod(p.shape)
                    num_nans += ((1 - p.grad.isfinite().int()).float()).sum()
                    torch.nan_to_num(p.grad, nan=0, posinf=1e5, neginf=-1e5, out=p.grad)

            optimizer.step()
            lr_scheduler.step()
            
            progbar.set_description(f"LOSSES: diff {np.mean(d_losses).item():02.3f} | ce {np.mean(c_losses).item():02.3f}")

            if accelerator.is_main_process and ((step + 1) % 500 == 0 or step == 10):
                if (step + 1) % 500 == 0 or step == 10:
                    pipe = unwrap(pipe)
                    pipe.save_pretrained(f'{args.work_dir}/current/')
                    print(f"Saved model to directory {f'{args.work_dir}/current/'}")
                elif (step + 1) % 5000 == 0 or step == 10:
                    pipe = unwrap(pipe)
                    pipe.save_pretrained(f'{args.work_dir}/epoch_{epoch}_step_{step}/')
                    print(f"Saved model to directory {f'{args.work_dir}/epoch_{epoch}_step_{step}/'}")

                d_losses = []
                c_losses = []

                batch = next(iter_val_loader)
                decoded_text = sample(batch)
                print(
                    f"Recon from text: {decoded_text[0].strip('!').replace('<|endoftext|>', '').replace('<|EOS|>', '')} \n"
                    f"Recon from image: {decoded_text[1].strip('!').replace('<|endoftext|>', '').replace('<|EOS|>', '')} \n"
                    f"True: {batch[1][0]}"
                )
