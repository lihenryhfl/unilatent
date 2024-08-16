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
from utils import get_lr, get_suffix_ids, get_dataloader
from utils import EmbedAdapterV1
from utils import EmbedAdapterV2
from utils import EmbedAdapterV3

parser = argparse.ArgumentParser(description="Training.")
parser.add_argument('--work_dir', default='/mnt/bn/us-aigc-temp/henry/data/clip2text/', help='the dir to save logs and models')
parser.add_argument('--load_from', default='', help='the dir to load from')
parser.add_argument('--adapter_version', default='v1', help='version of the pre-decoder / pre-denoiser adapter')
parser.add_argument('--batch_size', type=int, default=48)
parser.add_argument('--step_offset', type=int, default=0)
parser.add_argument('--num_steps', type=int, default=100_000)
parser.add_argument('--lam', type=float, default=1.)
parser.add_argument('--sample_and_exit', action='store_true')
parser.add_argument('--text', action='store_true')
parser.add_argument('--text_and_image', action='store_true')
parser.add_argument('--pretrain_adapter', action='store_true')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

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

if not args.load_from:
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float32)

    decoder_tokenizer = GPT2Tokenizer.from_pretrained('/mnt/bn/us-aigc-temp/henry/unilatent_weights/gpt_tokenizer/')
    decoder_tokenizer.add_special_tokens({'pad_token': decoder_tokenizer.eos_token})
    decoder_tokenizer.add_tokens([f'<|dataset{i}|>' for i in range(len(train_config['roots']))])
    pipe.decoder_tokenizer = decoder_tokenizer
    prefix_length = 78

    text_decoder = TextDecoder(prefix_length=prefix_length + 1, prefix_inner_dim=2048, prefix_hidden_dim=2048, 
                                vocab_size=len(decoder_tokenizer) + len(decoder_tokenizer.get_added_vocab()))
    pipe.text_decoder = text_decoder

    pipe.clip_image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float32)
    pipe.clip_image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float32)

    # transformer = SD3Transformer2DModel.from_config(pipe.transformer.config)
    # transformer.load_state_dict(pipe.transformer.state_dict())
    # pipe.transformer = transformer

    if args.adapter_version == 'v1':
        EmbedAdapter = EmbedAdapterV1
        adapter_len = -1
    elif args.adapter_version == 'v2':
        EmbedAdapter = EmbedAdapterV2
        adapter_len = -1
    elif args.adapter_version == 'v3':
        EmbedAdapter = EmbedAdapterV3
        adapter_len = 78
        print("ADAPTER_LEN", adapter_len)
    else:
        raise NotImplementedError

    image_encoder_adapter = image_decoder_adapter = None
    image_encoder_adapter = EmbedAdapter(1024, 2048, prefix_length, embed_pool=True, use_attn=True)
    # image_decoder_adapter = EmbedAdapter(2048, 2048, adapter_len, embed_pool=True, use_attn=True)

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
        image_decoder_adapter=image_decoder_adapter
    )
else:
    pipe = UniLatentPipeline.from_pretrained(args.load_from, torch_dtype=torch.float32)

val_loader = get_dataloader(args, val_config, val=True)

if args.debug or not args.sample_and_exit:
    dataloader = get_dataloader(args, train_config)
else:
    dataloader = val_loader

num_steps = args.num_steps

# models = [pipe.text_decoder]
# models = [pipe.text_encoder, pipe.text_encoder_2]
if args.text:
    models = [pipe.image_decoder_adapter]
    models2 = []
elif args.text_and_image:
    models = [pipe.image_encoder_adapter, pipe.text_decoder, pipe.image_decoder_adapter]
    models2 = []
else:
    models = [pipe.image_encoder_adapter]
    models2 = []

[x.train() for x in models if x is not None]
[x.train() for x in models2 if x is not None]

optimizer = torch.optim.AdamW(lr=2e-5, params=pipe.parameters(models=models))
# optimizer = torch.optim.AdamW(lr=1e-8, params=pipe.parameters(models=models))
optimizer.add_param_group(dict(params=pipe.parameters(models=models2), lr=1e-6))
lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=10000 + args.step_offset,
            num_training_steps=num_steps,
        )

for p in pipe.parameters():
    p.requires_grad = False

for p in pipe.parameters(models=models + models2):
    p.requires_grad = True

accelerator = Accelerator(
        # mixed_precision='fp16',
        mixed_precision='bf16',
        # gradient_accumulation_steps=config.gradient_accumulation_steps
    )

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
    accelerator.unwrap_model(pipe.transformer).save_pretrained(os.path.join(path, 'transformer'))
    accelerator.unwrap_model(pipe.text_encoder).save_pretrained(os.path.join(path, 'text_encoder'))
    accelerator.unwrap_model(pipe.text_encoder_2).save_pretrained(os.path.join(path, 'text_encoder_2'))
    accelerator.unwrap_model(pipe.clip_image_encoder).save_pretrained(os.path.join(path, 'clip_image_encoder'))
    accelerator.unwrap_model(pipe.text_decoder).save_pretrained(os.path.join(path, 'text_decoder'))
    accelerator.unwrap_model(pipe.vae).save_pretrained(os.path.join(path, 'vae'))
    accelerator.unwrap_model(pipe.image_encoder_adapter).save_pretrained(os.path.join(path, 'image_encoder_adapter'))

pipe = prepare(pipe)

def sample(batch):
    image, prompt = batch[0].to('cuda'), batch[1]
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = pipe.encode_text(prompt[:1])
        image_embeds, pooled_image_embeds = pipe.encode_image(image[:1], dtype=torch.float16)
        embeds = torch.cat([prompt_embeds, image_embeds])
        pooled_embeds = torch.cat([pooled_prompt_embeds, pooled_image_embeds])
        embeds = torch.cat([embeds, pooled_embeds], axis=1)
        suffix_input_ids = get_suffix_ids(batch, pipe.decoder_tokenizer, accelerator.device)
        decoded_tokens = pipe.text_decoder.generate_captions(embeds, 
                            eos_token_id=pipe.decoder_tokenizer.eos_token_id, device=accelerator.device,
                            suffix_input_ids=suffix_input_ids)[0]
        decoded_text = pipe.decoder_tokenizer.batch_decode(decoded_tokens)
    return decoded_text

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
    if args.pretrain_adapter:
        gradient_fixer = GradientFixer(pipe.text_decoder, pipe.decoder_tokenizer)
    
    d_losses = []
    c_losses = []
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
            image, prompt = batch[0].to('cuda'), batch[1]
            suffix_input_ids = get_suffix_ids(batch, pipe.decoder_tokenizer, accelerator.device)

            # run model
            if args.text:
                image_embeds, pooled_image_embeds = pipe.encode_text(prompt)
            else:
                image_embeds, pooled_image_embeds = pipe.encode_image(image, dtype=torch.float16)
            

            embeds = image_embeds
            pooled_embeds = pooled_image_embeds

            # model_output, target = pipe.embed_to_denoiser(image, embeds, pooled_embeds, index)
            model_output, target = pipe.embed_to_denoiser_v2(image, embeds, pooled_embeds)
            d_loss = ((model_output - target) ** 2).mean(axis=(1, 2, 3))
            if not d_loss.isfinite().all():
                d_c = len(d_loss) if len(d_loss.shape) > 0 else 1
                print(f"d_loss has nans: {d_loss.isfinite().sum() / len(d_loss) * 100}%")
            else:
                d_losses.append(d_loss.mean().item())
            
            d_loss = torch.nan_to_num(d_loss)

            if args.text_and_image:
                c_loss = pipe.embed_to_decoder(embeds, pooled_embeds, prompt, suffix_input_ids=suffix_input_ids)
            else:
                c_loss = torch.tensor(0., dtype=torch.float16, device=accelerator.device)

            if not c_loss.isfinite().all():
                n_c = len(c_loss) if len(c_loss.shape) > 0 else 1
                print(f"c_loss has nans: {c_loss.isfinite().sum() / n_c * 100}%")
            else:
                c_losses.append(c_loss.mean().item())
            
            c_loss = torch.nan_to_num(c_loss)

            loss = d_loss.mean() + (args.lam * c_loss.mean())
            accelerator.backward(loss)

            if args.pretrain_adapter:
                gradient_fixer.fix_gradients()

            num_params, num_nans = 0, 0
            for p in pipe.parameters():
                if p.grad is not None:
                    num_params += np.prod(p.shape)
                    num_nans += ((1 - p.grad.isfinite().int()).float()).sum()
                    torch.nan_to_num(p.grad, nan=0, posinf=1e5, neginf=-1e5, out=p.grad)

            grad_norm = accelerator.clip_grad_norm_(pipe.parameters(), 1.0)

            optimizer.step()
            lr_scheduler.step()
            
            progbar.set_description(
                f"LOSSES: diff {np.mean(d_losses).item():02.3f} "
                f"| ce {np.mean(c_losses).item():02.3f} "
                f"| lr: {get_lr(optimizer):.3e}")

            if accelerator.is_main_process and (step + 1) % 500 == 0:
                [x.eval() for x in models if x is not None]
                [x.eval() for x in models2 if x is not None]
                if (step + 1) % 5000 == 0:
                    pipe = unwrap(pipe)
                    pipe.save_pretrained(f'{args.work_dir}/step_{step}/')
                    print(f"Saved model to directory {f'{args.work_dir}/step_{step}/'}")
                elif (step + 1) % 500 == 0:
                    pipe = unwrap(pipe)
                    pipe.save_pretrained(f'{args.work_dir}/current/')

                d_losses = []
                c_losses = []

                if args.text_and_image:
                    batch = next(iter_val_loader)
                    decoded_text = sample(batch)
                    print(
                        f"Recon from text: {decoded_text[0].strip('!').replace('<|endoftext|>', '').replace('<|EOS|>', '')} \n"
                        f"Recon from image: {decoded_text[1].strip('!').replace('<|endoftext|>', '').replace('<|EOS|>', '')} \n"
                        f"True: {batch[1][0]}"
                    )

                [x.train() for x in models if x is not None]
                [x.train() for x in models2 if x is not None]
            
            step += 1
