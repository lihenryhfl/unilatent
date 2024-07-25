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
from utils import ReLength

parser = argparse.ArgumentParser(description="Training.")
parser.add_argument('--work_dir', default='/mnt/bn/us-aigc-temp/henry/data/clip2text/', help='the dir to save logs and models')
parser.add_argument('--batch_size', type=int, default=48)
args = parser.parse_args()

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float32)

decoder_tokenizer = GPT2Tokenizer.from_pretrained('/mnt/bn/us-aigc-temp/henry/unilatent_weights/gpt_tokenizer/')
decoder_tokenizer.add_special_tokens({'pad_token': decoder_tokenizer.eos_token})
pipe.decoder_tokenizer = decoder_tokenizer

# text_decoder = TextDecoder.from_pretrained('/mnt/bn/us-aigc-temp/henry/unilatent_weights/gpt/', 
#                     device_map=None, low_cpu_mem_usage=False, torch_dtype=torch.float32, ignore_mismatched_sizes=True)
# # slightly hacky -- cannot save wte weights since they are shared with lm_head, so we copy them back here
# text_decoder.transformer.transformer.wte.weight = text_decoder.transformer.lm_head.weight
# text_decoder.decode_prefix = torch.nn.Linear(1024, 768)
text_decoder = TextDecoder(prefix_length=78, prefix_inner_dim=2048, prefix_hidden_dim=2048, vocab_size=decoder_tokenizer.vocab_size + 1)
pipe.text_decoder = text_decoder

pipe.clip_image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float32)
pipe.clip_image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float32)

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
)

# pipe = UniLatentPipeline.from_pretrained('/mnt/bn/us-aigc-temp/henry/data/clip_test/', 
#                     device_map=None, low_cpu_mem_usage=False, torch_dtype=torch.float32)

data_config = {
    'type': 'FlexibleInternalDataMS',
    'roots': [
        # '/mnt/bn/us-aigc-temp/henry/coco_2014/val/val2014/',
        '/mnt/bn/aigc-us/zjl/laion-coco-aesthetic/data_max1024/',
        '/mnt/bn/aigc-us/zjl/recap_datacom_1b_aesthetic_subset/data/',
        # '/mnt/bn/aigc-us/zjl/openimages/data/',
        # '/mnt/bn/aigc-us/zjl/sharegpt4v_processed_data/data/'
    ],
    'json_lst': [
        # '/mnt/bn/us-aigc-temp/henry/test.json',
        '/mnt/bn/aigc-us/zjl/laion-coco-aesthetic/data_max1024/meta_data_coco_edited.json',
        '/mnt/bn/aigc-us/zjl/recap_datacom_1b_aesthetic_subset/data/aes5_meta_data_all.json'
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

num_epochs = 2

# models = [pipe.text_decoder]
models = [pipe.transformer]

optimizer = torch.optim.AdamW(lr=1e-4, params=pipe.parameters(models=models))
lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=100,
            num_training_steps=(len(dataloader) * num_epochs),
        )

for p in pipe.parameters():
    p.requires_grad = False

for p in pipe.parameters(models=models):
    p.requires_grad = True

for p in pipe.text_decoder.relength.parameters():
    p.requires_grad = False

for p in pipe.text_decoder.pooled_relength.parameters():
    p.requires_grad = False

accelerator = Accelerator(
        mixed_precision='fp16',
        # gradient_accumulation_steps=config.gradient_accumulation_steps
    )

def prepare(accelerator, pipe, inplace=True):
    return_dict = pipe.components if inplace else {}
    for name in pipe.components:
        return_dict[name] = accelerator.prepare(pipe.components[name])

    return return_dict

def unwrap(accelerator, pipe, inplace=True):
    return_dict = pipe.components if inplace else {}
    for name in pipe.components:
        return_dict[name] = accelerator.unwrap_model(pipe.components[name])

    return return_dict

prepare(accelerator, pipe)

for epoch in range(num_epochs):
    # progbar = tqdm(dataloader, mininterval=30, disable=not accelerator.is_main_process)
    progbar = tqdm(dataloader, disable=not accelerator.is_main_process)
    for step, batch in enumerate(progbar):
        optimizer.zero_grad()
        
        # prepare data
        image, prompt = batch[0].to('cuda'), batch[1]
        batch[1] = [x.strip('<|endoftext|>') for x in batch[1]]
        index = torch.randint(0, 1000, size=(len(image),))

        # run model
        prompt_embeds, pooled_prompt_embeds = pipe.encode_text(prompt, device=accelerator.device, dtype=torch.float16)
        # prompt_embeds = pipe.format_clip_prompt_embeds(prompt_embeds)
        # model_output, target = pipe.embed_to_denoiser(image, prompt_embeds, pooled_prompt_embeds, index, device=accelerator.device, dtype=torch.float16)
        model_output, target = pipe.diffusion_step(image, prompt, index)
        
        loss = ((model_output - target) ** 2).mean()
        # loss = pipe.embed_to_decoder(prompt_embeds, pooled_prompt_embeds, prompt, device=accelerator.device, dtype=torch.float16)
        # image_embeds, pooled_image_embeds = pipe.encode_image(image, device=accelerator.device, dtype=torch.float16)
        # loss = pipe.embed_to_decoder(image_embeds, pooled_image_embeds, prompt, device=accelerator.device, dtype=torch.float16)
        accelerator.backward(loss)

        grad_norm = accelerator.clip_grad_norm_(pipe.parameters(), 0.01)

        for p in pipe.parameters():
            if p.grad is not None:
                torch.nan_to_num(p.grad, nan=0, posinf=1e5, neginf=-1e5, out=p.grad)

        optimizer.step()
        lr_scheduler.step()
        
        progbar.set_description(f"loss: {loss.item():.3f}")

        if accelerator.is_main_process and ((step + 1) % 500 == 0 or step == 10):
            if (step + 1) % 500 == 0 or step == 10:
                pipe.save_pretrained(f'{args.work_dir}/epoch_{epoch}_step_{step}/')
                print(f"Saved model to directory {f'{args.work_dir}/epoch_{epoch}_step_{step}/'}")

            embed, pooled_embed = pipe.encode_image(image[:1])
            embed = torch.cat([embed, pooled_embed], axis=1)
            decoded_tokens = pipe.text_decoder.generate_captions(embed, 
                                eos_token_id=pipe.decoder_tokenizer.eos_token_id, device=accelerator.device)[0]
            decoded_text = pipe.decoder_tokenizer.batch_decode(decoded_tokens)
            print(decoded_text)
    
