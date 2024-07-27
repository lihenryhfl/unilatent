import argparse
import torch
from diffusers import StableDiffusion3Pipeline
from stable_diffusion3 import UniLatentPipeline, retrieve_timesteps

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

from unilatent.caption_decoder_v2 import TextDecoder

parser = argparse.ArgumentParser(description="Training.")
parser.add_argument('--work_dir', default='/mnt/bn/us-aigc-temp/henry/data/clip2text/', help='the dir to save logs and models')
parser.add_argument('--batch_size', type=int, default=48)
args = parser.parse_args()

# pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float32)

# decoder_tokenizer = GPT2Tokenizer.from_pretrained('/mnt/bn/us-aigc-temp/henry/unilatent_weights/gpt_tokenizer/')
# decoder_tokenizer.add_special_tokens({'pad_token': decoder_tokenizer.eos_token})
# pipe.decoder_tokenizer = decoder_tokenizer

# text_decoder = TextDecoder.from_pretrained('/mnt/bn/us-aigc-temp/henry/unilatent_weights/gpt/', 
#                     device_map=None, low_cpu_mem_usage=False, torch_dtype=torch.float32, ignore_mismatched_sizes=True)
# # slightly hacky -- cannot save wte weights since they are shared with lm_head, so we copy them back here
# text_decoder.transformer.transformer.wte.weight = text_decoder.transformer.lm_head.weight
# # text_decoder.decode_prefix = torch.nn.Linear(1024, 768)
# text_decoder = TextDecoder(prefix_length=257, prefix_inner_dim=1024, vocab_size=decoder_tokenizer.vocab_size)
# pipe.text_decoder = text_decoder

# pipe.clip_image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float32)
# pipe.clip_image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float32)

# pipe = UniLatentPipeline(
#     transformer=pipe.transformer,
#     scheduler=pipe.scheduler,
#     vae=pipe.vae,
#     text_encoder=pipe.text_encoder,
#     tokenizer=pipe.tokenizer,
#     text_encoder_2=pipe.text_encoder_2,
#     tokenizer_2=pipe.tokenizer_2,
#     clip_image_encoder=pipe.clip_image_encoder,
#     clip_image_processor=pipe.clip_image_processor,
#     text_decoder=pipe.text_decoder,
#     decoder_tokenizer=pipe.decoder_tokenizer,
# )

pipe = UniLatentPipeline.from_pretrained('/mnt/bn/us-aigc-temp/henry/data/clip_test/', 
                    device_map=None, low_cpu_mem_usage=False, torch_dtype=torch.float32)

data_config = {
    'type': 'FlexibleInternalDataMS',
    'roots': [
        '/mnt/bn/us-aigc-temp/henry/coco_2014/val/val2014/',
        # '/mnt/bn/aigc-us/zjl/laion-coco-aesthetic/data_max1024/',
        # '/mnt/bn/aigc-us/zjl/openimages/data/',
        # '/mnt/bn/aigc-us/zjl/sharegpt4v_processed_data/data/'
    ],
    'json_lst': [
        '/mnt/bn/us-aigc-temp/henry/test.json',
        # '/mnt/bn/aigc-us/zjl/laion-coco-aesthetic/data_max1024/meta_data_coco.json'
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

num_epochs = 100

optimizer = torch.optim.AdamW(lr=1e-4, params=pipe.parameters(models=[pipe.text_decoder]))
lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=50,
            num_training_steps=(len(dataloader) * num_epochs),
        )

for p in pipe.parameters():
    p.requires_grad = False

for p in pipe.parameters(models=[pipe.text_decoder]):
    p.requires_grad = True

accelerator = Accelerator(
        mixed_precision='fp16',
        # gradient_accumulation_steps=config.gradient_accumulation_steps
    )

(
    optimizer, 
    lr_scheduler,
    pipe.transformer,
    pipe.text_encoder, 
    pipe.text_encoder_2,
    pipe.clip_image_encoder,
    pipe.text_decoder,
    pipe.vae
) = accelerator.prepare(
    optimizer, 
    lr_scheduler,
    pipe.transformer,
    pipe.text_encoder, 
    pipe.text_encoder_2,
    pipe.clip_image_encoder,
    pipe.text_decoder,
    pipe.vae
)

for epoch in range(num_epochs):
    progbar = tqdm(dataloader)
    for step, batch in enumerate(progbar):
        with accelerator.accumulate(pipe.text_decoder):
            optimizer.zero_grad()
            
            # prepare data
            image, prompt = batch[0].to('cuda'), batch[1]
            batch[1] = [x.strip('<|endoftext|>') for x in batch[1]]
            index = torch.randint(0, 1000, size=(len(image),))

            # run model
            loss = pipe.decode_loss(image, prompt, device=accelerator.device, dtype=torch.float16)
            accelerator.backward(loss)

            grad_norm = accelerator.clip_grad_norm_(pipe.parameters(), 0.01)

            for p in pipe.parameters():
                if p.grad is not None:
                    torch.nan_to_num(p.grad, nan=0, posinf=1e5, neginf=-1e5, out=p.grad)

            optimizer.step()
            lr_scheduler.step()
        
        progbar.set_description(f"loss: {loss.item():.3f}")

        if accelerator.is_main_process and ((step + 1) % 2500 == 0 or step == 5):
            text_decoder = accelerator.unwrap_model(pipe.text_decoder)
            text_decoder.transformer.lm_head.weight = None
            text_decoder.save_pretrained(f'{args.work_dir}/epoch_{epoch}_step_{step}/')
            text_decoder.transformer.lm_head.weight = text_decoder.transformer.transformer.wte.weight

            image_embd = pipe.encode_image(image[:1], device=accelerator.device, dtype=torch.float16)
            generate_captions = pipe.text_decoder.module.generate_captions if isinstance(pipe.text_decoder, DDP) else pipe.text_decoder.generate_captions
            decoded_tokens = generate_captions(image_embd, 
                                eos_token_id=pipe.decoder_tokenizer.eos_token_id, device=accelerator.device)[0]
            decoded_text = pipe.decoder_tokenizer.batch_decode(decoded_tokens)
            print(decoded_text)
