# %%
import os

# %%
import argparse
import torch
import json
from unilatent import UniLatentPipeline

from data.builder import build_dataset, build_dataloader
from aspect_ratio_sampler import AspectRatioBatchSampler
from torch.utils.data import RandomSampler

from tqdm import tqdm
from accelerate import Accelerator

# parser = argparse.ArgumentParser(description="Training.")
# parser.add_argument('--work_dir', default='/mnt/bn/us-aigc-temp/henry/data/clip2text/', help='the dir to save logs and models')
# parser.add_argument('--batch_size', type=int, default=48)
# args = parser.parse_args()

# %%
data_config = {
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
dataset = build_dataset(
    data_config, resolution=512, aspect_ratio_type='ASPECT_RATIO_512',
    real_prompt_ratio=0.0, max_length=77, return_image_id=True
)
batch_sampler = AspectRatioBatchSampler(sampler=RandomSampler(dataset), dataset=dataset,
                                    batch_size=1, aspect_ratios=dataset.aspect_ratio, drop_last=True,
                                    ratio_nums=dataset.ratio_nums, valid_num=0)
dataloader = build_dataloader(dataset, batch_sampler=batch_sampler, num_workers=10)

accelerator = Accelerator(
        mixed_precision='fp16',
    )

# %%
def prepare(accelerator, pipe):
    (
        pipe.transformer,
        pipe.text_encoder, 
        pipe.text_encoder_2,
        pipe.clip_image_encoder,
        pipe.text_decoder,
        pipe.vae
    ) = accelerator.prepare(
        pipe.transformer,
        pipe.text_encoder, 
        pipe.text_encoder_2,
        pipe.clip_image_encoder,
        pipe.text_decoder,
        pipe.vae
    )

    return pipe

def dift_sampler(batch, pipe, index, block_num):
    index_ = torch.zeros(size=(1,), dtype=torch.long) + index
    embeds, pooled_embeds = pipe.dift_features(batch[0][:1], index_, return_layer=block_num)
    embeds = torch.cat([embeds[:1], pooled_embeds[:1]], axis=1)
    decoded_tokens = pipe.text_decoder.generate_captions(embeds, 
                        eos_token_id=pipe.decoder_tokenizer.eos_token_id, device=accelerator.device)[0]
    decoded_text = pipe.decoder_tokenizer.batch_decode(decoded_tokens)[0]
    return decoded_text

def clip_sampler(batch, pipe):
    embeds, pooled_embeds = pipe.encode_image(batch[0][:1])
    embeds = torch.cat([embeds[:1], pooled_embeds[:1]], axis=1)
    decoded_tokens = pipe.text_decoder.generate_captions(embeds, 
                        eos_token_id=pipe.decoder_tokenizer.eos_token_id, device=accelerator.device)[0]
    decoded_text = pipe.decoder_tokenizer.batch_decode(decoded_tokens)[0]
    return decoded_text

def generate_captions(pipe, dataloader, save_path, sampler, sampler_kwargs={}):
    json_list = []
    progbar = tqdm(iter(dataloader))
    for i, batch in enumerate(progbar):
        with torch.no_grad():
            decoded_text = sampler(batch, pipe, **sampler_kwargs)
        
        caption = decoded_text.strip('!').replace('<|endoftext|>', '').replace('<|EOS|>', '').strip()
        json_list.append({'image_id': batch[-1]['image_id'].item(), 'caption': caption})

        progbar.set_description(f"Image: {i:05d} | Predicted: {caption} | True: {batch[1][0]}")

        if (i + 1) % 50 == 0:
            with open(save_path, 'w') as f:
                test = json.dump(json_list, f)

    return json_list

# %%
# for block_num in [6, 12]:
for block_num in [12]:
    for index in [0, 250, 500, 750]:
        name = f'index_{index:03d}_block_{block_num}'
        save_path = f'/mnt/bn/us-aigc-temp/henry/data/captions/dift/dift_{name}_step_34999.json'
        load_path = f'/mnt/bn/us-aigc-temp/henry/data/dift/{name}/epoch_0_step_34999/'
        print(f"Loading pipeline for {name}:")
        pipe = UniLatentPipeline.from_pretrained(load_path, torch_dtype=torch.float32)

        pipe = prepare(accelerator, pipe)
        print(f"Running sampler for {name}:")
        sampler_kwargs = {'index': index, 'block_num': block_num}
        generate_captions(pipe, dataloader, save_path, dift_sampler, sampler_kwargs)

# %%



