from multiprocessing.connection import Pipe
import os
import json
import argparse
import torch
from diffusers import StableDiffusion3Pipeline
from unilatent import UniLatentPipeline

from tqdm import tqdm
from diffusers import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from transformers import (
    GPT2Tokenizer,
    CLIPVisionModel,
    CLIPImageProcessor,
)

from caption_decoder import TextDecoder as TextDecoderV0
from caption_decoder_v1 import TextDecoder
from utils import AttentionLayerAggregator, LayerAggregator, GradientFixer, SoftPrompter, get_suffix_ids, get_dataloader, unwrap
from utils import EmbedAdapterV1, EmbedAdapterV2, EmbedAdapterV3

from orig_transformer import SD3Transformer2DModel

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
parser.add_argument('--adapter_type', default='')
parser.add_argument('--pretrain', action='store_true')
parser.add_argument('--layer_aggregator', type=str, default='attention')
parser.add_argument('--pretrain_adapter', action='store_true')
parser.add_argument('--soft_prompter', action='store_true')
parser.add_argument('--flux', action='store_true')
parser.add_argument('--textv0', action='store_true')
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

if args.flux:
    from unilatent_flux import UniLatentFluxPipeline as UniLatentPipeline
    from diffusers import FluxPipeline as OrigPipeline
    from transformer_flux import FluxTransformer2DModel as TransformerClass
    orig_path = "black-forest-labs/FLUX.1-schnell"
    model_dim = 3072
else:
    OrigPipeline = StableDiffusion3Pipeline
    TransformerClass = SD3Transformer2DModel
    orig_path = "stabilityai/stable-diffusion-3-medium-diffusers"
    model_dim = 1536

if args.load_from:
    pipe = UniLatentPipeline.from_pretrained(args.load_from, torch_dtype=torch.float32)
else:
    pipe = OrigPipeline.from_pretrained(orig_path, torch_dtype=torch.float32)
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

    if args.clip:
        prefix_length = 259
        prefix_dim = 1024
        embed_pool = True
    elif args.clip_text:
        prefix_length += 1
        prefix_dim = model_dim
        embed_pool = False
    else:
        prefix_length += 1
        prefix_dim = model_dim
        embed_pool = False

    if args.textv0:
        TextDecoder = TextDecoderV0

    text_decoder = TextDecoder(
        prefix_length=prefix_length,
        prefix_inner_dim=prefix_dim,
        vocab_size=len(decoder_tokenizer) + len(decoder_tokenizer.get_added_vocab()))
    pipe.text_decoder = text_decoder

    pipe.clip_image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float32)
    pipe.clip_image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float32)

    transformer = TransformerClass.from_config(pipe.transformer.config)
    transformer.load_state_dict(pipe.transformer.state_dict())
    pipe.transformer = transformer

    soft_prompter = image_encoder_adapter = None
    if args.adapter_type == 'v1':
        dift_image_encoder_adapter = EmbedAdapterV1(prefix_dim, prefix_dim, prefix_length - 1, embed_pool=embed_pool)
    elif args.adapter_type == 'v2':
        dift_image_encoder_adapter = EmbedAdapterV2(prefix_dim, prefix_dim, prefix_length - 1, embed_pool=embed_pool, use_attn=True)
    elif args.adapter_type == 'v3':
        dift_image_encoder_adapter = EmbedAdapterV3(prefix_dim, prefix_dim, prefix_length - 1, embed_pool=embed_pool, use_attn=True)
    elif not args.adapter_type or args.adapter_type == 'none':
        dift_image_encoder_adapter = None

    layer_aggregator = None
    if args.layer_aggregator == 'attention':
        layer_aggregator = AttentionLayerAggregator(len(args.block_num), prefix_dim)
    elif args.layer_aggregator == 'no_attention':
        layer_aggregator = LayerAggregator(len(args.block_num))
    else:
        print(f"Not recognized: {args.layer_aggregator}")
        raise NotImplementedError

    if args.soft_prompter:
        soft_prompter = SoftPrompter(2048, 33)
        
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
        dift_image_encoder_adapter=dift_image_encoder_adapter,
        soft_prompter=soft_prompter,
        layer_aggregator=layer_aggregator
    )

# also re-initialize transformer
transformer = TransformerClass.from_config(pipe.transformer.config)
transformer.load_state_dict(pipe.transformer.state_dict())
pipe.register_modules(
    transformer=transformer
)

if args.layer_aggregator:
    layer_logits = pipe.layer_aggregator.layer_logits
    print(f"Layer logits: {layer_logits}")

val_loader = get_dataloader(args, val_config, val=True)

if args.debug or not args.sample_and_exit:
    dataloader = get_dataloader(args, train_config)
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

if args.soft_prompter:
    models.append(pipe.soft_prompter)

lr = 2e-3 if args.pretrain_adapter else 2e-5

num_steps = args.n_steps
optimizer = torch.optim.AdamW(lr=lr, params=pipe.parameters(models=models))
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
        mixed_precision='bf16',
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
    pipe.dift_image_encoder_adapter
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
    pipe.dift_image_encoder_adapter
)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

print(f"TOTAL TRANSFORMER LAYERS: {len(pipe.transformer.transformer_blocks)} | OUR CHOSEN BLOCK: {args.block_num}")

def sample(batch):
    with torch.no_grad():
        if len(batch[0]) > 1:
            print(f"Sample batch size is large ({len(batch[0])})! Is this really what we want?")
        image, prompt = batch[0].to(accelerator.device), batch[1]
        index = torch.zeros(size=(len(image),), dtype=torch.long) + args.index
        suffix_input_ids = get_suffix_ids(batch, pipe.decoder_tokenizer, accelerator.device)
        if args.clip:
            embeds, pooled_embeds = pipe.encode_image(image, dtype=torch.float16)
        else:
            if args.clip_text:
                embeds, pooled_embeds = pipe.encode_text(prompt)
            else:
                embeds = pooled_embeds = None

            embeds, pooled_embeds = pipe.dift_features(image, index, return_layers=args.block_num, 
                dataset_conditioning=True, num_aggregation_steps=args.n_agg, embed=embeds, pooled_embed=pooled_embeds)
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
    if args.pretrain_adapter:
        gradient_fixer = GradientFixer(pipe.text_decoder, pipe.decoder_tokenizer)

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
            suffix_input_ids = get_suffix_ids(batch, pipe.decoder_tokenizer, accelerator.device)
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

            if args.pretrain_adapter:
                gradient_fixer.fix_gradients()

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
                    pipe = unwrap(accelerator, pipe)
                    pipe.save_pretrained(f'{args.work_dir}/step_{step}/')
                    print(f"Saved model to directory {f'{args.work_dir}/step_{step}/'}")
                elif (step + 1) % 1000 == 0:
                    pipe = unwrap(accelerator, pipe)
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
    