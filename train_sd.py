import torch
from stable_diffusion3 import StableDiffusion3Pipeline, retrieve_timesteps

from diffusion.data.builder import build_dataset, build_dataloader
from diffusion.utils.data_sampler import AspectRatioBatchSampler
from torch.utils.data import RandomSampler

from tqdm import tqdm
from diffusers import get_cosine_schedule_with_warmup
from accelerate import Accelerator

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float32)

pipe = pipe.to("cuda")

data_config = {
    'type': 'FlexibleInternalDataMS',
    'roots': [
        # '/mnt/bn/us-aigc-temp/henry/coco_2014/val/val2014/',
        '/mnt/bn/aigc-us/zjl/laion-coco-aesthetic/data_max1024/',
        # '/mnt/bn/aigc-us/zjl/openimages/data/',
        # '/mnt/bn/aigc-us/zjl/sharegpt4v_processed_data/data/'
    ],
    'json_lst': [
        # '/mnt/bn/us-aigc-temp/henry/test.json',
    ],
    'load_vae_feat': False,
    'load_t5_feat': False
}
dataset = build_dataset(
    data_config, resolution=512, aspect_ratio_type='ASPECT_RATIO_512',
    real_prompt_ratio=0.0, max_length=77,
)
batch_sampler = AspectRatioBatchSampler(sampler=RandomSampler(dataset), dataset=dataset,
                                    batch_size=4, aspect_ratios=dataset.aspect_ratio, drop_last=True,
                                    ratio_nums=dataset.ratio_nums, valid_num=0)
dataloader = build_dataloader(dataset, batch_sampler=batch_sampler, num_workers=10)

num_epochs = 1

optimizer = torch.optim.AdamW(lr=1e-5, params=pipe.parameters())
lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=50,
            num_training_steps=(len(dataloader) * num_epochs),
        )

accelerator = Accelerator(
        # mixed_precision=config.mixed_precision,
        # mixed_precision='no',
        mixed_precision='fp16',
        # gradient_accumulation_steps=1
    )

pipe.transformer, optimizer, lr_scheduler = accelerator.prepare(pipe.transformer, optimizer, lr_scheduler)
pipe.text_encoder, pipe.text_encoder_2 = accelerator.prepare(pipe.text_encoder, pipe.text_encoder_2)

for epoch in range(num_epochs):
    progbar = tqdm(dataloader)
    for step, batch in enumerate(progbar):
        optimizer.zero_grad()
        
        # prepare data
        image, prompt = batch[0].to('cuda'), batch[1]
        batch[1] = [x.strip('<|endoftext|>') for x in batch[1]]
        index = torch.randint(0, 1000, size=(len(image),))

        # run model
        model_output, target = pipe.train_step(image, prompt, index)

        assert (model_output).isfinite().all(), model_output
        assert (target).isfinite().all(), target
        loss = ((model_output - target) ** 2).mean()
        accelerator.backward(loss)

        grad_norm = accelerator.clip_grad_norm_(pipe.parameters(), 0.01)

        for p in pipe.parameters():
            if p.grad is not None:
                torch.nan_to_num(p.grad, nan=0, posinf=1e5, neginf=-1e5, out=p.grad)

        optimizer.step()
        lr_scheduler.step()
        progbar.set_description(f"loss: {loss.item():.3f}")

        if (step + 1) % 2500 == 0:
            pipe.save_pretrained(f'/mnt/bn/us-aigc-temp/henry/data/unilatent/epoch_{epoch}_step_{step}/')
