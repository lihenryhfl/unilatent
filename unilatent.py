# Copyright 2024 Stability AI and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import numpy as np
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModel,
    CLIPImageProcessor,
    GPT2Tokenizer
)

from torch.nn.parallel import DistributedDataParallel as DDP

from caption_decoder_v1 import TextDecoder
from utils import pad_mask, EmbedAdapter, SoftPrompter, LayerAggregator

from diffusers import StableDiffusion3Pipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin
from diffusers.models.autoencoders import AutoencoderKL
from orig_transformer import SD3Transformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    is_torch_xla_available,
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusion3Pipeline

        >>> pipe = StableDiffusion3Pipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16
        ... )
        >>> pipe.to("cuda")
        >>> prompt = "A cat holding a sign that says hello world"
        >>> image = pipe(prompt).images[0]
        >>> image.save("sd3.png")
        ```
"""


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class UniLatentPipeline(StableDiffusion3Pipeline):
    r"""
    Args:
        transformer ([`SD3Transformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModelWithProjection`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant,
            with an additional added projection layer that is initialized with a diagonal matrix with the `hidden_size`
            as its dimension.
        text_encoder_2 ([`CLIPTextModelWithProjection`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the
            [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
            variant.
        text_encoder_3 ([`T5EncoderModel`]):
            Frozen text-encoder. Stable Diffusion 3 uses
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel), specifically the
            [t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`CLIPTokenizer`):
            Second Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_3 (`T5TokenizerFast`):
            Tokenizer of class
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
        clip_image_encoder (`CLIPVisionModel`):
            Image encoder of class
            [CLIPVisionModel].
        clip_image_processor (`CLIPImageProcessor`):
            Image processor of class
            [CLIPImageProcessor].
        text_decoder (`TextDecoder`)
            Text decoder of class
            [TextDecoder].
        decoder_tokenizer (`GPT2Tokenizer`)
            Tokenizer of class
            [GPT2Tokenizer]
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->text_decoder->clip_image_encoder->transformer->vae"
    _optional_components = [
        'image_decoder_adapter', 'image_encoder_adapter', 'soft_prompter', 'layer_aggregator', 'dift_image_encoder_adapter', 'dift_clip_adapter']
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds", "negative_pooled_prompt_embeds"]

    def __getattribute__(self, name):
        """
        Deals with issues accessing the model when it is wrapped by a DDP class.
        """
        obj = super().__getattribute__(name)
        if isinstance(obj, DDP):
            obj = obj.module

        return obj

    def __init__(
        self,
        transformer: SD3Transformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer_2: CLIPTokenizer,
        clip_image_encoder: CLIPVisionModel,
        clip_image_processor: CLIPImageProcessor,
        text_decoder: TextDecoder,
        decoder_tokenizer: GPT2Tokenizer,
        image_decoder_adapter: EmbedAdapter = None,
        image_encoder_adapter: EmbedAdapter = None,
        dift_image_encoder_adapter: EmbedAdapter = None,
        dift_clip_adapter: EmbedAdapter = None,
        soft_prompter: SoftPrompter = None,
        layer_aggregator: LayerAggregator = None,
    ):
        super(StableDiffusion3Pipeline).__init__()
        
        # for warning about truncation errors with CLIP
        self.num_warnings = 5

        if clip_image_encoder is not None:
            assert clip_image_processor is not None

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler,
            clip_image_encoder=clip_image_encoder,
            clip_image_processor=clip_image_processor,
            text_decoder=text_decoder,
            decoder_tokenizer=decoder_tokenizer,
            image_decoder_adapter=image_decoder_adapter,
            image_encoder_adapter=image_encoder_adapter,
            dift_image_encoder_adapter=dift_image_encoder_adapter,
            dift_clip_adapter=dift_clip_adapter,
            soft_prompter=soft_prompter,
            layer_aggregator=layer_aggregator,
        )
        self.text_encoder_3 = None
        self.tokenizer_3 = None
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        self.default_sample_size = (
            self.transformer.config.sample_size
            if hasattr(self, "transformer") and self.transformer is not None
            else 128
        )

    def save_pretrained(self, *args, **kwargs):
        self.text_decoder.transformer.lm_head.weight = None
        super().save_pretrained(*args, **kwargs)
        with torch.no_grad():
            self.text_decoder.transformer.lm_head.weight = self.text_decoder.transformer.transformer.wte.weight

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        model = super(cls, cls).from_pretrained(*args, low_cpu_mem_usage=False, device_map=None, **kwargs)
        assert model.text_decoder.transformer.transformer.wte.weight is not None
        model.text_decoder.transformer.lm_head.weight = model.text_decoder.transformer.transformer.wte.weight
        assert model.text_decoder.transformer.lm_head.weight is not None
        return model

    def encode_prompt_clip(
        self, 
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        clip_skip: Optional[int] = None
    ):
        prompt_embed, pooled_prompt_embed = self._get_clip_prompt_embeds(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            clip_skip=clip_skip,
            clip_model_index=0,
        )
        prompt_2_embed, pooled_prompt_2_embed = self._get_clip_prompt_embeds(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            clip_skip=clip_skip,
            clip_model_index=1,
        )
        clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)
        pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)

        return clip_prompt_embeds, pooled_prompt_embeds

    def format_clip_prompt_embeds(
        self, 
        clip_prompt_embeds: torch.FloatTensor,
        num_images_per_prompt: int = 1,
        clip_skip: Optional[int] = None,
        max_sequence_length: int = 256,
    ):
        B = len(clip_prompt_embeds)

        t5_prompt_embed = self._get_t5_prompt_embeds(
            prompt=[""] * B,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            device=clip_prompt_embeds.device,
        )

        clip_prompt_embeds = torch.nn.functional.pad(
            clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
        )

        prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

        return prompt_embeds

    def _scale_noise(self, sample, index, noise):
        assert (index >= 0).all() and (index < len(self.scheduler.sigmas)).all(), f"index: {index}, num sigmas: {len(self.scheduler.sigmas)}"
        sigma = self.scheduler.sigmas[index].to(sample.device).type(sample.dtype).reshape(-1, 1, 1, 1)
        noisy_sample = sigma * noise + (1.0 - sigma) * sample

        gt_noisy_samples = []
        for x, n, ind in zip(sample[:, None], noise[:, None], index[:, None]):
            gt_noisy_sample = self.scheduler.scale_noise(x, self.scheduler.timesteps[ind], n)
            gt_noisy_samples.append(gt_noisy_sample)
        gt_noisy_sample = torch.cat(gt_noisy_samples, axis=0)
        noisy_sample = gt_noisy_sample

        target = noise - sample
        
        return noisy_sample, self.scheduler.timesteps[index].to(sample.device).type(sample.dtype), target

    def encode_image(self, x, dtype=torch.float32):
        """
        Embeds image x into a shared latent variable z.
        """
        if x.min() < 0:
            assert x.min() >= -1.
            x = x * .5 + .5
        
        processed_list = self.clip_image_processor(x, do_rescale=False)['pixel_values']
        z = torch.tensor(np.stack(processed_list)).to(self.device).type(dtype)
        result = self.clip_image_encoder(z, output_hidden_states=True)
        image_embed, pooled_image_embed = result.hidden_states[-2], result.pooler_output
        # image_embed, pooled_image_embed = result.last_hidden_state, result.pooler_output

        B, N, C = image_embed.shape
        pooled_image_embed = pooled_image_embed.reshape(B, 1, C)

        if self._hasattr('image_encoder_adapter'):
            image_embed, pooled_image_embed = self.image_encoder_adapter(image_embed, pooled_image_embed)

        return image_embed, pooled_image_embed

    def encode_text(self, x):
        """
        Embeds text x into a shared latent variable z. 
        """

        # truncate x
        x = self.tokenizer.batch_decode(self.tokenizer(
            x,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).input_ids)
        texts = [s.replace(self.tokenizer.bos_token, '').replace(self.tokenizer.eos_token, '').strip().strip('!').strip() for s in x]

        x = self.tokenizer_2.batch_decode(self.tokenizer_2(
            x,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).input_ids)
        x = [s.replace(self.tokenizer_2.bos_token, '').replace(self.tokenizer_2.eos_token, '').strip().strip('!').strip() for s in x]

        (
            prompt_embeds,
            pooled_prompt_embeds
        ) = self.encode_prompt_clip(prompt=x)

        B, N, C = prompt_embeds.shape

        return prompt_embeds, pooled_prompt_embeds.reshape(B, 1, C)

    def embed_to_decoder(self, embed, pooled_embed, prompt, suffix_input_ids=None):
        B, N, C = embed.shape
        joint_embed = torch.cat([embed, pooled_embed], axis=1)

        processed_prompt = [txt + self.decoder_tokenizer.eos_token for txt in prompt]
        tokens = self.decoder_tokenizer(processed_prompt, return_tensors='pt', truncation=True, max_length=150, padding="longest")
        input_mask = tokens['attention_mask'].to(self.device)
        input_ids = tokens['input_ids'].to(self.device)
        
        assert input_ids.max() < self.text_decoder.transformer.transformer.wte.weight.shape[0], f"{input_ids.max()}, {self.text_decoder.transformer.transformer.wte.weight.shape}"
        llm_out = self.text_decoder(input_ids, joint_embed, attention_mask=input_mask, suffix_input_ids=suffix_input_ids)

        return llm_out.loss

    def decode_loss(self, image, prompt):
        image_embed, pooled_image_embed = self.encode_image(image)
        
        return self.embed_to_decoder(image_embed, pooled_image_embed, prompt)

    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        device = timesteps.device
        sigmas = self.scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        
        return sigma

    def embed_to_denoiser_v2(self, image, embeds, pooled_embeds, indices=None, return_layers=None):
        latent = self.vae.encode(image.to(self.device)).latent_dist.sample()
        latent = (latent - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        noise = torch.randn_like(latent)
        # Sample a random timestep for each image
        # for weighting schemes where we sample timesteps non-uniformly
        if indices is None:
            u = torch.normal(mean=0., std=1., size=(len(latent),), device="cpu")
            u = torch.nn.functional.sigmoid(u)
            indices = (u * self.scheduler.config.num_train_timesteps).long()
        timesteps = self.scheduler.timesteps[indices].to(device=latent.device)

        # Add noise according to flow matching.
        # zt = (1 - texp) * x + texp * z1
        sigmas = self.get_sigmas(timesteps, n_dim=latent.ndim, dtype=latent.dtype)
        noisy_latent = (1.0 - sigmas) * latent + sigmas * noise

        target = noise - latent

        # format prompt_embeds correctly
        B, N, C = embeds.shape
        if self._hasattr('image_decoder_adapter'):
            embeds, pooled_embeds = self.image_decoder_adapter(embeds, pooled_embeds)

        embeds = self.format_clip_prompt_embeds(embeds)
        pooled_embeds = pooled_embeds.reshape(B, C)

        model_output = self.transformer(
                    hidden_states=noisy_latent,
                    timestep=timesteps,
                    encoder_hidden_states=embeds,
                    pooled_projections=pooled_embeds,
                    joint_attention_kwargs=None,
                    return_layers=return_layers
                )

        if return_layers is not None:
            assert len(return_layers) > 0
            assert isinstance(model_output.hidden, tuple), f"{type(model_output)}, {type(model_output.hidden)}"
            return model_output.hidden, target

        return model_output.sample, target
        
    def diffusion_step(self, image, prompt, index):
        prompt_embeds, pooled_prompt_embeds = self.encode_text(prompt)

        return self.embed_to_denoiser(image, prompt_embeds, pooled_prompt_embeds, index)

    def parameters(self, models=None):
        if models is None:
            models = [self.components[k] for k in self.components]
        parameters = []
        for model in models:
            if not hasattr(model, 'parameters'):
                continue
            for p in model.parameters():
                parameters.append(p)

        return parameters

    def named_parameters(self, models=None):
        if models is None:
            models = [self.components[k] for k in self.components]
        parameters = []
        for model in models:
            if not hasattr(model, 'named_parameters'):
                continue
            for n, p in model.named_parameters():
                parameters.append((n, p))

        return parameters


######################## FOR DIFT #########################
    def dift_features(self, image, indices, embed=None, pooled_embed=None, 
        return_layers=12, num_aggregation_steps=1, dataset_conditioning=False,
        ):
        if not (isinstance(return_layers, list) or isinstance(return_layers, tuple)):
            return_layers = [return_layers]
        
        if embed is None:
            embed, pooled_embed = self.encode_text("")

        if self._hasattr('soft_prompter'):
            embed, pooled_embed = self.soft_prompter(embed, pooled_embed)

        if isinstance(indices, torch.Tensor):
            indices = (indices, )
        
        hidden_list = []
        for i in range(num_aggregation_steps):
            hiddens_over_times = []
            for index in indices:
                (_, hidden), _ = self.embed_to_denoiser_v2(
                    image,
                    embed, 
                    pooled_embed,
                    index,
                    return_layers=return_layers)
                hiddens_over_times.append(hidden)
            hidden = torch.cat(hiddens_over_times, axis=-1)
            hidden_list.append(hidden)
        
        # aggregate hidden across steps
        hidden = torch.stack(hidden_list).mean(dim=0)

        prefix_length = self.text_decoder.prefix_length
        if dataset_conditioning:
            prefix_length = prefix_length - 1
        else:
            assert False

        assert hidden.shape[1] >= prefix_length, f"{hidden.shape, self.text_decoder.prefix_length}"
        if self._hasattr('layer_aggregator'):
            hidden = self.layer_aggregator(hidden)
        
        if self._hasattr('dift_image_encoder_adapter'):
            assert hidden is not None
            hidden = self.dift_image_encoder_adapter(hidden)
        elif self._hasattr('image_encoder_adapter'):
            print(
                "Deprecated: Running image_encoder_adapter in the dift_features function in unilatent. ",
                "Please convert to dift_image_encoder_adapter instead."
            )
            assert hidden is not None
            hidden = self.dift_image_encoder_adapter(hidden)

        hidden = hidden[:, :prefix_length]
        
        # basic conversion to work with our framework
        embeds, pooled_embeds = hidden[:, :-1], hidden[:, -1:]
        if self._hasattr('dift_clip_adapter'):
            return self.dift_clip_adapter(embeds, pooled_embeds)
        else:
            return embeds, pooled_embeds

    def _hasattr(self, name):
        if not hasattr(self, name):
            return False
        elif getattr(self, name) is None:
            return False
        elif isinstance(getattr(self, name), list) and getattr(self, name)[0] is None:
            return False

        return True

    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead
            prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_3` and `text_encoder_3`. If not defined, `prompt` is
                will be used instead
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 5.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used instead
            negative_prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_3` and
                `text_encoder_3`. If not defined, `negative_prompt` is used instead
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 256): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            prompt_3,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            clip_skip=self.clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])
                assert self.joint_attention_kwargs is None

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type == "latent":
            image = latents
        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusion3PipelineOutput(images=image)