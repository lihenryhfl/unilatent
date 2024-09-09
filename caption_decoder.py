from typing import Optional

import numpy as np
import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from transformers.modeling_utils import ModuleUtilsMixin

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import ModelMixin

from utils import pad_mask


# Modified from ClipCaptionModel in https://github.com/thu-ml/unidiffuser/blob/main/libs/caption_decoder.py
class TextDecoder(ModelMixin, ConfigMixin, ModuleUtilsMixin):
    """
    Text decoder model for a image-text [UniDiffuser](https://arxiv.org/pdf/2303.06555.pdf) model. This is used to
    generate text from the UniDiffuser image-text embedding.

    Parameters:
        prefix_length (`int`):
            Max number of prefix tokens that will be supplied to the model.
        prefix_inner_dim (`int`):
            The hidden size of the incoming prefix embeddings. For UniDiffuser, this would be the hidden dim of the
            CLIP text encoder.
        vocab_size (`int`, *optional*, defaults to 50257):
            Vocabulary size of the GPT-2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`GPT2Model`] or [`TFGPT2Model`].
        n_positions (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        n_embd (`int`, *optional*, defaults to 768):
            Dimensionality of the embeddings and hidden states.
        n_layer (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        n_inner (`int`, *optional*, defaults to None):
            Dimensionality of the inner feed-forward layers. `None` will set it to 4 times n_embd
        activation_function (`str`, *optional*, defaults to `"gelu"`):
            Activation function, to be selected in the list `["relu", "silu", "gelu", "tanh", "gelu_new"]`.
        resid_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the embeddings.
        attn_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon to use in the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        scale_attn_weights (`bool`, *optional*, defaults to `True`):
            Scale attention weights by dividing by sqrt(hidden_size)..
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        scale_attn_by_inverse_layer_idx (`bool`, *optional*, defaults to `False`):
            Whether to additionally scale attention weights by `1 / layer_idx + 1`.
        reorder_and_upcast_attn (`bool`, *optional*, defaults to `False`):
            Whether to scale keys (K) prior to computing attention (dot-product) and upcast attention
            dot-product/softmax to float() when training with mixed precision.
    """

    _keys_to_ignore_on_load_unexpected = [r"h\.\d+\.attn\.bias", r"h\.\d+\.attn\.masked_bias"]

    @register_to_config
    def __init__(
        self,
        prefix_length: int,
        prefix_inner_dim: int,
        prefix_hidden_dim: int = None,
        vocab_size: int = 50257,  # Start of GPT2 config args
        n_positions: int = 1024,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        n_inner: Optional[int] = None,
        activation_function: str = "gelu_new",
        resid_pdrop: float = 0.1,
        embd_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        initializer_range: float = 0.02,
        scale_attn_weights: bool = True,
        use_cache: bool = True,
        scale_attn_by_inverse_layer_idx: bool = False,
        reorder_and_upcast_attn: bool = False,
        pretrained_transformer: bool = True,
    ):
        super().__init__()
        
        self.ticker = 0
        self.add_bos_embeds = False
        self.prefix_length = prefix_length
        self.prefix_inner_dim = prefix_inner_dim # in the future change this to input dim to be more clear
        self.prefix_hidden_dim = prefix_hidden_dim if prefix_hidden_dim is not None else prefix_inner_dim

        if self.prefix_inner_dim != self.prefix_hidden_dim:
            self.encode_prefix = nn.Sequential(
                nn.Linear(self.prefix_inner_dim, self.prefix_hidden_dim),
                nn.GELU()
            )
            self.decode_prefix = (
                nn.Linear(self.prefix_hidden_dim, n_embd)
            )
        else:
            self.encode_prefix = nn.Identity()

            self.decode_prefix = (
                nn.Linear(self.prefix_hidden_dim, n_embd)
            )

        # for debugging purposes
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        tokenizer.add_tokens([f'<|dataset{i}|>' for i in range(vocab_size - len(tokenizer))])
        self.tokenizer = tokenizer

        gpt_config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=n_inner,
            activation_function=activation_function,
            resid_pdrop=resid_pdrop,
            embd_pdrop=embd_pdrop,
            attn_pdrop=attn_pdrop,
            layer_norm_epsilon=layer_norm_epsilon,
            initializer_range=initializer_range,
            scale_attn_weights=scale_attn_weights,
            use_cache=use_cache,
            scale_attn_by_inverse_layer_idx=scale_attn_by_inverse_layer_idx,
            reorder_and_upcast_attn=reorder_and_upcast_attn,
        )
        
        if pretrained_transformer:
            transformer = GPT2LMHeadModel.from_pretrained('gpt2')
            orig_vocab_size = transformer.config.vocab_size
            transformer.config.vocab_size = vocab_size

            # create new wte module
            wte = torch.nn.modules.Embedding(vocab_size, n_embd)
            wte.weight.data[:orig_vocab_size] = transformer.transformer.wte.weight.data
            transformer.transformer.wte = wte

            # create new lm_head module
            lm_head = torch.nn.Linear(768, vocab_size, bias=False)
            lm_head.weight.data[:orig_vocab_size] = transformer.lm_head.weight.data
            transformer.lm_head = lm_head
        else:
            transformer = GPT2LMHeadModel(gpt_config)

        self.transformer = transformer

    def loss(self, lm_logits, labels, mask):
        lm_logits = self.zero_logits(lm_logits, labels=labels, mask=mask)
        # move labels to correct device to enable model parallelism
        labels = labels.to(lm_logits.device)

        # only use loss on last len(mask) entries
        offset = len(mask[0])

        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., -(offset+1):-1, :].contiguous()
        shift_labels = labels[..., -offset:].contiguous()

        # Flatten the tokens
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.reshape(*shift_labels.shape)
        
        loss = loss * mask.to(loss.device)
        true, pred = shift_labels, shift_logits.max(dim=-1).indices
        assert pred.max() < len(self.tokenizer), f"{pred.max()}, {len(self.tokenizer)}"
        self.ticker += 1
        if self.ticker > 20:
            print('PRED:', self.tokenizer.decode(pred[0]))
            print('TRUE:', self.tokenizer.decode(true[0]))
            self.ticker = 0

        return loss

    def get_prefix_embeds(self, x, suffix_input_ids):
        hidden = self.encode_prefix(x)
        prefix_embeds = self.decode_prefix(hidden)

        if suffix_input_ids is not None:
            suffix_embeds = self.transformer.transformer.wte(suffix_input_ids)
            prefix_embeds = torch.cat([prefix_embeds, suffix_embeds], axis=1)

        if self.add_bos_embeds:
            bos_token_ids = torch.full(size=(len(x), 1), fill_value=self.tokenizer.bos_token_id, dtype=torch.int64, device=x.device)
            bos_embeds = self.transformer.transformer.wte(bos_token_ids)
            prefix_embeds = torch.cat([bos_embeds, prefix_embeds], axis=1)

        return prefix_embeds

    def get_embedding_cat(self, input_ids, features, suffix_input_ids):
        if self.add_bos_embeds:
            input_ids = input_ids[:, 1:]
        embedding_text = self.transformer.transformer.wte(input_ids)
        prefix_embeds = self.get_prefix_embeds(features, suffix_input_ids)
        embedding_cat = torch.cat((prefix_embeds, embedding_text), dim=1)
        
        return embedding_cat

    def forward(
        self,
        input_ids: torch.Tensor,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        suffix_input_ids: Optional[torch.Tensor] = None,
        verbose: bool = False,
    ):
        """
        Args:
            input_ids (`torch.Tensor` of shape `(N, max_seq_len)`):
                Text tokens to use for inference.
            prefix_embeds (`torch.Tensor` of shape `(N, prefix_length, 768)`):
                Prefix embedding to preprend to the embedded tokens.
            attention_mask (`torch.Tensor` of shape `(N, prefix_length + max_seq_len, 768)`, *optional*):
                Attention mask for the prefix embedding.
            labels (`torch.Tensor`, *optional*):
                Labels to use for language modeling.
        """
        embedding_cat = self.get_embedding_cat(input_ids, features, suffix_input_ids)

        if attention_mask is not None:
            padded_mask = pad_mask(attention_mask, prefix_len=self.prefix_length).to(input_ids.device)

        if labels is None:
            dummy_token = self.get_dummy_token(input_ids.shape[0], input_ids.device)
            if self.add_bos_embeds:
                labels = torch.cat((input_ids[:, :1], dummy_token, input_ids[:, 1:]), dim=1)
            else:
                labels = torch.cat((dummy_token, input_ids), dim=1)

        assert embedding_cat.shape[:2] == padded_mask.shape, f"{embedding_cat.shape}, {padded_mask.shape}"
        assert embedding_cat.shape[:2] == labels.shape, f"{embedding_cat.shape}, {labels.shape}"
        assert embedding_cat.shape[-1] == 768, f"{embedding_cat.shape}"
        out = self.transformer(inputs_embeds=embedding_cat, labels=labels, attention_mask=padded_mask)

        # recompute loss our way
        out.loss = self.loss(out.logits, labels, attention_mask).mean()

        return out

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    @torch.no_grad()
    def generate_captions(self, features, eos_token_id, device, suffix_input_ids=None):
        """
        Generate captions given text embedding features. Returns list[L].

        Args:
            features (`torch.Tensor` of shape `(B, L, D)`):
                Text embedding features to generate captions from.
            eos_token_id (`int`):
                The token ID of the EOS token for the text decoder model.
            device:
                Device to perform text generation on.

        Returns:
            `List[str]`: A list of strings generated from the decoder model.
        """
        features = torch.split(features, 1, dim=0)
        generated_tokens = []
        generated_seq_lengths = []
        for feature in features:
            prefix_embeds = self.get_prefix_embeds(feature.to(device), suffix_input_ids)
            # Only support beam search for now
            output_tokens, seq_lengths = self.generate_beam(
                input_embeds=prefix_embeds, device=device, eos_token_id=eos_token_id
            )
            generated_tokens.append(output_tokens[0])
            generated_seq_lengths.append(seq_lengths[0])
        # generated_tokens = torch.stack(generated_tokens)
        generated_seq_lengths = torch.stack(generated_seq_lengths)
        return generated_tokens, generated_seq_lengths

    def zero_logits(self, logits, labels=None, mask=None):
        if labels is not None:
            labels = labels[:, -len(mask[0]):] * mask.to(labels.device)
            for idx in self.tokenizer.get_added_vocab().values():
                if idx == 50256:
                    continue
                assert (labels != idx).all(), f"{idx}, {labels}, {mask}"
        for idx in self.tokenizer.get_added_vocab().values():
            if idx == 50256:
                continue
            logits[..., idx] = -10000.0
        return logits

    @torch.no_grad()
    def generate_beam(
        self,
        input_ids=None,
        input_embeds=None,
        device=None,
        beam_size: int = 5,
        entry_length: int = 67,
        temperature: float = 1.0,
        eos_token_id: Optional[int] = None,
    ):
        """
        Generates text using the given tokenizer and text prompt or token embedding via beam search. This
        implementation is based on the beam search implementation from the [original UniDiffuser
        code](https://github.com/thu-ml/unidiffuser/blob/main/libs/caption_decoder.py#L89).

        Args:
            eos_token_id (`int`, *optional*):
                The token ID of the EOS token for the text decoder model.
            input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`, *optional*):
                Tokenizer indices of input sequence tokens in the vocabulary. One of `input_ids` and `input_embeds`
                must be supplied.
            input_embeds (`torch.Tensor` of shape `(batch_size, seq_len, hidden_size)`, *optional*):
                An embedded representation to directly pass to the transformer as a prefix for beam search. One of
                `input_ids` and `input_embeds` must be supplied.
            device:
                The device to perform beam search on.
            beam_size (`int`, *optional*, defaults to `5`):
                The number of best states to store during beam search.
            entry_length (`int`, *optional*, defaults to `67`):
                The number of iterations to run beam search.
            temperature (`float`, *optional*, defaults to 1.0):
                The temperature to use when performing the softmax over logits from the decoding model.

        Returns:
            `Tuple(torch.Tensor, torch.Tensor)`: A tuple of tensors where the first element is a tensor of generated
            token sequences sorted by score in descending order, and the second element is the sequence lengths
            corresponding to those sequences.
        """
        # Generates text until stop_token is reached using beam search with the desired beam size.
        stop_token_index = eos_token_id
        tokens = None
        scores = None
        seq_lengths = torch.ones(beam_size, device=device, dtype=torch.int)
        is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)

        if input_embeds is not None:
            generated = input_embeds
        else:
            generated = self.transformer.transformer.wte(input_ids)

        for i in range(entry_length):
            outputs = self.transformer(inputs_embeds=generated)
            logits = outputs.logits
            logits = self.zero_logits(logits)
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()

            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]

            next_token_embed = self.transformer.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break

        scores = scores / seq_lengths
        order = scores.argsort(descending=True)
        # tokens tensors are already padded to max_seq_length
        output_texts = [tokens[i] for i in order]
        output_texts = torch.stack(output_texts, dim=0)
        seq_lengths = torch.tensor([seq_lengths[i] for i in order], dtype=seq_lengths.dtype)
        return output_texts, seq_lengths