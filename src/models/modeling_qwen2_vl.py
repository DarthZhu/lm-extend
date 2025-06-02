# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""PyTorch Qwen2-VL model."""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss, LayerNorm

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    ModelOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig, Qwen2VLVisionConfig


if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func

    from transformers.modeling_flash_attention_utils import _flash_attention_forward
else:
    flash_attn_varlen_func = None


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "Qwen2VLConfig"


@dataclass
class Qwen2VLCausalLMOutputWithPast(ModelOutput):
    """
    Base class for Qwen2VL causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None


class Qwen2VLRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[Qwen2VLConfig] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            logger.warning_once(
                "`Qwen2VLRotaryEmbedding` can now be fully parameterized by passing the model config through the "
                "`config` argument. All other arguments will be removed in v4.46"
            )
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block. In contrast to other models, Qwen2_VL has different position ids for thw grids
        # So we expand the inv_freq to shape (3, ...)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).

    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension seperately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        mrope_section(`List(int)`):
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_vision(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    orig_dtype = tensor.dtype
    tensor = tensor.float()
    cos = freqs.cos()
    sin = freqs.sin()
    cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    output = (tensor * cos) + (rotate_half(tensor) * sin)
    output = output.to(orig_dtype)
    return output


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class PatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = LayerNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        return x


class VisionMlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, hidden_act: str) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = ACT2FN[hidden_act]
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class VisionAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, rotary_pos_emb: torch.Tensor = None
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        attention_mask = torch.full(
            [1, seq_length, seq_length], torch.finfo(q.dtype).min, device=q.device, dtype=q.dtype
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


class VisionFlashAttention2(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, rotary_pos_emb: torch.Tensor = None
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output


class VisionSdpaAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, rotary_pos_emb: torch.Tensor = None
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_output = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


QWEN2_VL_VISION_ATTENTION_CLASSES = {
    "eager": VisionAttention,
    "flash_attention_2": VisionFlashAttention2,
    "sdpa": VisionSdpaAttention,
}


class Qwen2VLVisionBlock(nn.Module):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = LayerNorm(config.embed_dim, eps=1e-6)
        self.norm2 = LayerNorm(config.embed_dim, eps=1e-6)
        mlp_hidden_dim = int(config.embed_dim * config.mlp_ratio)

        self.attn = QWEN2_VL_VISION_ATTENTION_CLASSES[attn_implementation](
            config.embed_dim, num_heads=config.num_heads
        )
        self.mlp = VisionMlp(dim=config.embed_dim, hidden_dim=mlp_hidden_dim, hidden_act=config.hidden_act)

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


# Copied from transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm
class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# Copied from transformers.models.qwen2.modeling_qwen2.Qwen2MLP
class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Qwen2VLAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: Qwen2VLConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.rope_scaling = config.rope_scaling

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = Qwen2VLRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        # print("here")
        # print(kwargs)
        # print(self.layer_idx)
        # print(self.num_heads)
        # print(self.head_dim)
        # input()
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
            
        if 'block_list' in kwargs:
            for h in kwargs['block_list']:
                if self.layer_idx==h[0]:
                    '''
                    if h[1]==0:
                        target_head = 1
                    elif h[1]==31:
                        target_head = 30
                    else:
                        target_head = h[1] - 1
                        
                    attn_weights[:, h[1], :, :] = attn_weights[:, target_head, :, :]
                    '''
                    attn_weights[:, h[1], :, :] = 0 

        # Fix precision issues in Qwen2-VL float16 inference
        # Replace inf values with zeros in attention weights to prevent NaN propagation
        if query_states.dtype == torch.float16:
            attn_weights = torch.where(torch.isinf(attn_weights), torch.zeros_like(attn_weights), attn_weights)

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Qwen2VLFlashAttention2(Qwen2VLAttention):
    """
    Qwen2VL flash attention module, following Qwen2VL attention module. This module inherits from `Qwen2VLAttention`
    as the weights of the module stays untouched. The only required change would be on the forward pass
    where it needs to correctly call the public API of flash attention and deal with padding tokens
    in case the input contains any of them. Additionally, for sliding window attention, we apply SWA only to the bottom
    config.max_window_layers layers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings

        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window
        else:
            sliding_window = None

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            sliding_window=sliding_window,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Qwen2VLSdpaAttention(Qwen2VLAttention):
    """
    Qwen2 attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `Qwen2Attention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from Qwen2Attention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "Qwen2VLModel is using Qwen2VLSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


QWEN2_VL_ATTENTION_CLASSES = {
    "eager": Qwen2VLAttention,
    "flash_attention_2": Qwen2VLFlashAttention2,
    "sdpa": Qwen2VLSdpaAttention,
}


class Qwen2VLDecoderLayer(nn.Module):
    def __init__(self, config: Qwen2VLConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.use_sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        self.self_attn = QWEN2_VL_ATTENTION_CLASSES["eager"](config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


QWEN2VL_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Qwen2VLConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Qwen2VL Model outputting raw hidden-states without any specific head on top.",
    QWEN2VL_START_DOCSTRING,
)
class Qwen2VLPreTrainedModel(PreTrainedModel):
    config_class = Qwen2VLConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2VLDecoderLayer", "Qwen2VLVisionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class Qwen2VisionTransformerPretrainedModel(Qwen2VLPreTrainedModel):
    config_class = Qwen2VLVisionConfig
    _no_split_modules = ["Qwen2VLVisionBlock"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.spatial_merge_size = config.spatial_merge_size

        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
        )

        head_dim = config.embed_dim // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [Qwen2VLVisionBlock(config, config._attn_implementation) for _ in range(config.depth)]
        )
        self.merger = PatchMerger(
            dim=config.hidden_size, context_dim=config.embed_dim, spatial_merge_size=config.spatial_merge_size
        )
        self.gradient_checkpointing = False

    def get_dtype(self) -> torch.dtype:
        return self.blocks[0].mlp.fc2.weight.dtype

    def get_device(self) -> torch.device:
        return self.blocks[0].mlp.fc2.weight.device

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for blk in self.blocks:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    blk.__call__, hidden_states, cu_seqlens, rotary_pos_emb
                )
            else:
                hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)

        return self.merger(hidden_states)


@add_start_docstrings(
    "The bare Qwen2VL Model outputting raw hidden-states without any specific head on top.",
    QWEN2VL_START_DOCSTRING,
)
class Qwen2VLModel(Qwen2VLPreTrainedModel):
    def __init__(self, config: Qwen2VLConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2VLDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2VLRotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    kwargs,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    # Copied from transformers.models.phi3.modeling_phi3.Phi3Model._update_causal_mask
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if (
            self.config._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # SlidingWindowCache or StaticCache
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        # DynamicCache or no cache
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    # Copied from transformers.models.mistral.modeling_mistral.MistralModel._prepare_4d_causal_attention_mask_with_cache_position with Mistral->Qwen2VL
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config: Qwen2VLConfig,
        past_key_values: Cache,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
            config (`Qwen2VLConfig`):
                The model's configuration class
            past_key_values (`Cache`):
                The cache class that is being used currently to generate
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            diagonal_attend_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            if config.sliding_window is not None:
                # if we have sliding window, we should not attend to tokens beyond sliding window length, so we mask them out also
                # the check is needed to verify is current checkpoint was trained with sliding window or not
                if not isinstance(past_key_values, SlidingWindowCache) or sequence_length > target_length:
                    sliding_attend_mask = torch.arange(target_length, device=device) <= (
                        cache_position.reshape(-1, 1) - config.sliding_window
                    )
                    diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        return causal_mask


QWEN2_VL_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        pixel_values (`torch.FloatTensor` of shape `(seq_length, num_channels * image_size * image_size)):
            The tensors corresponding to the input images. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`Qwen2VLImageProcessor.__call__`] for details. [`Qwen2VLProcessor`] uses
            [`Qwen2VLImageProcessor`] for processing images.
        pixel_values_videos (`torch.FloatTensor` of shape `(seq_length, num_channels * temporal_size * image_size * image_size)):
            The tensors corresponding to the input videos. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`Qwen2VLImageProcessor.__call__`] for details. [`Qwen2VLProcessor`] uses
            [`Qwen2VLImageProcessor`] for processing videos.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
"""


class Qwen2VLForConditionalGeneration(Qwen2VLPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen2VisionTransformerPretrainedModel._from_config(config.vision_config)
        self.model = Qwen2VLModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rope_deltas = None  # cache rope_deltas here

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with mordern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embeddin for text part.
            Examples:
                Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [3, 4, 5, 6, 7]
                text height position_ids: [3, 4, 5, 6, 7]
                text width position_ids: [3, 4, 5, 6, 7]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device
            )
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    @add_start_docstrings_to_model_forward(QWEN2_VL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Qwen2VLCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        >>> model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )
                video_mask = (
                    (input_ids == self.config.video_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and input_ids is not None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (cache_position is not None and cache_position[0] == 0) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if cache_position[0] != 0:
            pixel_values = None
            pixel_values_videos = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            model_inputs = {"input_ids": input_ids, "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = inputs_embeds.shape
                device = inputs_embeds.device
            else:
                batch_size, sequence_length = input_ids.shape
                device = input_ids.device

            attention_mask = self.model._prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_cache_shape(),
                dtype=self.lm_head.weight.dtype,
                device=device,
                cache_position=cache_position,
                batch_size=batch_size,
                config=self.config,
                past_key_values=past_key_values,
            )

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values_videos,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "cache_position": cache_position,
            }
        )
        return model_inputs




















































# # SPDX-License-Identifier: Apache-2.0

# # Adapted from
# # https://github.com/huggingface/transformers/blob/19e6e80e10118f855137b90740936c0b11ac397f/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py
# # Copyright 2024 The Qwen team.
# # Copyright 2023 The vLLM team.
# # Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# #
# # This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# # and OPT implementations in this library. It has been modified from its
# # original forms to accommodate minor architectural differences compared
# # to GPT-NeoX and OPT used by the Meta AI team that trained the model.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# """Inference-only Qwen2-VL model compatible with HuggingFace weights."""
# from functools import cached_property, partial
# from typing import (Any, Callable, Iterable, List, Literal, Mapping, Optional,
#                     Set, Tuple, Type, TypedDict, Union)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange, repeat
# from transformers import BatchFeature
# from transformers.models.qwen2_vl import (Qwen2VLImageProcessor,
#                                           Qwen2VLProcessor)
# from transformers.models.qwen2_vl.configuration_qwen2_vl import (
#     Qwen2VLConfig, Qwen2VLVisionConfig)
# from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

# from vllm.attention import AttentionMetadata
# from vllm.config import VllmConfig
# from vllm.distributed import parallel_state, tensor_model_parallel_all_gather
# from vllm.distributed import utils as dist_utils
# from vllm.logger import init_logger
# from vllm.model_executor import SamplingMetadata
# from vllm.model_executor.layers.activation import QuickGELU
# from vllm.model_executor.layers.linear import (ColumnParallelLinear,
#                                                RowParallelLinear)
# from vllm.model_executor.layers.quantization import QuantizationConfig
# from vllm.model_executor.layers.quantization.gptq import GPTQConfig
# from vllm.model_executor.layers.quantization.gptq_marlin import (
#     GPTQMarlinConfig)
# from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
# from vllm.model_executor.model_loader.weight_utils import default_weight_loader
# from vllm.model_executor.models.module_mapping import MultiModelKeys
# from vllm.multimodal import MULTIMODAL_REGISTRY
# from vllm.multimodal.inputs import (ImageItem, ModalityData,
#                                     MultiModalFieldConfig, MultiModalKwargs,
#                                     VideoItem)
# from vllm.multimodal.parse import (ImageSize, ModalityDataItems,
#                                    MultiModalDataItems, MultiModalDataParser)
# from vllm.multimodal.processing import (BaseMultiModalProcessor,
#                                         BaseProcessingInfo, PromptReplacement)
# from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
# from vllm.platforms import _Backend
# from vllm.sequence import IntermediateTensors
# from vllm.transformers_utils.config import uses_mrope

# from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsMultiModal, SupportsPP
# from vllm.model_executor.models.utils import (AutoWeightsLoader, WeightsMapper,
#                     init_vllm_registered_model, maybe_prefix,
#                     merge_multimodal_embeddings)
# from vllm.model_executor.models.vision import get_vit_attn_backend

# logger = init_logger(__name__)

# # For profile run
# _MAX_FRAMES_PER_VIDEO = 16

# # === Vision Inputs === #


# class Qwen2VLImagePixelInputs(TypedDict):
#     type: Literal["pixel_values"]
#     pixel_values: torch.Tensor
#     """Shape:
#     `(num_patches, num_channels * patch_size * patch_size)`
#     """

#     image_grid_thw: torch.Tensor
#     """Shape: `(num_images, 3)`
#     This should be in `(grid_t, grid_h, grid_w)` format.
#     """


# class Qwen2VLImageEmbeddingInputs(TypedDict):
#     type: Literal["image_embeds"]
#     image_embeds: torch.Tensor
#     """Supported types:
#     - List[`torch.Tensor`]: A list of tensors holding all images' features.
#         Each tensor holds an image's features.
#     - `torch.Tensor`: A tensor holding all images' features
#         (concatenation of all images' feature tensors).
    
#     Tensor shape: `(num_image_features, hidden_size)`
#     - `num_image_features` varies based on
#         the number and resolution of the images.
#     - `hidden_size` must match the hidden size of language model backbone.
#     """

#     image_grid_thw: torch.Tensor
#     """Shape: `(num_images, 3)`
#     This should be in `(grid_t, grid_h, grid_w)` format.
#     """


# Qwen2VLImageInputs = Union[Qwen2VLImagePixelInputs,
#                            Qwen2VLImageEmbeddingInputs]


# class Qwen2VLVideoPixelInputs(TypedDict):
#     type: Literal["pixel_values_videos"]
#     pixel_values_videos: torch.Tensor
#     """Shape:
#     `(num_patches,
#       num_channels * temporal_patch_size * patch_size * patch_size)`
#     """

#     video_grid_thw: torch.Tensor
#     """Shape: `(num_videos, 3)`

#     This should be in `(grid_t, grid_h, grid_w)` format.
#     """


# class Qwen2VLVideoEmbeddingInputs(TypedDict):
#     type: Literal["video_embeds"]
#     video_embeds: torch.Tensor
#     """Supported types:
#     - List[`torch.Tensor`]: A list of tensors holding all videos' features.
#         Each tensor holds an video's features.
#     - `torch.Tensor`: A tensor holding all videos' features
#         (concatenation of all videos' feature tensors).
    
#     Tensor shape: `(num_image_features, hidden_size)`
#     - `num_image_features` varies based on 
#         the number and resolution of the videos.
#     - `hidden_size` must match the hidden size of language model backbone.
#     """

#     video_grid_thw: torch.Tensor
#     """Shape: `(num_videos, 3)`
#     This should be in `(grid_t, grid_h, grid_w)` format.
#     """


# Qwen2VLVideoInputs = Union[Qwen2VLVideoPixelInputs,
#                            Qwen2VLVideoEmbeddingInputs]

# # === Vision Encoder === #


# class Qwen2VisionMLP(nn.Module):

#     def __init__(
#         self,
#         in_features: int,
#         hidden_features: int,
#         act_layer: Type[nn.Module] = QuickGELU,
#         quant_config: Optional[QuantizationConfig] = None,
#         prefix: str = "",
#     ):
#         super().__init__()
#         self.fc1 = ColumnParallelLinear(in_features,
#                                         hidden_features,
#                                         quant_config=quant_config,
#                                         prefix=f"{prefix}.fc1")
#         self.act = act_layer()
#         self.fc2 = RowParallelLinear(hidden_features,
#                                      in_features,
#                                      quant_config=quant_config,
#                                      prefix=f"{prefix}.fc2")

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x_parallel, _ = self.fc1(x)
#         x_parallel = self.act(x_parallel)
#         x, _ = self.fc2(x_parallel)
#         return x


# def rotate_half(x: torch.Tensor, interleaved: bool = False) -> torch.Tensor:
#     if not interleaved:
#         x1, x2 = x.chunk(2, dim=-1)
#         return torch.cat((-x2, x1), dim=-1)
#     else:
#         x1, x2 = x[..., ::2], x[..., 1::2]
#         return rearrange(torch.stack((-x2, x1), dim=-1),
#                          "... d two -> ... (d two)",
#                          two=2)


# def apply_rotary_emb_torch(x: torch.Tensor,
#                            cos: torch.Tensor,
#                            sin: torch.Tensor,
#                            interleaved: bool = False) -> torch.Tensor:
#     """
#     x: (batch_size, seqlen, nheads, headdim)
#     cos, sin: (seqlen, rotary_dim / 2) or (batch_size, seqlen, rotary_dim / 2)
#     """
#     ro_dim = cos.shape[-1] * 2
#     assert ro_dim <= x.shape[-1]
#     cos = repeat(
#         cos,
#         "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
#     sin = repeat(
#         sin,
#         "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
#     return torch.cat(
#         [
#             x[..., :ro_dim] * cos +
#             rotate_half(x[..., :ro_dim], interleaved) * sin, x[..., ro_dim:]
#         ],
#         dim=-1,
#     )


# def apply_rotary_pos_emb_vision(t: torch.Tensor,
#                                 freqs: torch.Tensor) -> torch.Tensor:
#     t_ = t.float()
#     cos = freqs.cos()
#     sin = freqs.sin()
#     output = apply_rotary_emb_torch(t_, cos, sin).type_as(t)
#     return output


# class Qwen2VisionAttention(nn.Module):

#     def __init__(
#         self,
#         embed_dim: int,
#         num_heads: int,
#         projection_size: int,
#         quant_config: Optional[QuantizationConfig] = None,
#         prefix: str = "",
#     ) -> None:
#         super().__init__()
#         # Per attention head and per partition values.
#         world_size = parallel_state.get_tensor_model_parallel_world_size()
#         self.tp_size = world_size
#         self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
#         self.hidden_size_per_attention_head = dist_utils.divide(
#             projection_size, num_heads)
#         self.num_attention_heads_per_partition = dist_utils.divide(
#             num_heads, world_size)

#         self.qkv = ColumnParallelLinear(input_size=embed_dim,
#                                         output_size=3 * projection_size,
#                                         quant_config=quant_config,
#                                         prefix=f"{prefix}.qkv")
#         self.proj = RowParallelLinear(input_size=projection_size,
#                                       output_size=embed_dim,
#                                       quant_config=quant_config,
#                                       prefix=f"{prefix}.proj")

#         # Detect attention implementation.
#         self.attn_backend: _Backend = get_vit_attn_backend(support_fa=True)
#         if self.attn_backend not in {
#                 _Backend.FLASH_ATTN, _Backend.TORCH_SDPA, _Backend.XFORMERS
#         }:
#             raise RuntimeError(
#                 f"Qwen2-VL does not support {self.attn_backend} backend now.")

#     def split_qkv(self, qkv: torch.Tensor) -> tuple[torch.Tensor, ...]:
#         # [s, b, 3 * head * head_dim]
#         seq_len, bs, _ = qkv.shape
#         if self.tp_size > 1:
#             qkv = tensor_model_parallel_all_gather(qkv)

#         # [s, b, 3 * head * head_dim] -> 3 * [s, b, head * head_dim]
#         q, k, v = qkv.chunk(3, dim=2)

#         # 3 * [s, b, head * head_dim]
#         if self.tp_size > 1:
#             splitter = partial(dist_utils.split_tensor_along_last_dim,
#                                num_partitions=self.tp_size)
#             q = splitter(q)[self.tp_rank]
#             k = splitter(k)[self.tp_rank]
#             v = splitter(v)[self.tp_rank]

#         # 3 * [s, b, head * head_dim] -> 3 * [s, b, head, head_dim]
#         new_shape = (seq_len, bs, self.num_attention_heads_per_partition,
#                      self.hidden_size_per_attention_head)
#         q, k, v = (x.view(*new_shape) for x in (q, k, v))
#         return q, k, v

#     def forward(
#         self,
#         x: torch.Tensor,
#         cu_seqlens: torch.Tensor,
#         rotary_pos_emb: torch.Tensor,
#     ) -> torch.Tensor:

#         # [s, b, c] --> [s, b, 3 * head * head_dim]
#         x, _ = self.qkv(x)

#         # [s, b, 3 * head * head_dim] -> 3 * [s, b, head, head_dim]
#         q, k, v = self.split_qkv(x)
#         batch_size = q.shape[1]

#         q, k, v = (rearrange(x, "s b ... -> b s ...").contiguous()
#                    for x in (q, k, v))
#         if rotary_pos_emb is not None:
#             q = apply_rotary_pos_emb_vision(q, rotary_pos_emb)
#             k = apply_rotary_pos_emb_vision(k, rotary_pos_emb)

#         if self.attn_backend == _Backend.FLASH_ATTN:
#             # from vllm_flash_attn.flash_attn_interface import (
#             #   flash_attn_varlen_func)
#             from flash_attn import flash_attn_varlen_func

#             q, k, v = (rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v])

#             max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
#             output = flash_attn_varlen_func(q,
#                                             k,
#                                             v,
#                                             cu_seqlens_q=cu_seqlens,
#                                             cu_seqlens_k=cu_seqlens,
#                                             max_seqlen_q=max_seqlen,
#                                             max_seqlen_k=max_seqlen,
#                                             dropout_p=0,
#                                             causal=False)

#             context_layer = rearrange(output,
#                                       "(b s) ... -> b s ...",
#                                       b=batch_size)
#         elif self.attn_backend == _Backend.TORCH_SDPA:
#             seq_length = q.size(1)
#             q, k, v = (rearrange(x, "b s h d -> b h s d") for x in [q, k, v])
#             attention_mask = torch.zeros([1, seq_length, seq_length],
#                                          device=q.device,
#                                          dtype=torch.bool)
#             for i in range(1, len(cu_seqlens)):
#                 attention_mask[..., cu_seqlens[i - 1]:cu_seqlens[i],
#                                cu_seqlens[i - 1]:cu_seqlens[i]] = True
#             output = F.scaled_dot_product_attention(q,
#                                                     k,
#                                                     v,
#                                                     attention_mask,
#                                                     dropout_p=0.0)
#             context_layer = rearrange(output, "b h s d -> b s h d ")
#         elif self.attn_backend == _Backend.XFORMERS:
#             from xformers import ops as xops
#             from xformers.ops.fmha.attn_bias import BlockDiagonalMask

#             seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
#             attn_bias = BlockDiagonalMask.from_seqlens(q_seqlen=seqlens,
#                                                        kv_seqlen=None)

#             context_layer = xops.memory_efficient_attention_forward(
#                 q, k, v, attn_bias=attn_bias, p=0, scale=None)
#         context_layer = rearrange(context_layer,
#                                   "b s h d -> s b (h d)").contiguous()

#         output, _ = self.proj(context_layer)
#         return output


# class Qwen2VisionBlock(nn.Module):

#     def __init__(
#         self,
#         dim: int,
#         num_heads: int,
#         mlp_ratio: float,
#         act_layer: Type[nn.Module] = QuickGELU,
#         norm_layer: Optional[Callable[[int], nn.Module]] = None,
#         quant_config: Optional[QuantizationConfig] = None,
#         prefix: str = "",
#     ) -> None:
#         super().__init__()
#         if norm_layer is None:
#             norm_layer = partial(nn.LayerNorm, eps=1e-6)
#         self.norm1 = norm_layer(dim)
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)

#         self.attn = Qwen2VisionAttention(embed_dim=dim,
#                                          num_heads=num_heads,
#                                          projection_size=dim,
#                                          quant_config=quant_config,
#                                          prefix=f"{prefix}.attn")
#         self.mlp = Qwen2VisionMLP(dim,
#                                   mlp_hidden_dim,
#                                   act_layer=act_layer,
#                                   quant_config=quant_config,
#                                   prefix=f"{prefix}.mlp")

#     def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor,
#                 rotary_pos_emb: torch.Tensor) -> torch.Tensor:
#         x = x + self.attn(self.norm1(x),
#                           cu_seqlens=cu_seqlens,
#                           rotary_pos_emb=rotary_pos_emb)
#         x = x + self.mlp(self.norm2(x))
#         return x


# class Qwen2VisionPatchEmbed(nn.Module):

#     def __init__(
#         self,
#         patch_size: int = 14,
#         temporal_patch_size: int = 2,
#         in_channels: int = 3,
#         embed_dim: int = 1152,
#     ) -> None:
#         super().__init__()
#         self.patch_size = patch_size
#         self.temporal_patch_size = temporal_patch_size
#         self.embed_dim = embed_dim

#         kernel_size = (temporal_patch_size, patch_size, patch_size)
#         self.proj = nn.Conv3d(in_channels,
#                               embed_dim,
#                               kernel_size=kernel_size,
#                               stride=kernel_size,
#                               bias=False)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         L, C = x.shape
#         x = x.view(L, -1, self.temporal_patch_size, self.patch_size,
#                    self.patch_size)
#         x = self.proj(x).view(L, self.embed_dim)
#         return x


# class Qwen2VisionPatchMerger(nn.Module):

#     def __init__(
#         self,
#         d_model: int,
#         context_dim: int,
#         norm_layer: Optional[Callable[[int], nn.Module]] = None,
#         spatial_merge_size: int = 2,
#         quant_config: Optional[QuantizationConfig] = None,
#         prefix: str = "",
#     ) -> None:
#         super().__init__()
#         self.hidden_size = context_dim * (spatial_merge_size**2)
#         if norm_layer is None:
#             norm_layer = partial(nn.LayerNorm, eps=1e-6)
#         self.ln_q = norm_layer(context_dim)
#         self.mlp = nn.ModuleList([
#             ColumnParallelLinear(self.hidden_size,
#                                  self.hidden_size,
#                                  bias=True,
#                                  quant_config=quant_config,
#                                  prefix=f"{prefix}.mlp.0"),
#             nn.GELU(),
#             RowParallelLinear(self.hidden_size,
#                               d_model,
#                               bias=True,
#                               quant_config=quant_config,
#                               prefix=f"{prefix}.mlp.2"),
#         ])

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.ln_q(x)
#         x = x.view(-1, self.hidden_size)

#         mlp_fc1, mlp_act, mlp_fc2 = self.mlp
#         x_parallel, _ = mlp_fc1(x)
#         x_parallel = mlp_act(x_parallel)
#         out, _ = mlp_fc2(x_parallel)
#         return out


# class Qwen2VisionRotaryEmbedding(nn.Module):

#     def __init__(self, dim: int, theta: float = 10000.0) -> None:
#         super().__init__()
#         self.dim = dim
#         self.theta = theta
#         inv_freq = 1.0 / (theta
#                           **(torch.arange(0, dim, 2, dtype=torch.float) / dim))
#         self.register_buffer("inv_freq", inv_freq, persistent=False)
#         self._seq_len_cached = 0
#         self._freqs_cached = None

#     def update_freqs_cache(self, seqlen: int) -> None:
#         if seqlen > self._seq_len_cached:
#             seqlen *= 2
#             self._seq_len_cached = seqlen
#             self.inv_freq = 1.0 / (self.theta**(torch.arange(
#                 0, self.dim, 2, dtype=torch.float, device=self.inv_freq.device)
#                                                 / self.dim))
#             seq = torch.arange(seqlen,
#                                device=self.inv_freq.device,
#                                dtype=self.inv_freq.dtype)
#             freqs = torch.outer(seq, self.inv_freq)
#             self._freqs_cached = freqs

#     def forward(self, seqlen: int) -> torch.Tensor:
#         self.update_freqs_cache(seqlen)
#         return self._freqs_cached[:seqlen]


# class Qwen2VisionTransformer(nn.Module):

#     def __init__(
#         self,
#         vision_config: Qwen2VLVisionConfig,
#         norm_eps: float = 1e-6,
#         quant_config: Optional[QuantizationConfig] = None,
#         prefix: str = "",
#     ) -> None:
#         super().__init__()

#         patch_size = vision_config.patch_size
#         temporal_patch_size = vision_config.temporal_patch_size
#         spatial_merge_size = vision_config.spatial_merge_size
#         in_channels = vision_config.in_channels
#         hidden_size = vision_config.hidden_size
#         embed_dim = vision_config.embed_dim
#         depth = vision_config.depth
#         num_heads = vision_config.num_heads
#         mlp_ratio = vision_config.mlp_ratio

#         self.spatial_merge_size = spatial_merge_size
#         self.num_heads = num_heads
#         self.embed_dim = embed_dim

#         self.patch_embed = Qwen2VisionPatchEmbed(
#             patch_size=patch_size,
#             temporal_patch_size=temporal_patch_size,
#             in_channels=in_channels,
#             embed_dim=embed_dim,
#         )

#         norm_layer = partial(nn.LayerNorm, eps=norm_eps)
#         head_dim = embed_dim // num_heads
#         self.rotary_pos_emb = Qwen2VisionRotaryEmbedding(head_dim // 2)

#         self.blocks = nn.ModuleList([
#             Qwen2VisionBlock(dim=embed_dim,
#                              num_heads=num_heads,
#                              mlp_ratio=mlp_ratio,
#                              norm_layer=norm_layer,
#                              quant_config=quant_config,
#                              prefix=f"{prefix}.blocks.{layer_idx}")
#             for layer_idx in range(depth)
#         ])
#         self.merger = Qwen2VisionPatchMerger(
#             d_model=hidden_size,
#             context_dim=embed_dim,
#             norm_layer=norm_layer,
#             quant_config=quant_config,
#             prefix=f"{prefix}.merger",
#         )

#     @property
#     def dtype(self) -> torch.dtype:
#         return self.patch_embed.proj.weight.dtype

#     @property
#     def device(self) -> torch.device:
#         return self.patch_embed.proj.weight.device

#     def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
#         pos_ids = []
#         for t, h, w in grid_thw:
#             hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
#             wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
#             hpos_ids = hpos_ids.reshape(
#                 h // self.spatial_merge_size,
#                 self.spatial_merge_size,
#                 w // self.spatial_merge_size,
#                 self.spatial_merge_size,
#             ).permute(0, 2, 1, 3).flatten()
#             wpos_ids = wpos_ids.reshape(
#                 h // self.spatial_merge_size,
#                 self.spatial_merge_size,
#                 w // self.spatial_merge_size,
#                 self.spatial_merge_size,
#             ).permute(0, 2, 1, 3).flatten()
#             pos_ids.append(
#                 torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
#         pos_ids = torch.cat(pos_ids, dim=0)
#         max_grid_size = grid_thw[:, 1:].max()
#         rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
#         rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
#         return rotary_pos_emb

#     def forward(
#         self,
#         x: torch.Tensor,
#         grid_thw: torch.Tensor,
#     ) -> torch.Tensor:
#         # patchify
#         x = x.to(device=self.device, dtype=self.dtype)
#         x = self.patch_embed(x)

#         # compute position embedding
#         rotary_pos_emb = self.rot_pos_emb(grid_thw)

#         # compute cu_seqlens
#         cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2],
#                                              grid_thw[:, 0]).cumsum(
#                                                  dim=0, dtype=torch.int32)
#         cu_seqlens = F.pad(cu_seqlens, (1, 0), "constant", 0)

#         # transformers
#         x = x.unsqueeze(1)
#         for blk in self.blocks:
#             x = blk(x, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)

#         # adapter
#         x = self.merger(x)

#         return x

#     def load_weights(self, weights: Iterable[Tuple[str,
#                                                    torch.Tensor]]) -> Set[str]:
#         stacked_params_mapping = [
#             # (param_name, shard_name, shard_id)
#             ("qkv_proj", "q_proj", "q"),
#             ("qkv_proj", "k_proj", "k"),
#             ("qkv_proj", "v_proj", "v"),
#         ]
#         params_dict = dict(self.named_parameters(remove_duplicate=False))
#         loaded_params: Set[str] = set()

#         for name, loaded_weight in weights:
#             for (param_name, weight_name, shard_id) in stacked_params_mapping:
#                 if weight_name not in name:
#                     continue
#                 name = name.replace(weight_name, param_name)

#                 param = params_dict[name]
#                 weight_loader = param.weight_loader
#                 weight_loader(param, loaded_weight, shard_id)
#                 break
#             else:
#                 param = params_dict[name]
#                 weight_loader = getattr(param, "weight_loader",
#                                         default_weight_loader)
#                 weight_loader(param, loaded_weight)
#             loaded_params.add(name)
#         return loaded_params


# class Qwen2VLEmbeddingItems(ModalityDataItems[dict[str, torch.Tensor],
#                                               dict[str, torch.Tensor]]):

#     def __init__(self, data: dict, modality: str) -> None:
#         super().__init__(data, modality)

#         grid_thw = data[f"{modality}_grid_thw"]
#         slice_idxs = [0] + grid_thw.prod(-1).cumsum_(0).tolist()
#         self._slices = [
#             slice(slice_idxs[i], slice_idxs[i + 1])
#             for i in range(len(grid_thw))
#         ]

#     def get_count(self) -> int:
#         return len(self.data[f"{self.modality}_grid_thw"])

#     def get(self, index: int) -> dict[str, torch.Tensor]:
#         out = {}
#         for k, v in self.data.items():
#             if v != f"{self.modality}_grid_thw":
#                 v = v[self._slices[index]]

#             out[k] = v

#         return out

#     def get_processor_data(self) -> Mapping[str, object]:
#         return {}

#     def get_passthrough_data(self) -> Mapping[str, object]:
#         return self.data


# class Qwen2VLImageEmbeddingItems(Qwen2VLEmbeddingItems):

#     def __init__(self, data: dict) -> None:
#         super().__init__(data, "image")


# class Qwen2VLVideoEmbeddingItems(Qwen2VLEmbeddingItems):

#     def __init__(self, data: dict) -> None:
#         super().__init__(data, "video")


# class Qwen2VLMultiModalDataParser(MultiModalDataParser):

#     def _parse_image_data(
#         self,
#         data: Union[dict[str, torch.Tensor], ModalityData[ImageItem]],
#     ) -> ModalityDataItems[Any, Any]:
#         if isinstance(data, dict):
#             return Qwen2VLEmbeddingItems(data, modality="image")

#         return super()._parse_image_data(data)

#     def _parse_video_data(
#         self,
#         data: Union[dict[str, torch.Tensor], ModalityData[VideoItem]],
#     ) -> ModalityDataItems[Any, Any]:
#         if isinstance(data, dict):
#             return Qwen2VLEmbeddingItems(data, modality="video")

#         return super()._parse_video_data(data)


# class Qwen2VLProcessingInfo(BaseProcessingInfo):

#     def get_hf_config(self):
#         return self.ctx.get_hf_config(Qwen2VLConfig)

#     def get_hf_processor(
#         self,
#         *,
#         min_pixels: Optional[int] = None,
#         max_pixels: Optional[int] = None,
#     ) -> Qwen2VLProcessor:
#         hf_processor = self.ctx.get_hf_processor(Qwen2VLProcessor)
#         image_processor = hf_processor.image_processor  # type: ignore
#         assert isinstance(image_processor, Qwen2VLImageProcessor)

#         if min_pixels:
#             image_processor.min_pixels = min_pixels
#         if max_pixels:
#             image_processor.max_pixels = max_pixels
#         if max_pixels or min_pixels:
#             image_processor.size = {
#                 "min_pixels": image_processor.min_pixels,
#                 "max_pixels": image_processor.max_pixels,
#             }

#         return hf_processor

#     def get_image_processor(
#         self,
#         *,
#         min_pixels: Optional[int] = None,
#         max_pixels: Optional[int] = None,
#     ):
#         hf_processor = self.get_hf_processor(min_pixels=min_pixels,
#                                              max_pixels=max_pixels)
#         image_processor = hf_processor.image_processor  # type: ignore
#         assert isinstance(image_processor, Qwen2VLImageProcessor)
#         return image_processor

#     def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
#         return {"image": None, "video": None}

#     def get_mm_max_tokens_per_item(
#         self,
#         seq_len: int,
#         mm_counts: Mapping[str, int],
#     ) -> Mapping[str, int]:
#         return {
#             "image": self.get_max_image_tokens(),
#             "video": self.get_max_video_tokens(seq_len),
#         }

#     def _get_vision_info(
#         self,
#         *,
#         image_width: int,
#         image_height: int,
#         num_frames: int = 1,
#         do_resize: bool = True,
#         image_processor: Optional[Qwen2VLImageProcessor],
#     ) -> tuple[ImageSize, int]:
#         if image_processor is None:
#             image_processor = self.get_image_processor()

#         hf_config = self.get_hf_config()
#         vision_config = hf_config.vision_config
#         patch_size = vision_config.patch_size
#         merge_size = vision_config.spatial_merge_size
#         temporal_patch_size = vision_config.temporal_patch_size

#         if do_resize:
#             resized_height, resized_width = smart_resize(
#                 height=image_height,
#                 width=image_width,
#                 factor=patch_size * merge_size,
#                 min_pixels=image_processor.min_pixels,
#                 max_pixels=image_processor.max_pixels,
#             )
#             preprocessed_size = ImageSize(width=resized_width,
#                                           height=resized_height)
#         else:
#             preprocessed_size = ImageSize(width=image_width,
#                                           height=image_height)

#         grid_t = max(num_frames // temporal_patch_size, 1)
#         grid_h = preprocessed_size.height // patch_size
#         grid_w = preprocessed_size.width // patch_size

#         num_patches = grid_t * grid_h * grid_w
#         num_vision_tokens = num_patches // (merge_size**2)

#         return preprocessed_size, num_vision_tokens

#     def get_num_image_tokens(
#         self,
#         *,
#         image_width: int,
#         image_height: int,
#         image_processor: Optional[Qwen2VLImageProcessor],
#     ) -> int:
#         _, num_image_tokens = self._get_vision_info(
#             image_width=image_width,
#             image_height=image_height,
#             image_processor=image_processor,
#         )
#         return num_image_tokens

#     def get_num_video_tokens(
#         self,
#         *,
#         image_width: int,
#         image_height: int,
#         num_frames: int,
#         image_processor: Optional[Qwen2VLImageProcessor],
#     ) -> int:
#         _, num_video_tokens = self._get_vision_info(
#             image_width=image_width,
#             image_height=image_height,
#             num_frames=num_frames,
#             image_processor=image_processor,
#         )
#         return num_video_tokens

#     def get_image_size_with_most_features(self) -> ImageSize:
#         max_image_size, _ = self._get_vision_info(
#             image_width=9999999,
#             image_height=9999999,
#             image_processor=None,
#         )
#         return max_image_size

#     def get_max_image_tokens(self) -> int:
#         target_width, target_height = self.get_image_size_with_most_features()

#         return self.get_num_image_tokens(
#             image_width=target_width,
#             image_height=target_height,
#             image_processor=None,
#         )

#     def _get_max_video_frames(self, max_tokens: int) -> int:
#         target_width, target_height = self.get_image_size_with_most_features()

#         num_frames = 0

#         while True:
#             next_num_frames = num_frames + 1
#             next_max_tokens = self.get_num_video_tokens(
#                 image_width=target_width,
#                 image_height=target_height,
#                 num_frames=next_num_frames,
#                 image_processor=None,
#             )

#             if next_max_tokens > max_tokens:
#                 break

#             num_frames = next_num_frames

#         return num_frames

#     def get_num_frames_with_most_features(self, seq_len: int) -> int:
#         mm_config = self.ctx.get_mm_config()
#         max_images = mm_config.limit_per_prompt.get("image", 1)
#         max_videos = mm_config.limit_per_prompt.get("video", 1)

#         max_image_tokens = self.get_max_image_tokens() * max_images
#         max_total_frames = self._get_max_video_frames(seq_len -
#                                                       max_image_tokens)
#         num_frames = min(max(max_total_frames // max(max_videos, 1), 1),
#                          _MAX_FRAMES_PER_VIDEO)

#         # Temporary workaround for https://github.com/huggingface/transformers/issues/35412
#         if num_frames > 1 and num_frames % 2 == 1:
#             num_frames += 1

#         return num_frames

#     def get_max_video_tokens(self, seq_len: int) -> int:
#         target_width, target_height = self.get_image_size_with_most_features()

#         return self.get_num_video_tokens(
#             image_width=target_width,
#             image_height=target_height,
#             num_frames=self.get_num_frames_with_most_features(seq_len),
#             image_processor=None,
#         )


# class Qwen2VLDummyInputsBuilder(BaseDummyInputsBuilder[Qwen2VLProcessingInfo]):

#     def get_dummy_processor_inputs(
#         self,
#         seq_len: int,
#         mm_counts: Mapping[str, int],
#     ) -> ProcessorInputs:
#         num_images = mm_counts.get("image", 0)
#         num_videos = mm_counts.get("video", 0)

#         hf_processor = self.info.get_hf_processor()
#         image_token: str = hf_processor.image_token
#         video_token: str = hf_processor.video_token

#         target_width, target_height = \
#             self.info.get_image_size_with_most_features()
#         target_num_frames = \
#             self.info.get_num_frames_with_most_features(seq_len)

#         mm_data = {
#             "image":
#             self._get_dummy_images(width=target_width,
#                                    height=target_height,
#                                    num_images=num_images),
#             "video":
#             self._get_dummy_videos(
#                 width=target_width,
#                 height=target_height,
#                 num_frames=target_num_frames,
#                 num_videos=num_videos,
#             )
#         }

#         return ProcessorInputs(
#             prompt_text=image_token * num_images + video_token * num_videos,
#             mm_data=mm_data,
#         )


# class Qwen2VLMultiModalProcessor(BaseMultiModalProcessor[Qwen2VLProcessingInfo]
#                                  ):

#     def _get_data_parser(self) -> MultiModalDataParser:
#         return Qwen2VLMultiModalDataParser()

#     def _get_prompt_replacements(
#         self,
#         mm_items: MultiModalDataItems,
#         hf_processor_mm_kwargs: Mapping[str, Any],
#         out_mm_kwargs: MultiModalKwargs,
#     ) -> list[PromptReplacement]:
#         hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
#         image_processor = self.info.get_image_processor(
#             **hf_processor_mm_kwargs)
#         tokenizer = self.info.get_tokenizer()
#         vocab = tokenizer.get_vocab()

#         # NOTE: Only Qwen2VLProcessor in transformers 4.47.0 has
#         # image_token and video_token registered
#         placeholder = {
#             "image": vocab[hf_processor.image_token],
#             "video": vocab[hf_processor.video_token],
#         }

#         merge_length = image_processor.merge_size**2

#         def get_replacement_qwen2vl(item_idx: int, modality: str):
#             grid_thw = out_mm_kwargs[f"{modality}_grid_thw"][item_idx]
#             assert isinstance(grid_thw, torch.Tensor)

#             num_tokens = int(grid_thw.prod()) // merge_length
#             return [placeholder[modality]] * num_tokens

#         return [
#             PromptReplacement(
#                 modality=modality,
#                 target=[placeholder[modality]],
#                 replacement=partial(get_replacement_qwen2vl,
#                                     modality=modality),
#             ) for modality in ("image", "video")
#         ]

#     def _get_mm_fields_config(
#         self,
#         hf_inputs: BatchFeature,
#         hf_processor_mm_kwargs: Mapping[str, object],
#     ) -> Mapping[str, MultiModalFieldConfig]:
#         image_grid_thw = hf_inputs.get("image_grid_thw", torch.empty((0, 3)))
#         image_grid_sizes = image_grid_thw.prod(-1)

#         video_grid_thw = hf_inputs.get("video_grid_thw", torch.empty((0, 3)))
#         video_grid_sizes = video_grid_thw.prod(-1)

#         return dict(
#             pixel_values=MultiModalFieldConfig.flat_from_sizes(
#                 "image", image_grid_sizes),
#             image_embeds=MultiModalFieldConfig.flat_from_sizes(
#                 "image", image_grid_sizes),
#             image_grid_thw=MultiModalFieldConfig.batched("image"),
#             pixel_values_videos=MultiModalFieldConfig.flat_from_sizes(
#                 "video", video_grid_sizes),
#             video_embeds=MultiModalFieldConfig.flat_from_sizes(
#                 "video", video_grid_sizes),
#             video_grid_thw=MultiModalFieldConfig.batched("video"),
#         )


# @MULTIMODAL_REGISTRY.register_processor(Qwen2VLMultiModalProcessor,
#                                         info=Qwen2VLProcessingInfo,
#                                         dummy_inputs=Qwen2VLDummyInputsBuilder)
# class Qwen2VLForConditionalGeneration(nn.Module, SupportsMultiModal,
#                                       SupportsLoRA, SupportsPP):
#     packed_modules_mapping = {
#         "qkv_proj": [
#             "q_proj",
#             "k_proj",
#             "v_proj",
#         ],
#         "gate_up_proj": [
#             "gate_proj",
#             "up_proj",
#         ],
#     }

#     # LoRA specific attributes
#     supported_lora_modules = [
#         "qkv_proj",
#         "o_proj",
#         "gate_up_proj",
#         "down_proj",
#         # vision tower
#         "qkv",
#         "attn.proj",  # Distinguish patch_embed.proj
#         "fc1",
#         "fc2",
#         # projector
#         "mlp.0",
#         "mlp.2"
#     ]
#     embedding_modules = {}
#     embedding_padding_modules = []

#     # To ensure correct weight loading and mapping.
#     hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={
#         "lm_head.": "language_model.lm_head.",
#         "model.": "language_model.model.",
#     })

#     def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
#         super().__init__()
#         config: Qwen2VLConfig = vllm_config.model_config.hf_config
#         quant_config = vllm_config.quant_config
#         multimodal_config = vllm_config.model_config.multimodal_config

#         self.config = config
#         self.multimodal_config = multimodal_config

#         self.visual = Qwen2VisionTransformer(
#             config.vision_config,
#             norm_eps=getattr(config, "rms_norm_eps", 1e-6),
#             quant_config=self._maybe_ignore_quant_config(quant_config),
#             prefix=maybe_prefix(prefix, "visual"),
#         )

#         self.language_model = init_vllm_registered_model(
#             vllm_config=vllm_config,
#             prefix=maybe_prefix(prefix, "language_model"),
#             architectures=["Qwen2ForCausalLM"],
#         )

#         self.make_empty_intermediate_tensors = (
#             self.language_model.make_empty_intermediate_tensors)

#     @cached_property
#     def sampler(self):
#         if hasattr(self.language_model, "sampler"):
#             return self.language_model.sampler

#         return get_sampler()

#     def _maybe_ignore_quant_config(self, quant_config: QuantizationConfig):
#         # GPTQ configs do not have a list of ignored modules, however AutoGPTQ
#         # seems to avoid vision encoder sections for some models.
#         # See: https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4
#         if isinstance(quant_config, (GPTQConfig, GPTQMarlinConfig)):
#             return None
#         return quant_config

#     def _validate_and_reshape_mm_tensor(self, mm_input: object,
#                                         name: str) -> torch.Tensor:
#         if not isinstance(mm_input, (torch.Tensor, list)):
#             raise ValueError(f"Incorrect type of {name}. "
#                              f"Got type: {type(mm_input)}")
#         if isinstance(mm_input, torch.Tensor):
#             if mm_input.ndim == 2:
#                 return mm_input
#             if mm_input.ndim != 3:
#                 raise ValueError(f"{name} should be 2D or batched 3D tensor. "
#                                  f"Got ndim: {mm_input.ndim} "
#                                  f"(shape={mm_input.shape})")
#             return torch.concat(list(mm_input))
#         else:
#             return torch.concat(mm_input)

#     def _parse_and_validate_image_input(
#             self, **kwargs: object) -> Optional[Qwen2VLImageInputs]:
#         pixel_values = kwargs.pop("pixel_values", None)
#         image_embeds = kwargs.pop("image_embeds", None)
#         image_grid_thw = kwargs.pop("image_grid_thw", None)

#         if pixel_values is None and image_embeds is None:
#             return None

#         if pixel_values is not None:
#             pixel_values = self._validate_and_reshape_mm_tensor(
#                 pixel_values, "image pixel values")
#             image_grid_thw = self._validate_and_reshape_mm_tensor(
#                 image_grid_thw, "image grid_thw")

#             if not isinstance(pixel_values, (torch.Tensor, list)):
#                 raise ValueError("Incorrect type of image pixel values. "
#                                  f"Got type: {type(pixel_values)}")

#             return Qwen2VLImagePixelInputs(type="pixel_values",
#                                            pixel_values=pixel_values,
#                                            image_grid_thw=image_grid_thw)

#         if image_embeds is not None:
#             image_embeds = self._validate_and_reshape_mm_tensor(
#                 image_embeds, "image embeds")
#             image_grid_thw = self._validate_and_reshape_mm_tensor(
#                 image_grid_thw, "image grid_thw")

#             if not isinstance(image_embeds, torch.Tensor):
#                 raise ValueError("Incorrect type of image embeddings. "
#                                  f"Got type: {type(image_embeds)}")
#             return Qwen2VLImageEmbeddingInputs(type="image_embeds",
#                                                image_embeds=image_embeds,
#                                                image_grid_thw=image_grid_thw)

#     def _parse_and_validate_video_input(
#             self, **kwargs: object) -> Optional[Qwen2VLVideoInputs]:
#         pixel_values_videos = kwargs.pop("pixel_values_videos", None)
#         video_embeds = kwargs.pop("video_embeds", None)
#         video_grid_thw = kwargs.pop("video_grid_thw", None)

#         if pixel_values_videos is None and video_embeds is None:
#             return None

#         if pixel_values_videos is not None:
#             pixel_values_videos = self._validate_and_reshape_mm_tensor(
#                 pixel_values_videos, "video pixel values")
#             video_grid_thw = self._validate_and_reshape_mm_tensor(
#                 video_grid_thw, "video grid_thw")

#             return Qwen2VLVideoPixelInputs(
#                 type="pixel_values_videos",
#                 pixel_values_videos=pixel_values_videos,
#                 video_grid_thw=video_grid_thw,
#             )

#         if video_embeds is not None:
#             video_embeds = self._validate_and_reshape_mm_tensor(
#                 video_embeds, "video embeds")
#             video_grid_thw = self._validate_and_reshape_mm_tensor(
#                 video_grid_thw, "video grid_thw")

#             if not isinstance(video_embeds, torch.Tensor):
#                 raise ValueError("Incorrect type of video embeddings. "
#                                  f"Got type: {type(video_embeds)}")
#             return Qwen2VLVideoEmbeddingInputs(type="video_embeds",
#                                                video_embeds=video_embeds,
#                                                video_grid_thw=video_grid_thw)

#     def _process_image_input(
#             self, image_input: Qwen2VLImageInputs) -> tuple[torch.Tensor, ...]:

#         grid_thw = image_input["image_grid_thw"]
#         assert grid_thw.ndim == 2

#         if image_input["type"] == "image_embeds":
#             image_embeds = image_input["image_embeds"].type(self.visual.dtype)
#         else:
#             pixel_values = image_input["pixel_values"].type(self.visual.dtype)
#             image_embeds = self.visual(pixel_values, grid_thw=grid_thw)

#         # Split concatenated embeddings for each image item.
#         merge_size = self.visual.spatial_merge_size
#         sizes = grid_thw.prod(-1) // merge_size // merge_size

#         return image_embeds.split(sizes.tolist())

#     def _process_video_input(
#             self, video_input: Qwen2VLVideoInputs) -> tuple[torch.Tensor, ...]:

#         grid_thw = video_input["video_grid_thw"]
#         assert grid_thw.ndim == 2

#         if video_input["type"] == "video_embeds":
#             video_embeds = video_input["video_embeds"].type(self.visual.dtype)
#         else:
#             pixel_values_videos = video_input["pixel_values_videos"].type(
#                 self.visual.dtype)
#             video_embeds = self.visual(pixel_values_videos, grid_thw=grid_thw)

#         # Split concatenated embeddings for each video item.
#         merge_size = self.visual.spatial_merge_size
#         sizes = grid_thw.prod(-1) // merge_size // merge_size

#         return video_embeds.split(sizes.tolist())

#     def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
#         modalities = {}

#         # Preserve the order of modalities if there are multiple of them
#         # from the order of kwargs.
#         for input_key in kwargs:
#             if input_key in ("pixel_values",
#                              "image_embeds") and "images" not in modalities:
#                 modalities["images"] = self._parse_and_validate_image_input(
#                     **kwargs)
#             if input_key in ("pixel_values_videos",
#                              "video_embeds") and "videos" not in modalities:
#                 modalities["videos"] = self._parse_and_validate_video_input(
#                     **kwargs)

#         return modalities

#     def get_multimodal_embeddings(
#             self, **kwargs) -> Optional[tuple[torch.Tensor, ...]]:

#         modalities = self._parse_and_validate_multimodal_inputs(**kwargs)
#         if not modalities:
#             return None

#         # The result multimodal_embeddings is tuple of tensors, with each
#         # tensor correspoending to a multimodal data item (image or video).
#         multimodal_embeddings: tuple[torch.Tensor, ...] = ()

#         # NOTE: It is important to iterate over the keys in this dictionary
#         # to preserve the order of the modalities.
#         for modality in modalities:
#             if modality == "images":
#                 image_input = modalities["images"]
#                 vision_embeddings = self._process_image_input(image_input)
#                 multimodal_embeddings += vision_embeddings
#             if modality == "videos":
#                 video_input = modalities["videos"]
#                 video_embeddings = self._process_video_input(video_input)
#                 multimodal_embeddings += video_embeddings

#         return multimodal_embeddings

#     def get_input_embeddings(
#         self,
#         input_ids: torch.Tensor,
#         multimodal_embeddings: Optional[tuple[torch.Tensor, ...]] = None,
#     ) -> torch.Tensor:
#         inputs_embeds = self.language_model.get_input_embeddings(input_ids)
#         if multimodal_embeddings is not None:
#             inputs_embeds = merge_multimodal_embeddings(
#                 input_ids, inputs_embeds, multimodal_embeddings,
#                 [self.config.image_token_id, self.config.video_token_id])
#         return inputs_embeds

#     def get_input_embeddings_v0(
#         self,
#         input_ids: torch.Tensor,
#         image_input: Optional[tuple[torch.Tensor, ...]] = None,
#         video_input: Optional[tuple[torch.Tensor, ...]] = None,
#     ) -> torch.Tensor:

#         inputs_embeds = self.get_input_embeddings(input_ids)
#         if image_input is not None:
#             image_embeds = self._process_image_input(image_input)
#             inputs_embeds = merge_multimodal_embeddings(
#                 input_ids,
#                 inputs_embeds,
#                 image_embeds,
#                 placeholder_token_id=self.config.image_token_id,
#             )

#         if video_input is not None:
#             video_embeds = self._process_video_input(video_input)
#             inputs_embeds = merge_multimodal_embeddings(
#                 input_ids,
#                 inputs_embeds,
#                 video_embeds,
#                 placeholder_token_id=self.config.video_token_id,
#             )
#         return inputs_embeds

#     def forward(
#         self,
#         input_ids: torch.Tensor,
#         positions: torch.Tensor,
#         kv_caches: List[torch.Tensor],
#         attn_metadata: AttentionMetadata,
#         intermediate_tensors: Optional[IntermediateTensors] = None,
#         inputs_embeds: Optional[torch.Tensor] = None,
#         **kwargs: object,
#     ) -> Union[torch.Tensor, IntermediateTensors]:
#         """Run forward pass for Qwen2-VL.

#         Args:
#             input_ids: Flattened (concatenated) input_ids corresponding to a
#                 batch.
#             positions: Flattened (concatenated) position ids corresponding to a
#                 batch.
#                 **NOTE**: If mrope is enabled (default setting for Qwen2-VL
#                 opensource models), the shape will be `(3, seq_len)`,
#                 otherwise it will be `(seq_len,).
#             pixel_values: Pixel values to be fed to a model.
#                 `None` if no images are passed.
#             image_grid_thw: Tensor `(n_images, 3)` of image 3D grid in LLM.
#                 `None` if no images are passed.
#             pixel_values_videos: Pixel values of videos to be fed to a model.
#                 `None` if no videos are passed.
#             video_grid_thw: Tensor `(n_videos, 3)` of video 3D grid in LLM.
#                 `None` if no videos are passed.
#         """

#         if intermediate_tensors is not None:
#             inputs_embeds = None

#         # NOTE: In v1, inputs_embeds is always generated at model runner from
#         # `get_multimodal_embeddings` and `get_input_embeddings`, this
#         # condition is only for v0 compatibility.
#         elif inputs_embeds is None:
#             image_input = self._parse_and_validate_image_input(**kwargs)
#             video_input = self._parse_and_validate_video_input(**kwargs)

#             if image_input is None and video_input is None:
#                 inputs_embeds = None
#             else:
#                 if uses_mrope(self.config):
#                     assert positions.ndim == 2 and positions.size(0) == 3, (
#                         "multimodal section rotary embedding requires "
#                         f"(3, seq_len) positions, but got {positions.size()}")
#                 inputs_embeds = self.get_input_embeddings_v0(
#                     input_ids,
#                     image_input=image_input,
#                     video_input=video_input)
#                 input_ids = None

#         hidden_states = self.language_model.model(
#             input_ids=input_ids,
#             positions=positions,
#             kv_caches=kv_caches,
#             attn_metadata=attn_metadata,
#             intermediate_tensors=intermediate_tensors,
#             inputs_embeds=inputs_embeds,
#         )
#         return hidden_states

#     def compute_logits(
#         self,
#         hidden_states: torch.Tensor,
#         sampling_metadata: SamplingMetadata,
#     ) -> Optional[torch.Tensor]:
#         return self.language_model.compute_logits(hidden_states,
#                                                   sampling_metadata)

#     def sample(
#         self,
#         logits: torch.Tensor,
#         sampling_metadata: SamplingMetadata,
#     ) -> Optional[SamplerOutput]:
#         return self.language_model.sample(logits, sampling_metadata)

#     def load_weights(self, weights: Iterable[Tuple[str,
#                                                    torch.Tensor]]) -> Set[str]:

#         loader = AutoWeightsLoader(self)
#         return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

#     def get_mm_mapping(self) -> MultiModelKeys:
#         """
#         Get the module prefix in multimodal models
#         """
#         return MultiModelKeys.from_string_field(
#             language_model="language_model",
#             connector="visual.",
#             tower_model="visual.merger.")