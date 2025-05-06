# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
"""Inference-only LLaMA model compatible with HuggingFace weights."""
from typing import Any, Dict, Iterable, Optional, Set, Tuple, Union

import time, math

import torch
from torch import nn
from transformers import LlamaConfig

from vllm.attention import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank, get_world_group
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.offload_buffer import OffloadBuffer
from vllm.forward_context import get_forward_context
from vllm.sequence import IntermediateTensors
from vllm.spec_decode.util import nvtx_range
from vllm.utils import GiB_bytes
from .interfaces import SupportsLoRA, SupportsPP
from .utils import (AutoWeightsLoader, PPMissingLayer, extract_layer_index,
                    is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix, smartoffload_make_layers)

logger = init_logger(__name__)

class LlamaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        prefix: str = "",
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class LlamaAttention(nn.Module):

    def __init__(self,
                 config: LlamaConfig,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 rope_theta: float = 10000,
                 rope_scaling: Optional[Dict[str, Any]] = None,
                 max_position_embeddings: int = 8192,
                 quant_config: Optional[QuantizationConfig] = None,
                 bias: bool = False,
                 bias_o_proj: bool = False,
                 cache_config: Optional[CacheConfig] = None,
                 prefix: str = "") -> None:
        super().__init__()
        layer_idx = extract_layer_index(prefix)
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
        self.head_dim = getattr(config, "head_dim",
                                self.hidden_size // self.total_num_heads)
        # Phi models introduced a partial_rotary_factor parameter in the config
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1)
        self.rotary_dim = int(partial_rotary_factor * self.head_dim)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias_o_proj,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        is_neox_style = True
        is_gguf = quant_config and quant_config.get_name() == "gguf"
        if is_gguf and config.model_type == "llama":
            is_neox_style = False

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.rotary_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=is_neox_style,
        )

        if hasattr(config, "interleaved_sliding_window"):
            interleaved_sliding_window = config.interleaved_sliding_window
            if isinstance(interleaved_sliding_window, int):
                sliding_window = interleaved_sliding_window
            elif isinstance(interleaved_sliding_window, list):
                sw_idx = layer_idx % len(interleaved_sliding_window)
                sliding_window = interleaved_sliding_window[sw_idx]
            else:
                raise ValueError(
                    f"{type(interleaved_sliding_window)} is not supported.")
        else:
            sliding_window = None

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class LlamaDecoderLayer(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = extract_layer_index(prefix)
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
                config, "original_max_position_embeddings", None):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        # Support abacusai/Smaug-72B-v0.1 with attention_bias
        # Support internlm/internlm-7b with bias
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False)
        bias_o_proj = attention_bias
        # support internlm/internlm3-8b with qkv_bias
        if hasattr(config, 'qkv_bias'):
            attention_bias = config.qkv_bias

        self.self_attn = LlamaAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads",
                                 config.num_attention_heads),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            bias_o_proj=bias_o_proj,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=getattr(config, "mlp_bias", False),
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    @nvtx_range("LlamaDecoderLayer.forward")
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(positions=positions,
                                       hidden_states=hidden_states)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile
class LlamaModel(nn.Module):

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = "",
                 layer_type: type[nn.Module] = LlamaDecoderLayer):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        
        cpu_offload_config = cache_config.cpu_offload_config
        self.smart_offload = False
        self.smart_offload_dynamic = cpu_offload_config.smart_offload_dynamic
        self.smart_offload_per_block_time = -1 # assume 1 milliseconds
        # self.smart_offload_pci_bandwidth = 23 # GB/s
        self.smart_offload_init_tokens = 0
        self.timing_layer_fwd = vllm_config.collect_layer_fwd_time

        self.config = config
        self.quant_config = quant_config
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size
        if get_pp_group().is_first_rank or (config.tie_word_embeddings
                                            and get_pp_group().is_last_rank):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                quant_config=quant_config,
            )
        else:
            self.embed_tokens = PPMissingLayer()
        
        self.offload_buffer = None
        if cpu_offload_config.method == "default":
            self.smart_offload = False
            self.start_layer, self.end_layer, self.layers = make_layers(
                config.num_hidden_layers,
                lambda prefix: layer_type(config=config,
                                        cache_config=cache_config,
                                        quant_config=quant_config,
                                        prefix=prefix),
                prefix=f"{prefix}.layers",
            )
            self.smart_offload_per_block_size =  self.get_per_decode_layer_bytes() / GiB_bytes # GB
        else:
            self.smart_offload = True
            # create offload buffer
            self.offload_buffer = OffloadBuffer(cpu_offload_config.smart_offload_interval,
                                                cpu_offload_config.param_offload_target)
            self.start_layer, self.end_layer, self.layers = smartoffload_make_layers(
                config.num_hidden_layers,
                lambda prefix: layer_type(config=config,
                                        cache_config=cache_config,
                                        quant_config=quant_config,
                                        prefix=prefix),
                prefix=f"{prefix}.layers",
                offload_fn=self.offload_buffer.create_module,
            )
            self.smart_offload_per_block_size =  self.offload_buffer.get_per_layer_params_Bytes() / GiB_bytes # GB
            
            
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

        self.avg_H2D_transfer_time = 0
        self.init_cpu_time = 0
        self.init_cuda_event = torch.cuda.Event(enable_timing=True)
        self.fwd_counts = 0
         
        logger.info(f"PP Rank {get_pp_group().rank_in_group}/{get_pp_group().ranks} "
                    f"TP Rank {get_tensor_model_parallel_rank()} include {len(self.layers)} layers,"
                    f" start layer: {self.start_layer} end layer: {self.end_layer}, per decode layer size: {self.smart_offload_per_block_size} GB")
    
    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)
    
    def get_per_decode_layer_bytes(self) -> int:
        module = self.layers[self.start_layer]
        
        total_bytes = 0
        for param in module.parameters():
            total_bytes += param.numel() * param.element_size()

        return total_bytes

    @nvtx_range("LlamaModel.forward")
    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        is_dummy_run = True if get_forward_context().attn_metadata is None else False
        if is_dummy_run == False and self.timing_layer_fwd:
            # skip the first fwd_counts since it is the profiling run
            layer_start_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.start_layer, self.end_layer)]
            layer_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.start_layer, self.end_layer)]
        
        if get_pp_group().is_first_rank:
            cur_timestamp = time.time_ns()
            relative_cpu_start_time = cur_timestamp - self.init_cpu_time
            
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
            if is_dummy_run == False and self.timing_layer_fwd: 
                logger.info(f"[abs_timestamp(ns): {cur_timestamp}] [relative_start(ns): {relative_cpu_start_time:.3f}]"
                            f" PP Rank {get_pp_group().rank_in_group} TP Rank {get_tensor_model_parallel_rank()} start forward on the model, fwd_counts: {self.fwd_counts}"
                            f" num_input_tokens: {hidden_states.shape[0]}, is_dummy_run: {is_dummy_run}")   
            
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        input_tokens = hidden_states.shape[0]
        # trigger reorganization of the offload buffer
        if (is_dummy_run == False and self.smart_offload and self.smart_offload_dynamic and \
            self.smart_offload_per_block_time != -1):
            cur_per_block_cmp_time = self.smart_offload_per_block_time * input_tokens / self.smart_offload_init_tokens # ms
            k = math.ceil(self.avg_H2D_transfer_time / cur_per_block_cmp_time)
            k = int(min(max(1, k), (self.end_layer - self.start_layer) / 2)) # because we have a duble buffer, so use (self.end_layer - self.start_layer) / 2 
            t_start = time.time_ns()
            self.offload_buffer.reorganize_resident_gpu_modules(k)
            t_end = time.time_ns()
            logger.info(f"PP Rank {get_pp_group().rank_in_group} TP Rank {get_tensor_model_parallel_rank()} fwd_counts: {self.fwd_counts} reorganize the modules, k = {k}, reorganize cost {((t_end - t_start) / 1e6):.3f} ms "
                         f"per_block_cmp_time: {cur_per_block_cmp_time} ms (init_per_block_time = {self.smart_offload_per_block_time}, input_tokens = {input_tokens} ), init_tokens = {self.smart_offload_init_tokens}), "
                         f"avg_H2D_transfer_time: {self.avg_H2D_transfer_time} ms")
        
        
        for layer in self.layers[self.start_layer:self.end_layer]:
            layer_idx = layer.layer_idx
            # preprocessing for the layer
            if is_dummy_run == False and self.timing_layer_fwd:
                # record the beginning of the layer forward
                layer_start_events[layer_idx - self.start_layer].record()
                
            hidden_states, residual = layer(positions, hidden_states, residual)
            # postprocessing for the layer
            if is_dummy_run == False and self.timing_layer_fwd:
                layer_end_events[layer_idx - self.start_layer].record()
            
            if (is_dummy_run == False and self.offload_buffer is not None \
                and (layer_idx + 1) % self.offload_buffer.get_offload_interval() == 0):
                # record the the computation event for this layer
                self.offload_buffer.compute_event.record()

        if not get_pp_group().is_last_rank:
            hidden_states = IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        else:
            hidden_states, _ = self.norm(hidden_states, residual)

        ## Print the time for each layer
        if is_dummy_run == False and self.timing_layer_fwd:
            layer_end_events[-1].synchronize()
            total_layer_fwd_time = 0 
            for idx in range(0, self.end_layer - self.start_layer):
                relative_start_time = self.init_cuda_event.elapsed_time(layer_start_events[idx])
                relative_end_time = self.init_cuda_event.elapsed_time(layer_end_events[idx])
                layer_fwd_time = layer_start_events[idx].elapsed_time(layer_end_events[idx])
                total_layer_fwd_time += layer_fwd_time
                logger.info(f"PP Rank {get_pp_group().rank_in_group} TP Rank {get_tensor_model_parallel_rank()} "
                            f"Layer {self.start_layer + idx} relative timing info, layer_relative_start: {relative_start_time:.3f} layer_relative_end: {relative_end_time:.3f} "
                            f"layer_fwd_time: {layer_fwd_time:.3f} ms, " 
                            f"fwd_counts: {self.fwd_counts} num_input_tokens: {input_tokens}")
        
            # update the compute time of per decode layer
            if self.smart_offload and self.smart_offload_dynamic and self.smart_offload_per_block_time == -1:
                self.smart_offload_per_block_time = total_layer_fwd_time / (self.end_layer - self.start_layer)
                self.smart_offload_init_tokens = input_tokens
                logger.info(f"+++++smart_offload_per_block average compute time: {self.smart_offload_per_block_time} fwd_counts: {self.fwd_counts} num_input_tokens: {input_tokens} ")
        
        
        if is_dummy_run == False and get_pp_group().is_last_rank and self.timing_layer_fwd:
            cur_timestamp = time.time_ns()
            relative_cpu_end_time = cur_timestamp - self.init_cpu_time
            # print the timestamp of the last rank (GPU time)  --> calculate the total time (lastrank - first ranjk, from log information)
            logger.info(f"[abs_timestamp(ns): {cur_timestamp}] [relative_end(ns): {relative_cpu_end_time:.3f}]"
                        f" PP Rank {get_pp_group().rank_in_group} TP Rank {get_tensor_model_parallel_rank()} stop forward on the model. fwd_counts: {self.fwd_counts} num_input_tokens: {input_tokens}")
        
        if is_dummy_run == False:
            self.fwd_counts += 1
            
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if (self.quant_config is not None and
                (scale_name := self.quant_config.get_cache_scale(name))):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = (loaded_weight if loaded_weight.dim() == 0 else
                                 loaded_weight[0])
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            if "scale" in name:
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
    
    
    def maybe_offload(self):
        if self.offload_buffer is not None:
           # using the smart offloading strategy to offload the model weights to CPU 
            for i in range(self.start_layer, self.end_layer):
                self.layers[i] = self.offload_buffer.maybe_offload(self.layers[i], i, 
                                                                    self.start_layer, self.end_layer)
            
            # self.avg_H2D_transfer_time = self.offload_buffer.get_avg_H2D_transfer_time()
            H2D_PCIe_bw = 23 # GB/s
            self.avg_H2D_transfer_time = (self.smart_offload_per_block_size / H2D_PCIe_bw ) * 1000 # ms
        else:
            H2D_PCIe_bw = 23 # GB/s
            self.avg_H2D_transfer_time = (self.smart_offload_per_block_size / H2D_PCIe_bw ) * 1000 # ms
            
        # logger.debug(f"+++++avg_H2D_transfer_time: {self.avg_H2D_transfer_time} on PP Rank {get_pp_group().rank_in_group} TP Rank {get_tensor_model_parallel_rank()}")
              
    def record_init_timestamp(self):
        self.init_cpu_time = time.time_ns()
        self.init_cuda_event.record()


class LlamaForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"]
    }

    # LoRA specific attributes
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings"
    }
    embedding_padding_modules = ["lm_head"]

    # Mistral/Llama models can also be loaded with --load-format mistral
    # from consolidated.safetensors checkpoints
    mistral_mapping = {
        "layers": "model.layers",
        "attention": "self_attn",
        "qscale_act": "input_scale",
        "qscale_weight": "weight_scale",
        "kv_fake_quantizer.qscale_act": "kv_scale",
        "wq": "q_proj",
        "wk": "k_proj",
        "wv": "v_proj",
        "wo": "o_proj",
        "attention_norm": "input_layernorm",
        "feed_forward": "mlp",
        "w1": "gate_proj",
        "w2": "down_proj",
        "w3": "up_proj",
        "ffn_norm": "post_attention_layernorm",
        "tok_embeddings": "model.embed_tokens",
        "output": "lm_head",
        "norm": "model.norm",
    }

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = "",
                 layer_type: type[nn.Module] = LlamaDecoderLayer):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        self.config = config
        self.lora_config = lora_config

        self.model = self._init_model(vllm_config=vllm_config,
                                      prefix=maybe_prefix(prefix, "model"),
                                      layer_type=layer_type)

        if get_pp_group().is_last_rank:
            self.unpadded_vocab_size = config.vocab_size
            if lora_config:
                self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                padding_size=(
                    DEFAULT_VOCAB_PADDING_SIZE
                    # We need bigger padding if using lora for kernel
                    # compatibility
                    if not lora_config else
                    lora_config.lora_vocab_padding_size),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(
                    self.model.embed_tokens)

            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                    config.vocab_size,
                                                    logit_scale)
        else:
            self.lm_head = PPMissingLayer()

        self.sampler = get_sampler()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)
        
        ## Used for SmartOffload relative time
        # Step 1: call torch.distribute.barrier() to synchronize all ranks
        current_device = torch.cuda.current_device()
        torch.cuda.synchronize() # in case of some GPUs has unfinished tasks.
        get_world_group().barrier()
        # Step 2: call self.model to record the initial timestamp
        self.model.record_init_timestamp()
        logger.info(f"PP Rank {get_pp_group().rank_in_group}/{get_pp_group().rank} "
                     f"TP Rank {get_tensor_model_parallel_rank()} "
                     f"WORLD RANK {get_world_group().local_rank}/{get_world_group().rank} the gpu device is {current_device}")

    def _init_model(self,
                    vllm_config: VllmConfig,
                    prefix: str = "",
                    layer_type: type[nn.Module] = LlamaDecoderLayer):
        return LlamaModel(vllm_config=vllm_config,
                          prefix=prefix,
                          layer_type=layer_type)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    @nvtx_range("LlamaForCausalLM.forward")
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        model_output = self.model(input_ids, positions, intermediate_tensors,
                                  inputs_embeds)
        return model_output

    @nvtx_range("LlamaForCausalLM.compute_logits")
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    @nvtx_range("LlamaForCausalLM.sample")
    def sample(self, logits: torch.Tensor,
               sampling_metadata: SamplingMetadata) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        
        loaded_weights = loader.load_weights(
            self.maybe_remap_mistral(name, loaded_weight)
            for name, loaded_weight in weights)
        
        self.model.maybe_offload()
        
        return loaded_weights

    # This function is used to remap the mistral format as
    # used by Mistral and Llama <=2
    def maybe_remap_mistral(
        self,
        name: str,
        loaded_weight: torch.Tensor,
    ) -> Tuple[str, torch.Tensor]:

        def permute(w: torch.Tensor, n_heads: int):
            attn_in = self.config.head_dim * n_heads
            attn_out = self.config.hidden_size

            return w.view(n_heads, attn_in // n_heads // 2, 2,
                          attn_out).transpose(1, 2).reshape(attn_in, attn_out)

        mapping = self.mistral_mapping
        modules = name.split(".")

        # rotary embeds should be sliced
        if "wk" in modules and modules[-1] == "weight":
            loaded_weight = permute(loaded_weight,
                                    self.config.num_key_value_heads)
        elif "wq" in modules and modules[-1] == "weight":
            loaded_weight = permute(loaded_weight,
                                    self.config.num_attention_heads)

        num_modules = len(modules)
        for i in range(num_modules):
            item = modules[i]
            next_item = modules[i + 1] if i < num_modules - 1 else None

            combined_item = (f"{item}.{next_item}"
                             if next_item is not None else None)

            if combined_item in mapping:
                name = name.replace(combined_item, mapping[combined_item])
            elif item in mapping and mapping[item] not in name:
                name = name.replace(item, mapping[item])

        return name, loaded_weight
