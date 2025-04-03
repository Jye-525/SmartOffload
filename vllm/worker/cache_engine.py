"""CacheEngine class for managing the KV cache."""
from typing import List, Tuple
import math

import torch

from vllm.attention import get_attn_backend
from vllm.config import CacheConfig, DeviceConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, LayerBlockType,
                        get_dtype_size, is_pin_memory_available, GiB_bytes)

logger = init_logger(__name__)

class SecondaryGpuCache:
    """Secondary GPU cache for KV cache swapping memory."""
    def __init__(self, device:str, dtype: torch.dtype) -> None:
        self.allocated = {}
        self._device = device
        self._dtype = dtype
        self.current_device = torch.cuda.current_device()
        logger.debug(f"+++++++++++++Secondary GPU cache is created on device {self._device}. Current device is {self.current_device}.")
        
        
    def can_allocate(self, layer_id: int, kv_cache_shape: Tuple[int, ...]) -> bool:
        # check if the current device has enough space to allocate the required space
        required_kv_cache_size = math.prod(kv_cache_shape) * get_dtype_size(self._dtype)
        logger.debug(f"++++++++++++Layer {layer_id} required {required_kv_cache_size/GiB_bytes} GB on secondary GPU cache")
        # Step 1: get the free memory in the current device
        free_mem = (
            torch.cuda.memory_reserved()  # Total memory reserved by PyTorch
            - torch.cuda.memory_allocated()  # Memory actually in use
        )
        free_gpu_mem, total_gpu_memory = torch.cuda.mem_get_info()
        logger.debug(f"+++++++++++++Secondary GPU cache free memory: free_mem_1={free_mem/GiB_bytes} GB, free_mem_2={free_gpu_mem/GiB_bytes} GB, total_gpu_memory={total_gpu_memory/GiB_bytes} GB")

        # Step 2: check if the free memory is greater than the required size
        #  --- we may need a threshold to avoid all the memory being used
        if free_gpu_mem >= required_kv_cache_size:
            return True
        else:
            logger.debug("Not enough memory to allocate secondary GPU cache")
            return False

    def allocate(self, 
            layer_id: int, 
            kv_cache_shape: Tuple[int, ...], 
            org_src_to_dst: torch.Tensor):
        # Step 1: call torch.zeros to allocate the required space
        target_gpu_cache=torch.zeros(kv_cache_shape,
                dtype=self._dtype,
                device=self._device)
        # Step 2: create a mapping of the orignal dst to the new dst
        block_table = self.allocated.setdefault(layer_id, {})
        for i in range(kv_cache_shape[1]):
            block_table[org_src_to_dst[i][1]] = {i:target_gpu_cache}
            
            
    def swap_out(self, layer_id: int, gpu_src: torch.Tensor, org_src_to_dst: torch.Tensor) -> None:
        block_table = self.allocated.get(layer_id, None)
        assert block_table is not None, "No block table found for the layer"
        
        for i in range(len(org_src_to_dst)):
            # Step 1: get the original source and destination block Ids
            org_src_block_id, org_dst_block_id = org_src_to_dst[i]
            # Step 2: Extract the block of org_src_block_id from the gpu_src: key and value
            src_key_block, src_value_block = gpu_src[0, org_src_block_id], gpu_src[1, org_src_block_id]
            # Step 4: Extract the block of dst key and value buffer from the corresponding secondary gpu buffer  
            dst_key_block, dst_value_block = block_table[org_dst_block_id][i][0, i], block_table[org_dst_block_id][i][1, i]
            # Step 3: Copy the data from the source block to the destination block
            dst_key_block.copy_(src_key_block)
            dst_value_block.copy_(src_value_block)
            
    def is_cached(self, layer_id: int, org_src_to_dst: torch.Tensor) -> bool:
        # Check if the blocks_to_swap_in are cached on secondary GPU cache.
        # Step 1: Get the block table for the layer with layer_id
        # If the block table is None, it means that the layer is not cached
        block_table = self.allocated.get(layer_id, None)
        if block_table is None:
            return False
        # Step 2: check if org_src block id is in the block_table of the layer with layer_id
        # In our case, either all the blocks are cached or none of them are cached.
        # (To Do) For swap-in, may be partial on secondary GPU cache, and partial on GPU cache 
        for i in range(len(org_src_to_dst)):
            org_src_block_id, _ = org_src_to_dst[i]
            if org_src_block_id not in block_table:
                return False
        
        return True
    
    def swap_in(self, layer_id: int, gpu_dst: torch.Tensor, org_src_to_dst: torch.Tensor) -> None:
        # Step 1: check if the blocks_to_swap_in are cached on secondary GPU cache.
        block_table = self.allocated.get(layer_id, None)
        assert block_table is not None, "No block table found for the layer"
        
        for i in range(len(org_src_to_dst)): 
            # Step 2: get the original source and destination block Ids
            org_src_block_id, org_dst_block_id = org_src_to_dst[i]
            # Step 3: Extract the block of org_dst_block_id from the gpu_dst: key and value
            dst_key_block, dst_value_block = gpu_dst[0, org_dst_block_id], gpu_dst[1, org_dst_block_id]
            # Step 4: Extract the block of src key and value buffer from corresponding secondary gpu buffer
            src_key_block, src_value_block = block_table[org_src_block_id][i][0, i], block_table[org_src_block_id][i][1, i]
            # Step 4: Copy the data from the src block to the dst block
            dst_key_block.copy_(src_key_block)
            dst_value_block.copy_(src_value_block)
            

class CacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        device_config: DeviceConfig,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.device_config = device_config

        self.head_size = model_config.get_head_size()
        # Models like Jamba, have mixed typed layers, E.g Mamba
        self.num_attention_layers = model_config.get_num_layers_by_block_type(
            parallel_config, LayerBlockType.attention)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        if self.num_gpu_blocks:
            self.num_gpu_blocks //= parallel_config.pipeline_parallel_size
        self.num_cpu_blocks = cache_config.num_cpu_blocks
        if self.num_cpu_blocks:
            self.num_cpu_blocks //= parallel_config.pipeline_parallel_size

        if cache_config.cache_dtype == "auto":
            self.dtype = model_config.dtype
        else:
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        # Get attention backend.
        self.attn_backend = get_attn_backend(self.head_size,
                                             model_config.dtype,
                                             cache_config.cache_dtype,
                                             self.block_size,
                                             model_config.is_attention_free)

        # Initialize the cache.
        self.gpu_cache = self._allocate_kv_cache(
            self.num_gpu_blocks, self.device_config.device_type)
        self.cpu_cache = self._allocate_kv_cache(self.num_cpu_blocks, "cpu")
        self.secondary_gpu_cache = SecondaryGpuCache(self.device_config.device_type, self.dtype)

    def _allocate_kv_cache(
        self,
        num_blocks: int,
        device: str,
    ) -> List[torch.Tensor]:
        """Allocates KV cache on the specified device."""
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_kv_heads, self.head_size)
        pin_memory = is_pin_memory_available() if device == "cpu" else False
        kv_cache: List[torch.Tensor] = []
        for _ in range(self.num_attention_layers):
            # null block in CpuGpuBlockAllocator requires at least that
            # block to be zeroed-out.
            # We zero-out everything for simplicity.
            kv_cache.append(
                torch.zeros(kv_cache_shape,
                            dtype=self.dtype,
                            pin_memory=pin_memory,
                            device=device))
        return kv_cache

    def swap_in(self, src_to_dst: torch.Tensor) -> None:
        logger.debug(f"+++++++++++++++swap_in is triggered. num_blocks on each layer {len(src_to_dst)}")
        for i in range(self.num_attention_layers):
            if self.secondary_gpu_cache.is_cached(i, src_to_dst):
                logger.debug(f"+++++++++++++++swap_in from secondary gpu cache++++++++++++++")
                self.secondary_gpu_cache.swap_in(i, self.gpu_cache[i], src_to_dst)
            else:
                logger.debug(f"+++++++++++++++swap_in from cpu cache++++++++++++++")
                self.attn_backend.swap_blocks(self.cpu_cache[i], self.gpu_cache[i],
                                            src_to_dst)

    def swap_out(self, src_to_dst: torch.Tensor) -> None:
        logger.debug(f"+++++++++++++++swap_out is triggered. num_blocks on each layer {len(src_to_dst)}")
        for i in range(self.num_attention_layers):
            num_blocks_to_swap = len(src_to_dst)
            kv_cache_shape = self.attn_backend.get_kv_cache_shape(
                num_blocks_to_swap, self.block_size, self.num_kv_heads, self.head_size)
            if self.secondary_gpu_cache.can_allocate(kv_cache_shape, self.dtype):
                logger.debug(f"+++++++++++++++swap_out from gpu cache to secondary gpu cache++++++++++++++")
                self.secondary_gpu_cache.allocate(i, kv_cache_shape, self.dtype, src_to_dst)
                self.secondary_gpu_cache.swap_out(i, self.gpu_cache[i], src_to_dst)
            else:
                logger.debug(f"+++++++++++++++swap_out from gpu cache to cpu cache++++++++++++++")
                self.attn_backend.swap_blocks(self.gpu_cache[i], self.cpu_cache[i],
                                            src_to_dst)

    def copy(self, src_to_dsts: torch.Tensor) -> None:
        self.attn_backend.copy_blocks(self.gpu_cache, src_to_dsts)

    @staticmethod
    def get_cache_block_size(
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_attention_layers = model_config.get_num_layers_by_block_type(
            parallel_config, LayerBlockType.attention)

        key_cache_block = cache_config.block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_attention_layers * (key_cache_block + value_cache_block)
        if cache_config.cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        dtype_size = get_dtype_size(dtype)
        return dtype_size * total
