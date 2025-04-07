"""CacheEngine class for managing the KV cache."""
from typing import List, Tuple
import math, time

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
        # logger.debug(f"+++++++++++++Secondary GPU cache is created on device {self._device}. Current device is {self.current_device}.")
        
        
    def can_allocate(self, layer_id: int, kv_cache_shape: Tuple[int, ...]) -> bool:
        # check if the current device has enough space to allocate the required space
        t_start = time.time_ns()
        required_kv_cache_size = math.prod(kv_cache_shape) * get_dtype_size(self._dtype)
        t_end_1 = time.time_ns()
        # Step 1: get the free memory in the current device
        free_gpu_mem1 = (
            torch.cuda.memory_reserved()  # Total memory reserved by PyTorch
            - torch.cuda.memory_allocated()  # Memory actually in use
        )
        free_gpu_mem2, total_gpu_memory = torch.cuda.mem_get_info()
        total_gpu_free_mem = free_gpu_mem1 + free_gpu_mem2 # Total free memory on the device
        t_end_2 = time.time_ns()
        
        # Step 2: check if the free memory is greater than the required size
        #  --- we may need a threshold to avoid all the memory being used
        if total_gpu_free_mem >= required_kv_cache_size:
            logger.debug(f"+++++Layer {layer_id} required {total_gpu_free_mem/GiB_bytes:.3f} GB on the secondary GPU cache for kv cache {kv_cache_shape}. "
                         f"It takes {(t_end_2 - t_start)/1e6:.3f} ms to calculate the size ({(t_end_1 - t_start)/1e6:.3f} ms) and free memory ({(t_end_2 - t_end_1)/1e6:.3f}).")
            return True
        else:
            logger.debug("Not enough memory to allocate secondary GPU cache")
            return False

    def allocate(self, 
            layer_id: int, 
            kv_cache_shape: Tuple[int, ...], 
            org_src_to_dst: torch.Tensor):
        t_start = time.time_ns()
        # Step 1: call torch.zeros to allocate the required space
        target_gpu_cache=torch.zeros(kv_cache_shape,
                dtype=self._dtype,
                device=self._device)
        t_end_1 = time.time_ns()
        # Step 2: create a mapping of the orignal dst to the new dst
        block_table = self.allocated.setdefault(layer_id, {})
        for i in range(kv_cache_shape[1]):
            block_table[org_src_to_dst[i][1].item()] = {i:target_gpu_cache}
        t_end_2 = time.time_ns()
        logger.debug(f"+++++++++++++Layer {layer_id} allocate memory take {(t_end_1 - t_start)/1e6:.3f} ms, "
                     f"build block_table for {len(org_src_to_dst)} blocks take {(t_end_2 - t_end_1)/1e6:.3f} ms, "
                     f"total cost {(t_end_2 - t_start)/1e6:.3f} ms.")
            
            
    def swap_out(self, layer_id: int, gpu_src: torch.Tensor, org_src_to_dst: torch.Tensor) -> None:
        t_start = time.time_ns() 
        t_gpu_start = torch.cuda.Event(enable_timing=True)
        t_gpu_end = torch.cuda.Event(enable_timing=True)
        t_gpu_start.record()
        block_table = self.allocated.get(layer_id, None)
        assert block_table is not None, "No block table found for the layer"
        
        num_blocks_to_swap = len(org_src_to_dst)
        for i in range(num_blocks_to_swap):
            # Step 1: get the original source and destination block Ids
            org_src_block_id, org_dst_block_id = org_src_to_dst[i][0].item(), org_src_to_dst[i][1].item()
            # Step 2: Extract the block of org_src_block_id from the gpu_src: key and value
            src_key_block, src_value_block = gpu_src[0, org_src_block_id], gpu_src[1, org_src_block_id]
            # Step 4: Extract the block of dst key and value buffer from the corresponding secondary gpu buffer  
            dst_key_block, dst_value_block = block_table[org_dst_block_id][i][0, i], block_table[org_dst_block_id][i][1, i]
            # Step 3: Copy the data from the source block to the destination block
            dst_key_block.copy_(src_key_block)
            dst_value_block.copy_(src_value_block)
        t_gpu_end.record()
        torch.cuda.synchronize()
        t_end_1 = time.time_ns()
        logger.debug(f"+++++++++++++Layer {layer_id} swap_out {num_blocks_to_swap} blocks to secondary gpu cache take cpu_time {(t_end_1 - t_start)/1e6:.3f} ms, "
                     f"gpu_time {(t_gpu_start.elapsed_time(t_gpu_end)):.3f} ms.")
            
    def is_cached(self, layer_id: int, org_src_to_dst: torch.Tensor) -> bool:
        # Step 1: Get the block table for the layer with layer_id
        # If the block table is None, it means that the layer is not cached
        t_start = time.time_ns()
        is_cached = False
        block_table = self.allocated.get(layer_id, None)
        if block_table is not None:
            # Step 2: check if org_src block id is in the block_table of the layer with layer_id
            # In our case, either all the blocks are cached or none of them are cached.
            # (To Do) For swap-in, may be partial on secondary GPU cache, and partial on GPU cache 
            is_cached = True
            for i in range(len(org_src_to_dst)):
                org_src_block_id = org_src_to_dst[i][0].item()
                if org_src_block_id not in block_table:
                    is_cached = False
                    break
        t_end = time.time_ns()
        if layer_id == 0:
            logger.debug(f"+++++++++++++Layer {layer_id} check if cached {len(org_src_to_dst)} blocks take {(t_end - t_start)/1e6:.3f} ms.")    
        return is_cached
    
    def swap_in(self, layer_id: int, gpu_dst: torch.Tensor, org_src_to_dst: torch.Tensor) -> None:
        # Step 1: check if the blocks_to_swap_in are cached on secondary GPU cache.
        t_start = time.time_ns()
        t_gpu_start = torch.cuda.Event(enable_timing=True)
        t_gpu_end = torch.cuda.Event(enable_timing=True)
        t_gpu_start.record()
        block_table = self.allocated.get(layer_id, None)
        assert block_table is not None, "No block table found for the layer"
        
        for i in range(len(org_src_to_dst)): 
            # Step 2: get the original source and destination block Ids
            org_src_bid, org_dst_bid = org_src_to_dst[i][0].item(), org_src_to_dst[i][1].item()
            # Step 3: Extract the block of org_dst_block_id from the gpu_dst: key and value
            dst_key_block, dst_value_block = gpu_dst[0, org_dst_bid], gpu_dst[1, org_dst_bid]
            # Step 4: Extract the block of src key and value buffer from corresponding secondary gpu buffer
            snd_gpu_cache_bid = list(block_table[org_src_bid].keys())[0]
            src_key_block, src_value_block = block_table[org_src_bid][snd_gpu_cache_bid][0, snd_gpu_cache_bid], block_table[org_src_bid][snd_gpu_cache_bid][1, snd_gpu_cache_bid]
            # Step 4: Copy the data from the src block to the dst block
            dst_key_block.copy_(src_key_block)
            dst_value_block.copy_(src_value_block)
        t_gpu_end.record()
        torch.cuda.synchronize()
        t_end = time.time_ns()
        logger.debug(f"+++++++++++++Layer {layer_id} swap_in {len(org_src_to_dst)} blocks from secondary gpu cache take cpu_time {(t_end - t_start)/1e6:.3f} ms, "
                     f"gpu_time {(t_gpu_start.elapsed_time(t_gpu_end)):.3f} ms.")
            

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
        self.enable_secondary_gpu_cache = True
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
        if self.enable_secondary_gpu_cache:
            start_time = time.time_ns()
            t_gpu_start = torch.cuda.Event(enable_timing=True)
            t_gpu_end = torch.cuda.Event(enable_timing=True)
            t_gpu_start.record()
            num_blocks_to_swap = len(src_to_dst) 
            for i in range(self.num_attention_layers):
                if self.secondary_gpu_cache.is_cached(i, src_to_dst):
                    self.secondary_gpu_cache.swap_in(i, self.gpu_cache[i], src_to_dst)
                else:
                    self.attn_backend.swap_blocks(self.cpu_cache[i], self.gpu_cache[i],
                                                src_to_dst)
            t_gpu_end.record()
            torch.cuda.synchronize()
            end_time = time.time_ns()
            logger.debug(f"+++++++++++++++swap_in {num_blocks_to_swap} from secondary gpu cache took cpu_time {(end_time - start_time)/1e6:.3f} ms, gpu_time {(t_gpu_start.elapsed_time(t_gpu_end)):.3f} ms.")
        else:
            start_time = time.time_ns()
            t_gpu_start = torch.cuda.Event(enable_timing=True)
            t_gpu_end = torch.cuda.Event(enable_timing=True)
            t_gpu_start.record()
            num_blocks_to_swap = len(src_to_dst)
            for i in range(self.num_attention_layers):
                self.attn_backend.swap_blocks(self.cpu_cache[i], self.gpu_cache[i],
                                            src_to_dst)
            t_gpu_end.record()
            torch.cuda.synchronize()
            end_time = time.time_ns()
            logger.debug(f"+++++++++++++++swap_in {num_blocks_to_swap} blocks from cpu cache took cpu_time {(end_time - start_time)/1e6:.3f} ms, gpu_time {(t_gpu_start.elapsed_time(t_gpu_end)):.3f} ms.")

    def swap_out(self, src_to_dst: torch.Tensor) -> None:
        logger.debug(f"In cache_engine swap_out, src_to_dst tensor is allocated on device: {src_to_dst.device}")
        # Question 1: when to released the secondary gpu cache?
        if self.enable_secondary_gpu_cache:
            start_time = time.time_ns()
            t_gpu_start = torch.cuda.Event(enable_timing=True)
            t_gpu_end = torch.cuda.Event(enable_timing=True)
            t_gpu_start.record()
            num_blocks_to_swap = len(src_to_dst)
            kv_cache_shape = self.attn_backend.get_kv_cache_shape(
                num_blocks_to_swap, self.block_size, self.num_kv_heads, self.head_size)
            for i in range(self.num_attention_layers):
                if self.secondary_gpu_cache.can_allocate(i, kv_cache_shape):
                    self.secondary_gpu_cache.allocate(i, kv_cache_shape, src_to_dst)
                    self.secondary_gpu_cache.swap_out(i, self.gpu_cache[i], src_to_dst)
                else:
                    self.attn_backend.swap_blocks(self.gpu_cache[i], self.cpu_cache[i],
                                                src_to_dst)
            t_gpu_end.record()
            torch.cuda.synchronize()
            end_time = time.time_ns()
            logger.debug(f"+++++++++++++++Swap out {num_blocks_to_swap} blocks per layer to the second gpu cache take cpu_time {(end_time - start_time)/1e6:.3f} ms, gpu_time {(t_gpu_start.elapsed_time(t_gpu_end)):.3f} ms.")
        else:
            start_time = time.time_ns()
            num_blocks_to_swap = len(src_to_dst)
            t_gpu_start = torch.cuda.Event(enable_timing=True)
            t_gpu_end = torch.cuda.Event(enable_timing=True)
            t_gpu_start.record()
            for i in range(self.num_attention_layers):
                self.attn_backend.swap_blocks(self.gpu_cache[i], self.cpu_cache[i],
                                            src_to_dst)
            t_gpu_end.record()
            torch.cuda.synchronize()
            end_time = time.time_ns()
            logger.debug(f"+++++++++++++++Swap out {num_blocks_to_swap} blocks per layer to cpu cache take cpu_time {(end_time - start_time)/1e6:.3f} ms, gpu_time {(t_gpu_start.elapsed_time(t_gpu_end)):.3f} ms.")

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
