"""CacheEngine class for managing the KV cache."""
from typing import List, Tuple
import math, time

import torch

from vllm.attention import get_attn_backend
from vllm.config import CacheConfig, DeviceConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, LayerBlockType,
                        get_dtype_size, is_pin_memory_available, GiB_bytes)

from collections import deque

logger = init_logger(__name__)

class SecondaryGPUCache:
    def __init__(self, block_size, num_kv_heads, head_size, cpu_cache, dtype=torch.float16, device="cuda", fraction_of_gpu_memory=0.05):
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.dtype = dtype
        self.device = device
        self.cpu_cache = cpu_cache
        self.dtype_size = torch.tensor([], dtype=dtype).element_size()
        self.block_shape = (2, block_size, num_kv_heads, head_size)
        self.block_bytes = torch.empty(self.block_shape, dtype=dtype).element_size() * torch.empty(self.block_shape, dtype=dtype).numel()
        self.gpu_to_gpu_stream = torch.cuda.Stream(device=self.device)
        self.gpu_to_host_stream = torch.cuda.Stream(device=self.device)

        self.blocks: Dict[int, torch.Tensor] = {}
        self.available_blocks = deque()  # holds reusable s_ids
        self.buffer = deque()  # holds (s_id, event)
        self.s_to_h: Dict[int, int] = {}
        self.h_to_s: Dict[int, int] = {}

        self.s_id_counter = 0
        self.max_blocks = self._get_max_blocks(fraction_of_gpu_memory)
        self._preallocate_blocks(self.max_blocks)
        print(f"=== SecondaryGPUCache has allocated {self.max_blocks} blocks of size {self.block_shape} on {self.device} with dtype {self.dtype}.")

    def _get_max_blocks(self, fraction):
        total_mem = torch.cuda.get_device_properties(self.device).total_memory
        allocatable = int(total_mem * fraction)
        return allocatable // self.block_bytes

    def _preallocate_blocks(self, num_blocks):
        for _ in range(num_blocks):
            s_id = self.s_id_counter
            self.blocks[s_id] = torch.empty(self.block_shape, dtype=self.dtype, device=self.device)
            self.available_blocks.append(s_id)
            self.s_id_counter += 1

    def grow_cache(self, size_in_bytes):
        num_new_blocks = size_in_bytes // self.block_bytes
        self._preallocate_blocks(num_new_blocks)
        self.max_blocks += num_new_blocks

    def shrink_cache(self, size_in_bytes_to_shrink):
        num_to_remove = size_in_bytes_to_shrink // self.block_bytes
        removed = 0
        while removed < num_to_remove and self.buffer:
            s_id, event = self.buffer[-1]
            if not event.query():
                event.synchronize()
            h_id = self.s_to_h.pop(s_id)
            self.h_to_s.pop(h_id)
            self.buffer.pop()
            del self.self.blocks[s_id]
            removed += 1


    def swap_out(self, layer_id: int, gpu_src_blocks: torch.Tensor, src_ids: torch.Tensor):
        try:
            num_blocks = src_ids.shape[0]
            p_ids = src_ids[:, 0].tolist()
            h_ids = src_ids[:, 1].tolist()

            s_ids = [None]*num_blocks
            blocks_to_copy = []
            dst_keys_list = [None]*num_blocks
            dst_vals_list = [None]*num_blocks

            # Track start time
            t_start = time.time_ns()

            # Step 1: Reuse or evict to get free blocks
            for i, h_id in enumerate(h_ids):
                if not self.available_blocks:
                    evicted_s_id, evicted_event = self.buffer[-1]
                    if not evicted_event.query():
                        evicted_event.synchronize(self.gpu_to_host_stream)
                    evicted_h_id = self.s_to_h.pop(evicted_s_id)
                    self.h_to_s.pop(evicted_h_id)
                    self.buffer.pop()
                    self.available_blocks.append(evicted_s_id)

                s_id = self.available_blocks.popleft()
                self.s_to_h[s_id] = h_id
                self.h_to_s[h_id] = s_id
                s_ids[i] = s_id
                # For approach-1
                # blocks_to_copy.append((s_id, h_id))

                # For approach-2
                block = self.blocks[s_id]
                dst_keys_list[i] = block[0]
                dst_vals_list[i] = block[1]

            t_evict = time.time_ns()


            # --- Approach 1: Tensor by tensor copy ---
            # with torch.cuda.stream(self.gpu_to_gpu_stream):
            #     for i, (s_id, _) in enumerate(blocks_to_copy):
            #         p_id = p_ids[i]
            #         block = self.blocks[s_id]
            #         block[0].copy_(gpu_src_blocks[0][p_id], non_blocking=True)
            #         block[1].copy_(gpu_src_blocks[1][p_id], non_blocking=True)
            #         event = torch.cuda.Event()
            #         event.record(self.gpu_to_gpu_stream)
            #         self.buffer.appendleft((s_id, event))


            # --- Approach 2: Batch copy [BUT THIS CREATES A NEW MEMORY ALLOCATION FOR STACKING] ---
            src_keys = gpu_src_blocks[0][p_ids]
            src_vals = gpu_src_blocks[1][p_ids]
            dst_keys = torch.stack(dst_keys_list)
            dst_vals = torch.stack(dst_vals_list)

            t_ready = time.time_ns()

            event = torch.cuda.Event()
            # Copy primary KV cache to secondary GPU KV cache 
            dst_keys.copy_(src_keys)
            dst_vals.copy_(src_vals)
            
            # Enqueue lazy copy to the host memory too.
            with torch.cuda.stream(self.gpu_to_host_stream):
                self.cpu_cache[layer_id][0][h_ids].copy_(dst_keys, non_blocking=True)
                self.cpu_cache[layer_id][1][h_ids].copy_(dst_vals, non_blocking=True)
                event.record(self.gpu_to_host_stream)
            
            for s_id in s_ids:
                self.buffer.appendleft((s_id, event))

            t_end = time.time_ns()

            if True:
                total_bytes = 2 * self.block_size * self.num_kv_heads * self.head_size * self.dtype_size * num_blocks
                print(
                    f"[Layer {layer_id}] swap_out {num_blocks} blocks | "
                    f"Evict: {(t_evict - t_start)/1e6:.3f} ms | "
                    f"Ready: {(t_ready - t_evict)/1e6:.3f} ms | "
                    f"Copy: {(t_end - t_ready)/1e6:.3f} ms | "
                    f"Total: {(t_end - t_start)/1e6:.3f} ms | "
                    f"Bytes moved: {total_bytes} Bytes"
                )

        except Exception as e:
            print(f"[Layer {layer_id}] swap_out failed: {e}")
            import pdb
            pdb.set_trace()

    def swap_in(self, layer_id, gpu_dst_blocks: torch.Tensor, dst_ids: torch.Tensor):
        try:
            start_time = time.time_ns()
            num_blocks = dst_ids.shape[0]
            from_cpu = 0
            for i in range(num_blocks):
                h_id, p_id = dst_ids[i].tolist()
                s_id = self.h_to_s.get(h_id, None)
                if s_id is None:
                    # Not available in the Secondary GPU cache, need to transfer in from the CPU.
                    gpu_dst_blocks[0][p_id].copy_(self.cpu_cache[layer_id][0][h_id], non_blocking=True)
                    gpu_dst_blocks[1][p_id].copy_(self.cpu_cache[layer_id][0][h_id], non_blocking=True)
                    from_cpu += 1
                    continue
                block = self.blocks[s_id]
                gpu_dst_blocks[0][p_id].copy_(block[0], non_blocking=True)
                gpu_dst_blocks[1][p_id].copy_(block[1], non_blocking=True)
            total_bytes = 2 * self.block_size * self.num_kv_heads * self.head_size * self.dtype_size * num_blocks
            print(f"[Layer {layer_id}] Swap in {num_blocks} blocks of total size {total_bytes} took {(time.time_ns() - start_time)/1e6:.3f} ms. From CPU: {from_cpu} blocks.")
        except Exception as e:
            print(f"Error in swap_in: {e}")
            import pdb; pdb.set_trace()
    



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
        self.secondary_gpu_cache = SecondaryGPUCache(self.block_size, self.num_kv_heads, self.head_size, self.cpu_cache, self.dtype, self.device_config.device_type)

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
                self.secondary_gpu_cache.swap_in(i, self.gpu_cache[i], src_to_dst)
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
        print(f"In cache_engine swap_out, src_to_dst tensor is allocated on device: {src_to_dst.device}")
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
                self.secondary_gpu_cache.swap_out(i, self.gpu_cache[i], src_to_dst)
            t_gpu_end.record()
            torch.cuda.synchronize()
            end_time = time.time_ns()
            print(f"+++++++++++++++Swap out {num_blocks_to_swap} blocks per layer to the second gpu cache take cpu_time {(end_time - start_time)/1e6:.3f} ms, gpu_time {(t_gpu_start.elapsed_time(t_gpu_end)):.3f} ms.")
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
