import torch
from typing import Union
from typing import List
import logging
from vllm.spec_decode.util import nvtx_range
from typing import TYPE_CHECKING, Tuple
if TYPE_CHECKING:
    from vllm.attention.layer import Attention

DEBUG_MODE = False
logging.basicConfig(level=logging.DEBUG if DEBUG_MODE else logging.CRITICAL)
logger = logging.getLogger(__name__)

def print_mem_stats():
    return torch.cuda.memory_stats()["allocated_bytes.all.current"]

class LayerManager:
    def __init__(self, layer_id: int, start_layer: int, end_layer: int, compute_stream: torch.cuda.Stream, trf_stream_to: List[torch.cuda.Stream]):
        self.layer_id = layer_id
        self.module = None
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.compute_stream = compute_stream
        self.cuda_device = compute_stream.device
        self.cpu_device = torch.device('cpu')
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.tensors = {"kv_cache": torch.empty((0,), device=self.cpu_device)}
        self.pinned_cpu_tensors = {k: v.to(device=self.cpu_device) for k, v in self.tensors.items()}
        self.is_recording_compute_event = False
        self.trf_stream_to = trf_stream_to
        self.on_gpu_static = True
        self.kv_map = None
        self.kv_holders = {}

    def register_module(self, module: torch.nn.Module):
        self.module = module

    def add_tensor(self, name: str, p: Union[torch.nn.Parameter, torch.Tensor], kv_map: Tuple[int, int, str] = None):
        # assert name not in self.tensors, f"Parameter/tensor {name} already exists"
        if kv_map is not None:  
            logger.info(f"Layer {self.layer_id}, resident={self.on_gpu_static} adding kv tensor. is p a tensor? {torch.is_tensor(p)}")
        if (not self.on_gpu_static):
            t = p
            if not torch.is_tensor(p):
                t = torch.empty_strided(size=p.data.size(),
                                            stride=p.data.stride(),
                                            dtype=p.data.dtype,
                                            layout=p.data.layout,
                                            device=self.cpu_device,
                                            pin_memory=True)
                t.data.copy_(p.data)
            t = t.to(device=self.cpu_device).pin_memory()
            p.data = t
        self.tensors[name] = p
        # Q1: This one may be wrong, because the model weights will directly load to the GPU memory if it is on_gpu_static = True
        # The self.pinned_cpu_tensors[name] is always 0.
        self.pinned_cpu_tensors[name] = p.to(device=self.cpu_device).pin_memory()
        if name in "kv_cache":
            assert kv_map is not None, f"kv_map must be provided for kv_cache"
            self.kv_map = kv_map
            layer_id, idx, layer_name = kv_map
            self.kv_holders["runners_kv_cache"][idx] = p
            self.kv_holders["forward_context"][layer_name].kv_cache = [p]

        if self.layer_id == self.start_layer:
            self.move_to_cuda()

    @nvtx_range("LayerManager::move_to_cuda")
    def move_to_cuda(self):
        device = self.cuda_device
        for name, t in self.pinned_cpu_tensors.items():
            with torch.cuda.stream(self.trf_stream_to[device]):
                self.tensors[name] = t.to(device=device, non_blocking=True)

    def _async_update_kv_pointers(self, to_device):
        if self.kv_map is not None:
            self.trf_stream_to[to_device].synchronize()
            layer_id, idx, layer_name = self.kv_map
            if to_device == self.cpu_device:
                self.tensors["kv_cache"] = self.pinned_cpu_tensors["kv_cache"]
            p = self.tensors["kv_cache"]
            assert p.device == to_device, f"KV cache for {self.layer_id} is on {p.device} not on desired device {to_device}"
            self.kv_holders["runners_kv_cache"][idx] = p
            self.kv_holders["forward_context"][layer_name].kv_cache = [p]

    @nvtx_range("LayerManager::move_to_cpu")
    def move_to_cpu(self):
        device = self.cpu_device
        for name, t in self.tensors.items():
            if name == "kv_cache":
                with torch.cuda.stream(self.trf_stream_to[device]):
                    # We do a cpu_tensor.copy_() to avoid reallocation overhead on the CPU.
                    self.pinned_cpu_tensors[name].copy_(self.tensors[name], non_blocking=True)
            else:
                t = self.pinned_cpu_tensors[name]
                self.tensors[name] = t
        self.map_module_tensors(to_device=self.cpu_device)

    def print_mappings(self):
        for name, t in self.tensors.items():
            logger.info(f"Layer {self.layer_id}, resident={self.on_gpu_static} tensor {name}: {t.device}, {t.shape}")

    def map_module_tensors(self, to_device: torch.device):
        if self.module is None:
            return
        for name, t in self.module.named_parameters():
            t.data = self.tensors[name] if to_device == self.cuda_device else self.pinned_cpu_tensors[name]

    @nvtx_range("LayerManager::begin_compute")
    def begin_compute(self):
        self.is_recording_compute_event = True
        self.start_event.record(self.compute_stream)
        if not self.on_gpu_static:
            self.trf_stream_to[self.cuda_device].synchronize()    
            self.map_module_tensors(to_device=self.cuda_device)
            self._async_update_kv_pointers(to_device=self.cuda_device)
        logger.debug(f"Layer {self.layer_id}, resident={self.on_gpu_static} compute start event recorded")
        
    
    def end_compute(self):
        assert self.is_recording_compute_event, "Compute end event registered without start"
        self.end_event.record(self.compute_stream)
        self.end_event.synchronize()
        self.is_recording_compute_event = False
        # logger.debug(f"Layer {self.layer_id}, resident={self.on_gpu_static}  compute end event recorded")
        # if not self.on_gpu_static:
        logger.debug(f"***** Layer {self.layer_id}, {self.on_gpu_static}, takes {self.start_event.elapsed_time(self.end_event):.3f} ms *****")
    
    # Question: No one use this funtion now? 
    def point_to_cpu_buffers(self):
        for name, t in self.tensors.items():
            t = self.pinned_cpu_tensors[name]
            self.tensors[name] = t

class SmartBufferManager:
    _instance = None

    def __init__(self, start_layer: int, end_layer: int, k: int):
        assert k > 0, "k must be greater than 0"
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.cuda_device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.cpu_device = torch.device('cpu')
        self.trf_stream_to = {}
        self.trf_stream_to[self.cuda_device] = torch.cuda.Stream()
        self.trf_stream_to[self.cpu_device] = torch.cuda.Stream()
        self.compute_stream = torch.cuda.Stream()
        self.layers = {layer_id: LayerManager(layer_id, start_layer, end_layer, self.compute_stream, self.trf_stream_to) for layer_id in range(start_layer, end_layer)}
        self.k_layers = self.set_k_layers(k)
        self.prev_dynamic_layer = start_layer
        SmartBufferManager._instance = self

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            raise Exception("SmartBufferManager not initialized")
        return cls._instance
    
    def register_module(self, layer_id: int, module: torch.nn.Module):
        assert layer_id in self.layers, f"Layer {layer_id} not in range"
        assert self.layers[layer_id] is not None, f"Layer {layer_id} already exists"
        self.layers[layer_id].register_module(module)

    def get_layer(self, layer_id: int):
        assert layer_id in self.layers, f"Layer {layer_id} not in range"
        assert self.layers[layer_id] is not None, f"Layer {layer_id} already exists"
        return self.layers[layer_id]
    
    def add_tensor(self, layer_id: int, name: str, p: Union[torch.nn.Parameter, torch.Tensor]):
        assert layer_id in self.layers, f"Layer {layer_id} not in range"
        assert self.layers[layer_id] is not None, f"Layer {layer_id} already exists"
        with nvtx_range(f"SmartBufferManager::add_tensor_{layer_id}"):
            self.layers[layer_id].add_tensor(name, p)

    def set_kv_holders(self, runners_kv_cache: List[torch.Tensor], forward_context: dict[str, "Attention"]):
        for layer_id in range(self.start_layer, self.end_layer):
            self.layers[layer_id].kv_holders["runners_kv_cache"] = runners_kv_cache
            self.layers[layer_id].kv_holders["forward_context"] = forward_context

    def add_kv_tensor(self, p: torch.Tensor, kv_mapping: Tuple[int, int, str]):
        layer_id, idx, layer_name = kv_mapping
        assert layer_id in self.layers, f"Layer {layer_id} not in range"
        assert self.layers[layer_id] is not None, f"Layer {layer_id} already exists"
        with nvtx_range(f"SmartBufferManager::add_kv_tensor_{layer_id}"):
            self.layers[layer_id].add_tensor("kv_cache", p=p, kv_map=kv_mapping)

    def begin_compute(self, layer_id: int):
        assert layer_id in self.layers, f"Layer {layer_id} not in range"
        assert self.layers[layer_id] is not None, f"Layer {layer_id} already exists"
        with nvtx_range(f"SmartBufferManager::begin_compute_{layer_id}_kv_to_cpu_prev_{self.prev_dynamic_layer}"):
            if (not self.layers[layer_id].on_gpu_static) and (self.k_layers is not None) and (self.prev_dynamic_layer != layer_id):
                self.layers[self.prev_dynamic_layer]._async_update_kv_pointers(to_device=self.cpu_device)
                self.prev_dynamic_layer = layer_id
        self.layers[layer_id].begin_compute()
        
    def end_compute(self, layer_id: int):
        assert layer_id in self.layers, f"Layer {layer_id} not in range"
        assert self.layers[layer_id] is not None, f"Layer {layer_id} already exists"
        self.layers[layer_id].end_compute()
        if (not self.layers[layer_id].on_gpu_static) and (self.k_layers is not None):
            # This is a dynamic GPU parameter, ofload it's KV cache and fetch the `k`th parameter
            self.layers[layer_id].move_to_cpu()
            
            next_k_layer = layer_id + self.k_layers
            if next_k_layer >= self.end_layer:
                next_k_layer = self.start_layer
            logger.debug(f"***** Layer {layer_id} is a dynamic GPU parameter, offloading it's KV cache and fetching the {next_k_layer} parameter *****")
            assert not self.layers[next_k_layer].on_gpu_static, f"Layer {next_k_layer} is a static GPU resident"
            with nvtx_range(f"SmartBufferManager::move_{next_k_layer}_to_cuda"):
                self.layers[next_k_layer].move_to_cuda()
        
        logger.info(f"Memory after performing layer {layer_id} computation. Used memory: {print_mem_stats() / (1024**2):.2f} MB")

    def set_k_layers(self, k: int):
        if k > self.end_layer - self.start_layer:
            return None
        for idx, layer_id in enumerate(range(self.start_layer, self.end_layer)):
            if idx % k == 0:
                with nvtx_range(f"SmartBufferManager::move_{layer_id}_to_cpu"):
                    self.layers[layer_id].on_gpu_static = False
                    self.layers[layer_id].move_to_cpu()
            else:
                with nvtx_range(f"SmartBufferManager::move_{layer_id}_to_cuda"):
                    self.layers[layer_id].on_gpu_static = True
                    self.layers[layer_id].move_to_cuda()
        return k

