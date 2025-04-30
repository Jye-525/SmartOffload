import torch
import torch.nn as nn
from torch.func import functional_call
from collections import deque
from typing import List
from vllm.utils import is_pin_memory_available
from vllm.spec_decode.util import nvtx_range
import time

from vllm.logger import init_logger
logger = init_logger(__name__)

class PerLayerParameters:
    def __init__(self):
        self.parameters = {} # Ensure it's a new dictionary for each instance

class EventsTracker:
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.prefetch_events = [torch.cuda.Event() for _ in range(num_layers)]
        self.recording_events = [False for _ in range(num_layers)]
        
    def record(self, layer_idx: int, stream: torch.cuda.Stream):
        assert 0 <= layer_idx < self.num_layers, f"Event index out of range,it should be within [0, {self.num_layers})"
        self.prefetch_events[layer_idx].record(stream)
        self.recording_events[layer_idx] = True
        
    def synchronize(self, layer_idx: int):
        assert 0 <= layer_idx < self.num_layers, f"Event index out of range,it should be within [0, {self.num_layers})"
        if self.recording_events[layer_idx]:
            self.prefetch_events[layer_idx].synchronize()
            self.recording_events[layer_idx] = False
        
    def is_record(self, layer_idx: int):
        assert 0 <= layer_idx < self.num_layers, f"Event index out of range,it should be within [0, {self.num_layers})"
        return self.recording_events[layer_idx]

class PrefetchObject:
    def __init__(self, module: torch.nn.Module, to_dynamic_gpu_buffer: bool):
        self.module = module # The layer to be prefetched
        self.to_dynamic_gpu_buffer = to_dynamic_gpu_buffer  # Whether prefetching to dynamic GPU buffer or to resident GPU buffer

class LayersManager:
    def __init__(self, start_layer_idx: int, end_layer_idx: int, device: str):
        self.device = device
        self.s_layer_idx = start_layer_idx
        self.e_layer_idx = end_layer_idx
        self.n_layers = end_layer_idx - start_layer_idx
        self.cpu_buffers = [PerLayerParameters() for _ in range(self.n_layers)]
        self.events_tracker = EventsTracker(self.n_layers)
        self.dynamic_gpu_buffers = [None for _ in range(2)] 
        self.prefetch_buffer_index = 0
        self.prefetched_modules = deque()
        self.to_fetch_modules = deque() # include the modules to be fetched to the dynamic GPU buffer and static GPU buffer 
        self.layer_metadata = [
            {
                'total_bytes': 0,   # Total memory required in bytes
                'total_elems': 0,   # Total number of elements across all parameters 
                'dtype': None,      # Data type (torch.bfloat16, etc.)
                'parameters': {}    # Dictionary to store the offset of each individual parameter
            } 
            for _ in range(self.n_layers)
        ] 
        self.layers = [None for _ in range(self.n_layers)]
        self.data_mv_stream = torch.cuda.Stream()
        
    def create_module_on_cpu(self, module: torch.nn.Module, layer_idx: int):
        inter_layer_idx = layer_idx - self.s_layer_idx
        # Create a CPU buffer and copy the parameters of this module layer to the CPU buffer (pinned memory)
        pin_memory = is_pin_memory_available()
        total_elems = 0
        elem_size = 0
        cpu_buffer = self.cpu_buffers[inter_layer_idx]
        layer_dict = self.layer_metadata[inter_layer_idx]
        layer_params = self.layer_metadata[inter_layer_idx]['parameters']
        for name, p in module.named_parameters():
            cpu_data = torch.empty_strided(size=p.data.size(),
                                            stride=p.data.stride(),
                                            dtype=p.data.dtype,
                                            layout=p.data.layout,
                                            device='cpu',
                                            pin_memory=pin_memory)
            cpu_data.copy_(p.data) # if delete this line.
            p.data = cpu_data
            # store the parameters in the CPU buffer
            cpu_buffer.parameters[name] = cpu_data
            # store the metadata of the parameters of per layer
            total_elems += p.data.numel()
            layer_params[name] = p.data.numel() 
            if layer_dict['dtype'] is None:
                layer_dict['dtype'] = p.data.dtype
                elem_size = p.data.element_size()
            
        layer_dict['total_elems'] = total_elems
        layer_dict['total_bytes'] = total_elems * elem_size
        
        # save each modules and init the dynamic GPU buffers
        self.layers[layer_idx - self.s_layer_idx] = module
        if layer_idx == self.e_layer_idx - 1:
            # Initialize the dynamic GPU buffers
            layer_dict = self.layer_metadata[layer_idx - self.s_layer_idx]
            for buf_idx in range(2):
                # Create a GPU buffer for this layer
                gpu_data = torch.zeros(layer_dict['total_elems'],
                                    dtype=layer_dict['dtype'], device=self.device)
                self.dynamic_gpu_buffers[buf_idx] = gpu_data
        
        return module
    
    
    def prefetch_to_dynamic_gpu_buffer(self, module: torch.nn.Module, inner_layer_idx: int):
        dst_buf = self.dynamic_gpu_buffers[self.prefetch_buffer_index]
        self.__async_h2d_copy_event(module, dst_buf, inner_layer_idx) 
        self.prefetched_modules.append(PrefetchObject(module, True))
        self.prefetch_buffer_index ^= 1 # switch the prefetch buffer index
        return module
    
    def prefetch_to_static_gpu_buffer(self, inner_layer_idx: int):
        layer_dict = self.layer_metadata[inner_layer_idx]
        layer_params = layer_dict['parameters']
        layer = self.layers[inner_layer_idx]
        with torch.cuda.stream(self.data_mv_stream):
            gpu_data = torch.zeros(layer_dict['total_elems'],
                                dtype=layer_dict['dtype'], device=self.device)
            start_offset = 0
            for name, p in layer.named_parameters():
                end_offset = start_offset + layer_params[name]
                gpu_data[start_offset:end_offset].copy_(p.data.view(-1), non_blocking=True)
                p.data = gpu_data[start_offset:end_offset].view_as(p.data) 
        self.events_tracker.record(inner_layer_idx, self.data_mv_stream)
        # No need to add it to prefetched_modules
    
    def process_offloaded_layer(self, module: torch.nn.Module, inner_layer_idx: int, cur_k_value: int):
        # if the current layer is the kth and 2kth layer, asyncronously copy the data to the dynamic GPU buffer
        if (inner_layer_idx + 1) in [cur_k_value, 2 * cur_k_value]:
            self.prefetch_to_dynamic_gpu_buffer(module, inner_layer_idx)
        else:
            # add the layer to the to_prefetch_modules
            self.to_fetch_modules.append(PrefetchObject(module, True))
        return module
    
    def process_static_layer(self, module: torch.nn.Module, inner_layer_idx: int, cur_k_value: int):
        # get the device of the current module
        device = next(module.parameters()).device
        if device == torch.device("cpu"):
            # Need to asynchrnously prefetch the module to the static GPU buffer
            if inner_layer_idx + 1 < 2 * cur_k_value:
                self.prefetch_to_static_gpu_buffer(inner_layer_idx)
            else:
                self.to_fetch_modules.append(PrefetchObject(module, False))
            
    def is_record(self, layer_idx: int):
        return self.events_tracker.is_record(layer_idx)
    
    def synchronize(self, layer_idx: int):
        self.events_tracker.synchronize(layer_idx)
        
    def sync_fetch_to_gpu(self, module: torch.nn.Module, inner_layer_idx: int):
        # Step 1: check if the current layer is an offloaded layer or not
        layer_dict = self.layer_metadata[inner_layer_idx]
        layer_params = layer_dict['parameters']
        gpu_data = torch.empty(layer_dict['total_elems'], 
                            dtype=layer_dict['dtype'], device=self.device)
        start_offset = 0
        for name, p in module.named_parameters():
            end_offset = start_offset + layer_params[name]
            gpu_data[start_offset:end_offset].copy_(p.data.view(-1))
            p.data = gpu_data[start_offset:end_offset].view_as(p.data)
            start_offset = end_offset
             
        return module
    
    def is_at_offload_boundary(self, inner_layer_idx: int, cur_k_value: int):
        # check if the current layer is a layer right after an offloaded layer, i.e, the offload boundary
        next_idx = inner_layer_idx + 1
        return (
            inner_layer_idx < self.n_layers and
            next_idx > cur_k_value and
            (cur_k_value == 1 or next_idx % cur_k_value == 1)
        )
               
    def maybe_prefetch_module_k_and_2k(self, total_offloaded_layers: int, cur_k_value: int):
        if total_offloaded_layers > 2:
            # Asynchronously prefetch the first kth and 2kth layer to the dynamic GPU buffer
            self.prefetched_modules.clear()
            self.prefetch_buffer_index = 0
            
            module_to_prefetch_k = self.to_fetch_modules.popleft().module
            inner_k_idx = module_to_prefetch_k.layer_idx - self.s_layer_idx
            assert inner_k_idx + 1 == cur_k_value, f"Error: the module to prefetch is not the kth layer, {module_to_prefetch_k.layer_idx + 1} != {self.s_layer_idx + cur_k_value}, k={cur_k_value}" 
            self.prefetch_to_dynamic_gpu_buffer(module_to_prefetch_k, inner_k_idx)
                    
            module_to_prefetch_2k = self.to_fetch_modules.popleft().module
            inner_2k_idx = module_to_prefetch_2k.layer_idx - self.s_layer_idx
            assert (inner_2k_idx) + 1 == 2 * cur_k_value, f"Error: the module to prefetch is not the 2kth layer, {module_to_prefetch_2k.layer_idx + 1} != {self.s_layer_idx + 2 * cur_k_value}, k={cur_k_value}"
            self.prefetch_to_dynamic_gpu_buffer(module_to_prefetch_2k, inner_2k_idx)
             
    
    def maybe_prefetch_future_modules(self, compute_event: torch.cuda.Event, cur_layer_idx: int, cur_k_value: int):
        # the next layer in to_fetch_modules is the one needed to be prefetched to dynamic GPU buffer or static buffer
        if not self.to_fetch_modules:
            # no modules to prefetched, which means all the layers are already on GPU
            # Step 1: if the current layer is a right after the kth layer, 
            # calling compute_event.synchronize to make sure the layer prefeteched to dynamic buffer finished the computation
            if self.is_at_offload_boundary(cur_layer_idx, cur_k_value):
                compute_event.synchronize()
            return
        
        to_fetch_obj = self.to_fetch_modules[0]
        if not to_fetch_obj.to_dynamic_gpu_buffer:
            # fetch the module to a static gpu buffer, need to allocate GPU memory
            module_to_prefetch = self.to_fetch_modules.popleft().module
            self.prefetch_to_static_gpu_buffer(module_to_prefetch.layer_idx - self.s_layer_idx)
        else:
            if self.is_at_offload_boundary(cur_layer_idx, cur_k_value):
                compute_event.synchronize()
                # Maybe trigger prefetch to the dynamic GPU buffer
                module_to_offload = self.prefetched_modules.popleft().module
                assert module_to_offload.layer_idx == cur_layer_idx - 1, f"OffloadBuffer.wraper_forward: the module to offload is not the previous layer, {module_to_offload.layer_idx} != {cur_layer_idx - 1}"
                cpu_buffer = self.cpu_buffers[module_to_offload.layer_idx - self.s_layer_idx]
                for name, p in module_to_offload.named_parameters():
                    p.data = cpu_buffer.parameters[name]
                self.to_fetch_modules.append(PrefetchObject(module_to_offload, True))
                
                # prefetch the next layer to the dynamic GPU buffer
                to_prefetch_layer_idx = cur_layer_idx + 2 * cur_k_value - 1
                if to_prefetch_layer_idx < self.n_layers: 
                    module_to_prefetch = self.to_fetch_modules.popleft().module
                    inner_layer_idx = module_to_prefetch.layer_idx - self.s_layer_idx
                    assert inner_layer_idx == to_prefetch_layer_idx, f"Error: the module to prefetch is not the 2kth layer, {module_to_prefetch.layer_idx} != {to_prefetch_layer_idx}, k={cur_k_value}"
                    self.prefetch_to_dynamic_gpu_buffer(module_to_prefetch, inner_layer_idx)
        
        
    def maybe_offload_last_layer(self, cur_layer_idx: int, total_offloaded_layers: int):
        if total_offloaded_layers > 2:
            # offload the last layer to the CPU buffer
            module_to_offload = self.prefetched_modules.popleft().module
            assert module_to_offload.layer_idx == cur_layer_idx, f"Error: the module to offload is not the last layer, {module_to_offload.layer_idx} != {cur_layer_idx}"
            cpu_buffer = self.cpu_buffers[module_to_offload.layer_idx - self.s_layer_idx]
            for name, p in module_to_offload.named_parameters():
                p.data = cpu_buffer.parameters[name]
            self.to_fetch_modules.append(PrefetchObject(module_to_offload, True))
            
    def reset_all_modules(self):
        # Step 1: set the modules in prefetched_modules to the CPU buffer 
        while self.prefetched_modules:
            module_to_offload = self.prefetched_modules.popleft().module
            cpu_buffer = self.cpu_buffers[module_to_offload.layer_idx - self.s_layer_idx]
            for name, p in module_to_offload.named_parameters():
                p.data = cpu_buffer.parameters[name]
        
        # Step 2: clear the prefetched_modules and to_fetch_modules
        self.prefetched_modules.clear()
        self.to_fetch_modules.clear()
        self.prefetch_buffer_index = 0
        
    def get_layer(self, inner_layer_idx: int): 
        return self.layers[inner_layer_idx]
        
    def __async_h2d_copy_event(self, layer: torch.nn.Module, dst_buf: torch.Tensor, layer_idx: int):
        # Asynchronous copy the data to the GPU buffer
        with torch.cuda.stream(self.data_mv_stream):
            start_offset = 0
            for name, p in layer.named_parameters():
                end_offset = start_offset + p.data.numel()
                dst_buf[start_offset:end_offset].copy_(p.data.view(-1), non_blocking=True)
                p.data = dst_buf[start_offset:end_offset].view_as(p.data)
                start_offset = end_offset
        self.events_tracker.record(layer_idx, self.data_mv_stream)

   
class OffloadBuffer:
    def __init__(self, k: int, param_offload_target:str):
        self.curr_k_value = k
        self.start_layer = -1 # record the start layer in this pipeline rank, will be updated in create_module
        self.end_layer = -1 
        self.param_offload_target = param_offload_target
        self.layers_manager = None      
        self.compute_event = torch.cuda.Event()
        self.original_forwards = []
        self.wraper_forwards = []
        self.total_offloaded_layers = 0
        self.first_fwd_after_reorganize = True
        print(f"+++++++++++ init OffloadBuffer with k={k}, param_offload_target={param_offload_target}")
        
        
    def create_module(self, module: torch.nn.Module, layer_idx: int, start_layer: int, end_layer: int):
        # get the original device of the module
        self.device = next(module.parameters()).device 
        if self.device == torch.device("cpu"):
            return module
        
        assert self.curr_k_value > 0, "The init_k should be greater than 0 when smart offload is enabled"
        
        if self.layers_manager is None:
            # initialize the parameters used in this pipeline rank
            self.layers_manager = LayersManager(start_layer, end_layer, self.device)
            self.start_layer = start_layer
            self.end_layer = end_layer
            self.original_forwards = [None for _ in range(end_layer - start_layer)]
            self.wraper_forwards = [None for _ in range(end_layer - start_layer)]
            self.total_offloaded_layers = (end_layer - start_layer) // self.curr_k_value
        
        assert self.layers_manager is not None, "OffloadBuffer: layers_manager is None, please check the initialization"   
        # create a wrapper forward function for the module
        def forward(*args, **kwargs):
            return self.wraper_forward(module, args=args, kwargs=kwargs)
        
        inter_layer_idx = layer_idx - start_layer
        # Update the original forward and wrapper forward function pointer for the module
        self.original_forwards[inter_layer_idx] = module.forward
        module.forward = forward 
        self.wraper_forwards[inter_layer_idx] = module.forward

        # Step 1: Create a CPU buffer and copy the parameters of this module layer to the CPU buffer (pinned memory)
        module = self.layers_manager.create_module_on_cpu(module, inter_layer_idx)
        
        return module
    
        
    @nvtx_range("OffloadBuffer.maybe_offload")    
    def maybe_offload(self, module: torch.nn.Module, layer_idx: int, start_layer: int, end_layer: int):
        # This function is called once when start vLLM server
        load_module_time = 0
        inner_layer_idx = layer_idx - start_layer
        # Step 1: check if current layer is an offload layer
        if (inner_layer_idx + 1) % self.curr_k_value == 0:
            tt_start = time.time_ns()
            module = self.layers_manager.process_offloaded_layer(module, inner_layer_idx, self.curr_k_value) 
            if self.curr_k_value == 1:
               self.layers_manager.synchronize(inner_layer_idx)
            tt_end = time.time_ns()
            load_module_time = (tt_end - tt_start) / 1e6
        else:
            tt_start = time.time_ns()
            # Step2: this layer should be put on GPU after model weights loading
            module = self.layers_manager.sync_fetch_to_gpu(module, inner_layer_idx)
            tt_end = time.time_ns()
            load_module_time = (tt_end - tt_start) / 1e6
            # self.H2D_transfer_times.append(load_module_time)
            
        logger.debug(f"OffloadBuffer.maybe_offload: load layer_idx={layer_idx} cost {load_module_time:.3f} ms")
        return module
    
    
    @nvtx_range("OffloadBuffer.wraper_forward")
    def wraper_forward(self, module: torch.nn.Module, args=None, kwargs=None):
        # intercept the original forward function
        cur_layer_idx = module.layer_idx
         
        if cur_layer_idx == self.start_layer:
            if not self.first_fwd_after_reorganize:
                # prefetch the kth and 2kth layer to the dynamic GPU buffer
                with nvtx_range(f"wraper_fwd_step1_{cur_layer_idx}"):
                    self.layers_manager.maybe_prefetch_module_k_and_2k(self.total_offloaded_layers, self.curr_k_value)
            else:
                self.first_fwd_after_reorganize = False 
        
        # Step 1: prefetch the to dynamic or static GPU buffer
        with nvtx_range(f"wraper_fwd_step2_{cur_layer_idx}"):       
            self.layers_manager.maybe_prefetch_future_modules(self.compute_event, cur_layer_idx, self.curr_k_value)
        
        # Step 2: synchonize the event to make sure the required data for current forward is finished
        with nvtx_range(f"wraper_fwd_step3_{cur_layer_idx}"):
            self.layers_manager.synchronize(cur_layer_idx - self.start_layer)
        
        # Step 3: Perform the foward
        # run the forward
        with nvtx_range(f"wraper_fwd_step4_{cur_layer_idx}"):
            inner_layer_idx = cur_layer_idx - self.start_layer
            module.forward = self.original_forwards[inner_layer_idx]
            output = functional_call(module,
                                    module.state_dict(),
                                    args=args,
                                    kwargs=kwargs)
            # restore the forward function to the wrapper forward function
            module.forward = self.wraper_forwards[inner_layer_idx]
            
        # Step 4: process the last layer
        if cur_layer_idx == self.end_layer - 1 and (cur_layer_idx + 1) % self.curr_k_value == 0:
            self.compute_event.synchronize()
            self.layers_manager.maybe_offload_last_layer(cur_layer_idx, self.total_offloaded_layers)
            
        return output
    
    @nvtx_range("OffloadBuffer.reorg_gpu")
    def reorganize_resident_gpu_modules(self, k: int):
        if self.curr_k_value == k:
            return
        
        old_k_value = self.curr_k_value
        self.curr_k_value = k
        print(f"changing the k value from {old_k_value} to {self.curr_k_value}", flush=True) 
        self.layers_manager.reset_all_modules()
        
        # Step 2: rearrange all the modules
        for inner_layer_idx in range(0, self.end_layer - self.start_layer):
            layer_module = self.layers_manager.get_layer(inner_layer_idx)
            assert layer_module is not None, f"OffloadBuffer.reorganize_resident_gpu_modules: the layer module is None, {inner_layer_idx}"
            if (inner_layer_idx + 1) % self.curr_k_value == 0:
                # dynamic layers to preftech to dynamic buffer
                self.layers_manager.process_offloaded_layer(layer_module, inner_layer_idx, self.curr_k_value)
            else:
                # static layers
                self.layers_manager.process_static_layer(layer_module, inner_layer_idx, self.curr_k_value)
            
        self.first_fwd_after_reorganize = True # reset the first forward pass after reorganize
        self.total_offloaded_layers = (self.end_layer - self.start_layer) // self.curr_k_value            
        
        # logger.debug(f"Reorganizing the resident GPU modules from stride of {old_k_value} stride size of {self.curr_k_value}, "
        #              f"reorganizing layers cost {((t_end - t_start) / 1e6):.3f} ms, "
        #              f"per layer time = {[f'{k}={v:.3f}' for k, v in times_per_layer.items()]}, "
        #              f"total_offloaded_layers={self.total_offloaded_layers},"
        #              f"aaa cost = {((t_start_aaa - t_start) / 1e6):.3f} ms")
        
        logger.debug(f"Reorganizing the resident GPU modules from stride of {old_k_value} stride size of {self.curr_k_value}")

    def get_offload_interval(self):
        return self.curr_k_value
    
    def get_per_layer_params_Bytes(self):
        # Calculate the total size of parameters in bytes
        assert self.layers_manager.layer_metadata is not None, "OffloadBuffer: per_layer_info is None, please call create_module first"
        total_size = self.layers_manager.layer_metadata[0]['total_bytes']
        return total_size

    
# NOTES:
# 1. We use two dynamic GPU buffers to store the prefetching and computing operations, which are used to switch between transfers and computations alternately.
# 2. Use a generic prefetch_events to record the prefetching events, and a generic recording_events to check if that element is being prefetched or not (only if it is being prefetched, check prefetch_events[i].sycnhronize()).
# 3. Use a static_cpu_buffers to permanently store the CPU buffers. This way you can always lookup the CPU based buffer in the static_cpu_buffers datastructure.
# 4. Use a dynamic_gpu_buffers to store the GPU buffers, which are used to switch between transfers and computations alternately.
# 5. Use a resident_gpu_buffers to store the `k` transformer blocks on the GPU. The resident_gpu_buffer will be dynamically allocated at every forward pass using `reorganize_resident_gpu_modules`.
# 6. Remove the `initial_offload_layers`, assume that we do equi-spaced offloading, i.e,. offload every `k` layers.
#    6.1 replace with initial_offload_layers with k
