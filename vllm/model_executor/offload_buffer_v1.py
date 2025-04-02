import torch
import torch.nn as nn
from torch.func import functional_call
from collections import deque
from typing import List
from vllm.utils import is_pin_memory_available
from vllm.spec_decode.util import nvtx_range

from vllm.logger import init_logger
logger = init_logger(__name__)

class PerLayerParameters:
    def __init__(self):
        self.parameters = {} # Ensure it's a new dictionary for each instance
    
class OffloadBuffer:
    def __init__(self, cpu_offload_layers:List[int], param_offload_target:str):
        self.initial_offload_layers = cpu_offload_layers  
        self.param_offload_target = param_offload_target
        self.layers = 0 # record the number of layers in this pipeline rank, will be updated in create_module
        self.start_layer = -1 # record the start layer in this pipeline rank, will be updated in create_module
        self.end_layer = -1 # record the last layer in this pipeline rank, will be updated in create_module
        self.cpu_buffer = None # the copy of all the transformer block layers.
        self.gpu_buffer = None # Assume the model has uniform trabsformer block structure. (A single buffer for prefetch) 
        self.original_forwards = []
        self.wraper_forwards = []
        self.offloaded_modules = deque()
        self.prefetched_modules = deque()
        self.prefetch_event = torch.cuda.Event()
        self.compute_event = torch.cuda.Event()
        self.data_mv_stream = torch.cuda.Stream() # create a new stream for H2D data transfer
    
    def __initialization(self, start_layer: int, end_layer: int):
        self.layers = end_layer - start_layer
        self.start_layer = start_layer
        self.end_layer = end_layer
        # initialize the cpu_buffer (the buffer on host memory)
        self.cpu_buffer = [PerLayerParameters() for _ in range(self.layers)]
        self.gpu_buffer = PerLayerParameters()
        self.original_forwards = [None for _ in range(self.layers)]
        self.wraper_forwards = [None for _ in range(self.layers)]
        # add the start_layer to the initial_offload_layers since we reuse the same buffer
        if self.initial_offload_layers is None and len(self.initial_offload_layers) > 0:
            self.initial_offload_layers.insert(0, start_layer)
        logger.debug(f"OffloadBuffer.__initialization: initial_offload_layers={self.initial_offload_layers}")
    
        
    def create_module(self, module: torch.nn.Module, layer_idx: int, start_layer: int, end_layer: int):
        # get the original device of the module
        self.device = next(module.parameters()).device 
        if self.device == torch.device("cpu") or \
            (self.initial_offload_layers is None or len(self.initial_offload_layers) == 0):
            # if device is CPU or the initial_offload_layers is None or empty,
            # we don't need to offload the module, so just return the original module
            return module
        
        logger.debug(f"OffloadBuffer.create_module: layer_idx={layer_idx}, start_layer={start_layer}, end_layer={end_layer}")
        
        if self.layers == 0:
            # initialize and update parameters used in this pipeline rank
            self.__initialization(start_layer, end_layer)
        
        if layer_idx == start_layer:
            # create a GPU buffer on the device based on the size of the start layer, which will be used for switch and prefetch.
            for name, p in module.named_parameters():
                gpu_data = torch.empty_strided(size=p.data.size(),
                                                stride=p.data.stride(),
                                                dtype=p.data.dtype,
                                                layout=p.data.layout,
                                                device=self.device)
                gpu_data.copy_(p.data)
                self.gpu_buffer.parameters[name] = gpu_data
        
            
        # Create a CPU buffer and copy the parameters of this module layer to the CPU buffer (pinned memory)
        pin_memory = is_pin_memory_available()
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
            self.cpu_buffer[layer_idx].parameters[name] = cpu_data
            
        # create a wrapper forward function for the module
        def forward(*args, **kwargs):
            return self.wraper_forward(module, args=args, kwargs=kwargs)
        
        # Update the original forward and wrapper forward function pointer for the module
        self.original_forwards[layer_idx] = module.forward
        module.forward = forward 
        self.wraper_forwards[layer_idx] = module.forward

        return module
    
    def maybe_offload(self, module: torch.nn.Module, layer_idx: int, start_layer: int, end_layer: int):
        # We will offload the module layers in reverse order
        if layer_idx in self.initial_offload_layers:
            if layer_idx == start_layer:
                # the first layer should be prefetched to the GPU buffer during the model weights loading
                with torch.cuda.stream(self.data_mv_stream):
                    for name, p in module.named_parameters():
                        gpu_copy = self.gpu_buffer.parameters[name]
                        gpu_copy.copy_(p.data) # synchronous copy the data to the GPU buffer
                        p.data = gpu_copy
                    self.prefetch_event.record(self.data_mv_stream)
                self.prefetched_modules.append(module)
                # wait for the prefetching to finish data movement
                self.prefetch_event.synchronize()
                for name, p in module.named_parameters():
                    p.data = self.gpu_buffer.parameters[name]
                logger.debug(f"OffloadBuffer.maybe_offload: layer_idx={layer_idx} put to ${self.device}")
            else:
                # this layer should be on the CPU after model weights loading
                self.offloaded_modules.append(module)
                logger.debug(f"OffloadBuffer.maybe_offload: layer_idx={layer_idx} put to CPU")
        else:
            # this layer should be on the GPU after model weights loading
            for name, p in module.named_parameters():
                gpu_data = torch.empty_strided(size=p.data.size(),
                                            stride=p.data.stride(),
                                            dtype=p.data.dtype,
                                            layout=p.data.layout,
                                            device=self.device)
                gpu_data.copy_(p.data)
                p.data = gpu_data 
            logger.debug(f"OffloadBuffer.maybe_offload: layer_idx={layer_idx} put to ${self.device}")
        return module
        
    
    @nvtx_range("OffloadBuffer.wraper_forward")
    def wraper_forward(self, module: torch.nn.Module, args=None, kwargs=None):
        # intercept the original forward function
        current_device = next(module.parameters()).device
        cur_layer_idx = module.layer_idx
        logger.debug(f"OffloadBuffer.wraper_forward on layer {cur_layer_idx} on device {current_device}")
        
        assert self.prefetched_modules, "The prefetched_modules should not be empty"
        assert self.offloaded_modules, "The offloaded_modules should not be empty"
        # Step 1: check if the current layer is on the top of the prefetch queue, 
        # if yes, we need to wait for the prefetching to finish data movement.
        # if no, it means the layer is already on the GPU device.
        last_prefetch_module = self.prefetched_modules[0]
        if cur_layer_idx == last_prefetch_module.layer_idx:
            # wait for the prefetching to finish data movement
            self.prefetch_event.synchronize()
            # update the data of the top module in the prefetched_modules
            for name, p in last_prefetch_module.named_parameters():
                p.data = self.gpu_buffer.parameters[name]
                
        # Step 2: check if we need to trigger the prefetching
        if cur_layer_idx == last_prefetch_module.layer_idx + 1:
            # Step 1: call the compute_event synchronize to make sure the last layer has finished computation
            # sicne prefetching will overrite the GPU buffer.
            self.compute_event.synchronize()
            # step 1: offload the last prefeteched layer stored on the GPU buffer to the CPU buffer
            module_to_offload = self.prefetched_modules.popleft()
            for name, p in module_to_offload.named_parameters():
                p.data = self.cpu_buffer[module_to_offload.layer_idx].parameters[name]
            self.offloaded_modules.append(module_to_offload)  
            
            # step 2: prefetch the next offloaded layer to the GPU buffer using the data movement stream
            module_to_prefetch = self.offloaded_modules.popleft()
            with torch.cuda.stream(self.data_mv_stream):
                for name, p in module_to_prefetch.named_parameters():
                    gpu_copy = self.gpu_buffer.parameters[name]
                    gpu_copy.copy_(p.data)
                self.prefetch_event.record(self.data_mv_stream)
            self.prefetched_modules.append(module_to_prefetch)
            logger.debug(f"OffloadBuffer.wraper_forward: prefetch is triggered, current layer {cur_layer_idx}, offload layer {module_to_offload.layer_idx}, " 
                         f"prefetch layer {module_to_prefetch.layer_idx}")
            
        # perform the forward function using the original forward function
        module.forward = self.original_forwards[module.layer_idx]
        output = functional_call(module,
                                module.state_dict(),
                                args=args,
                                kwargs=kwargs)
        # restore the forward function to the wrapper forward function
        module.forward = self.wraper_forwards[module.layer_idx]
        
        # Step 3: check if the current layer is the last layer and it is the last prefetched layer
        # if yes, print a warning message to indicate the prefetch of the start layer is triggered, will will delay 
        # the computation of the current iteration.
        if cur_layer_idx == self.end_layer - 1 and cur_layer_idx == last_prefetch_module.layer_idx:
            logger.warning(f"[Warn] OffloadBuffer.wraper_forward: current layer {cur_layer_idx} is the last layer and the last prefetched layer, "
                           f"the prefetch of the start layer will be triggered, which will delay the computation of the current iteration.")
            # Step 1: call synchronize to wait the computation of the current layer to finish
            self.compute_event.synchronize()
            # Step 2: prefetch the layer 0 to the GPU buffer
            # step 1: offload the current layer stored on the GPU buffer
            module_to_offload = self.prefetched_modules.popleft()
            for name, p in module_to_offload.named_parameters():
                p.data = self.cpu_buffer[module_to_offload.layer_idx].parameters[name]
            self.offloaded_modules.append(module_to_offload) 
            # step 2: prefetch the next offloaded layer to the GPU buffer
            module_to_prefetch = self.offloaded_modules.popleft()
            with torch.cuda.stream(self.data_mv_stream):
                for name, p in module_to_prefetch.named_parameters():
                    gpu_copy = self.gpu_buffer.parameters[name]
                    gpu_copy.copy_(p.data)
                    # p.data = gpu_copy
                self.prefetch_event.record(self.data_mv_stream)
            self.prefetched_modules.append(module_to_prefetch)
            logger.debug(f"OffloadBuffer.wraper_forward: prefetch is triggered, current layer {cur_layer_idx}, offload layer {module_to_offload.layer_idx}, " 
                         f"prefetch layer {module_to_prefetch.layer_idx}")
        return output
