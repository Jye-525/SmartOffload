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
    
class OffloadBuffer:
    def __init__(self, k: int, param_offload_target:str):
        self.init_k = k 
        self.param_offload_target = param_offload_target
        self.nlayers = 0 # record the number of layers in this pipeline rank, will be updated in create_module
        self.start_layer = -1 # record the start layer in this pipeline rank, will be updated in create_module
        self.end_layer = -1 # record the last layer in this pipeline rank, will be updated in create_module
        self.cpu_buffers = None # the copy of all the transformer block layers.
        self.curr_k_value = self.init_k # Question: Do we need to set this to the initial K?
        self.resident_gpu_buffers = {} # store the `self.nlayers - self.nlayers // k` transformer blocks statically on the GPU.
        self.prefetch_events = {}   # to record prefetch completions.
        self.recording_events = {}
        self.dynamic_gpu_buffers = None # Assume the model has uniform trabsformer block structure. (A single buffer for prefetch) 
        self.prefetch_buffer_index = 0 # the index of the prefetch buffer
        
        self.original_forwards = []
        self.wraper_forwards = []
        self.offloaded_modules = deque()
        self.prefetched_modules = deque()
        # self.prefetch_event = torch.cuda.Event()
        self.compute_event = torch.cuda.Event()
        self.data_mv_stream = torch.cuda.Stream() # create a new stream for H2D data transfer
        self.layers = {} # store the transformer block layers between (start_layer, end_layer)
        logger.debug(f"OffloadBuffer.__init__: init_k={self.init_k}")
    
    def __initialization(self, start_layer: int, end_layer: int):
        self.nlayers = end_layer - start_layer
        self.start_layer = start_layer
        self.end_layer = end_layer
        # initialize the cpu_buffer (the buffer on host memory)
        self.cpu_buffers = [PerLayerParameters() for _ in range(self.nlayers)] 
        self.dynamic_gpu_buffers = [PerLayerParameters() for _ in range(2)] # one buffer for prefetch and one buffer for active compute
        
        self.prefetch_events = {i: torch.cuda.Event() for i in range(self.nlayers)}
        self.recording_events = {i: False for i in range(self.nlayers)}
        self.original_forwards = [None for _ in range(self.nlayers)]
        self.wraper_forwards = [None for _ in range(self.nlayers)]
        logger.debug(f"OffloadBuffer.__initialization: start_layer={start_layer}, end_layer={end_layer}, init_k={self.init_k}")
    
    ### Question 2: When do we call this function? at the beginning of each forward pass?
    ###  - This part adds some overhead  
    def reorganize_resident_gpu_modules(self, k: int):
        if self.curr_k_value == k:
            return
        logger.debug(f"OffloadBuffer.reorganize_resident_gpu_modules: k is changed from stride size of {self.curr_k_value} to stride size of {k}, need to reorganize the layers. current_resident_gpu_buffers = {self.resident_gpu_buffers.keys()}")
        self.curr_k_value = k
        self.resident_gpu_buffers = {} # deallocate all the previously allocated residents from GPU.

        # Step 1: check if the layer in the dynamic buffer is the start layer and the kth layer
        #  - If no, we need to prefetch the start layer and the kth layer to the dynamic gpu buffer
        t_start = time.time_ns()
        with torch.cuda.stream(self.data_mv_stream):
            # check which if the required start_layer and kth layer is already in the dynamic gpu buffer
            cur_num_prefetched_modules = len(self.prefetched_modules)
            matched_info = []
            for i in range(cur_num_prefetched_modules):
                if self.prefetched_modules[i].layer_idx == self.start_layer:
                    # this layer is already in the dynamic gpu buffer
                    matched_info.append((self.start_layer, i))
                elif self.prefetched_modules[i].layer_idx == self.start_layer + k:
                    # this layer is already in the dynamic gpu buffer
                    matched_info.append((self.start_layer + k, i))
            
            if len(matched_info) == 2:
                # both the start layer and the kth layer are already in the dynamic gpu buffer
                logger.debug(f"OffloadBuffer.reorganize_resident_gpu_modules: both the start layer and the kth layer are already in the dynamic gpu buffer, no need to prefetch.")
            elif len(matched_info) == 1:
                tt_start = time.time_ns()
                # one of the start layer or the kth layer is already in the dynamic gpu buffer
                to_prefetch_buffer_idx = matched_info[0][0]^1 
                # prefetch the other one from the cpu buffer to the corresponding dynamic gpu buffer
                if matched_info[0][0] == self.start_layer:
                    # the kth layer is already in the dynamic gpu buffer
                    for name, p in self.layers[k].named_parameters():
                        gpu_copy = self.dynamic_gpu_buffers[to_prefetch_buffer_idx].parameters[name]
                        # gpu_copy.copy_(self.cpu_buffers[0].parameters[name].data)
                        gpu_copy.copy_(p.data, non_blocking=True)
                        # p.data = gpu_copy
                    self.prefetch_events[k].record(self.data_mv_stream)
                    self.recording_events[k] = True
                    if cur_num_prefetched_modules > 1:
                        self.prefetched_modules[to_prefetch_buffer_idx] = self.layers[k]
                    else:
                        self.prefetched_modules.append(self.layers[k])
                else:
                    # the kth layer is already in the dynamic gpu buffer
                    for name, p in self.layers[0].named_parameters():
                        gpu_copy = self.dynamic_gpu_buffers[to_prefetch_buffer_idx].parameters[name]
                        # gpu_copy.copy_(self.cpu_buffers[0].parameters[name].data)
                        gpu_copy.copy_(p.data, non_blocking=True)
                        # p.data = gpu_copy
                    self.prefetch_events[0].record(self.data_mv_stream)
                    self.recording_events[0] = True
                    if cur_num_prefetched_modules > 1:
                        self.prefetched_modules[to_prefetch_buffer_idx] = self.layers[0]
                    else:
                        self.prefetched_modules.append(self.layers[0])
                
                tt_end = time.time_ns() 
                logger.debug(f"OffloadBuffer.reorganize_resident_gpu_modules: one of the start layer and the kth layer are already in the dynamic gpu buffer, no need to prefetch."
                             f", to_prefetch_buffer_idx={to_prefetch_buffer_idx}, matched_info={matched_info},"
                             f"dynbuf_0 device={next(iter(self.dynamic_gpu_buffers[0].parameters.values())).device}, "
                             f"a_device={next(self.prefetched_modules[0].parameters()).device}, "
                             f"prefetch one layer cost {((tt_end - t_start) / 1e6):.3f} ms, ")
                
            else:
                self.prefetched_modules.clear()
                # both the start layer and the kth layer are not in the dynamic gpu buffer
                tt_start = time.time_ns()
                self.prefetch_buffer_index = 0 # reset the prefetch buffer index
                for name, p in self.layers[0].named_parameters():
                    gpu_copy = self.dynamic_gpu_buffers[self.prefetch_buffer_index].parameters[name]
                    # gpu_copy.copy_(self.cpu_buffers[0].parameters[name].data)
                    gpu_copy.copy_(p.data, non_blocking=True)
                    # p.data = gpu_copy
                self.prefetch_events[0].record(self.data_mv_stream)
                self.recording_events[0] = True
                self.prefetched_modules.append(self.layers[0])
                
                for name, p in self.layers[k].named_parameters():
                    gpu_copy = self.dynamic_gpu_buffers[self.prefetch_buffer_index].parameters[name]
                    # gpu_copy.copy_(self.cpu_buffers[k].parameters[name].data)
                    gpu_copy.copy_(p.data, non_blocking=True)
                    # p.data = gpu_copy
                self.prefetch_events[k].record(self.data_mv_stream)
                self.recording_events[k] = True
                self.prefetched_modules.append(self.layers[k])
                
                tt_end = time.time_ns()
                logger.debug(f"OffloadBuffer.reorganize_resident_gpu_modules: both the start layer and the kth layer are not in the dynamic gpu buffer, "
                         f"prefetch them cost {((tt_end - tt_start) / 1e6):.3f} ms")
            
        t_end1 = time.time_ns() 
        
        # organize the layers resides on the CPU and GPU
        self.offloaded_modules.clear()
        time_lists = []
        with torch.cuda.stream(self.data_mv_stream):
            for i in range(1, self.nlayers):
                ttt_start = time.time_ns()
                if i % k != 0:
                    # 1. Check if layer[i] is on the GPU device
                    #   - If yes, we just need to put that buffer to the resident_gpu_buffers
                    #   - If no, we create a gpu buffer and copy data from the cpu buffer to the gpu buffer
                    if next(self.layers[i].parameters()).device == torch.device("cpu"):
                        self.resident_gpu_buffers[i] = PerLayerParameters() 
                        for name, p in self.layers[i].named_parameters():
                            self.resident_gpu_buffers[i].parameters[name] = torch.empty_strided(size=p.data.size(),
                                                        stride=p.data.stride(),
                                                        dtype=p.data.dtype,
                                                        layout=p.data.layout,
                                                        device=self.device)
                            self.resident_gpu_buffers[i].parameters[name].copy_(p.data, non_blocking=True)
                            # p.data = self.resident_gpu_buffers[i].parameters[name]
                        self.prefetch_events[i].record(self.data_mv_stream)
                        self.recording_events[i] = True
                    else:
                        # this layer is already on the GPU.
                        self.resident_gpu_buffers[i] = PerLayerParameters()
                        for name, p in self.layers[i].named_parameters():
                            self.resident_gpu_buffers[i].parameters[name] = p.data
                        self.recording_events[i] = False
                else:
                    # this layer should be on the CPU after reorganize
                    if i != 0 and i != k:
                        for name, p in self.layers[i].named_parameters():
                            p.data = self.cpu_buffers[i].parameters[name]
                        self.offloaded_modules.append(self.layers[i])
                ttt_end = time.time_ns()
                time_lists.append(f'{(ttt_end - ttt_start) / 1e6:.3f}')
        
        t_end2 = time.time_ns()
            
        logger.debug(f"Reorganizing the resident GPU modules to stride size of {k}, self.resident_gpu_buffers.keys() = {self.resident_gpu_buffers.keys()}, "
                     f"prefetech to dynamic gpu buffer cost {((t_end1 - t_start) / 1e6):.3f} ms, "
                     f"reorganizing the resident GPU modules cost {((t_end2 - t_start) / 1e6):.3f} ms"
                     f", time_lists={time_lists}")

        
    def create_module(self, module: torch.nn.Module, layer_idx: int, start_layer: int, end_layer: int):
        # get the original device of the module
        self.device = next(module.parameters()).device 
        if self.device == torch.device("cpu"):
            return module
        
        assert self.init_k > 0, "The init_k should be greater than 0 when smart offload is enabled"
        
        if self.nlayers == 0:
            # initialize and update parameters used in this pipeline rank
            self.__initialization(start_layer, end_layer)
        
        inter_layer_idx = layer_idx - start_layer # used for indexing various buffers and envents
        logger.debug(f"OffloadBuffer.create_module: layer_idx={layer_idx}, start_layer={start_layer}, end_layer={end_layer}, inter_layer_idx={inter_layer_idx}, device={self.device}")
        
        if layer_idx == start_layer or layer_idx == start_layer + self.curr_k_value:
            # Put the first layer (i.e., start_layer) and the kth layer (i.e., start_layer + k) to the dynamic GPU buffer
            logger.debug(f"OffloadBuffer.create_module: layer_idx={layer_idx} will load to {self.device}, prefetch_buffer_index={self.prefetch_buffer_index}")
            for name, p in module.named_parameters():
                # reuse the p.data space, we don't need to create a new tensor
                self.dynamic_gpu_buffers[self.prefetch_buffer_index].parameters[name] = p.data
            if self.curr_k_value < end_layer:
                # only one layer needs to be prefetched, so we don't need to switch the prefetch buffer index
                self.prefetch_buffer_index ^= 1 # switch the prefetch buffer index
            else:
                # create a gpu buffer for slot 1 in dynamic_gpu buffer
                for name, p in module.named_parameters():
                    gpu_data = torch.empty_strided(size=p.data.size(),
                                            stride=p.data.stride(),
                                            dtype=p.data.dtype,
                                            layout=p.data.layout,
                                            device=self.device)
                    gpu_data.copy_(p.data) # if delete this line.
                    # reuse the p.data space, we don't need to create a new tensor
                    self.dynamic_gpu_buffers[self.prefetch_buffer_index^1].parameters[name] = gpu_data 
            
            logger.debug(f"=====After OffloadBuffer.create_module: layer_idx={layer_idx} will load to {self.device}, prefetch_buffer_index={self.prefetch_buffer_index}")
            
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
            self.cpu_buffers[inter_layer_idx].parameters[name] = cpu_data
            
        # create a wrapper forward function for the module
        def forward(*args, **kwargs):
            return self.wraper_forward(module, args=args, kwargs=kwargs)
        
        # Update the original forward and wrapper forward function pointer for the module
        self.original_forwards[inter_layer_idx] = module.forward
        module.forward = forward 
        self.wraper_forwards[inter_layer_idx] = module.forward

        # save the transformer block layer in the layers dict
        self.layers[layer_idx] = module
        
        return module
    
    def maybe_offload(self, module: torch.nn.Module, layer_idx: int, start_layer: int, end_layer: int):
        # We will offload the module layers in reverse order
        inner_layer_idx = layer_idx - start_layer
        # Step 1: check if current layer is an offload layer
        if inner_layer_idx % self.curr_k_value == 0:
            # the current layer is an offloaded layer
            if layer_idx == start_layer or layer_idx == start_layer + self.curr_k_value:
                # Only load the first layer (i.e., start_layer) and the kth layer (i.e., start_layer + k) to the dynamic GPU buffer
                logger.debug(f"OffloadBuffer.maybe_offload: layer_idx={layer_idx} put to {self.device}, inner_layer_idx={inner_layer_idx}, prefetch_buffer_index={self.prefetch_buffer_index}")
                with torch.cuda.stream(self.data_mv_stream):
                    for name, p in module.named_parameters():
                        gpu_copy = self.dynamic_gpu_buffers[self.prefetch_buffer_index].parameters[name] 
                        gpu_copy.copy_(p.data, non_blocking=True) # synchronous copy the data to the GPU buffer
                        # p.data = gpu_copy
                    self.prefetch_events[inner_layer_idx].record(self.data_mv_stream)
                    self.recording_events[inner_layer_idx] = True
                self.prefetched_modules.append(module)
                
                # wait for the prefetching to finish data movement
                self.prefetch_events[inner_layer_idx].synchronize()
                self.recording_events[inner_layer_idx] = False 
                for name, p in module.named_parameters():
                    p.data = self.dynamic_gpu_buffers[self.prefetch_buffer_index].parameters[name]
                logger.debug(f"OffloadBuffer.maybe_offload: layer_idx={layer_idx} put to {self.device} module device={next(module.parameters()).device}, inner_layer_idx={inner_layer_idx}, prefetch_buffer_index={self.prefetch_buffer_index}")
                if self.curr_k_value < end_layer:
                    # only one layer needs to be prefetched, so we don't need to switch the prefetch buffer index
                    self.prefetch_buffer_index ^= 1
            else:
                # this layer should be on the CPU after model weights loading
                self.offloaded_modules.append(module)
                logger.debug(f"OffloadBuffer.maybe_offload: layer_idx={layer_idx} put to CPU")
        else:
            # Step2: this layer should be put on GPU after model weights loading
            self.resident_gpu_buffers[layer_idx] = PerLayerParameters()
            for name, p in module.named_parameters():
                gpu_data = torch.empty_strided(size=p.data.size(),
                                            stride=p.data.stride(),
                                            dtype=p.data.dtype,
                                            layout=p.data.layout,
                                            device=self.device)
                gpu_data.copy_(p.data)
                p.data = gpu_data
                self.resident_gpu_buffers[layer_idx].parameters[name] = gpu_data
                self.recording_events[layer_idx] = False
            logger.debug(f"OffloadBuffer.maybe_offload: layer_idx={layer_idx} put to {self.device}, module device={next(module.parameters()).device}, self.layer[layer_idx].device={next(self.layers[layer_idx].parameters()).device}")
        return module
        
    
    @nvtx_range("OffloadBuffer.wraper_forward")
    def wraper_forward(self, module: torch.nn.Module, args=None, kwargs=None):
        # intercept the original forward function
        current_device = next(module.parameters()).device
        cur_layer_idx = module.layer_idx
        
        assert self.prefetched_modules, "The prefetched_modules should not be empty"
        # assert self.offloaded_modules, "The offloaded_modules should not be empty"
        # Step 1: check if the current layer is on the top of the prefetch queue, 
        # if yes, we need to wait for the prefetching to finish data movement.
        # if no, it means the layer is already on the GPU device.
        last_prefetch_module = self.prefetched_modules[0]
        logger.debug(f"OffloadBuffer.wraper_forward: current_layer {cur_layer_idx}, last_prefetch_module.layer_idx={last_prefetch_module.layer_idx}, "
                     f"resident_gpu_buffers.keys = {self.resident_gpu_buffers.keys()}, recording_events[{cur_layer_idx - self.start_layer}]={self.recording_events[cur_layer_idx - self.start_layer]}")
        
        if self.recording_events[cur_layer_idx - self.start_layer]:
            tt_start = time.time_ns()
            self.prefetch_events[cur_layer_idx - self.start_layer].synchronize()
            self.recording_events[cur_layer_idx - self.start_layer] = False
            
            if cur_layer_idx in self.resident_gpu_buffers:
                for name, p in module.named_parameters():
                    p.data = self.resident_gpu_buffers[cur_layer_idx].parameters[name]
            else:
                if cur_layer_idx == last_prefetch_module.layer_idx:
                    for name, p in module.named_parameters():
                        p.data = self.dynamic_gpu_buffers[self.prefetch_buffer_index].parameters[name]
                else:
                    for name, p in module.named_parameters():
                        p.data = self.dynamic_gpu_buffers[self.prefetch_buffer_index^1].parameters[name] 
            tt_end = time.time_ns()
            logger.debug(f"OffloadBuffer.wraper_forward: layer {cur_layer_idx} finished data movement, current device={next(module.parameters()).device}, "
                         f"module device={next(module.parameters()).device}, "
                         f"cost {((tt_end - tt_start) / 1e6):.3f} ms")
        
        # if cur_layer_idx == last_prefetch_module.layer_idx:
        #     # logger.debug(f"OffloadBuffer.wraper_forward: layer {cur_layer_idx} prefetched from cpu haven't finished... current device={next(module.parameters()).device}, module device={next(last_prefetch_module.parameters()).device}, event idx={cur_layer_idx - self.start_layer}")
        #     # wait for the prefetching to finish data movement
        #     tt_start = time.time_ns()
        #     self.prefetch_events[cur_layer_idx - self.start_layer].synchronize()
        #     self.recording_events[cur_layer_idx - self.start_layer] = False
        #     # update the data of the top module in the prefetched_modules
        #     for name, p in last_prefetch_module.named_parameters():
        #         # p.data = self.gpu_buffer.parameters[name]
        #         p.data = self.dynamic_gpu_buffers[self.prefetch_buffer_index].parameters[name]
        #     tt_end = time.time_ns()
        #     logger.debug(f"OffloadBuffer.wraper_forward: layer {cur_layer_idx} fininshed preftech, prefetch_idx={self.prefetch_buffer_index}, current device={next(module.parameters()).device}, "
        #                  f"last_prefetch_module device={next(last_prefetch_module.parameters()).device},"
        #                  f"prefetch time cost {((tt_end - tt_start) / 1e6):.3f} ms")
        
        # # Step 2: check if the current layer involve data movement
        # if self.recording_events[cur_layer_idx - self.start_layer]:
        #     tt_start = time.time_ns()
        #     self.prefetch_events[cur_layer_idx - self.start_layer].synchronize()
        #     self.recording_events[cur_layer_idx - self.start_layer] = False
        #     if cur_layer_idx in self.resident_gpu_buffers:
        #         for name, p in module.named_parameters():
        #             p.data = self.resident_gpu_buffers[cur_layer_idx].parameters[name]
        #     else:
        #         # this layer is on the dynamic gpu buffer
        #         for name, p in module.named_parameters():
        #             p.data = self.dynamic_gpu_buffers[self.prefetch_buffer_index^1].parameters[name]
        #     tt_end = time.time_ns()
        #     logger.debug(f"OffloadBuffer.wraper_forward: layer {cur_layer_idx} finished data movement, current device={next(module.parameters()).device}, "
        #                  f"cost {((tt_end - tt_start) / 1e6):.3f} ms")
                
        # Step 2: check if we need to trigger the prefetching
        if cur_layer_idx == last_prefetch_module.layer_idx + 1:
            # Step 1: call the compute_event synchronize to make sure the last layer has finished computation
            # sicne prefetching will overrite the GPU buffer.
            self.compute_event.synchronize()
            if len(self.offloaded_modules) == 0:
                # the offloaded_modules is empty, there is no prefetch since all the layers are on the GPU
                logger.debug(f"OffloadBuffer.wraper_forward: No prefetching is triggered, since less than 2 layers are offloaded and result in all layers will be on GPU.") 
            else:
                # Prefetch or offload will happen
                module_to_offload = self.prefetched_modules.popleft()
                # step 1: offload the last prefeteched layer stored on the GPU buffer to the CPU buffer
                for name, p in module_to_offload.named_parameters():
                    p.data = self.cpu_buffers[module_to_offload.layer_idx - self.start_layer].parameters[name]
                self.offloaded_modules.append(module_to_offload)
                logger.debug(f"OffloadBuffer.wraper_forward: offloading layer {module_to_offload.layer_idx} to CPU buffer index {module_to_offload.layer_idx - self.start_layer}, "
                            f"module_to_offload device={next(module_to_offload.parameters()).device}, offload layer {module_to_offload.layer_idx}")  
                
                # step 2: prefetch the next offloaded layer to the GPU buffer using the data movement stream
                try:
                    module_to_prefetch = self.offloaded_modules.popleft()
                    logger.debug(f"OffloadBuffer.wraper_forward: prefetching layer {module_to_prefetch.layer_idx} to GPU buffer index {self.prefetch_buffer_index}, module to prefetch device={next(module_to_prefetch.parameters()).device}, ")
                    with torch.cuda.stream(self.data_mv_stream):
                        for name, p in module_to_prefetch.named_parameters():
                            gpu_copy = self.dynamic_gpu_buffers[self.prefetch_buffer_index].parameters[name]
                            gpu_copy.copy_(p.data, non_blocking=True)
                            # p.data = gpu_copy
                        self.prefetch_events[module_to_prefetch.layer_idx - self.start_layer].record(self.data_mv_stream)
                        self.recording_events[module_to_prefetch.layer_idx - self.start_layer] = True
                    self.prefetched_modules.append(module_to_prefetch)
                    logger.debug(f"OffloadBuffer.wraper_forward: prefetching layer {module_to_prefetch.layer_idx} to GPU buffer index {self.prefetch_buffer_index}, current layer {cur_layer_idx}, offload layer {module_to_offload.layer_idx}, " 
                                f"prefetch_events index {module_to_prefetch.layer_idx - self.start_layer}, "
                                f"module_to_prefetch device={next(module_to_prefetch.parameters()).device}, offload layer {module_to_prefetch.layer_idx}")
                    self.prefetch_buffer_index ^= 1 # switch the prefetch buffer index
                except Exception as e:
                    logger.error(f"OffloadBuffer.wraper_forward: prefetch is triggered, but the offloaded_modules is empty, current layer {cur_layer_idx}, "
                                f"offload layer {module_to_offload.layer_idx}, exception: {e}")
                    import pdb; pdb.set_trace()
             
        # perform the forward function using the original forward function
        logger.debug(f"OffloadBuffer.wraper_forward: layer {module.layer_idx} start its foward computation on {next(module.parameters()).device}...")
        module.forward = self.original_forwards[module.layer_idx - self.start_layer]
        output = functional_call(module,
                                module.state_dict(),
                                args=args,
                                kwargs=kwargs)
        # restore the forward function to the wrapper forward function
        module.forward = self.wraper_forwards[module.layer_idx - self.start_layer]
        logger.debug(f"OffloadBuffer.wraper_forward: layer {module.layer_idx} finsih its foward computation on {next(module.parameters()).device}, last_prefetch_module.layer_idx={last_prefetch_module.layer_idx}")
        
        # Step 3: check if the current layer is the last layer and it is the last prefetched layer
        # if yes, print a warning message to indicate the prefetch of the start layer is triggered, will will delay 
        # the computation of the current iteration.
        if cur_layer_idx == self.end_layer - 1 and cur_layer_idx == last_prefetch_module.layer_idx + self.curr_k_value:
            logger.warning(f"[Warn] OffloadBuffer.wraper_forward: current layer {cur_layer_idx} is the last layer and the last - 1 is prefetched layer, "
                           f"the prefetch of the kth layer will be triggered, which will delay the computation of the current iteration.")
            # Step 1: call synchronize to wait the computation of the current layer to finish
            self.compute_event.synchronize()
            if len(self.offloaded_modules) != 0:
                # Step 2: prefetch the layer 0 to the GPU buffer
                # step 1: offload the current layer stored on the GPU buffer
                module_to_offload = self.prefetched_modules.popleft()
                for name, p in module_to_offload.named_parameters():
                    p.data = self.cpu_buffers[module_to_offload.layer_idx - self.start_layer].parameters[name]
                self.offloaded_modules.append(module_to_offload) 
                # step 2: prefetch the next offloaded layer to the GPU buffer
                module_to_prefetch = self.offloaded_modules.popleft()
                with torch.cuda.stream(self.data_mv_stream):
                    for name, p in module_to_prefetch.named_parameters():
                        gpu_copy = self.dynamic_gpu_buffers[self.prefetch_buffer_index].parameters[name]
                        gpu_copy.copy_(p.data, non_blocking=True)
                        # p.data = gpu_copy
                    self.prefetch_events[module_to_prefetch.layer_idx - self.start_layer].record(self.data_mv_stream)
                    self.recording_events[module_to_prefetch.layer_idx - self.start_layer] = True
                self.prefetched_modules.append(module_to_prefetch)
                logger.debug(f"Warn-OffloadBuffer.wraper_forward: prefetch layer {module_to_prefetch.layer_idx} to GPU buffer index {self.prefetch_buffer_index}, current layer {cur_layer_idx}, offload layer {module_to_offload.layer_idx}")
                self.prefetch_buffer_index ^= 1 # switch the prefetch buffer index
        return output
    
    def get_offload_interval(self):
        return self.curr_k_value
    
    
# NOTES:
# 1. We use two dynamic GPU buffers to store the prefetching and computing operations, which are used to switch between transfers and computations alternately.
# 2. Use a generic prefetch_events to record the prefetching events, and a generic recording_events to check if that element is being prefetched or not (only if it is being prefetched, check prefetch_events[i].sycnhronize()).
# 3. Use a static_cpu_buffers to permanently store the CPU buffers. This way you can always lookup the CPU based buffer in the static_cpu_buffers datastructure.
# 4. Use a dynamic_gpu_buffers to store the GPU buffers, which are used to switch between transfers and computations alternately.
# 5. Use a resident_gpu_buffers to store the `k` transformer blocks on the GPU. The resident_gpu_buffer will be dynamically allocated at every forward pass using `reorganize_resident_gpu_modules`.
# 6. Remove the `initial_offload_layers`, assume that we do equi-spaced offloading, i.e,. offload every `k` layers.
#    6.1 replace with initial_offload_layers with k
