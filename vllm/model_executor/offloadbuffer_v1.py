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
        self.curr_k_value = k 
        self.param_offload_target = param_offload_target
        
        self.nlayers = 0 # record the number of layers in this pipeline rank, will be updated in create_module
        self.start_layer = -1 # record the start layer in this pipeline rank, will be updated in create_module
        self.end_layer = -1 # record the last layer in this pipeline rank, will be updated in create_module
        self.cpu_buffers = None # the copy of all the transformer block layers.
        self.dynamic_gpu_buffers = None # Assume the model has uniform trabsformer block structure. (A single buffer for prefetch) 
        self.prefetch_buffer_index = 0 # the index of the prefetch buffer
        self.resident_gpu_buffers = {} # store the `self.nlayers - self.nlayers // k` transformer blocks statically on the GPU.
        
        self.prefetch_events = []   # to record prefetch the prefetch events
        self.recording_events = []  # to check if the layer is being prefetched or not
        
        self.compute_event = torch.cuda.Event()
        self.data_mv_stream = torch.cuda.Stream() # create a new stream for H2D data transfer
        # self.data_mv_streams = [torch.cuda.Stream() for _ in range(2)]
        
        self.original_forwards = []
        self.wraper_forwards = []
        
        self.offloaded_modules = deque()
        self.prefetched_modules = deque()
        # self.prefetch_event = torch.cuda.Event()
        self.layers = [] # store the transformer block layers between (start_layer, end_layer)
        self.first_fwd_after_reorganize = True # used to check if this is the first forward pass after initialization
        self.H2D_transfer_times = []
        logger.debug(f"OffloadBuffer.__init__: init_k={self.curr_k_value}")
    
    def __initialization(self, start_layer: int, end_layer: int, module: torch.nn.Module):
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.nlayers = end_layer - start_layer
        # initialize the cpu_buffer (the buffer on host memory)
        self.cpu_buffers = [PerLayerParameters() for _ in range(self.nlayers)] 
        # (TODO) Question: Can we make the dynamic_gpu_buffers number dynamic?
        #  - If nlayers // cur_key_value > 2, we crate a duble buffer for prefetching
        #  - If nlayers // cur_key_value <= 2, we create a single buffer for prefetching
        self.dynamic_gpu_buffers = [PerLayerParameters() for _ in range(2)]
        params_dict = dict(module.named_parameters()) # Assume that each transformer block module has same parameter names 
        for buffer in self.dynamic_gpu_buffers:
            for name, p in params_dict.items():
                gpu_data = torch.empty_strided(size=p.data.size(),
                                            stride=p.data.stride(),
                                            dtype=p.data.dtype,
                                            layout=p.data.layout,
                                            device='cuda')
                gpu_data.copy_(p.data)
                buffer.parameters[name] = gpu_data
        
        self.prefetch_events = [torch.cuda.Event() for _ in range(self.nlayers)]
        self.recording_events = [False for _ in range(self.nlayers)]
        self.original_forwards = [None for _ in range(self.nlayers)]
        self.wraper_forwards = [None for _ in range(self.nlayers)]
        self.layers = [None for _ in range(self.nlayers)]
        self.total_offloaded_layers = self.nlayers // self.curr_k_value
        logger.debug(f"OffloadBuffer.__initialization: start_layer={start_layer}, end_layer={end_layer}, init_k={self.curr_k_value}, total_offloaded_layers={self.total_offloaded_layers}")
    
    ### Question 2: When do we call this function? at the beginning of each forward pass?
    ###  - This part adds some overhead
    @nvtx_range("OffloadBuffer.reorganize")  
    def reorganize_resident_gpu_modules(self, k: int):
        if self.curr_k_value == k:
            return
        old_k_value = self.curr_k_value
       
        t_start = time.time_ns()
        self.curr_k_value = k
        self.resident_gpu_buffers = {} # deallocate all the previously allocated residents from GPU.
        # Step 1: reset the prefetch buffer to make the modules in self.prefetched_modules points to the CPU buffers
        while self.prefetched_modules:
            module = self.prefetched_modules.popleft()
            layer_idx = module.layer_idx
            # Get references to avoid repeated lookups
            layer = self.layers[layer_idx]
            cpu_buffer = self.cpu_buffers[layer_idx - self.start_layer]
            for name, p in layer.named_parameters():
                p.data = cpu_buffer.parameters[name]
                
        self.prefetched_modules.clear()
        self.offloaded_modules.clear()
        self.prefetch_buffer_index = 0 # reset the prefetch buffer index
        times_per_layer = {}
        for layer_idx in range(self.start_layer, self.end_layer):
            tt_start = time.time_ns()
            if (layer_idx + 1) % self.curr_k_value == 0:
                # the layer should be offloaded to the CPU except the kth and 2kth layer
                if (layer_idx + 1 ) in [self.curr_k_value, 2 * self.curr_k_value]:
                    buffer_idx = (self.prefetch_buffer_index ^ 1 if (layer_idx + 1) == 2 * self.curr_k_value
                                    else self.prefetch_buffer_index)
                    self.__prefetch(self.layers[layer_idx], layer_idx - self.start_layer, buffer_idx, non_blocking=True)
                else:
                    # this layer should be on the CPU after reorganize
                    for name, p in self.layers[layer_idx].named_parameters():
                        p.data = self.cpu_buffers[layer_idx - self.start_layer].parameters[name]
                    self.offloaded_modules.append(self.layers[layer_idx])
            else:
                # these layers shold be reside on the GPU after reorganize
                if next(self.layers[layer_idx].parameters()).device == torch.device("cpu"):
                    ### cache nvtx to see if cuda free trigger before or after this.
                    with torch.cuda.stream(self.data_mv_stream):
                        with nvtx_range(f"reorganize_{layer_idx}_to_gpu"):
                            self.resident_gpu_buffers[layer_idx] = PerLayerParameters() 
                            for name, p in self.layers[layer_idx].named_parameters():
                                self.resident_gpu_buffers[layer_idx].parameters[name] = torch.empty_strided(size=p.data.size(),
                                                            stride=p.data.stride(),
                                                            dtype=p.data.dtype,
                                                            layout=p.data.layout,
                                                            device=self.device)
                                self.resident_gpu_buffers[layer_idx].parameters[name].copy_(p.data, non_blocking=True)
                                p.data = self.resident_gpu_buffers[layer_idx].parameters[name]
                    self.prefetch_events[layer_idx].record(self.data_mv_stream)
                    self.recording_events[layer_idx] = True
                else:
                    # this layer is already on the GPU.
                    self.resident_gpu_buffers[layer_idx] = PerLayerParameters()
                    for name, p in self.layers[layer_idx].named_parameters():
                        self.resident_gpu_buffers[layer_idx].parameters[name] = p.data
                    self.recording_events[layer_idx] = False
            tt_start = time.time_ns()
            times_per_layer[layer_idx] = (tt_start - t_start) / 1e6
        
        t_end = time.time_ns()
        self.first_fwd_after_reorganize = True # reset the first forward pass after reorganize
        self.total_offloaded_layers = self.nlayers // self.curr_k_value            
        
        logger.debug(f"Reorganizing the resident GPU modules from stride of {old_k_value} stride size of {self.curr_k_value}, "
                     f"resident_gpu_buffers layer ids = {self.resident_gpu_buffers.keys()}, "
                     f"reorganizing layers cost {((t_end - t_start) / 1e6):.3f} ms, "
                     f"per layer time = {[f'{k}={v:.3f}' for k, v in times_per_layer.items()]}, "
                     f"total_offloaded_layers={self.total_offloaded_layers}")
        
        
        
    def create_module(self, module: torch.nn.Module, layer_idx: int, start_layer: int, end_layer: int):
        # get the original device of the module
        self.device = next(module.parameters()).device 
        if self.device == torch.device("cpu"):
            return module
        
        assert self.curr_k_value > 0, "The init_k should be greater than 0 when smart offload is enabled"
        
        if self.nlayers == 0:
            # initialize the parameters used in this pipeline rank
            self.__initialization(start_layer, end_layer, module)
        
        inter_layer_idx = layer_idx - start_layer # used for indexing various buffers and envents
        
        # Step 1: Create a CPU buffer and copy the parameters of this module layer to the CPU buffer (pinned memory)
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

        # save the transformer block layer in the layers List
        self.layers[inter_layer_idx] = module
        
        return module
    
    def __prefetch(self, module: torch.nn.Module, inner_layer_idx: int, buffer_index: int, non_blocking: bool):
        # prefetch the kth layer to the dynamic GPU buffer slot 0
        with torch.cuda.stream(self.data_mv_stream):
            for name, p in module.named_parameters():
                gpu_copy = self.dynamic_gpu_buffers[buffer_index].parameters[name] 
                gpu_copy.copy_(p.data, non_blocking=non_blocking) # Asynchronous copy the data to the GPU buffer
                p.data = gpu_copy
            self.prefetch_events[inner_layer_idx].record(self.data_mv_stream)
            self.recording_events[inner_layer_idx] = True
        self.prefetched_modules.append(module)
    
    @nvtx_range("OffloadBuffer.maybe_offload")    
    def maybe_offload(self, module: torch.nn.Module, layer_idx: int, start_layer: int, end_layer: int):
        load_module_time = 0
        inner_layer_idx = layer_idx - start_layer
        # Step 1: check if current layer is an offload layer
        if (inner_layer_idx + 1) % self.curr_k_value == 0:
            tt_start = time.time_ns()
            # the current layer is an offloaded layer
            if (inner_layer_idx + 1) in [self.curr_k_value, 2 * self.curr_k_value]:
                buffer_idx = (self.prefetch_buffer_index ^ 1 
                                if (inner_layer_idx + 1) == 2 * self.curr_k_value 
                                else self.prefetch_buffer_index)
                
                # prefetch the kth and 2kth layer to the dynamic GPU buffer slot 0
                self.__prefetch(module, inner_layer_idx, buffer_idx, non_blocking=False)
                
                if self.curr_k_value == 1:
                     self.prefetch_events[inner_layer_idx].synchronize()
                     self.recording_events[inner_layer_idx] = False
                
                logger.debug(f"OffloadBuffer.maybe_offload: layer_idx={layer_idx} put to {self.device} module device={next(module.parameters()).device}, inner_layer_idx={inner_layer_idx}, prefetch_buffer_index={buffer_idx}") 
            # elif (inner_layer_idx + 1) == 2 * self.curr_k_value:
            #     # prefetch the 2kth layer to the dynamic GPU buffer slot 1
            #     with torch.cuda.stream(self.data_mv_stream):
            #         for name, p in module.named_parameters():
            #             gpu_copy = self.dynamic_gpu_buffers[self.prefetch_buffer_index^1].parameters[name] 
            #             gpu_copy.copy_(p.data, non_blocking=True)
            #             p.data = gpu_copy
            #         self.prefetch_events[inner_layer_idx].record(self.data_mv_stream)
            #         self.recording_events[inner_layer_idx] = True
            #     self.prefetched_modules.append(module)    
            #     logger.debug(f"OffloadBuffer.maybe_offload: layer_idx={layer_idx} put to {self.device} module device={next(module.parameters()).device}, inner_layer_idx={inner_layer_idx}, prefetch_buffer_index={self.prefetch_buffer_index^1}")
            else:
                # this layer should be on the CPU after model weights loading
                self.offloaded_modules.append(module)
                logger.debug(f"OffloadBuffer.maybe_offload: layer_idx={layer_idx} put to CPU. inner_layer_idx={inner_layer_idx}, module device={next(module.parameters()).device}")
            tt_end = time.time_ns()
            load_module_time = (tt_end - tt_start) / 1e6
        else:
            tt_start = time.time_ns()
            # Step2: this layer should be put on GPU after model weights loading
            # Copy data from CPU to GPU in synchronous mode using cuda default stream
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
                self.recording_events[inner_layer_idx] = False
                print(f"+++++++ Is gpu_data contiguous? {gpu_data.is_contiguous()}")
            tt_end = time.time_ns()
            load_module_time = (tt_end - tt_start) / 1e6
            self.H2D_transfer_times.append(load_module_time)
            
        logger.debug(f"OffloadBuffer.maybe_offload: layer_idx={layer_idx} put to {next(module.parameters()).device} cost {load_module_time:.3f} ms")
        return module
        
    
    @nvtx_range("OffloadBuffer.wraper_forward")
    def wraper_forward(self, module: torch.nn.Module, args=None, kwargs=None):
        # intercept the original forward function
        cur_layer_idx = module.layer_idx
        cur_inner_layer_idx = cur_layer_idx - self.start_layer
        
        tt_start = time.time_ns() 
        # Step 1: Always prefetech the kth, and 2kth layer to the dynamic GPU buffer at the first layer
        if cur_layer_idx == self.start_layer:
            if self.first_fwd_after_reorganize == True:
                self.first_fwd_after_reorganize = False
            elif self.first_fwd_after_reorganize == False and self.total_offloaded_layers > 2:
                self.prefetched_modules.clear()
                self.prefetch_buffer_index = 0 # reset the prefetch buffer index 
                # prefetch the kth and 2kth layer to the dynamic GPU buffer
                ttt_start_1 = time.time_ns()
                with torch.cuda.stream(self.data_mv_stream):
                    module_to_prefetch_k = self.offloaded_modules.popleft()
                    inner_k_idx = module_to_prefetch_k.layer_idx - self.start_layer
                    assert inner_k_idx + 1 == self.curr_k_value, f"OffloadBuffer.wraper_forward: the module to prefetch is not the kth layer, {module_to_prefetch_k.layer_idx + 1} != {self.start_layer + self.curr_k_value}, k={self.curr_k_value}"
                    buffer_idx = self.prefetch_buffer_index
                    for name, p in module_to_prefetch_k.named_parameters():
                        gpu_copy = self.dynamic_gpu_buffers[buffer_idx].parameters[name] 
                        gpu_copy.copy_(p.data, non_blocking=True)
                        p.data = gpu_copy
                    self.prefetch_events[inner_k_idx].record(self.data_mv_stream)
                    self.recording_events[inner_k_idx] = True
                    self.prefetched_modules.append(module_to_prefetch_k)
                    
                    module_to_prefetch_2k = self.offloaded_modules.popleft()
                    inner_2k_idx = module_to_prefetch_2k.layer_idx - self.start_layer
                    buffer_idx = self.prefetch_buffer_index^1
                    assert (inner_2k_idx) + 1 == 2 * self.curr_k_value, f"OffloadBuffer.wraper_forward: the module to prefetch is not the 2kth layer, {module_to_prefetch_2k.layer_idx} != {self.start_layer + 2 * self.curr_k_value}, k={self.curr_k_value}"
                    for name, p in module_to_prefetch_2k.named_parameters():
                        gpu_copy = self.dynamic_gpu_buffers[buffer_idx].parameters[name] 
                        gpu_copy.copy_(p.data, non_blocking=True)
                        p.data = gpu_copy
                    self.prefetch_events[inner_2k_idx].record(self.data_mv_stream)
                    self.recording_events[inner_2k_idx] = True
                    self.prefetched_modules.append(module_to_prefetch_2k) 
                    
                ttt_start_2 = time.time_ns()
                logger.debug(f"OffloadBuffer.wraper_forward: First layer to prefetch layer {inner_k_idx} and {inner_2k_idx} to the dynamic GPU buffer"
                             f", cost {(ttt_start_2 - ttt_start_1)/1e6:.3f} ms")       
        
        tt_end1 = time.time_ns()  
        # Step 2: check if the current layer is prefetching data
        if self.recording_events[cur_inner_layer_idx]:
            self.prefetch_events[cur_inner_layer_idx].synchronize()
            self.recording_events[cur_inner_layer_idx] = False
        tt_end2 = time.time_ns() 
        
        # Step 3: Check if the current layer is every (kth + 1) layer
        if ((cur_inner_layer_idx + 1) > self.curr_k_value and \
            cur_inner_layer_idx < self.nlayers):
    
            if ((self.curr_k_value == 1) or (cur_inner_layer_idx + 1) % self.curr_k_value == 1):
                # Step 2: current layer is right the layer after every kth layer
                # 2.1: Wait the previous computation to finish
                ttt_1 = time.time_ns()
                self.compute_event.synchronize()
                ttt_2 = time.time_ns()
                # 2.2: Check if need to prefetch the next kth layer
                ttt_4 = []
                if len(self.offloaded_modules) != 0:
                    ttt_4_1 = time.time_ns()
                    # offload the (current layer - 1) to the CPU buffer
                    module_to_offload = self.prefetched_modules.popleft()
                    assert module_to_offload.layer_idx == cur_layer_idx - 1, f"OffloadBuffer.wraper_forward: the module to offload is not the previous layer, {module_to_offload.layer_idx} != {cur_layer_idx - 1}"
                    for name, p in module_to_offload.named_parameters():
                        p.data = self.cpu_buffers[module_to_offload.layer_idx - self.start_layer].parameters[name]
                    self.offloaded_modules.append(module_to_offload)
                    ttt_4_2 = time.time_ns()
                    
                    # prefetch the next 2 * kth layer to the corresponding dynamic gpu buffer
                    prefetech_layer_idx = cur_layer_idx + 2 * self.curr_k_value - 1 
                    if prefetech_layer_idx < self.end_layer:
                        # only trigger preftech when the prefetech layer is less then the end layer
                        module_to_prefetch = self.offloaded_modules.popleft()
                        assert module_to_prefetch.layer_idx == prefetech_layer_idx, f"OffloadBuffer.wraper_forward: the module to prefetch is not the next kth layer, {module_to_prefetch.layer_idx} != {prefetech_layer_idx}, k={self.curr_k_value}"
                        self.__prefetch(module_to_prefetch, module_to_prefetch.layer_idx - self.start_layer, self.prefetch_buffer_index, non_blocking=True)
                        # with torch.cuda.stream(self.data_mv_stream):
                        #     for name, p in module_to_prefetch.named_parameters():
                        #         gpu_copy = self.dynamic_gpu_buffers[self.prefetch_buffer_index].parameters[name]
                        #         gpu_copy.copy_(p.data, non_blocking=True)
                        #         p.data = gpu_copy
                        #     self.prefetch_events[module_to_prefetch.layer_idx - self.start_layer].record(self.data_mv_stream)
                        #     self.recording_events[module_to_prefetch.layer_idx - self.start_layer] = True
                        # self.prefetched_modules.append(module_to_prefetch)
                        self.prefetch_buffer_index ^= 1 # switch the prefetch buffer index
                    ttt_4_3 = time.time_ns()
                    ttt_4.append((ttt_4_2 - ttt_4_1)/1e6)
                    ttt_4.append((ttt_4_3 - ttt_4_2)/1e6)
                    ttt_4.append((ttt_4_3 - ttt_4_1)/1e6)
                
                print(f"OffloadBuffer.wraper_forward: layer_idx={cur_layer_idx} offload and prefetch stage, current_stream={torch.cuda.current_stream()}"
                      f"stage_1(wait/sync) = {(ttt_2 - ttt_1)/1e6:.3f} ms, "
                      f"stage_3(offload_prefetch) = {[f'{value:.3f}' for value in ttt_4]}")    # logger.debug(f"OffloadBuffer.wraper_forward: layer_idx={layer_idx} put to {self.device} module device={next(module.parameters()).device}, inner_layer_idx={inter_layer_idx}, prefetch_buffer_index={self.prefetch_buffer_index}")           
            
        tt_end3 = time.time_ns()        
        # Step 4: run the forward function
        module.forward = self.original_forwards[cur_inner_layer_idx]
        output = functional_call(module,
                                module.state_dict(),
                                args=args,
                                kwargs=kwargs)
        # restore the forward function to the wrapper forward function
        module.forward = self.wraper_forwards[cur_inner_layer_idx]
        tt_end4 = time.time_ns()
        
        if cur_layer_idx == self.end_layer - 1 and (cur_layer_idx + 1) % self.curr_k_value == 0:
            # the last layer is also an offloaded layer
            self.compute_event.synchronize()
            # self.data_mv_stream.wait_event(self.compute_event) # Wait for the compute event to finish
            if len(self.offloaded_modules) != 0:
                # offload the last layer to the CPU buffer
                module_to_offload = self.prefetched_modules.popleft()
                assert module_to_offload.layer_idx == cur_layer_idx, f"OffloadBuffer.wraper_forward: the module to offload is not the last layer, {module_to_offload.layer_idx} != {cur_layer_idx}"
                for name, p in module_to_offload.named_parameters():
                    p.data = self.cpu_buffers[module_to_offload.layer_idx - self.start_layer].parameters[name]
                self.offloaded_modules.append(module_to_offload)
        tt_end5 = time.time_ns()
        
        logger.debug(f"OffloadBuffer.wraper_forward: perform forward with layer_idx={cur_layer_idx}, inner_layer_idx={cur_inner_layer_idx} cost: "
                     f"stage1={((tt_end1 - tt_start) / 1e6):.3f} ms, "
                     f"stage2={((tt_end2 - tt_end1) / 1e6):.3f} ms, "
                     f"stage3={((tt_end3 - tt_end2) / 1e6):.3f} ms, "
                     f"stage4={((tt_end4 - tt_end3) / 1e6):.3f} ms, "
                     f"stage5={((tt_end5 - tt_end4) / 1e6):.3f} ms, "
                     f"total={((tt_end5 - tt_start) / 1e6):.3f} ms")
        
        return output
    
    def get_offload_interval(self):
        return self.curr_k_value
    
    def get_per_layer_params_Bytes(self):
        # Calculate the total size of parameters in bytes
        assert self.layers is not None, "OffloadBuffer: layers is None, please call create_module first"
        total_size = 0
        for p in self.layers[0].parameters():
            total_size += p.data.numel() * p.data.element_size()
        return total_size
    
    def get_avg_H2D_transfer_time(self):
        assert self.H2D_transfer_times is not None, "OffloadBuffer: H2D transfer times is None, please call maybe_offload first"
        print(f"OffloadBuffer.get_avg_H2D_transfer_time: H2D transfer times = {[f'{value:.3f}' for value in self.H2D_transfer_times]}")
        return sum(self.H2D_transfer_times) / len(self.H2D_transfer_times) 
    
    
# NOTES:
# 1. We use two dynamic GPU buffers to store the prefetching and computing operations, which are used to switch between transfers and computations alternately.
# 2. Use a generic prefetch_events to record the prefetching events, and a generic recording_events to check if that element is being prefetched or not (only if it is being prefetched, check prefetch_events[i].sycnhronize()).
# 3. Use a static_cpu_buffers to permanently store the CPU buffers. This way you can always lookup the CPU based buffer in the static_cpu_buffers datastructure.
# 4. Use a dynamic_gpu_buffers to store the GPU buffers, which are used to switch between transfers and computations alternately.
# 5. Use a resident_gpu_buffers to store the `k` transformer blocks on the GPU. The resident_gpu_buffer will be dynamically allocated at every forward pass using `reorganize_resident_gpu_modules`.
# 6. Remove the `initial_offload_layers`, assume that we do equi-spaced offloading, i.e,. offload every `k` layers.
#    6.1 replace with initial_offload_layers with k
