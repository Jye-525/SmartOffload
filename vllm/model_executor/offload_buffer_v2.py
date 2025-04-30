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
        # self.resident_gpu_buffers = {} # store the `self.nlayers - self.nlayers // k` transformer blocks statically on the GPU.
        
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
        
        self.per_layer_info = []
        logger.debug(f"OffloadBuffer.__init__: init_k={self.curr_k_value}")
    
    def __initialization(self, start_layer: int, end_layer: int):
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.nlayers = end_layer - start_layer
        # initialize the cpu_buffer (the buffer on host memory)
        self.cpu_buffers = [PerLayerParameters() for _ in range(self.nlayers)]  
        self.prefetch_events = [torch.cuda.Event() for _ in range(self.nlayers)]
        self.recording_events = [False for _ in range(self.nlayers)]
        self.original_forwards = [None for _ in range(self.nlayers)]
        self.wraper_forwards = [None for _ in range(self.nlayers)]
        self.layers = [None for _ in range(self.nlayers)]
        self.per_layer_info = [{'total_bytes': 0, 'total_elems': 0, 'dtype': None, 'parameters': {}} for _ in range(self.nlayers)]
        self.total_offloaded_layers = self.nlayers // self.curr_k_value
        logger.debug(f"OffloadBuffer.__initialization: start_layer={start_layer}, end_layer={end_layer}, init_k={self.curr_k_value}, total_offloaded_layers={self.total_offloaded_layers}")
    
    def __init_dynamic_gpu_buffers(self):
        # (TODO) Question: Can we make the dynamic_gpu_buffers number dynamic?
        #  - If nlayers // cur_key_value > 2, we crate a duble buffer for prefetching
        #  - If nlayers // cur_key_value <= 2, we create a single buffer for prefetching
        self.dynamic_gpu_buffers = [PerLayerParameters() for _ in range(2)]
        # - Assume each transformer blcok has same architecture, so use the first layer information 
        named_params_dict = dict(self.layers[0].named_parameters())
        # to create the dynamic GPU buffer
        layer_dict = self.per_layer_info[0]
        for buffer in self.dynamic_gpu_buffers:
            # Step 1: create a GPU buffer for thos layer
            # gpu_data = torch.empty(layer_dict['total_elems'],
            #                     dtype=layer_dict['dtype'], device=self.device)
            gpu_data = torch.zeros(layer_dict['total_elems'],
                                dtype=layer_dict['dtype'], device=self.device)
            layer_params = layer_dict['parameters'] 
            start_offset = 0
            for name, p in named_params_dict.items():
                end_offset = start_offset + layer_params[name]
                gpu_data[start_offset:end_offset].copy_(p.data.view(-1))
                buffer.parameters[name] = gpu_data[start_offset:end_offset]
                start_offset = end_offset
        
    
    ### Question 2: When do we call this function? at the beginning of each forward pass?
    ###  - This part adds some overhead
    @nvtx_range("OffloadBuffer.reorganize")  
    def reorganize_resident_gpu_modules(self, k: int):
        if self.curr_k_value == k:
            return
        t_start = time.time_ns()
        
        old_k_value = self.curr_k_value
        self.curr_k_value = k
        
        # Step 1: Put the layers in the prefetched_modules to the CPU buffer beore reorganize
        with nvtx_range(f"reorg_step1_{old_k_value}_to_{k}"):
            while self.prefetched_modules:
                module = self.prefetched_modules.popleft()
                layer_idx = module.layer_idx
                # Get references to avoid repeated lookups
                layer = self.layers[layer_idx]
                cpu_buffer = self.cpu_buffers[layer_idx - self.start_layer]
                for name, p in layer.named_parameters():
                    p.data = cpu_buffer.parameters[name]
        
        # Step 2: clear the prefetched modules and offloaded modules
        with nvtx_range(f"reorg_step2_{old_k_value}_to_{k}"):   
            self.prefetched_modules.clear()
            self.offloaded_modules.clear()
            self.prefetch_buffer_index = 0 # reset the prefetch buffer index
            t_start_aaa = time.time_ns()
        
        
        with nvtx_range(f"reorg_step3_{old_k_value}_to_{k}"):
            times_per_layer = {}
            for layer_idx in range(self.start_layer, self.end_layer):
                tt_start = time.time_ns()
                layer = self.layers[layer_idx]
                if (layer_idx + 1) % self.curr_k_value == 0:
                    # the layer should be offloaded to the CPU except the kth and 2kth layer
                    with nvtx_range(f"reorg_step3_{old_k_value}_to_{k}_{layer_idx}"):
                        if (layer_idx + 1 ) in [self.curr_k_value, 2 * self.curr_k_value]:
                            buffer_idx = (self.prefetch_buffer_index ^ 1 if (layer_idx + 1) == 2 * self.curr_k_value
                                            else self.prefetch_buffer_index)
                            self.__prefetch(layer, layer_idx - self.start_layer, buffer_idx, non_blocking=True)
                        else:
                            # this layer should be on the CPU after reorganize
                            if next(layer.parameters()).device != torch.device("cpu"):
                                cpu_buffer = self.cpu_buffers[layer_idx - self.start_layer]
                                for name, p in layer.named_parameters():
                                    p.data = cpu_buffer.parameters[name]
                            self.offloaded_modules.append(layer)
                else:
                    # If the layer is on GPU, asynchronously move it to the GPU buffer;
                    # otherwise, do nothing and reuse the GPU buffer
                    if next(layer.parameters()).device == torch.device("cpu"):
                        ### cache nvtx to see if cuda free trigger before or after this.
                        with nvtx_range(f"reorg_step3_{layer_idx}_to_gpu"):
                            # self.resident_gpu_buffers[layer_idx] = PerLayerParameters() 
                            layer_dict = self.per_layer_info[layer_idx - self.start_layer]
                            layer_params = layer_dict['parameters']
                            with torch.cuda.stream(self.data_mv_stream):
                                # gpu_data = torch.empty(layer_dict['total_elems'],
                                #                     dtype=layer_dict['dtype'], device=self.device)
                                gpu_data = torch.zeros(layer_dict['total_elems'],
                                                    dtype=layer_dict['dtype'], device=self.device)
                                start_offset = 0
                                for name, p in layer.named_parameters():
                                    end_offset = start_offset + layer_params[name]
                                    gpu_data[start_offset:end_offset].copy_(p.data.view(-1), non_blocking=True)
                                    p.data = gpu_data[start_offset:end_offset].view_as(p.data) 
                                    # self.resident_gpu_buffers[layer_idx].parameters[name] = gpu_data[start_offset:end_offset] 
                        self.prefetch_events[layer_idx].record(self.data_mv_stream)
                        self.recording_events[layer_idx] = True
                    
                tt_end = time.time_ns()
                times_per_layer[layer_idx] = (tt_end - tt_start) / 1e6
        
        t_end = time.time_ns()
        self.first_fwd_after_reorganize = True # reset the first forward pass after reorganize
        self.total_offloaded_layers = self.nlayers // self.curr_k_value            
        
        logger.debug(f"Reorganizing the resident GPU modules from stride of {old_k_value} stride size of {self.curr_k_value}, "
                     f"reorganizing layers cost {((t_end - t_start) / 1e6):.3f} ms, "
                     f"per layer time = {[f'{k}={v:.3f}' for k, v in times_per_layer.items()]}, "
                     f"total_offloaded_layers={self.total_offloaded_layers},"
                     f"aaa cost = {((t_start_aaa - t_start) / 1e6):.3f} ms")
    
        
    def create_module(self, module: torch.nn.Module, layer_idx: int, start_layer: int, end_layer: int):
        # get the original device of the module
        self.device = next(module.parameters()).device 
        if self.device == torch.device("cpu"):
            return module
        
        assert self.curr_k_value > 0, "The init_k should be greater than 0 when smart offload is enabled"
        
        if self.nlayers == 0:
            # initialize the parameters used in this pipeline rank
            self.__initialization(start_layer, end_layer)
        
        inter_layer_idx = layer_idx - start_layer # used for indexing various buffers and envents
        
        # Step 1: Create a CPU buffer and copy the parameters of this module layer to the CPU buffer (pinned memory)
        pin_memory = is_pin_memory_available()
        elems_per_layer = 0
        elem_size = 0
        cpu_buffer = self.cpu_buffers[inter_layer_idx]
        layer_params = self.per_layer_info[inter_layer_idx]['parameters']
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
            elems_per_layer += p.data.numel()
            layer_params[name] = p.data.numel()
            if self.per_layer_info[inter_layer_idx]['dtype'] is None:
                self.per_layer_info[inter_layer_idx]['dtype'] = p.data.dtype
                elem_size = p.data.element_size()
        self.per_layer_info[inter_layer_idx]['total_elems'] = elems_per_layer
        self.per_layer_info[inter_layer_idx]['total_bytes'] = elems_per_layer * elem_size 
             
        # create a wrapper forward function for the module
        def forward(*args, **kwargs):
            return self.wraper_forward(module, args=args, kwargs=kwargs)
        
        # Update the original forward and wrapper forward function pointer for the module
        self.original_forwards[inter_layer_idx] = module.forward
        module.forward = forward 
        self.wraper_forwards[inter_layer_idx] = module.forward

        # save the transformer block layer in the layers List (why do we need this?)
        self.layers[inter_layer_idx] = module
        
        if layer_idx == end_layer - 1:
            self.__init_dynamic_gpu_buffers()
        
        return module
    
    def __prefetch(self, module: torch.nn.Module, inner_layer_idx: int, buffer_index: int, non_blocking: bool):
        # prefetch the given layer to the dynamic GPU buffer slot buffer_index
        dynamic_gpu_buffer = self.dynamic_gpu_buffers[buffer_index]
        for name, p in module.named_parameters():
            with torch.cuda.stream(self.data_mv_stream):
                gpu_copy = dynamic_gpu_buffer.parameters[name] 
                gpu_copy.copy_(p.data.view(-1), non_blocking=non_blocking) # Asynchronous copy the data to the GPU buffer
                p.data = gpu_copy.view_as(p.data)
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
                # prefetch the kth and 2kth layer to the dynamic GPU buffer slot 0
                buffer_idx = (self.prefetch_buffer_index ^ 1 
                                if (inner_layer_idx + 1) == 2 * self.curr_k_value 
                                else self.prefetch_buffer_index)
                
                self.__prefetch(module, inner_layer_idx, buffer_idx, non_blocking=False)
                
                if self.curr_k_value == 1 and self.recording_events[inner_layer_idx] == True:
                     self.prefetch_events[inner_layer_idx].synchronize()
                     self.recording_events[inner_layer_idx] = False
                
                logger.debug(f"OffloadBuffer.maybe_offload: layer_idx={layer_idx} put to {self.device} module device={next(module.parameters()).device}, inner_layer_idx={inner_layer_idx}, prefetch_buffer_index={buffer_idx}") 
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
            # self.resident_gpu_buffers[layer_idx] = PerLayerParameters()
            # create a GPU buffer for this layer
            layer_dict = self.per_layer_info[inner_layer_idx]
            layer_params = layer_dict['parameters']
            gpu_data = torch.empty(layer_dict['total_elems'], 
                                dtype=layer_dict['dtype'], device=self.device)
            start_offset = 0
            for name, p in module.named_parameters():
                end_offset = start_offset + layer_params[name]
                gpu_data[start_offset:end_offset].copy_(p.data.view(-1))
                p.data = gpu_data[start_offset:end_offset].view_as(p.data)
                start_offset = end_offset     
            self.recording_events[inner_layer_idx] = False
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
        
        # tt_start = time.time_ns() 
        # Step 1: Always prefetech the kth, and 2kth layer to the dynamic GPU buffer at the first layer
        with nvtx_range(f"wraper_fwd_step1_{cur_layer_idx}"):
            if cur_layer_idx == self.start_layer:
                if self.first_fwd_after_reorganize == True:
                    self.first_fwd_after_reorganize = False
                elif self.first_fwd_after_reorganize == False and self.total_offloaded_layers > 2:
                    self.prefetched_modules.clear()
                    self.prefetch_buffer_index = 0 # reset the prefetch buffer index 
                    # prefetch the kth and 2kth layer to the dynamic GPU buffer
                    # ttt_start_1 = time.time_ns()
                    module_to_prefetch_k = self.offloaded_modules.popleft()
                    inner_k_idx = module_to_prefetch_k.layer_idx - self.start_layer
                    assert inner_k_idx + 1 == self.curr_k_value, f"OffloadBuffer.wraper_forward: the module to prefetch is not the kth layer, {module_to_prefetch_k.layer_idx + 1} != {self.start_layer + self.curr_k_value}, k={self.curr_k_value}"
                    buffer_idx = self.prefetch_buffer_index
                    self.__prefetch(module_to_prefetch_k, inner_k_idx, buffer_idx, non_blocking=True)
                    
                    module_to_prefetch_2k = self.offloaded_modules.popleft()
                    inner_2k_idx = module_to_prefetch_2k.layer_idx - self.start_layer
                    buffer_idx = self.prefetch_buffer_index^1
                    assert (inner_2k_idx) + 1 == 2 * self.curr_k_value, f"OffloadBuffer.wraper_forward: the module to prefetch is not the 2kth layer, {module_to_prefetch_2k.layer_idx} != {self.start_layer + 2 * self.curr_k_value}, k={self.curr_k_value}"
                    self.__prefetch(module_to_prefetch_2k, inner_2k_idx, buffer_idx, non_blocking=True) 
                        
                    # ttt_start_2 = time.time_ns()
                    # logger.debug(f"OffloadBuffer.wraper_forward: First layer to prefetch layer {inner_k_idx} and {inner_2k_idx} to the dynamic GPU buffer"
                    #             f", cost {(ttt_start_2 - ttt_start_1)/1e6:.3f} ms")       
            
        # tt_end1 = time.time_ns()  
        # Step 2: check if the current layer is prefetching data
        with nvtx_range(f"wraper_fwd_step2_{cur_layer_idx}"):
            if self.recording_events[cur_inner_layer_idx]:
                self.prefetch_events[cur_inner_layer_idx].synchronize()
                self.recording_events[cur_inner_layer_idx] = False
        # tt_end2 = time.time_ns() 
        
        # Step 3: Check if the current layer is every (kth + 1) layer
        with nvtx_range(f"wraper_fwd_step3_{cur_layer_idx}"):
            if ((cur_inner_layer_idx + 1) > self.curr_k_value and \
                cur_inner_layer_idx < self.nlayers):
        
                if ((self.curr_k_value == 1) or (cur_inner_layer_idx + 1) % self.curr_k_value == 1):
                    # Step 2: current layer is right the layer after every kth layer
                    # 2.1: Wait the previous computation to finish
                    # ttt_1 = time.time_ns()
                    self.compute_event.synchronize()
                    # ttt_2 = time.time_ns()
                    # 2.2: Check if need to prefetch the next kth layer
                    # ttt_4 = []
                    if len(self.offloaded_modules) > 0:
                        # ttt_4_1 = time.time_ns()
                        # offload the (current layer - 1) to the CPU buffer
                        module_to_offload = self.prefetched_modules.popleft()
                        assert module_to_offload.layer_idx == cur_layer_idx - 1, f"OffloadBuffer.wraper_forward: the module to offload is not the previous layer, {module_to_offload.layer_idx} != {cur_layer_idx - 1}"
                        cpu_buffer = self.cpu_buffers[module_to_offload.layer_idx - self.start_layer]
                        for name, p in module_to_offload.named_parameters():
                            p.data = cpu_buffer.parameters[name]
                        self.offloaded_modules.append(module_to_offload)
                        # ttt_4_2 = time.time_ns()
                        
                        # prefetch the next 2 * kth layer to the corresponding dynamic gpu buffer
                        prefetech_layer_idx = cur_layer_idx + 2 * self.curr_k_value - 1 
                        if prefetech_layer_idx < self.end_layer:
                            # only trigger preftech when the prefetech layer is less then the end layer
                            module_to_prefetch = self.offloaded_modules.popleft()
                            assert module_to_prefetch.layer_idx == prefetech_layer_idx, f"OffloadBuffer.wraper_forward: the module to prefetch is not the next kth layer, {module_to_prefetch.layer_idx} != {prefetech_layer_idx}, k={self.curr_k_value}"
                            self.__prefetch(module_to_prefetch, module_to_prefetch.layer_idx - self.start_layer, self.prefetch_buffer_index, non_blocking=True)
                            self.prefetch_buffer_index ^= 1 # switch the prefetch buffer index
                        # ttt_4_3 = time.time_ns()
                        # ttt_4.append((ttt_4_2 - ttt_4_1)/1e6)
                        # ttt_4.append((ttt_4_3 - ttt_4_2)/1e6)
                        # ttt_4.append((ttt_4_3 - ttt_4_1)/1e6)
                    
                    # print(f"OffloadBuffer.wraper_forward: layer_idx={cur_layer_idx} offload and prefetch stage, current_stream={torch.cuda.current_stream()}"
                    #       f"stage_1(wait/sync) = {(ttt_2 - ttt_1)/1e6:.3f} ms, "
                    #       f"stage_3(offload_prefetch) = {[f'{value:.3f}' for value in ttt_4]}")    # logger.debug(f"OffloadBuffer.wraper_forward: layer_idx={layer_idx} put to {self.device} module device={next(module.parameters()).device}, inner_layer_idx={inter_layer_idx}, prefetch_buffer_index={self.prefetch_buffer_index}")           
                
        # tt_end3 = time.time_ns()        
        # Step 4: run the forward function
        with nvtx_range(f"wraper_fwd_step4_{cur_layer_idx}"):
            module.forward = self.original_forwards[cur_inner_layer_idx]
            output = functional_call(module,
                                    module.state_dict(),
                                    args=args,
                                    kwargs=kwargs)
            # restore the forward function to the wrapper forward function
            module.forward = self.wraper_forwards[cur_inner_layer_idx]
        # tt_end4 = time.time_ns()
        
        with nvtx_range(f"wraper_fwd_step5_{cur_layer_idx}"):
            if cur_layer_idx == self.end_layer - 1 and (cur_layer_idx + 1) % self.curr_k_value == 0:
                # the last layer is also an offloaded layer
                self.compute_event.synchronize()
                if len(self.offloaded_modules) > 0:
                    # offload the last layer to the CPU buffer
                    module_to_offload = self.prefetched_modules.popleft()
                    assert module_to_offload.layer_idx == cur_layer_idx, f"OffloadBuffer.wraper_forward: the module to offload is not the last layer, {module_to_offload.layer_idx} != {cur_layer_idx}"
                    cpu_buffer = self.cpu_buffers[module_to_offload.layer_idx - self.start_layer]
                    for name, p in module_to_offload.named_parameters():
                        p.data = cpu_buffer.parameters[name]
                    self.offloaded_modules.append(module_to_offload)
        # tt_end5 = time.time_ns()
        
        # logger.debug(f"OffloadBuffer.wraper_forward: perform forward with layer_idx={cur_layer_idx}, inner_layer_idx={cur_inner_layer_idx} cost: "
        #              f"stage1={((tt_end1 - tt_start) / 1e6):.3f} ms, "
        #              f"stage2={((tt_end2 - tt_end1) / 1e6):.3f} ms, "
        #              f"stage3={((tt_end3 - tt_end2) / 1e6):.3f} ms, "
        #              f"stage4={((tt_end4 - tt_end3) / 1e6):.3f} ms, "
        #              f"stage5={((tt_end5 - tt_end4) / 1e6):.3f} ms, "
        #              f"total={((tt_end5 - tt_start) / 1e6):.3f} ms")
        
        return output
    
    def get_offload_interval(self):
        return self.curr_k_value
    
    def get_per_layer_params_Bytes(self):
        # Calculate the total size of parameters in bytes
        assert self.per_layer_info is not None, "OffloadBuffer: per_layer_info is None, please call create_module first"
        total_size = self.per_layer_info[0]['total_bytes']
        # for p in self.layers[0].parameters():
        #     total_size += p.data.numel() * p.data.element_size()
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
