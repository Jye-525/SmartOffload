import torch
import torch.nn as nn
from torch.func import functional_call
from collections import deque
from vllm.utils import is_pin_memory_available
from vllm.spec_decode.util import nvtx_range

class PerLayerParameters:
    def __init__(self):
        self.parameters = {} # Ensure it's a new dictionary for each instance
    

class OffloadBuffer:
    # Question: who and when to init the OffloadBuffer in vLLM?
    def __init__(self, cpu_offload_layers, cpu_offload_type):
        self.cpu_offload_layers = cpu_offload_layers
        self.cpu_offload_type = cpu_offload_type
        self.layers = 0 # record the number of layers in this pipeline rank, will be updated in the `maybe_offload` function
        self.cpu_buffer = None
        self.gpu_buffer = None # Assume every transformer layer has the same architecture.
        # record which modules are on device and which modules are offloaded
        self.device_modules = deque()
        self.offloaded_modules = deque()
        self.prefetched_modules = deque()
        self.original_forwards = None
        self.wraper_forwards = None
        self.next_offload_gpu_buffer_idx = 0 # record the current idx of the GPU buffer for ringbuffer idx
        self.ondemand_mv = False # whether to use on-demand data transfer 
        self.data_mv_stream = torch.cuda.Stream() # create a new stream for H2D data transfer
        self.start_layer = -1
        self.end_layer = -1
        self.test_case = "offload_from_end" # offload_from_beginning or offload_from_end 
        
    def create_module(self, module: torch.nn.Module, layer_idx: int, start_layer: int, end_layer: int):
        # This function is to create a copy of the module and put the parameters on CPU.
        print(f"OffloadBuffer.create_module: layer_idx={layer_idx}, start_layer={start_layer}, end_layer={end_layer}")
        # get the original device of the module
        self.device = next(module.parameters()).device
         
        if self.device == torch.device("cpu"):
            # if self.device is CPU, it means that the model is runing on CPU.
            # so we don't need to offload the module to CPU and intercept the forward function
            return module
        
        if self.cpu_offload_layers == 0:
            # disable the model weights offloading
            return module
        
        if self.layers == 0:
            # update the number of layers in this pipeline rank
            self.start_layer = start_layer
            self.end_layer = end_layer
            self.layers = end_layer - start_layer
            self.device_layers = self.layers - self.cpu_offload_layers
            # initialize the cpu_buffer (the buffer on host memory)
            self.cpu_buffer = [PerLayerParameters() for _ in range(self.layers)]
            self.gpu_buffer = [PerLayerParameters() for _ in range(self.device_layers)]
            self.original_forwards = [None for _ in range(self.layers)]
            self.wraper_forwards = [None for _ in range(self.layers)]
            print(f"OffloadBuffer.create_module: layers={self.layers}, device_layers={self.device_layers}, len(self.cpu_buffer) = {len(self.cpu_buffer)} "
                  f"len(self.gpu_buffer) = {len(self.gpu_buffer)} len(self.original_forwards) = {len(self.original_forwards)} "
                  f"len(self.wraper_forwards) = {len(self.wraper_forwards)}", flush=True)
            
        # Update the original forward function for this given module layer_idx
        self.original_forwards[layer_idx] = module.forward
        # create a copy of the parameters of the module and put them on CPU
        pin_memory = is_pin_memory_available()
        for name, p in module.named_parameters():
            cpu_data = torch.empty_strided(size=p.data.size(),
                                            stride=p.data.stride(),
                                            dtype=p.data.dtype,
                                            layout=p.data.layout,
                                            device='cpu',
                                            pin_memory=pin_memory)
            cpu_data.copy_(p.data)
            p.data = cpu_data
            self.cpu_buffer[layer_idx].parameters[name] = cpu_data
            
            
        # create the GPU buffer for the device layers
        if layer_idx < start_layer + (self.layers - self.cpu_offload_layers):
                device_layer_idx = layer_idx - start_layer
                # print(f"OffloadBuffer.create_module: layer_idx={layer_idx}, gpu_buffer idx = {device_layer_idx}", flush=True)
                for name, p in module.named_parameters():
                    gpu_data = torch.empty_strided(size=p.data.size(),
                                                stride=p.data.stride(),
                                                dtype=p.data.dtype,
                                                layout=p.data.layout,
                                                device=self.device)
                    gpu_data.copy_(p.data)
                    self.gpu_buffer[device_layer_idx].parameters[name] = gpu_data
            
             
        # create a wrapper forward function for the module
        def forward(*args, **kwargs):
            return self.wraper_forward(module, args=args, kwargs=kwargs)
        
        module.forward = forward 
        self.wraper_forwards[layer_idx] = module.forward

        return module
    
    def maybe_offload(self, module: torch.nn.Module, layer_idx: int, start_layer: int, end_layer: int):
        # We will offload the module layers in reverse order
        if self.ondemand_mv and self.test_case == "offload_from_beginning":
            # offload the first self.cpu_offload_layers layers to the CPU
            if layer_idx >= start_layer + self.cpu_offload_layers:
                device_layer_idx = layer_idx - (start_layer + self.cpu_offload_layers)
                #    print(f"OffloadBuffer.maybe_offload: layer_idx={layer_idx} put to device {self.device}, device_layer_idx on GPU buffer {device_layer_idx}", flush=True)
                for name, p in module.named_parameters():
                    gpu_copy = self.gpu_buffer[device_layer_idx].parameters[name]
                    gpu_copy.copy_(p.data)
                    p.data = gpu_copy        
                self.device_modules.append(module)
            else:
                # put this module layer on the CPU
                self.offloaded_modules.append(module)
                # print(f"OffloadBuffer.maybe_offload: layer_idx={layer_idx} put to CPU", flush=True) 
        else:
            if layer_idx < start_layer + (self.layers - self.cpu_offload_layers):
                device_layer_idx = layer_idx - start_layer
                #    print(f"OffloadBuffer.maybe_offload: layer_idx={layer_idx} put to device {self.device}, device_layer_idx on GPU buffer {device_layer_idx}", flush=True)
                for name, p in module.named_parameters():
                    gpu_copy = self.gpu_buffer[device_layer_idx].parameters[name]
                    gpu_copy.copy_(p.data)
                    p.data = gpu_copy
                        
                self.device_modules.append(module)
            else:
                # put this module layer on the CPU
                self.offloaded_modules.append(module)
                # print(f"OffloadBuffer.maybe_offload: layer_idx={layer_idx} put to CPU", flush=True) 
            
        return module
        
    
    @nvtx_range("OffloadBuffer.wraper_forward")
    def wraper_forward(self, module: torch.nn.Module, args=None, kwargs=None):
        # intercept the original forward function
        # perform forward using the original forward function
        # perform data transfer between CPU and GPU
        current_device = next(module.parameters()).device
        # print(f"OffloadBuffer.forward, layer_idx={module.layer_idx}, current_device = {current_device}", flush=True)
        
        # layer_fwd_start_list = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        # layer_fwd_end_list = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        # get the original forward function of the module
        module.forward = self.original_forwards[module.layer_idx]
        # layer_fwd_start_list[0].record() 
        if current_device == torch.device("cpu"):
            print(f"ERROR: OffloadBuffer.wraper_forward: Error if go here since we do prefetch....Layer_idx={module.layer_idx}, test_case={self.test_case}", flush=True)
            # on-demand data transfer case will go here. Always reuse the last GPU_buffer slot for on-demand data transfer
            if self.test_case == "offload_from_beginning":
                # use the first GPU buffer slot for on-demand data transfer
                # Step 1: get the data of the last GPU buffer slot
                module_to_offload = self.device_modules.popleft()
                for name, p in module_to_offload.named_parameters():
                    p.data = self.cpu_buffer[module_to_offload.layer_idx].parameters[name]
                self.offloaded_modules.append(module_to_offload)
                
                # Step 2: copy the data of the current module to the first GPU buffer slot and remove the first module from the offloaded_modules
                for name, p in module.named_parameters():
                    gpu_copy = self.gpu_buffer[0].parameters[name]
                    gpu_copy.copy_(p.data, non_blocking=True)
                    p.data = gpu_copy
                self.device_modules.appendleft(module)
                self.offloaded_modules.popleft()
                print(f"[offload_from_beginning] OffloadBuffer.wraper_forward: module_to_offload.layer_idx={module_to_offload.layer_idx}, module.layer_idx={module.layer_idx}", flush=True)
            else:
                # use the last GPU buffer slot for on-demand data transfer
                # Step 1: get the data of the last GPU buffer slot
                module_to_offload = self.device_modules.pop()
                for name, p in module_to_offload.named_parameters():
                    p.data = self.cpu_buffer[module_to_offload.layer_idx].parameters[name]
                self.offloaded_modules.append(module_to_offload)
                
                # Step 2: copy the data of the current module to the last GPU buffer slot and remove the first module from the offloaded_modules
                for name, p in module.named_parameters():
                    gpu_copy = self.gpu_buffer[-1].parameters[name]
                    gpu_copy.copy_(p.data, non_blocking=True)
                    p.data = gpu_copy
                self.device_modules.append(module)
                self.offloaded_modules.popleft()
                print(f"[offload_from_end] OffloadBuffer.wraper_forward: module_to_offload.layer_idx={module_to_offload.layer_idx}, module.layer_idx={module.layer_idx}", flush=True)
                 
            # get the device_state of the module
            device_state = module.state_dict()
        else:
            print(f"OffloadBuffer.wraper_forward: layer_idx={module.layer_idx}, current_device={current_device}", flush=True)
            device_state = module.state_dict()
        # layer_fwd_end_list[0].record()
        
        # layer_fwd_start_list[1].record()    
        output = functional_call(module,
                                device_state,
                                args=args,
                                kwargs=kwargs)
        # layer_fwd_end_list[1].record()
        # restore the forward function to the wrapper one
        module.forward = self.wraper_forwards[module.layer_idx]
        
        if self.ondemand_mv == False:
            # for non on-demand data transfer, we will prefetch the next module to device immediately
            # get the current module to be offloaded from teh device_modules
            # layer_fwd_start_list[2].record()
            if module.layer_idx < self.start_layer + self.cpu_offload_layers:
                print(f"offload_buffer.wraper_forward: prefetch is triggered, current layer_idx = {module.layer_idx}", flush=True)
                # trigger prefetching of the offloded module 
                module_to_offload = self.device_modules.popleft()
                assert module_to_offload.layer_idx == module.layer_idx, f"module_to_offload.layer_idx={module_to_offload.layer_idx}, module.layer_idx={module.layer_idx}"
                for name, p in module_to_offload.named_parameters():
                    p.data = self.cpu_buffer[module_to_offload.layer_idx].parameters[name]
                self.offloaded_modules.append(module_to_offload)
                # layer_fwd_end_list[2].record()
                
                # prefetch an offloaded module to the device the gpu_buffer_idx is self.next_offload_gpu_buffer_idx
                # layer_fwd_start_list[3].record()
                module_to_prefetch = self.offloaded_modules.popleft()
                # preftech the offloaded layer using a different stream
                with torch.cuda.stream(self.data_mv_stream):
                    for name, p in module_to_prefetch.named_parameters():
                        gpu_copy = self.gpu_buffer[self.next_offload_gpu_buffer_idx].parameters[name]
                        gpu_copy.copy_(p.data, non_blocking=True)
                        p.data = gpu_copy    
                self.prefetched_modules.append(module_to_prefetch)
                tmp_idx = self.next_offload_gpu_buffer_idx
                self.next_offload_gpu_buffer_idx = (self.next_offload_gpu_buffer_idx + 1) % self.cpu_offload_layers
                print(f"OffloadBuffer.wraper_forward: layer_idx={module.layer_idx}, offload_layer={module_to_offload.layer_idx} "
                      f"prefetch_layer={module_to_prefetch.layer_idx}, gpu_slot_idx_for_prefetched_layer={tmp_idx}, "
                      f"next_offload_gpu_buffer_idx={self.next_offload_gpu_buffer_idx}", flush=True)
            
            if module.layer_idx >= self.end_layer - self.cpu_offload_layers:
                # trigger prefetching of the first layers in the offloaded_modules
                # trigger prefetching of the offloded module
                print(f"offload_buffer.wraper_forward: prefetch is triggered to fetch layers for next iteration, current layer_idx = {module.layer_idx}", flush=True) 
                module_to_offload = self.prefetched_modules.popleft()
                for name, p in module_to_offload.named_parameters():
                    p.data = self.cpu_buffer[module_to_offload.layer_idx].parameters[name]
                self.offloaded_modules.append(module_to_offload)
                # layer_fwd_end_list[2].record()
                
                # prefetch an offloaded module to the device the gpu_buffer_idx is self.next_offload_gpu_buffer_idx
                # layer_fwd_start_list[3].record()
                module_to_prefetch = self.offloaded_modules.popleft()
                with torch.cuda.stream(self.data_mv_stream):
                    for name, p in module_to_prefetch.named_parameters():
                        gpu_copy = self.gpu_buffer[self.next_offload_gpu_buffer_idx].parameters[name]
                        gpu_copy.copy_(p.data, non_blocking=True)
                        p.data = gpu_copy    
                self.prefetched_modules.append(module_to_prefetch)
                tmp_idx = self.next_offload_gpu_buffer_idx
                self.next_offload_gpu_buffer_idx = (self.next_offload_gpu_buffer_idx + 1) % self.cpu_offload_layers
                print(f"+++++++OffloadBuffer.wraper_forward: layer_idx={module.layer_idx}, offload_layer={module_to_offload.layer_idx} "
                      f"prefetch_layer={module_to_prefetch.layer_idx}, gpu_slot_idx_for_prefetched_layer={tmp_idx}, "
                      f"next_offload_gpu_buffer_idx={self.next_offload_gpu_buffer_idx}", flush=True) 
                
                if module.layer_idx == self.end_layer - 1:
                    # reset the next_offload_gpu_buffer_idx to 0
                    self.next_offload_gpu_buffer_idx = 0
                    # put prefetch_modules to the device_modules
                    while self.prefetched_modules:
                        self.device_modules.appendleft(self.prefetched_modules.pop())
                  
                  
            # layer_fwd_end_list[3].record()
            
            # for i in range(4):
            #     layer_fwd_end_list[i].synchronize()
            
            # print(f"OffloadBuffer.wraper_forward: layer_idx={module.layer_idx}, data_mv_0={layer_fwd_start_list[0].elapsed_time(layer_fwd_end_list[0]):.3f} "
            #       f"fwd = {layer_fwd_start_list[1].elapsed_time(layer_fwd_end_list[1]):.3f} "
            #       f"offload = {layer_fwd_start_list[2].elapsed_time(layer_fwd_end_list[2]):.3f} offload_layer={module_to_offload.layer_idx} "
            #       f"prefetch = {layer_fwd_start_list[3].elapsed_time(layer_fwd_end_list[3]):.3f} prefetch_layer={module_to_prefetch.layer_idx} "
            #       f"the prefetched module will use gpu_buffer idx {self.next_offload_gpu_buffer_idx}", flush=True)
            # update the next_offload_gpu_buffer_idx
            # self.next_offload_gpu_buffer_idx = (self.next_offload_gpu_buffer_idx + 1) % self.device_layers
        return output
