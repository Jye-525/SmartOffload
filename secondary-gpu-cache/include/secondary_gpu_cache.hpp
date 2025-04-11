#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <deque>
#include <unordered_map>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <memory>
#include <string>
#include <torch/extension.h>
#include "debug.hpp"
#define __BENCHMARK

class SecondaryGPUCache {
public:
    SecondaryGPUCache(size_t block_size,
                      size_t num_kv_heads,
                      size_t head_size,
                      torch::Tensor cpu_tensor,
                      size_t cpu_stride,
                      size_t dtype_size,
                      float gpu_fraction = 0.05,
                      int device_id = 0);

    ~SecondaryGPUCache();

    void swap_out(int layer_id,
                  torch::Tensor gpu_src_keys,
                  torch::Tensor gpu_src_vals,
                  const std::vector<std::pair<int, int>>& src_ids);

    void swap_in(int layer_id,
                 torch::Tensor gpu_dst_keys,
                 torch::Tensor gpu_dst_vals,
                 const std::vector<std::pair<int, int>>& dst_ids);

    void grow_cache(size_t size_in_bytes);
    void shrink_cache(size_t size_in_bytes_to_shrink);

private:
    struct Block {
        void* device_ptr;
        size_t offset;
        cudaEvent_t event;
        bool in_use = false;
        int host_id = -1;
    };

    size_t block_size_;
    size_t num_kv_heads_;
    size_t head_size_;
    size_t dtype_size_;
    size_t block_bytes_;
    size_t max_blocks_;
    int device_id_;

    void* device_mem_base_;
    size_t device_mem_size_;

    void* cpu_cache_ptr_;
    size_t cpu_stride_;

    cudaStream_t gpu_to_host_stream_;
    cudaStream_t gpu_to_gpu_stream_;

    std::deque<size_t> available_blocks_;
    std::deque<size_t> fifo_buffer_;

    std::unordered_map<int, size_t> h_to_s_;
    std::unordered_map<size_t, int> s_to_h_;

    std::vector<Block> blocks_;

    void initialize_vmm(size_t num_blocks);
    void destroy_vmm();
    void evict_block_if_needed();
};
