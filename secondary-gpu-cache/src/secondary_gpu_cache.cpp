#include "secondary_gpu_cache.hpp"
#include <iostream>
#include <cstring>

SecondaryGPUCache::SecondaryGPUCache(size_t block_size,
                                     size_t num_kv_heads,
                                     size_t head_size,
                                     torch::Tensor cpu_tensor,
                                     size_t cpu_stride,
                                     size_t dtype_size,
                                     float gpu_fraction,
                                     int device_id)
    : block_size_(block_size), num_kv_heads_(num_kv_heads),
      head_size_(head_size), dtype_size_(dtype_size),
      cpu_stride_(cpu_stride), device_id_(device_id) {
    
    TIMER_START(init_timer);
    block_bytes_ = 2 * block_size_ * num_kv_heads_ * head_size_ * dtype_size_;
    cpu_cache_ptr_ = cpu_tensor.data_ptr();

    cudaSetDevice(device_id_);
    cudaStreamCreate(&gpu_to_gpu_stream_);
    cudaStreamCreate(&gpu_to_host_stream_);

    size_t total_mem;
    // cudaDeviceGetAttribute((int*)&total_mem, cudaDevAttrTotalGlobalMem, device_id);
    cudaMemGetInfo(nullptr, &total_mem);
    size_t allocatable = static_cast<size_t>(total_mem * gpu_fraction);
    max_blocks_ = allocatable / block_bytes_;
    initialize_vmm(max_blocks_);
    TIMER_STOP(init_timer, "GPU cache initialized");
}

SecondaryGPUCache::~SecondaryGPUCache() {
    destroy_vmm();
    cudaStreamDestroy(gpu_to_gpu_stream_);
    cudaStreamDestroy(gpu_to_host_stream_);
}

void SecondaryGPUCache::initialize_vmm(size_t num_blocks) {
    device_mem_size_ = num_blocks * block_bytes_;
    cudaMalloc(&device_mem_base_, device_mem_size_);

    for (size_t i = 0; i < num_blocks; ++i) {
        Block blk;
        blk.device_ptr = static_cast<char*>(device_mem_base_) + i * block_bytes_;
        blk.offset = i * block_bytes_;
        cudaEventCreateWithFlags(&blk.event, cudaEventDisableTiming);
        available_blocks_.push_back(i);
        blocks_.push_back(blk);
    }
}

void SecondaryGPUCache::destroy_vmm() {
    for (auto& blk : blocks_) {
        cudaEventDestroy(blk.event);
    }
    cudaFree(device_mem_base_);
}

void SecondaryGPUCache::evict_block_if_needed() {
    
    if (available_blocks_.empty() && !fifo_buffer_.empty()) {
        TIMER_START(eviction_timer);
        size_t evict_id = fifo_buffer_.back();
        fifo_buffer_.pop_back();

        Block& blk = blocks_[evict_id];
        if (cudaEventQuery(blk.event) != cudaSuccess) {
            cudaEventSynchronize(blk.event);
        }

        int h_id = s_to_h_[evict_id];
        s_to_h_.erase(evict_id);
        h_to_s_.erase(h_id);
        available_blocks_.push_back(evict_id);
        TIMER_STOP(eviction_timer, "Evicted a block");
    }
    
}

void SecondaryGPUCache::swap_out(int layer_id,
                                  torch::Tensor gpu_src_keys,
                                  torch::Tensor gpu_src_vals,
                                  const std::vector<std::pair<int, int>>& src_ids) {
    
    TIMER_START(swap_out_timer);
    double mem_find_time_us = 0.0;
    double gpu_to_gpu_time_us = 0.0;
    double host_trf_time_us = 0.0;
    double buffer_mgmt_time_us = 0.0;

    for (const auto& [p_id, h_id] : src_ids) {
        auto t0 = std::chrono::high_resolution_clock::now();
        evict_block_if_needed();

        size_t s_id = available_blocks_.front();
        available_blocks_.pop_front();

        Block& blk = blocks_[s_id];
        void* dst_ptr = blk.device_ptr;
        const void* src_key = gpu_src_keys[p_id].data_ptr();
        const void* src_val = gpu_src_vals[p_id].data_ptr();
        auto t1 = std::chrono::high_resolution_clock::now();
        cudaMemcpyAsync(dst_ptr, src_key, block_bytes_ / 2, cudaMemcpyDeviceToDevice, gpu_to_gpu_stream_);
        cudaMemcpyAsync(static_cast<char*>(dst_ptr) + block_bytes_ / 2,
                        src_val, block_bytes_ / 2, cudaMemcpyDeviceToDevice, gpu_to_gpu_stream_);
        auto t2 = std::chrono::high_resolution_clock::now();

        void* cpu_dst = static_cast<char*>(cpu_cache_ptr_) + h_id * cpu_stride_;
        cudaMemcpyAsync(cpu_dst, dst_ptr, block_bytes_, cudaMemcpyDeviceToHost, gpu_to_host_stream_);
        cudaEventRecord(blk.event, gpu_to_host_stream_);
        auto t3 = std::chrono::high_resolution_clock::now();

        h_to_s_[h_id] = s_id;
        s_to_h_[s_id] = h_id;
        fifo_buffer_.push_front(s_id);
        auto t4 = std::chrono::high_resolution_clock::now();
        mem_find_time_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
        gpu_to_gpu_time_us += std::chrono::duration<double, std::micro>(t2 - t1).count();
        host_trf_time_us += std::chrono::duration<double, std::micro>(t3 - t2).count();
        buffer_mgmt_time_us += std::chrono::duration<double, std::micro>(t4 - t3).count();
    }
    std::cout << "[swap_out timing breakdown] "
              << "mem_find_time: " << mem_find_time_us / 1000.0 << " ms, "
              << "gpu_to_gpu_time: " << gpu_to_gpu_time_us / 1000.0 << " ms, "
              << "host_trf_time: " << host_trf_time_us / 1000.0 << " ms, "
              << "buffer_mgmt_time: " << buffer_mgmt_time_us / 1000.0 << " ms\n";

    TIMER_STOP(swap_out_timer, "Swap out completed");
}

void SecondaryGPUCache::swap_in(int layer_id,
                                 torch::Tensor gpu_dst_keys,
                                 torch::Tensor gpu_dst_vals,
                                 const std::vector<std::pair<int, int>>& dst_ids) {
    TIMER_START(swap_in_timer);
    for (const auto& [h_id, p_id] : dst_ids) {
        auto it = h_to_s_.find(h_id);
        void* dst_key = gpu_dst_keys[p_id].data_ptr();
        void* dst_val = gpu_dst_vals[p_id].data_ptr();

        if (it == h_to_s_.end()) {
            void* cpu_src = static_cast<char*>(cpu_cache_ptr_) + h_id * cpu_stride_;
            cudaMemcpyAsync(dst_key, cpu_src, block_bytes_ / 2, cudaMemcpyHostToDevice, gpu_to_gpu_stream_);
            cudaMemcpyAsync(dst_val, static_cast<char*>(cpu_src) + block_bytes_ / 2,
                            block_bytes_ / 2, cudaMemcpyHostToDevice, gpu_to_gpu_stream_);
        } else {
            Block& blk = blocks_[it->second];
            void* src_ptr = blk.device_ptr;
            cudaMemcpyAsync(dst_key, src_ptr, block_bytes_ / 2, cudaMemcpyDeviceToDevice, gpu_to_gpu_stream_);
            cudaMemcpyAsync(dst_val, static_cast<char*>(src_ptr) + block_bytes_ / 2,
                            block_bytes_ / 2, cudaMemcpyDeviceToDevice, gpu_to_gpu_stream_);
        }
    }
    cudaStreamSynchronize(gpu_to_gpu_stream_);
    TIMER_STOP(swap_in_timer, "Swap in completed");
}

void SecondaryGPUCache::grow_cache(size_t size_in_bytes) {
    size_t new_blocks = size_in_bytes / block_bytes_;
    for (size_t i = 0; i < new_blocks; ++i) {
        Block blk;
        blk.device_ptr = static_cast<char*>(device_mem_base_) + blocks_.size() * block_bytes_;
        blk.offset = blocks_.size() * block_bytes_;
        cudaEventCreateWithFlags(&blk.event, cudaEventDisableTiming);
        available_blocks_.push_back(blocks_.size());
        blocks_.push_back(blk);
    }
    max_blocks_ += new_blocks;
}

void SecondaryGPUCache::shrink_cache(size_t size_in_bytes_to_shrink) {
    size_t num_to_remove = size_in_bytes_to_shrink / block_bytes_;
    size_t removed = 0;
    while (removed < num_to_remove && !fifo_buffer_.empty()) {
        size_t s_id = fifo_buffer_.back();
        fifo_buffer_.pop_back();
        Block& blk = blocks_[s_id];
        cudaEventSynchronize(blk.event);

        int h_id = s_to_h_[s_id];
        s_to_h_.erase(s_id);
        h_to_s_.erase(h_id);
        available_blocks_.push_back(s_id);
        ++removed;
    }
}