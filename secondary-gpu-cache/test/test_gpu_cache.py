import torch
from secondary_gpu_cache import handle as SecondaryGPUCache
import math
import time

block_size = 32
num_kv_heads = 16
head_size = 16
dtype = torch.float16
dtype_size = torch.tensor([], dtype=dtype).element_size()
fraction = 0.5
device_id = 0
num_blocks = 102400
cpu_stride = 2 * block_size * num_kv_heads * head_size * dtype_size
block_shape = (2, num_blocks, block_size, num_kv_heads, head_size)

cpu_buffer = torch.zeros(block_shape, dtype=dtype, pin_memory=True)

# Create secondary GPU cache
cache = SecondaryGPUCache(
    block_size,
    num_kv_heads,
    head_size,
    cpu_buffer,
    cpu_stride,
    dtype_size,
    fraction,
    device_id
)

# Create dummy GPU source blocks
gpu_src = torch.randn(block_shape, dtype=dtype, device="cuda")
gpu_src.fill_(0.5)

src_ids = [(i, i) for i in range(80)]


for i in range(3):
    t = time.time_ns()
    cache.swap_out(0, gpu_src[0], gpu_src[1], src_ids)
    print("Swap out time for ", i, (time.time_ns() - t) / 1e6, "ms. ", " moved ", (math.prod(block_shape) * len(src_ids) /num_blocks ) / (1<<20), "MB")
cpu_buffer.fill_(0.7)

# Prepare dst_ids for swap_in: (host_id, primary_id)
dst_ids = [(i, i) for i in range(80)]

# Create GPU destination blocks
gpu_dst = torch.zeros_like(gpu_src)

# Swap in from secondary GPU cache
t = time.time_ns()
cache.swap_in(0, gpu_dst[0], gpu_dst[1], dst_ids)
print("Swap in time: ", (time.time_ns() - t) / 1e6, "ms")
print("Swap in/out test completed successfully.")
