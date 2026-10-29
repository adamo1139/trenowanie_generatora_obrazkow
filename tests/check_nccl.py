"""NCCL sanity + all-reduce bandwidth check, independent of training (no data, no model):

  torchrun --nproc_per_node 8 tests/check_nccl.py
  NCCL_DEBUG=INFO torchrun --nproc_per_node 8 tests/check_nccl.py   # shows which transport NCCL picks (SHM / P2P / NET)

The last line estimates the DDP gradient all-reduce time per training step (bf16 grads, --grad-comm bf16).
"""
import os
import time

import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

x = torch.ones(1, device="cuda")
dist.all_reduce(x)
assert x.item() == world, x.item()
if rank == 0:
    print(f"all_reduce OK on {world} ranks ({torch.cuda.get_device_name()})", flush=True)

busbw = {}
for mb in (1, 16, 64, 256):
    t = torch.randn(mb * 2**20 // 2, dtype=torch.bfloat16, device="cuda")
    for _ in range(3):
        dist.all_reduce(t)
    torch.cuda.synchronize()
    start, n = time.time(), 10
    for _ in range(n):
        dist.all_reduce(t)
    torch.cuda.synchronize()
    dt = (time.time() - start) / n
    busbw[mb] = mb * 2**20 * 2 * (world - 1) / world / dt / 1e9
    if rank == 0:
        print(f"all_reduce {mb:4d} MB bf16: {dt * 1000:8.2f} ms   bus bandwidth {busbw[mb]:6.1f} GB/s", flush=True)

if rank == 0 and world > 1:
    for name, params in (("50m", 63e6), ("116m", 116e6)):
        size = params * 2  # bf16 gradients
        print(f"~{size * 2 * (world - 1) / world / (busbw[256] * 1e9) * 1000:.0f} ms of gradient all-reduce per step "
              f"for --size {name} (partly overlapped with backward)")
dist.destroy_process_group()
