"""perf_cuda / compare_memory_kernel_perf 的第一个调用方。

Phase 1 需要先魔改 `benchmark/perf.py` 的 `perf_cuda`，加一个 `warmups: int = 1` 参数，
把原来写死的一次 warmup `f()` 换成 `for _ in range(warmups): f()`。
本脚本用 `warmups=0/1/5` 观察 GPU kernel 的冷启动开销。

跑法（需要 GPU）：
    python lab/bench_profiler/microbench.py
"""
from __future__ import annotations

import platform

if platform.system() != "Linux":
    raise RuntimeError("This experiment must run on the Linux CUDA server.")

import torch

if not torch.cuda.is_available():
    raise RuntimeError("This experiment requires an NVIDIA CUDA GPU.")

from minisgl.benchmark.perf import compare_memory_kernel_perf, perf_cuda


def main() -> None:
    torch.cuda.set_device(0)
    N = 1 << 22  # 4M 个 float32
    x = torch.randn(N, device="cuda")
    y = torch.randn(N, device="cuda")

    # 纯 copy：每次调用读 N*4 + 写 N*4 bytes
    for w in (0, 1, 5):
        ms = perf_cuda(lambda: y.copy_(x), warmups=w)
        gbs = (2 * N * 4) / (ms * 1e6)
        print(f"warmups={w}: {ms:.3f} ms/call -> {gbs:.1f} GB/s  (3090 峰值约 936 GB/s)")

    # 对比「分配新输出」vs「in-place」，二者搬动字节数相同（读 x + 读 y + 写结果）
    compare_memory_kernel_perf(
        baseline=lambda: torch.add(x, y),
        our_impl=lambda: torch.add(x, y, out=y),
        memory_footprint=3 * N * 4,
        description="[elementwise add] alloc vs inplace ",
    )


if __name__ == "__main__":
    main()
