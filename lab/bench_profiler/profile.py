"""Phase 3 的 profiler 驱动：驱动魔改后的 `Scheduler.report_profile`。

前提：已经按 bench_profiler.ipynb Phase 3 在 `scheduler/scheduler.py` 里加了
`self._profile` 容器、`_forward` 里的 CUDA Event 计时、以及 `report_profile` 方法。

`LLM` 是 `Scheduler` 的子类，所以 `llm.report_profile()` 直接可用。
用 dummy weight（不需要真实权重），prefill budget 设小以区分 prefill/decode。

跑法（需要 GPU）：
    python lab/bench_profiler/profile.py
"""
from __future__ import annotations

import platform

if platform.system() != "Linux":
    raise RuntimeError("This experiment must run on the Linux CUDA server.")

import torch

if not torch.cuda.is_available():
    raise RuntimeError("This experiment requires an NVIDIA CUDA GPU.")

from minisgl.core import SamplingParams
from minisgl.llm import LLM


def main() -> None:
    llm = LLM(
        "Qwen/Qwen3-0.6B",
        use_dummy_weight=True,
        max_extend_tokens=32,  # 设小，强制长 prompt 走 chunked prefill
        max_seq_len_override=1024,
        max_running_req=7,
    )
    # 1 条 200-token 长 prompt（拆 7 chunk）+ 6 条 4-token 短 prompt
    prompts = [[1000 + i % 5000 for i in range(200)]] + [[i + 2000 for i in range(4)]] * 6
    llm.generate(prompts, SamplingParams(ignore_eos=True, max_tokens=30))
    llm.report_profile()


if __name__ == "__main__":
    main()
