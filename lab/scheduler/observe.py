"""离线驱动 Scheduler 的最小 workload，配合 scheduler.py 里的 per-step 观测一起用。

`observe.py` 构造一个「1 长 + N 短」的混合负载，并把 prefill budget 设得很小，
强制长 prompt 被 chunked prefill 切成多块，从而让 prefill 和 decode 在时间上交错。
这样你在 scheduler.py 里加的观测日志就能看清 prefill-first vs decode-first 的差别。

三种变体负载，分别对应 scheduler.ipynb 里的排序魔改：

    python lab/scheduler/observe.py                 # 默认：1 长 + N 短（Phase 0/1/2/5 用）
    python lab/scheduler/observe.py --priority      # 递进 priority（Phase 4 用，需先给 SamplingParams 加 priority）
    python lab/scheduler/observe.py --shared-prefix 100   # 共享前缀 + 不同尾巴（Phase 6 LPM 用）

注意：本脚本本身不打印调度轨迹。轨迹来自你在 scheduler.py 的
`_schedule_next_batch` 里加的 per-step 日志（见 lab/scheduler/scheduler.ipynb 的 Phase 0）。
"""
from __future__ import annotations

import platform

if platform.system() != "Linux":
    raise RuntimeError("This experiment must run on the Linux CUDA server.")

import torch

if not torch.cuda.is_available():
    raise RuntimeError("This experiment requires an NVIDIA CUDA GPU.")

import argparse

from minisgl.core import SamplingParams
from minisgl.llm import LLM


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--budget", type=int, default=32, help="max_extend_tokens（prefill budget）")
    ap.add_argument("--long-len", type=int, default=200)
    ap.add_argument("--num-short", type=int, default=6)
    ap.add_argument("--decode-len", type=int, default=30)
    ap.add_argument(
        "--priority",
        action="store_true",
        help="给每条请求分配递进的 priority（Phase 4 用；要求 SamplingParams 已有 priority 字段）",
    )
    ap.add_argument(
        "--shared-prefix",
        type=int,
        default=0,
        help=">0 时生成「共享前缀 + 不同尾巴」的负载（Phase 6 LPM 用）",
    )
    args = ap.parse_args()

    if args.shared_prefix > 0:
        # LPM 负载：所有请求共享同一个 prefix，尾巴各不相同
        prefix = list(range(args.shared_prefix))
        prompt_ids = [prefix + [9000 + i] * 4 for i in range(args.num_short + 1)]
    else:
        # R0：一条长 prompt（会被 chunked prefill 切成多块）
        # R1..RN：若干短 prompt
        prompt_ids = [
            [1000 + i % 5000 for i in range(args.long_len)],
            *[[2000 + i for i in range(4)]] * args.num_short,
        ]

    if args.priority:
        # 递进 priority：R0(长) priority=0 最低，R1..RN 依次升高 → 高 priority 先 prefill
        sps = [
            SamplingParams(ignore_eos=True, max_tokens=args.decode_len, priority=i)
            for i in range(len(prompt_ids))
        ]
    else:
        sps = SamplingParams(ignore_eos=True, max_tokens=args.decode_len)

    llm = LLM(
        args.model,
        use_dummy_weight=True,          # 只看调度，不看权重；随机权重即可
        max_extend_tokens=args.budget,  # 关键：设小才能触发 chunked prefill
        max_seq_len_override=1024,
        max_running_req=args.num_short + 1,
    )
    llm.generate(prompt_ids, sps)
    print("[observe] done")


if __name__ == "__main__":
    main()
