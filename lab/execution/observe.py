"""GPU-only workload for the execution lab; does not alter mini-sglang."""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from collections import Counter
from pathlib import Path

if platform.system() != "Linux":
    raise RuntimeError("Run this lab on a Linux CUDA server")

import torch

if not torch.cuda.is_available():
    raise RuntimeError("This lab requires an NVIDIA CUDA GPU")

from minisgl.core import SamplingParams
from minisgl.env import ENV
from minisgl.llm import LLM


class TraceLLM(LLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batches: list[dict[str, int | str]] = []

    def _schedule_next_batch(self):
        forward_input = super()._schedule_next_batch()
        if forward_input is not None:
            batch = forward_input.batch
            self.batches.append(
                {
                    "phase": batch.phase,
                    "size": batch.size,
                    "padded_size": batch.padded_size,
                    "tokens": sum(req.extend_len for req in batch.reqs),
                }
            )
        return forward_input


def workload(round_id: int):
    base = 1000 + round_id * 1000
    prompts = [
        [base + i for i in range(80)],
        [base + 200 + i for i in range(80)],
    ]
    params = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=12)
    return prompts, params


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--graph", choices=["on", "off"], required=True)
    ap.add_argument("--trace", type=Path)
    ap.add_argument("--summary", type=Path)
    ap.add_argument("--check-markers", action="store_true")
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()
    if args.bench and args.trace:
        ap.error("run profiler and benchmark separately")

    llm = TraceLLM(
        args.model,
        use_dummy_weight=True,
        max_extend_tokens=32,
        max_seq_len_override=256,
        max_running_req=4,
        cuda_graph_max_bs=4 if args.graph == "on" else 0,
    )
    overlap = "off" if ENV.DISABLE_OVERLAP_SCHEDULING else "on"
    if args.bench:
        durations = []
        for round_id in range(6):
            llm.batches.clear()
            prompts, params = workload(round_id)
            torch.cuda.synchronize()
            start = time.perf_counter()
            results = llm.generate(prompts, params)
            torch.cuda.synchronize()
            if round_id:
                durations.append((time.perf_counter() - start) * 1000)
        assert all(len(item["token_ids"]) == 12 for item in results)
        summary = {
            "graph": args.graph,
            "overlap": overlap,
            "median_ms": statistics.median(durations),
            "runs_ms": durations,
            "batches": llm.batches,
        }
    else:
        prompts, params = workload(0)
        if args.trace:
            from torch.profiler import ProfilerActivity, profile

            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                results = llm.generate(prompts, params)
                torch.cuda.synchronize()
            args.trace.parent.mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(str(args.trace))
            markers = Counter({event.key: event.count for event in prof.key_averages()})
        else:
            results = llm.generate(prompts, params)
            markers = Counter()
        assert all(len(item["token_ids"]) == 12 for item in results)
        phases = Counter(batch["phase"] for batch in llm.batches)
        assert phases["prefill"] > 1 and phases["decode"] > 0, llm.batches
        if args.check_markers:
            if not args.trace:
                ap.error("--check-markers requires --trace")
            required = {"lab.eager_forward", "lab.prepare_batch", "lab.process_result"}
            if args.graph == "on":
                required.add("lab.graph_replay")
            missing = sorted(name for name in required if markers[name] == 0)
            assert not missing, f"missing profiler ranges: {missing}"
            if args.graph == "off":
                assert markers["lab.graph_replay"] == 0, markers
        summary = {
            "graph": args.graph,
            "overlap": overlap,
            "token_ids": [item["token_ids"] for item in results],
            "batches": llm.batches,
            "markers": {name: markers[name] for name in sorted(markers) if name.startswith("lab.")},
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
