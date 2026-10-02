"""Scheduler lab workload and behavioral checks. Run on a Linux CUDA server.

The trace is collected outside the scheduler implementation so the experiment
requires no permanent logging in production code.
"""
from __future__ import annotations

import argparse
import platform

if platform.system() != "Linux":
    raise RuntimeError("This experiment must run on the Linux CUDA server.")

import torch

if not torch.cuda.is_available():
    raise RuntimeError("This experiment requires an NVIDIA CUDA GPU.")

from minisgl.core import SamplingParams
from minisgl.llm import LLM


class TraceLLM(LLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace: list[tuple[str, tuple[int, ...], int]] = []

    def _schedule_next_batch(self):
        forward_input = super()._schedule_next_batch()
        if forward_input is not None:
            batch = forward_input.batch
            step = (
                batch.phase,
                tuple(req.uid for req in batch.reqs),
                sum(req.extend_len for req in batch.reqs),
            )
            self.trace.append(step)
            print(f"[sched] {step[0]:7s} reqs={list(step[1])} tokens={step[2]}", flush=True)
        return forward_input


def check_mixed(
    trace: list[tuple[str, tuple[int, ...], int]],
    interval: int | None,
    limit: int | None,
) -> None:
    phases = [phase for phase, _, _ in trace]
    if limit == 1:
        prefill = [reqs for phase, reqs, _ in trace if phase == "prefill"]
        assert all(len(reqs) <= 1 for reqs in prefill), prefill
        assert prefill[0] == (0,) and prefill.count((1,)) > 1, prefill
        return
    assert trace[0][:2] == ("prefill", (0, 1)), trace[:4]
    if interval in (None, 0):
        assert phases[:4] == ["prefill"] * 4, phases[:8]
    elif interval == 2:
        assert phases[:4] == ["prefill", "decode", "decode", "prefill"], phases[:8]
        assert trace[1][1] == trace[2][1] == (0,), trace[:4]
        assert trace[3][1] == (1,), trace[:4]
    else:
        raise ValueError("--check supports --interval 0 or 2 for the mixed case")


def check_admission(
    trace: list[tuple[str, tuple[int, ...], int]], limit: int | None
) -> None:
    prefill = [reqs for phase, reqs, _ in trace if phase == "prefill"]
    admitted = [uid for reqs in prefill for uid in reqs]
    assert sorted(admitted) == list(range(6)), prefill
    if limit is None:
        assert prefill == [tuple(range(6))], prefill
    elif limit == 2:
        assert len(prefill) >= 3 and all(len(reqs) <= 2 for reqs in prefill), prefill
    else:
        raise ValueError("--check supports --prefill-max-requests 2 for the admission case")


def check_chunk_limit(
    trace: list[tuple[str, tuple[int, ...], int]], chunk_size: int | None
) -> None:
    prefill = [(reqs, tokens) for phase, reqs, tokens in trace if phase == "prefill"]
    assert sum(tokens for _, tokens in prefill) == 84, prefill
    assert prefill[0][0] == (0, 1), prefill
    if chunk_size is None:
        assert prefill[0][1] == 64, prefill
    elif chunk_size == 16:
        assert prefill[0][1] == 16, prefill
        assert all(tokens <= 16 for _, tokens in prefill), prefill
        assert sum(1 in reqs for reqs, _ in prefill) > 2, prefill
    else:
        raise ValueError("--check supports --chunk-size 16 for the chunk-limit case")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument(
        "--case", choices=["mixed", "long-first", "no-decode", "admission", "chunk-limit"], default="mixed"
    )
    ap.add_argument("--budget", type=int, default=32, help="prefill token budget")
    ap.add_argument("--shared-prefix", type=int, default=0, help="observe prefix-cache reuse")
    ap.add_argument("--interval", type=int, default=None)
    ap.add_argument("--prefill-max-requests", type=int, default=None)
    ap.add_argument("--chunk-size", type=int, default=None)
    ap.add_argument("--check", action="store_true", help="assert the expected observable behavior")
    args = ap.parse_args()
    if args.check and args.budget != (64 if args.case == "chunk-limit" else 32):
        ap.error("--check requires budget 64 for chunk-limit and 32 for other cases")
    if args.check and args.case == "long-first":
        ap.error("--case long-first is an observation-only workload")

    if args.shared_prefix > 0:
        if args.case != "mixed" or args.check:
            ap.error("--shared-prefix is an observation-only workload")
        prefix = list(range(args.shared_prefix))
        prompts = [prefix + [9000 + i] * 4 for i in range(7)]
        params = SamplingParams(ignore_eos=True, max_tokens=5)
    elif args.case == "mixed":
        # R0 enters decode while R1 still needs prefill chunks.
        prompts = [[2000, 2001, 2002, 2003], [1000 + i for i in range(200)]]
        params = [
            SamplingParams(ignore_eos=True, max_tokens=24),
            SamplingParams(ignore_eos=True, max_tokens=4),
        ]
    elif args.case == "long-first":
        prompts = [
            [1000 + i for i in range(200)],
            *[[2000 + i * 4 + j for j in range(4)] for i in range(6)],
        ]
        params = SamplingParams(ignore_eos=True, max_tokens=10)
    elif args.case == "no-decode":
        prompts = [[1000 + i for i in range(200)]]
        params = SamplingParams(ignore_eos=True, max_tokens=4)
    elif args.case == "chunk-limit":
        prompts = [[2000 + i for i in range(4)], [1000 + i for i in range(80)]]
        params = SamplingParams(ignore_eos=True, max_tokens=8)
    else:
        # Six requests fit the token budget, so batch size exposes admission limits.
        prompts = [[2000 + i * 4 + j for j in range(4)] for i in range(6)]
        params = SamplingParams(ignore_eos=True, max_tokens=5)

    options = dict(
        use_dummy_weight=True,
        max_extend_tokens=args.budget,
        max_seq_len_override=1024,
        max_running_req=8,
        cuda_graph_max_bs=8,
    )
    if args.interval is not None:
        options["prefill_decode_interval"] = args.interval
    if args.prefill_max_requests is not None:
        options["prefill_max_requests"] = args.prefill_max_requests
    if args.chunk_size is not None:
        options["chunked_prefill_size"] = args.chunk_size

    llm = TraceLLM(args.model, **options)
    results = llm.generate(prompts, params)
    assert len(results) == len(prompts)
    expected_lengths = (
        [sp.max_tokens for sp in params]
        if isinstance(params, list)
        else [params.max_tokens] * len(prompts)
    )
    assert [len(result["token_ids"]) for result in results] == expected_lengths
    if args.check:
        if args.case == "mixed":
            check_mixed(llm.trace, args.interval, args.prefill_max_requests)
        elif args.case == "no-decode":
            phases = [phase for phase, _, _ in llm.trace]
            assert phases[:4] == ["prefill"] * 4, phases[:8]
        elif args.case == "chunk-limit":
            check_chunk_limit(llm.trace, args.chunk_size)
        else:
            check_admission(llm.trace, args.prefill_max_requests)
        print("[check] PASS")
    print("[observe] done")


if __name__ == "__main__":
    main()
