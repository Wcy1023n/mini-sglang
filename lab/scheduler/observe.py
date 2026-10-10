"""Scheduler lab workload and behavioral checks. Run on a Linux CUDA server.

The external [sched] trace is separate from the mini scheduler's opt-in
[mini-observe] log that the student adds in Phase 0.
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


class BurstTraceLLM(TraceLLM):
    """Submit the whole comparison workload before its first scheduling step."""

    def offline_receive_msg(self, blocking: bool = False):
        if not self.pending_requests:
            return super().offline_receive_msg(blocking=blocking)
        messages = []
        while self.pending_requests:
            messages.extend(super().offline_receive_msg(blocking=False))
        return messages


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


def check_long_first(trace: list[tuple[str, tuple[int, ...], int]]) -> None:
    prefill = [reqs for phase, reqs, _ in trace if phase == "prefill"]
    assert sum(0 in reqs for reqs in prefill) == 7, prefill
    assert all(any(uid in reqs for reqs in prefill) for uid in range(1, 7)), prefill


def check_stage_order(
    trace: list[tuple[str, tuple[int, ...], int]], expected: str
) -> None:
    assert trace[:6] == [("prefill", (0,), 32)] * 6, trace[:8]
    if expected == "prefill-first":
        assert trace[6][:2] == ("prefill", (1, 2, 3, 4, 5, 6)), trace[6:8]
    else:
        assert all(
            phase == "decode" and reqs == (0,)
            for phase, reqs, _ in trace[6:15]
        ), trace[6:16]
        assert trace[15][:2] == ("prefill", (1, 2, 3, 4, 5, 6)), trace[15:17]


def check_priority_baseline(trace: list[tuple[str, tuple[int, ...], int]]) -> None:
    prefill = [reqs for phase, reqs, _ in trace if phase == "prefill"]
    assert prefill[0] == (0, 1), prefill
    assert sorted(uid for reqs in prefill for uid in reqs) == [0, 1, 2, 3], prefill


def check_shortest_baseline(trace: list[tuple[str, tuple[int, ...], int]]) -> None:
    prefill = [reqs for phase, reqs, _ in trace if phase == "prefill"]
    assert prefill[0] == (0,), prefill
    assert all(any(uid in reqs for reqs in prefill[1:]) for uid in range(1, 5)), prefill


def check_lpm_baseline(trace: list[tuple[str, tuple[int, ...], int]]) -> None:
    prefill = [reqs for phase, reqs, _ in trace if phase == "prefill"]
    assert prefill[0] == (0, 1), prefill
    assert trace[0][2] == 128, trace[:3]
    assert sorted(set(uid for reqs in prefill for uid in reqs)) == [0, 1, 2], prefill


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument(
        "--case",
        choices=[
            "mixed", "long-first", "stage-order", "no-decode", "admission",
            "chunk-limit", "priority-theory", "shortest-theory", "lpm-theory",
        ],
        default="mixed",
    )
    ap.add_argument("--budget", type=int, default=32, help="prefill token budget")
    ap.add_argument("--observe", type=int, choices=[0, 1], default=0)
    ap.add_argument("--shared-prefix", type=int, default=0, help="observe prefix-cache reuse")
    ap.add_argument("--interval", type=int, default=None)
    ap.add_argument("--prefill-max-requests", type=int, default=None)
    ap.add_argument("--chunk-size", type=int, default=None)
    ap.add_argument("--check", action="store_true", help="assert the expected observable behavior")
    ap.add_argument("--expect-chunks", type=int, choices=[1, 7])
    ap.add_argument(
        "--expect-stage-order", choices=["prefill-first", "decode-first"]
    )
    args = ap.parse_args()
    if args.expect_chunks is not None and args.case != "long-first":
        ap.error("--expect-chunks requires --case long-first")
    if args.expect_stage_order is not None and (args.case != "stage-order" or args.budget != 32):
        ap.error("--expect-stage-order requires --case stage-order --budget 32")
    checked_budget = {
        "chunk-limit": 64,
        "priority-theory": 16,
        "shortest-theory": 24,
        "lpm-theory": 128,
    }.get(args.case, 32)
    if args.check and args.budget != checked_budget:
        ap.error(f"--check requires budget {checked_budget} for this case")
    if args.check and args.case == "stage-order":
        ap.error("use --expect-stage-order for the stage-order workload")

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
    elif args.case in ("long-first", "stage-order"):
        prompts = [
            [1000 + i for i in range(200 if args.case == "long-first" else 192)],
            *[[2000 + i * 4 + j for j in range(4)] for i in range(6)],
        ]
        params = SamplingParams(ignore_eos=True, max_tokens=10)
    elif args.case == "no-decode":
        prompts = [[1000 + i for i in range(200)]]
        params = SamplingParams(ignore_eos=True, max_tokens=4)
    elif args.case == "chunk-limit":
        prompts = [[2000 + i for i in range(4)], [1000 + i for i in range(80)]]
        params = SamplingParams(ignore_eos=True, max_tokens=8)
    elif args.case == "priority-theory":
        prompts = [[2000 + i * 8 + j for j in range(8)] for i in range(4)]
        params = SamplingParams(ignore_eos=True, max_tokens=5)
    elif args.case == "shortest-theory":
        prompts = [[1000 + i for i in range(24)]] + [
            [2000 + i * 4 + j for j in range(4)] for i in range(4)
        ]
        params = SamplingParams(ignore_eos=True, max_tokens=5)
    elif args.case == "lpm-theory":
        prompts = [
            [5000 + i for i in range(64)],
            [2000 + i for i in range(64)] + [4000 + i for i in range(64)],
            [1000 + i for i in range(128)] + [3000 + i for i in range(64)],
        ]
        params = SamplingParams(ignore_eos=True, max_tokens=5)
    else:
        # Six requests fit the token budget, so batch size exposes admission limits.
        prompts = [[2000 + i * 4 + j for j in range(4)] for i in range(6)]
        params = SamplingParams(ignore_eos=True, max_tokens=5)

    options = dict(
        use_dummy_weight=True,
        observe=args.observe,
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

    use_burst = args.case in ("priority-theory", "shortest-theory", "lpm-theory")
    llm = (BurstTraceLLM if use_burst else TraceLLM)(args.model, **options)
    if args.case == "lpm-theory":
        llm.generate(
            [[1000 + i for i in range(128)], [2000 + i for i in range(64)]],
            SamplingParams(ignore_eos=True, max_tokens=5),
        )
        llm.trace.clear()
    results = llm.generate(prompts, params)
    assert len(results) == len(prompts)
    expected_lengths = (
        [sp.max_tokens for sp in params]
        if isinstance(params, list)
        else [params.max_tokens] * len(prompts)
    )
    assert [len(result["token_ids"]) for result in results] == expected_lengths
    if args.expect_chunks is not None:
        actual = sum(
            phase == "prefill" and 0 in reqs for phase, reqs, _ in llm.trace
        )
        assert actual == args.expect_chunks, (actual, llm.trace)
        print("[check] chunks PASS")
    if args.expect_stage_order is not None:
        check_stage_order(llm.trace, args.expect_stage_order)
        print("[check] stage-order PASS")
    if args.check:
        if args.case == "mixed":
            check_mixed(llm.trace, args.interval, args.prefill_max_requests)
        elif args.case == "no-decode":
            phases = [phase for phase, _, _ in llm.trace]
            assert phases[:4] == ["prefill"] * 4, phases[:8]
        elif args.case == "chunk-limit":
            check_chunk_limit(llm.trace, args.chunk_size)
        elif args.case == "long-first":
            check_long_first(llm.trace)
        elif args.case == "priority-theory":
            check_priority_baseline(llm.trace)
        elif args.case == "shortest-theory":
            check_shortest_baseline(llm.trace)
        elif args.case == "lpm-theory":
            check_lpm_baseline(llm.trace)
        else:
            check_admission(llm.trace, args.prefill_max_requests)
        print("[check] PASS")
    print("[observe] done")


if __name__ == "__main__":
    main()
