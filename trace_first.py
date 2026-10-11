#!/usr/bin/env python3
"""Interactive first single-request trace for mini-sglang."""
from __future__ import annotations

import argparse
import os
import sys
from functools import wraps
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MINISGL_DISABLE_OVERLAP_SCHEDULING", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["PATH"] = "/root/miniconda3/bin:" + os.environ.get("PATH", "")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "python"))

import torch

from minisgl.core import SamplingParams
from minisgl.engine.engine import Engine
from minisgl.llm import LLM
from minisgl.scheduler.prefill import PrefillManager
from minisgl.scheduler.scheduler import Scheduler


def request_state(reqs: Any) -> list[tuple[int, int, int, int]]:
    return [(r.uid, r.cached_len, r.device_len, r.remain_len) for r in reqs]


def trace_stop(label: str, before: Callable[..., None] | None = None):
    def decorator(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            print(f"\n{'=' * 72}\nTRACE: {label}\n{'=' * 72}", flush=True)
            if before is not None:
                before(*args, **kwargs)
            breakpoint()
            return function(*args, **kwargs)

        return wrapped

    return decorator


def show_user_msg(_self: Scheduler, msg) -> None:
    print(f"message = {type(msg).__name__}, uid = {getattr(msg, 'uid', None)}", flush=True)
    input_ids = getattr(msg, "input_ids", None)
    if input_ids is not None:
        print(f"input_ids.shape = {tuple(input_ids.shape)}", flush=True)


def show_prefill(self: PrefillManager, prefill_budget: int) -> None:
    print(f"prefill_budget = {prefill_budget}", flush=True)
    print(f"pending = {[p.uid for p in self.pending_list]}", flush=True)


def show_batch(_self: Scheduler, batch) -> None:
    print(f"phase = {batch.phase}", flush=True)
    print(f"requests(uid, cached_len, device_len, remain_len) = {request_state(batch.reqs)}", flush=True)


def show_forward_input(_self: Scheduler, forward_input) -> None:
    show_batch(_self, forward_input.batch)
    print(f"input_tuple shapes = {[tuple(x.shape) for x in forward_input.input_tuple]}", flush=True)
    print(f"write_tuple shapes = {[tuple(x.shape) for x in forward_input.write_tuple]}", flush=True)


def show_forward(_self: Engine, batch, _args) -> None:
    show_batch(_self, batch)
    print(f"input_ids.shape = {tuple(batch.input_ids.shape)}", flush=True)
    print(f"positions.shape = {tuple(batch.positions.shape)}", flush=True)
    print(f"out_loc.shape = {tuple(batch.out_loc.shape)}", flush=True)


def show_result(_self: Scheduler, last_data) -> None:
    if last_data is None:
        print("last_data = None", flush=True)
        return
    batch = last_data[0].batch
    print(f"completed phase = {batch.phase}", flush=True)
    print(f"requests(uid, cached_len, device_len, remain_len) = {request_state(batch.reqs)}", flush=True)


def install_trace_points() -> None:
    # These wrappers exist only in this process; mini-sglang source is unchanged.
    LLM.generate = trace_stop("LLM.generate: request enters offline scheduler")(LLM.generate)
    Scheduler._process_one_msg = trace_stop("Scheduler._process_one_msg: UserMsg arrives", show_user_msg)(
        Scheduler._process_one_msg
    )
    PrefillManager.schedule_next_batch = trace_stop("PrefillManager.schedule_next_batch", show_prefill)(
        PrefillManager.schedule_next_batch
    )
    Scheduler._prepare_batch = trace_stop("Scheduler._prepare_batch: make GPU metadata", show_batch)(
        Scheduler._prepare_batch
    )
    Scheduler._forward = trace_stop("Scheduler._forward: invoke Engine", show_forward_input)(
        Scheduler._forward
    )
    Engine.forward_batch = trace_stop("Engine.forward_batch: eager model forward + sampling", show_forward)(
        Engine.forward_batch
    )
    Scheduler._process_last_data = trace_stop("Scheduler._process_last_data: consume next token", show_result)(
        Scheduler._process_last_data
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/workspace/models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompt", default="Explain KV cache in one short sentence.")
    parser.add_argument("--max-tokens", type=int, default=4)
    args = parser.parse_args()

    print("Environment:")
    print(f"  model = {args.model}")
    print(f"  torch = {torch.__version__}")
    print("  CUDA inspection is deferred so Engine initializes CUDA first.")
    print("Constructing LLM; model loading happens before tracing begins...")

    llm = LLM(
        model_path=args.model,
        dtype=torch.bfloat16,
        max_running_req=1,
        max_extend_tokens=256,
        page_size=1,
        cache_type="naive",
        cuda_graph_max_bs=0,
    )
    print(f"  cuda = {torch.cuda.is_available()}")
    print(f"  gpu = {torch.cuda.get_device_name(0)}")

    install_trace_points()
    print("\nTrace ready. At each stop use 'where', inspect variables, then type 'c'.")
    print("Useful: p batch.phase | p batch.reqs | p self.pending_list | p self.running_reqs")
    result = llm.generate([args.prompt], SamplingParams(temperature=0.0, max_tokens=args.max_tokens))
    print("\nFinal result:")
    print(result)


if __name__ == "__main__":
    main()
