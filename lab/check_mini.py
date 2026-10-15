"""Check the Mini-SGLang environment and optionally exercise inference paths.

Run from the repository root on a Linux NVIDIA CUDA machine.
The offline smoke test never passes lab-specific feature flags such as observe.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
results: list[tuple[str, str, str]] = []


def record(name: str, status: str, detail: str) -> None:
    results.append((name, status, detail))
    print(f"[{status:4}] {name}: {detail}", flush=True)


def check_openmp() -> None:
    value = os.environ.get("OMP_NUM_THREADS")
    if value is None:
        record("OMP_NUM_THREADS", "PASS", "unset; using the runtime default")
        return
    parts = [part.strip() for part in value.split(",")]
    valid = all(re.fullmatch(r"[0-9]+", part) and part.lstrip("0") for part in parts)
    if valid:
        record("OMP_NUM_THREADS", "PASS", f"positive-integer list: {value!r}")
    else:
        record(
            "OMP_NUM_THREADS",
            "WARN",
            f"invalid value {value!r}; unset OMP_NUM_THREADS before rerunning",
        )


def check_environment() -> bool:
    python_status = (
        "FAIL" if sys.version_info < (3, 10) else "WARN" if sys.version_info >= (3, 13) else "PASS"
    )
    record("Python", python_status, f"{sys.version.split()[0]} (repository setup recommends 3.12)")
    record("platform", "PASS" if platform.system() == "Linux" else "FAIL", platform.platform())
    record("repository", "PASS" if Path.cwd().resolve() == ROOT else "WARN", str(ROOT))
    check_openmp()

    for package in ("minisgl", "transformers", "flashinfer", "sgl_kernel", "tvm_ffi"):
        try:
            found = importlib.util.find_spec(package) is not None
        except (ImportError, ValueError):
            found = False
        record(f"import {package}", "PASS" if found else "FAIL", "found" if found else "missing")

    if shutil.which("nvidia-smi"):
        try:
            probe = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            record("nvidia-smi", "PASS" if probe.returncode == 0 else "WARN", (probe.stdout or probe.stderr).strip())
        except (OSError, subprocess.TimeoutExpired) as error:
            record("nvidia-smi", "WARN", f"{type(error).__name__}: {error}")
    else:
        record("nvidia-smi", "WARN", "command not found; PyTorch CUDA check below is authoritative")

    try:
        import torch
    except Exception as error:
        record("PyTorch", "FAIL", f"{type(error).__name__}: {error}")
        return False

    record("PyTorch", "PASS", torch.__version__)
    if not torch.cuda.is_available():
        record("CUDA", "FAIL", "torch.cuda.is_available() is False")
        return False

    count = torch.cuda.device_count()
    record("CUDA", "PASS" if count else "FAIL", f"{count} GPU(s), PyTorch CUDA {torch.version.cuda}")
    if not count:
        return False
    for index in range(count):
        try:
            props = torch.cuda.get_device_properties(index)
            free, total = torch.cuda.mem_get_info(index)
            record(
                f"GPU {index}",
                "PASS",
                f"{props.name}, sm_{props.major}{props.minor}, "
                f"free {free / 2**30:.1f}/{total / 2**30:.1f} GiB",
            )
        except Exception as error:
            record(f"GPU {index}", "FAIL", f"{type(error).__name__}: {error}")

    return True


def check_offline(model: str, real_weights: bool, graph: bool) -> None:
    from minisgl.core import SamplingParams
    from minisgl.llm import LLM
    from minisgl.utils import cached_load_hf_config, load_tokenizer

    config = cached_load_hf_config(model)
    record("model config", "PASS", f"{model}: {config.model_type}")
    tokenizer = load_tokenizer(model)
    record("tokenizer", "PASS", f"{type(tokenizer).__name__}, vocab size {len(tokenizer)}")

    class TraceLLM(LLM):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.batches: list[tuple[str, int]] = []

        def _schedule_next_batch(self):
            forward_input = super()._schedule_next_batch()
            if forward_input is not None:
                batch = forward_input.batch
                self.batches.append((batch.phase, sum(req.extend_len for req in batch.reqs)))
            return forward_input

    # 创建 LLM 类
    llm = TraceLLM(
        model,
        use_dummy_weight=not real_weights,
        max_extend_tokens=32,
        max_seq_len_override=256,
        max_running_req=2,
        num_page_override=128,
        cuda_graph_bs=[1, 2] if graph else [],
    )
    try:
        long_prompt = list(range(1000, 1064))
        params = SamplingParams(ignore_eos=True, max_tokens=4)
        # 开始 generate 
        output = llm.generate([long_prompt, [2000, 2001, 2002, 2003]], params)
        lengths = [len(item["token_ids"]) for item in output]
        assert lengths == [4, 4], f"unexpected output lengths: {lengths}"
        phases = [phase for phase, _ in llm.batches]
        assert "prefill" in phases and "decode" in phases, f"missing phase: {phases}"
        assert sum(phase == "prefill" for phase, _ in llm.batches) >= 2, llm.batches
        record("offline prefill/decode", "PASS", f"output lengths {lengths}; batches {llm.batches}")

        llm.batches.clear()
        reused = llm.generate([long_prompt + [3000]], params)
        assert len(reused[0]["token_ids"]) == 4
        new_prefill = sum(tokens for phase, tokens in llm.batches if phase == "prefill")
        assert 0 < new_prefill < len(long_prompt) + 1, llm.batches
        record("radix prefix reuse", "PASS", f"65-token follow-up used {new_prefill} prefill token(s)")
        if graph:
            record("CUDA Graph", "PASS", "enabled for this smoke run; use execution lab for replay/ablation evidence")
    finally:
        llm.shutdown()


def check_server(server_url: str, timeout: float) -> None:
    base = server_url.rstrip("/")
    with urlopen(f"{base}/v1/models", timeout=timeout) as response:
        payload = json.load(response)
    models = payload.get("data", [])
    assert models and models[0].get("id"), payload
    model = models[0]["id"]
    record("HTTP model list", "PASS", model)

    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "Say hello."}],
            "max_tokens": 4,
            "ignore_eos": True,
            "stream": True,
        }
    ).encode()
    request = Request(
        f"{base}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        events = [line.decode("utf-8").strip() for line in response if line.startswith(b"data: ")]
    assert events and events[-1] == "data: [DONE]", events[-3:]
    chunks = [json.loads(event[6:]) for event in events if event != "data: [DONE]"]
    assert chunks and all(chunk.get("choices") for chunk in chunks), events[:3]
    record("HTTP token streaming", "PASS", f"{len(chunks)} SSE chunk(s), ended with [DONE]")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="run GPU offline inference with dummy weights")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="local model directory or HF model ID")
    parser.add_argument("--real-weights", action="store_true", help="load actual weights during --smoke")
    parser.add_argument("--graph", action="store_true", help="enable CUDA Graph during --smoke")
    parser.add_argument("--server-url", help="check an already running Mini-SGLang HTTP server")
    parser.add_argument("--timeout", type=float, default=120, help="HTTP request timeout in seconds")
    parser.add_argument("--verbose", action="store_true", help="show tracebacks for failed checks")
    parser.add_argument("--_offline-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (args.real_weights or args.graph) and not (args.smoke or args._offline_worker):
        parser.error("--real-weights and --graph require --smoke")

    # GPU environment probes initialize CUDA; Engine must initialize it in a fresh process.
    if args._offline_worker:
        try:
            check_offline(args.model, args.real_weights, args.graph)
            return 0
        except Exception as error:
            record("offline smoke", "FAIL", f"{type(error).__name__}: {error}")
            if args.verbose:
                traceback.print_exc()
            return 1

    ready = check_environment()
    if args.smoke:
        if ready and not any(status == "FAIL" for _, status, _ in results):
            try:
                command = [sys.executable, str(Path(__file__).resolve()), "--_offline-worker", "--model", args.model]
                for option in ("real_weights", "graph", "verbose"):
                    if getattr(args, option):
                        command.append("--" + option.replace("_", "-"))
                worker = subprocess.run(command)
                record("offline smoke", "PASS" if worker.returncode == 0 else "FAIL", f"fresh process exit code {worker.returncode}")
            except Exception as error:
                record("offline smoke", "FAIL", f"{type(error).__name__}: {error}")
                if args.verbose:
                    traceback.print_exc()
        else:
            record("offline smoke", "SKIP", "fix the environment failures above first")

    if args.server_url:
        try:
            check_server(args.server_url, args.timeout)
        except Exception as error:
            record("HTTP server", "FAIL", f"{type(error).__name__}: {error}")
            if args.verbose:
                traceback.print_exc()

    passed = sum(status == "PASS" for _, status, _ in results)
    failed = sum(status == "FAIL" for _, status, _ in results)
    warned = sum(status == "WARN" for _, status, _ in results)
    skipped = sum(status == "SKIP" for _, status, _ in results)
    print(f"\nSummary: {passed} PASS, {failed} FAIL, {warned} WARN, {skipped} SKIP")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
