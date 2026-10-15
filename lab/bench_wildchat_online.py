"""Real WildChat prompts against mini's HTTP server; no CUDA import in the client.

Prepare: python lab/bench_wildchat_online.py --prepare-only
Run:     python lab/bench_wildchat_online.py --num-requests 1000 --request-rate 2

This is a serving workload, not a model quality evaluation or a faithful
production arrival trace. Output length is controlled with ignore_eos=True.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DATASET = "allenai/WildChat-4.8M"
SOURCE = f"https://huggingface.co/datasets/{DATASET}"
VIEWER = "https://datasets-server.huggingface.co"


def get_json(endpoint, **params):
    url = f"{VIEWER}/{endpoint}?{urllib.parse.urlencode(params)}"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                return json.load(response)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def prepare(path, count):
    if path.exists():
        print(f"Using cached data: {path}; remove it to download a larger pool.")
        return
    splits = get_json("splits", dataset=DATASET)["splits"]
    split = next(s for s in splits if s["split"] == "train")
    records, seen, offset = [], set(), 0
    while len(records) < count:
        page = get_json("rows", dataset=DATASET, config=split["config"],
                        split="train", offset=offset, length=100)
        if not page["rows"]:
            raise RuntimeError(f"Only found {len(records)} eligible prompts")
        for entry in page["rows"]:
            if entry.get("truncated_cells"):
                continue
            conversation = entry["row"].get("conversation") or []
            user = next((t for t in conversation if t.get("role") == "user"), None)
            if not user or user.get("language") not in {"English", "Chinese"}:
                continue
            text = user.get("content") or ""
            if not text.strip() or user.get("redacted") or user.get("toxic"):
                continue
            digest = hashlib.sha256(text.encode()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            records.append({"row_idx": entry["row_idx"], "prompt": text,
                            "language": user["language"], "sha256": digest})
            if len(records) == count:
                break
        offset += len(page["rows"])
        print(f"[dataset] scanned={offset} selected={len(records)}/{count}", flush=True)
        if offset >= page["num_rows_total"] and len(records) < count:
            raise RuntimeError("Dataset exhausted before the requested sample count")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".jsonl.partial")
    temporary.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    temporary.replace(path)
    manifest = {"source": SOURCE, "config": split["config"], "split": "train",
                "selection": "First distinct eligible English/Chinese first-user prompts",
                "count": count, "rows_scanned": offset,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "downloaded_at": datetime.now(timezone.utc).isoformat()}
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))


def stats(values):
    if not values:
        return None
    values = sorted(values)
    return {"mean": sum(values) / len(values),
            **{f"p{p}": values[min(len(values) - 1, math.ceil(len(values) * p / 100) - 1)]
               for p in (50, 90, 99)}}


async def run(args):
    from openai import AsyncOpenAI
    from transformers import AutoTokenizer

    async with AsyncOpenAI(base_url=args.base_url, api_key="dummy", max_retries=0,
                          timeout=args.timeout) as client:
        model = args.model or (await client.models.list()).data[0].id
        tokenizer = AutoTokenizer.from_pretrained(model)
        pool = []
        for line in args.data.read_text().splitlines():
            record = json.loads(line)
            messages = [{"role": "user", "content": record["prompt"]}]
            input_len = len(tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True))
            if input_len + args.output_tokens > args.max_seq_len:
                continue
            record["input_tokens"] = input_len
            pool.append(record)
        random.Random(args.seed).shuffle(pool)
        if len(pool) < args.num_requests + args.warmup:
            raise RuntimeError(f"Only {len(pool)} prompts fit --max-seq-len; "
                               "prepare a larger pool using --pool-size and a new --data path")

        async def send(record, output_tokens):
            return await client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": record["prompt"]}],
                stream=True, temperature=0, max_tokens=output_tokens,
                extra_body={"ignore_eos": True, "top_k": 1})

        for record in pool[args.num_requests:args.num_requests + args.warmup]:
            stream = await send(record, min(16, args.output_tokens))
            async with stream:
                async for _ in stream:
                    pass

        args.output_dir.mkdir(parents=True, exist_ok=False)
        manifest = {"source": SOURCE, "model": model, "num_requests": args.num_requests,
                    "request_rate": args.request_rate, "arrival": "Poisson" if args.request_rate else "burst",
                    "max_concurrency": args.max_concurrency, "seed": args.seed,
                    "output_tokens": args.output_tokens, "max_seq_len": args.max_seq_len,
                    "warmup": args.warmup, "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
                    "measurement": "mini emits one acknowledgement chunk per generated token; "
                                   "finish-only chunks excluded, empty-content acknowledgements included",
                    "cache_state": "Client does not flush server cache; restart server for cold comparisons"}
        (args.output_dir / "config.json").write_text(json.dumps(manifest, indent=2))
        semaphore = asyncio.Semaphore(args.max_concurrency)
        baseline = time.perf_counter()
        rng, arrival = random.Random(args.seed), 0.0
        with (args.output_dir / "requests.jsonl").open("w") as log:
            async def one(index, record, scheduled):
                await asyncio.sleep(max(0, baseline + scheduled - time.perf_counter()))
                async with semaphore:
                    started = time.perf_counter()
                    result = {"request_id": index, "row_idx": record["row_idx"],
                              "prompt_sha256": record["sha256"], "input_tokens": record["input_tokens"],
                              "scheduled_s": scheduled, "sent_s": started - baseline,
                              "client_queue_ms": max(0, (started - baseline - scheduled) * 1000)}
                    ticks, finished, server_id = [], False, None
                    print(f"[request] send id={index} input={record['input_tokens']}", flush=True)
                    try:
                        stream = await send(record, args.output_tokens)
                        async with stream:
                            async for chunk in stream:
                                server_id = chunk.id
                                if not chunk.choices:
                                    continue
                                choice = chunk.choices[0]
                                if choice.finish_reason is not None:
                                    finished = True
                                else:
                                    ticks.append(time.perf_counter())
                        if not finished or len(ticks) != args.output_tokens:
                            raise RuntimeError(f"Expected {args.output_tokens} mini token acknowledgements "
                                               f"and finish marker; got {len(ticks)}, finished={finished}")
                        result.update(status="ok", output_tokens=len(ticks),
                                      ttft_ms=(ticks[0] - started) * 1000,
                                      tpot_ms=((ticks[-1] - ticks[0]) * 1000 / (len(ticks) - 1)
                                               if len(ticks) > 1 else None),
                                      itl_ms=[(b - a) * 1000 for a, b in zip(ticks, ticks[1:])])
                    except Exception as error:
                        result.update(status="failed", error=f"{type(error).__name__}: {error}")
                    ended = time.perf_counter()
                    result.update(server_request_id=server_id, done_s=ended - baseline,
                                  e2e_ms=(ended - started) * 1000,
                                  token_ack_s=[tick - baseline for tick in ticks])
                    log.write(json.dumps(result) + "\n")
                    log.flush()
                    print(f"[request] {result['status']} id={index} server={server_id} "
                          f"e2e={result['e2e_ms']:.1f}ms", flush=True)
                    return result

            tasks = []
            for index, record in enumerate(pool[:args.num_requests]):
                tasks.append(asyncio.create_task(one(index, record, arrival)))
                if args.request_rate:
                    arrival += rng.expovariate(args.request_rate)
            results = await asyncio.gather(*tasks)
        duration = time.perf_counter() - baseline
        good = [r for r in results if r["status"] == "ok"]
        summary = {"successful": len(good), "failed": len(results) - len(good), "duration_s": duration,
                   "request_throughput": len(good) / duration,
                   "output_token_throughput": sum(r["output_tokens"] for r in good) / duration,
                   "ttft_ms": stats([r["ttft_ms"] for r in good]),
                   "tpot_ms": stats([r["tpot_ms"] for r in good if r["tpot_ms"] is not None]),
                   "itl_ms": stats([v for r in good for v in r["itl_ms"]]),
                   "e2e_ms": stats([r["e2e_ms"] for r in good]),
                   "client_queue_ms": stats([r["client_queue_ms"] for r in results])}
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        print(f"Artifacts: {args.output_dir}")
        return 1 if summary["failed"] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--data", type=Path, default=Path("/tmp/mini-wildchat.jsonl"))
    parser.add_argument("--pool-size", type=int, default=2000)
    parser.add_argument("--num-requests", type=int, default=1000)
    parser.add_argument("--request-rate", type=float, default=2, help="Poisson requests/s; 0 means burst")
    parser.add_argument("--max-concurrency", type=int, default=32)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--base-url", default="http://127.0.0.1:1919/v1")
    parser.add_argument("--model")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/tmp/mini-bench") / datetime.now().strftime("%Y%m%d-%H%M%S"))
    args = parser.parse_args()
    if (min(args.pool_size, args.num_requests, args.max_concurrency, args.output_tokens) < 1
            or args.warmup < 0 or args.request_rate < 0 or not math.isfinite(args.request_rate)
            or not math.isfinite(args.timeout) or args.timeout <= 0
            or args.max_seq_len <= args.output_tokens):
        parser.error("Invalid counts, rate, timeout, or sequence length")
    prepare(args.data, args.pool_size)
    return 0 if args.prepare_only else asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
