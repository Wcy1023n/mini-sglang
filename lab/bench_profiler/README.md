# Lab: 魔改 benchmark / profiler —— 把 TTFT/TPOT 拆到调度器里

## 目标

**不写新的 benchmark 脚本，但也不只读**。bench_profiler lab 的核心是**魔改**：改现有 benchmark 脚本、改调度器，把端到端 benchmark 报出来的 TTFT / TPOT 这两个黑盒数字，一路拆到调度器内部 prefill / decode 各自的 GPU 时间。每步魔改都逼你走一遍一条真实的调用路径。

三个产出：

1. 魔改 `perf_cuda`（`benchmark/perf.py`），理解 GPU 计时方法（CUDA Event + warmup + CUDA graph 捕获），并写出 `perf_cuda` 的第一个调用方；
2. 魔改 `client.py` 的指标计算，理解 `tics` 列表 → TTFT / TPOT / E2E / throughput 的推导；
3. **魔改 `scheduler._forward` 加 CUDA-event 计时，建一个真正的 profiler**，看到 TTFT 里的 prefill 时间、TPOT 里的 decode 时间——这一步强制走完 `run_forever → _schedule_next_batch → _forward → engine.forward_batch` 整条调用路径。

## 为什么用「魔改」学 benchmark

TTFT / TPOT 这两个数字，光看定义是记不住的。只有当你亲手改出「TTFT 里哪部分是 prefill」的时候，才会真正明白：

- `benchmark/perf.py` 的 `perf_cuda` 用 **CUDA Event** 而非 `time.time()`——`time.time()` 在 CPU 上计时，测的是「提交 kernel 的墙钟」，对短 kernel 会严重高估（launch 开销被算进去）。`perf_cuda` 和 `compare_memory_kernel_perf` 目前全仓库零调用者（`grep -r perf_cuda` 只有定义没有调用），你写的 microbench 就是第一个调用方。
- `benchmark/client.py` 的 `process_benchmark_results` 把一次请求的 `tics` 列表拆成 TTFT（第一个 delta）和 TPOT（后续 delta）——SGLang 线上 benchmark 就是这么干的。
- 但 client 能看到的 TTFT 是个**黑盒**：client 侧的 TTFT 把「prefill + 采出第一个 token」混在了一起。要拆开，只能进调度器加计时——这正是 Phase 3 的魔改。

## 现有资产盘点（先看这张地图）

| 文件 | 角色 | 关键符号 |
|---|---|---|
| `benchmark/perf.py` | **kernel 级计时器** | `perf_cuda`（`:10`）、`compare_memory_kernel_perf`（`:54`） |
| `benchmark/client.py` | **在线 client + 指标计算** | `benchmark_one`（`:202`）、`process_benchmark_results`（`:320`） |
| `benchmark/offline/bench.py` | 离线吞吐（`LLM` 直接 `generate`） | 整个文件 42 行 |
| `benchmark/online/bench_simple.py` | 在线、固定并发 | `TEST_BS`（`:34`）、`MAX_INPUT`（`:36`） |
| `scheduler/scheduler.py` | **调度器主循环 + `_forward`** | `run_forever`（`:120`）、`_schedule_next_batch`（`:219`）、`_forward`（`:227`） |
| `engine/engine.py` | `forward_batch`（真正跑模型 + sample） | `:191` |

## 对标 SGLang 主仓库

每个 Phase 的魔改都不是凭空出题，而是对标 SGLang 主仓库里的一个真实功能。mini-sglang 的 `benchmark/` 是 SGLang 主仓库功能的最小版——先用小魔改走一遍调用链，将来承接 SGLang 主仓库功能时直接平移。

| SGLang 主仓库功能 | 位置 | mini-sglang 的最小版 | 本 lab 魔改 |
|---|---|---|---|
| `bench_serving`：TTFT / ITL / TPOT / E2E / throughput | `python/sglang/bench_serving.py` | `benchmark/client.py` 的 `process_benchmark_results`（有 TTFT/TPOT/E2E/throughput，**缺 ITL 明细**） | Phase 2：补 ITL 明细 + `TPOT=(e2e−ttft)/(n−1)` 公式 |
| `bench_serving --request-rate`（Poisson 到达） | `bench_serving.py` | `benchmark_trace`（固定时间戳重放） | Phase 4 进阶 |
| `bench_offline_throughput` | `python/sglang/bench_offline_throughput.py` | `benchmark/offline/bench.py` | Phase 4 |
| torch profiler（HTTP `/start_profile` `/end_profile`、`--profile`） | `sglang/srt/` + HTTP API | **无**（只有 `perf_cuda` / `compare_memory_kernel_perf` 两个 CUDA-event 微基准） | Phase 3：魔改 `scheduler._forward` 加 CUDA-event 计时 |
| `bench_one_batch`（单 batch / kernel 微基准） | `python/sglang/bench_one_batch.py` | `benchmark/perf.py` 的 `perf_cuda` / `compare_memory_kernel_perf`（零调用者） | Phase 1：写第一个调用方 |

练完之后要承接的关键差异：

- SGLang 的 `bench_serving.py` 比 mini-sglang 的 `client.py` 多一个 **ITL（inter-token latency）** 指标，TPOT 用 `(e2e − ttft) / (输出 token 数 − 1)` 公式，而不是对 `deltas[1:]` 简单求均值。Phase 2 就是把 ITL 明细和 TPOT 公式补上。
- SGLang 用 **torch profiler**（HTTP `/start_profile` `/end_profile` + `--profile`）做端到端 profiling，产出 `*.trace.json` 喂 Perfetto / chrome://tracing。mini-sglang 完全没有这层——Phase 3 用 CUDA-event 做一个最小版（拆 prefill/decode 各自耗时），为理解 torch profiler 的 per-step 拆解打底。

---

## 验收标准

| # | 检查 | 现象 |
|---|---|---|
| 1 | 魔改 `perf_cuda` 成功 | `microbench.py` 输出合理 GB/s；能解释 `warmups=0` vs `warmups=5` 的冷启动差 |
| 2 | 魔改 `client.py` 指标成功 | 新增的 `TTFT / TPOT` 比值随输入长度变化 |
| 3 | 魔改 `_forward` 的 profiler 跑通 | 输出 prefill / decode 各自的 steps 和 avg ms |
| 4 | 复用 offline/online bench 收口 | 得到真实 TTFT/TPOT/throughput，并用 Phase 3 的 profiler 解释 |

---

## Phase 1 — 魔改 `perf_cuda`：GPU 计时

先读 `benchmark/perf.py:10-51` 的 `perf_cuda`，搞清楚三步：warmup（`f()` 一次）→ 捕获 CUDA graph（`cuda_graph_repetitions` 次）→ 用 CUDA Event 计时 `repetitions` 次 replay。

**魔改 1**：给 `perf_cuda` 加一个 `warmups: int = 1` 参数，把原来写死的一次 warmup `f()` 换成 `for _ in range(warmups): f()`（默认 1，行为不变）。然后写 `lab/bench_profiler/microbench.py`，用 `perf_cuda` 测一个短 kernel，对比 `warmups=0 / 1 / 5`：

```python
from __future__ import annotations
import torch
from minisgl.benchmark.perf import perf_cuda

def main() -> None:
    torch.cuda.set_device(0)
    N = 1 << 22
    x = torch.randn(N, device="cuda")
    y = torch.randn(N, device="cuda")
    for w in (0, 1, 5):
        ms = perf_cuda(lambda: y.copy_(x), warmups=w)
        gbs = (2 * N * 4) / (ms * 1e6)
        print(f"warmups={w}: {ms:.3f} ms/call -> {gbs:.1f} GB/s")
```

**checkpoint 1**：`warmups=0` 明显慢于 `warmups=1`（第一次调用触发 kernel JIT + 缓存冷启动），`warmups=1` 和 `warmups=5` 基本持平。回答三问（答案见提示 A）：

1. 为什么用 `torch.cuda.Event` 而非 `time.time()`？
2. 为什么 `tic.record()` 之前要 `torch.cuda.synchronize()`？
3. `cuda_graph_repetitions` 为什么能把「N 次调用」捕获成一个 graph 再 replay？

> 进阶：把 `f` 换成 `torch.mm`（计算密集），用同样的 `perf_cuda` 测，算出来的 GB/s 会「突破纸面峰值」——`torch.mm` 不是 memory-bound，用 GB/s 衡量没意义。再顺手用 `compare_memory_kernel_perf`（`perf.py:54`）对比 `torch.add(x, y)` vs `torch.add(x, y, out=y)`，理解 `memory_footprint / (dur * 1e6)` 的单位是 bytes ÷ (ms × 10⁶) = GB/s。

## Phase 2 — 魔改 `client.py`：把「TPOT」拆回 ITL / TPOT（对标 `bench_serving`）

读 `benchmark_one`（`client.py:202-248`）和 `process_benchmark_results`（`client.py:320-404`）。`tics` 的累积方式是 `tics = [time.perf_counter()]` 然后每收一个 stream chunk `append` 一次，所以 `deltas[0] = TTFT`、`deltas[1:]` = 每个 token 之间的间隔。

这里有个**命名陷阱**，也是 SGLang `bench_serving.py` 和 mini-sglang `client.py` 的关键差异：mini-sglang 把 `deltas[1:]` 直接标成了 **TPOT**，但 SGLang 里 `deltas[1:]` 叫 **ITL（inter-token latency，逐 token 间隔）**，而 **TPOT** 是一个派生量：`TPOT = (e2e − ttft) / (输出 token 数 − 1)`。

**魔改 2**：把 `process_benchmark_results` 的「TPOT」改正确：

1. 把 `accum_times`（`client.py:333` 的 `deltas[1:]`）改标签为 **ITL**，并保留 p95/p99/max 明细（SGLang 也是这么报 ITL 的）；
2. 新增逐请求的 **TPOT**：`(e2e − ttft) / (输出 token 数 − 1)`，再聚合 avg/p50/p90/p99。

**checkpoint 2**：能解释（答案见提示 B）：

1. `first_times`（`client.py:332`）= `deltas[0]`，是什么？（TTFT）
2. `accum_times`（`client.py:333`）= `deltas[1:]`，是什么？（ITL，逐 token 间隔）
3. 为什么 `times[int(len(times)*0.9)]` 之前要先 `sort()`？（p90/p99 要先排序）
4. 新加的 TPOT `(e2e−ttft)/(n−1)` 和 ITL 的均值是什么关系？（`e2e−ttft` = `deltas[1:]` 之和，除以 `n−1` 就是 ITL 的均值——所以 SGLang 的 TPOT ≈ 平均 ITL，但二者分开报：ITL 报分布，TPOT 报单值）
5. **关键**：TTFT 里到底混了什么，client 能拆开吗？

第 5 问的答案是「不能」——client 看到的 TTFT 是「prefill + 采第一个 token」的总和。要拆开，只有进调度器加计时。这就是 Phase 3。

## Phase 3 — 魔改 `scheduler._forward`：建 profiler（核心）

目标：在调度器的 forward 路径上插 CUDA Event，把每一步的 GPU 时间按 `prefill` / `decode` 分开累计，最后打出一份 profiler 报告。

**先走一遍调用路径**（改之前必须先搞清楚在改哪一环）：

```
Scheduler.run_forever            (scheduler.py:120)
  └─ overlap_loop / normal_loop  (scheduler.py:83 / :108)
       └─ _schedule_next_batch   (scheduler.py:219)  ← 选 prefill 还是 decode
            └─ _prepare_batch    (scheduler.py:204)
       └─ _forward               (scheduler.py:227)  ← 本 Phase 的魔改点
            └─ engine.forward_batch (engine.py:191)  ← 真正跑模型 + sample
```

**魔改 3**：三步，全在 `scheduler/scheduler.py`。

① `__init__` 里加一个累计容器：

```python
self._profile: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {"prefill": [], "decode": []}
```

② `_forward` 里，用两个 CUDA Event 包住 `engine.forward_batch`：

```python
def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
    batch, sample_args, input_mapping, output_mapping = forward_input
    batch.input_ids = self.token_pool[input_mapping]
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    forward_output = self.engine.forward_batch(batch, sample_args)
    end.record()
    self._profile[batch.phase].append((start, end))
    self.token_pool[output_mapping] = forward_output.next_tokens_gpu
    self.decode_manager.filter_reqs(forward_input.batch.reqs)
    return forward_output
```

> `start.record()` 不带 stream 参数，用的就是当前 stream。`_forward` 是在 `overlap_loop` 的 `with self.engine_stream_ctx:` 里被调用的（`scheduler.py:101`），当前 stream = engine stream，event 落在 GPU 计算的那条 stream 上，计的才是真 GPU 时间。

③ 加一个报告方法，把所有 step 汇总：

```python
def report_profile(self) -> None:
    torch.cuda.synchronize()
    for phase, events in self._profile.items():
        if not events:
            continue
        total = sum(s.elapsed_time(e) for s, e in events)
        print(f"[profile] {phase:7s} steps={len(events):4d} total={total:8.2f}ms avg={total / len(events):6.3f}ms")
```

然后写 `lab/bench_profiler/profile.py` 驱动（`LLM` 就是 `Scheduler` 的子类，`report_profile` 直接可用）：

```python
from minisgl.core import SamplingParams
from minisgl.llm import LLM

def main():
    llm = LLM("Qwen/Qwen3-0.6B", use_dummy_weight=True, max_extend_tokens=32)
    prompts = [[1000 + i % 5000 for i in range(200)]] + [[i + 2000 for i in range(4)]] * 6
    llm.generate(prompts, SamplingParams(ignore_eos=True, max_tokens=30))
    llm.report_profile()
```

跑 `python lab/bench_profiler/profile.py`。

**checkpoint 3**：输出类似：

```
[profile] prefill steps=  13 total=  xx.xx ms avg=  x.xxx ms
[profile] decode  steps= 203 total=  xx.xx ms avg=  x.xxx ms
```

回答（答案见提示 C）：

1. `steps=13` 和 `steps=203` 分别从哪来？（13 = 1 条长 prompt 拆 7 chunk + 6 条短 prompt 各 1 步；203 = 7 个请求各 `max_tokens − 1 = 29` 步 = 7×29）
2. 为什么 `decode` 的 `avg` 比 `prefill` 小很多？（decode 每步只有 1 个 token，prefill 每步几十个 token）
3. profiler 和 Phase 2 的 TTFT/TPOT 怎么对上？（TTFT ≈ 一次 prefill 的耗时，TPOT ≈ 一次 decode 的耗时——现在能在调度器里分别看到）

## Phase 4 — 复用 offline/online bench 收口

有了 Phase 3 的 profiler，再回头看端到端数字就不一样了。

**offline**：读 `benchmark/offline/bench.py`（42 行），把模型路径魔改成 InternLM2 本地路径（本机只有 Qwen3-0.6B 的 config/tokenizer 缓存，权重没下），跑 `python benchmark/offline/bench.py` 得到 tok/s。小魔改 `num_seqs` / `max_output_len` 看吞吐怎么变。

**online**：起 `python -m minisgl --model /root/models/internlm2_5-1_8b-chat --max-seq-len-override 4096`，另开终端跑 `python benchmark/online/bench_simple.py`，得到带 p50/p90/p99 的 TTFT/TPOT/E2E/throughput。小魔改 `TEST_BS=[1,8,64]`，观察并发对三个指标的影响。

**checkpoint 4**：现在能用 Phase 3 的 profiler 语言解释 TTFT/TPOT/throughput 三个数字——TTFT 大 = prefill 慢（输入长或排队），TPOT 大 = decode 慢（并发高、GPU 竞争），吞吐 = 总 token / 墙钟时间。

## 建议的推进顺序

Phase 1 → 2 → 3 → 4，顺序不能乱：Phase 3 的 profiler 建立在 Phase 1 的 CUDA Event 方法上，Phase 4 的解释建立在 Phase 3 的 profiler 上。

## 环境注意事项

- 需要 GPU（`perf.py` 和所有 benchmark 都在 CUDA 上）。
- Phase 3 的 `profile.py` 用 `use_dummy_weight=True`，**不需要真实权重**，且能看到 prefill/decode 的区分。
- Phase 4 的 offline/online 要真实权重，依赖 model lab 已完成（InternLM2 在本地）。
- 后端走 `fi`，别动 `--attention-backend`。
- Phase 1、3 魔改的是共享源码 `perf.py` / `scheduler.py`，做完记得还原或用环境变量包一层。

## 卡住了再看

<details>
<summary>提示 A：perf_cuda 三问</summary>

1. `time.time()` 在 CPU 上计时，测的是提交 wall-clock；短 kernel 的 launch 开销（CPU 提交 GPU）会被算进去。CUDA Event 记录在 GPU stream 上，测的是纯 GPU 执行时间。
2. `synchronize()` 清空 GPU 上排队的 kernel，否则前面的 kernel 耗时会被算进第一个 `tic.record()`。
3. 把 N 次调用捕获进 CUDA graph 后 `replay` 一次提交，launch 开销被摊薄到 N 次里。

</details>

<details>
<summary>提示 B：client 指标推导（ITL vs TPOT）</summary>

`tics = [t0, t1, ..., tK]`，`t0` = 发送，`t1` = 第一个 token chunk，`t2..tK` = 后续 token。`deltas = [t1-t0, t2-t1, ...]`；TTFT = `deltas[0]`（`first_times`）；ITL = `deltas[1:]`（`accum_times`）；E2E = `tics[-1] - tics[0]`；TPOT = `(e2e − ttft) / (输出 token 数 − 1)`。`e2e − ttft` = `deltas[1:]` 之和，所以 TPOT ≈ 平均 ITL。长输入 → prefill 重 → TTFT/ITL 比值大；短输入 → 比值小。TTFT 里混了 prefill + 首 token，client 拆不开。

</details>

<details>
<summary>提示 C：profiler 数字来源</summary>

`profile.py` 的 workload：1 条 200-token 长 prompt（`max_extend_tokens=32` → 拆 7 chunk）+ 6 条 4-token 短 prompt。prefill steps = 7 + 6 = 13；decode steps = 7 请求 × (`max_tokens − 1` = 29) = 203（第 1 个 token 由 prefill 那步产出）。decode avg 小是因为每步只算 1 个 token。

</details>
