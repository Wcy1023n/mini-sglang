# SGLang 系统实验毕业表

## 先检查运行环境

在 Linux + NVIDIA CUDA 服务器的仓库根目录运行：

```bash
python lab/check_mini.py
python lab/check_mini.py --smoke --model Qwen/Qwen3-0.6B
```

第一条只检查 Python、依赖、GPU 和 CUDA；第二条使用 dummy weight 跑离线 prefill/decode 与 Radix Cache 前缀复用，仍需要模型 config/tokenizer（可以把 `--model` 换成本地模型目录）。想检查真实权重，加 `--real-weights`；想顺带启用 CUDA Graph，加 `--graph`。若 HTTP 服务已启动，再运行 `python lab/check_mini.py --server-url http://127.0.0.1:1919` 检查模型列表和 token streaming。脚本逐项报告 PASS/FAIL/WARN/SKIP；失败时可加 `--verbose` 看堆栈。

这个自检不传 scheduler lab 的 `observe` 参数，因此可以在 Phase 0 尚未实现时先确认 mini 的基础推理路径。它不能替代各 notebook 的专项验收。

若出现 `libgomp: Invalid value for environment variable OMP_NUM_THREADS`，脚本会报告 WARN。可先在同一终端运行 `unset OMP_NUM_THREADS`，再重跑自检与 `--smoke`；这只移除当前终端的设置，不修改你的 shell 配置。

每项都以“能画调用链、提出反例、修改一处真实机制、用行为和测量验证”为通过标准。`已有实验`表示有可运行的题目，**不表示你已经完成**。各 notebook 中未实现的功能由学习者完成。

| # | 毕业能力 | 当前实验入口 | 尚缺的证据 |
|---|---|---|---|
| 1 | HTTP 到 token streaming 的完整 lifecycle | 待补 | 跨 API server、tokenizer、scheduler、detokenizer 的同一请求轨迹与 SSE 顺序检查 |
| 2 | prefill 与 decode 的 workload 差异 | [Scheduler](scheduler/scheduler.ipynb)、[Execution](execution/execution.ipynb) | 用相同模型实测 phase 的 token 数、shape 与 GPU 时间 |
| 3 | HF model 到 serving model、权重 load/shard | [Model Support](model_support/model_support.ipynb) | Gemma 3 文本短上下文对拍后，再做 TP>1 的 shard 验收 |
| 4 | token 到 GPU physical page | [KV Cache](kv_cache/kv_cache.ipynb)、[Execution](execution/execution.ipynb) | 从 request page table 追到 attention backend 实际读取的位置 |
| 5 | Radix Cache 复用与 evict | [KV Cache](kv_cache/kv_cache.ipynb) | 共享前缀树形图、完整性失败/通过、锁定与淘汰轨迹 |
| 6 | token/KV/prefill/decode 联合调度 | [Scheduler](scheduler/scheduler.ipynb) | 准入失败与阶段选择反例，两项新增功能的行为验收 |
| 7 | Chunked Prefill 与 TTFT/ITL | [Scheduler](scheduler/scheduler.ipynb)、[Execution](execution/execution.ipynb) | 独立 chunk 上限通过，再以在线并发负载测 TTFT/ITL |
| 8 | attention backend 如何消费 metadata | [Execution](execution/execution.ipynb) | 仍需针对 page table、positions、prefill/decode metadata 的定向行为题 |
| 9 | CUDA Graph 主要优化 decode | [Execution](execution/execution.ipynb) | graph ON/OFF trace、padding 解释、稳态 benchmark |
| 10 | Overlap Scheduling 重叠哪两件事 | [Execution](execution/execution.ipynb) | 两轮 batch 时序图、ON/OFF trace 与输出等价性 |
| 11 | TP row/column parallel 的通信 | 待补 | 两卡的局部张量/collective 轨迹和数值对拍 |
| 12 | ON/OFF ablation + profiler/benchmark | [Execution](execution/execution.ipynb)、[Bench Profiler](bench_profiler/bench_profiler.ipynb) | 固定负载、原始样本、trace、性能结论的适用范围 |

建议顺序：Model Support → Scheduler → KV Cache → Execution → Bench Profiler。做完已有实验后，再补第 1、8、11 项的动手题；这三项目前不能据此判定毕业。
