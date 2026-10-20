# mini-sglang 实验路线：基础 TODO 后转向 SGLang 主仓库

六个 notebook 的题目统一使用三种标签：

- **【必做】**：读透 mini 源码 TODO/NOTE 所在的调用链，完成现状基线、关键反例和资源/输出验证。这是转场门槛；其中有些题要做小范围实现，例如 scheduler 的观测开关。
- **【小改选做】**：在 mini 中做局部、可撤销的修改，加深对机制的理解；不妨碍转场。
- **【大改理论】**：只交设计图、不变量、失败/通过测试计划与尚未验证的边界。完整实现应等到主仓库有明确、仍开放的需求再考虑，不是 mini 的毕业条件。

| Lab | 必做 | 小改选做 | 大改理论 |
|---|---|---|---|
| [Lifecycle](lifecycle/lifecycle.ipynb) | 请求双向链、sampling 参数缺口、Unicode NOTE | batch tokenization | 完整 sampling 参数支持 |
| [Model Support](model_support/model_support.ipynb) | 权重加载链、Gemma 前向图 | InternLM 配置归一化 | 完整 InternLM/Gemma 接入与对拍 |
| [Scheduler](scheduler/scheduler.ipynb) | `observe`、chunk/token budget、公平性轨迹、源码 NOTE | 阶段对照、chunk 上限、A/B 配置 | priority/SRPT/LPM、Mixed Batch、host cache/估计策略 |
| [KV Cache](kv_cache/kv_cache.ipynb) | HiCache/MLA 的 H0/M0 缺口分析、Radix 树、CacheManager 账本、源码 NOTE | 树完整性、MRU 小改 | HiCache 恢复、MLA pool/模型、LFU 完整接入 |
| [Execution](execution/execution.ipynb) | baseline、overlap 时序、ablation、metadata/graph NOTE | profiler 标记 | FA4/backend 扩展 |
| [Bench Profiler](bench_profiler/bench_profiler.ipynb) | CUDA 计时、scheduler profiler | ITL/TPOT 修正、bench 参数收口 | — |

**转场节点**：按建议顺序完成全部【必做】并留存最小复现、轨迹和解释，即结束 mini 基础练习。然后到 [SGLang 主仓库](https://github.com/sgl-project/sglang) 找当前开放、尚无明确认领或活跃 PR 的具体 issue；先在主仓库复现问题、找现有测试与性能基线，再提交聚焦的改动。不要把 mini 中的 TODO 直接当作主仓库仍缺同一功能的证据；例如主仓库已有 HiCache 和 MLA 实现。主仓库的 `TODO` 注释只是调查线索，选题应以现有代码、issue 和维护者讨论为准。

## 先检查运行环境

在 Linux + NVIDIA CUDA 服务器的仓库根目录运行：

```bash
python lab/check_mini.py
python lab/check_mini.py --smoke --model Qwen/Qwen3-0.6B
```

第一条只检查 Python、依赖、GPU 和 CUDA；第二条使用 dummy weight 跑离线 prefill/decode 与 Radix Cache 前缀复用，仍需要模型 config/tokenizer（可以把 `--model` 换成本地模型目录）。想检查真实权重，加 `--real-weights`；想顺带启用 CUDA Graph，加 `--graph`。若 HTTP 服务已启动，再运行 `python lab/check_mini.py --server-url http://127.0.0.1:1919` 检查模型列表和 token streaming。脚本逐项报告 PASS/FAIL/WARN/SKIP；失败时可加 `--verbose` 看堆栈。

这个自检不传 scheduler lab 的 `observe` 参数，因此可以在 Phase 0 尚未实现时先确认 mini 的基础推理路径。它不能替代各 notebook 的专项验收。

`--smoke` 在独立 Python 子进程中初始化 mini，避免环境探测提前初始化 CUDA；只配置 128 页 KV cache，减少基础检查占用的显存。

若出现 `libgomp: Invalid value for environment variable OMP_NUM_THREADS`，脚本会报告 WARN。可先在同一终端运行 `unset OMP_NUM_THREADS`，再重跑自检与 `--smoke`；这只移除当前终端的设置，不修改你的 shell 配置。

【必做】题以“能画调用链、提出反例、用行为和测量验证”为通过标准；要求修改代码的题还须验证改动。下表是能力索引，不表示学习者已经完成，也不把【大改理论】的完整实现列为毕业条件。

| # | 毕业能力 | 当前实验入口 | 尚缺的证据 |
|---|---|---|---|
| 1 | HTTP 到 token streaming 的完整 lifecycle | [Lifecycle](lifecycle/lifecycle.ipynb) | 跨 API server、tokenizer、scheduler、detokenizer 的同一请求轨迹与 SSE 顺序检查 |
| 2 | prefill 与 decode 的 workload 差异 | [Scheduler](scheduler/scheduler.ipynb)、[Execution](execution/execution.ipynb) | 用相同模型实测 phase 的 token 数、shape 与 GPU 时间 |
| 3 | HF model 到 serving model、权重 load/shard | [Model Support](model_support/model_support.ipynb) | 必做权重加载链和 Gemma 前向图；真实模型接入、对拍和 TP shard 属于【大改理论】 |
| 4 | token 到 GPU physical page | [KV Cache](kv_cache/kv_cache.ipynb)、[Execution](execution/execution.ipynb) | 必做 MHA page/表/attention 路径，以及 HiCache/MLA 缺口；host 恢复和 MLA pool 实作属于【大改理论】 |
| 5 | Radix Cache 复用与 evict | [KV Cache](kv_cache/kv_cache.ipynb) | 树形图、锁定、淘汰与 page 账本；host 命中与恢复只作设计分析 |
| 6 | token/KV/prefill/decode 联合调度 | [Scheduler](scheduler/scheduler.ipynb) | 必做准入失败与阶段选择反例；A/B/C 选做，D Mixed Batch 只作理论分析 |
| 7 | Chunked Prefill 与 TTFT/ITL | [Scheduler](scheduler/scheduler.ipynb)、[Execution](execution/execution.ipynb) | 独立 chunk 上限通过，再以在线并发负载测 TTFT/ITL |
| 8 | attention backend 如何消费 metadata | [Execution](execution/execution.ipynb) | 仍需针对 page table、positions、prefill/decode metadata 的定向行为题 |
| 9 | CUDA Graph 主要优化 decode | [Execution](execution/execution.ipynb) | graph ON/OFF trace、padding 解释、稳态 benchmark |
| 10 | Overlap Scheduling 重叠哪两件事 | [Execution](execution/execution.ipynb) | 两轮 batch 时序图、ON/OFF trace 与输出等价性 |
| 11 | TP row/column parallel 的通信 | 待补 | 两卡的局部张量/collective 轨迹和数值对拍 |
| 12 | ON/OFF ablation + profiler/benchmark | [Execution](execution/execution.ipynb)、[Bench Profiler](bench_profiler/bench_profiler.ipynb) | 固定负载、原始样本、trace、性能结论的适用范围 |

建议顺序：Lifecycle → Model Support → Scheduler → KV Cache → Execution → Bench Profiler。每章先完成【必做】，有余力再挑【小改选做】；【大改理论】读需求并写验证方案即可。KV Cache 先做源码 TODO 的 H0/M0，再做 Radix 树和资源账本；Scheduler 的 D 只分析真正同轮混批的约束。第 8、11 项还缺独立的完整动手题，不列入这次 mini 转场门槛。

## 每道题从哪里读代码

各 notebook 的具体题目都有「本题文件导航」，按顺序列出仓库根目录下的真实文件、阅读重点、负载/验收文件和可暂时跳过的模块。InternLM2 的 Phase 1–5 单列文件导航表；Scheduler Phase 0 从 `python/minisgl/env.py` 的全局 `ENV` 开关追到 scheduler/prefill/decode，另有不启动 CUDA 的源码定位格。导航用于找入口，不提供实现补丁或行号。

## mini-sglang 源码 TODO/NOTE 索引

原有 lab 的题目和验收材料保留，执行范围以题目标签为准。以下章节把 mini-sglang 源码留下的 TODO/NOTE 接到相应实验；`NOTE` 需要用轨迹、边界输入或资源账本验证。KV Cache 的两个 TODO 是该 lab 首先调查的任务：完成 H0/M0 现状和反例后转场，不要求在 mini 完整实现 HiCache/MLA。索引指向教程，具体源码位置和练习写在各节内。

| 源码范围 | 原有标记 | 教程入口 |
|---|---|---|
| API、tokenizer、detokenizer | 两处 sampling 参数 TODO、batch tokenization TODO、CJK Unicode Note | [Lifecycle：请求链与输入输出](lifecycle/lifecycle.ipynb) |
| 阶段选择、prefill 准入、请求表 | decode-first TODO、host cache TODO、估计策略 TODO、cached part / token 写入 / dummy request NOTE | [Scheduler：源码留白](scheduler/scheduler.ipynb) |
| Radix tree、缓存接口、KV pool | **优先调查 HiCache TODO 与 MLA TODO 的 H0/M0**；再验证三处树 NOTE、`evict` 文档 Note 和 kernel 测试的重复索引 NOTE | [KV Cache：TODO 主线与缓存边界](kv_cache/kv_cache.ipynb) |
| page table、attention backend、CUDA graph、overlap | 全局表和 backend 切片 NOTE、FlashInfer workspace NOTE、graph/NCCL 与双重释放 NOTE、FA4 TODO | [Execution：执行约束](execution/execution.ipynb) |

KV Cache 的 H0/M0 要交现状和行为证据；Phase 0/3 必做，Phase 1/2 小改选做，Phase 4 的完整 LFU 仅作理论分析。其他 lab 的源码标记按对应标签完成；FA4 等硬件扩展留给主仓库有明确需求时再验收。
