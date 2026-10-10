# Scheduler Lab 原实验核对

以用户提供的原版 `sche.md` 为对照。当前教程是 [scheduler.ipynb](scheduler.ipynb)；它保留原 Phase 0–6 的学习目标，同时新增 A/B/C。原版中给出最终补丁或预设答案的内容没有复制进题面，验收改为可观察行为与需要自行推导的问题。

## Phase 0 从哪里开始

`LLM(...)` 的类定义在 [python/minisgl/llm/llm.py](../../python/minisgl/llm/llm.py)，先搜索 `class LLM` 并阅读 `__init__`；`from minisgl.llm import LLM` 通过包的 [__init__.py](../../python/minisgl/llm/__init__.py) 导出它。`observe` 是本题要新增的接口。验收脚本创建 `TraceLLM`，它继承 `LLM`，因此只搜索 `LLM(` 可能找不到实际构造入口。

notebook 的 **「Phase 0 源码导航：先找到 LLM」** 提供逐文件入口、构造/执行调用路线、阅读问题与可跳过的模块，附带一格只读源码的定位代码，不启动 CUDA。先完成导航，再运行原有的开关对照验收格。

## 原实验对应关系

| 原实验 | 当前形式 | notebook 验收 |
|---|---|---|
| Phase 0：搭观测 | 给 mini 增加 `observe=0/1` 日志开关，对照外部轨迹 | notebook 比较开关两种输出；`long-first --observe 0 --check` |
| Phase 1：decode-first | 可撤销的动手对照；使用能区分两种顺序的负载 | `stage-order --expect-stage-order ...` |
| Phase 2：chunked prefill | 两种总 budget 的动手观察 | `long-first --expect-chunks ...` |
| Phase 3：round-robin | 理论题：两阶段都可运行时的交替及空缺处理 | `mixed --check` 验证饥饿反例，再交推导轨迹 |
| Phase 4：priority | 理论题：waiting 准入与 running 抢占 | `priority-theory --check` 验证 FCFS 基线，再交优先级预期 |
| Phase 5：SRPT | 理论题：原始长度、剩余 prefill 与完整请求工作量 | `shortest-theory --check` 验证长请求先行，再交反例 |
| Phase 6：LPM | 理论题：缓存复用与排序效果分离 | `lpm-theory --check` 验证 FCFS 基线，再交 LPM 预期 |

`observe.py` 的 `[sched]` 是实验脚本提供的外部轨迹；Phase 0 的 `[mini-observe]` 是学生要在 mini 内实现的可开关日志。除 Phase 0 的 `observe=1` 对照，所有实验验收都传 `--observe 0`，脚本默认值也为 0。理论题的代码块验收的是**提出问题所依赖的真实基线**，不代表策略已经实现。A `prefill_decode_interval`、B `prefill_max_requests`、C `chunked_prefill_size` 各有另外的行为验收。原版 Phase 1 使用 200-token 长 prompt 时，最后一块可能与短请求合批，不能稳定显示 decode-first 造成的等待；当前专用负载用 192 token 消除了这个歧义。
