# Scheduler Lab 原实验核对

以用户提供的原版 `sche.md` 为对照。当前教程是 [scheduler.ipynb](scheduler.ipynb)；它保留原 Phase 0–6 的学习目标，同时新增 A/B/C。原版中给出最终补丁或预设答案的内容没有复制进题面，验收改为可观察行为与需要自行推导的问题。

## Phase 0 从哪里开始

从 [python/minisgl/env.py](../../python/minisgl/env.py) 开始：mini 已有各模块可导入的模块级 `ENV` 单例，scheduler 已用它控制 overlap。Phase 0 沿用这套机制，新增由 `MINISGL_OBSERVE=0/1` 控制、默认关闭的进程级全局观测开关。`SchedulerConfig` 是实例配置类，导入它本身不会得到同一份配置对象。

接着读三个日志位置：[python/minisgl/scheduler/scheduler.py](../../python/minisgl/scheduler/scheduler.py) 的选批次、[python/minisgl/scheduler/prefill.py](../../python/minisgl/scheduler/prefill.py) 的准入、[python/minisgl/scheduler/decode.py](../../python/minisgl/scheduler/decode.py) 的 running 集合。三个模块读取同一个进程内的全局配置；开关无须经 `LLM` 或 manager 的构造参数传递。

验收脚本的 `--observe 0/1` 在导入 mini **之前**设置 `MINISGL_OBSERVE`，不再把 `observe` 传给 `LLM`。在线服务也可从启动环境读取该值；各进程有自己的 `ENV`，本题不要求跨进程动态改值。notebook 的 **「Phase 0 源码导航：全局开关放哪里，三个模块去哪里读」** 提供这些文件的阅读重点与只读定位格。

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

`observe.py` 的 `[sched]` 是实验脚本提供的外部轨迹；Phase 0 要在 mini 内实现 `[mini-observe]`（选中 batch）、`[mini-prefill]`（准入）、`[mini-decode]`（running）三处可开关日志。除 Phase 0 的 `observe=1` 对照，所有实验验收都传 `--observe 0`，脚本默认值也为 0。理论题的代码块验收的是**提出问题所依赖的真实基线**，不代表策略已经实现。A `prefill_decode_interval`、B `prefill_max_requests`、C `chunked_prefill_size` 各有另外的行为验收。原版 Phase 1 使用 200-token 长 prompt 时，最后一块可能与短请求合批，不能稳定显示 decode-first 造成的等待；当前专用负载用 192 token 消除了这个歧义。
