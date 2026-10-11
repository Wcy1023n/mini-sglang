# Scheduler Lab 原实验核对

以用户提供的原版 `sche.md` 为对照。当前教程是 [scheduler.ipynb](scheduler.ipynb)；它保留原 Phase 0–6 的学习目标，同时新增 A/B/C。原版中给出最终补丁或预设答案的内容没有复制进题面，验收改为可观察行为与需要自行推导的问题。

## Phase 0 从哪里开始

从 [SchedulerConfig](../../python/minisgl/scheduler/config.py) 开始：验收脚本通过现有离线调用传入 `observe=0/1`，额外关键字参数已经转交配置。然后看 [Scheduler](../../python/minisgl/scheduler/scheduler.py) 怎样创建 [PrefillManager](../../python/minisgl/scheduler/prefill.py) 和 [DecodeManager](../../python/minisgl/scheduler/decode.py)。三个位置共用开关，但两个 manager 是独立对象；本题不要求在 `LLM` 类中保存开关或修改其方法。[ServerArgs](../../python/minisgl/server/args.py) 继承 `SchedulerConfig`，在线调度器也使用这套配置定义；本题只验收离线调用，在线命令行参数解析暂不验收。

notebook 的 **「Phase 0 源码导航：从配置到三个观测位置」** 提供逐文件入口、开关传递问题与只读源码定位格，不启动 CUDA。完成导航后运行开关对照验收格。

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
