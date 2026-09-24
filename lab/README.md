# Lab: 给 Mini-SGLang 接入 `internlm/internlm2_5-1_8b-chat`

## 目标

让下面这条命令跑起来，且**贪心解码结果与 HF transformers 逐 token 一致**：

```bash
python -m minisgl --model /root/models/internlm2_5-1_8b-chat --shell
```

模型已经下载到 `/root/models/internlm2_5-1_8b-chat`（3.6G，含 `modeling_internlm2.py`，所以 HF 侧可以直接跑）。

## 为什么挑这个模型

它的**骨架是标准的 Llama** —— RMSNorm、SwiGLU、GQA、RoPE，无 bias，`head_dim = 2048/16 = 128`（合法）。

这意味着 `models/internlm2.py` 本身几乎是抄 `models/llama.py`，**不值得花时间**。真正的时间应该花在下面这三件事上，而这正是模型接入最容易翻车的地方：

1. **checkpoint 的键名体系 ≠ 运行时的模块树**
2. **checkpoint 里 qkv 的物理排布 ≠ 运行时 `qkv_proj` 期望的排布**
3. **HF config 的字段命名/语义 ≠ 归一化后的 `ModelConfig`**

## 验收标准

按顺序，三个都得过：

| # | 检查 | 命令 |
|---|---|---|
| 1 | 注册 / config / 网络结构正确 | `python -m minisgl --model /root/models/internlm2_5-1_8b-chat --dummy-weight --max-seq-len-override 4096` 能启动并发请求 |
| 2 | 真实权重全部被消费 | 去掉 `--dummy-weight`，启动时**不报** `Unexpected keys in state_dict`（`layers/base.py:52`） |
| 3 | 数值正确 | `python lab/compare.py` → `PASS -- 16/16 greedy tokens match` |

`lab/hf_ref.py` 已经跑过并产出了 `lab/hf_ref.json`（HF 的 ground truth）：

```
INPUT_IDS : [1, 1, 92543, 1008, 364, ...]          # 注意有两个 BOS，是 chat template + add_bos_token 叠加的结果
STEP0_TOP5: [(60862, 16.875), (60361, 12.625), ...]
GREEDY    : '张量并行是一种计算模型，它将输入张量分解成多个'
```

`lab/compare.py` 用**相同的 input_ids** 直接喂给 mini-sglang（绕开输入侧分词），只比对模型本身。

---

## 五个接入点

### ① `python/minisgl/utils/hf.py` — 基础设施

**症状**：连 config 都读不出来。

```
ValueError: The repository /root/models/internlm2_5-1_8b-chat contains custom code
which must be executed to correctly load the model.
```

**原因**：`model_type: internlm2` 不在 transformers 4.57 的内置注册表里，config 和 tokenizer 都靠仓库内 `config.json` 的 `auto_map` 指向 `configuration_internlm2.py` / `tokenization_internlm2*.py`。而 `hf.py:32` 的 `AutoConfig.from_pretrained(model_path)` 和 `hf.py:18` 的 `AutoTokenizer.from_pretrained(model_path)` 都没开 `trust_remote_code`。

**额外坑**：tokenizer 还必须 `use_fast=False`。

原因是 `sentencepiece` 与 `protobuf` 的双向版本冲突：

- 本机 `sentencepiece` 被迫锁在 **0.1.99**（唯一能解析这个 vocab 的版本，见"环境注意事项"）
- 但 0.1.99 自带的 `sentencepiece_model_pb2.py` 是旧版 `protoc` 生成的，与本机的 **protobuf 6.31** 不兼容
- `use_fast=True`（默认）会走慢→快转换路径，该路径需要 `from sentencepiece import sentencepiece_model_pb2`，于是撞上：

```
TypeError: Descriptors cannot be created directly.
```

`use_fast=False` 直接绕开整个转换路径，走纯 Python 的 sentencepiece 实现。

> 另一条路是 `use_fast=True` + 环境变量 `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`（已验证可行），但那会让 protobuf 退化成纯 Python 解析且污染全局环境，**不推荐**。

> ⚠️ 注意这是**共享代码**，改动会影响所有模型。想清楚 `trust_remote_code=True` 的代价再动手（它会执行仓库里的任意 Python）。

### ② `python/minisgl/models/config.py` — config 归一化

**症状**：模型能建起来，但 `get_rope` 抛 `KeyError`。

**原因**：InternLM2 的 config 里是

```json
"rope_scaling": {"type": "dynamic", "factor": 2.0}
```

注意 key 是 **`type`** 而不是 `rope_type`，而 `layers/rotary.py:65` 硬编码读 `rope_scaling["rope_type"]`。

**要想清楚的第二层**：就算把 key 改名，`_get_rope`（`rotary.py:65`）也只实现了 `default` / `llama3` / `yarn`，没有 `dynamic`。

这里**不要急着去实现 dynamic NTK**，先去读 `modeling_internlm2.py:144` 的 `InternLM2DynamicNTKScalingRotaryEmbedding.forward`：

```python
seq_len = torch.max(position_ids) + 1
if seq_len > self.max_position_embeddings:   # <- 32768
    base = ...重新计算...
```

也就是说，**只要序列长度不超过 32768，dynamic 缩放完全不生效，等价于普通 RoPE**。`config.py` 里做归一化的正确姿势是什么？

### ③ `python/minisgl/models/register.py` — 注册

一行的事。`architectures[0]` 是 `InternLM2ForCausalLM`。

### ④ `python/minisgl/models/internlm2.py` — 模型定义

抄 `models/llama.py`，改类名即可 —— 结构完全一致。

**但有个关键约束**：`layers/base.py:32` 的 `load_state_dict` 是**靠属性名反射遍历 `__dict__`** 来匹配权重 key 的。所以 ④ 和 ⑤ 是**耦合**的：你在 ⑤ 里把 checkpoint key 重映射成什么名字，这里就必须是什么属性名。

建议先定 ⑤ 的重命名方案，再写 ④。这样 ④ 基本就是无脑对齐。

### ⑤ `python/minisgl/models/weight.py` — 权重映射（**重点**）

checkpoint 里只有 10 种 key：

```
model.tok_embeddings.weight              model.layers.N.attention.wqkv.weight
model.norm.weight                        model.layers.N.attention.wo.weight
output.weight                            model.layers.N.attention_norm.weight
                                         model.layers.N.ffn_norm.weight
                                         model.layers.N.feed_forward.w1.weight
                                         model.layers.N.feed_forward.w2.weight
                                         model.layers.N.feed_forward.w3.weight
```

而 `weight.py` 里那套 TP 切分 / 融合机制，**全部依赖后缀匹配**：

- `_SPLIT_DIM_0`（`weight.py:13`）认 `.q_proj`/`.k_proj`/`.v_proj`/`.gate_proj`/`.up_proj`
- `_SPLIT_DIM_1`（`weight.py:14`）认 `.o_proj`/`.down_proj`
- `_MERGE_GROUPS`（`weight.py:17`）把 `.q/k/v_proj` 融成 `.qkv_proj`、`.gate/up_proj` 融成 `.gate_up_proj`

在这里**一个都匹配不上**。

**子任务 A — 键名重映射**：加一层 rename，把 checkpoint 的 key 翻译成运行时的 key。提示：`w1`/`w3` 其实对应 gate/up，翻译成 `.gate_proj`/`.up_proj` 就能**直接复用现成的 `_MERGE_GROUPS`**。

**子任务 B — `wqkv` 的排布陷阱** ⚠️

这是本次实验**唯一一个不查源码就一定会踩**的坑。

`wqkv` 是已经融合好的权重，形状 `[(16 + 2*8) * 128, 2048]`。你可能会想当然地认为它是 `[q0..q15, k0..k7, v0..v7]` 顺序拼的 —— **不是**。

它的真实排布见 `modeling_internlm2.py:314`：

```python
qkv_states = rearrange(qkv_states, "b q (h gs d) -> b q h gs d",
                       gs=2 + self.num_key_value_groups, d=self.head_dim)
query = qkv_states[..., :gs, :]
key   = qkv_states[..., -2, :]
value = qkv_states[..., -1, :]
```

而 mini-sglang 的 `AttentionLayer.forward`（`layers/attention.py:49`）期望的是：

```python
q, k, v = qkv.split([self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1)
```

即整块的 `[所有 q | 所有 k | 所有 v]`。

> 如果你只做 A 不做 B，模型能启动、权重能 load、**一个报错都没有**，但对拍会在**第 0 个 token 就错**，输出是一堆乱码。这是最典型的"静默错误"。

---

## 建议的推进顺序

每个 Phase 都有明确的 checkpoint，不要跳。

```
Phase 1  改 hf.py（接入点①）
         checkpoint: python -c "from minisgl.utils import cached_load_hf_config;
                                 print(cached_load_hf_config('/root/models/internlm2_5-1_8b-chat'))"
                     能打印出 config

Phase 2  改 config.py（接入点②）
         checkpoint: ModelConfig.from_hf(...) 能构造成功，且 rotary_config.scaling 是合理值

Phase 3  改 register.py + 新建 models/internlm2.py（接入点③④）
         checkpoint: python -m minisgl --model /root/models/internlm2_5-1_8b-chat \
                         --dummy-weight --max-seq-len-override 4096
                     能起来、能返回 200（输出乱码是对的，因为权重是随机的）

Phase 4  改 weight.py 的子任务 A（键名重映射）
         checkpoint: 去掉 --dummy-weight 能启动，不报 Unexpected keys

Phase 5  改 weight.py 的子任务 B（wqkv 反交错）
         checkpoint: python lab/compare.py  →  PASS 16/16
```

**Phase 3 的 `--dummy-weight` 是个很有用的中间态**：它走 `engine.py:140` 的分支，完全不调用 `load_weight`，所以能把"注册/config/结构"的问题和"权重映射"的问题彻底隔离开。别跳过它。

## 环境注意事项

- 后端已经确认走 **`fi`（flashinfer）**：sm_86 上 `engine.py:223` 的 auto 逻辑不会选 `fa`（本机 `sgl_kernel` 有 ABI 不兼容，是坏的），所以别去动 `--attention-backend`。
- `sentencepiece` 已固定到 **0.1.99**。0.2.x 无法解析 InternLM2 的 `tokenizer.model`（`piece must not include null character`），文件本身没问题（sha256 与 HF 一致）。**不要升级它**——升级后 slow tokenizer 直接坏掉。
- 由此产生的连锁反应：0.1.99 的 `sentencepiece_model_pb2.py` 与本机 `protobuf 6.31` 不兼容，所以 fast tokenizer 也走不通。详见接入点 ①。
- 底座已验证可用：`--dummy-weight` + Qwen3-0.6B 在这台 3090 上能正常起服务并返回 200。

## 卡住了再看

<details>
<summary>提示 A：键名重映射表</summary>

| checkpoint | 运行时 |
|---|---|
| `model.tok_embeddings` | `model.embed_tokens` |
| `model.layers.N.attention.wqkv` | `model.layers.N.self_attn.qkv_proj` |
| `model.layers.N.attention.wo` | `model.layers.N.self_attn.o_proj` |
| `model.layers.N.attention_norm` | `model.layers.N.input_layernorm` |
| `model.layers.N.ffn_norm` | `model.layers.N.post_attention_layernorm` |
| `model.layers.N.feed_forward.w1` | `model.layers.N.mlp.gate_proj` |
| `model.layers.N.feed_forward.w3` | `model.layers.N.mlp.up_proj` |
| `model.layers.N.feed_forward.w2` | `model.layers.N.mlp.down_proj` |
| `output` | `lm_head` |
| `model.norm` | `model.norm`（不变） |

注意 `w1`/`w3` → `gate_proj`/`up_proj` 是刻意的：这样 `_MERGE_GROUPS` 会自动把它们拼成 `gate_up_proj`，不用另外写逻辑。

</details>

<details>
<summary>提示 B：wqkv 的 reshape 思路</summary>

`num_kv_heads = 8`，`num_key_value_groups = num_qo_heads / num_kv_heads = 2`，`head_dim = 128`。

把 `wqkv`（`[4096, 2048]`）reshape 成 `[num_kv_heads, gs+2, head_dim, hidden]` = `[8, 4, 128, 2048]`，你就得到了 8 个组，每组是 `(q, q, k, v)`：

```
[:, 0:2] -> 这一组对应的 q heads
[:, 2]   -> 这一组对应的 k head
[:, 3]   -> 这一组对应的 v head
```

剩下就是 `reshape` 拍平 + `torch.cat` 成 `[q_all, k_all, v_all]`。

顺手可以做个小验证：这个变换应该是**可逆的双射**，且 q 部分共有 `16*128 = 2048` 行、k/v 各 `8*128 = 1024` 行。

</details>

<details>
<summary>提示 C：这个变换为什么必须做（不想做的话）</summary>

其实还有一条路：不重排权重，而是**在 `RopeAttn` 里按 InternLM2 的布局切 q/k/v**。但那样 `AttentionLayer` 就要为这个模型特化，会污染共享代码 —— 而且 TP 切分逻辑也得跟着改。

把异构性**收敛在 `weight.py` 这一层**（加载期做一次性的重排），让运行时看到的永远是统一的布局，是更干净的工程选择。这也是 SGLang/vLLM 这类框架的通行做法：**模型定义层只认一种规范布局，脏活全在权重加载层**。

</details>
