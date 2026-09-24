# 权重加载调用链（Phase 4/5 调查笔记）

这篇笔记记录「从启动到权重落地」的完整调用链，以及 rename 该插在哪一行、为什么。
rename 表和 wqkv 反交错的思路见 `README.md` 的接入点 ⑤ 和「卡住了再看」，这里只补
README 里没有的**调用链**和**精确插入点**。

## 完整调用链

```
python -m minisgl                              # __main__.py → server → launch_server
  └─ engine/engine.py  Engine.__init__
        engine.py:50   torch.device("meta")    # 结构先建在 meta 设备上（不占显存）
        engine.py:51   create_model(config)    # 建结构 = 你的 internlm2.py，权重为空
        engine.py:52   model.load_state_dict( _load_weight_state_dict(config) )
        │
        └─ engine.py:139  _load_weight_state_dict
              ├─ use_dummy_weight=True  → engine.py:141 随机权重，绕过 load_weight（Phase 3 走这里）
              └─ 否则 → engine.py:146  load_weight(model_path, device)
                    │
                    └─ weight.py:75  load_weight        # 生成器，逐个 yield (名字, 张量)
                          weight.py:95   raw = f.get_tensor(name)        # checkpoint 原名
                          weight.py:96   name = removeprefix("language_model.")
                          ★★★ rename 插在 96 与 97 之间 ★★★
                          weight.py:97   _shard_tensor(name, ...)        # TP 切分，按后缀
                          weight.py:100  _get_merge_info(name)           # 融合，按后缀
                          weight.py:121  yield name, tensor
        │
        └─ layers/base.py:32  BaseOP.load_state_dict   # 按 __dict__ 属性名反射
              base.py:43   state_dict.pop(前缀.属性名)   # 必须 == rename 后的 key
              base.py:52   state_dict 有剩余 → "Unexpected keys"（Phase 4 要消灭它）
```

## 为什么 rename 必须在 `_shard_tensor` 之前

`_shard_tensor`（weight.py:97）和 `_get_merge_info`（weight.py:100）都**按后缀认 key**：

- `.gate_proj` / `.up_proj` → `_SPLIT_DIM_0`（切分）＋ `_MERGE_GROUPS`（拼成 `.gate_up_proj`）
- `.o_proj` / `.down_proj` → `_SPLIT_DIM_1`（切分）

这些后缀只在 rename 之后才存在（checkpoint 里叫 `w1`/`w3`/`wo`）。所以必须先 rename，
后续的切分和融合才能认出这些后缀、做对处理。

## checkpoint 的 10 种 key（含形状）

```
model.tok_embeddings.weight            [92544, 2048]
model.layers.N.attention.wqkv.weight   [4096, 2048]
model.layers.N.attention.wo.weight
model.layers.N.attention_norm.weight
model.layers.N.ffn_norm.weight
model.layers.N.feed_forward.w1.weight
model.layers.N.feed_forward.w2.weight
model.layers.N.feed_forward.w3.weight
model.norm.weight
output.weight
```

## 运行时 key（= internlm2.py 里的属性路径）

```
model.embed_tokens.weight
model.layers.N.self_attn.qkv_proj.weight
model.layers.N.self_attn.o_proj.weight
model.layers.N.input_layernorm.weight
model.layers.N.post_attention_layernorm.weight
model.layers.N.mlp.gate_up_proj.weight     # = w1(gate) + w3(up) 融合而成
model.layers.N.mlp.down_proj.weight
model.norm.weight
lm_head.weight
```

两侧名字的逐条对照（rename 表）见 `README.md` 的「提示 A」。

## 插入点的代码形状

`weight.py` 第 95-97 行，改成：

```python
raw = f.get_tensor(name)
name = name.removeprefix("language_model.")
for old, new in _RENAME:                 # ← 你加的这段
    name = name.replace(old, new)
tensor = _shard_tensor(name, raw, tp_info.rank, tp_info.size, config.num_kv_heads)
```

`_RENAME` 建议定义在 `weight.py` 顶部（`_SPLIT_DIM_0` 等常量的位置），与现有风格一致。

## 一个容易踩的点

`attention_norm` 里也包含子串 `attention`。rename 必须用**完整子串**（`attention.wqkv`、
`attention.wo`、`attention_norm`、`ffn_norm`、`feed_forward.w1`…），不能用 `attention → self_attn`
这种朴素替换，否则 `attention_norm` 会被错改成 `self_attn_norm`。
