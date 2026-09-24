"""HF transformers reference for internlm2_5-1_8b-chat.

Runs the *stock* HuggingFace stack over the same input_ids that `compare.py`
feeds to mini-sglang, and dumps the ground truth to hf_ref.json.

Two quirks worth knowing, both of which are why this is a standalone script:

1. The repo ships its own `modeling_internlm2.py` (via `auto_map`), and that
   remote code is incompatible with transformers 4.57's `generate()` -- its
   `_update_causal_mask` still assumes the pre-4.47 cache API. A plain forward
   pass is fine, so we hand-roll greedy decoding instead.
2. The tokenizer needs `use_fast=False`: the slow->fast sentencepiece converter
   is broken for this vocab under transformers 4.57.

Usage:  python lab/hf_ref.py
"""

import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PATH = os.environ.get("MODEL_PATH", "/root/models/internlm2_5-1_8b-chat")
PROMPT = "用一句话解释什么是张量并行。"
NUM_TOKENS = 16
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_ref.json")


def main() -> None:
    tok = AutoTokenizer.from_pretrained(PATH, trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        PATH, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    text = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True
    )
    ids = tok(text, return_tensors="pt").input_ids.to("cuda")
    input_ids = ids[0].tolist()
    print("INPUT_IDS:", input_ids)

    with torch.no_grad():
        step0 = model(ids).logits[0, -1].float()
    top = torch.topk(step0, 5)
    print("STEP0_TOP5:", [(int(t), round(float(v), 4)) for t, v in zip(top.indices, top.values)])

    seq, greedy = ids, []
    with torch.no_grad():
        for _ in range(NUM_TOKENS):
            nxt = int(model(seq).logits[0, -1].argmax())
            greedy.append(nxt)
            seq = torch.cat([seq, torch.tensor([[nxt]], device="cuda")], dim=1)

    print("GREEDY_IDS:", greedy)
    print("GREEDY_TEXT:", repr(tok.decode(greedy)))
    json.dump({"input_ids": input_ids, "greedy_ids": greedy}, open(OUT, "w"), indent=2)
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
