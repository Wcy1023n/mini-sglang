"""Acceptance test: mini-sglang vs. the HuggingFace reference.

Feeds the *same* input_ids to mini-sglang and compares the greedy continuation
against `hf_ref.json`. Bypassing tokenization on the input side keeps the
comparison purely about the model + weight loading.

Build with `python lab/model_support/hf_ref.py` first, then run
`python lab/model_support/compare.py`.
"""

import json
import os

from server_only import require_server_model_path

MODEL_PATH = require_server_model_path()

from minisgl.core import SamplingParams
from minisgl.llm import LLM

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    ref = json.load(open(os.path.join(HERE, "hf_ref.json")))
    input_ids, expected = ref["input_ids"], ref["greedy_ids"]

    llm = LLM(MODEL_PATH, cuda_graph_max_bs=1, max_running_req=1)
    out = llm.generate(
        [input_ids],
        SamplingParams(temperature=0.0, max_tokens=len(expected), ignore_eos=True),
    )
    got = out[0]["token_ids"]

    print("input_ids :", input_ids)
    print("hf        :", expected)
    print("minisgl   :", got)
    print("minisgl txt:", repr(out[0]["text"]))

    if got == expected:
        print(f"\nPASS -- {len(expected)}/{len(expected)} greedy tokens match")
        return

    n = next((i for i, (a, b) in enumerate(zip(got, expected)) if a != b), min(len(got), len(expected)))
    print(f"\nFAIL -- first divergence at index {n}: hf={expected[n] if n < len(expected) else None} "
          f"minisgl={got[n] if n < len(got) else None}")
    print(f"       matched prefix: {n}/{len(expected)}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
