from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Set

from minisgl.core import Batch, Req
from minisgl.env import ENV


@dataclass
class DecodeManager:
    page_size: int
    running_reqs: Set[Req] = field(default_factory=set)

    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        self.running_reqs = {req for req in self.running_reqs.union(reqs) if req.can_decode}

    def remove_req(self, req: Req) -> None:
        self.running_reqs.discard(req)

    def abort_req(self, uid: int) -> Req | None:
        for req in self.running_reqs:
            if req.uid == uid:
                self.running_reqs.remove(req)
                return req
        return None

    @property
    def inflight_tokens(self) -> int:
        # 每个请求额外预留约 1 个 page 的余量 + 所有的 remain 预算
        tokens_reserved = (self.page_size - 1) * len(self.running_reqs)  # 1 page reserved
        return sum(req.remain_len for req in self.running_reqs) + tokens_reserved

    # 无需限制 decode 数量，因为早在准入阶段就用 reserved_size 与 inflight_tokens 约束了
    def schedule_next_batch(self) -> Batch | None:
        if not self.runnable:
            return None

        reqs=sorted(self.running_reqs, key=lambda req: req.uid)
        if ENV.OBSERVE:
            print(
                f"[mini-decode]: reqs={[req.uid for req in reqs]}",
                flush=True,
            )
        return Batch(reqs, phase="decode")

    @property
    def runnable(self) -> bool:
        return len(self.running_reqs) > 0
