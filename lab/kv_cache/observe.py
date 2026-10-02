"""Small CPU-only behavior probes for the KV cache lab.

Run from the repository root in an environment with mini-sglang installed.
The integrity and LFU probes fail until their lab features are implemented.
"""
from __future__ import annotations

import argparse
import time

import torch

from minisgl.core import Context, set_global_ctx
from minisgl.kvcache.radix_cache import RadixPrefixCache


def tensor(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32)


def make_cache(page_size: int = 1, policy: str = "default") -> RadixPrefixCache:
    set_global_ctx(Context(page_size=page_size))
    if policy == "default":
        return RadixPrefixCache(torch.device("cpu"))
    return RadixPrefixCache(torch.device("cpu"), eviction_policy=policy)


def probe_split_and_lock() -> None:
    cache = make_cache()
    cache.insert_prefix(tensor([11, 12, 13, 14]), tensor([100, 101, 102, 103]))
    match = cache.match_prefix(tensor([11, 12, 99]))
    print("[match] cached_len=", match.cuda_handle.cached_len)
    print("[before lock] size=", cache.size_info)
    assert match.cuda_handle.cached_len == 2
    assert cache.size_info.total_size == 4

    cache.lock_handle(match.cuda_handle)
    print("[locked] size=", cache.size_info)
    assert cache.size_info.evictable_size == 2
    assert cache.size_info.protected_size == 2
    assert match.cuda_handle.get_matched_indices().tolist() == [100, 101]

    evicted = cache.evict(2).tolist()
    print("[evicted] indices=", evicted)
    assert evicted == [102, 103]
    assert cache.match_prefix(tensor([11, 12, 99])).cuda_handle.cached_len == 2
    cache.lock_handle(match.cuda_handle, unlock=True)
    assert cache.size_info.evictable_size == 2
    assert cache.size_info.protected_size == 0
    print("[check] PASS")


def probe_radix_tree() -> None:
    cache = make_cache()
    cache.insert_prefix(tensor([11, 12, 13, 14]), tensor([100, 101, 102, 103]))
    cache.insert_prefix(tensor([11, 12, 15, 16]), tensor([100, 101, 200, 201]))
    shared = cache.match_prefix(tensor([11, 12, 99])).cuda_handle
    print("[branch] shared_len=", shared.cached_len, "size=", cache.size_info)
    assert shared.cached_len == 2
    assert shared.get_matched_indices().tolist() == [100, 101]
    assert cache.size_info.total_size == 6

    cache.lock_handle(shared)
    assert cache.size_info.protected_size == 2
    assert cache.size_info.evictable_size == 4
    evicted = cache.evict(4).tolist()
    print("[branch] evicted=", evicted, "size=", cache.size_info)
    assert set(evicted) == {102, 103, 200, 201}
    assert cache.match_prefix(tensor([11, 12, 99])).cuda_handle.cached_len == 2
    cache.lock_handle(shared, unlock=True)
    assert cache.size_info.protected_size == 0
    assert cache.size_info.evictable_size == 2
    assert cache.evict(2).tolist() == [100, 101]
    assert cache.size_info.total_size == 0
    print("[check] PASS")


def probe_integrity() -> None:
    cache = make_cache()
    cache.insert_prefix(tensor([11, 12, 13, 14]), tensor([100, 101, 102, 103]))
    cache.insert_prefix(tensor([11, 12, 15, 16]), tensor([100, 101, 200, 201]))
    shared = cache.match_prefix(tensor([11, 12, 99])).cuda_handle
    cache.lock_handle(shared)
    cache.check_integrity()

    cache.evictable_size += 1
    try:
        try:
            cache.check_integrity()
        except (AssertionError, RuntimeError, ValueError):
            pass
        else:
            raise AssertionError("integrity check missed a corrupted size counter")
    finally:
        cache.evictable_size -= 1

    child = next(iter(cache.root_node.children.values()))
    parent = child._parent
    child._parent = None
    try:
        try:
            cache.check_integrity()
        except (AssertionError, RuntimeError, ValueError):
            pass
        else:
            raise AssertionError("integrity check missed a broken parent link")
    finally:
        child._parent = parent

    cache.check_integrity()
    cache.lock_handle(shared, unlock=True)
    cache.evict(4)
    cache.check_integrity()
    print("[check] PASS")


def probe_page_alignment() -> None:
    cache = make_cache(page_size=4)
    cache.insert_prefix(tensor([1, 2, 3, 4, 5, 6]), tensor([40, 41, 42, 43, 44, 45]))
    hit = cache.match_prefix(tensor([1, 2, 3, 4, 5, 6]))
    miss = cache.match_prefix(tensor([1, 2, 3, 9]))
    print("[page] hit=", hit.cuda_handle.cached_len, "miss=", miss.cuda_handle.cached_len)
    print("[page] size=", cache.size_info)
    assert hit.cuda_handle.cached_len == 4
    assert miss.cuda_handle.cached_len == 0
    assert cache.size_info.total_size == 4
    print("[check] PASS")


def probe_eviction(expected: str) -> None:
    cache = make_cache()
    a_ids, b_ids = tensor([11, 12]), tensor([21, 22])
    cache.insert_prefix(a_ids, tensor([100, 101]))
    time.sleep(0.002)
    cache.insert_prefix(b_ids, tensor([200, 201]))
    time.sleep(0.002)
    a_handle = cache.match_prefix(a_ids).cuda_handle
    assert a_handle.cached_len == 2
    evicted = cache.evict(2).tolist()
    print("[eviction] recently_used=A, indices=", evicted)
    print("[eviction] size=", cache.size_info)
    expected_indices = [200, 201] if expected == "lru" else [100, 101]
    assert evicted == expected_indices
    surviving_ids = a_ids if expected == "lru" else b_ids
    assert cache.match_prefix(surviving_ids).cuda_handle.cached_len == 2
    print("[check] PASS")


def probe_frequency(policy: str, lock_b: bool) -> None:
    cache = make_cache(policy=policy)
    a_ids, b_ids = tensor([11, 12]), tensor([21, 22])
    a_values, b_values = tensor([100, 101]), tensor([200, 201])
    cache.insert_prefix(a_ids, a_values)
    cache.insert_prefix(b_ids, b_values)
    for _ in range(2):
        time.sleep(0.002)
        assert cache.insert_prefix(a_ids, a_values).cached_len == 2
    time.sleep(0.002)
    assert cache.insert_prefix(b_ids, b_values).cached_len == 2

    if lock_b:
        b_handle = cache.match_prefix(b_ids).cuda_handle
        cache.lock_handle(b_handle)
    evicted = cache.evict(2).tolist()
    print("[frequency] policy=", policy, "locked_b=", lock_b, "indices=", evicted)
    print("[frequency] size=", cache.size_info)
    expected = [100, 101] if policy == "default" or lock_b else [200, 201]
    assert evicted == expected
    surviving_ids = b_ids if expected == [100, 101] else a_ids
    assert cache.match_prefix(surviving_ids).cuda_handle.cached_len == 2
    if lock_b:
        assert cache.size_info.protected_size == 2
        assert cache.size_info.evictable_size == 0
        cache.lock_handle(b_handle, unlock=True)
        assert cache.size_info.protected_size == 0
        assert cache.size_info.evictable_size == 2
    print("[check] PASS")


def probe_manager() -> None:
    from minisgl.scheduler.cache import CacheManager

    set_global_ctx(Context(page_size=1))
    page_table = torch.empty((1, 4), dtype=torch.int32)
    manager = CacheManager(4, 1, page_table, type="radix", radix_eviction_policy="lfu")
    occupied = manager._allocate(4)
    assert occupied.tolist() == [0, 1, 2, 3]

    a_ids, b_ids = tensor([11, 12]), tensor([21, 22])
    manager.prefix_cache.insert_prefix(a_ids, occupied[:2])
    manager.prefix_cache.insert_prefix(b_ids, occupied[2:])
    for _ in range(2):
        time.sleep(0.002)
        manager.prefix_cache.insert_prefix(a_ids, occupied[:2])
    time.sleep(0.002)
    manager.prefix_cache.insert_prefix(b_ids, occupied[2:])

    recycled = manager._allocate(2)
    print("[manager] recycled indices=", recycled.tolist())
    assert recycled.tolist() == [2, 3]
    manager._free(recycled)
    manager.check_integrity()
    print("[check] PASS")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--case",
        choices=["split-lock", "radix-tree", "integrity", "page", "eviction", "frequency", "locked-frequency", "manager"],
        required=True,
    )
    ap.add_argument("--policy", choices=["default", "lfu"], default="default")
    ap.add_argument("--expect", choices=["lru", "mru"])
    args = ap.parse_args()
    if args.case == "split-lock":
        probe_split_and_lock()
    elif args.case == "radix-tree":
        probe_radix_tree()
    elif args.case == "integrity":
        probe_integrity()
    elif args.case == "page":
        probe_page_alignment()
    elif args.case == "eviction":
        if args.expect is None:
            ap.error("--expect is required for the eviction probe")
        probe_eviction(args.expect)
    elif args.case == "manager":
        probe_manager()
    else:
        probe_frequency(args.policy, lock_b=args.case == "locked-frequency")


if __name__ == "__main__":
    main()
