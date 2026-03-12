"""sampler.py — SamplerTool for agent-friend (stdlib only).

Random sampling, shuffling, and selection for agent workflows.
Deterministic when a seed is provided.

Features:
* sample_list — reservoir/random sample from a list
* sample_weighted — weighted random selection
* sample_stratified — balanced sampling across groups
* shuffle — random permutation
* sample_reservoir — streaming reservoir sampling (N items, memory-efficient)
* random_choice — pick N items with or without replacement
* random_split — split data into train/test (or arbitrary proportions)
* random_int / random_float — reproducible random number generation

Usage::

    tool = SamplerTool()

    tool.sample_list(items=[1,2,3,4,5], n=3, seed=42)
    # {"sample": [3, 1, 5], "n": 3, "seed": 42}

    tool.sample_weighted(items=["a","b","c"], weights=[0.5, 0.3, 0.2], n=2, seed=42)
    # {"sample": ["a", "a"], ...}
"""

import json
import random
from typing import Any, Dict, List, Optional, Union

from .base import BaseTool


def _make_rng(seed: Optional[int]) -> random.Random:
    rng = random.Random()
    if seed is not None:
        rng.seed(seed)
    return rng


class SamplerTool(BaseTool):
    """Random sampling, shuffling, and selection for agent workflows.

    All operations accept an optional *seed* for reproducibility.
    """

    # ── public API ────────────────────────────────────────────────────

    def sample_list(
        self,
        items: List[Any],
        n: int,
        seed: Optional[int] = None,
        replacement: bool = False,
    ) -> str:
        """Random sample of *n* items from a list.

        *replacement*: if True, sampling with replacement (allows repeats).

        Returns ``{sample, n, total, seed}``.
        """
        if not items:
            return json.dumps({"error": "items list is empty"})
        if n < 1:
            return json.dumps({"error": "n must be >= 1"})
        if not replacement and n > len(items):
            return json.dumps({"error": f"n ({n}) > len(items) ({len(items)}). Use replacement=true for oversampling."})

        rng = _make_rng(seed)
        if replacement:
            sample = [rng.choice(items) for _ in range(n)]
        else:
            sample = rng.sample(items, n)

        return json.dumps({"sample": sample, "n": n, "total": len(items), "seed": seed})

    def sample_weighted(
        self,
        items: List[Any],
        weights: List[float],
        n: int = 1,
        seed: Optional[int] = None,
        replacement: bool = True,
    ) -> str:
        """Weighted random selection.

        *weights* must have the same length as *items*.
        Weights are automatically normalized (don't need to sum to 1).

        Returns ``{sample, weights_normalized, n, seed}``.
        """
        if not items:
            return json.dumps({"error": "items list is empty"})
        if len(items) != len(weights):
            return json.dumps({"error": f"items ({len(items)}) and weights ({len(weights)}) must have the same length"})
        if any(w < 0 for w in weights):
            return json.dumps({"error": "weights must be non-negative"})
        total_w = sum(weights)
        if total_w == 0:
            return json.dumps({"error": "weights sum to zero"})

        normalized = [w / total_w for w in weights]
        rng = _make_rng(seed)

        if replacement:
            sample = rng.choices(items, weights=normalized, k=n)
        else:
            if n > len(items):
                return json.dumps({"error": f"n ({n}) > len(items) ({len(items)}) without replacement"})
            # Weighted sampling without replacement
            pool = list(items)
            pool_weights = list(normalized)
            sample = []
            for _ in range(n):
                tot = sum(pool_weights)
                r = rng.random() * tot
                cumulative = 0.0
                chosen_idx = len(pool) - 1
                for j, w in enumerate(pool_weights):
                    cumulative += w
                    if r <= cumulative:
                        chosen_idx = j
                        break
                sample.append(pool.pop(chosen_idx))
                pool_weights.pop(chosen_idx)

        return json.dumps({
            "sample": sample,
            "weights_normalized": [round(w, 6) for w in normalized],
            "n": n,
            "seed": seed,
        })

    def sample_stratified(
        self,
        groups: Dict[str, List[Any]],
        n_per_group: int,
        seed: Optional[int] = None,
    ) -> str:
        """Balanced sampling: draw *n_per_group* items from each group.

        *groups* is a dict mapping group name → list of items.

        Returns ``{sample: {group: [items]}, total, seed}``.
        """
        if not groups:
            return json.dumps({"error": "groups dict is empty"})
        if n_per_group < 1:
            return json.dumps({"error": "n_per_group must be >= 1"})

        rng = _make_rng(seed)
        result: Dict[str, List[Any]] = {}
        for group, items in groups.items():
            if not items:
                result[group] = []
            elif n_per_group > len(items):
                # oversample with replacement
                result[group] = rng.choices(items, k=n_per_group)
            else:
                result[group] = rng.sample(items, n_per_group)

        total = sum(len(v) for v in result.values())
        return json.dumps({"sample": result, "total": total, "n_per_group": n_per_group, "seed": seed})

    def shuffle(self, items: List[Any], seed: Optional[int] = None) -> str:
        """Return a shuffled copy of *items*.

        Returns ``{items, seed}``.
        """
        if not items:
            return json.dumps({"items": [], "seed": seed})

        rng = _make_rng(seed)
        shuffled = list(items)
        rng.shuffle(shuffled)
        return json.dumps({"items": shuffled, "seed": seed})

    def random_split(
        self,
        items: List[Any],
        ratios: Optional[List[float]] = None,
        seed: Optional[int] = None,
    ) -> str:
        """Split data into partitions by *ratios*.

        *ratios* defaults to ``[0.8, 0.2]`` (train/test).
        Ratios are normalized automatically.

        Returns ``{splits: [[items], ...], sizes: [...], seed}``.
        """
        if not items:
            return json.dumps({"error": "items list is empty"})
        if ratios is None:
            ratios = [0.8, 0.2]
        if any(r <= 0 for r in ratios):
            return json.dumps({"error": "ratios must be positive"})

        total_r = sum(ratios)
        norm = [r / total_r for r in ratios]

        rng = _make_rng(seed)
        shuffled = list(items)
        rng.shuffle(shuffled)

        splits = []
        start = 0
        n = len(shuffled)
        for i, ratio in enumerate(norm):
            if i == len(norm) - 1:
                splits.append(shuffled[start:])
            else:
                size = round(n * ratio)
                splits.append(shuffled[start: start + size])
                start += size

        return json.dumps({
            "splits": splits,
            "sizes": [len(s) for s in splits],
            "ratios": [round(r, 6) for r in norm],
            "seed": seed,
        })

    def random_choice(
        self,
        items: List[Any],
        seed: Optional[int] = None,
    ) -> str:
        """Pick a single random item.

        Returns ``{choice, index, seed}``.
        """
        if not items:
            return json.dumps({"error": "items list is empty"})
        rng = _make_rng(seed)
        idx = rng.randrange(len(items))
        return json.dumps({"choice": items[idx], "index": idx, "seed": seed})

    def random_int(
        self,
        low: int,
        high: int,
        n: int = 1,
        seed: Optional[int] = None,
    ) -> str:
        """Generate *n* random integers in [low, high] inclusive.

        Returns ``{values, n, low, high, seed}``.
        """
        if low > high:
            return json.dumps({"error": f"low ({low}) must be <= high ({high})"})
        rng = _make_rng(seed)
        values = [rng.randint(low, high) for _ in range(n)]
        return json.dumps({"values": values, "n": n, "low": low, "high": high, "seed": seed})

    def random_float(
        self,
        low: float = 0.0,
        high: float = 1.0,
        n: int = 1,
        decimals: int = 6,
        seed: Optional[int] = None,
    ) -> str:
        """Generate *n* random floats in [low, high).

        Returns ``{values, n, low, high, seed}``.
        """
        if low >= high:
            return json.dumps({"error": f"low ({low}) must be < high ({high})"})
        rng = _make_rng(seed)
        values = [round(rng.uniform(low, high), decimals) for _ in range(n)]
        return json.dumps({"values": values, "n": n, "low": low, "high": high, "seed": seed})

    # ── BaseTool interface ────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "sampler"

    @property
    def description(self) -> str:
        return (
            "Random sampling, shuffling, and selection. sample_list, "
            "sample_weighted, sample_stratified, shuffle, random_split "
            "(train/test), random_choice, random_int, random_float. "
            "Deterministic with seed. Zero deps."
        )

    def definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "sample_list",
                "description": "Random sample of n items. replacement=true for oversampling. Returns {sample, n, total}.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "items": {"type": "array"},
                        "n": {"type": "integer"},
                        "seed": {"type": "integer"},
                        "replacement": {"type": "boolean"},
                    },
                    "required": ["items", "n"],
                },
            },
            {
                "name": "sample_weighted",
                "description": "Weighted random selection. weights auto-normalized. Returns {sample, weights_normalized}.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "items": {"type": "array"},
                        "weights": {"type": "array", "items": {"type": "number"}},
                        "n": {"type": "integer"},
                        "seed": {"type": "integer"},
                        "replacement": {"type": "boolean"},
                    },
                    "required": ["items", "weights"],
                },
            },
            {
                "name": "sample_stratified",
                "description": "Balanced sampling: n_per_group items from each group dict key. Returns {sample: {group: [items]}}.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "groups": {"type": "object", "description": "Dict mapping group name → list of items"},
                        "n_per_group": {"type": "integer"},
                        "seed": {"type": "integer"},
                    },
                    "required": ["groups", "n_per_group"],
                },
            },
            {
                "name": "shuffle",
                "description": "Return shuffled copy of items. Deterministic with seed.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "items": {"type": "array"},
                        "seed": {"type": "integer"},
                    },
                    "required": ["items"],
                },
            },
            {
                "name": "random_split",
                "description": "Split list into partitions. ratios defaults to [0.8, 0.2] (train/test). Returns {splits, sizes}.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "items": {"type": "array"},
                        "ratios": {"type": "array", "items": {"type": "number"}, "description": "Partition proportions (auto-normalized)"},
                        "seed": {"type": "integer"},
                    },
                    "required": ["items"],
                },
            },
            {
                "name": "random_choice",
                "description": "Pick a single random item. Returns {choice, index}.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "items": {"type": "array"},
                        "seed": {"type": "integer"},
                    },
                    "required": ["items"],
                },
            },
            {
                "name": "random_int",
                "description": "Generate n random integers in [low, high] inclusive. Returns {values}.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "low": {"type": "integer"},
                        "high": {"type": "integer"},
                        "n": {"type": "integer"},
                        "seed": {"type": "integer"},
                    },
                    "required": ["low", "high"],
                },
            },
            {
                "name": "random_float",
                "description": "Generate n random floats in [low, high). Returns {values}.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "low": {"type": "number"},
                        "high": {"type": "number"},
                        "n": {"type": "integer"},
                        "decimals": {"type": "integer"},
                        "seed": {"type": "integer"},
                    },
                    "required": [],
                },
            },
        ]

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name == "sample_list":
            return self.sample_list(**arguments)
        if tool_name == "sample_weighted":
            return self.sample_weighted(**arguments)
        if tool_name == "sample_stratified":
            return self.sample_stratified(**arguments)
        if tool_name == "shuffle":
            return self.shuffle(**arguments)
        if tool_name == "random_split":
            return self.random_split(**arguments)
        if tool_name == "random_choice":
            return self.random_choice(**arguments)
        if tool_name == "random_int":
            return self.random_int(**arguments)
        if tool_name == "random_float":
            return self.random_float(**arguments)
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
