"""stats_tool.py — StatsTool for agent-friend (stdlib only).

Descriptive statistics for agent data analysis — no numpy, no pandas.

Features:
* stats_describe — mean, median, std, min, max, percentiles
* stats_histogram — frequency histogram with configurable bins
* stats_correlation — Pearson correlation coefficient between two series
* stats_normalize — min-max or z-score normalization
* stats_outliers — detect outliers using IQR or z-score method
* stats_moving_average — simple and exponential moving averages
* stats_frequency — frequency count for categorical data

Usage::

    tool = StatsTool()

    data = [2, 4, 4, 4, 5, 5, 7, 9]
    result = tool.stats_describe(data)
    # {"count": 8, "mean": 5.0, "median": 4.5, "std": 2.0, ...}
"""

import json
import math
from collections import Counter
from typing import Any, Dict, List, Optional, Union

from .base import BaseTool

Number = Union[int, float]


def _mean(values: List[float]) -> float:
    return sum(values) / len(values)


def _sorted_values(values: List[float]) -> List[float]:
    return sorted(values)


def _median(sorted_v: List[float]) -> float:
    n = len(sorted_v)
    mid = n // 2
    if n % 2 == 1:
        return sorted_v[mid]
    return (sorted_v[mid - 1] + sorted_v[mid]) / 2.0


def _std(values: List[float], mean: float, population: bool = False) -> float:
    n = len(values)
    denom = n if population else n - 1
    if denom == 0:
        return 0.0
    variance = sum((x - mean) ** 2 for x in values) / denom
    return math.sqrt(variance)


def _percentile(sorted_v: List[float], p: float) -> float:
    """Linear interpolation percentile."""
    n = len(sorted_v)
    if n == 1:
        return sorted_v[0]
    idx = (p / 100.0) * (n - 1)
    lo = int(idx)
    hi = lo + 1
    if hi >= n:
        return sorted_v[-1]
    frac = idx - lo
    return sorted_v[lo] * (1 - frac) + sorted_v[hi] * frac


class StatsTool(BaseTool):
    """Descriptive statistics for numeric data. No numpy, no pandas."""

    # ── public API ────────────────────────────────────────────────────

    def stats_describe(
        self,
        values: List[Number],
        percentiles: Optional[List[float]] = None,
    ) -> str:
        """Compute descriptive statistics for a numeric list.

        *percentiles* defaults to [25, 50, 75].

        Returns ``{count, mean, median, std, variance, min, max,
        range, percentiles: {p25, p50, p75, ...}}``.
        """
        if not values:
            return json.dumps({"error": "values list is empty"})

        try:
            v = [float(x) for x in values]
        except (TypeError, ValueError) as exc:
            return json.dumps({"error": f"Non-numeric value: {exc}"})

        if percentiles is None:
            percentiles = [25, 50, 75]

        sv = _sorted_values(v)
        m = _mean(v)
        med = _median(sv)
        std = _std(v, m)
        variance = std ** 2 if len(v) > 1 else 0.0

        pct = {f"p{int(p)}": round(_percentile(sv, p), 6) for p in percentiles}

        return json.dumps({
            "count": len(v),
            "mean": round(m, 6),
            "median": round(med, 6),
            "std": round(std, 6),
            "variance": round(variance, 6),
            "min": sv[0],
            "max": sv[-1],
            "range": round(sv[-1] - sv[0], 6),
            "sum": round(sum(v), 6),
            "percentiles": pct,
        })

    def stats_histogram(
        self,
        values: List[Number],
        bins: int = 10,
    ) -> str:
        """Compute a frequency histogram.

        Returns ``{bins: [{range_start, range_end, count, frequency}], total}``.
        *frequency* is count/total.
        """
        if not values:
            return json.dumps({"error": "values list is empty"})
        if bins < 1 or bins > 1000:
            return json.dumps({"error": "bins must be between 1 and 1000"})

        try:
            v = [float(x) for x in values]
        except (TypeError, ValueError) as exc:
            return json.dumps({"error": f"Non-numeric value: {exc}"})

        mn, mx = min(v), max(v)
        total = len(v)

        if mn == mx:
            return json.dumps({
                "bins": [{"range_start": mn, "range_end": mx, "count": total, "frequency": 1.0}],
                "total": total,
            })

        width = (mx - mn) / bins
        counts = [0] * bins
        for x in v:
            idx = min(int((x - mn) / width), bins - 1)
            counts[idx] += 1

        result_bins = []
        for i in range(bins):
            start = mn + i * width
            end = mn + (i + 1) * width
            result_bins.append({
                "range_start": round(start, 6),
                "range_end": round(end, 6),
                "count": counts[i],
                "frequency": round(counts[i] / total, 6),
            })

        return json.dumps({"bins": result_bins, "total": total, "bin_width": round(width, 6)})

    def stats_correlation(self, x: List[Number], y: List[Number]) -> str:
        """Pearson correlation coefficient between two series.

        Returns ``{r, r_squared, interpretation}`` where interpretation
        is one of: strong_positive, moderate_positive, weak_positive,
        none, weak_negative, moderate_negative, strong_negative.
        """
        if not x or not y:
            return json.dumps({"error": "x and y must be non-empty"})
        if len(x) != len(y):
            return json.dumps({"error": f"x and y must have the same length (got {len(x)} and {len(y)})"})
        if len(x) < 2:
            return json.dumps({"error": "Need at least 2 data points"})

        try:
            xv = [float(v) for v in x]
            yv = [float(v) for v in y]
        except (TypeError, ValueError) as exc:
            return json.dumps({"error": f"Non-numeric value: {exc}"})

        mx = _mean(xv)
        my = _mean(yv)
        num = sum((xi - mx) * (yi - my) for xi, yi in zip(xv, yv))
        dx = math.sqrt(sum((xi - mx) ** 2 for xi in xv))
        dy = math.sqrt(sum((yi - my) ** 2 for yi in yv))

        if dx == 0.0 or dy == 0.0:
            return json.dumps({"error": "One or both series have zero variance"})

        r = num / (dx * dy)
        r = max(-1.0, min(1.0, r))  # clamp float errors

        abs_r = abs(r)
        if abs_r >= 0.9:
            interp = "strong_positive" if r > 0 else "strong_negative"
        elif abs_r >= 0.7:
            interp = "moderate_positive" if r > 0 else "moderate_negative"
        elif abs_r >= 0.4:
            interp = "weak_positive" if r > 0 else "weak_negative"
        else:
            interp = "none"

        return json.dumps({
            "r": round(r, 6),
            "r_squared": round(r ** 2, 6),
            "interpretation": interp,
        })

    def stats_normalize(
        self,
        values: List[Number],
        method: str = "minmax",
    ) -> str:
        """Normalize a numeric list.

        *method*:
        * ``minmax`` — scale to [0, 1]
        * ``zscore`` — subtract mean, divide by std

        Returns ``{values: [...], method, original_min, original_max}``.
        """
        if not values:
            return json.dumps({"error": "values list is empty"})

        try:
            v = [float(x) for x in values]
        except (TypeError, ValueError) as exc:
            return json.dumps({"error": f"Non-numeric value: {exc}"})

        if method == "minmax":
            mn, mx = min(v), max(v)
            rng = mx - mn
            if rng == 0:
                normalized = [0.0] * len(v)
            else:
                normalized = [(x - mn) / rng for x in v]
            return json.dumps({
                "values": [round(x, 6) for x in normalized],
                "method": method,
                "original_min": mn,
                "original_max": mx,
            })
        elif method == "zscore":
            m = _mean(v)
            s = _std(v, m)
            if s == 0:
                normalized = [0.0] * len(v)
            else:
                normalized = [(x - m) / s for x in v]
            return json.dumps({
                "values": [round(x, 6) for x in normalized],
                "method": method,
                "original_mean": round(m, 6),
                "original_std": round(s, 6),
            })
        else:
            return json.dumps({"error": f"Unknown method '{method}'. Valid: minmax, zscore."})

    def stats_outliers(
        self,
        values: List[Number],
        method: str = "iqr",
        threshold: float = 1.5,
    ) -> str:
        """Detect outliers.

        *method*:
        * ``iqr`` — values outside Q1 - threshold*IQR .. Q3 + threshold*IQR
        * ``zscore`` — |z-score| > threshold

        Returns ``{outliers: [{index, value}], clean: [...], method}``.
        """
        if not values:
            return json.dumps({"error": "values list is empty"})

        try:
            v = [float(x) for x in values]
        except (TypeError, ValueError) as exc:
            return json.dumps({"error": f"Non-numeric value: {exc}"})

        sv = _sorted_values(v)

        if method == "iqr":
            q1 = _percentile(sv, 25)
            q3 = _percentile(sv, 75)
            iqr = q3 - q1
            lo = q1 - threshold * iqr
            hi = q3 + threshold * iqr
            outliers = [{"index": i, "value": x} for i, x in enumerate(v) if x < lo or x > hi]
            clean = [x for x in v if lo <= x <= hi]
        elif method == "zscore":
            m = _mean(v)
            s = _std(v, m)
            if s == 0:
                return json.dumps({"outliers": [], "clean": v, "method": method})
            outliers = [{"index": i, "value": x} for i, x in enumerate(v) if abs((x - m) / s) > threshold]
            outlier_idxs = {o["index"] for o in outliers}
            clean = [x for i, x in enumerate(v) if i not in outlier_idxs]
        else:
            return json.dumps({"error": f"Unknown method '{method}'. Valid: iqr, zscore."})

        return json.dumps({
            "outliers": outliers,
            "clean": clean,
            "outlier_count": len(outliers),
            "method": method,
        })

    def stats_moving_average(
        self,
        values: List[Number],
        window: int = 3,
        kind: str = "simple",
        alpha: float = 0.3,
    ) -> str:
        """Compute moving averages.

        *kind*:
        * ``simple`` (SMA) — unweighted rolling mean
        * ``exponential`` (EMA) — `alpha * x + (1-alpha) * prev`

        Returns ``{values: [...], window, kind, original_count}``.
        """
        if not values:
            return json.dumps({"error": "values list is empty"})
        if window < 1:
            return json.dumps({"error": "window must be >= 1"})

        try:
            v = [float(x) for x in values]
        except (TypeError, ValueError) as exc:
            return json.dumps({"error": f"Non-numeric value: {exc}"})

        if kind == "simple":
            result = []
            for i in range(len(v)):
                start = max(0, i - window + 1)
                result.append(round(_mean(v[start: i + 1]), 6))
        elif kind == "exponential":
            if not (0 < alpha <= 1):
                return json.dumps({"error": "alpha must be in (0, 1]"})
            result = [v[0]]
            for x in v[1:]:
                ema = alpha * x + (1 - alpha) * result[-1]
                result.append(round(ema, 6))
        else:
            return json.dumps({"error": f"Unknown kind '{kind}'. Valid: simple, exponential."})

        return json.dumps({
            "values": result,
            "window": window,
            "kind": kind,
            "original_count": len(v),
        })

    def stats_frequency(self, values: List[Any], top_n: int = 20) -> str:
        """Frequency count for categorical or discrete data.

        Returns ``{frequencies: [{value, count, percent}], total, unique}``.
        """
        if not values:
            return json.dumps({"error": "values list is empty"})

        total = len(values)
        counter = Counter(str(v) for v in values)
        most_common = counter.most_common(top_n)

        return json.dumps({
            "frequencies": [
                {"value": k, "count": c, "percent": round(c / total * 100, 2)}
                for k, c in most_common
            ],
            "total": total,
            "unique": len(counter),
        })

    # ── BaseTool interface ────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "stats"

    @property
    def description(self) -> str:
        return (
            "Descriptive statistics for numeric data. describe (mean/median/std/percentiles), "
            "histogram, Pearson correlation, min-max/z-score normalization, outlier detection "
            "(IQR/z-score), moving averages (SMA/EMA), frequency counts. No numpy. Zero deps."
        )

    def definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "stats_describe",
                "description": "Descriptive stats: count/mean/median/std/variance/min/max/range/percentiles.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "values": {"type": "array", "items": {"type": "number"}, "description": "Numeric data"},
                        "percentiles": {"type": "array", "items": {"type": "number"}, "description": "Default [25, 50, 75]"},
                    },
                    "required": ["values"],
                },
            },
            {
                "name": "stats_histogram",
                "description": "Frequency histogram. Returns bins with range, count, and frequency.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "values": {"type": "array", "items": {"type": "number"}, "description": "Numeric data"},
                        "bins": {"type": "integer", "description": "Number of bins (default 10)"},
                    },
                    "required": ["values"],
                },
            },
            {
                "name": "stats_correlation",
                "description": "Pearson correlation between x and y. Returns r, r_squared, interpretation.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "array", "items": {"type": "number"}, "description": "First series"},
                        "y": {"type": "array", "items": {"type": "number"}, "description": "Second series"},
                    },
                    "required": ["x", "y"],
                },
            },
            {
                "name": "stats_normalize",
                "description": "Normalize values. method: minmax (0..1) or zscore (mean=0, std=1).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "values": {"type": "array", "items": {"type": "number"}, "description": "Numeric data"},
                        "method": {"type": "string", "description": "minmax | zscore"},
                    },
                    "required": ["values"],
                },
            },
            {
                "name": "stats_outliers",
                "description": "Detect outliers. method: iqr (threshold=IQR multiplier) or zscore (threshold=sigma).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "values": {"type": "array", "items": {"type": "number"}, "description": "Numeric data"},
                        "method": {"type": "string", "description": "iqr | zscore"},
                        "threshold": {"type": "number", "description": "IQR: 1.5 standard, 3.0 extreme. Z-score: 2.0 or 3.0."},
                    },
                    "required": ["values"],
                },
            },
            {
                "name": "stats_moving_average",
                "description": "Moving average. kind: simple (SMA) or exponential (EMA, alpha=0.3).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "values": {"type": "array", "items": {"type": "number"}, "description": "Numeric data"},
                        "window": {"type": "integer", "description": "Window size (default 3)"},
                        "kind": {"type": "string", "description": "simple | exponential"},
                        "alpha": {"type": "number", "description": "EMA smoothing factor (0,1]"},
                    },
                    "required": ["values"],
                },
            },
            {
                "name": "stats_frequency",
                "description": "Frequency count for categorical data. Returns [{value, count, percent}].",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "values": {"type": "array", "description": "Data values"},
                        "top_n": {"type": "integer", "description": "Return top N most frequent"},
                    },
                    "required": ["values"],
                },
            },
        ]

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name == "stats_describe":
            return self.stats_describe(**arguments)
        if tool_name == "stats_histogram":
            return self.stats_histogram(**arguments)
        if tool_name == "stats_correlation":
            return self.stats_correlation(**arguments)
        if tool_name == "stats_normalize":
            return self.stats_normalize(**arguments)
        if tool_name == "stats_outliers":
            return self.stats_outliers(**arguments)
        if tool_name == "stats_moving_average":
            return self.stats_moving_average(**arguments)
        if tool_name == "stats_frequency":
            return self.stats_frequency(**arguments)
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
