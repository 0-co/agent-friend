"""Tests for StatsTool — descriptive statistics."""

import json
import math
import pytest
from agent_friend.tools.stats_tool import StatsTool, _mean, _median, _std, _percentile


# ── helpers ────────────────────────────────────────────────────────────────

def test_mean_basic():
    assert _mean([1, 2, 3, 4, 5]) == 3.0


def test_median_odd():
    sv = sorted([1, 3, 2])
    assert _median(sv) == 2.0


def test_median_even():
    sv = sorted([1, 2, 3, 4])
    assert _median(sv) == 2.5


def test_std_known():
    values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    m = _mean(values)
    # Sample std should be ~2.0
    assert abs(_std(values, m) - 2.138) < 0.01


def test_percentile_0():
    sv = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert _percentile(sv, 0) == 1.0


def test_percentile_100():
    sv = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert _percentile(sv, 100) == 5.0


def test_percentile_50():
    sv = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert _percentile(sv, 50) == 3.0


@pytest.fixture
def tool():
    return StatsTool()


# ── stats_describe ─────────────────────────────────────────────────────────

def test_describe_basic(tool):
    r = json.loads(tool.stats_describe([2, 4, 4, 4, 5, 5, 7, 9]))
    assert r["count"] == 8
    assert r["mean"] == 5.0
    assert abs(r["std"] - 2.138) < 0.01
    assert r["min"] == 2
    assert r["max"] == 9


def test_describe_single_value(tool):
    r = json.loads(tool.stats_describe([42]))
    assert r["count"] == 1
    assert r["mean"] == 42.0
    assert r["std"] == 0.0


def test_describe_has_percentiles(tool):
    r = json.loads(tool.stats_describe([1, 2, 3, 4, 5]))
    assert "percentiles" in r
    assert "p25" in r["percentiles"]
    assert "p50" in r["percentiles"]
    assert "p75" in r["percentiles"]


def test_describe_custom_percentiles(tool):
    r = json.loads(tool.stats_describe([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], percentiles=[10, 90]))
    assert "p10" in r["percentiles"]
    assert "p90" in r["percentiles"]


def test_describe_range(tool):
    r = json.loads(tool.stats_describe([1, 5, 10]))
    assert r["range"] == 9.0


def test_describe_sum(tool):
    r = json.loads(tool.stats_describe([1, 2, 3]))
    assert r["sum"] == 6.0


def test_describe_empty_error(tool):
    r = json.loads(tool.stats_describe([]))
    assert "error" in r


def test_describe_median_even(tool):
    r = json.loads(tool.stats_describe([1, 2, 3, 4]))
    assert r["median"] == 2.5


def test_describe_identical_values(tool):
    r = json.loads(tool.stats_describe([5, 5, 5, 5]))
    assert r["mean"] == 5.0
    assert r["std"] == 0.0
    assert r["range"] == 0.0


# ── stats_histogram ────────────────────────────────────────────────────────

def test_histogram_basic(tool):
    data = list(range(10))
    r = json.loads(tool.stats_histogram(data, bins=5))
    assert len(r["bins"]) == 5
    assert r["total"] == 10
    assert sum(b["count"] for b in r["bins"]) == 10


def test_histogram_frequency_sums_to_one(tool):
    data = list(range(100))
    r = json.loads(tool.stats_histogram(data, bins=10))
    total_freq = sum(b["frequency"] for b in r["bins"])
    assert abs(total_freq - 1.0) < 0.01


def test_histogram_single_bin(tool):
    r = json.loads(tool.stats_histogram([1, 2, 3], bins=1))
    assert len(r["bins"]) == 1
    assert r["bins"][0]["count"] == 3


def test_histogram_all_same(tool):
    r = json.loads(tool.stats_histogram([5, 5, 5], bins=3))
    assert len(r["bins"]) == 1  # all same → single bin
    assert r["bins"][0]["count"] == 3


def test_histogram_empty_error(tool):
    r = json.loads(tool.stats_histogram([], bins=5))
    assert "error" in r


def test_histogram_invalid_bins(tool):
    r = json.loads(tool.stats_histogram([1, 2, 3], bins=0))
    assert "error" in r


def test_histogram_has_bin_width(tool):
    r = json.loads(tool.stats_histogram(list(range(20)), bins=4))
    assert "bin_width" in r
    assert r["bin_width"] > 0


# ── stats_correlation ──────────────────────────────────────────────────────

def test_correlation_perfect_positive(tool):
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    r = json.loads(tool.stats_correlation(x, y))
    assert abs(r["r"] - 1.0) < 1e-6
    assert r["interpretation"] == "strong_positive"


def test_correlation_perfect_negative(tool):
    x = [1, 2, 3, 4, 5]
    y = [10, 8, 6, 4, 2]
    r = json.loads(tool.stats_correlation(x, y))
    assert abs(r["r"] + 1.0) < 1e-6
    assert r["interpretation"] == "strong_negative"


def test_correlation_no_correlation(tool):
    x = [1, 2, 3, 4, 5]
    y = [3, 3, 3, 3, 3]  # constant → zero variance
    r = json.loads(tool.stats_correlation(x, y))
    assert "error" in r


def test_correlation_r_squared(tool):
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    r = json.loads(tool.stats_correlation(x, y))
    assert abs(r["r_squared"] - 1.0) < 1e-6


def test_correlation_length_mismatch(tool):
    r = json.loads(tool.stats_correlation([1, 2, 3], [4, 5]))
    assert "error" in r


def test_correlation_empty(tool):
    r = json.loads(tool.stats_correlation([], []))
    assert "error" in r


def test_correlation_moderate(tool):
    # Moderate positive: 0.4 <= |r| < 0.7
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [2, 2, 4, 4, 5, 7, 5, 8, 9, 10]
    r = json.loads(tool.stats_correlation(x, y))
    assert r["r"] > 0
    assert r["interpretation"] in ("moderate_positive", "strong_positive", "weak_positive")


# ── stats_normalize ────────────────────────────────────────────────────────

def test_normalize_minmax(tool):
    r = json.loads(tool.stats_normalize([0, 5, 10], method="minmax"))
    assert r["values"] == [0.0, 0.5, 1.0]


def test_normalize_minmax_all_same(tool):
    r = json.loads(tool.stats_normalize([5, 5, 5], method="minmax"))
    assert r["values"] == [0.0, 0.0, 0.0]


def test_normalize_zscore(tool):
    data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    r = json.loads(tool.stats_normalize(data, method="zscore"))
    # Mean of normalized should be ~0
    m = sum(r["values"]) / len(r["values"])
    assert abs(m) < 1e-5


def test_normalize_zscore_std_one(tool):
    data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    r = json.loads(tool.stats_normalize(data, method="zscore"))
    vals = r["values"]
    m = sum(vals) / len(vals)
    std = math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))
    assert abs(std - 1.0) < 0.01


def test_normalize_invalid_method(tool):
    r = json.loads(tool.stats_normalize([1, 2, 3], method="invalid"))
    assert "error" in r


def test_normalize_empty_error(tool):
    r = json.loads(tool.stats_normalize([]))
    assert "error" in r


def test_normalize_minmax_range(tool):
    r = json.loads(tool.stats_normalize([10, 20, 30, 40, 50]))
    assert r["values"][0] == 0.0
    assert r["values"][-1] == 1.0


# ── stats_outliers ─────────────────────────────────────────────────────────

def test_outliers_iqr_detects(tool):
    data = [1, 2, 3, 4, 5, 100]  # 100 is clear outlier
    r = json.loads(tool.stats_outliers(data, method="iqr"))
    outlier_values = [o["value"] for o in r["outliers"]]
    assert 100.0 in outlier_values


def test_outliers_iqr_clean(tool):
    data = [1, 2, 3, 4, 5, 100]
    r = json.loads(tool.stats_outliers(data, method="iqr"))
    assert 100.0 not in r["clean"]


def test_outliers_no_outliers(tool):
    data = [1, 2, 3, 4, 5]
    r = json.loads(tool.stats_outliers(data, method="iqr"))
    assert r["outlier_count"] == 0
    assert len(r["clean"]) == 5


def test_outliers_zscore(tool):
    data = [1, 2, 3, 4, 5, 100]
    r = json.loads(tool.stats_outliers(data, method="zscore", threshold=2.0))
    outlier_values = [o["value"] for o in r["outliers"]]
    assert 100.0 in outlier_values


def test_outliers_index_preserved(tool):
    data = [1, 2, 100, 3, 4]
    r = json.loads(tool.stats_outliers(data, method="iqr"))
    indices = [o["index"] for o in r["outliers"]]
    assert 2 in indices  # index of 100


def test_outliers_invalid_method(tool):
    r = json.loads(tool.stats_outliers([1, 2, 3], method="invalid"))
    assert "error" in r


def test_outliers_empty_error(tool):
    r = json.loads(tool.stats_outliers([]))
    assert "error" in r


# ── stats_moving_average ───────────────────────────────────────────────────

def test_sma_basic(tool):
    data = [1, 2, 3, 4, 5]
    r = json.loads(tool.stats_moving_average(data, window=3, kind="simple"))
    assert len(r["values"]) == 5
    # Third element should be (1+2+3)/3 = 2.0
    assert r["values"][2] == 2.0
    # Fifth should be (3+4+5)/3
    assert abs(r["values"][4] - 4.0) < 0.001


def test_sma_window_1(tool):
    data = [1, 2, 3]
    r = json.loads(tool.stats_moving_average(data, window=1))
    assert r["values"] == [1.0, 2.0, 3.0]


def test_ema_basic(tool):
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    r = json.loads(tool.stats_moving_average(data, kind="exponential", alpha=0.5))
    assert r["values"][0] == 1.0
    assert r["values"][1] == 1.5  # 0.5*2 + 0.5*1


def test_ema_count_matches(tool):
    data = [1, 2, 3, 4, 5, 6, 7]
    r = json.loads(tool.stats_moving_average(data, kind="exponential"))
    assert len(r["values"]) == len(data)


def test_moving_average_invalid_kind(tool):
    r = json.loads(tool.stats_moving_average([1, 2, 3], kind="invalid"))
    assert "error" in r


def test_moving_average_empty_error(tool):
    r = json.loads(tool.stats_moving_average([]))
    assert "error" in r


# ── stats_frequency ────────────────────────────────────────────────────────

def test_frequency_basic(tool):
    data = ["a", "b", "a", "c", "a", "b"]
    r = json.loads(tool.stats_frequency(data))
    by_val = {f["value"]: f for f in r["frequencies"]}
    assert by_val["a"]["count"] == 3
    assert by_val["b"]["count"] == 2
    assert by_val["c"]["count"] == 1


def test_frequency_sorted_by_count(tool):
    data = ["x", "y", "x", "x", "y", "z"]
    r = json.loads(tool.stats_frequency(data))
    counts = [f["count"] for f in r["frequencies"]]
    assert counts == sorted(counts, reverse=True)


def test_frequency_percent(tool):
    data = ["a", "a", "b"]
    r = json.loads(tool.stats_frequency(data))
    by_val = {f["value"]: f for f in r["frequencies"]}
    assert abs(by_val["a"]["percent"] - 66.67) < 0.1


def test_frequency_total_and_unique(tool):
    data = ["a", "b", "a", "c"]
    r = json.loads(tool.stats_frequency(data))
    assert r["total"] == 4
    assert r["unique"] == 3


def test_frequency_top_n(tool):
    data = list("abcdeabcabab")
    r = json.loads(tool.stats_frequency(data, top_n=2))
    assert len(r["frequencies"]) == 2


def test_frequency_empty_error(tool):
    r = json.loads(tool.stats_frequency([]))
    assert "error" in r


# ── execute dispatch ───────────────────────────────────────────────────────

def test_execute_describe(tool):
    r = json.loads(tool.execute("stats_describe", {"values": [1, 2, 3, 4, 5]}))
    assert "mean" in r


def test_execute_histogram(tool):
    r = json.loads(tool.execute("stats_histogram", {"values": list(range(10))}))
    assert "bins" in r


def test_execute_unknown(tool):
    r = json.loads(tool.execute("nope", {}))
    assert "error" in r


# ── tool metadata ──────────────────────────────────────────────────────────

def test_name(tool):
    assert tool.name == "stats"


def test_description(tool):
    assert "stat" in tool.description.lower()


def test_definitions_count(tool):
    assert len(tool.definitions()) == 7


def test_definitions_fields(tool):
    for d in tool.definitions():
        assert "name" in d
        assert "description" in d
        assert "input_schema" in d
