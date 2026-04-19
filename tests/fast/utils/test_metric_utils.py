"""Unit tests for miles.utils.metric_utils."""

from __future__ import annotations

import math

import pytest

from miles.utils.metric_utils import compute_at_k, compute_at_k_over_metrics, compute_pass_rate


class TestComputeAtK:
    def test_basic_avg_and_max(self):
        # 2 groups of 4. group 0 = [0,0,1,1], group 1 = [0.5, 0.5, 0.5, 0.5]
        # avg = 0.5 overall; max = mean([max(0,0,1,1), max(0.5×4)]) = (1 + 0.5)/2 = 0.75
        got = compute_at_k([0.0, 0.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], group_size=4)
        assert math.isclose(got["avg@4"], 0.5)
        assert math.isclose(got["max@4"], 0.75)

    def test_single_group(self):
        got = compute_at_k([0.2, 0.4, 0.8], group_size=3)
        assert math.isclose(got["avg@3"], (0.2 + 0.4 + 0.8) / 3)
        assert math.isclose(got["max@3"], 0.8)

    def test_group_size_one_returns_avg_only(self):
        # With group_size == 1, avg@1 == max@1 == raw mean. Still report both
        # for a consistent schema.
        got = compute_at_k([0.2, 0.4, 0.6], group_size=1)
        assert math.isclose(got["avg@1"], 0.4)
        assert math.isclose(got["max@1"], 0.4)

    def test_empty_flat_values(self):
        assert compute_at_k([], group_size=4) == {}

    def test_mismatched_length_raises(self):
        with pytest.raises(AssertionError):
            compute_at_k([0.1, 0.2, 0.3], group_size=4)


class TestComputePassRateContract:
    """Make sure compute_pass_rate still matches its documented contract, since
    compute_at_k lives next door and shares group-reshape logic."""

    def test_binary_gate_counts_only_ones(self):
        # Group 0: [1, 1, 0, 0.9] → only 2 ones → pass@1 = 2/4 = 0.5
        got = compute_pass_rate([1.0, 1.0, 0.0, 0.9], group_size=4)
        # pass@1 is probability that a uniformly chosen sample is correct.
        assert math.isclose(got["pass@1"], 0.5)

    def test_group_size_one_returns_empty(self):
        assert compute_pass_rate([1.0, 0.0, 1.0], group_size=1) == {}


class TestComputeAtKOverMetrics:
    def test_emits_avg_max_for_all_metrics(self):
        streams = {
            "item_f1": [0.5, 0.5, 1.0, 0.0],  # 1 group of 4
            "row_f1": [1.0, 1.0, 1.0, 0.0],
        }
        got = compute_at_k_over_metrics(streams, group_size=4)
        # No binary metrics, so no pass@k keys.
        assert "item_f1/avg@4" in got
        assert "item_f1/max@4" in got
        assert "row_f1/avg@4" in got
        assert "row_f1/max@4" in got
        assert not any("pass@" in k for k in got)
        assert math.isclose(got["item_f1/avg@4"], 0.5)
        assert math.isclose(got["item_f1/max@4"], 1.0)

    def test_binary_metric_gets_pass_at_k(self):
        streams = {
            "em": [1.0, 0.0, 0.0, 0.0],
            "token_f1": [0.5, 0.6, 0.7, 0.8],  # continuous, pass@k should NOT appear
        }
        got = compute_at_k_over_metrics(streams, group_size=4, binary_metrics={"em"})
        # em has pass@1/2/4; token_f1 doesn't.
        assert "em/pass@1" in got
        assert "em/pass@2" in got
        assert "em/pass@4" in got
        assert "em/avg@4" in got
        assert "em/max@4" in got
        assert "token_f1/avg@4" in got
        assert "token_f1/max@4" in got
        assert not any(k.startswith("token_f1/pass") for k in got)

    def test_mismatched_group_falls_back_to_mean(self):
        # 5 values, group_size=4 → doesn't divide, skip @k, report mean.
        streams = {"item_f1": [0.1, 0.2, 0.3, 0.4, 0.5]}
        got = compute_at_k_over_metrics(streams, group_size=4)
        assert "item_f1/mean" in got
        assert "item_f1/avg@4" not in got
        assert math.isclose(got["item_f1/mean"], 0.3)

    def test_empty_streams(self):
        assert compute_at_k_over_metrics({}, group_size=4) == {}

    def test_empty_values_skipped(self):
        got = compute_at_k_over_metrics({"item_f1": []}, group_size=4)
        assert got == {}
