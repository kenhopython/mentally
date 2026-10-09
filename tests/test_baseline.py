# -*- coding: utf-8 -*-
"""baseline 模組測試"""
import numpy as np

from baseline import baseline as b


def test_compute_baseline_stats():
    r = b.compute_baseline([10, 20, 30, 40])
    assert r["mean"] == 25.0
    assert r["n"] == 4
    assert r["min"] == 10.0 and r["max"] == 40.0


def test_compute_baseline_ignores_nan():
    r = b.compute_baseline([10, None, 20, float("nan")])
    assert r["n"] == 2
    assert r["mean"] == 15.0


def test_zscore():
    assert b.zscore(45, 40, 5) == 1.0
    assert b.zscore(45, 40, 0) is None   # std=0 → None


def test_categorize_boundaries():
    assert b.categorize(0.5) == "within_baseline"
    assert b.categorize(1.5) == "slightly_above"
    assert b.categorize(-1.5) == "slightly_below"
    assert b.categorize(2.5) == "significantly_above"
    assert b.categorize(-2.5) == "significantly_below"
    assert b.categorize(None) == "insufficient"


def test_status_insufficient():
    assert b.status({"n": 3}) == "INSUFFICIENT_DATA"
    assert b.status({"n": 30}) == "READY"


def test_update_feature_merges():
    bf = b.compute_baseline([10, 20, 30])
    bf2 = b.update_feature(bf, [40, 50])
    assert bf2["n"] == 5
    assert abs(bf2["mean"] - 30.0) < 1e-6
