# -*- coding: utf-8 -*-
"""data quality 測試"""
from api.quality import check_signal_quality


def _full():
    return {"delta": 0.12, "theta": 0.21, "alpha": 0.34, "beta": 0.25, "gamma": 0.08,
            "theta_beta_ratio": 0.84, "alpha_beta_ratio": 1.36, "alpha_theta_ratio": 1.62,
            "heart_rate": 78, "mean_rr": 769, "rmssd": 32, "sdnn": 41, "pnn50": 18,
            "lf": 500, "hf": 300, "lf_hf_ratio": 1.67, "total_power": 1000,
            "scl": 3.2, "scr_amplitude": 0.8, "scr_frequency": 5.2, "tonic": 2.8,
            "phasic": 0.4, "eda_variability": 0.5, "eda_latency": 2.1}


def test_good_quality():
    r = check_signal_quality(_full())
    assert r["status"] == "GOOD"
    assert r["issues"] == []


def test_missing_data_lowers_score():
    r = check_signal_quality({"alpha": 0.3, "rmssd": 30, "scl": 3.0})
    assert r["score"] < 0.8
    assert r["status"] in ("MODERATE", "LOW")


def test_impossible_negative_value():
    d = _full()
    d["rmssd"] = -5
    r = check_signal_quality(d)
    assert any("rmssd" in i for i in r["issues"])


def test_extreme_hr():
    d = _full()
    d["heart_rate"] = 400
    r = check_signal_quality(d)
    assert any("heart_rate" in i for i in r["issues"])
