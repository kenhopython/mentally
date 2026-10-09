# -*- coding: utf-8 -*-
"""prediction 管線測試"""
import pytest

from api import prediction


def _payload():
    return {"user_id": "1",
            "eeg": {"delta": 0.12, "theta": 0.21, "alpha": 0.34, "beta": 0.25, "gamma": 0.08},
            "hrv": {"heart_rate": 78, "mean_rr": 769, "rmssd": 32, "sdnn": 41, "pnn50": 18,
                    "lf": 500, "hf": 300, "lf_hf_ratio": 1.67, "total_power": 1000},
            "eda": {"scl": 3.2, "scr_amplitude": 0.8, "scr_frequency": 5.2, "tonic": 2.8,
                    "phasic": 0.4, "eda_variability": 0.5, "eda_latency": 2.1}}


def test_normal_prediction():
    r = prediction.run_ml(_payload())
    assert r["prediction"]["emotion"] in ("positive", "neutral", "negative")
    assert 0.0 <= r["prediction"]["confidence"] <= 1.0
    p = r["prediction"]["probabilities"]
    assert abs(sum(p.values()) - 1.0) < 0.01
    assert set(p) == {"positive", "neutral", "negative"}
    assert r["data_quality"]["status"] == "GOOD"


def test_missing_data_still_runs():
    r = prediction.run_ml({"user_id": "1", "eeg": {"alpha": 0.3}, "hrv": {"rmssd": 30}})
    assert r["prediction"]["emotion"] in ("positive", "neutral", "negative")
    # 完整度低 → 品質非 GOOD
    assert r["data_quality"]["status"] in ("MODERATE", "LOW")


def test_extreme_data_no_crash():
    p = _payload()
    p["hrv"]["rmssd"] = 9999
    r = prediction.run_ml(p)
    assert r["prediction"]["emotion"] in ("positive", "neutral", "negative")


def test_no_baseline():
    r = prediction.run_ml({**_payload(), "user_id": "999999"})
    assert r["baseline"]["status"] == "INSUFFICIENT_DATA"


def test_invalid_input_empty():
    # 空輸入不應崩潰，但品質為 LOW
    r = prediction.run_ml({})
    assert r["data_quality"]["status"] == "LOW"
