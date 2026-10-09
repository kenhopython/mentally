# -*- coding: utf-8 -*-
"""API 端點測試（TestClient，不真實呼叫 DeepSeek）"""
import pytest
from fastapi.testclient import TestClient

from api import deepseek
from api.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _no_real_deepseek(monkeypatch):
    monkeypatch.setattr(deepseek, "load_api_key", lambda: "")   # 讓 /predict 不走真實 API


def _payload():
    return {"user_id": "1",
            "eeg": {"delta": 0.12, "theta": 0.21, "alpha": 0.34, "beta": 0.25, "gamma": 0.08},
            "hrv": {"heart_rate": 78, "mean_rr": 769, "rmssd": 32, "sdnn": 41, "pnn50": 18,
                    "lf": 500, "hf": 300, "lf_hf_ratio": 1.67, "total_power": 1000},
            "eda": {"scl": 3.2, "scr_amplitude": 0.8, "scr_frequency": 5.2, "tonic": 2.8,
                    "phasic": 0.4, "eda_variability": 0.5, "eda_latency": 2.1}}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert "DEEPSEEK_API_KEY" not in str(body)


def test_model_info():
    r = client.get("/model-info")
    assert r.status_code == 200
    assert "name" in r.json()


def test_predict():
    r = client.post("/predict", json=_payload())
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"]["emotion"] in ("positive", "neutral", "negative")
    assert "probabilities" in body["prediction"]
    assert "deepseek" in body


def test_predict_includes_fusion_fields():
    """未設定 API key → DeepSeek 棄權，結果應為純 ML 且標籤不變。"""
    r = client.post("/predict", json=_payload())
    p = r.json()["prediction"]
    assert p["consensus"] == "ml_only"
    assert p["ml_emotion"] == p["emotion"]        # 棄權時標籤必相同
    assert p["confidence"] == p["confidence_ml"]  # 棄權時信心度不調整
    assert "llm_vote" in r.json()


def test_predict_invalid_input():
    r = client.post("/predict", json={})
    assert r.status_code == 200
    assert r.json()["data_quality"]["status"] == "LOW"


def test_baseline_get():
    r = client.get("/baseline/1")
    assert r.status_code == 200
    assert "baseline_status" in r.json()


def test_trend():
    r = client.get("/trend/1")
    assert r.status_code == 200
    assert "trend" in r.json()


def test_batch():
    r = client.post("/predict/batch", json={"items": [_payload(), _payload()]})
    assert r.status_code == 200
    assert r.json()["count"] == 2
