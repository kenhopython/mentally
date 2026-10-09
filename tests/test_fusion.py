# -*- coding: utf-8 -*-
"""雙分類器投票融合測試（ML ＋ DeepSeek），不真實呼叫 API。"""
import json

import pytest

from api import deepseek, fusion


class _FakeResp:
    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def _ml_result(emotion="negative", confidence=0.5):
    return {
        "prediction": {"emotion": emotion, "confidence": confidence,
                       "probabilities": {"positive": 0.2, "neutral": 0.3, "negative": 0.5}},
        "data_quality": {"status": "GOOD", "score": 0.95},
        "baseline": {"status": "READY", "comparison": {}},
        "trend": {},
    }


@pytest.fixture(autouse=True)
def _clear():
    deepseek.clear_vote_cache()
    yield
    deepseek.clear_vote_cache()


# --------------------------- 融合規則 ---------------------------

def test_agreement_keeps_label_and_boosts_confidence():
    r = fusion.fuse(_ml_result("negative", 0.5),
                    {"status": "ok", "label": "negative", "llm_confidence": 0.7})
    p = r["prediction"]
    assert p["emotion"] == "negative"          # 標籤不變
    assert p["model_agreement"] is True
    assert p["consensus"] == "high"
    assert p["confidence_ml"] == 0.5
    assert p["confidence"] > 0.5               # 提高
    assert p["confidence"] <= 1.0


def test_disagreement_keeps_ml_label_and_penalises_confidence():
    r = fusion.fuse(_ml_result("negative", 0.5),
                    {"status": "ok", "label": "neutral", "llm_confidence": 0.6})
    p = r["prediction"]
    assert p["emotion"] == "negative"          # 以 ML 為準
    assert p["llm_emotion"] == "neutral"
    assert p["model_agreement"] is False
    assert p["consensus"] == "low"
    assert p["confidence"] < 0.5               # 打折
    assert p["confidence"] > 0.0


@pytest.mark.parametrize("status", ["unavailable", "throttled", "no_api_key",
                                    "invalid_label", "skipped"])
def test_abstain_falls_back_to_ml(status):
    r = fusion.fuse(_ml_result("neutral", 0.42), {"status": status, "label": None})
    p = r["prediction"]
    assert p["emotion"] == "neutral"
    assert p["consensus"] == "ml_only"
    assert p["model_agreement"] is None
    assert p["confidence"] == 0.42             # 完全不動


def test_confidence_never_exceeds_one():
    r = fusion.fuse(_ml_result("positive", 0.99),
                    {"status": "ok", "label": "positive"})
    assert r["prediction"]["confidence"] <= 1.0


def test_vote_result_attached():
    r = fusion.fuse(_ml_result(), {"status": "ok", "label": "negative",
                                   "llm_confidence": 0.8, "reason": "EDA 偏高"})
    assert r["llm_vote"]["label"] == "negative"
    assert r["llm_vote"]["llm_confidence"] == 0.8


def test_low_quality_skips_vote():
    res = _ml_result()
    res["data_quality"]["status"] = "LOW"
    v = fusion.llm_vote(res)                   # 不應呼叫 API
    assert v["status"] == "skipped"


# --------------------------- DeepSeek vote() ---------------------------

def test_vote_no_key(monkeypatch):
    monkeypatch.setattr(deepseek, "load_api_key", lambda: "")
    v = deepseek.vote({"probabilities": {}})
    assert v["status"] == "no_api_key"
    assert v["label"] is None


def test_vote_ok(monkeypatch):
    monkeypatch.setattr(deepseek, "load_api_key", lambda: "k")
    monkeypatch.setattr(deepseek.requests, "post", lambda *a, **k: _FakeResp(
        json.dumps({"label": "negative", "llm_confidence": 0.66, "reason": "r"})))
    v = deepseek.vote({"probabilities": {"negative": 0.5}})
    assert v["status"] == "ok"
    assert v["label"] == "negative"
    assert v["llm_confidence"] == 0.66


def test_vote_rejects_unknown_label(monkeypatch):
    monkeypatch.setattr(deepseek, "load_api_key", lambda: "k")
    monkeypatch.setattr(deepseek.requests, "post", lambda *a, **k: _FakeResp(
        json.dumps({"label": "ANGER", "llm_confidence": 0.9})))
    v = deepseek.vote({"probabilities": {"negative": 0.5}})
    assert v["status"] == "invalid_label"
    assert v["label"] is None                  # 髒標籤不得進入預測


def test_vote_is_cached(monkeypatch):
    monkeypatch.setattr(deepseek, "load_api_key", lambda: "k")
    calls = []

    def counting_post(*a, **k):
        calls.append(1)
        return _FakeResp(json.dumps({"label": "neutral", "llm_confidence": 0.5}))

    monkeypatch.setattr(deepseek.requests, "post", counting_post)
    inp = {"probabilities": {"neutral": 0.4}}
    deepseek.vote(inp)
    v2 = deepseek.vote(inp)
    assert len(calls) == 1                     # 第二次走快取
    assert v2["cached"] is True


def test_vote_throttled_when_too_soon(monkeypatch):
    monkeypatch.setattr(deepseek, "load_api_key", lambda: "k")
    monkeypatch.setattr(deepseek, "VOTE_MIN_INTERVAL", 9999.0)
    monkeypatch.setattr(deepseek.requests, "post", lambda *a, **k: _FakeResp(
        json.dumps({"label": "neutral", "llm_confidence": 0.5})))
    deepseek.vote({"probabilities": {"a": 1}})            # 第一次成功
    v = deepseek.vote({"probabilities": {"b": 2}})        # 不同輸入 → 需呼叫，但被節流
    assert v["status"] == "throttled"


def test_vote_api_error(monkeypatch):
    monkeypatch.setattr(deepseek, "load_api_key", lambda: "k")

    def boom(*a, **k):
        raise RuntimeError("timeout")

    monkeypatch.setattr(deepseek.requests, "post", boom)
    assert deepseek.vote({"probabilities": {"a": 1}})["status"] == "unavailable"
