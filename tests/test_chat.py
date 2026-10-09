# -*- coding: utf-8 -*-
"""AI 對話測試（不真實呼叫 API）"""
import pytest

from api import chat, device


class _FakeResp:
    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


@pytest.fixture(autouse=True)
def _reset_latest():
    saved = dict(device.latest)
    device.latest.update({"result": None, "source": None, "ts": 0.0})
    yield
    device.latest.update(saved)


def _fake_result():
    return {
        "prediction": {"emotion": "negative", "confidence": 0.72,
                       "probabilities": {"positive": 0.1, "neutral": 0.18, "negative": 0.72},
                       "model_agreement": True},
        "data_quality": {"status": "GOOD", "score": 0.92},
        "baseline": {"status": "READY", "comparison": {
            "scl": {"z_score": 2.1, "category": "significantly_above"},
            "rmssd": {"z_score": -1.4, "category": "slightly_below"},
            "alpha": {"z_score": 0.3, "category": "within_baseline"},
            "beta": {"z_score": None},
        }},
        "trend": {"trend": "stable"},
    }


# --------------------------- build_context ---------------------------

def test_context_without_data():
    ctx = chat.build_context()
    assert ctx["has_data"] is False


def test_context_reads_latest_result():
    device.store_result(_fake_result(), source="simulated")
    ctx = chat.build_context()
    assert ctx["has_data"] is True
    assert "負面" in ctx["ml_emotion"]
    assert ctx["confidence_percent"] == 72
    assert ctx["data_quality"] == "GOOD"


def test_context_sorts_deviations_and_skips_nulls():
    device.store_result(_fake_result(), source="simulated")
    devs = chat.build_context()["top_deviations_from_personal_baseline"]
    assert [d["feature"] for d in devs] == ["scl", "rmssd", "alpha"]   # |z| 由大到小
    assert all(d["z_score"] is not None for d in devs)                 # beta 的 None 被剔除


def test_context_does_not_leak_raw_input():
    """上下文只含算好的結果，不該出現模型的原始特徵輸入。"""
    device.store_result(_fake_result(), source="simulated")
    ctx = chat.build_context()
    assert not any(k in ctx for k in ("eeg", "hrv", "eda", "features", "raw"))


# --------------------------- _clean ---------------------------

def test_clean_filters_bad_roles_and_blanks():
    out = chat._clean([{"role": "system", "content": "x"},
                       {"role": "user", "content": "  "},
                       {"role": "user", "content": "hi"},
                       {"role": "assistant", "content": "yo"},
                       "not-a-dict"])
    assert out == [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]


def test_clean_truncates_long_messages():
    out = chat._clean([{"role": "user", "content": "字" * 5000}])
    assert len(out[0]["content"]) == chat.MAX_CHARS


def test_clean_keeps_only_last_n():
    msgs = [{"role": "user", "content": f"m{i}"} for i in range(30)]
    out = chat._clean(msgs)
    assert len(out) == chat.MAX_HISTORY
    assert out[-1]["content"] == "m29"


# --------------------------- reply ---------------------------

def test_reply_empty_messages():
    assert chat.reply([])["status"] == "unavailable"


def test_reply_without_key(monkeypatch):
    monkeypatch.setattr(chat, "_load_api_key", lambda: "")
    r = chat.reply([{"role": "user", "content": "你好"}])
    assert r["status"] == "unavailable"
    assert "DEEPSEEK_API_KEY" in r["reply"]


def test_reply_success(monkeypatch):
    monkeypatch.setattr(chat, "_load_api_key", lambda: "fake")
    monkeypatch.setattr(chat.requests, "post", lambda *a, **k: _FakeResp("  我覺得還不錯  "))
    r = chat.reply([{"role": "user", "content": "你好"}])
    assert r["status"] == "ok"
    assert r["reply"] == "我覺得還不錯"


def test_reply_api_error_is_graceful(monkeypatch):
    monkeypatch.setattr(chat, "_load_api_key", lambda: "fake")

    def boom(*a, **k):
        raise RuntimeError("timeout")

    monkeypatch.setattr(chat.requests, "post", boom)
    r = chat.reply([{"role": "user", "content": "你好"}])
    assert r["status"] == "unavailable"
    assert r["reply"]                                       # 仍有可顯示的訊息


def test_chat_endpoint(monkeypatch):
    from fastapi.testclient import TestClient
    from api.main import app
    monkeypatch.setattr(chat, "_load_api_key", lambda: "fake")
    monkeypatch.setattr(chat.requests, "post", lambda *a, **k: _FakeResp("ok 回覆"))
    c = TestClient(app)
    r = c.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert r.json()["reply"] == "ok 回覆"
