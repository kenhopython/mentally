# -*- coding: utf-8 -*-
"""DeepSeek 客戶端測試（不真實呼叫 API）"""
import json

import pytest

from api import deepseek


class _FakeResp:
    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


GOOD_JSON = json.dumps({
    "summary": "s", "eeg_analysis": "e", "hrv_analysis": "h", "eda_analysis": "d",
    "multimodal_analysis": "m", "baseline_analysis": "b", "trend_analysis": "t",
    "possible_state": "p", "self_regulation": ["a", "b"], "limitations": "l",
}, ensure_ascii=False)


def test_parse_json_fenced():
    d = deepseek._parse_json("```json\n" + GOOD_JSON + "\n```")
    assert d["summary"] == "s"


def test_parse_json_plain():
    d = deepseek._parse_json(GOOD_JSON)
    assert d["eeg_analysis"] == "e"


def test_no_key_returns_unavailable(monkeypatch):
    monkeypatch.setattr(deepseek, "load_api_key", lambda: "")
    r = deepseek.explain({"prediction": {}})
    assert r["status"] == "unavailable"


def test_success_parses_json(monkeypatch):
    monkeypatch.setattr(deepseek, "load_api_key", lambda: "fake-key")
    monkeypatch.setattr(deepseek.requests, "post", lambda *a, **k: _FakeResp(GOOD_JSON))
    r = deepseek.explain({"prediction": {}})
    assert r["status"] == "ok"
    assert r["summary"] == "s"
    assert r["self_regulation"] == ["a", "b"]


def test_api_error_returns_unavailable(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("timeout")
    monkeypatch.setattr(deepseek, "load_api_key", lambda: "fake-key")
    monkeypatch.setattr(deepseek.requests, "post", boom)
    r = deepseek.explain({"prediction": {}})
    assert r["status"] == "unavailable"
    assert r["analysis"] is None
