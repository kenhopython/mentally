# -*- coding: utf-8 -*-
"""Excel 批量預測測試：欄位映射、標籤欄忽略、端點行為。

端點現在每一列都走「ML + DeepSeek 投票 + 解釋」的完整管線（與即時接收相同），
所以端點測試一律把 DeepSeek 換成假實作，避免真的打 API、也避免測試被網路左右。
"""
import io

import pytest
from fastapi.testclient import TestClient

from api import deepseek, excel_io, main


# --------------------------- 欄位映射 ---------------------------

def test_map_column_accepts_short_and_csv_names():
    assert excel_io.map_column("alpha") == "alpha"
    assert excel_io.map_column("eeg_alpha_power") == "alpha"
    assert excel_io.map_column("heart_rate") == "heart_rate"
    assert excel_io.map_column("heart_rate_bpm") == "heart_rate"
    assert excel_io.map_column("lf_hf") == "lf_hf_ratio"
    assert excel_io.map_column("不存在的欄位") is None


def test_labels_and_metadata_are_ignored():
    for col in ("emotion_label", "physiological_pattern", "clinical_status",
                "dataset_split", "data_quality", "eeg_alpha_zscore", "age", "sex"):
        assert excel_io.is_ignored(col), f"{col} 應該被忽略（不是模型輸入）"
    assert not excel_io.is_ignored("alpha")
    assert not excel_io.is_ignored("heart_rate")


def test_read_rows_ignores_label_and_normalizes_id():
    csv = (
        "sample_id,person_id,delta,theta,alpha,beta,gamma,heart_rate,emotion_label\n"
        "1,1,0.10,0.18,0.42,0.14,0.06,65,positive\n"
        "2,1,0.08,0.26,0.20,0.42,0.12,105,negative\n"
    )
    parsed = excel_io.read_rows(io.BytesIO(csv.encode("utf-8")), "test.csv")

    assert len(parsed["rows"]) == 2
    assert "emotion_label" in parsed["ignored"]           # 標籤欄被忽略
    assert "emotion_label" not in parsed["mapped"]
    assert parsed["rows"][0]["sample_id"] == "1"          # 整數不帶 .0
    assert parsed["rows"][0]["features"]["heart_rate"] == 65
    assert "emotion_label" not in parsed["rows"][0]["features"]


def test_read_rows_skips_empty_rows():
    csv = "delta,theta,alpha,beta,gamma\n0.1,0.2,0.3,0.4,0.5\n,,,,\n"
    parsed = excel_io.read_rows(io.BytesIO(csv.encode("utf-8")), "test.csv")
    assert len(parsed["rows"]) == 1


# --------------------------- 端點 ---------------------------

@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "SIM_AUTOSTART", False)
    main._stop_simulator()
    # 預設讓 DeepSeek 拿不到 key → 棄權，端點測試不會真的打 API
    monkeypatch.setattr(deepseek, "load_api_key", lambda: "")
    with TestClient(main.app) as c:
        yield c
    main._stop_simulator()


# 29 個特徵的一筆「品質 GOOD」的列（取自真實測試集）。只有 8 個特徵時
# 品質檢查會判 LOW，llm_vote 直接棄權，根本走不到 DeepSeek —— 所以測試
# 要用完整特徵，確定會真的跑到投票 + 解釋那一段。
_GOOD_FEATURES = {
    "delta": 0.358972, "theta": 0.248369, "alpha": 0.22702, "beta": 0.425499,
    "gamma": 0.097854, "relative_delta": 0.264394, "relative_theta": 0.182931,
    "relative_alpha": 0.167208, "relative_beta": 0.313394, "relative_gamma": 0.072073,
    "theta_beta_ratio": 0.583711, "alpha_beta_ratio": 0.533538, "alpha_theta_ratio": 0.914045,
    "heart_rate": 68.091, "mean_rr": 881.18, "rmssd": 54.598, "sdnn": 37.528,
    "pnn50": 71.548, "lf": 1096.473, "hf": 211.19, "lf_hf_ratio": 5.1919,
    "total_power": 1509.649, "scl": 3.40754, "scr_amplitude": 0.50248,
    "scr_frequency": 1.69817, "tonic": 2.4074, "phasic": 1.13016,
    "eda_variability": 0.76256, "eda_latency": 2.4183,
}
_FEATURE_COLS = list(_GOOD_FEATURES)


def _good_csv(n=3):
    cols = ["sample_id", "person_id"] + _FEATURE_COLS
    lines = [",".join(cols)]
    for i in range(n):
        vals = [str(i + 1), "1"] + [str(_GOOD_FEATURES[c]) for c in _FEATURE_COLS]
        lines.append(",".join(vals))
    return "\n".join(lines).encode("utf-8")


def test_predict_excel_runs_full_pipeline_for_every_row(client, monkeypatch):
    """每一列都走完整管線（DeepSeek 投票 + 解釋），不是只有 ML。"""
    calls = {"vote": 0, "explain": 0}

    def fake_vote(deepseek_input, respect_throttle=True):
        calls["vote"] += 1
        assert respect_throttle is False, "批次呼叫不該被 3 秒節流擋掉"
        return {"status": "ok", "label": "negative", "llm_confidence": 0.7, "reason": "test"}

    def fake_explain(deepseek_input):
        calls["explain"] += 1
        return {"status": "ok", "summary": "測試用解釋"}

    monkeypatch.setattr(deepseek, "vote", fake_vote)
    monkeypatch.setattr(deepseek, "explain", fake_explain)

    r = client.post("/predict/excel",
                    files={"file": ("test.csv", _good_csv(3), "text/csv")})
    assert r.status_code == 200, r.text
    d = r.json()

    assert d["count"] == 3
    assert d["summary"][0]["sample_id"] == "1"

    # 三列全部有 AI 結果（不是 deferred），且 DeepSeek 投票 + 解釋各被呼叫 3 次
    assert calls["vote"] == 3
    assert calls["explain"] == 3
    for res in d["results"]:
        assert res["deepseek"]["status"] == "ok"
        assert res["llm_vote"]["status"] == "ok"
        assert res["prediction"]["llm_emotion"] == "negative"

    # 每一列 summary 都帶 AI 的票與一致性
    assert all(s["llm_emotion"] == "negative" for s in d["summary"])
    assert d["model_agreement"]["agree"] + d["model_agreement"]["disagree"] <= 3

    # 原始數據預覽：每列結果帶原始特徵（JSON 可序列化的原生數值），分組齊全
    assert [g["group"] for g in d["feature_groups"]] == ["EEG", "HRV", "EDA"]
    feats = d["results"][0]["_excel"]["features"]
    assert "heart_rate" in feats and isinstance(feats["heart_rate"], (int, float))


def test_predict_excel_without_api_key_abstains_cleanly(client):
    """沒有 API key 時，每一列都棄權、標籤仍以 ML 為準（不是 deferred 偷懶）。"""
    r = client.post("/predict/excel",
                    files={"file": ("test.csv", _good_csv(2), "text/csv")})
    assert r.status_code == 200, r.text
    d = r.json()

    for res in d["results"]:
        assert res["llm_vote"]["status"] == "no_api_key"
        assert res["deepseek"]["status"] == "unavailable"
        p = res["prediction"]
        assert p["consensus"] == "ml_only"
        assert p["llm_emotion"] is None
        assert p["emotion"] in ("positive", "neutral", "negative")


def test_predict_excel_rejects_wrong_extension(client):
    r = client.post("/predict/excel",
                    files={"file": ("test.txt", b"hello", "text/plain")})
    assert r.status_code == 422


def test_predict_excel_rejects_no_features(client):
    csv = "sample_id,name\n1,甲\n2,乙\n".encode("utf-8")
    r = client.post("/predict/excel",
                    files={"file": ("test.csv", csv, "text/csv")})
    assert r.status_code == 422
