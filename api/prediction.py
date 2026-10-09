# -*- coding: utf-8 -*-
"""
api/prediction.py — ML 預測管線（Step 1–13；DeepSeek 由 main.py 接 Step 14–15）

Input：payload（{user_id, eeg/hrv/eda}）。
Output：ML 結果 dict（不含 deepseek；含 baseline/trend/data_quality）。
流程：validate → feature engineering → data quality → feature vector（與訓練同順序）
      → load best model + scaler → predict_proba → baseline 比較 → trend。
連接方式：被 api/main.py 的 /predict、/predict/batch 呼叫。
"""
import json
import logging
import os
import time

import joblib
import numpy as np

from api import features, quality, trends
from api.features import compute_derived, normalize_input, SHORT_TO_CSV
from baseline import baseline as baseline_mod
from preprocessing import config

log = logging.getLogger("api.prediction")

TRAINED_DIR = os.path.join(config.BASE_DIR, "trained_models")
MODEL_VERSION = "1.0"

_cache = {}


def load_artifacts() -> dict:
    """載入 best model + preprocessor（快取）。"""
    if "model" in _cache:
        return _cache
    with open(os.path.join(TRAINED_DIR, "best_model.json"), encoding="utf-8") as f:
        best = json.load(f)
    model_name = best["best_model"]
    model = joblib.load(os.path.join(TRAINED_DIR, f"{model_name}.pkl"))
    preproc = joblib.load(os.path.join(TRAINED_DIR, "preprocessor.pkl"))
    _cache.update({"model": model, "preproc": preproc, "model_name": model_name})
    return _cache


def load_baselines() -> dict:
    path = os.path.join(TRAINED_DIR, "baselines.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _normalize_user(uid: str) -> str:
    """把 'U00001' 正規化成 '1'（對應 person_id）。"""
    uid = str(uid)
    if uid.upper().startswith("U") and uid[1:].isdigit():
        return str(int(uid[1:]))
    return uid


def _get_baseline(user_id: str) -> dict:
    baselines = load_baselines()
    return baselines.get(user_id) or baselines.get(_normalize_user(user_id)) or {}


def build_vector(derived: dict, feature_names: list) -> np.ndarray:
    """依訓練時的 feature order 建 29 維向量（缺者 NaN，由 imputer 補）。"""
    csv_values = {SHORT_TO_CSV[k]: v for k, v in derived.items()
                  if k in SHORT_TO_CSV and v is not None}
    vec = np.full(len(feature_names), np.nan, dtype=float)
    for i, col in enumerate(feature_names):
        if col in csv_values:
            vec[i] = float(csv_values[col])
    return vec.reshape(1, -1)


def _preprocess(vec: np.ndarray, preproc: dict) -> np.ndarray:
    X = preproc["imputer"].transform(vec)
    X = np.clip(X, preproc["clip_lo"], preproc["clip_hi"])
    return preproc["scaler"].transform(X)


def _group_signals(derived: dict) -> dict:
    return {
        "eeg": {k: derived.get(k) for k in features.EEG_SHORT + features.RELATIVE_SHORT
                if k in derived},
        "hrv": {k: derived.get(k) for k in features.HRV_SHORT if k in derived},
        "eda": {k: derived.get(k) for k in features.EDA_SHORT if k in derived},
    }


def run_ml(payload: dict) -> dict:
    # Step 1 validate + 攤平
    flat = normalize_input(payload)
    user_id = str(payload.get("user_id") or payload.get("person_id") or "unknown")

    # Step 2 feature engineering（補算 relative / ratio）
    derived = compute_derived(flat)

    # Step 3 data quality
    dq = quality.check_signal_quality(derived)

    # Step 4–6 feature vector（同訓練順序）＋載入模型
    art = load_artifacts()
    X = _preprocess(build_vector(derived, art["preproc"]["feature_names"]), art["preproc"])

    # Step 7–9 predict + proba
    proba = art["model"].predict_proba(X)[0]
    label_order = art["preproc"]["label_order"]
    pred_idx = int(np.argmax(proba))
    emotion = label_order[pred_idx]
    probabilities = {label_order[i]: round(float(p), 4) for i, p in enumerate(proba)}

    # Step 10–11 baseline 比較
    baseline = _get_baseline(user_id)
    baseline_status = ("READY" if baseline and
                       any(baseline_mod.status(v) == "READY" for v in baseline.values())
                       else "INSUFFICIENT_DATA")
    comparison = baseline_mod.compare(
        {k: derived.get(k) for k in features.BASELINE_SHORT}, baseline)

    # Step 12 trend
    trend = trends.get_emotion_trend(user_id)

    return {
        "user_id": user_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "data_quality": {"score": dq["score"], "status": dq["status"]},
        "model": {"name": art["model_name"], "version": MODEL_VERSION},
        "prediction": {
            "emotion": emotion,
            "confidence": round(float(confidence_of(proba, pred_idx)), 4),
            "probabilities": probabilities,
        },
        "signals": _group_signals(derived),
        "baseline": {"status": baseline_status, "comparison": comparison},
        "trend": trend,
    }


def confidence_of(proba, idx):
    return proba[idx]


def to_deepseek_input(result: dict) -> dict:
    """把 ML 結果轉成傳給 DeepSeek 的 JSON（只含已算好的數值）。

    同一份 payload 有兩個用途：
      1. fusion.llm_vote()  → 請 DeepSeek 投第二票（此時尚無投票結果）
      2. deepseek.explain() → 請 DeepSeek 寫解釋（此時已含投票結果）
    """
    cmp = result["baseline"]["comparison"]
    pred = result["prediction"]

    def z(f):
        v = cmp.get(f) or {}
        return v.get("z_score")

    payload = {
        "prediction": pred,
        "probabilities": pred["probabilities"],
        "eeg": {"alpha_zscore": z("alpha"), "beta_zscore": z("beta"),
                "theta_zscore": z("theta")},
        "hrv": {"rmssd_zscore": z("rmssd"), "sdnn_zscore": z("sdnn")},
        "eda": {"scl_zscore": z("scl"), "scr_frequency_zscore": z("scr_frequency")},
        "baseline_status": result["baseline"]["status"],
        "data_quality": result["data_quality"]["score"],
        "trend": result["trend"],
    }

    # 融合完成後才有的欄位：讓解釋層知道「雙模型是否一致」
    if "consensus" in pred:
        payload["model_consensus"] = {
            "ml_emotion": pred.get("ml_emotion"),
            "llm_emotion": pred.get("llm_emotion"),
            "model_agreement": pred.get("model_agreement"),
            "consensus": pred.get("consensus"),
            "final_confidence": pred.get("confidence"),
            "ml_confidence": pred.get("confidence_ml"),
        }
    return payload
