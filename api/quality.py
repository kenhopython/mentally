# -*- coding: utf-8 -*-
"""
api/quality.py — 訊號品質檢查

Input：derived_flat（含衍生特徵的短名 dict）。
Output：{"score": 0-1, "status": "GOOD"/"MODERATE"/"LOW", "issues": [...]}。
Processing：檢查缺失、不可能值（負值/超範圍）、訊號完整度。
連接方式：被 api/prediction.py 的 Step 2 呼叫。
"""
from api.features import BASELINE_SHORT

# 不得為負值的特徵
_NONNEG = ["delta", "theta", "alpha", "beta", "gamma", "rmssd", "sdnn", "pnn50",
           "lf", "hf", "total_power", "scl", "scr_amplitude", "scr_frequency",
           "tonic", "phasic", "eda_variability", "eda_latency"]


def check_signal_quality(derived_flat: dict) -> dict:
    provided = {k for k, v in derived_flat.items() if v is not None}
    completeness = len(provided & set(BASELINE_SHORT)) / len(BASELINE_SHORT)

    issues = []
    for f in _NONNEG:
        v = derived_flat.get(f)
        if v is not None and v < 0:
            issues.append(f"{f} 為負值（不可能）")

    hr = derived_flat.get("heart_rate")
    if hr is not None and not (20 <= hr <= 250):
        issues.append(f"heart_rate {hr} 超出合理範圍 (20–250)")
    pnn50 = derived_flat.get("pnn50")
    if pnn50 is not None and not (0 <= pnn50 <= 100):
        issues.append(f"pnn50 {pnn50} 超出 0–100")

    penalty = min(0.5, 0.1 * len(issues))
    score = max(0.0, completeness - penalty)

    status = "GOOD" if score >= 0.8 else ("MODERATE" if score >= 0.5 else "LOW")
    return {"score": round(score, 4), "status": status, "issues": issues}
