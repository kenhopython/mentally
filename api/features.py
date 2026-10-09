# -*- coding: utf-8 -*-
"""
api/features.py — 特徵名稱對應與衍生特徵計算（單一事實來源）

短名（API 輸入）→ CSV 欄位（模型訓練用），以及由頻帶功率補算 relative / ratio。
"""
# 基線與輸入用（24 個：EEG 8 + HRV 9 + EDA 7）
EEG_SHORT = ["delta", "theta", "alpha", "beta", "gamma",
             "theta_beta_ratio", "alpha_beta_ratio", "alpha_theta_ratio"]
HRV_SHORT = ["heart_rate", "mean_rr", "rmssd", "sdnn", "pnn50",
             "lf", "hf", "lf_hf_ratio", "total_power"]
EDA_SHORT = ["scl", "scr_amplitude", "scr_frequency", "tonic", "phasic",
             "eda_variability", "eda_latency"]
BASELINE_SHORT = EEG_SHORT + HRV_SHORT + EDA_SHORT   # 24

# 衍生 relative（模型有、但輸入通常不給，由頻帶功率補算）
RELATIVE_SHORT = ["relative_delta", "relative_theta", "relative_alpha",
                  "relative_beta", "relative_gamma"]

# 短名 → CSV 欄位（24 + 5 = 29 個模型特徵）
SHORT_TO_CSV = {
    # EEG 頻帶
    "delta": "eeg_delta_power", "theta": "eeg_theta_power", "alpha": "eeg_alpha_power",
    "beta": "eeg_beta_power", "gamma": "eeg_gamma_power",
    "relative_delta": "eeg_relative_delta", "relative_theta": "eeg_relative_theta",
    "relative_alpha": "eeg_relative_alpha", "relative_beta": "eeg_relative_beta",
    "relative_gamma": "eeg_relative_gamma",
    "theta_beta_ratio": "theta_beta_ratio", "alpha_beta_ratio": "alpha_beta_ratio",
    "alpha_theta_ratio": "alpha_theta_ratio",
    # HRV
    "heart_rate": "heart_rate_bpm", "mean_rr": "mean_rr_ms", "rmssd": "hrv_rmssd_ms",
    "sdnn": "hrv_sdnn_ms", "pnn50": "hrv_pnn50_percent", "lf": "hrv_lf_ms2",
    "hf": "hrv_hf_ms2", "lf_hf_ratio": "hrv_lf_hf_ratio", "total_power": "hrv_total_power_ms2",
    # EDA
    "scl": "eda_scl_uS", "scr_amplitude": "eda_scr_amplitude_uS",
    "scr_frequency": "eda_scr_frequency_per_min", "tonic": "eda_tonic_uS",
    "phasic": "eda_phasic_uS", "eda_variability": "eda_variability",
    "eda_latency": "eda_response_latency_s",
}

BANDS = ["delta", "theta", "alpha", "beta", "gamma"]


def normalize_input(payload: dict) -> dict:
    """把 {eeg/hrv/eda} 巢狀輸入攤平成 {短名: value}。"""
    flat = {}
    for key, value in payload.items():
        if isinstance(value, dict) and key in ("eeg", "hrv", "eda"):
            for k, v in value.items():
                # 支援 lf_hf / lf_hf_ratio 兩種寫法
                flat[k if k != "lf_hf" else "lf_hf_ratio"] = v
        elif key not in ("eeg", "hrv", "eda", "user_id", "person_id", "data_quality",
                         "data_quality_score", "baseline", "timestamp"):
            flat[key] = value
    return flat


def compute_derived(flat: dict) -> dict:
    """由頻帶功率補算 relative_* 與 ratio（若輸入未提供）。"""
    out = dict(flat)
    if all(out.get(b) is not None for b in BANDS):
        total = sum(out[b] for b in BANDS)
        if total > 0:
            for b in BANDS:
                out.setdefault(f"relative_{b}", out[b] / total)
        if out.get("theta") and out.get("beta"):
            out.setdefault("theta_beta_ratio", out["theta"] / out["beta"])
        if out.get("alpha") and out.get("beta"):
            out.setdefault("alpha_beta_ratio", out["alpha"] / out["beta"])
        if out.get("alpha") and out.get("theta"):
            out.setdefault("alpha_theta_ratio", out["alpha"] / out["theta"])
    return out
