# -*- coding: utf-8 -*-
"""
models/build_simulator_anchors.py — 從訓練資料算出模擬裝置的錨點

為什麼需要這支程式：
  模擬裝置（api/simulator.py）原本用「憑感覺寫」的錨點（例如假設平靜 =
  心率 65 / SCL 2.0），結果那些數值落在訓練資料根本不存在的位置，
  模型只好全部猜「negative」—— 展示時情緒永遠是負面，一看就知道是壞的。
  錨點就該是「訓練資料裡各情緒類別的平均值」，這支程式把它算出來。

Output：api/simulator_anchors.json
  {
    "source": 資料檔名, "n": 各類別樣本數,
    "anchors": {"positive": {<特徵>: 平均}, ...},
    "spread":  {<特徵>: 合併標準差},
    "features": [<特徵名>...]        # 模擬裝置會產生的 19 個欄位
  }

執行：python -m models.build_simulator_anchors
"""
import glob
import json
import os

import pandas as pd

from preprocessing import config

# 訓練資料欄名 → 模擬裝置／ML 用的短名
COL2KEY = {
    "eeg_delta_power": "delta", "eeg_theta_power": "theta", "eeg_alpha_power": "alpha",
    "eeg_beta_power": "beta", "eeg_gamma_power": "gamma",
    "heart_rate_bpm": "hr", "hrv_rmssd_ms": "rmssd", "hrv_sdnn_ms": "sdnn",
    "hrv_pnn50_percent": "pnn50", "hrv_lf_ms2": "lf", "hrv_hf_ms2": "hf",
    "hrv_total_power_ms2": "total_power",
    "eda_scl_uS": "scl", "eda_scr_amplitude_uS": "scr_amplitude",
    "eda_scr_frequency_per_min": "scr_frequency", "eda_tonic_uS": "tonic",
    "eda_phasic_uS": "phasic", "eda_variability": "eda_variability",
    "eda_response_latency_s": "eda_latency",
}

LABELS = ["positive", "neutral", "negative"]
OUT_PATH = os.path.join(config.BASE_DIR, "api", "simulator_anchors.json")


def find_data() -> str:
    files = sorted(glob.glob(os.path.join(config.BASE_DIR, "data", "*.csv")))
    for f in files:
        if "training" in os.path.basename(f).lower() or "100K" in os.path.basename(f):
            return f
    if files:
        return files[0]
    raise FileNotFoundError("data/ 底下找不到訓練用的 CSV")


def main() -> None:
    path = find_data()
    df = pd.read_csv(path, usecols=list(COL2KEY) + ["emotion_label", "data_quality"])
    # 與訓練管線一致：只用品質良好的樣本
    df = df[df["data_quality"].astype(str).str.lower() == "good"]
    df = df[df["emotion_label"].isin(LABELS)]
    df = df.rename(columns=COL2KEY)
    keys = list(COL2KEY.values())

    means = df.groupby("emotion_label")[keys].mean()
    # 每類別各算標準差再取平均 → 一個共用的「散佈程度」，用來產生雜訊
    spread = df.groupby("emotion_label")[keys].std().mean()

    payload = {
        "source": os.path.basename(path),
        "n": {lab: int((df["emotion_label"] == lab).sum()) for lab in LABELS},
        "anchors": {lab: {k: round(float(means.loc[lab, k]), 4) for k in keys} for lab in LABELS},
        "spread": {k: round(float(spread[k]), 4) for k in keys},
        "features": keys,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"資料來源：{payload['source']}")
    print(f"樣本數：{payload['n']}")
    print(f"已寫入：{OUT_PATH}\n")
    for lab in LABELS:
        print(f"  {lab:9s} " + "  ".join(f"{k}={payload['anchors'][lab][k]:g}" for k in keys[:6]))


if __name__ == "__main__":
    main()
