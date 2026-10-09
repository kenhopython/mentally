# -*- coding: utf-8 -*-
"""
baseline/baseline.py — 個人基線（per-user baseline）

Input：歷史特徵值、目前特徵、baseline dict。
Output：compare → {feature: {raw, baseline_mean, z_score, category}}。
- 每個 user_id 建立自己的基線（不用全體平均）。
- 樣本不足 → status = INSUFFICIENT_DATA，不強行建立。
- 分類：within_baseline / slightly_above / slightly_below / significantly_above / significantly_below。
連接方式：被 api/prediction.py 與 api/main.py 呼叫。
"""
import json
import math
import os

import numpy as np
import pandas as pd

from api.features import BASELINE_SHORT, SHORT_TO_CSV

EPS = 1e-6
MIN_SAMPLES = 10   # 建立基線所需最少歷史樣本數


def compute_baseline(values: list) -> dict:
    """由歷史值算 mean/std/median/min/max/n（忽略 None/NaN），並附 sum/sumsq 供線上更新。"""
    v = np.asarray([float(x) for x in values if x is not None], dtype=float)
    v = v[~np.isnan(v)]
    n = len(v)
    if n == 0:
        return {"mean": None, "std": None, "median": None, "min": None, "max": None,
                "n": 0, "sum": 0.0, "sumsq": 0.0}
    s = float(v.sum())
    return {
        "mean": s / n, "std": float(v.std(ddof=0)), "median": float(np.median(v)),
        "min": float(v.min()), "max": float(v.max()), "n": n,
        "sum": s, "sumsq": float((v ** 2).sum()),
    }


def status(baseline_feature: dict) -> str:
    """基線是否可用。"""
    n = (baseline_feature or {}).get("n", 0)
    return "READY" if n >= MIN_SAMPLES else "INSUFFICIENT_DATA"


def zscore(value, mean, std):
    if value is None or mean is None or std is None:
        return None
    if math.isnan(float(std)) or std < EPS:
        return None
    return float((value - mean) / std)


def categorize(z) -> str:
    """z-score → within / slightly_* / significantly_*。"""
    if z is None or (isinstance(z, float) and math.isnan(z)):
        return "insufficient"
    if z >= 2:
        return "significantly_above"
    if z >= 1:
        return "slightly_above"
    if z > -1:
        return "within_baseline"
    if z > -2:
        return "slightly_below"
    return "significantly_below"


def compare(current: dict, baseline: dict) -> dict:
    """目前特徵 vs 個人基線 → 每特徵 z-score + 分類。"""
    out = {}
    for feature, value in current.items():
        b = baseline.get(feature) or {}
        z = zscore(value, b.get("mean"), b.get("std"))
        out[feature] = {
            "raw": value,
            "baseline_mean": b.get("mean"),
            "z_score": round(z, 3) if z is not None else None,
            "category": categorize(z),
        }
    return out


def update_feature(bf: dict, new_values: list) -> dict:
    """把新有效樣本合併進基線（線上更新 mean/std/min/max）。"""
    new = np.asarray([float(x) for x in new_values if x is not None], dtype=float)
    new = new[~np.isnan(new)]
    if len(new) == 0:
        return bf
    bf = bf or {"mean": None, "std": None, "median": None, "min": None, "max": None,
                "n": 0, "sum": 0.0, "sumsq": 0.0}
    s = bf.get("sum", 0.0) + float(new.sum())
    ss = bf.get("sumsq", 0.0) + float((new ** 2).sum())
    n = bf.get("n", 0) + len(new)
    mean = s / n
    var = ss / n - mean ** 2
    return {
        "mean": mean, "std": math.sqrt(var) if var > 0 else 0.0,
        "median": mean,   # 無原始值時以 mean 近似 median
        "min": min(bf.get("min", float("inf")), float(new.min())),
        "max": max(bf.get("max", float("-inf")), float(new.max())),
        "n": n, "sum": s, "sumsq": ss,
    }


def build_person_baselines(csv_path: str, out_path: str) -> dict:
    """由訓練 CSV 建立 per-person 基線（24 特徵），存成 JSON。"""
    df = pd.read_csv(csv_path)
    baselines = {}
    for pid, grp in df.groupby("person_id"):
        b = {}
        for short in BASELINE_SHORT:
            col = SHORT_TO_CSV.get(short)
            if col and col in df.columns:
                b[short] = compute_baseline(grp[col].tolist())
        baselines[str(pid)] = b

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(baselines, f, ensure_ascii=False, indent=2)
    return baselines
