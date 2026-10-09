# -*- coding: utf-8 -*-
"""
api/excel_io.py — 讀取「預測用 Excel / CSV」

用途：把使用者上傳的 Excel 每一列，轉成 ML 能吃的 {短名特徵: 值} dict。
     —— 供準確率驗證：Excel 裡只有生理特徵，沒有 emotion_label（心情標籤）。

欄位辨識規則：
  1. 認識的欄位（特徵）→ 轉成短名（支援「短名」與「訓練 CSV 長名」兩種寫法）。
  2. 標籤／中繼資料（emotion_label、physiological_pattern、clinical_status、
     *_zscore、age、sex…）→ 一律忽略，**絕不當特徵**。
  3. 不認識的欄位 → 回報給前端，讓使用者知道哪幾欄被跳過。
連接方式：被 api/main.py 的 POST /predict/excel 呼叫。
"""
import logging

import pandas as pd

from api.features import SHORT_TO_CSV

log = logging.getLogger("api.excel_io")

# CSV 長名 → 短名（反向對照；讓使用者可直接上傳訓練 CSV，無需改名）
CSV_TO_SHORT = {csv: short for short, csv in SHORT_TO_CSV.items()}

# 別名（裝置/不同來源可能用不同拼法）
ALIASES = {"lf_hf": "lf_hf_ratio"}

# 標籤／中繼資料／診斷衍生欄 —— 這些「不是模型輸入」，看到一律跳過
IGNORED_COLUMNS = {
    "sample_id", "person_id", "timestamp", "age", "sex", "sleep_hours",
    "exercise_level", "caffeine_level", "measurement_hour",
    "emotion_label", "physiological_pattern", "pattern_note", "clinical_status",
    "synthetic", "dataset_split", "data_quality", "data_quality_score",
}


def map_column(col: str):
    """欄位名 → 短名特徵；不是特徵（也不屬於忽略清單）回 None。"""
    c = str(col).strip()
    if c in SHORT_TO_CSV:
        return c
    if c in CSV_TO_SHORT:
        return CSV_TO_SHORT[c]
    if c in ALIASES:
        return ALIASES[c]
    return None


def is_ignored(col: str) -> bool:
    c = str(col).strip()
    if c in IGNORED_COLUMNS:
        return True
    if c.endswith("_zscore"):      # eeg_alpha_zscore 等是診斷欄，不是輸入
        return True
    return False


def _id_str(v):
    """把 sample_id/person_id 轉成乾淨的字串。

    pandas 的 iterrows 會把整數欄 upcast 成 float（例如 69753 → 69753.0），
    這裡把「整數值」還原成不帶 .0 的字串，避免 join 時 69753 對不上 69753.0。
    """
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return str(int(v)) if float(v).is_integer() else str(v)
    return str(v).strip()


def read_rows(file_like, filename: str) -> dict:
    """讀取上傳的 Excel/CSV，回傳：
      {"rows": [{"features": {short: val}, "sample_id": ..., "person_id": ...}],
       "mapped": [...], "ignored": [...], "unknown": [...]}
    """
    if str(filename).lower().endswith(".csv"):
        df = pd.read_csv(file_like)
    else:
        df = pd.read_excel(file_like)

    mapped = {}      # 原始欄名 → 短名
    ignored = []
    unknown = []
    for col in df.columns:
        m = map_column(col)
        if m:
            mapped[col] = m
        elif is_ignored(col):
            ignored.append(str(col).strip())
        else:
            unknown.append(str(col).strip())

    has_sid = "sample_id" in df.columns
    has_pid = "person_id" in df.columns

    rows = []
    for _, row in df.iterrows():
        feats = {}
        for col, short in mapped.items():
            v = row[col]
            if v is None or (isinstance(v, float) and pd.isna(v)):
                continue
            feats[short] = v
        if not feats:                      # 整列都是空的 → 跳過
            continue
        rows.append({
            "features": feats,
            "sample_id": _id_str(row.get("sample_id")) if has_sid else None,
            "person_id": _id_str(row.get("person_id")) if has_pid else None,
        })

    return {"rows": rows, "mapped": [str(c).strip() for c in mapped],
            "ignored": ignored, "unknown": unknown}
