# -*- coding: utf-8 -*-
"""
preprocessing/clean.py — 資料清理

Input：CSV 路徑。
Output：清理後的 DataFrame（特徵 + 標籤 + person_id + 品質欄位）。
Processing：
  1. 載入 CSV
  2. 丟棄無用/敏感/會洩漏標籤的欄位（見 config.DROP_COLUMNS）
  3. 移除缺失標籤的列
  4. 依 data quality 過濾（預設只保留 "good"）
Error Handling：檔案不存在/欄位缺失 → 明確拋出並記錄。
連接方式：被 feature_engineering.run_pipeline() 呼叫。
"""
import logging

import pandas as pd

from preprocessing import config

log = logging.getLogger("preprocessing.clean")


def load(csv_path: str) -> pd.DataFrame:
    """載入 CSV（或 xlsx，依副檔名）。"""
    if csv_path.endswith(".csv"):
        df = pd.read_csv(csv_path)
    elif csv_path.endswith(".xlsx"):
        df = pd.read_excel(csv_path, sheet_name="training_sample")
    else:
        raise ValueError(f"不支援的資料格式：{csv_path}")
    log.info("載入資料：%d 列 × %d 欄", len(df), len(df.columns))
    return df


def drop_unused(df: pd.DataFrame) -> pd.DataFrame:
    """丟棄非模型特徵 / 敏感 / 會洩漏標籤的欄位。"""
    cols = [c for c in config.DROP_COLUMNS if c in df.columns]
    dropped = [c for c in config.DROP_COLUMNS if c not in df.columns]
    if dropped:
        log.warning("以下欄位不存在於資料中（略過）：%s", dropped)
    log.info("丟棄 %d 個欄位：%s", len(cols), cols)
    return df.drop(columns=cols)


def drop_missing_target(df: pd.DataFrame) -> pd.DataFrame:
    """移除缺失標籤的列。"""
    before = len(df)
    df = df.dropna(subset=[config.TARGET_COLUMN])
    log.info("移除缺失標籤：%d 列", before - len(df))
    return df


def filter_by_quality(df: pd.DataFrame, keep: str = "good") -> pd.DataFrame:
    """依 data_quality 過濾（預設只保留 good）。"""
    if config.QUALITY_COLUMN not in df.columns:
        return df
    before = len(df)
    df = df[df[config.QUALITY_COLUMN] == keep]
    log.info("品質過濾（保留 %s）：移除 %d 列", keep, before - len(df))
    return df


def clean(csv_path: str) -> pd.DataFrame:
    """完整清理流程。"""
    df = load(csv_path)
    df = drop_unused(df)
    df = drop_missing_target(df)
    df = filter_by_quality(df)
    # 數值型別確保（避免字串混入）
    for c in config.FEATURE_COLUMNS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.reset_index(drop=True)


def report_missing(df: pd.DataFrame) -> pd.Series:
    """回報各欄缺失值數量。"""
    return df.isnull().sum()[df.isnull().sum() > 0]
