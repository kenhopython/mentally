# -*- coding: utf-8 -*-
"""
preprocessing/feature_engineering.py — 特徵工程 + 以 person_id 切分

Input：clean() 後的 DataFrame。
Output：train/val/test 的 (X, y)、fitted preprocessor（imputer + 剪裁 + scaler）。
Processing（防 data leakage：所有統計量只在 train 上 fit）：
  1. 以 person_id 為單位切分 80/10/10（同一個人絕不跨 split）。
  2. 標籤編碼（positive/neutral/negative → 0/1/2）。
  3. 缺失值中位數插補（fit on train）。
  4. 極端值 winsorize（0.5%/99.5% 分位數剪裁，fit on train）。
  5. StandardScaler（fit on train）。
  6. 儲存 scaler + feature_config.json。
連接方式：被訓練腳本（models/train_*.py）呼叫。
"""
import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from preprocessing import clean, config

log = logging.getLogger("preprocessing.feature_engineering")

OUT_DIR = os.path.join(config.BASE_DIR, "trained_models")


def split_by_person(df: pd.DataFrame, seed: int = config.SEED):
    """以 person_id 為單位切分 train/val/test（80/10/10），杜絕同一人跨 split。"""
    persons = df[config.PERSON_ID_COLUMN].unique()
    rng = np.random.default_rng(seed)
    persons = rng.permutation(persons)

    n = len(persons)
    n_val = int(round(n * config.VAL_FRAC))
    n_test = int(round(n * config.TEST_FRAC))
    n_train = n - n_val - n_test

    val_ids = set(persons[:n_val])
    test_ids = set(persons[n_val:n_val + n_test])
    train_ids = set(persons[n_val + n_test:])

    train = df[df[config.PERSON_ID_COLUMN].isin(train_ids)].copy()
    val = df[df[config.PERSON_ID_COLUMN].isin(val_ids)].copy()
    test = df[df[config.PERSON_ID_COLUMN].isin(test_ids)].copy()

    log.info("person 切分：train=%d人 / val=%d人 / test=%d人", n_train, n_val, n_test)
    # 防洩漏驗證
    assert set(train[config.PERSON_ID_COLUMN]).isdisjoint(val_ids)
    assert set(train[config.PERSON_ID_COLUMN]).isdisjoint(test_ids)
    assert set(val[config.PERSON_ID_COLUMN]).isdisjoint(test_ids)
    return train, val, test


def encode_target(y: pd.Series) -> np.ndarray:
    """positive/neutral/negative → 0/1/2。"""
    mapping = {label: i for i, label in enumerate(config.LABEL_ORDER)}
    return y.map(mapping).to_numpy(dtype=int)


def build_preprocessor(train_df: pd.DataFrame) -> dict:
    """在 train 上 fit imputer + 剪裁邊界 + scaler。"""
    X_train = train_df[config.FEATURE_COLUMNS].to_numpy(dtype=float)

    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train)
    X_imp = imputer.transform(X_train)

    # winsorize：0.5% / 99.5% 分位數剪裁
    lo = np.percentile(X_imp, 0.5, axis=0)
    hi = np.percentile(X_imp, 99.5, axis=0)
    X_clip = np.clip(X_imp, lo, hi)

    scaler = StandardScaler()
    scaler.fit(X_clip)

    return {
        "imputer": imputer,
        "clip_lo": lo,
        "clip_hi": hi,
        "scaler": scaler,
        "feature_names": list(config.FEATURE_COLUMNS),
        "label_order": list(config.LABEL_ORDER),
    }


def transform(df: pd.DataFrame, preproc: dict) -> np.ndarray:
    """套用 preprocessor（impute → clip → scale）。"""
    X = df[config.FEATURE_COLUMNS].to_numpy(dtype=float)
    X = preproc["imputer"].transform(X)
    X = np.clip(X, preproc["clip_lo"], preproc["clip_hi"])
    return preproc["scaler"].transform(X)


def save_preprocessor(preproc: dict, out_dir: str = OUT_DIR) -> None:
    """儲存 scaler.pkl、preprocessor.pkl 與 feature_config.json（避免部署後特徵順序錯誤）。"""
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(preproc["scaler"], os.path.join(out_dir, "scaler.pkl"))
    # 完整 preprocessor（imputer + 剪裁邊界 + scaler），供 API 對新資料做一致前處理
    joblib.dump(preproc, os.path.join(out_dir, "preprocessor.pkl"))

    cfg = {
        "feature_names": preproc["feature_names"],
        "feature_order": list(range(len(preproc["feature_names"]))),
        "label_order": preproc["label_order"],
        "preprocessing": ["median_impute", "winsorize_0.5_99.5", "standard_scale"],
        "n_features": len(preproc["feature_names"]),
        "version": "1.0",
    }
    with open(os.path.join(out_dir, "feature_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    log.info("已儲存 scaler.pkl + feature_config.json 至 %s", out_dir)


def run_pipeline(csv_path: str) -> dict:
    """完整 preprocessing 管線入口。"""
    df = clean.clean(csv_path)

    train, val, test = split_by_person(df)
    preproc = build_preprocessor(train)
    save_preprocessor(preproc)

    return {
        "X_train": transform(train, preproc),
        "X_val": transform(val, preproc),
        "X_test": transform(test, preproc),
        "y_train": encode_target(train[config.TARGET_COLUMN]),
        "y_val": encode_target(val[config.TARGET_COLUMN]),
        "y_test": encode_target(test[config.TARGET_COLUMN]),
        "person_train": train[config.PERSON_ID_COLUMN].to_numpy(),
        "person_test": test[config.PERSON_ID_COLUMN].to_numpy(),
        "preprocessor": preproc,
        "train_df": train, "val_df": val, "test_df": test,
    }
