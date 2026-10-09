# -*- coding: utf-8 -*-
"""
models/common.py — 訓練腳本共用：載入 preprocessed 資料 + 儲存模型

Input：無（內部呼叫 preprocessing.run_pipeline）。
Output：data dict（X/y train/val/test + preprocessor）。
連接方式：被 train_*.py 與 model_comparison.py 呼叫。
"""
import logging
import os

import joblib

from preprocessing import config, feature_engineering

log = logging.getLogger("models.common")

_CACHE = None


def get_data() -> dict:
    """執行（或快取）preprocessing 管線，回傳切分後的資料。"""
    global _CACHE
    if _CACHE is None:
        csv_path = config.DATA_1M_CSV if os.path.exists(config.DATA_1M_CSV) else config.DATA_100K_CSV
        log.info("使用資料：%s", os.path.basename(csv_path))
        _CACHE = feature_engineering.run_pipeline(csv_path)
    return _CACHE


def save_model(model, name: str) -> str:
    """儲存模型到 trained_models/<name>.pkl。"""
    path = os.path.join(config.BASE_DIR, "trained_models", f"{name}.pkl")
    joblib.dump(model, path)
    log.info("已儲存模型：%s", path)
    return path
