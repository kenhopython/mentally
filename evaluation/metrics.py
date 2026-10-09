# -*- coding: utf-8 -*-
"""
evaluation/metrics.py — 模型評估指標

Input：y_true、y_pred、y_proba、label_names。
Output：dict（accuracy/precision/recall/f1/roc_auc/confusion_matrix/classification_report）。
連接方式：被 models/* 訓練腳本與 model_comparison.py 呼叫。
"""
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)


def evaluate_model(y_true, y_pred, y_proba, label_names) -> dict:
    """回傳完整評估指標。"""
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, target_names=label_names, zero_division=0),
    }
    try:
        metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")), 4)
    except Exception:
        metrics["roc_auc"] = None
    return metrics


def top_features(feature_names, importances, n=10) -> list:
    """回傳 feature importance 前 n 名（依重要性降冪）。"""
    order = np.argsort(importances)[::-1]
    return [{"feature": feature_names[i], "importance": round(float(importances[i]), 6)}
            for i in order[:n]]
