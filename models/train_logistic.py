# -*- coding: utf-8 -*-
"""
models/train_logistic.py — Logistic Regression baseline

Input：data dict（X/y train/val/test）。
Output：dict（name/model/metrics/path），模型存 trained_models/logistic_regression.pkl。
要求：StandardScaler（已在 preprocessing 完成）、class_weight="balanced"、predict_proba。
"""
import logging

from sklearn.linear_model import LogisticRegression

from evaluation import metrics as ev
from models import common
from preprocessing import config

log = logging.getLogger("models.logistic")


def train(data: dict) -> dict:
    model = LogisticRegression(
        class_weight="balanced", max_iter=3000,
        random_state=config.SEED, n_jobs=-1,
    )
    model.fit(data["X_train"], data["y_train"])

    y_pred = model.predict(data["X_val"])
    y_proba = model.predict_proba(data["X_val"])
    m = ev.evaluate_model(data["y_val"], y_pred, y_proba, config.LABEL_ORDER)

    path = common.save_model(model, "logistic_regression")
    return {"name": "logistic_regression", "model": model, "metrics": m, "path": path}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    r = train(common.get_data())
    print("accuracy:", r["metrics"]["accuracy"], "| f1_macro:", r["metrics"]["f1_macro"])
