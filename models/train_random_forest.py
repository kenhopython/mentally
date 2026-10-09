# -*- coding: utf-8 -*-
"""
models/train_random_forest.py — Random Forest classifier

Input：data dict。
Output：dict（name/model/metrics/feature_importances/path）。
要求：class imbalance handling（class_weight="balanced"）、predict_proba、
      feature_importance、模型存 trained_models/random_forest.pkl。
"""
import logging

from sklearn.ensemble import RandomForestClassifier

from evaluation import metrics as ev
from models import common
from preprocessing import config

log = logging.getLogger("models.random_forest")


def train(data: dict) -> dict:
    model = RandomForestClassifier(
        n_estimators=300, max_depth=20,
        class_weight="balanced", random_state=config.SEED, n_jobs=-1,
    )
    model.fit(data["X_train"], data["y_train"])

    y_pred = model.predict(data["X_val"])
    y_proba = model.predict_proba(data["X_val"])
    m = ev.evaluate_model(data["y_val"], y_pred, y_proba, config.LABEL_ORDER)

    importances = ev.top_features(
        data["preprocessor"]["feature_names"], model.feature_importances_,
    )

    path = common.save_model(model, "random_forest")
    return {
        "name": "random_forest", "model": model, "metrics": m,
        "feature_importances": importances, "path": path,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    r = train(common.get_data())
    print("accuracy:", r["metrics"]["accuracy"], "| f1_macro:", r["metrics"]["f1_macro"])
    print("Top 5 features:", [f["feature"] for f in r["feature_importances"][:5]])
