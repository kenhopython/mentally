# -*- coding: utf-8 -*-
"""
models/train_svm.py — SVM classifier（linear / rbf）

Input：data dict、kernel（"linear" / "rbf"）。
Output：dict（name/model/metrics/path）。
要求：SVC(probability=True)、StandardScaler（已在 preprocessing 完成）、predict_proba。
注意：SVC 對大樣本較慢；大資料集建議抽樣（見 model_comparison.py 的 subsample）。
"""
import logging

from sklearn.svm import SVC

from evaluation import metrics as ev
from models import common
from preprocessing import config

log = logging.getLogger("models.svm")


def train(data: dict, kernel: str = "rbf") -> dict:
    model = SVC(
        kernel=kernel, probability=True, class_weight="balanced",
        random_state=config.SEED,
    )
    model.fit(data["X_train"], data["y_train"])

    y_pred = model.predict(data["X_val"])
    y_proba = model.predict_proba(data["X_val"])
    m = ev.evaluate_model(data["y_val"], y_pred, y_proba, config.LABEL_ORDER)

    name = f"svm_{kernel}"
    path = common.save_model(model, name)
    return {"name": name, "model": model, "metrics": m, "path": path}


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    kernel = sys.argv[1] if len(sys.argv) > 1 else "rbf"
    r = train(common.get_data(), kernel=kernel)
    print(kernel, "| accuracy:", r["metrics"]["accuracy"], "| f1_macro:", r["metrics"]["f1_macro"])
