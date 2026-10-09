# -*- coding: utf-8 -*-
"""
models/model_comparison.py — 訓練並比較三個模型，選出最佳作為 production classifier

Input：無（可傳 sample=N 對 train 抽樣加速測試）。
Output：model_comparison.csv + best_model.json。
流程：LR / RF / SVM(linear) / SVM(rbf) / MLP(自建) → 在 validation set 比較 → 選最佳。
"""
import csv
import json
import logging
import os

import numpy as np

from models import (common, train_logistic, train_mlp, train_random_forest,
                    train_svm)
from preprocessing import config

log = logging.getLogger("models.comparison")

OUT_DIR = os.path.join(config.BASE_DIR, "trained_models")


def subsample_train(data: dict, n: int) -> dict:
    """對 train 抽樣（均勻），加速測試；val/test 不動。"""
    idx = np.random.default_rng(config.SEED).choice(len(data["X_train"]), n, replace=False)
    data = dict(data)
    data["X_train"] = data["X_train"][idx]
    data["y_train"] = data["y_train"][idx]
    log.info("train 抽樣至 %d 列（測試用）", n)
    return data


def run(sample: int | None = None) -> dict:
    data = common.get_data()
    if sample:
        data = subsample_train(data, sample)

    log.info("訓練 Logistic Regression …")
    lr = train_logistic.train(data)
    log.info("訓練 Random Forest …")
    rf = train_random_forest.train(data)
    log.info("訓練 SVM linear …")
    svm_lin = train_svm.train(data, kernel="linear")
    log.info("訓練 SVM rbf …")
    svm_rbf = train_svm.train(data, kernel="rbf")
    log.info("訓練自建 MLP 神經網路 …")
    mlp = train_mlp.train(data)

    results = {r["name"]: r for r in (lr, rf, svm_lin, svm_rbf, mlp)}

    # 選最佳（validation f1_macro）
    best = max(results, key=lambda k: results[k]["metrics"]["f1_macro"])
    best_info = {
        "best_model": best,
        "metrics": {k: results[best]["metrics"][k]
                    for k in ("accuracy", "f1_macro", "roc_auc")},
        "label_order": config.LABEL_ORDER,
    }
    with open(os.path.join(OUT_DIR, "best_model.json"), "w", encoding="utf-8") as f:
        json.dump(best_info, f, ensure_ascii=False, indent=2)

    _write_csv(results)
    return {"results": results, "best": best, "best_info": best_info}


def _write_csv(results: dict) -> None:
    path = os.path.join(OUT_DIR, "model_comparison.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "accuracy", "precision", "recall", "f1", "roc_auc"])
        for name, r in results.items():
            m = r["metrics"]
            w.writerow([name, m["accuracy"], m["precision_macro"], m["recall_macro"],
                        m["f1_macro"], m["roc_auc"] if m["roc_auc"] is not None else ""])
    log.info("已寫入 model_comparison.csv → %s", path)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sample = int(sys.argv[1]) if len(sys.argv) > 1 else None
    out = run(sample=sample)
    print("\n=== 模型比較 ===")
    for name, r in out["results"].items():
        m = r["metrics"]
        print(f"  {name:22s} acc={m['accuracy']}  f1={m['f1_macro']}  roc_auc={m['roc_auc']}")
    print(f"\n最佳模型（validation f1）：{out['best']}")
