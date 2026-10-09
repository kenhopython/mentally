# -*- coding: utf-8 -*-
"""
models/train_mlp.py — 自建神經網路 MLP（多層感知器，3 分類）

Input：data dict（來自 models.common.get_data()，已按 person_id 切分）。
Output：dict（name/model/metrics/path），模型存成 trained_models/my_mlp.pkl。
網路：29 特徵 → Dense(64, ReLU) → Dense(32, ReLU) → Dense(3, Softmax)
要求：predict_proba（與 LR/SVM 介面一致，供 api/prediction.py 直接載入）。
注意：
  - 不使用 early_stopping（它會在 train 內部再切一份驗證集，造成同一人跨 split
    的輕微洩漏）；改用固定 max_iter，模型選擇一律看外部 validation set。
  - MLPClassifier 沒有 class_weight，改用 sample_weight 做類別平衡。
"""
import logging

import numpy as np
from sklearn.neural_network import MLPClassifier

from evaluation import metrics as ev
from models import common
from preprocessing import config

log = logging.getLogger("models.mlp")

HIDDEN_LAYERS = (64, 32)
MAX_ITER = 300
LEARNING_RATE = 1e-3
ALPHA = 1e-4          # L2 正則化


def balanced_weights(y: np.ndarray) -> np.ndarray:
    """依類別頻率倒數給樣本權重（等效 class_weight='balanced'）。"""
    classes, counts = np.unique(y, return_counts=True)
    freq = dict(zip(classes, counts))
    n = len(y)
    return np.array([n / (len(classes) * freq[c]) for c in y], dtype=float)


def train(data: dict) -> dict:
    model = MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYERS,
        activation="relu",
        solver="adam",
        alpha=ALPHA,
        learning_rate_init=LEARNING_RATE,
        max_iter=MAX_ITER,
        early_stopping=False,        # 見檔頭說明：避免內部再切驗證集
        random_state=config.SEED,    # 可重現
    )
    model.fit(data["X_train"], data["y_train"],
              sample_weight=balanced_weights(data["y_train"]))

    y_pred = model.predict(data["X_val"])
    y_proba = model.predict_proba(data["X_val"])
    m = ev.evaluate_model(data["y_val"], y_pred, y_proba, config.LABEL_ORDER)

    name = "my_mlp"
    path = common.save_model(model, name)
    return {"name": name, "model": model, "metrics": m, "path": path}


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sample = int(sys.argv[1]) if len(sys.argv) > 1 else None
    data = common.get_data()
    if sample:
        from models.model_comparison import subsample_train
        data = subsample_train(data, sample)
    r = train(data)
    m = r["metrics"]
    print(f"my_mlp | accuracy: {m['accuracy']} | f1_macro: {m['f1_macro']} "
          f"| roc_auc: {m['roc_auc']}")
