# -*- coding: utf-8 -*-
"""自建 MLP 神經網路測試：模型已訓練完成，驗證可載入、介面與其他模型一致。"""
import os

import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier

from preprocessing import config

MODEL_PATH = os.path.join(config.BASE_DIR, "trained_models", "my_mlp.pkl")

pytestmark = pytest.mark.skipif(not os.path.exists(MODEL_PATH),
                                reason="尚未訓練 my_mlp.pkl（先跑 python -m models.train_mlp）")


@pytest.fixture(scope="module")
def model():
    import joblib
    return joblib.load(MODEL_PATH)


def test_architecture(model):
    assert isinstance(model, MLPClassifier)
    assert model.hidden_layer_sizes == (64, 32)      # 兩層隱藏層
    assert model.n_outputs_ == 3                     # 三分類


def test_predict_proba_shape(model):
    import joblib
    pre = joblib.load(os.path.join(config.BASE_DIR, "trained_models", "preprocessor.pkl"))
    x = np.zeros((2, len(pre["feature_names"])))
    p = model.predict_proba(x)
    assert p.shape == (2, 3)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)   # 機率和為 1


def test_labels_match_order(model):
    assert set(model.classes_) == {0, 1, 2}


def test_reproducible_under_seed():
    """同一個 seed 重訓，權重初始值應一致（可重現性）。"""
    from models.train_mlp import HIDDEN_LAYERS
    a = MLPClassifier(hidden_layer_sizes=HIDDEN_LAYERS, random_state=config.SEED,
                      max_iter=1, early_stopping=False)
    b = MLPClassifier(hidden_layer_sizes=HIDDEN_LAYERS, random_state=config.SEED,
                      max_iter=1, early_stopping=False)
    X = np.random.default_rng(0).normal(size=(30, 29))
    y = np.tile([0, 1, 2], 10)
    a.fit(X, y)
    b.fit(X, y)
    assert np.allclose(a.coefs_[0], b.coefs_[0])
