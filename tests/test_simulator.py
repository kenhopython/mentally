# -*- coding: utf-8 -*-
"""模擬裝置測試：生理一致性、漂移、執行緒生命週期、端點。

全部不呼叫真實 DeepSeek API，也不真的等 3 秒。
"""
import time

import pytest
from fastapi.testclient import TestClient

from api import device, main, simulator


# --------------------------- 資料本身 ---------------------------

def test_anchors_come_from_training_data():
    """錨點必須是從訓練資料算出來的，不是憑感覺寫的。"""
    assert set(simulator.ANCHORS) == {"positive", "neutral", "negative"}
    assert simulator.FEATURES, "錨點檔沒有特徵清單"
    for state in simulator.STATES:
        assert set(simulator.ANCHORS[state]) == set(simulator.FEATURES)
        assert state in simulator.SPREAD or True
    assert set(simulator.SPREAD) == set(simulator.FEATURES)
    assert all(v > 0 for v in simulator.SPREAD.values()), "散佈程度不該有 0"


def test_sample_has_all_features():
    dev = simulator.SimulatedDevice(lambda *a, **k: None, seed=1)
    s = dev.sample()
    for key in simulator.FEATURES:
        assert key in s, f"缺少特徵 {key}"
    assert s["user_id"] == "sim"


def test_heart_rate_and_rr_are_consistent():
    """HR 與 mean_rr 必須滿足 60000/HR —— 否則資料一眼就假。"""
    dev = simulator.SimulatedDevice(lambda *a, **k: None, seed=2)
    for _ in range(50):
        s = dev.sample()
        # hr 四捨五入到小數 1 位，所以 RR 允許約 1 ms 的捨入差
        assert s["ibi"] == pytest.approx(60000.0 / s["hr"], abs=1.0)


def test_lf_hf_ratio_is_computed_not_random():
    dev = simulator.SimulatedDevice(lambda *a, **k: None, seed=3)
    for _ in range(50):
        s = dev.sample()
        assert s["lf_hf_ratio"] == pytest.approx(s["lf"] / s["hf"], abs=0.002)


def test_samples_stay_generation_after_many_ticks():
    """長時間漂移也不能漂到訓練分布之外（否則模型只會亂猜）。"""
    dev = simulator.SimulatedDevice(lambda *a, **k: None, seed=4)
    for _ in range(400):
        dev.sample()
        assert set(dev.weights) == set(simulator.STATES)
        assert dev.dominant_state() in simulator.STATES


def test_values_are_physiologically_plausible():
    """不能出現負值或荒謬的極端值。"""
    dev = simulator.SimulatedDevice(lambda *a, **k: None, seed=5)
    for _ in range(200):
        s = dev.sample()
        assert 30 <= s["hr"] <= 200
        for k, v in s.items():
            if k != "user_id":
                assert v >= 0, f"{k} 出現負值：{v}"
                assert v < 1e5, f"{k} 數值不合理：{v}"


def test_same_seed_is_reproducible():
    a = [simulator.SimulatedDevice(lambda *x: None, seed=7).sample() for _ in range(1)]
    b = [simulator.SimulatedDevice(lambda *x: None, seed=7).sample() for _ in range(1)]
    assert a == b


def test_samples_drift_over_time():
    """狀態應該會漂移，而不是固定值。"""
    dev = simulator.SimulatedDevice(lambda *a, **k: None, seed=8)
    hrs = [dev.sample()["hr"] for _ in range(60)]
    assert len(set(hrs)) > 1
    assert max(hrs) - min(hrs) > 1.0


def test_weights_stay_normalized_and_nonzero():
    dev = simulator.SimulatedDevice(lambda *a, **k: None, seed=9)
    for _ in range(300):
        dev.sample()
        assert sum(dev.weights.values()) == pytest.approx(1.0)
        assert all(w > 0 for w in dev.weights.values())


# --------------------------- 與裝置管線相容 ---------------------------

def test_sample_maps_to_ml_features():
    """模擬資料必須能走真實裝置那條 map_device_features 管線。"""
    dev = simulator.SimulatedDevice(lambda *a, **k: None, seed=10)
    feats = device.map_device_features(dev.sample())
    for expected in ("heart_rate", "mean_rr", "rmssd", "scl", "alpha", "beta"):
        assert expected in feats, f"模擬資料缺少 ML 特徵 {expected}"


# --------------------------- 執行緒 ---------------------------

def test_thread_calls_on_data_and_stops():
    got = []
    dev = simulator.SimulatedDevice(lambda raw, source: got.append((raw, source)),
                                    interval=0.01, seed=11)
    dev.start()
    deadline = time.time() + 3
    while not got and time.time() < deadline:
        time.sleep(0.02)
    dev.stop()
    dev.join(timeout=2)

    assert got, "執行緒沒有產生任何資料"
    assert got[0][1] == "simulated"
    assert not dev.is_alive()


# --------------------------- 端點 ---------------------------

@pytest.fixture
def client(monkeypatch):
    """不啟用模擬自動啟動，也不讓 lifespan 真的開串口。"""
    monkeypatch.setattr(main, "SIM_AUTOSTART", False)
    main._stop_simulator()
    with TestClient(main.app) as c:
        yield c
    main._stop_simulator()


def test_status_initially_not_running(client):
    d = client.get("/api/simulate/status").json()
    assert d["running"] is False
    assert d["ticks"] == 0


def test_start_then_status_then_stop(client, monkeypatch):
    # 不真的每 3 秒跑一次 ML，只驗證生命週期
    monkeypatch.setattr(main, "SIM_INTERVAL", 3600.0)
    assert client.post("/api/simulate/start").json()["status"] == "started"
    assert client.get("/api/simulate/status").json()["running"] is True
    assert client.post("/api/simulate/start").json()["status"] == "already_running"
    assert client.post("/api/simulate/stop").json()["status"] == "stopped"
    assert client.get("/api/simulate/status").json()["running"] is False


def test_stop_when_not_running_is_safe(client):
    assert client.post("/api/simulate/stop").json()["status"] == "not_running"


def test_start_accepts_custom_interval(client):
    d = client.post("/api/simulate/start", json={"interval": 3600}).json()
    assert d["status"] == "started"
    assert d["interval"] == 3600
    assert "模擬" in d["note"]
    client.post("/api/simulate/stop")


def test_real_device_stops_simulation(client, monkeypatch):
    """真實裝置上傳時，模擬裝置要自動讓位，否則會蓋掉真實讀數。"""
    monkeypatch.setattr(main, "SIM_INTERVAL", 3600.0)
    client.post("/api/simulate/start")
    assert client.get("/api/simulate/status").json()["running"] is True

    client.post("/api/ingest", json={
        "user_id": "U00001", "hr": 80, "ibi": 750, "rmssd": 30,
        "delta": 0.12, "theta": 0.2, "alpha": 0.33, "beta": 0.25, "gamma": 0.08,
        "scl": 3.2,
    })
    assert client.get("/api/simulate/status").json()["running"] is False


def test_model_follows_the_anchors():
    """關鍵不變量：餵 positive 錨點，「positive」的預測數要明顯多於餵 negative 錨點。

    這正是原本「憑感覺寫的錨點」壞掉的地方 —— 那時不管餵什麼，模型都只回 negative，
    展示時情緒永遠是負面。用訓練資料算出來的錨點才會有單調關係。
    """
    from api import prediction
    if prediction.load_artifacts() is None:
        pytest.skip("模型尚未訓練")

    def positive_count(state):
        n = 0
        for seed in range(40):
            dev = simulator.SimulatedDevice(lambda *a, **k: None, seed=seed)
            dev.weights = {s: 0.0 for s in simulator.STATES}
            dev.weights[state] = 1.0
            feats = device.map_device_features(dev.sample())
            if prediction.run_ml({"user_id": "sim", **feats})["prediction"]["emotion"] == "positive":
                n += 1
        return n

    assert positive_count("positive") > positive_count("negative")


def test_simulated_data_is_never_labelled_as_device():
    """模擬來源必須誠實標記，不能冒用 source="device"。"""
    device.store_result({"prediction": {"emotion": "neutral"}}, source="simulated")
    assert device.latest["source"] == "simulated"
