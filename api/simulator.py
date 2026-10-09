# -*- coding: utf-8 -*-
"""
api/simulator.py — 模擬裝置（沒有 Arduino 也能讓網站跑起來）

Input：無（背景執行緒自己產生）。
Output：每隔 interval 秒（預設 5 秒，見 api/main.py 的 SIM_INTERVAL）呼叫一次
        on_data(device_json)，格式與真實 ESP 上傳完全相同。
Processing：
  1. 在三個情緒錨點（positive / neutral / negative）之間做平滑隨機漫步
     —— 狀態會慢慢漂移，不是固定值，所以圖表看起來像真的。
  2. 錨點與散佈程度**不是憑感覺寫的**，是從訓練資料算出來的每個類別平均值
     與標準差（見 models/build_simulator_anchors.py → api/simulator_anchors.json）。
     憑感覺寫的錨點會落在訓練分布之外，模型只會全部猜同一個類別，展示時一眼就假。
  3. 加上以真實標準差為尺度的雜訊，再修正生理一致性：
     heart_rate 與 ibi 綁定（60000/HR）、lf_hf_ratio 由 lf/hf 算出。

連接方式：被 api/main.py 的 lifespan 自動啟動 / /api/simulate/start 啟動，
          資料走與 /api/ingest 相同的一條管線。

⚠ 產生的資料一律標記 source="simulated"，前端會顯示「模擬資料」，
  不得當成真實量測或臨床數據。
"""
import json
import logging
import os
import random
import threading

log = logging.getLogger("api.simulator")

ANCHOR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulator_anchors.json")

STATES = ["positive", "neutral", "negative"]
MIN_WEIGHT = 0.02     # 避免某個錨點權重歸零後回不來
START_WEIGHTS = {"positive": 0.25, "neutral": 0.50, "negative": 0.25}

# 每次漫步最多移動多少（越小越平滑）。這是「情緒多久換一次」的旋鈕：
#   0.12（預設）→ 主導情緒平均約每分鐘換一次，最像真人
#   0.40        → 約每 20 秒換一次，展示時比較看得出來
#   1.00        → 幾乎每筆都在換，一看就是假的
DRIFT = float(os.getenv("SIM_DRIFT", "0.12"))

# 雜訊尺度：1.0 = 與訓練資料的散佈程度相同（最真實）。
# 調小會讓資料更集中、情緒更穩定，但也就更不像真人。
NOISE_SCALE = float(os.getenv("SIM_NOISE_SCALE", "1.0"))

# 極端值保護（避免雜訊把數值推到荒謬的地方）
BOUNDS = {"hr": (35.0, 190.0), "rmssd": (5.0, 200.0), "sdnn": (5.0, 250.0),
          "pnn50": (0.0, 100.0), "lf": (20.0, 5000.0), "hf": (20.0, 5000.0),
          "total_power": (50.0, 8000.0), "scl": (0.2, 25.0)}


class _MissingAnchors(RuntimeError):
    pass


def _load_anchors() -> dict:
    if not os.path.exists(ANCHOR_PATH):
        raise _MissingAnchors(
            f"找不到 {ANCHOR_PATH}；請先執行：python -m models.build_simulator_anchors")
    with open(ANCHOR_PATH, encoding="utf-8") as f:
        return json.load(f)


_ANCHOR_DATA = _load_anchors()
ANCHORS = _ANCHOR_DATA["anchors"]
SPREAD = _ANCHOR_DATA["spread"]
FEATURES = _ANCHOR_DATA["features"]


class SimulatedDevice(threading.Thread):
    """背景執行緒：每隔 interval 秒產生一筆模擬裝置資料。"""

    def __init__(self, on_data, interval: float = 3.0, seed: int | None = None):
        super().__init__(daemon=True, name="SimulatedDevice")
        self.on_data = on_data
        self.interval = interval
        self.rng = random.Random(seed)
        self._stop = threading.Event()
        self.weights = dict(START_WEIGHTS)
        self.ticks = 0

    # -- 狀態漫步 --------------------------------------------------------
    def _drift(self):
        """權重做小步隨機漫步後正規化（情緒狀態平滑漂移）。"""
        for s in STATES:
            self.weights[s] = max(MIN_WEIGHT,
                                  self.weights[s] + self.rng.uniform(-DRIFT, DRIFT))
        total = sum(self.weights.values())
        for s in STATES:
            self.weights[s] /= total

    def _blend(self, key: str) -> float:
        return sum(self.weights[s] * ANCHORS[s][key] for s in STATES)

    def dominant_state(self) -> str:
        return max(self.weights, key=self.weights.get)

    def sample(self) -> dict:
        """產生一筆裝置格式的 JSON（與 ESP 上傳的欄位相同）。"""
        self._drift()
        self.ticks += 1

        out = {}
        for key in FEATURES:
            value = self._blend(key) + self.rng.gauss(0, SPREAD[key] * NOISE_SCALE)
            lo, hi = BOUNDS.get(key, (0.0, float("inf")))
            out[key] = round(max(lo, min(hi, value)), 4)

        # 生理一致性：HR 與 RR 必須互相吻合（60000 / HR）
        out["ibi"] = round(60000.0 / out["hr"], 1)
        # LF/HF 由 LF 與 HF 算出，不另外加雜訊
        out["lf_hf_ratio"] = round(out["lf"] / max(1e-6, out["hf"]), 3)
        out["user_id"] = "sim"
        return out

    # -- 執行緒 ----------------------------------------------------------
    def run(self):
        log.info("模擬裝置啟動（每 %.1f 秒一筆）", self.interval)
        while not self._stop.is_set():
            try:
                self.on_data(self.sample(), source="simulated")
            except Exception as e:
                log.warning("模擬裝置產生資料失敗：%s", e)
            self._stop.wait(self.interval)
        log.info("模擬裝置停止")

    def stop(self):
        self._stop.set()
