# -*- coding: utf-8 -*-
"""
api/device.py — 真實裝置資料接收（Arduino 串口 / ESP 無線）

Input：裝置 JSON（hr / ibi / hrv / eda / attention / meditation，或已算好的
       delta/theta/alpha/beta/gamma、lf/hf/rmssd/sdnn…）。
Output：map_device_features → {ML 短名特徵: 值}，餵給 prediction.run_ml()。
Processing：把裝置原始欄位映射到 ML 的 29 個特徵（缺者留空，由 imputer 補）。
連接方式：被 api/main.py 的 POST /api/ingest 與 SerialReader 呼叫。
"""
import json
import logging
import threading
import time

log = logging.getLogger("api.device")

# 裝置欄位 → ML 短名特徵（可擴充：韌體算越多，預測越準）
DEVICE_FEATURE_MAP = {
    "hr": "heart_rate", "ibi": "mean_rr", "hrv": "rmssd",
    "eda": "scl", "eda_raw": "scl",
    "delta": "delta", "theta": "theta", "alpha": "alpha", "beta": "beta", "gamma": "gamma",
    "lf": "lf", "hf": "hf", "lf_hf_ratio": "lf_hf_ratio", "sdnn": "sdnn",
    "pnn50": "pnn50", "total_power": "total_power",
    "scr_frequency": "scr_frequency", "tonic": "tonic", "phasic": "phasic",
    "scr_amplitude": "scr_amplitude", "eda_variability": "eda_variability",
    "eda_latency": "eda_latency",
    "mean_rr": "mean_rr", "heart_rate": "heart_rate", "rmssd": "rmssd", "scl": "scl",
}

# 最新一筆裝置結果（供 GET /api/latest 即時推播）
latest = {"result": None, "source": None, "ts": 0.0}


def map_device_features(raw: dict) -> dict:
    """把裝置原始 JSON 映射成 ML 短名特徵 dict。"""
    out = {}
    for dev_key, ml_key in DEVICE_FEATURE_MAP.items():
        if dev_key in raw and raw[dev_key] is not None:
            out[ml_key] = raw[dev_key]
    return out


def store_result(result: dict, source: str) -> None:
    latest["result"] = result
    latest["source"] = source
    latest["ts"] = time.time()


class SerialReader(threading.Thread):
    """背景執行緒：讀 Arduino 串口，每行 JSON → on_data 回呼。"""
    def __init__(self, port: str, baud: int, on_data):
        super().__init__(daemon=True, name="DeviceSerialReader")
        self.port = port
        self.baud = baud
        self.on_data = on_data
        self.connected = False
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        try:
            import serial
        except ImportError:
            log.warning("未安裝 pyserial，串口讀取停用")
            return
        while not self._stop.is_set():
            try:
                with serial.Serial(self.port, self.baud, timeout=1) as ser:
                    self.connected = True
                    log.info("已連接 Arduino 串口：%s", self.port)
                    while not self._stop.is_set():
                        line = ser.readline().decode("utf-8", errors="ignore").strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(data, dict):
                            self.on_data(data, source="device")
            except Exception as e:
                self.connected = False
                log.warning("串口 %s 無法開啟（%s），3 秒後重試", self.port, e)
            self._stop.wait(3)
