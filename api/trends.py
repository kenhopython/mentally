# -*- coding: utf-8 -*-
"""
api/trends.py — 情緒趨勢分析 + 預測歷史儲存

Input：user_id。
Output：{"last_7": {...}, "last_30": {...}, "trend": "..."}。
- 只描述歷史分類結果（negative_increasing 等），不做心理健康結論。
連接方式：被 api/main.py 的 GET /trend/{user_id} 與 /predict 呼叫。
"""
import json
import os
import threading
import time
from collections import Counter

from preprocessing import config

HISTORY_PATH = os.path.join(config.BASE_DIR, "data", "prediction_history.json")

# 模擬裝置是背景執行緒、每幾秒就寫一次，會與 API 請求執行緒同時讀寫同一個檔案。
# load → append → save 不是原子操作，沒有鎖會互相覆蓋、甚至寫出壞掉的 JSON。
_lock = threading.Lock()


def _load() -> dict:
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save(hist: dict) -> None:
    """原子寫入：先寫暫存檔再置換，避免中斷時留下半截 JSON。

    Windows 的 os.replace() 只要目標檔案正被另一個行程開著，就會丟
    PermissionError (WinError 5) —— 而「另一個行程」是真實存在的：
    伺服器在跑（模擬裝置每 5 秒寫一次），同時又跑 pytest 或另一個工具時，
    兩個行程各有一個 _lock，跨行程完全擋不住。實測同一條測試會交替通過／失敗。
    這裡重試幾次（短暫後退），真的搶不到才放棄。
    """
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    tmp = HISTORY_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(hist, f, ensure_ascii=False, indent=2)

    last_err = None
    for attempt in range(5):
        try:
            os.replace(tmp, HISTORY_PATH)
            return
        except PermissionError as e:      # 目標檔被其他行程暫用
            last_err = e
            time.sleep(0.02 * (attempt + 1))   # 20ms, 40ms, 60ms, 80ms
    try:
        os.remove(tmp)
    except OSError:
        pass
    raise last_err


def append_prediction(user_id: str, emotion: str, ts: float | None = None) -> None:
    """記錄一筆預測結果（供趨勢分析）。"""
    with _lock:
        hist = _load()
        hist.setdefault(user_id, []).append({"ts": ts if ts is not None else time.time(),
                                             "emotion": emotion})
        # 每個使用者保留最近 200 筆
        hist[user_id] = hist[user_id][-200:]
        _save(hist)


def _trend_direction(emotions: list) -> str:
    if len(emotions) < 3:
        return "insufficient"
    recent = emotions[-7:]
    prev = emotions[-14:-7]
    neg_recent = recent.count("negative") / len(recent)
    neg_prev = prev.count("negative") / max(1, len(prev))
    if neg_recent > 0.5 and neg_recent > neg_prev + 0.1:
        return "negative_increasing"
    if neg_recent < 0.3:
        return "stable_or_improving"
    return "stable"


def get_emotion_trend(user_id: str) -> dict:
    hist = _load().get(user_id, [])
    emotions = [r["emotion"] for r in hist if r.get("emotion") in
                ("positive", "neutral", "negative")]
    last_7 = dict(Counter(emotions[-7:]))
    last_30 = dict(Counter(emotions[-30:]))
    return {
        "last_7": last_7,
        "last_30": last_30,
        "trend": _trend_direction(emotions),
    }
