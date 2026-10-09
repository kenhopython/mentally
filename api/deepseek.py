# -*- coding: utf-8 -*-
"""
api/deepseek.py — DeepSeek 雙角色層：① 解釋者 ② 第二個分類器投票者

Input：to_deepseek_input() 產出的 dict（只含已算好的數值）。
Output：
  explain() → {status:"ok", summary, eeg_analysis, ...}
              {status:"unavailable", analysis:None, reason}
  vote()    → {status:"ok", label:"negative", llm_confidence, reason}
              {status:"unavailable"|"throttled", label:None, reason}
- API key 從 .env 讀取，不寫死在 source。
- DeepSeek 掛掉 ≠ ML 掛掉：呼叫端仍可回傳 ML 結果。
- vote() 有快取與最小間隔（裝置每秒上傳一次時不會打爆 API）。
"""
import json
import logging
import os
import time

import requests
from dotenv import load_dotenv

from preprocessing import config

log = logging.getLogger("api.deepseek")

PROMPT_PATH = os.path.join(config.BASE_DIR, "prompts", "deepseek_system_prompt.txt")
VOTE_PROMPT_PATH = os.path.join(config.BASE_DIR, "prompts", "deepseek_vote_prompt.txt")

_KEYS = ["summary", "eeg_analysis", "hrv_analysis", "eda_analysis", "multimodal_analysis",
         "baseline_analysis", "trend_analysis", "possible_state", "self_regulation",
         "limitations"]

LABELS = ("positive", "neutral", "negative")

# --- vote() 的成本控制 ---------------------------------------------------
VOTE_MIN_INTERVAL = float(os.getenv("DEEPSEEK_VOTE_MIN_INTERVAL", "3"))  # 秒
VOTE_CACHE_TTL = float(os.getenv("DEEPSEEK_VOTE_TTL", "120"))            # 秒
_vote_cache = {}          # key -> (timestamp, verdict)
_vote_last_call = [0.0]   # 上次真正呼叫 API 的時間


def load_api_key() -> str:
    load_dotenv(os.path.join(config.BASE_DIR, ".env"))
    return os.getenv("DEEPSEEK_API_KEY", "")


def load_system_prompt() -> str:
    if os.path.exists(PROMPT_PATH):
        with open(PROMPT_PATH, encoding="utf-8") as f:
            return f.read()
    return "你是一個 EEG＋HRV＋EDA 生理訊號分析助手。"


def load_vote_prompt() -> str:
    if os.path.exists(VOTE_PROMPT_PATH):
        with open(VOTE_PROMPT_PATH, encoding="utf-8") as f:
            return f.read()
    return ("你是生理訊號分類器。只能輸出 JSON："
            '{"label":"positive|neutral|negative","llm_confidence":0-1,"reason":"..."}')


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError("無法解析 DeepSeek JSON")


def _cache_key(deepseek_input: dict) -> str:
    """把輸入量化成快取鍵：z-score 與機率取到小數 2 位，避免裝置微幅抖動就打一次 API。"""
    parts = []
    for group in ("eeg", "hrv", "eda"):
        for k, v in (deepseek_input.get(group) or {}).items():
            parts.append(f"{k}={round(v, 2) if isinstance(v, (int, float)) else v}")
    for k, v in (deepseek_input.get("probabilities") or {}).items():
        parts.append(f"p_{k}={round(v, 2) if isinstance(v, (int, float)) else v}")
    return "|".join(parts)


def vote(deepseek_input: dict, respect_throttle: bool = True) -> dict:
    """第二個分類器：請 DeepSeek 從三個標籤中投一票（temperature=0，可重現）。

    respect_throttle=False 用於「批次」情境（例如上傳 Excel 逐筆分析）：
    那是使用者主動按一次、定量筆數的工作，不是裝置每秒串流，
    被最小間隔擋掉只會讓後面每一列都變成 throttled 而失去意義。
    快取仍然生效。

    回傳 {"status": "ok", "label": ..., "llm_confidence": ..., "reason": ...}
    或   {"status": "unavailable"|"throttled"|"no_api_key", "label": None, "reason": ...}
    """
    key = load_api_key()
    if not key:
        return {"status": "no_api_key", "label": None, "reason": "no_api_key"}

    ck = _cache_key(deepseek_input)
    now = time.time()

    hit = _vote_cache.get(ck)
    if hit and now - hit[0] < VOTE_CACHE_TTL:
        return {**hit[1], "cached": True}

    # 最小間隔節流：裝置串流時保護 API
    if respect_throttle and now - _vote_last_call[0] < VOTE_MIN_INTERVAL:
        return {"status": "throttled", "label": None,
                "reason": f"呼叫間隔 < {VOTE_MIN_INTERVAL}s"}

    try:
        _vote_last_call[0] = now
        resp = requests.post(
            f"{config.DEEPSEEK_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": config.DEEPSEEK_MODEL,
                "messages": [
                    {"role": "system", "content": load_vote_prompt()},
                    {"role": "user", "content": json.dumps(deepseek_input, ensure_ascii=False)},
                ],
                "temperature": 0,          # 固定 → 同一筆輸入結果可重現
                "response_format": {"type": "json_object"},
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = _parse_json(resp.json()["choices"][0]["message"]["content"])

        label = str(data.get("label", "")).strip().lower()
        if label not in LABELS:            # 模型亂回 → 視為棄權，不讓髒標籤進到預測
            return {"status": "invalid_label", "label": None,
                    "reason": f"非允許標籤：{label!r}"}

        conf = data.get("llm_confidence", 0.5)
        try:
            conf = max(0.0, min(1.0, float(conf)))
        except (TypeError, ValueError):
            conf = 0.5

        verdict = {"status": "ok", "label": label, "llm_confidence": round(conf, 3),
                   "reason": str(data.get("reason", ""))[:80]}
        _vote_cache[ck] = (now, verdict)
        return verdict
    except Exception as e:
        log.warning("DeepSeek 投票失敗：%s", e)
        return {"status": "unavailable", "label": None, "reason": str(e)}


def clear_vote_cache() -> None:
    """清空投票快取（測試用）。"""
    _vote_cache.clear()
    _vote_last_call[0] = 0.0


def explain(deepseek_input: dict) -> dict:
    """把 ML 結果交給 DeepSeek 解釋，回傳結構化 dict。"""
    key = load_api_key()
    if not key:
        return {"status": "unavailable", "analysis": None, "reason": "no_api_key"}

    system = load_system_prompt()
    user_msg = json.dumps(deepseek_input, ensure_ascii=False, indent=2)

    try:
        resp = requests.post(
            f"{config.DEEPSEEK_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": config.DEEPSEEK_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                "temperature": 0.4,
                "response_format": {"type": "json_object"},
            },
            timeout=25,
        )
        resp.raise_for_status()
        data = _parse_json(resp.json()["choices"][0]["message"]["content"])
        return {"status": "ok", **{k: data.get(k, "") for k in _KEYS}}
    except Exception as e:
        log.warning("DeepSeek 解釋失敗：%s", e)
        return {"status": "unavailable", "analysis": None, "reason": str(e)}
