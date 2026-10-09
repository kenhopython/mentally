# -*- coding: utf-8 -*-
"""
api/chat.py — AI 對話（DeepSeek）

Input：messages = [{"role": "user"|"assistant", "content": str}, ...]
Output：{"status": "ok", "reply": str} 或 {"status": "unavailable", "reply": str, "reason": str}

上下文（context）由伺服器自己組，不信任前端送來的數字：
  取 api.device.latest 的最新一筆預測，只挑「已經算好」的欄位（情緒、機率、
  信心度、雙模型一致性、訊號品質、重要特徵 z-score）。
  → 對話助手永遠不會看到模型的原始輸入，也就不可能自己捏造數值。

成本控制：只送最後 MAX_HISTORY 則訊息；temperature=0.7。
連接方式：被 api/main.py 的 POST /api/chat 呼叫。
"""
import json
import logging
import os

import requests
from dotenv import load_dotenv

from preprocessing import config

log = logging.getLogger("api.chat")

PROMPT_PATH = os.path.join(config.BASE_DIR, "prompts", "deepseek_chat_prompt.txt")

MAX_HISTORY = 10          # 送給 API 的最大訊息數（不含 system）
MAX_CHARS = 800           # 單則訊息長度上限
MAX_CONTEXT_FEATURES = 6  # 摘要中最多列幾個 z-score

_EMOTION_ZH = {"positive": "正面（放鬆）", "neutral": "中性（平穩）",
               "negative": "負面（緊張／壓力）"}

_FALLBACK_PROMPT = ("你是心情監測系統的 AI 對話助手。用繁體中文、親切溫和的語氣，"
                    "不做醫療診斷，只引用對話中提供的數值，約 200 字內。")


def load_prompt() -> str:
    if os.path.exists(PROMPT_PATH):
        with open(PROMPT_PATH, encoding="utf-8") as f:
            return f.read()
    return _FALLBACK_PROMPT


def _load_api_key() -> str:
    load_dotenv(os.path.join(config.BASE_DIR, ".env"))
    return os.getenv("DEEPSEEK_API_KEY", "")


def build_context() -> dict:
    """從最新一筆預測組出對話上下文（只含已算好的值）。"""
    try:
        from api import device
    except ImportError:                                  # pragma: no cover
        return {"has_data": False}

    result = device.latest.get("result")
    if not result:
        return {"has_data": False,
                "note": "目前還沒有任何量測結果。請使用者先在「即時監測」按一個情境按鈕。"}

    pred = result.get("prediction") or {}
    quality = result.get("data_quality") or {}
    baseline = result.get("baseline") or {}

    # 只挑 |z| 最大的幾個特徵，避免上下文過長
    cmp = baseline.get("comparison") or {}
    top_z = sorted(
        ((k, v) for k, v in cmp.items() if isinstance(v, dict) and v.get("z_score") is not None),
        key=lambda kv: abs(kv[1]["z_score"]), reverse=True,
    )[:MAX_CONTEXT_FEATURES]

    ctx = {
        "has_data": True,
        "ml_emotion": _EMOTION_ZH.get(pred.get("emotion"), pred.get("emotion")),
        "probabilities": pred.get("probabilities"),
        "confidence_percent": (round(pred["confidence"] * 100) if pred.get("confidence") is not None else None),
        "model_agreement": pred.get("model_agreement"),
        "data_quality": quality.get("status"),
        "baseline_status": baseline.get("status"),
        "top_deviations_from_personal_baseline": [
            {"feature": k, "z_score": v["z_score"], "category": v.get("category")} for k, v in top_z
        ],
        "trend": result.get("trend"),
    }
    return ctx


def _clean(messages: list) -> list:
    """只留合法的 role/content，並限制長度。"""
    out = []
    for m in messages or []:
        if not isinstance(m, dict):      # 前端可能送來任何東西，壞資料直接跳過
            continue
        role = m.get("role")
        content = m.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        content = content.strip()[:MAX_CHARS]
        if content:
            out.append({"role": role, "content": content})
    return out[-MAX_HISTORY:]


def reply(messages: list) -> dict:
    """回覆一則對話。messages 為前端累積的歷史（含最新一則 user）。"""
    cleaned = _clean(messages)
    if not cleaned:
        return {"status": "unavailable", "reply": "請先輸入訊息。", "reason": "empty_messages"}

    key = _load_api_key()
    if not key:
        return {"status": "unavailable", "reason": "no_api_key",
                "reply": "（AI 尚未設定，請在專案根目錄的 .env 填入 DEEPSEEK_API_KEY。）"}

    context = build_context()
    system = load_prompt() + "\n\n目前感測數據摘要（JSON）：\n" + json.dumps(
        context, ensure_ascii=False, indent=2)

    try:
        resp = requests.post(
            f"{config.DEEPSEEK_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": config.DEEPSEEK_MODEL,
                "messages": [{"role": "system", "content": system}] + cleaned,
                "temperature": 0.7,
            },
            timeout=25,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        return {"status": "ok", "reply": text, "context_has_data": context.get("has_data", False)}
    except Exception as e:
        log.warning("DeepSeek 對話失敗：%s", e)
        return {"status": "unavailable", "reason": str(e),
                "reply": "抱歉，我暫時無法回應（可能網路或服務問題）。請稍後再試。"}
