# -*- coding: utf-8 -*-
"""
api/fusion.py — ML ＋ DeepSeek 雙分類器投票融合

Input：prediction.run_ml() 的結果 dict。
Output：同一個 dict，但 prediction 區塊多了投票欄位。

投票規則（依使用者指定）：
  ┌ ML 與 DeepSeek 一致  → 標籤不變，信心度提升（雙方互相印證）
  ├ 兩者不一致           → 標籤一律以 ML 為準，信心度打折並標記 model_disagreement
  └ DeepSeek 棄權/逾時   → 標籤與信心度都用純 ML，標記 consensus = "ml_only"

為什麼不一致時不採用 DeepSeek：LLM 的輸出無法離線重現、也沒有校準過的機率，
把它當成「第二意見」而非決策者，才不會讓整份預測結果失去可驗證性。

信心度調整是有界的（不會被拉到 0 或 1）：
  一致 → conf + (1 - conf) * AGREE_BOOST
  分歧 → conf * DISAGREE_PENALTY
原始的 ML 信心度保留在 confidence_ml，前端與報告都能看到調整前後。
連接方式：被 api/main.py 的 _predict_full() 呼叫。
"""
import logging

from api import deepseek

log = logging.getLogger("api.fusion")

AGREE_BOOST = 0.15        # 一致時，往 1.0 靠攏的比例（剩餘空間的 15%）
DISAGREE_PENALTY = 0.85   # 分歧時，信心度乘上的折扣

# 只要 status != "ok"（unavailable / throttled / no_api_key / invalid_label /
# skipped / deferred…）一律視為棄權 → consensus = "ml_only"，絕不讓 LLM 影響標籤。


def llm_vote(result: dict, quality_status: str | None = None,
             respect_throttle: bool = True) -> dict:
    """請 DeepSeek 投第二票。品質不足時直接棄權，不浪費 API。

    respect_throttle=False 供批次使用（見 api.deepseek.vote）。
    """
    status = quality_status or (result.get("data_quality") or {}).get("status")
    if status == "LOW":
        return {"status": "skipped", "label": None, "reason": "訊號品質不足"}

    from api.prediction import to_deepseek_input
    return deepseek.vote(to_deepseek_input(result), respect_throttle=respect_throttle)


def fuse(result: dict, vote_result: dict) -> dict:
    """把投票結果併進 ML 結果（就地修改並回傳）。"""
    pred = result["prediction"]
    ml_label = pred["emotion"]
    ml_conf = float(pred["confidence"])

    pred["ml_emotion"] = ml_label
    pred["confidence_ml"] = round(ml_conf, 4)

    llm_label = vote_result.get("label")

    if vote_result.get("status") != "ok" or llm_label is None:
        # DeepSeek 沒投票 → 純 ML
        pred["llm_emotion"] = None
        pred["model_agreement"] = None
        pred["consensus"] = "ml_only"
        fused = ml_conf
    elif llm_label == ml_label:
        pred["llm_emotion"] = llm_label
        pred["model_agreement"] = True
        pred["consensus"] = "high"
        fused = ml_conf + (1.0 - ml_conf) * AGREE_BOOST
    else:
        # 分歧 → 標籤仍以 ML 為準，只調降信心度並標記
        pred["llm_emotion"] = llm_label
        pred["model_agreement"] = False
        pred["consensus"] = "low"
        fused = ml_conf * DISAGREE_PENALTY

    pred["confidence"] = round(min(1.0, max(0.0, fused)), 4)

    result["llm_vote"] = {
        "status": vote_result.get("status"),
        "label": llm_label,
        "llm_confidence": vote_result.get("llm_confidence"),
        "reason": vote_result.get("reason", ""),
        "cached": bool(vote_result.get("cached", False)),
    }
    if pred["model_agreement"] is False:
        log.info("模型分歧：ML=%s / DeepSeek=%s（採 ML）", ml_label, llm_label)
    return result
