# -*- coding: utf-8 -*-
"""
api/main.py — Model API（FastAPI）

啟動：cd <專案根> && uvicorn api.main:app --host 0.0.0.0 --port 8001

端點：/health  /model-info  /baseline/{user_id}  /baseline/{user_id}/update
      /trend/{user_id}  /predict  /predict/batch  /api/chat
      /api/ingest  /api/latest
      /api/simulate/start  /api/simulate/stop  /api/simulate/status
      /predict/excel

沒有接 Arduino 也能跑：啟動時會自動開啟「模擬裝置」（見 api/simulator.py），
每 SIM_INTERVAL 秒產生一筆模擬生理資料，走與真實裝置相同的管線，
前端會誠實標示為「模擬資料」。
"""
import json
import logging
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

# 讓 `python api/main.py` 直接執行（含 VSCode 執行按鈕）也能找到 `api` 套件
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from api import chat, deepseek, device, excel_io, fusion, prediction, simulator, trends
from api.features import (BASELINE_SHORT, EDA_SHORT, EEG_SHORT, HRV_SHORT,
                          RELATIVE_SHORT, compute_derived, normalize_input)
from baseline import baseline as baseline_mod
from preprocessing import config

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api.main")

VERSION = "1.0.0"
DATA_QUALITY_THRESHOLD = 0.7
BASELINE_PATH = os.path.join(config.BASE_DIR, "trained_models", "baselines.json")


# ---------------------------------------------------------------------------
# 完整預測流程（ML + trend + DeepSeek），供 /predict 與裝置 /api/ingest 共用
# ---------------------------------------------------------------------------
def _predict_full(payload: dict, use_deepseek: bool = True,
                  respect_throttle: bool = True) -> dict:
    """完整預測。use_deepseek=False 時略過 LLM（模擬串流用，避免每 3 秒燒一次 API）。

    respect_throttle=False 供「批次」情境（例如上傳 Excel 逐列分析）：那是使用者
    主動按一次、定量筆數的工作，不是裝置每秒串流，被最小間隔擋掉只會讓第 2 列
    之後全都變成 throttled。
    """
    # Step 1–13：ML 預測（含基線、趨勢）
    result = prediction.run_ml(payload)

    # Step 14：DeepSeek 投第二票 → 融合（一致提高信心度／分歧以 ML 為準並標記）
    if use_deepseek:
        vote = fusion.llm_vote(result, respect_throttle=respect_throttle)
    else:
        vote = {"status": "deferred", "label": None,
                "reason": "即時串流中略過 AI 投票以節省 API"}
    result = fusion.fuse(result, vote)

    # Step 15：DeepSeek 解釋最終結果
    trends.append_prediction(result["user_id"], result["prediction"]["emotion"])
    if not use_deepseek:
        result["deepseek"] = {"status": "deferred", "analysis": None}
    elif result["data_quality"]["status"] == "LOW":
        result["deepseek"] = {"status": "skipped", "analysis":
                              "目前訊號品質不足，無法可靠分析，建議重新測量。"}
    else:
        result["deepseek"] = deepseek.explain(prediction.to_deepseek_input(result))
    return result


_serial = None


def _start_serial():
    global _serial
    if config.SERIAL_PORT and _serial is None:
        def on_data(data, source):
            try:
                _predict_full(data)
            except Exception as e:
                log.warning("裝置資料預測失敗：%s", e)
        _serial = device.SerialReader(config.SERIAL_PORT, config.SERIAL_BAUD, on_data)
        _serial.start()


# ---------------------------------------------------------------------------
# 模擬裝置：沒有 Arduino 也能讓網站自己跑起來
# ---------------------------------------------------------------------------
_simulator = None
SIM_INTERVAL = float(os.getenv("SIM_INTERVAL", "5"))
SIM_AUTOSTART = os.getenv("SIM_AUTOSTART", "1").lower() not in ("0", "false", "no")
# 0 = 串流時完全不呼叫 LLM（預設，零 API 成本）；設 5 = 每 5 筆做一次完整 AI 分析
SIM_DEEPSEEK_EVERY = int(os.getenv("SIM_DEEPSEEK_EVERY", "0"))


def _on_simulated_data(raw: dict, source: str):
    """模擬資料走與真實裝置完全相同的管線（map → predict → store）。"""
    global _simulator
    try:
        features = device.map_device_features(raw)
        if not features:
            return
        tick = _simulator.ticks if _simulator else 0
        use_llm = SIM_DEEPSEEK_EVERY > 0 and tick % SIM_DEEPSEEK_EVERY == 0
        result = _predict_full({"user_id": "sim", **features}, use_deepseek=use_llm)
        device.store_result(result, source=source)
    except Exception as e:
        log.warning("模擬資料預測失敗：%s", e)


def _start_simulator(interval: float = None) -> dict:
    global _simulator
    if _simulator is not None and _simulator.is_alive():
        return {"status": "already_running", "interval": _simulator.interval}
    _simulator = simulator.SimulatedDevice(_on_simulated_data,
                                           interval=interval or SIM_INTERVAL)
    _simulator.start()
    return {"status": "started", "interval": _simulator.interval,
            "deepseek_every": SIM_DEEPSEEK_EVERY,
            "note": "產生的是模擬資料，前端會標示「模擬資料」"}


def _stop_simulator() -> bool:
    """停止模擬裝置（回傳先前是否在跑）。"""
    global _simulator
    if _simulator is None:
        return False
    if _simulator.is_alive():
        _simulator.stop()
    _simulator = None
    return True


@asynccontextmanager
async def lifespan(app):
    _start_serial()
    if SIM_AUTOSTART:
        _start_simulator()
        log.info("模擬裝置已自動啟動（每 %.1f 秒一筆；SIM_AUTOSTART=0 可關閉）", SIM_INTERVAL)
    yield
    _stop_simulator()
    if _serial:
        _serial.stop()


app = FastAPI(title="EEG+HRV+EDA 情緒分析 API", version=VERSION, lifespan=lifespan)


# ---------------------------------------------------------------------------
# 基線 store 讀寫
# ---------------------------------------------------------------------------
def _load_baseline_store() -> dict:
    if os.path.exists(BASELINE_PATH):
        with open(BASELINE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_baseline_store(store: dict) -> None:
    os.makedirs(os.path.dirname(BASELINE_PATH), exist_ok=True)
    with open(BASELINE_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)


def _format_baseline(uid: str, baseline: dict) -> dict:
    """把內部 {feature: {mean,std,...}} 轉成 API 回傳的 {eeg/hrv/eda: {<feature>_mean,...}}。"""
    groups = {"eeg": EEG_SHORT, "hrv": HRV_SHORT, "eda": EDA_SHORT}
    out = {"user_id": uid,
           "baseline_status": ("READY" if baseline and any(
               baseline_mod.status(v) == "READY" for v in baseline.values())
               else "INSUFFICIENT_DATA")}
    for gname, feats in groups.items():
        g = {}
        for f in feats:
            b = baseline.get(f) or {}
            g[f"{f}_mean"] = b.get("mean")
            g[f"{f}_std"] = b.get("std")
        out[gname] = g
    return out


# ---------------------------------------------------------------------------
# 端點
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    key = deepseek.load_api_key()
    art_loaded = prediction.load_artifacts() is not None
    return {
        "status": "ok",
        "model_loaded": art_loaded,
        "deepseek_configured": bool(key),
        "version": VERSION,
    }


@app.get("/model-info")
def model_info():
    art = prediction.load_artifacts()
    return {"name": art["model_name"], "version": "1.0",
            "n_features": len(art["preproc"]["feature_names"]),
            "label_order": art["preproc"]["label_order"]}


@app.get("/baseline/{user_id}")
def get_baseline(user_id: str):
    store = _load_baseline_store()
    uid = prediction._normalize_user(user_id)
    baseline = store.get(uid) or store.get(user_id) or {}
    return _format_baseline(user_id, baseline)


@app.post("/baseline/{user_id}/update")
def update_baseline(user_id: str, payload: dict = Body(...)):
    """用新的有效資料更新基線；data_quality 過低者不納入。"""
    flat = compute_derived(normalize_input(payload))
    dq = payload.get("data_quality_score")
    if dq is None:
        dq = 1.0 if str(payload.get("data_quality", "good")).lower() in ("good", "ok") else 0.5
    if float(dq) < DATA_QUALITY_THRESHOLD:
        raise HTTPException(status_code=422, detail="data_quality too low to update baseline")

    store = _load_baseline_store()
    uid = prediction._normalize_user(user_id)
    baseline = store.get(uid, {})
    for f in BASELINE_SHORT:
        if f in flat and flat[f] is not None:
            baseline[f] = baseline_mod.update_feature(baseline.get(f), [flat[f]])
    store[uid] = baseline
    _save_baseline_store(store)
    return {"user_id": user_id, "baseline_status":
            ("READY" if any(baseline_mod.status(v) == "READY" for v in baseline.values())
             else "INSUFFICIENT_DATA")}


@app.get("/trend/{user_id}")
def get_trend(user_id: str):
    uid = prediction._normalize_user(user_id)
    return trends.get_emotion_trend(uid)


@app.post("/predict")
def predict(payload: dict = Body(...)):
    """預測情緒（模擬／完整特徵輸入）。"""
    result = _predict_full(payload)
    # 存成「最新一筆」：AI 對話（api/chat.build_context）才讀得到上下文。
    # source 標記為 simulated，前端據此誠實顯示資料來源。
    device.store_result(result, source="simulated")
    return result


# 上傳 Excel 的批次上限（避免一次回傳超大 JSON 拖垮網頁）
EXCEL_MAX_ROWS = 5000
EXCEL_MAX_FULL = 200       # 完整結果（可逐筆套用到網頁）只回前 200 列
# AI 分析同時打幾條 API。實測每列約 5.7 秒（投票 1.8s + 解釋 3.8s），
# 序列跑 300 列要等近半小時；並行 4 條約 7 分鐘。
# 不要開太大：DeepSeek 有速率限制，而且這裡的瓶頸是等網路，不是 CPU。
EXCEL_AI_WORKERS = max(1, min(8, int(os.getenv("EXCEL_AI_WORKERS", "4"))))

# 原始特徵的展示分組（前端「原始數據預覽」用）。EEG 含頻帶 + 衍生 relative。
EXCEL_FEATURE_GROUPS = [("EEG", EEG_SHORT + RELATIVE_SHORT), ("HRV", HRV_SHORT), ("EDA", EDA_SHORT)]


def _json_num(v):
    """把 numpy 標量（np.float64 / np.int64）轉成 JSON 可序列化的原生 float/int。"""
    if isinstance(v, bool):
        return v
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)


@app.post("/predict/excel")
def predict_excel(file: UploadFile = File(...)):
    """上傳 Excel/CSV，逐列預測情緒。

    **每一列都走與「即時接收資料」完全相同的完整管線**：
    ML 預測 → DeepSeek 投票 → 融合 → DeepSeek 解釋（不是只有 ML）。

    與 /predict（即時）的唯一差別：批次情境不套用 3 秒最小間隔節流
    （respect_throttle=False），否則第 2 列之後全會被 throttled；快取仍然生效。

    注意（重要）：融合規則下 **DeepSeek 不會改變預測標籤**，
    它只提供第二意見與信心度調整，最終 emotion 永遠以 ML 為準。
    因此準確率不受影響，但 CSV 另附 llm_emotion 欄位可單獨比對。

    用途：驗證模型準確率。Excel 只放生理特徵、不放 emotion_label。
    """
    filename = file.filename or ""
    if not filename.lower().endswith((".xlsx", ".xls", ".csv")):
        raise HTTPException(status_code=422, detail="只接受 .xlsx / .xls / .csv")

    try:
        parsed = excel_io.read_rows(file.file, filename)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"無法讀取檔案：{e}")

    rows = parsed["rows"]
    if not rows:
        raise HTTPException(status_code=422,
                            detail="檔案裡沒有可用的特徵欄位（請參照範本 Excel 的欄位名稱）")
    if len(rows) > EXCEL_MAX_ROWS:
        raise HTTPException(
            status_code=422,
            detail=f"一次最多 {EXCEL_MAX_ROWS} 列（收到 {len(rows)} 列）。"
                   f"大量資料請用 python -m models.predict_excel 命令列工具")

    # 每一列都做完整分析（ML + DeepSeek 投票 + 解釋），與即時接收一模一樣。
    # DeepSeek 是等網路的慢操作，用執行緒池並行；pool.map 會保持輸入順序，
    # 所以回傳結果的次序不會亂。
    def _full(item):
        i, r = item
        uid = r["person_id"] or r["sample_id"] or f"excel_{i}"
        return _predict_full({"user_id": uid, **r["features"]},
                             use_deepseek=True, respect_throttle=False)

    with ThreadPoolExecutor(max_workers=EXCEL_AI_WORKERS) as pool:
        analyzed = list(pool.map(_full, enumerate(rows)))

    results = []    # 前 EXCEL_MAX_FULL 列的完整結果（供網頁逐筆渲染）
    summary = []    # 所有列的摘要（供表格 + 下載 CSV）
    for i, (r, res) in enumerate(zip(rows, analyzed)):
        res["_excel"] = {"index": i, "sample_id": r["sample_id"],
                         "person_id": r["person_id"],
                         "features": {k: _json_num(v) for k, v in r["features"].items()}}

        summary.append({
            "index": i, "sample_id": r["sample_id"], "person_id": r["person_id"],
            "emotion": res["prediction"]["emotion"],
            "confidence": res["prediction"]["confidence"],
            "quality": res["data_quality"]["status"],
            "llm_emotion": res["prediction"].get("llm_emotion"),
            "model_agreement": res["prediction"].get("model_agreement"),
        })
        if i < EXCEL_MAX_FULL:
            results.append(res)

    dist = Counter(s["emotion"] for s in summary)
    agreed = sum(1 for s in summary if s["model_agreement"] is True)
    disagreed = sum(1 for s in summary if s["model_agreement"] is False)
    note = (f"全部 {len(summary)} 列都做了完整分析（ML + DeepSeek 投票 + 解釋），"
            "與即時接收資料相同。DeepSeek 只提供第二意見與信心度調整，"
            "**最終標籤一律以 ML 為準**，所以準確率不受影響。"
            "下載 CSV 後以 sample_id 對回原資料的 emotion_label 即可計算準確率。")
    return {
        "count": len(summary),
        "emotion_distribution": {k: dist.get(k, 0) for k in ("positive", "neutral", "negative")},
        "mapped_columns": parsed["mapped"],
        "ignored_columns": parsed["ignored"],
        "unknown_columns": parsed["unknown"],
        "model_agreement": {"agree": agreed, "disagree": disagreed,
                            "rate": round(agreed / len(summary), 3) if summary else None},
        "feature_groups": [{"group": g, "features": fs} for g, fs in EXCEL_FEATURE_GROUPS],
        "summary": summary,
        "results": results,
        "note": note,
    }


@app.post("/api/ingest")
def ingest_device(raw: dict = Body(...)):
    """真實裝置上傳（ESP 無線／串口），映射為 ML 特徵後跑預測。

    真實資料優先：一旦收到真實裝置上傳，就自動關掉模擬裝置，
    否則模擬資料會在幾秒內蓋掉真實讀數，前端來源標示也會互相閃爍。
    """
    if _simulator is not None and _simulator.is_alive():
        _stop_simulator()
        log.info("收到真實裝置資料，已自動停止模擬裝置")

    features = device.map_device_features(raw)
    if not features:
        raise HTTPException(status_code=422, detail="no valid device features")
    payload = {"user_id": str(raw.get("user_id") or raw.get("person_id") or "device"),
               **features}
    result = _predict_full(payload)
    device.store_result(result, source="device")
    return result


@app.get("/api/latest")
def latest():
    """最新一筆裝置預測結果（供前端即時監測輪詢）。"""
    return device.latest


@app.post("/api/simulate/start")
def simulate_start(payload: dict = Body(default={})):
    """啟動模擬裝置。body 可帶 {"interval": 3}（秒）。"""
    interval = payload.get("interval")
    return _start_simulator(float(interval) if interval else None)


@app.post("/api/simulate/stop")
def simulate_stop():
    """停止模擬裝置。"""
    return {"status": "stopped" if _stop_simulator() else "not_running"}


@app.get("/api/simulate/status")
def simulate_status():
    """模擬裝置狀態（前端用來決定按鈕顯示啟動或停止）。"""
    running = _simulator is not None and _simulator.is_alive()
    return {"running": running,
            "interval": _simulator.interval if running else None,
            "ticks": _simulator.ticks if running else 0,
            "autostart": SIM_AUTOSTART}


@app.post("/api/chat")
def ai_chat(payload: dict = Body(...)):
    """AI 對話。payload = {"messages": [{"role":"user","content":"..."}, ...]}。

    上下文由伺服器自己從最新預測組出（見 api/chat.build_context），不信任前端數字。
    """
    return chat.reply(payload.get("messages", []))


@app.post("/predict/batch")
def predict_batch(payload: dict = Body(...)):
    items = payload.get("items", [])
    if not items:
        raise HTTPException(status_code=422, detail="items 為空")

    # 批次刻意「不」逐筆呼叫 DeepSeek 投票（成本考量）：
    # 每筆仍帶上 fusion 欄位以維持回傳格式一致，但 consensus 固定為 ml_only。
    results = []
    for item in items:
        r = prediction.run_ml(item)
        r = fusion.fuse(r, {"status": "skipped", "label": None, "reason": "batch 模式不投票"})
        results.append(r)
        trends.append_prediction(r["user_id"], r["prediction"]["emotion"])

    # 一次 DeepSeek summary（不逐筆呼叫）
    counts = Counter(r["prediction"]["emotion"] for r in results)
    summary_input = {
        "batch_size": len(results),
        "emotion_distribution": dict(counts),
        "data_quality": {"score": round(sum(r["data_quality"]["score"] for r in results) / len(results), 3)},
    }
    ds = deepseek.explain(summary_input) if len(results) > 0 else \
        {"status": "unavailable", "analysis": None}

    return {"count": len(results), "results": results, "deepseek_summary": ds}


# 伺服前端靜態檔案（單一伺服器：API + Website，開 http://localhost:8001/）
FRONTEND_DIR = os.path.join(config.BASE_DIR, "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


def _find_port_holder(port: int) -> list:
    """找出佔用某個埠的 PID（Windows netstat / Unix lsof）。找不到回傳空 list。"""
    import re
    import subprocess

    try:
        if os.name == "nt":
            out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True,
                                 timeout=10).stdout
            pids = {m.group(1) for m in re.finditer(
                rf"^\s*TCP\s+\S*:{port}\s+\S+\s+LISTENING\s+(\d+)", out, re.M)}
        else:
            out = subprocess.run(["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
                                 capture_output=True, text=True, timeout=10).stdout
            pids = {p.strip() for p in out.split() if p.strip()}
    except Exception:
        return []
    return sorted(pids)


if __name__ == "__main__":
    import uvicorn

    print("\n  心情監測系統 — EEG + HRV + EDA 多模態生理訊號分析")
    print("  啟動後請用瀏覽器打開： http://localhost:8001/")
    print("  （0.0.0.0 是「監聽所有網卡」的意思，不是網址，打不開）\n", flush=True)

    try:
        uvicorn.run(app, host="0.0.0.0", port=8001)
    except (SystemExit, OSError) as e:
        # 最常見的啟動失敗：8001 已經被另一個 Python 佔著（例如上一次沒關乾淨）
        holders = _find_port_holder(8001)
        print("\n" + "=" * 62)
        print("  啟動失敗：埠 8001 已經被佔用。")
        if holders:
            print(f"  佔用的 PID：{', '.join(holders)}")
            print("\n  解法（在 PowerShell 執行）：")
            for pid in holders:
                print(f"      taskkill /F /PID {pid}")
        else:
            print("\n  解法：關掉其他正在執行的 main.py，或重開機。")
        print("\n  另一個可能：你只是想開網站 —— 那它其實已經在跑了，")
        print("  直接開 http://localhost:8001/ 就好，不用再啟動一次。")
        print("=" * 62 + "\n")
        raise SystemExit(1) from e

