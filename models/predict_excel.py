# -*- coding: utf-8 -*-
"""
models/predict_excel.py — 命令列批量預測（測準確率，不限列數）

網頁的 /predict/excel 一次最多 5000 列；要跑整個測試集（1 萬列以上）或算準確率，
用這支命令列工具。

用法：
  # 1) 只預測，輸出每列的預測情緒 + 信心度
  python -m models.predict_excel --input data/prediction_input_sample.xlsx \
                                 --output data/predictions.csv

  # 2) 同時給「原始訓練 CSV」當真值 → 自動 join sample_id 並算出準確率 / f1 / 混淆矩陣
  python -m models.predict_excel --input data/prediction_input_sample.xlsx \
                                 --output data/predictions.csv \
                                 --truth data/EEG_HRV_EDA_synthetic_training_100K.csv

只跑純 ML（不呼叫 DeepSeek），與網頁批量模式一致。
"""
import argparse
import os

import pandas as pd

from api import excel_io, prediction
from preprocessing import config

LABEL_ORDER = config.LABEL_ORDER   # ["positive", "neutral", "negative"]


def predict(input_path: str, output_path: str, truth_path: str | None = None):
    with open(input_path, "rb") as f:
        parsed = excel_io.read_rows(f, input_path)

    rows = parsed["rows"]
    if not rows:
        raise SystemExit("檔案裡沒有可用的特徵欄位")

    prediction.load_artifacts()

    pred_rows = []
    for i, r in enumerate(rows):
        uid = r["person_id"] or r["sample_id"] or f"row_{i}"
        res = prediction.run_ml({"user_id": uid, **r["features"]})
        pred_rows.append({
            "sample_id": r["sample_id"], "person_id": r["person_id"],
            "predicted_emotion": res["prediction"]["emotion"],
            "confidence": round(res["prediction"]["confidence"], 4),
        })

    out_df = pd.DataFrame(pred_rows)
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"預測 {len(out_df)} 列 → {output_path}")
    print(f"分布：{out_df['predicted_emotion'].value_counts().to_dict()}")

    if truth_path:
        truth = pd.read_csv(truth_path, usecols=["sample_id", "emotion_label"])
        # sample_id 一邊是字串（excel_io 轉過）一邊是 int，統一成字串再 join
        out_df["sample_id"] = out_df["sample_id"].astype(str)
        truth["sample_id"] = truth["sample_id"].astype(str)
        merged = out_df.merge(truth, on="sample_id", how="inner")
        if merged.empty:
            raise SystemExit("--truth 的 sample_id 對不上（請確認是同一份資料）")

        y_true = merged["emotion_label"]
        y_pred = merged["predicted_emotion"]
        n = len(merged)
        acc = (y_true == y_pred).mean()

        from sklearn.metrics import classification_report, confusion_matrix
        print(f"\n=== 準確率（{n} 列可對回真值）===")
        print(f"accuracy = {acc:.4f}")
        print("\n分類報告：")
        print(classification_report(y_true, y_pred, labels=LABEL_ORDER, zero_division=0))
        print("混淆矩陣（列=真值，行=預測）：")
        print(pd.DataFrame(confusion_matrix(y_true, y_pred, labels=LABEL_ORDER),
                           index=["真:" + l for l in LABEL_ORDER],
                           columns=["預:" + l for l in LABEL_ORDER]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Excel/CSV 批量預測（純 ML）")
    ap.add_argument("--input", required=True, help="輸入 .xlsx/.csv")
    ap.add_argument("--output", required=True, help="輸出預測 .csv")
    ap.add_argument("--truth", default=None, help="（選填）原始 CSV，含 sample_id + emotion_label")
    args = ap.parse_args()
    predict(args.input, args.output, args.truth)
