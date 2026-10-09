# -*- coding: utf-8 -*-
"""
models/make_prediction_excel.py — 產生「預測用 Excel」（測準確率用）

目的：從訓練資料的「測試集」切出一份 Excel，內容只有生理特徵 + 樣本識別，
      **沒有 emotion_label**。這樣使用者上傳後看模型預測什麼，再拿 sample_id
      回原 CSV 對回真實標籤，就能自己算準確率 —— 不會先偷看答案。

為什麼用測試集：這群人（person）從沒出現在模型訓練裡，測出來的準確率才公平。

輸出：data/prediction_input_sample.xlsx
  欄位 = sample_id + person_id + 29 個生理特徵（短名），共 31 欄。

執行：
  python -m models.make_prediction_excel              # 預設 200 列
  python -m models.make_prediction_excel --n 1000     # 1000 列
  python -m models.make_prediction_excel --n 0        # 全部測試集
"""
import argparse
import glob
import os

import pandas as pd

from api.features import SHORT_TO_CSV
from preprocessing import config
from preprocessing import feature_engineering

OUT_DEFAULT = os.path.join(config.BASE_DIR, "data", "prediction_input_sample.xlsx")


def find_training_csv() -> str:
    files = sorted(glob.glob(os.path.join(config.BASE_DIR, "data", "*.csv")))
    for f in files:
        if "training" in os.path.basename(f).lower() or "100K" in os.path.basename(f):
            return f
    if files:
        return files[0]
    raise FileNotFoundError("data/ 底下找不到訓練用 CSV")


def main(n: int = 200, out: str | None = None) -> str:
    path = find_training_csv()
    df = pd.read_csv(path)

    # 與訓練完全相同的 person 切分（seed 相同 → 結果可重現），取「測試集」
    _train, _val, test = feature_engineering.split_by_person(df)
    src = test
    if n and n < len(src):
        src = src.sample(n, random_state=config.SEED)

    out_df = pd.DataFrame()
    out_df["sample_id"] = src["sample_id"].to_numpy()
    out_df["person_id"] = src["person_id"].to_numpy()
    # 29 個模型特徵，用「短名」當欄位（api/excel_io.py 兩種寫法都認得）
    for short, csv_col in SHORT_TO_CSV.items():
        out_df[short] = src[csv_col].to_numpy()

    out_path = out or OUT_DEFAULT
    out_df.to_excel(out_path, index=False)

    print(f"測試集共 {len(test)} 列（person={test['person_id'].nunique()} 人）")
    print(f"匯出 {len(out_df)} 列 → {out_path}")
    print(f"欄位（{len(out_df.columns)} 欄）：sample_id, person_id + 29 個生理特徵")
    print("（無 emotion_label —— 要算準確率，用 sample_id 回原 CSV 對回標籤）")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="產生預測用 Excel（無情緒標籤）")
    ap.add_argument("--n", type=int, default=200, help="匯出列數；0 = 全部測試集")
    ap.add_argument("--out", type=str, default=None, help="輸出路徑")
    args = ap.parse_args()
    main(args.n, args.out)
