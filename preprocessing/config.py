# -*- coding: utf-8 -*-
"""
preprocessing/config.py — 資料欄位定義（**照實際 CSV 欄位**，非假設）

欄位來源：EEG_HRV_EDA_synthetic_training_100K_sample.xlsx 的 training_sample 工作表。
"""
import os

# 重現性
SEED = 42

# ---------------------------------------------------------------------------
# 特徵欄位（實際欄位名，共 29 個生理特徵）
# ---------------------------------------------------------------------------
EEG_FEATURES = [
    "eeg_delta_power", "eeg_theta_power", "eeg_alpha_power",
    "eeg_beta_power", "eeg_gamma_power",
    "eeg_relative_delta", "eeg_relative_theta", "eeg_relative_alpha",
    "eeg_relative_beta", "eeg_relative_gamma",
    "theta_beta_ratio", "alpha_beta_ratio", "alpha_theta_ratio",
]

HRV_FEATURES = [
    "heart_rate_bpm", "mean_rr_ms", "hrv_rmssd_ms", "hrv_sdnn_ms",
    "hrv_pnn50_percent", "hrv_lf_ms2", "hrv_hf_ms2", "hrv_lf_hf_ratio",
    "hrv_total_power_ms2",
]

EDA_FEATURES = [
    "eda_scl_uS", "eda_scr_amplitude_uS", "eda_scr_frequency_per_min",
    "eda_tonic_uS", "eda_phasic_uS", "eda_variability", "eda_response_latency_s",
]

FEATURE_COLUMNS = EEG_FEATURES + HRV_FEATURES + EDA_FEATURES   # 29 個

# ---------------------------------------------------------------------------
# 標籤 / 切分 / 品質
# ---------------------------------------------------------------------------
TARGET_COLUMN = "emotion_label"
PERSON_ID_COLUMN = "person_id"
QUALITY_COLUMN = "data_quality"
QUALITY_SCORE_COLUMN = "data_quality_score"

# 標籤順序（predict_proba 的輸出順序與此一致）
LABEL_ORDER = ["positive", "neutral", "negative"]

# 個人基線所需的重要特徵（用於 baseline 比較 + DeepSeek 解釋）
BASELINE_FEATURES = [
    "eeg_alpha_power", "eeg_beta_power", "eeg_theta_power",
    "hrv_rmssd_ms", "hrv_sdnn_ms", "hrv_lf_hf_ratio",
    "eda_scl_uS", "eda_scr_frequency_per_min",
]

# ---------------------------------------------------------------------------
# 要丟棄的欄位：ID／時間、非生理情境欄位、敏感欄位、會洩漏標籤的欄位、
# 以及資料自帶的「全域 z-score」（個人基線 z 由我們自行以 person_id 重算）。
# ---------------------------------------------------------------------------
DROP_COLUMNS = [
    "sample_id", "timestamp",
    "age", "sex", "sleep_hours", "exercise_level", "caffeine_level",
    "measurement_hour",                 # 情境/人口學欄位（非 EEG+HRV+EDA 核心）
    "clinical_status", "pattern_note", "physiological_pattern", "synthetic",
    "dataset_split",                    # 全為 train，改以 person_id 重切
    "eeg_alpha_zscore", "eeg_beta_zscore", "hrv_rmssd_zscore",
    "hrv_sdnn_zscore", "eda_scl_zscore",
]

# ---------------------------------------------------------------------------
# 路徑
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_100K_CSV = os.path.join(BASE_DIR, "data", "EEG_HRV_EDA_synthetic_training_100K.csv")
# 完整 1M 資料（使用者稍後提供時放到此路徑）
DATA_1M_CSV = os.path.join(BASE_DIR, "data", "EEG_HRV_EDA_synthetic_training_1M.csv")

# 切分比例（以 person_id 為單位）
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1

# ---------------------------------------------------------------------------
# DeepSeek（解釋層；API Key 由 api/deepseek.py 從 .env 讀取，不寫死在這裡）
# ---------------------------------------------------------------------------
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# ---------------------------------------------------------------------------
# 真實裝置（Arduino USB 串口；留空 = 停用，只用 ESP 無線）
# ---------------------------------------------------------------------------
SERIAL_PORT = os.getenv("SERIAL_PORT", "")      # 例 "COM3"；留空停用串口
SERIAL_BAUD = 115200
