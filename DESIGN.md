# EEG＋HRV＋EDA 多模態生理訊號分析系統 — 系統設計書（Phase 1，修訂版）

> 青少年情緒與心理狀態「輔助分析」研究作品
> 版本：Phase 1 設計稿（本階段不寫完整程式碼，只輸出架構與規格）
> 修訂：經多專家審查（26 項發現，0 critical / 10 major / 16 minor）全部修正後定稿

---

## 0. 定位聲明（防跑偏，全程遵守）

本系統的核心輸出**不是**「診斷某人患有某種心理疾病」，而是：

> 根據 EEG、HRV、EDA 的多模態生理訊號，對使用者**目前的情緒／心理生理狀態進行輔助分析**，並分類為 **POSITIVE（正面）/ NEUTRAL（中性）/ NEGATIVE（負面）**，同時輸出 **confidence（來自分類器）** 與 **data quality（資料品質）**。

三條鐵律（貫穿所有模組）：

1. **Python 算數值，DeepSeek 不解釋原始訊號、不猜數值、不編造 confidence。** DeepSeek 只做：解釋已算好的特徵、引用 RAG 研究、綜合三訊號、生成報告與**自我調節建議**。
2. **不把研究性分析描述成醫療診斷**；不使用「焦慮症」「憂鬱症」等疾病標籤；統一使用「自我調節建議」，不使用臨床導向的「介入」措辭。
3. **不單用任一訊號下結論**；必須綜合 EEG＋HRV＋EDA＋個人 Baseline＋時間趨勢＋資料品質。

> 用語規範：全文統一用「**自我調節建議**」（對應你規格 §19「介入建議」章節的實質內容），避免「介入」這類治療性用語，維持「輔助分析」定位。

### 與現有 V1 的關係（遷移說明）

| 現有 V1 檔案（Flask） | 新架構去處（FastAPI） |
|---|---|
| `app.py`（Flask + SocketIO 路由） | `backend/main.py` + `backend/api/routes/*`（FastAPI + WebSocket） |
| `ai_analysis.py`（rule_based / DeepSeek） | 拆成 `services/eeg_analyzer.py`、`hrv_analyzer.py`、`eda_analyzer.py`、`baseline.py`、`fusion.py`、`classifier.py`、`llm/*` |
| `storage.py`（單一 `readings` 表） | `backend/db/models.py`（11 張正規化資料表，SQLAlchemy） |
| `serial_reader.py` | `services/ingestion/serial_reader.py` |
| `templates/index.html` + `static/` | 前端 Dashboard（結構見 §9） |
| `config.py` | `backend/config.py` |
| `arduino/*`（感測節點、無線橋接） | **維持不變**（資料格式沿用現有 JSON） |

---

## 1. 完整系統架構圖

```
┌─────────────────────────────────────────────────────────────────────┐
│                          可穿戴裝置層                                │
│   NeuroSky TGAM (EEG) ｜ Pulse Sensor (HRV/PPG) ｜ Grove GSR (EDA)  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ 類比/串口
┌──────────────────────────────▼──────────────────────────────────────┐
│                       擷取層  Arduino Nano/Uno                       │
│   讀感測器 → 打包 JSON → 每秒輸出一筆（含該秒內所有 IBI 陣列）       │
└───────────────┬──────────────────────────────┬──────────────────────┘
                │ USB 串口                      │ TX→RX
┌───────────────▼──────────────┐  ┌─────────────▼────────────────────┐
│  有線：serial_reader          │  │  無線：ESP8266/ESP32 (WiFi)       │
│  (pyserial)                   │  │  HTTP POST /api/v1/ingest         │
└───────────────┬───────────────┘  └─────────────┬────────────────────┘
                └────────────────┬───────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend（核心）                         │
│                                                                      │
│ ① 資料接收 + 品質檢查  Signal Quality Check (EEG/HRV/EDA)            │
│ ② 分析引擎  EEG Analyzer │ HRV Analyzer │ EDA Analyzer               │
│            （filtering → artifact removal → 頻譜/時域 → 特徵）      │
│ ③ 特徵提取  保留原始值 + 衍生特徵 + 分級(Low/Normal/High)           │
│ ④ 個人 Baseline  mean/std/median/min/max → Z-score 比較              │
│ ⑤ 多模態融合  Feature Vector（z-score 化後拼接，early-fusion）       │
│ ⑥ 分類器  Rule-based / ML（輸出 3-class scores + confidence）        │
│ ⑦ 歷史趨勢分析  Daily / Weekly / Long-term                          │
│ ⑧ RAG 知識庫  Embedding → Pre-filter → Vector Search → Rerank       │
│ ⑨ DeepSeek 只做解釋（Tool Calling 取得 Python 算好的數值）           │
│ ⑩ 即時提醒 + 自我調節建議                                           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ REST / WebSocket
┌──────────────────────────────▼──────────────────────────────────────┐
│                  網站 Dashboard（即時 + 歷史 + 圖譜 + 建議）          │
│   目前情緒｜Confidence｜Data Quality｜EEG/HRV/EDA 圖｜趨勢｜Baseline  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. 資料流圖

```
感測器 ──► Arduino ──► ESP8266/ESP32 ──(WiFi JSON)──► POST /api/v1/ingest
                                                        │
                                           ┌────────────▼────────────┐
                                           │ 1. 資料接收 + 品質檢查   │
                                           │    EEG/HRV/EDA quality   │
                                           └────────────┬────────────┘
                                     品質不足 │          │ 品質通過
                         ┌───────────────────┘          ▼
                         ▼                     ┌──────────────────────┐
              predicted_state = NULL           │ 2. 分模組分析          │
              data_quality = LOW               │  EEG/HRV/EDA Analyzer │
              「目前訊號品質不足，              └──────────┬───────────┘
               暫時無法可靠分析」                         ▼
                                              ┌──────────────────────┐
                                              │ 3. 特徵提取（原始值保留）│
                                              │  alpha/beta/theta …    │
                                              │  RMSSD/SDNN/LF/HF …    │
                                              │  SCL/SCR/tonic/phasic …│
                                              └──────────┬───────────┘
                                                         ▼
                                              ┌──────────────────────┐
                                              │ 4. 個人 Baseline 比較  │
                                              │  z = (x - μ) / σ      │
                                              └──────────┬───────────┘
                                                         ▼
                                              ┌──────────────────────┐
                                              │ 5. 多模態融合(early)  │
                                              │  z-score 化特徵向量   │
                                              └──────────┬───────────┘
                                                         ▼
                                              ┌──────────────────────┐
                                              │ 6. 分類器             │
                                              │  predicted_state +    │
                                              │  confidence(+type)    │
                                              └──────┬──────────┬────┘
                                                     │          │
                                          ┌──────────▼──┐  ┌────▼──────────┐
                                          │ 7. 歷史趨勢 │  │ 8. RAG 知識庫 │
                                          │ daily/week  │  │  pre-filter→  │
                                          │ long-term   │  │  檢索相關研究 │
                                          └──────────┬──┘  └────┬──────────┘
                                                     │           │
                                                     ▼           ▼
                                          ┌──────────────────────────┐
                                          │ 9. DeepSeek（Tool Calling）│
                                          │   分類器結果已注入上下文   │
                                          │   綜合三訊號 + 研究 + 解釋 │
                                          └──────────┬───────────────┘
                                                     ▼
                                          ┌──────────────────────────┐
                                          │ 10. 報告 + 提醒 + 自我調節│
                                          │     WebSocket 推播到前端  │
                                          └──────────────────────────┘
```

> **品質不足的表示**：`predicted_state = NULL` 且 `data_quality = LOW`，前端顯示「目前訊號品質不足，暫時無法可靠分析」。`DATA_QUALITY_LOW` 是資料品質狀態，**不是**情緒狀態的其中一個值。

---

## 3. 資料庫 Schema

資料庫：SQLite（開發/比賽）→ 可換 PostgreSQL；ORM 用 SQLAlchemy。

> 命名規範：`hr ≡ heart_rate`（心率）、`ibi ≡ rr_interval`（心跳間隔，一律以 **ms** 儲存）。原始量測用簡稱（`hr`/`ibi`），衍生特徵用全稱（`heart_rate`/`rr_interval`），兩者等價。

### 3.1 users

| 欄位 | 型別 | 說明 |
|---|---|---|
| id | INT PK | 使用者 ID |
| alias | TEXT | 匿名代號（**不存真名/學號，符合隱私**） |
| created_at | DATETIME | 建立時間 |
| device_type | TEXT | 裝置型別（如 TGAM+Pulse+GSR） |

### 3.2 sensor_data（原始量測，一列＝一秒）

| 欄位 | 型別 | 說明 |
|---|---|---|
| id | INT PK | |
| user_id | FK→users | |
| timestamp | FLOAT/REAL | 接收時間（Unix 秒） |
| source | TEXT | `arduino` / `esp8266` |
| eeg_raw | JSON/REAL | EEG 原始值（512Hz raw 或 TGAM 頻帶原始值） |
| eeg_signal_quality | INT | TGAM 0–200（0 最好） |
| hr | REAL | 心率 bpm（≡ heart_rate） |
| ibi_list | JSON | **該秒內所有心跳間隔（ms）陣列**，供重建完整 RR 序列 |
| eda_raw | REAL | EDA 原始 ADC/μS |
| data_quality | TEXT | `OK` / `LOW` / 細部旗標 |

> **逐拍完整性**：Pulse Sensor 逐拍產生 IBI，HR>60bpm 時每秒多於一拍，故存 `ibi_list`（陣列）而非單一 `ibi`。HRV 特徵一律由跨秒組成的完整 RR 序列計算，避免 RMSSD/SDNN/pNN50 失真。

### 3.3 eeg_features

| 欄位 | 型別 | 說明 |
|---|---|---|
| id | INT PK | |
| user_id | FK | |
| timestamp | FLOAT | |
| sensor_data_id | FK | 對應原始量測 |
| delta_power / theta_power / alpha_power / beta_power / gamma_power | REAL | 絕對功率 |
| alpha_rel / beta_rel / theta_rel | REAL | 相對功率（%） |
| beta_alpha_ratio / theta_beta_ratio / alpha_theta_ratio | REAL | 比值 |
| attention / meditation | INT | TGAM 0–100 |
| alpha_level / beta_level / theta_level | TEXT | `Low`/`Normal`/`High`（對 Baseline） |
| z_scores | JSON | 各特徵的 z-score（feature → z），可選擇性落庫 |
| data_quality | TEXT | |

### 3.4 hrv_features

| 欄位 | 型別 | 說明 |
|---|---|---|
| id | INT PK | |
| user_id / timestamp / sensor_data_id | — | 同上 |
| heart_rate | REAL | bpm |
| rr_interval | REAL | 平均心跳間隔 ms |
| rmssd | REAL | ms |
| sdnn | REAL | ms |
| pnn50 | REAL | % |
| lf | REAL | ms² |
| hf | REAL | ms² |
| lf_hf | REAL | 比值 |
| z_scores | JSON | 各特徵 z-score |
| data_quality | TEXT | |

> **誠實原則**：不可算的指標其**數值欄位存 NULL**，並在 `data_quality` 標記 `Unavailable`；**不以字串填入 REAL 欄位、不填 0 冒充**。詳見 §4.2 各指標的窗長門檻。

### 3.5 eda_features

| 欄位 | 型別 | 說明 |
|---|---|---|
| id | INT PK | |
| user_id / timestamp / sensor_data_id | — | 同上 |
| scl | REAL | 窗內皮膚電導均值（μS） |
| tonic_level | REAL | Tonic 成分（逐時低通，μS） |
| phasic_component | REAL | Phasic 成分 = 訊號 − tonic（**可正可負**，μS） |
| scr_count | INT | SCR 次數 |
| scr_amplitude_mean | REAL | SCR 平均振幅（μS，正值） |
| scr_frequency | REAL | SCR 頻率（次/min） |
| eda_baseline | REAL | EDA 基線（μS） |
| eda_variation | REAL | EDA 變化量（μS） |
| z_scores | JSON | 各特徵 z-score |
| data_quality | TEXT | |

### 3.6 baseline（個人基線）

| 欄位 | 型別 | 說明 |
|---|---|---|
| id | INT PK | |
| user_id | FK | |
| signal | TEXT | `eeg` / `hrv` / `eda` |
| feature | TEXT | 特徵名（如 `alpha_power`、`rmssd`、`scl`） |
| mean / std / median / min / max | REAL | 統計量 |
| sample_count | INT | 基線樣本數 |
| updated_at | DATETIME | 更新時間 |

> **基線穩定門檻**：`sample_count` 需 ≥ 最小門檻（預設 30）；`std < ε`（如 1e-6，樣本恆定）時**不計算 z-score**，輸出「baseline insufficient」，避免除以零與不穩定分級。

### 3.7 emotion_results（情緒分類結果，核心表）

| 欄位 | 型別 | 說明 |
|---|---|---|
| id | INT PK | |
| user_id | FK | |
| timestamp | FLOAT | |
| positive_score | REAL | 正面分數 |
| neutral_score | REAL | 中性分數 |
| negative_score | REAL | 負面分數 |
| predicted_state | TEXT | `POSITIVE`/`NEUTRAL`/`NEGATIVE`，品質不足時 **NULL** |
| confidence | REAL | 由分類器輸出（**非 DeepSeek**） |
| confidence_type | TEXT | `rule_softmax`（規則版）/ `ml_proba`（ML 版） |
| data_quality | TEXT | `OK` / `LOW` |
| engine | TEXT | `rule-based` / `ml` |

> **confidence 來源明確**：
> - rule-based：3-class scores 經 **softmax** 後取最大值作為 confidence（相對信心，非統計機率）；另可記錄 top−second 的 margin 作輔助。
> - ML：`predict_proba` 的最大值（真機率）。
> - 兩者量綱不同，故以 `confidence_type` 區分，前端與 DeepSeek 不得把兩種當同一尺度比較。

### 3.8 historical_analysis（趨勢分析）

| 欄位 | 型別 | 說明 |
|---|---|---|
| id | INT PK | |
| user_id | FK | |
| period | TEXT | `daily` / `weekly` / `long_term` |
| start_ts / end_ts | FLOAT | 時間範圍 |
| summary | JSON | 各訊號趨勢 + 情緒趨勢摘要 |
| trend_sequence | JSON | 例如 `["POSITIVE","NEUTRAL","NEGATIVE",…]` |

### 3.9 knowledge_documents（RAG 知識片段）

| 欄位 | 型別 | 說明 |
|---|---|---|
| id | INT PK | |
| doc_id | TEXT UNIQUE | 知識片段唯一識別（如 `eda-scr-001`） |
| source_id | FK→research_sources | 引用文獻 |
| title | TEXT | |
| content | JSON | 知識內容（definition/interpretation/unit/population/limitations/keywords，見 §5.1） |
| signal / feature | TEXT | 供 metadata filter |
| metadata | JSON | 其他 metadata |
| embedding | BLOB/向量 | 供 FAISS/餘弦檢索（維度與 embedder 一致） |
| created_at | DATETIME | |

### 3.10 research_sources（研究文獻來源，正規化表）

| 欄位 | 型別 | 說明 |
|---|---|---|
| id | INT PK | |
| title / authors | TEXT | |
| source | TEXT | 期刊/會議 |
| year | INT | |
| doi | TEXT | |
| url | TEXT | |
| abstract | TEXT | |
| signal / feature | TEXT | 對應訊號/特徵 |
| evidence_level | TEXT | 證據等級 |

> **職責區分**：`research_sources` 存「文獻本體」；`knowledge_documents` 存「知識片段」並以 `source_id` 引用文獻。文獻中繼資料（source/year/doi/evidence_level）只存一份，避免兩表不同步。

### 3.11 analysis_logs（分析日誌）

| 欄位 | 型別 | 說明 |
|---|---|---|
| id | INT PK | |
| user_id | FK | |
| timestamp | FLOAT | |
| action | TEXT | 如 `classify`、`rag_search`、`deepseek_explain` |
| input / output | JSON | 輸入輸出 |
| engine | TEXT | |
| latency_ms | INT | 耗時 |
| error | TEXT | 錯誤（若有） |

---

## 4. EEG / HRV / EDA Feature Schema

> 記錄合約：每個特徵的 **raw_value = 特徵欄位本身**；**baseline_value 存於 baseline 表**（join）；**z_score 即時計算**，可選擇性落庫於各表 `z_scores` JSON 欄位；`timestamp` 隨列儲存。原始值永不丟棄。

### 4.0 時間窗與對齊規格（融合的基礎）

不同訊號的可靠分析窗長不同，融合以「固定節奏 + 取最近完成窗」對齊：

| 模組 | 分析窗 | 輸出頻率 |
|---|---|---|
| EEG | 1s（TGAM 每秒） | 1s |
| EDA（SCR/tonic） | 30s 滑窗 | 10s |
| HRV 短時（RMSSD/pNN50） | 30–60s | 30s |
| HRV 頻域（LF/HF/SDNN） | ≥5 分鐘（標準） | 5 分鐘 |

- **融合/分類節奏**：預設每 **10s** 執行一次，取各模組「≤ 該時間戳的最近完成窗」特徵。
- **未滿窗**：該特徵標 Unavailable（NULL）；融合時以 mask 遮罩。
- **整模組不可用**：不丟棄該筆，但標註「多模態不完整」並降低 confidence。

### 4.1 EEG 特徵

| 特徵 | 定義 | 單位 | 分級 |
|---|---|---|---|
| delta_power | 0.5–4 Hz 功率 | μV²（或相對單位） | — |
| theta_power | 4–8 Hz | 同上 | Low/Normal/High |
| alpha_power | 8–13 Hz | 同上 | Low/Normal/High |
| beta_power | 13–30 Hz | 同上 | Low/Normal/High |
| gamma_power | 30–50 Hz | 同上 | — |
| alpha_rel / beta_rel / theta_rel | 頻帶功率 ÷ 總功率 | % | — |
| beta_alpha_ratio | β / α | 比值 | — |
| theta_beta_ratio | θ / β | 比值 | — |
| alpha_theta_ratio | α / θ | 比值 | — |
| attention / meditation | TGAM 直接輸出 | 0–100 | — |

- **頻帶定義（重要）**：標準頻帶（上表）一律由 **raw EEG（512Hz）經 FFT/帶通重新計算**。NeuroSky TGAM 內建頻帶邊界與標準不同（delta 約 0.5–2.75、theta 約 3.5–6.75、alpha/beta 拆 low/high 兩段、gamma 拆 low/mid 且無單一 30–50 Hz）。若直接採用 TGAM 內建頻帶，必須**明確寫出映射關係**，並在 RAG 解釋時標註「頻帶定義與標準臨床定義不同」，避免特徵值與研究詮釋的系統性偏誤。
- **分級（Low/Normal/High）**：對個人 Baseline 的 z-score：`z < -1 → Low`，`-1 ≤ z ≤ 1 → Normal`，`z > 1 → High`。
- **研究性解釋方向（非診斷）**：Alpha 較明顯→放鬆/平靜參考；Beta 較高→專注/思考/較高心理喚醒參考（若伴隨其他壓力指標可作緊張/高喚醒參考）；Theta 變化→注意力/疲勞/認知狀態參考。

### 4.2 HRV 特徵

| 特徵 | 定義 | 單位 | 可靠窗長 |
|---|---|---|---|
| heart_rate | 60000 / RR_ms | bpm | 逐拍 |
| rr_interval | 相鄰心跳間隔 | ms | 逐拍 |
| rmssd | √(mean(ΔRR²)) | ms | 30–60s 即可 |
| pnn50 | NN50 ÷ 總數 | % | 30–60s 即可 |
| sdnn | std(RR) | ms | 長度依賴，≥5min 或 24h |
| lf | 0.04–0.15 Hz 功率 | ms² | ≥5min（標準） |
| hf | 0.15–0.4 Hz 功率 | ms² | ≥5min（標準） |
| lf_hf | LF ÷ HF | 比值 | ≥5min（標準） |

- **Unavailable 分級（重要修正）**：
  - **LF/HF**：Task Force 標準為 **5 分鐘**（2 分鐘僅最低門檻、需標註較不穩定）。
  - **SDNN**：具長度依賴性，標準 ≥5 分鐘或 24h。
  - **RMSSD / pNN50**：可在超短窗（30–60s）可靠計算，**不需**與 LF/HF 一併標 Unavailable。
  - 不可算者：數值欄位存 **NULL** + `data_quality` 標 `Unavailable`，不填 0、不假造。
- HRV 主要觀察：自律神經狀態、壓力相關變化、身體恢復、情緒穩定度。
- **限制**：HRV 受睡眠/運動/咖啡因/呼吸/時間/身體狀況影響，**不能單獨判病**。

### 4.3 EDA 特徵

| 特徵 | 定義 | 單位 |
|---|---|---|
| scl | 窗內皮膚電導**均值** | μS |
| tonic_level | 逐時低通成分 | μS |
| phasic_component | 訊號 − tonic（**可正可負**） | μS |
| scr_count | 皮電反應次數 | 次/窗 |
| scr_amplitude_mean | SCR 平均振幅（正值） | μS |
| scr_frequency | scr_count ÷ (窗長秒 ÷ 60) | 次/min |
| eda_baseline | EDA 基線 | μS |
| eda_variation | EDA 變化量 | μS |

- **名詞澄清**：SCL 與 tonic 是同源量——`scl` 取窗內均值、`tonic_level` 取逐時低通，兩者不重複計算；`phasic_component` 為殘差（可負），與正向的 SCR 振幅並列時清楚區分。
- **窗長固定**：EDA 分析窗固定 **60 秒**（或於 schema 明示窗長），使 `scr_frequency = scr_count / (window_sec / 60)` 可換算、不矛盾。
- EDA 反映 **Physiological Arousal（情緒喚醒程度）**。
- **限制**：EDA 不能區分焦慮/興奮/驚嚇/緊張/害怕，**必須與 EEG、HRV 一起分析**。

### 4.4 多模態融合（Feature Vector 規格）

> 目標：真正的特徵向量融合，**不是**「每個訊號各自投票再合併」。

- **融合方式：early-fusion**。所有特徵先做 **z-score 正規化**（對個人 Baseline），再**拼接成單一向量**送分類器。
- **正規化**：`z = (x - baseline_mean) / baseline_std`；無 Baseline 時用族群預設或標註。
- **特徵向量欄位（示例）**：
  - EEG：`alpha_z, beta_z, theta_z, beta_alpha_ratio, theta_beta_ratio`
  - HRV：`rmssd_z, sdnn_z, pnn50, lf_hf`
  - EDA：`scl_z, scr_frequency, tonic_slope`
  - 趨勢：`hrv_slope, eda_slope, emotion_slope`（近期斜率）
- **Unavailable 處理**：缺失特徵以 **mask 遮罩**（該維度排除並記錄 mask flag）；整模組缺失時標註「多模態不完整」、降低 confidence。
- **輸出**：融合向量 → `classify_emotion` → `{predicted_state, confidence, confidence_type, scores{positive/neutral/negative}}`。

---

## 5. RAG 知識庫 Schema

### 5.1 知識文件格式（沿用使用者指定）

```json
{
  "id": "eda-scr-001",
  "title": "Skin Conductance Response 與情緒喚醒",
  "signal": "EDA",
  "feature": "SCR",
  "definition": "…",
  "interpretation": "…",
  "unit": "μS",
  "population": "青少年 / 成人",
  "limitations": "無法單獨區分情緒種類",
  "evidence_level": "research",
  "source": "期刊/書籍",
  "year": 2025,
  "doi": "10.xxxx/xxxx",
  "keywords": ["EDA", "SCR", "arousal"]
}
```

> 文獻中繼資料（source/year/doi/evidence_level）正規化到 `research_sources` 表，`knowledge_documents` 以 `source_id` 引用，避免重複。

### 5.2 知識庫涵蓋主題（至少，對應使用者清單）

EEG 基礎、EEG Frequency Bands、Alpha、Beta、Theta、HRV、RMSSD、SDNN、LF、HF、LF/HF、EDA、SCL、SCR、Tonic/Phasic、Stress、焦慮相關生理研究、情緒喚醒、認知負荷、放鬆、疲勞、多模態生理訊號研究。

### 5.3 檢索管線（pre-filter 修正）

```
Query → Embedding → Metadata Pre-filter（先以 signal/feature 用 SQLite 篩出候選 id 集合）
      → Vector Search（在候選子集內做餘弦相似，FAISS 或記憶體索引）
      → Rerank（依相關度，evidence_level 作 tie-break）
      → Top-K → 提供給 DeepSeek
```

- **為何 pre-filter**：若先取 top-K 再依 signal/feature 過濾，相似度前 K 名若屬錯的訊號，過濾後可能剩極少甚至 0 筆，造成「有資料卻被丟棄」的假陰性。故**先以 metadata 篩出候選集合，再於集合內做向量檢索**。
- **Embedding**：預設本地 `sentence-transformers`（多語，離線可用）；可切 DeepSeek embedding。
- **誠實原則**：知識庫沒有相關資料時，DeepSeek 必須回答「目前知識庫沒有足夠資料支持這個結論」，不得捏造研究。

---

## 6. Tool Calling Schema

DeepSeek 只透過工具取得**Python 算好的數值**。以下 13 個工具，標 ★ 為 DeepSeek 可呼叫。

| 工具 | 觸發者 | Input | Output |
|---|---|---|---|
| `check_signal_quality` | 系統 | sensor_data_id | `{eeg, hrv, eda, overall}` |
| ★ `analyze_eeg` | 系統/DeepSeek | window_id | **讀取**已存 eeg_features（Python 已算好，不重算） |
| ★ `analyze_hrv` | 系統/DeepSeek | window_id | **讀取**已存 hrv_features |
| ★ `analyze_eda` | 系統/DeepSeek | window_id | **讀取**已存 eda_features |
| `calculate_baseline` | 系統 | user_id, signal | baseline 統計量 |
| `compare_baseline` | 系統 | feature, value | `{raw_value, baseline_value, z_score}` |
| `calculate_zscore` | 系統 | value, mean, std | z（σ<ε 時回傳 baseline insufficient） |
| `classify_emotion` | 系統 | feature_vector | `{predicted_state, confidence, confidence_type, scores{positive/neutral/negative}}` |
| ★ `get_historical_data` | DeepSeek | user_id, range | 歷史資料 |
| `get_emotion_trend` | 系統 | user_id, period | 趨勢序列 |
| ★ `search_knowledge_base` | DeepSeek | query, filters | Top-K 研究片段 |
| `search_research` | 系統/RAG | query | 文獻（解析/回填引用） |
| `generate_report` | 系統 | 分析結果 | 報告 |

> **關鍵約定**：
> - `analyze_*` 雖可由 DeepSeek 呼叫，但**只讀取資料庫中已由 Python 算好的特徵**（input 為 window_id），**不重新跑 DSP**——保證單一數值來源，避免「重算」與「已算好快取」兩套不一致。
> - **分類器結果注入**：`classify_emotion` 在 DeepSeek 之前由系統執行，其 `predicted_state` 與 `confidence` 直接**注入 DeepSeek 的 prompt 上下文**（「系統分類結果」區塊），DeepSeek 不需（也不得）自行呼叫計算。
> - `compare_baseline` 輸出欄位統一為 `raw_value / baseline_value / z_score`，與 §4 記錄合約一致。

---

## 7. DeepSeek System Prompt

> 核心 System Prompt（可微調語氣，但規則與格式必須保留）。

```
你是一個「EEG＋HRV＋EDA 生理訊號研究分析助手」。

你的任務是：
根據系統計算好的 EEG、HRV、EDA 特徵，以及 RAG 知識庫提供的研究資料，
分析使用者目前的生理狀態與情緒相關變化。

你不能直接從原始訊號猜測結果。
你必須使用系統提供的分析數據。

你必須遵守：
1. 不可以僅根據單一指標判斷情緒。
2. 必須綜合 EEG、HRV、EDA。
3. 優先比較個人 Baseline。
4. 必須考慮時間趨勢。
5. 必須考慮資料品質。
6. 不可以捏造數據。
7. 不可以捏造研究。
8. 不可以捏造 confidence。
9. 如果資料不足，必須明確說明。
10. 不可以把研究性分析描述成醫療診斷。
11. 不可以直接判斷使用者患有心理疾病。
12. 正面／中性／負面只是系統的情緒狀態分類。
13. 如果多個訊號結果互相矛盾，必須指出矛盾。
14. 所有研究性解釋都必須優先引用 RAG 提供的資料。
15. 【目前情緒狀態】與【分析信心】只能原樣引用系統注入的「分類器結果」
    （predicted_state 與 confidence），不得自行計算、修改或另做推斷；
    若你與分類器看法不同，以分類器為準。

回答格式：
【目前資料品質】
【EEG 分析】
【HRV 分析】
【EDA 分析】
【三種訊號綜合分析】
【目前情緒狀態】（原樣引用分類器 predicted_state）
【與個人 Baseline 比較】
【近期趨勢】
【分析信心】（原樣引用分類器 confidence，不自行估算）
【可能原因】
【自我調節建議】
【研究限制】
```

---

## 8. API Endpoint 設計

Base URL：`/api/v1`

| 方法 | 路徑 | 用途 | 查詢/參數 |
|---|---|---|---|
| POST | `/ingest` | 接收裝置資料（ESP 無線 / 串口匯入） | body: 量測 JSON |
| POST | `/analyze` | 手動觸發完整分析管線（回填/重跑） | body: window 範圍 |
| GET | `/emotion/current` | 目前情緒狀態 | `?user_id=` |
| GET | `/emotion/history` | 歷史情緒結果 | `?user_id=&start=&end=&page=&limit=` |
| GET | `/features/eeg` `/hrv` `/eda` | 各模組特徵 | `?user_id=&start=&end=` |
| GET / POST | `/baseline` | 查詢/建立個人基線 | `?user_id=&signal=` |
| GET | `/trends` | 日/週/長期趨勢 | `?user_id=&period=` |
| GET | `/quality` | 目前訊號品質 | `?user_id=` |
| POST | `/rag/search` | 知識庫檢索 | body: query + filters |
| POST | `/ai/explain` | DeepSeek 綜合解釋（Tool Calling） | body: window_id |
| POST | `/ai/chat` | AI 對話（帶數據上下文） | body: messages |
| WS | `/ws` | WebSocket 即時推播（完整路徑 `/api/v1/ws`） | — |

> **分析觸發器**：`/ingest` 內建背景排程——每當一個融合窗（預設 10s）完成，自動跑一次「品質→特徵→融合→分類」管線；`/analyze` 僅供手動重跑/回填。資料流圖的「自動連續」與 API 的「手動端點」因此一致。

---

## 9. 前端 Dashboard 結構

```
┌──────────────────────────────────────────────────────────────┐
│ 頂列：品牌 ｜ 連線狀態（無線/串口）｜ 訊號品質指示 ｜ 提醒    │
├──────────────────────────────────────────────────────────────┤
│ 主角區：目前情緒（正面/中性/負面）＋ Confidence ＋ Data Quality│
│        （品質不足時顯示「目前訊號品質不足，暫時無法可靠分析」）│
├───────────────┬───────────────┬──────────────────────────────┤
│ EEG 區塊       │ HRV 區塊       │ EDA 區塊                     │
│ 即時圖 + 頻帶  │ 即時圖 + RMSSD │ 即時圖 + SCL/SCR            │
├───────────────┴───────────────┴──────────────────────────────┤
│ Emotion Timeline（情緒時間軸）＋ Historical Records           │
├──────────────────────────────────────────────────────────────┤
│ Personal Baseline（個人基線）＋ Trend Analysis（日/週/長期）  │
├──────────────────────────────────────────────────────────────┤
│ AI 分析報告（DeepSeek 解釋）＋ AI 對話                        │
└──────────────────────────────────────────────────────────────┘
```

一眼可見：**EEG、HRV、EDA、目前情緒、Confidence、Data Quality、圖表、時間軸、歷史、基線、趨勢**。

---

## 10. AI 分析流程

```
1. 收到一筆/一窗資料
2. Signal Quality Check（EEG noise/電極、HRV 異常 RR、EDA 脫落/跳變）
   └─ 品質不足 → predicted_state=NULL + data_quality=LOW + 「目前訊號品質不足」
3. 分模組特徵提取（Python）：EEG / HRV / EDA（保留原始值）
4. 個人 Baseline 比較 → Z-score（σ 過小/樣本不足 → baseline insufficient）
5. 多模態融合（early-fusion，z-score 化後拼接）
6. 分類器（Rule-based softmax / ML predict_proba）→ predicted_state + confidence(+type)
7. 歷史趨勢分析（日/週/長期）
8. RAG 檢索（pre-filter → vector search → rerank）
9. DeepSeek 綜合解釋 + 自我調節建議（分類器結果已注入上下文）
10. 報告 + 即時提醒 + 自我調節建議 → WebSocket 推播前端
```

---

## 附錄：Phase 2 模組化編碼標註

Phase 2 將逐模組產出可運行的 FastAPI 程式碼。每個模組必須說明：

- **Input**：輸入資料與格式
- **Output**：輸出資料與格式
- **Processing**：處理步驟（演算法/公式）
- **Data Format**：資料結構（Pydantic schema）
- **Error Handling**：錯誤與品質不足處理
- **與其他模組的連接方式**：呼叫/被呼叫關係

模組順序：`signal_quality → eeg_analyzer → hrv_analyzer → eda_analyzer → baseline → fusion → classifier → trends → rag → llm(deepseek+tools) → api routes → dashboard`。

---

## 修訂記錄（本次審查修正）

多專家審查共提出 **26 項發現（0 critical / 10 major / 16 minor）**，經對抗式驗證**全部屬實**，本版已全數修正。主要修正：

1. **HRV 心率公式**：`60/RR` → `60000/RR_ms`（單位 ms 一致）。
2. **逐拍完整性**：sensor_data 改存 `ibi_list`（該秒內所有 IBI 陣列），HRV 由完整 RR 序列計算。
3. **TGAM 頻帶**：標準頻帶一律由 raw EEG(512Hz) 重算；採 TGAM 內建頻帶須明確映射並標註。
4. **DeepSeek 不編造 state/confidence**：分類器結果注入 prompt 上下文，新增規則 15（原樣引用，不得自行計算）。
5. **特徵記錄合約**：raw_value=欄位本身、baseline 存 baseline 表、z_score 即時算（可落 z_scores JSON）。
6. **多模態融合**：明確定義 early-fusion + z-score 正規化 + 特徵向量欄位清單 + mask 處理。
7. **confidence 來源**：新增 `confidence_type`（rule_softmax / ml_proba），明確定義公式。
8. **RAG 檢索順序**：改為 metadata pre-filter → vector search → rerank（避免 post-filter 假陰性）。
9. **時間窗對齊**：新增 §4.0 窗長/輸出頻率/對齊規格。
10. **用語統一**：「介入建議」→「自我調節建議」；`state`→`predicted_state`；`hr/ibi`≡`heart_rate/rr_interval` 別名標註。
11. **品質表示**：`predicted_state` 僅三類，品質不足 = `predicted_state=NULL` + `data_quality=LOW`。
12. **其餘**：HRV Unavailable 窗長分級、EDA scl/tonic/phasic 澄清、baseline σ=0 防護、knowledge_documents 補全欄位、research_sources 正規化關聯、analyze_* 改為讀快取、API 觸發器/參數補齊等。
