# 多模態生理訊號與 AI 青少年心情監測系統

基於 **EEG + HRV + EDA** 三種生理訊號，結合機器學習與 DeepSeek 分析青少年心情狀態（正面 / 中性 / 負面）的網頁監測系統。

> 澳門青少年科學家大會參賽專案

---

## 我要運行哪個檔案

```powershell
git clone https://github.com/kenhopython/mentally.git
cd mentally
python api/main.py
```

然後用瀏覽器開 **http://localhost:8001/**

**不用接 Arduino，網站一開就有資料在跑**（見下方「模擬裝置」）。

- **不要打 `0.0.0.0:8001`** —— 那是「監聽所有網卡」的意思，不是網址。
- 埠被佔用時，程式會自己印出佔用的 PID 和 `taskkill` 解法。

---

## 模擬裝置（沒有硬體也能展示）

啟動伺服器時會**自動開啟模擬裝置**（[api/simulator.py](api/simulator.py)），
每 5 秒產生一筆模擬生理資料，前端開頁就自動進入「即時監測」開始輪詢，
所有分頁（即時監測／生理訊號／個人分析／歷史紀錄）都會動起來。

模擬資料刻意維持**生理一致性與分布真實性**，避免一眼看穿是假的：

- **錨點從訓練資料算出來**，不是憑感覺寫的：
  `python -m models.build_simulator_anchors` 會算出三個情緒類別的特徵平均值與
  標準差，存成 [api/simulator_anchors.json](api/simulator_anchors.json)。
  （一開始是用手寫的「平靜 = 心率 65 / SCL 2.0」那種數字，結果落在訓練分布
  之外，模型不管餵什麼都只回 negative —— 展示時情緒永遠是負面。）
- 在三個情緒錨點之間做**平滑隨機漫步**，數值會慢慢漂移，不是固定值
- 雜訊尺度 = 訓練資料的實際標準差（`SIM_NOISE_SCALE`，預設 1.0）
- 心率與平均 RR 綁定：`mean_rr = 60000 / HR`
- `LF/HF` 由 `LF ÷ HF` 算出，不是獨立亂數

| 控制方式 | 說明 |
|---|---|
| 前端 `🧪 模擬裝置` 按鈕 | 啟動／停止模擬裝置 |
| `POST /api/simulate/start` | 啟動（body 可帶 `{"interval": 3}`） |
| `POST /api/simulate/stop` | 停止 |
| `GET /api/simulate/status` | 目前狀態與已產生筆數 |

**成本**：串流預設**完全不呼叫 DeepSeek**（`SIM_DEEPSEEK_EVERY=0`），
只跑本地 ML，開一整天也不會花錢。按「😌 平靜 / 🙂 正常 / 😰 壓力」
才會有完整的雙分類器投票與 AI 解釋。要讓串流每隔幾筆也做完整分析，
在 `.env` 設 `SIM_DEEPSEEK_EVERY=5`。

| 環境變數 | 預設 | 說明 |
|---|---|---|
| `SIM_AUTOSTART` | `1` | 開伺服器時自動啟動模擬裝置；設 `0` 關閉 |
| `SIM_INTERVAL` | `5` | 每幾秒產生一筆 |
| `SIM_DEEPSEEK_EVERY` | `0` | 每 N 筆做一次完整 AI 分析；`0` = 都不做 |
| `SIM_NOISE_SCALE` | `1.0` | 雜訊尺度；`1.0` = 與訓練資料相同的散佈程度 |
| `SIM_DRIFT` | `0.12` | 情緒變化快慢；`0.12` 約每分鐘換一次，`0.40` 約每 20 秒換一次 |

> **兩個「幾秒」是不一樣的東西**：`SIM_INTERVAL` 是**資料更新頻率**（每 5 秒一筆，
> 數值一直在動）；`SIM_DRIFT` 才是**情緒標籤多久換一次**。因為是平滑漂移，
> 預設大約每分鐘才會從「中性」變成「負面」—— 想讓展示時明顯看得出情緒在變，
> 把 `SIM_DRIFT` 調到 `0.40`。

> **模擬畫面的情緒比例會偏負面，這是誠實的結果**：這份合成資料的三個類別
> 分布高度重疊，模型驗證 accuracy 只有約 0.50（ROC-AUC 0.68）。
> 餵入真實分布的模擬資料，模型當然也就只有那個準度。
> 我們選擇讓它照實顯示，而不是把模擬資料調成「好看的」樣子。

> **真實裝置優先**：一旦有資料 `POST /api/ingest`，模擬裝置會**自動停止**，
> 免得模擬值在幾秒內蓋掉真實讀數。
>
> ⚠ 模擬資料一律標記 `source="simulated"`，前端顯示「⚠ 目前為「模擬資料」示範」，
> **不得當成真實量測或臨床數據**。

---

## 系統架構

```
Arduino Nano/Uno（感測節點，arduino/sensor_node/sensor_node.ino）
   ├─ Pulse Sensor  → 心率 + HRV
   ├─ Grove GSR     → 皮膚電 EDA
   └─ NeuroSky TGAM → EEG 腦電波
           │
           ├── 有線：USB 串口（api/device.py 的 SerialReader）
           ├── 無線：ESP8266/ESP32（arduino/wifi_bridge/wifi_bridge.ino）
           │            └─ HTTP POST /api/ingest
           └── 沒有硬體：api/simulator.py 模擬裝置（同管線，標記為模擬資料）
                                    │
                     FastAPI（api/main.py，port 8001）
                       ├─ ML 預測管線（api/prediction.py）
                       ├─ 個人基線（baseline/）
                       ├─ DeepSeek 投票 + 解釋（api/fusion.py, api/deepseek.py）
                       ├─ AI 對話（api/chat.py）
                       └─ 伺服前端（frontend/）
```

---

## 功能一覽（5 分頁 + 浮動對話框）

- **即時監測**：心情狀態、模型信心度、訊號品質、EEG 頻帶、個人基線摘要
  （開頁自動開始輪詢；`📡 即時監測` 可暫停）
- **生理訊號**：EEG / HRV / EDA 各指標明細與圖表
- **個人分析**：目前值 vs 個人基線（z-score）
- **AI 分析**：ML 結果、雙分類器投票、主要影響特徵、DeepSeek 解釋
- **歷史紀錄**：情緒分類比例與時間軸
- **💬 AI 對話框**（右下角浮動）：參考最新量測結果回答問題
- **📂 Excel 批量預測**：上傳生理特徵 Excel，每一列都做完整 ML + DeepSeek 分析並下載結果（測準確率用）

---

## Excel 批量預測（測模型準確率）

想驗證這個網站「預測準不準」，上傳一份**只有生理特徵、沒有情緒標籤**的 Excel，
系統逐列預測情緒 + 信心度。你拿結果 CSV 的 `sample_id` 回原始資料對回
`emotion_label`，就能算準確率。

**完整分析**：每一列都走與「即時接收資料」**完全相同**的管線 ——
ML 預測 → DeepSeek 投票 → 融合 → DeepSeek 解釋（不是只有 ML）。
所以上傳 Excel 跟接裝置一樣，每一筆都會同時呼叫 MLP 與 DeepSeek。

> **⚠ DeepSeek 不會改變預測標籤**。照融合規則，DeepSeek 永遠只是「第二意見」，
> 一致時提高信心度、分歧時打折並標記，**最終 emotion 一律以 ML 為準**。
> 所以加不加 AI 分析，準確率完全一樣；AI 的價值在於解釋與一致性檢查。
> CSV 因此另闢 `llm_emotion` / `model_agreement` 兩欄，你可以單獨算 DeepSeek 自己的準確率來對照。

**用法**：

1. 按控制列的「📂 上傳 Excel」，選一份 `.xlsx` / `.xls` / `.csv`。
2. 系統逐列預測 → 跳出結果視窗（總列數、三類分布、每列 ML 預測 + 🤖 DeepSeek 那一票），
   **並自動開始逐筆套用到主頁面**（即時監測／生理訊號／個人分析／AI 分析），不用手動按。
3. 結果視窗下方有 **📋 原始數據預覽**：顯示目前這一列的 29 個特徵值，依 EEG / HRV / EDA 分組。
   點表格任一列（或等自動播放）都會更新預覽與主頁面。
4. 「⬇ 下載預測結果 CSV」→ 匯出
   `sample_id, person_id, predicted_emotion, confidence, llm_emotion, model_agreement, quality`。
5. 關閉結果視窗後主頁面會繼續輪播上傳的資料；按「📡 即時監測」即可停掉 Excel 播放、回到模擬／即時資料。

| 環境變數 | 預設 | 說明 |
|---|---|---|
| `EXCEL_AI_WORKERS` | `4` | 同時打幾條 DeepSeek API（1–8）。**這是速度的關鍵**，見下方 |

**為什麼要做並行**：實測每列約 **5.7 秒**（投票 1.8s + 解釋 3.8s，瓶頸是等 DeepSeek 回應，
不是 CPU）。序列跑 300 列要等近半小時；改成執行緒池並行 4 條後，實測 10 列 **20.4 秒**、
300 列約 7 分鐘。`ThreadPoolExecutor.map` 會保持順序，所以結果列的次序不會亂。
每列會打 2 次 API（投票 + 解釋），資料越多越久也越貴 —— 這是你要求「每列都完整分析」的代價。

**產一份測試 Excel**（從測試集切出，無標籤）：

```powershell
python -m models.make_prediction_excel --n 300     # 300 列；--n 0 = 全部測試集
# → data/prediction_input_sample.xlsx（sample_id + person_id + 29 個特徵，無 emotion_label）
```

**算準確率**（命令列，不限列數，自動 join 真值）：

```powershell
python -m models.predict_excel --input data/prediction_input_sample.xlsx \
                               --output data/predictions.csv \
                               --truth data/EEG_HRV_EDA_synthetic_training_100K.csv
# → 印出 accuracy / 分類報告 / 混淆矩陣
```

網頁端點：`POST /predict/excel`（multipart 上傳，只收 `file`；每一列都做完整
ML + DeepSeek 分析，上限 5000 列）。命令列工具 `models/predict_excel.py` 仍走純 ML。

---

## 雙分類器情緒預測（ML ＋ DeepSeek 投票）

系統有**兩個分類器**對同一筆生理資料各投一票：

| 投票者 | 角色 | 檔案 |
|---|---|---|
| ① 機器學習模型 | 由 100K 合成資料訓練，輸出三類機率 | [api/prediction.py](api/prediction.py) |
| ② DeepSeek | 讀同一份特徵與 ML 機率，從三個標籤中選一個 | [api/deepseek.py](api/deepseek.py) `vote()` |

**融合規則**（[api/fusion.py](api/fusion.py)）：

| 情況 | 最終標籤 | 信心度 |
|---|---|---|
| 兩票一致 | 不變 | 上調 `conf + (1-conf)×0.15` |
| 兩票不一致 | **以 ML 為準**，標記 `model_agreement=false` | 下修 `conf × 0.85` |
| DeepSeek 棄權／逾時／品質不足 | 純 ML | 不調整 |

回傳欄位：`prediction.confidence`（融合後）、`confidence_ml`（原始 ML）、
`ml_emotion`、`llm_emotion`、`model_agreement`、`consensus`，以及 `llm_vote`。

> **為什麼不一致時不用 DeepSeek？** LLM 的輸出無法離線重現、也沒有校準過的機率。
> 把它當「第二意見」而非決策者，整份預測才保持可驗證。

**成本控制**：`vote()` 有 120 秒快取（特徵量化到小數 2 位）與最小呼叫間隔
（`DEEPSEEK_VOTE_MIN_INTERVAL`，預設 3 秒），裝置每秒串流也不會打爆 API。
`/predict/batch` 刻意不逐筆投票。

---

## AI 對話框

- 端點：`POST /api/chat`，body `{"messages": [{"role":"user","content":"…"}]}`
- 實作：[api/chat.py](api/chat.py)、提示詞 [prompts/deepseek_chat_prompt.txt](prompts/deepseek_chat_prompt.txt)

**上下文由伺服器自己組，不信任前端送來的數字**：`build_context()` 從
`api/device.latest` 取最新一筆**已算好**的結果（情緒、機率、信心度、雙模型一致性、
訊號品質、偏離個人基線最多的 6 個 z-score），對話助手看不到模型的原始輸入，
也就不可能自己捏造數值。

其他保護：
- 只送最後 10 則訊息，單則上限 800 字；`_clean()` 會過濾掉非 dict、非法 role、空訊息
- 提示詞明文禁止醫療診斷字眼；訊號品質不足時要求主動說明「參考價值有限」
- 使用者表達強烈負面情緒時，引導去找信任的大人／老師／輔導室
- 對話歷史存在瀏覽器 `sessionStorage`，關掉分頁即清空，不上傳

---

## 機器學習模型

29 個生理特徵（EEG 13 + HRV 9 + EDA 7）→ 三分類（positive / neutral / negative）。

| 模型 | 訓練檔 | 驗證 accuracy | ROC-AUC |
|---|---|---|---|
| Logistic Regression | models/train_logistic.py | 0.497 | 0.682 |
| Random Forest | models/train_random_forest.py | 0.503 | 0.675 |
| SVM（linear）| models/train_svm.py | 0.498 | 0.685 |
| SVM（rbf）| models/train_svm.py | 0.496 | 0.674 |
| **MLP 自建神經網路** | models/train_mlp.py | 0.475 | 0.652 |

> 三分類約 50% 是這份合成資料的天花板（`emotion_label` 是 `physiological_pattern`
> 加雜訊生出來的）；但 ROC-AUC 皆 > 0.5，代表訊號是真實存在的。

**重新訓練**：

```powershell
python -m models.model_comparison            # 全部模型一起比，選最佳存 best_model.json
python -m models.train_mlp                   # 只訓 MLP 神經網路
python -m models.build_simulator_anchors     # 重算模擬裝置的錨點（換資料集後要跑）
```

- 切分嚴格**按 person_id**（80/10/10），杜絕同一人跨 split 的資料洩漏
- 前處理：中位數補值 → 縮尾（0.5/99.5 百分位）→ StandardScaler
- MLP 類別不平衡用 `sample_weight` 平衡；刻意**不開** `early_stopping`
  （它會在 train 內部再切驗證集，造成同一人跨 split 的輕微洩漏）
- 模型存成 `trained_models/*.pkl`，`api/prediction.py` 直接載入 `best_model.json` 指定的模型

---

## 檔案說明

| 路徑 | 用途 |
|---|---|
| `api/main.py` | **唯一入口**：FastAPI 伺服器 + 所有端點 + 伺服前端 |
| `api/prediction.py` | ML 預測管線（特徵工程 → 品質 → 向量 → predict_proba → 基線 → 趨勢） |
| `api/features.py` | 特徵名稱對應與衍生特徵（relative / ratio） |
| `api/quality.py` | 訊號品質檢查 |
| `api/fusion.py` | ML ＋ DeepSeek 雙分類器投票融合 |
| `api/deepseek.py` | DeepSeek 解釋 + 投票（`explain()` / `vote()`） |
| `api/chat.py` | AI 對話 |
| `api/device.py` | 真實裝置資料映射 + 串口讀取 |
| `api/simulator.py` | 模擬裝置（沒有 Arduino 也能讓網站自己跑） |
| `api/simulator_anchors.json` | 由訓練資料算出的模擬錨點（`models/build_simulator_anchors.py` 產生） |
| `api/excel_io.py` | 讀取預測用 Excel/CSV，欄位映射（忽略標籤欄） |
| `api/trends.py` | 情緒趨勢統計 |
| `baseline/baseline.py` | 個人基線（z-score、線上更新） |
| `preprocessing/` | 資料清理 + 特徵工程（訓練用） |
| `models/` | 五個模型的訓練腳本 + 比較 |
| `models/make_prediction_excel.py` | 從測試集產生「預測用 Excel」（無情緒標籤） |
| `models/predict_excel.py` | 命令列批量預測 + 算準確率（可 join 真值） |
| `evaluation/metrics.py` | 評估指標（accuracy / f1 / roc_auc / 混淆矩陣） |
| `frontend/` | 網頁（index.html + css + js） |
| `prompts/` | DeepSeek 三份提示詞（解釋 / 投票 / 對話） |
| `arduino/` | 感測節點 + 無線橋接韌體 |
| `tests/` | 86 個單元測試（`pytest tests/`） |
| `trained_models/` | 已訓練模型與基線（`*.pkl`、`best_model.json`） |

---

## 安裝與啟動

```powershell
pip install -r requirements.txt        # 安裝套件
copy .env.example .env                  # 建立 .env（API Key 要自己填，見下一節）
python api/main.py                      # 啟動，開 http://localhost:8001/
```

> 沒有填 `DEEPSEEK_API_KEY` 也能跑 —— 網站照常運作、ML 預測照常輸出，
> 只是 AI 投票／解釋／對話會停用（回傳 `no_api_key`）。

**這個 repo 裡沒有附的大檔**（因為檔案太大，見 `.gitignore`）：

| 檔案 | 大小 | 用途 | 沒有它會怎樣 |
|---|---|---|---|
| `data/EEG_HRV_EDA_synthetic_training_100K.csv` | 39 MB | 訓練／重訓用資料集 | 只能跑預測，不能重訓 |
| `EEG_HRV_EDA_synthetic_training_100K_sample.xlsx` | 32 MB | 原始資料樣本 | 同上 |
| `trained_models/random_forest.pkl` | 646 MB | Random Forest 模型 | 不影響，網站用的是 `svm_linear` |
| `trained_models/baselines.json` | 5.6 MB | 個人基線 | 不影響，缺檔時自動跳過 |

**網站執行必需的模型檔有附在 repo 裡**（`best_model.json`、`svm_linear.pkl`、
`preprocessor.pkl`、`scaler.pkl`、`feature_config.json`），所以 clone 下來就能直接跑。

接上 Arduino：
- **有線**：把 `arduino/sensor_node/sensor_node.ino` 燒進 Nano/Uno，USB 接電腦，
  在 `.env` 加 `SERIAL_PORT=COM3`（改成你的 COM 口）。
- **無線（推薦）**：把 `arduino/wifi_bridge/wifi_bridge.ino` 燒進 ESP8266/ESP32
  （Arduino IDE 需安裝 **WiFiManager** 函式庫）。開機後連熱點 **`MoodSense-Setup`**
  設定 WiFi 與伺服器位址，設定會記住、不用重燒。

---

## 設定 DeepSeek API Key（要自己申請）

**這個專案不附任何 API Key，每個人都要用自己申請的。**
沒有它網站照樣跑得起來 —— ML 預測完全不受影響，只有 AI 投票／解釋／對話會停用
（回傳 `no_api_key`）。所以只想先看看網站長怎樣，可以跳過這一節。

### ① 申請一把金鑰

1. 到 <https://platform.deepseek.com> 註冊帳號
2. 左側選單 → **API keys** → **Create new API key**
3. 複製產生的字串（`sk-` 開頭的一長串英數字）
4. ⚠ 金鑰**只顯示一次**，關掉頁面就再也看不到，先貼到安全的地方

> DeepSeek 是付費 API，呼叫前需要先儲值。

### ② 放進專案根目錄的 `.env`

金鑰要放在**專案根目錄**（跟 `README.md`、`api/` 同一層）一個叫 `.env` 的檔案裡。
這個檔案不在 repo 裡，要自己建立：

```
你的專案資料夾\
├─ .env            ← 金鑰放這裡（自己建立）
├─ .env.example    ← 範本，照著複製
├─ api\
├─ frontend\
└─ README.md
```

在專案根目錄執行：

```powershell
copy .env.example .env
```

然後用文字編輯器打開 `.env`，把 `DEEPSEEK_API_KEY` 那一行改成你的金鑰：

```ini
DEEPSEEK_API_KEY=sk-貼上你的金鑰
```

| 注意事項 | 說明 |
|---|---|
| 等號兩邊不加空格 | `DEEPSEEK_API_KEY=sk-xxx` ✅　`DEEPSEEK_API_KEY = sk-xxx` ❌ |
| 值不加引號 | 直接寫，不要包 `"` 或 `'` |
| 檔名開頭有一個點 | 是 `.env`，不是 `env`，也不是 `.env.txt` |
| 存成 UTF-8 | Windows 記事本預設編碼可能不是 |

### ③ 重啟伺服器

`.env` 只在**啟動時讀一次**。改完要關掉伺服器（`Ctrl+C`）再重新 `python api/main.py`。

### ④ 確認有沒有生效

```powershell
curl http://localhost:8001/health
```

回傳裡看到 `"deepseek_configured": true` 就代表讀到了。
如果是 `false`，回頭檢查檔名是不是 `.env`、位置是不是在專案根目錄。

也可以在網頁按「😌 平靜」，看「AI 分析」分頁有沒有出現解釋文字。

### ⚠ 三件絕對不要做的事

1. **不要把 `.env` 上傳 GitHub** —— 它已經列在 `.gitignore` 裡，不要把它從裡面拿掉
2. **不要把金鑰寫進任何 `.py` 檔** —— 程式是從 `.env` 讀的，寫死等於直接公開
3. **不要把金鑰貼給任何人** —— 金鑰等於你的帳號額度，外流會被盜刷

> 金鑰不小心外洩時，**光是刪檔案沒有用**：要登入 <https://platform.deepseek.com>
> 的 API keys 頁面把舊金鑰 **revoke（作廢）**，再建一把新的填回 `.env`。

---

## 隱私設計

- **本地優先**：感測資料寫在本地，不上傳第三方。
- **匿名化**：只存生理數值，不存姓名/學號等個資。
- **金鑰不外洩**：`DEEPSEEK_API_KEY` 放在 `.env`（已被 `.gitignore` 排除），
  **不要**寫進任何 `.py` 原始碼。
