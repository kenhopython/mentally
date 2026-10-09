/* 前端邏輯 — 研究型多模態生理訊號分析平台
   只呼叫現有後端 API，不修改後端。 */
const API = "";

const EMOTION = {
  positive: { label: "正面", icon: "😊", cls: "positive", color: "var(--green)" },
  neutral:  { label: "中性", icon: "😐", cls: "neutral", color: "var(--blue)" },
  negative: { label: "負面", icon: "😟", cls: "negative", color: "var(--orange)" },
};

const FEATURE_LABELS = {
  delta: "Delta (δ)", theta: "Theta (θ)", alpha: "Alpha (α)", beta: "Beta (β)", gamma: "Gamma (γ)",
  relative_delta: "相對 δ", relative_theta: "相對 θ", relative_alpha: "相對 α",
  relative_beta: "相對 β", relative_gamma: "相對 γ",
  theta_beta_ratio: "θ/β 比值", alpha_beta_ratio: "α/β 比值", alpha_theta_ratio: "α/θ 比值",
  heart_rate: "心率", mean_rr: "平均 RR", rmssd: "RMSSD", sdnn: "SDNN", pnn50: "pNN50",
  lf: "LF", hf: "HF", lf_hf_ratio: "LF/HF", total_power: "總功率",
  scl: "SCL", scr_amplitude: "SCR 振幅", scr_frequency: "SCR 頻率",
  tonic: "Tonic", phasic: "Phasic", eda_variability: "EDA 變異", eda_latency: "EDA 潛伏期",
};
const UNITS = { heart_rate: "bpm", mean_rr: "ms", rmssd: "ms", sdnn: "ms", pnn50: "%",
  lf: "ms²", hf: "ms²", total_power: "ms²", scl: "μS", scr_amplitude: "μS",
  scr_frequency: "次/分", tonic: "μS", phasic: "μS", eda_variability: "μS", eda_latency: "s" };
const CAT = { within_baseline: ["正常", "within"], slightly_above: ["略高", "above"],
  significantly_above: ["顯著高", "above"], slightly_below: ["略低", "below"],
  significantly_below: ["顯著低", "below"], insufficient: ["資料不足", "insufficient"] };

const EEG_ORDER = ["delta", "theta", "alpha", "beta", "gamma", "relative_delta", "relative_theta",
  "relative_alpha", "relative_beta", "relative_gamma", "theta_beta_ratio", "alpha_beta_ratio", "alpha_theta_ratio"];
const HRV_ORDER = ["heart_rate", "mean_rr", "rmssd", "sdnn", "pnn50", "lf", "hf", "lf_hf_ratio", "total_power"];
const EDA_ORDER = ["scl", "scr_amplitude", "scr_frequency", "tonic", "phasic", "eda_variability", "eda_latency"];
const BASELINE_ORDER = ["delta", "theta", "alpha", "beta", "gamma", "theta_beta_ratio", "alpha_beta_ratio", "alpha_theta_ratio",
  "heart_rate", "mean_rr", "rmssd", "sdnn", "pnn50", "lf", "hf", "lf_hf_ratio", "total_power",
  "scl", "scr_amplitude", "scr_frequency", "tonic", "phasic", "eda_variability", "eda_latency"];

const PRESETS = {
  calm: { user_id: "U00001",
    eeg: { delta: 0.10, theta: 0.18, alpha: 0.42, beta: 0.14, gamma: 0.06 },
    hrv: { heart_rate: 65, mean_rr: 923, rmssd: 55, sdnn: 62, pnn50: 45, lf: 400, hf: 500, lf_hf_ratio: 0.8, total_power: 1200 },
    eda: { scl: 2.0, scr_amplitude: 0.3, scr_frequency: 1.5, tonic: 1.8, phasic: 0.2, eda_variability: 0.3, eda_latency: 2.5 } },
  normal: { user_id: "U00001",
    eeg: { delta: 0.12, theta: 0.21, alpha: 0.34, beta: 0.25, gamma: 0.08 },
    hrv: { heart_rate: 78, mean_rr: 769, rmssd: 32, sdnn: 41, pnn50: 18, lf: 500, hf: 300, lf_hf_ratio: 1.67, total_power: 1000 },
    eda: { scl: 3.2, scr_amplitude: 0.8, scr_frequency: 5.2, tonic: 2.8, phasic: 0.4, eda_variability: 0.5, eda_latency: 2.1 } },
  stress: { user_id: "U00001",
    eeg: { delta: 0.08, theta: 0.26, alpha: 0.20, beta: 0.42, gamma: 0.12 },
    hrv: { heart_rate: 105, mean_rr: 571, rmssd: 18, sdnn: 22, pnn50: 5, lf: 800, hf: 120, lf_hf_ratio: 6.5, total_power: 950 },
    eda: { scl: 7.5, scr_amplitude: 1.6, scr_frequency: 9.5, tonic: 6.8, phasic: 0.9, eda_variability: 1.2, eda_latency: 1.2 } },
};

let lastResult = null;
let sessionHistory = [];   // 本次 session 的預測時間軸

function cvar(n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }
function setText(id, t) { const e = document.getElementById(id); if (e) e.textContent = t; }
function fmt(v, d) { return (v == null) ? "—" : Number(v).toFixed(d ?? 2); }
function zOf(cmp, k) { return (cmp[k] && cmp[k].z_score != null) ? cmp[k].z_score : null; }

/* ---------- Chart.js ---------- */
Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
const gridOpt = { responsive: true, maintainAspectRatio: false, animation: false,
  plugins: { legend: { display: false } },
  scales: { x: { grid: { display: false }, ticks: { color: cvar("--text-3"), font: { size: 11 } } },
            y: { grid: { color: cvar("--line") }, ticks: { color: cvar("--text-3"), font: { size: 11 } } } } };
const chartEEG = new Chart(document.getElementById("chart-eeg"), {
  type: "bar", data: { labels: ["δ", "θ", "α", "β", "γ"],
    datasets: [{ data: [0, 0, 0, 0, 0], backgroundColor: cvar("--blue"), borderRadius: 4 }] }, options: gridOpt });
const chartBaseline = new Chart(document.getElementById("chart-baseline"), {
  type: "bar", data: { labels: [], datasets: [{ data: [], backgroundColor: [], borderRadius: 4 }] },
  options: { ...gridOpt, indexAxis: "y",
    scales: { x: { grid: { color: cvar("--line") }, ticks: { color: cvar("--text-3") } },
              y: { grid: { display: false }, ticks: { color: cvar("--text-3") } } } } });
const chartHistory = new Chart(document.getElementById("chart-history"), {
  type: "bar", data: { labels: ["正面", "中性", "負面"],
    datasets: [{ data: [0, 0, 0], backgroundColor: [cvar("--green"), cvar("--blue"), cvar("--orange")], borderRadius: 4 }] },
  options: gridOpt });

/* ---------- 分頁 ---------- */
document.querySelectorAll(".nav-btn").forEach(btn => btn.addEventListener("click", () => {
  document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
  document.getElementById("panel-" + btn.dataset.tab).classList.add("active");
}));
document.querySelectorAll("[data-goto]").forEach(btn => btn.addEventListener("click", () => {
  document.querySelector(`.nav-btn[data-tab="${btn.dataset.goto}"]`).click();
}));
document.querySelectorAll(".range-btn").forEach(btn => btn.addEventListener("click", () => {
  document.querySelectorAll(".range-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  renderHistoryChart(btn.dataset.range);
}));

/* ---------- 渲染：即時監測 ---------- */
function renderMonitor(d) {
  const lowQuality = d.data_quality.status === "LOW";
  const em = EMOTION[d.prediction.emotion] || EMOTION.neutral;
  const hero = document.getElementById("hero");
  hero.className = "hero " + em.cls;

  if (lowQuality) {
    setText("emotion-icon", "⚠"); setText("emotion-text", "無法可靠分析");
    setText("emotion-sub", "訊號品質不足，請重新測量。");
    setText("hero-confidence", "—");
  } else {
    setText("emotion-icon", em.icon); setText("emotion-text", em.label);
    setText("emotion-sub", `機率 正面 ${Math.round(d.prediction.probabilities.positive * 100)}% · 中性 ${Math.round(d.prediction.probabilities.neutral * 100)}% · 負面 ${Math.round(d.prediction.probabilities.negative * 100)}%`);
    setText("hero-confidence", Math.round(d.prediction.confidence * 100) + "%");
  }
  setText("hero-quality", d.data_quality.status + " " + Math.round(d.data_quality.score * 100) + "%");
  renderConsensusChip(d);

  // 基線 strip
  const cmp = d.baseline.comparison || {};
  const strip = document.getElementById("baseline-strip");
  strip.innerHTML = "";
  [["EDA", "scl"], ["HRV", "rmssd"], ["Alpha", "alpha"], ["Beta", "beta"]].forEach(([name, k]) => {
    const z = zOf(cmp, k);
    if (z == null) return;
    const s = document.createElement("span");
    s.className = z > 0.8 ? "up" : (z < -0.8 ? "down" : "ok");
    s.textContent = `${name} ${z > 0.8 ? "↑" : (z < -0.8 ? "↓" : "≈")} ${Math.abs(z).toFixed(1)} SD`;
    strip.appendChild(s);
  });

  // 喚醒提示
  const sclZ = zOf(cmp, "scl"), rmssdZ = zOf(cmp, "rmssd");
  const arousal = document.getElementById("arousal-note");
  if (lowQuality) arousal.textContent = "";
  else if ((sclZ != null && sclZ > 1) || (rmssdZ != null && rmssdZ < -1))
    arousal.textContent = "⚠ 目前生理喚醒程度較平時高，建議休息或進行 1–2 分鐘緩慢呼吸。";
  else arousal.textContent = "目前訊號與個人基線接近，狀態平穩。";

  // KPI
  const s = d.signals || {};
  setText("kpi-hr", fmt((s.hrv || {}).heart_rate, 0));
  setText("kpi-rmssd", fmt((s.hrv || {}).rmssd, 1));
  setText("kpi-scl", fmt((s.eda || {}).scl, 2));
  setText("kpi-hr-sub", baselineSub(cmp.heart_rate));
  setText("kpi-rmssd-sub", baselineSub(cmp.rmssd));
  setText("kpi-scl-sub", baselineSub(cmp.scl));

  // EEG 頻帶
  renderBandBars(s.eeg || {});
  // 基線摘要
  renderBaselineSummary(cmp);
  // AI 摘要
  renderAIBrief(d);
}

/* 雙分類器一致性徽章（ML ＋ DeepSeek 投票） */
function renderConsensusChip(d) {
  const el = document.getElementById("consensus-chip");
  const p = d.prediction || {};
  const llm = (d.llm_vote || {}).label;
  el.className = "consensus-chip";
  if (p.consensus === "high") {
    el.classList.add("agree");
    el.textContent = `✓ 雙模型一致（ML ＋ DeepSeek 皆為 ${EMOTION[p.emotion]?.label || p.emotion}）`;
  } else if (p.consensus === "low") {
    el.classList.add("disagree");
    const l = EMOTION[llm]?.label || llm;
    el.textContent = `⚠ 模型分歧：DeepSeek 認為是「${l}」，最終採用 ML 結果（信心度已下修）`;
  } else {
    el.classList.add("ml-only");
    const why = (d.llm_vote || {}).status;
    el.textContent = why === "skipped"
      ? "僅 ML 投票（訊號品質不足，未呼叫 DeepSeek）"
      : (why === "deferred"
      ? "僅 ML 投票（即時串流中，未呼叫 DeepSeek）"
      : "僅 ML 投票（DeepSeek 未參與本次預測）");
  }
}

function baselineSub(c) {
  if (!c || c.z_score == null) return "—";
  const [lbl] = CAT[c.category] || ["—"];
  return (c.z_score > 0.8 ? "↑ " : (c.z_score < -0.8 ? "↓ " : "≈ ")) + lbl;
}

function renderBandBars(eeg) {
  const el = document.getElementById("band-bars");
  el.innerHTML = "";
  [["alpha", "Alpha"], ["beta", "Beta"], ["theta", "Theta"], ["delta", "Delta"]].forEach(([k, label]) => {
    const pct = Math.round((eeg["relative_" + k] ?? 0) * 100);
    const row = document.createElement("div");
    row.className = "band-row";
    row.innerHTML = `<span class="band-label">${label}</span>` +
      `<div class="band-track"><div class="band-fill" style="width:${pct}%"></div></div>` +
      `<span class="band-pct">${pct}%</span>`;
    el.appendChild(row);
  });
}

function renderBaselineSummary(cmp) {
  const el = document.getElementById("baseline-summary");
  el.innerHTML = "";
  [["HRV", "rmssd"], ["EDA", "scl"], ["Alpha", "alpha"]].forEach(([name, k]) => {
    const c = cmp[k];
    if (!c) return;
    const row = document.createElement("div");
    row.className = "bs-row";
    const arrow = c.z_score > 0.8 ? "↑" : (c.z_score < -0.8 ? "↓" : "≈");
    const cls = c.z_score > 0.8 ? "up" : (c.z_score < -0.8 ? "down" : "ok");
    row.innerHTML = `<span class="bs-name">${name}</span><span class="${cls}">${arrow} ${CAT[c.category][0]}</span>`;
    el.appendChild(row);
  });
}

function renderAIBrief(d) {
  const el = document.getElementById("ai-brief");
  const ds = d.deepseek || {};
  const summary = ds.summary || (ds.status === "unavailable" ? "（DeepSeek 未設定或不可用）" : "");
  el.innerHTML = "";
  const p = document.createElement("p");
  p.textContent = summary;
  el.appendChild(p);

  const ul = document.createElement("ul");
  ul.className = "key-features";
  topFeatures(d.baseline.comparison || {}, 3).forEach(([k]) => {
    const li = document.createElement("li");
    const c = d.baseline.comparison[k];
    li.textContent = `• ${FEATURE_LABELS[k] || k} ${c.z_score > 0 ? "高於" : "低於"}個人基線`;
    ul.appendChild(li);
  });
  el.appendChild(ul);
}

/* ---------- 渲染：生理訊號 ---------- */
function renderSignals(d) {
  const s = d.signals || {};
  const e = s.eeg || {};
  chartEEG.data.datasets[0].data = [e.delta || 0, e.theta || 0, e.alpha || 0, e.beta || 0, e.gamma || 0];
  chartEEG.update();
  renderKV("signals-eeg", e, EEG_ORDER);
  renderKV("signals-hrv", s.hrv || {}, HRV_ORDER);
  renderKV("signals-eda", s.eda || {}, EDA_ORDER);
}
function renderKV(elId, group, order) {
  const el = document.getElementById(elId);
  el.innerHTML = "";
  order.forEach(k => {
    const v = group[k];
    if (v == null) return;
    const row = document.createElement("div");
    row.className = "kv-row";
    const u = UNITS[k] ? " " + UNITS[k] : "";
    row.innerHTML = `<span class="k">${FEATURE_LABELS[k] || k}</span><span class="v">${fmt(v)}${u}</span>`;
    el.appendChild(row);
  });
}

/* ---------- 渲染：個人分析 ---------- */
function renderPersonal(d) {
  setText("baseline-status", d.baseline.status === "READY" ? "基線就緒" : "基線不足");
  const cmp = d.baseline.comparison || {};
  const tbody = document.getElementById("baseline-tbody");
  tbody.innerHTML = "";
  BASELINE_ORDER.forEach(k => {
    const c = cmp[k];
    if (!c) return;
    const [lbl, cls] = CAT[c.category] || ["—", "insufficient"];
    const z = c.z_score == null ? "—" : (c.z_score > 0 ? "+" : "") + c.z_score.toFixed(2);
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${FEATURE_LABELS[k] || k}</td><td>${fmt(c.raw)}</td>` +
      `<td>${fmt(c.baseline_mean)}</td>` +
      `<td>${z}</td><td><span class="cat ${cls}">${lbl}</span></td>`;
    tbody.appendChild(tr);
  });
  // 基線圖（關鍵特徵）
  const keys = ["alpha", "beta", "theta", "rmssd", "sdnn", "lf_hf", "scl", "scr_frequency"];
  const lbl = { alpha: "Alpha", beta: "Beta", theta: "Theta", rmssd: "RMSSD", sdnn: "SDNN", lf_hf: "LF/HF", scl: "SCL", scr_frequency: "SCR 頻率" };
  const have = keys.filter(k => cmp[k] && cmp[k].z_score != null);
  chartBaseline.data.labels = have.map(k => lbl[k]);
  chartBaseline.data.datasets[0].data = have.map(k => cmp[k].z_score);
  chartBaseline.data.datasets[0].backgroundColor = have.map(k => cmp[k].z_score >= 0 ? cvar("--orange") : cvar("--blue"));
  chartBaseline.update();
}

/* ---------- 渲染：AI 分析 ---------- */
function renderAI(d) {
  const em = EMOTION[d.prediction.emotion] || EMOTION.neutral;
  const mr = document.getElementById("model-result");
  mr.innerHTML = "";
  [["模型", d.model.name], ["Prediction", em.label], ["Confidence", Math.round(d.prediction.confidence * 100) + "%"]].forEach(([l, v]) => {
    const item = document.createElement("div");
    item.className = "mr-item";
    item.innerHTML = `<div class="mr-label">${l}</div><div class="mr-value">${v}</div>`;
    mr.appendChild(item);
  });

  renderVoteCard(d);

  // 主要影響特徵（相對個人基線，來自後端 z-score）
  const fl = document.getElementById("feature-list");
  fl.innerHTML = "";
  const tops = topFeatures(d.baseline.comparison || {}, 5);
  const maxZ = Math.max(...tops.map(([, v]) => Math.abs(v.z_score)), 1);
  tops.forEach(([k, v]) => {
    const row = document.createElement("div");
    row.className = "feature-row";
    const w = Math.round(Math.abs(v.z_score) / maxZ * 100);
    const dir = v.z_score > 0 ? "↑ 高於基線" : "↓ 低於基線";
    row.innerHTML = `<span class="f-name">${FEATURE_LABELS[k] || k}</span>` +
      `<div class="f-track"><div class="f-fill" style="width:${w}%"></div></div>` +
      `<span class="f-val">${dir} ${Math.abs(v.z_score).toFixed(1)} SD</span>`;
    fl.appendChild(row);
  });

  // DeepSeek 完整
  const ds = d.deepseek || {};
  setText("deepseek-status", ds.status === "ok" ? "DeepSeek" : (ds.status || ""));
  setText("ai-summary", ds.summary || ds.analysis || "");
  setText("ai-eeg", ds.eeg_analysis || ""); setText("ai-hrv", ds.hrv_analysis || "");
  setText("ai-eda", ds.eda_analysis || ""); setText("ai-multimodal", ds.multimodal_analysis || "");
  setText("ai-baseline", ds.baseline_analysis || ""); setText("ai-trend", ds.trend_analysis || "");
  setText("ai-possible", ds.possible_state || ""); setText("ai-limitations", ds.limitations || "");
  const ul = document.getElementById("ai-suggestions");
  ul.innerHTML = "";
  (ds.self_regulation || []).forEach(s => { const li = document.createElement("li"); li.textContent = s; ul.appendChild(li); });
}

/* 雙分類器投票卡：兩票 + 最終結果 */
function renderVoteCard(d) {
  const p = d.prediction || {};
  const v = d.llm_vote || {};
  const nameOf = k => (EMOTION[k]?.label || k || "—");

  const mlLabel = p.ml_emotion || p.emotion;
  document.getElementById("vote-ml-label").textContent = nameOf(mlLabel);
  setText("vote-ml-sub", `信心度 ${Math.round((p.confidence_ml ?? p.confidence) * 100)}%`);

  const llmBox = document.getElementById("vote-llm");
  llmBox.className = "vote-box";
  if (v.label) {
    document.getElementById("vote-llm-label").textContent = nameOf(v.label);
    setText("vote-llm-sub", `自評信心 ${Math.round((v.llm_confidence || 0) * 100)}%`
      + (v.reason ? ` · ${v.reason}` : "") + (v.cached ? "（快取）" : ""));
    llmBox.classList.add(v.label === mlLabel ? "agree" : "disagree");
  } else {
    document.getElementById("vote-llm-label").textContent = "未投票";
    setText("vote-llm-sub", v.status === "throttled" ? "呼叫過於頻繁，本筆跳過"
      : (v.status === "skipped" ? "訊號品質不足"
      : (v.status === "deferred" ? "即時串流中跳過以節省 API（按情境按鈕可得完整分析）"
      : "DeepSeek 無法使用")));
    llmBox.classList.add("abstain");
  }

  const finalBox = document.getElementById("vote-final");
  finalBox.className = "vote-box";
  document.getElementById("vote-final-label").textContent = nameOf(p.emotion);
  const agree = p.model_agreement;
  setText("vote-final-sub", agree === true ? "雙模型一致，信心度上調"
    : (agree === false ? "模型分歧，以 ML 為準，信心度下修" : "僅 ML 投票"));
  finalBox.classList.add(agree === true ? "agree" : (agree === false ? "disagree" : "abstain"));

  setText("vote-status", agree === true ? "一致" : (agree === false ? "分歧" : "單一模型"));
}

function topFeatures(cmp, n) {
  return Object.entries(cmp).filter(([, v]) => v && v.z_score != null)
    .sort((a, b) => Math.abs(b[1].z_score) - Math.abs(a[1].z_score)).slice(0, n);
}

/* ---------- 渲染：歷史 ---------- */
function renderHistory() {
  const tl = document.getElementById("timeline");
  tl.innerHTML = "";
  if (!sessionHistory.length) {
    tl.innerHTML = `<div class="note">尚無本次預測紀錄，點上方範例按鈕開始。</div>`;
    return;
  }
  sessionHistory.slice().reverse().forEach(h => {
    const row = document.createElement("div");
    row.className = "tl-row";
    const em = EMOTION[h.emotion] || EMOTION.neutral;
    row.innerHTML = `<span class="tl-time">${h.time}</span>` +
      `<span class="tl-dot ${h.emotion}"></span><span>${em.label}</span>` +
      `<span class="note" style="margin:0">${h.preset}</span>`;
    tl.appendChild(row);
  });
  renderHistoryChart(document.querySelector(".range-btn.active").dataset.range);
}

function renderHistoryChart(range) {
  const d = lastResult ? (lastResult.trend || {}) : {};
  let counts;
  if (range === "session") counts = { positive: 0, neutral: 0, negative: 0, ...countSession() };
  else if (range === "last7") counts = d.last_7 || {};
  else counts = d.last_30 || {};
  chartHistory.data.datasets[0].data = [counts.positive || 0, counts.neutral || 0, counts.negative || 0];
  chartHistory.update();
}
function countSession() {
  const c = { positive: 0, neutral: 0, negative: 0 };
  sessionHistory.forEach(h => { if (c[h.emotion] != null) c[h.emotion]++; });
  return c;
}

/* ---------- 主流程 ---------- */
async function predict(payload, presetName) {
  const res = await fetch(API + "/predict", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
  const d = await res.json();
  lastResult = d;
  // 記入 session 時間軸
  sessionHistory.push({ time: new Date().toLocaleTimeString("zh-Hant", { hour: "2-digit", minute: "2-digit" }),
    emotion: d.prediction.emotion, preset: presetName });
  if (sessionHistory.length > 50) sessionHistory.shift();

  renderMonitor(d); renderSignals(d); renderPersonal(d); renderAI(d); renderHistory();
  setSource("simulated");
}

function setSource(source) {
  const chip = document.querySelector(".source-chip");
  const note = document.getElementById("source-note");
  chip.classList.remove("real", "simulated", "excel");
  if (source === "real") {
    chip.textContent = "真實裝置";
    chip.classList.add("real");
    note.textContent = "● 正在接收真實裝置資料";
  } else if (source === "excel") {
    chip.textContent = "Excel 資料";
    chip.classList.add("excel");
    note.textContent = "📂 目前顯示 Excel 批次預測的某一列";   // applyExcelRow 會再依該列補上 AI 說明
  } else {
    chip.textContent = "模擬資料";
    chip.classList.add("simulated");
    note.textContent = "⚠ 目前為「模擬資料」示範";
  }
}

// 即時監測（輪詢 /api/latest 取得真實裝置資料）
let livePolling = null;
async function fetchLatest() {
  try {
    const r = await fetch(API + "/api/latest");
    const d = await r.json();
    if (d && d.result) {
      lastResult = d.result;
      renderMonitor(d.result); renderSignals(d.result); renderPersonal(d.result); renderAI(d.result); renderHistory();
      // 誠實標示來源：device = 真實裝置上傳，simulated = 網頁情境按鈕
      setSource(d.source === "device" ? "real" : "simulated");
    } else {
      setSource("real");
      document.getElementById("source-note").textContent = "○ 等待裝置資料…（請接上 ESP/Arduino）";
    }
  } catch (e) { /* 略過 */ }
}
function startLivePolling() {
  if (livePolling) return;
  stopExcelPlay();   // 回到即時監測 → 停掉 Excel 播放
  fetchLatest();
  livePolling = setInterval(fetchLatest, 2000);
  document.getElementById("btn-live").textContent = "⏸ 停止監測";
}
function stopLivePolling() {
  clearInterval(livePolling); livePolling = null;
  document.getElementById("btn-live").textContent = "📡 即時監測";
}
document.getElementById("btn-live").addEventListener("click", () => {
  if (livePolling) stopLivePolling(); else startLivePolling();
});

/* ---------- 模擬裝置（沒有 Arduino 也能讓網站自己跑） ---------- */
const btnSim = document.getElementById("btn-sim");
function renderSimButton(running) {
  btnSim.textContent = running ? "🧪 模擬裝置：運行中" : "🧪 模擬裝置：已停止";
  btnSim.classList.toggle("active", running);
}
async function simStatus() {
  try {
    const r = await fetch(API + "/api/simulate/status");
    const d = await r.json();
    renderSimButton(!!d.running);
    return !!d.running;
  } catch (e) { return false; }
}
btnSim.addEventListener("click", async () => {
  const running = btnSim.classList.contains("active");
  btnSim.disabled = true;
  try {
    await fetch(API + (running ? "/api/simulate/stop" : "/api/simulate/start"), { method: "POST" });
    if (await simStatus()) startLivePolling();   // 以伺服器實際狀態為準
  } catch (e) { /* 略過 */ }
  btnSim.disabled = false;
});


document.querySelectorAll("[data-preset]").forEach(btn => {
  const name = { calm: "平靜", normal: "正常", stress: "壓力" }[btn.dataset.preset];
  btn.addEventListener("click", () => predict(PRESETS[btn.dataset.preset], name));
});

/* ---------- AI 對話框 ---------- */
const CHAT_KEY = "mood_chat_history";
let chatHistory = loadChat();
let chatBusy = false;

function loadChat() {
  try { return JSON.parse(sessionStorage.getItem(CHAT_KEY)) || []; } catch (e) { return []; }
}
function saveChat() {
  try { sessionStorage.setItem(CHAT_KEY, JSON.stringify(chatHistory.slice(-40))); } catch (e) {}
}

function chatBubble(role, text) {
  const el = document.createElement("div");
  el.className = "chat-msg " + role;
  el.textContent = text;
  document.getElementById("chat-body").appendChild(el);
  document.getElementById("chat-body").scrollTop = 1e9;
  return el;
}

function renderChat() {
  const body = document.getElementById("chat-body");
  body.innerHTML = "";
  if (!chatHistory.length) {
    chatBubble("sys", "嗨！我是 AI 對話助手。\n先到「即時監測」按一個情境按鈕，我就能參考你的數據回答。");
  } else {
    chatHistory.forEach(m => chatBubble(m.role === "user" ? "user" : "ai", m.content));
  }
  updateChatSub();
}

function updateChatSub() {
  const has = !!lastResult;
  setText("chat-sub", has
    ? `已讀取最新結果：${(EMOTION[lastResult.prediction.emotion] || {}).label || "—"}`
    : "尚無量測結果，先按一個情境按鈕");
}

function toggleChat(open) {
  const panel = document.getElementById("chat-panel");
  const fab = document.getElementById("chat-fab");
  const isOpen = open ?? !panel.classList.contains("open");
  panel.classList.toggle("open", isOpen);
  fab.classList.toggle("open", isOpen);
  panel.setAttribute("aria-hidden", String(!isOpen));
  if (isOpen) {
    updateChatSub();
    document.getElementById("chat-text").focus();
  }
}

async function sendChat(text) {
  text = (text || "").trim();
  if (!text || chatBusy) return;
  chatBusy = true;
  document.getElementById("chat-send").disabled = true;

  // 對話開始時，先在氣泡區留下系統提示（第一次才顯示）
  const body = document.getElementById("chat-body");
  if (body.querySelector(".chat-msg.sys")) body.innerHTML = "";

  chatHistory.push({ role: "user", content: text });
  chatBubble("user", text);
  saveChat();

  const typing = chatBubble("ai typing", "思考中…");
  try {
    const r = await fetch(API + "/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: chatHistory.slice(-10) }),
    });
    const d = await r.json();
    typing.remove();
    chatHistory.push({ role: "assistant", content: d.reply });
    chatBubble("ai", d.reply);
    saveChat();
    if (d.status !== "ok") setText("chat-sub", "AI 暫時無法回應");
    else updateChatSub();
  } catch (e) {
    typing.remove();
    chatBubble("sys", "連線失敗，請稍後再試。");
  } finally {
    chatBusy = false;
    document.getElementById("chat-send").disabled = false;
    document.getElementById("chat-text").focus();
  }
}

document.getElementById("chat-fab").addEventListener("click", () => toggleChat(true));
document.getElementById("chat-close").addEventListener("click", () => toggleChat(false));
document.getElementById("chat-clear").addEventListener("click", () => {
  chatHistory = []; saveChat(); renderChat();
});
document.getElementById("chat-form").addEventListener("submit", e => {
  e.preventDefault();
  const box = document.getElementById("chat-text");
  const t = box.value;
  box.value = "";
  box.style.height = "auto";
  sendChat(t);
});
document.getElementById("chat-text").addEventListener("input", e => {
  e.target.style.height = "auto";
  e.target.style.height = Math.min(e.target.scrollHeight, 104) + "px";
});
document.getElementById("chat-text").addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); e.target.form.requestSubmit(); }
});
document.querySelectorAll(".chat-chip").forEach(b => {
  b.addEventListener("click", () => sendChat(b.dataset.q));
});
renderChat();

/* ---------- 啟動 ---------- */
(async () => {
  try {
    const r = await fetch(API + "/health");
    const h = await r.json();
    document.getElementById("status-dot").classList.add("on");
    setText("status-text", h.model_loaded ? "已連接" : "模型未載入");
  } catch (e) {
    setText("status-text", "未連接");
  }
  renderHistory();
  // 開頁即跑：後端啟動時已自動開啟模擬裝置，前端也自動開始輪詢。
  renderSimButton(await simStatus());
  startLivePolling();
})();

/* ---------- Excel 批量預測（測準確率） ---------- */
let excelData = null;      // 後端回傳的 {count, summary, results, ...}
let excelPlayTimer = null;
let excelPlayIdx = 0;

const btnExcel = document.getElementById("btn-excel");
const fileInput = document.getElementById("excel-file");
const excelModal = document.getElementById("excel-modal");

function openExcelModal() { excelModal.classList.add("open"); excelModal.setAttribute("aria-hidden", "false"); }
function closeExcelModal() {
  // 關閉結果視窗不停止播放：讓主頁面繼續輪播上傳的資料。
  // 要回到即時監測，按「📡 即時監測」即可（會自動停掉 Excel 播放）。
  excelModal.classList.remove("open"); excelModal.setAttribute("aria-hidden", "true");
}
btnExcel.addEventListener("click", () => fileInput.click());
document.getElementById("excel-close").addEventListener("click", closeExcelModal);
excelModal.addEventListener("click", e => { if (e.target === excelModal) closeExcelModal(); });

fileInput.addEventListener("change", async () => {
  const file = fileInput.files[0];
  if (!file) return;
  fileInput.value = "";   // 允許重複選同一個檔案

  const body = document.getElementById("excel-body");
  body.innerHTML =
    `<p class="note">⏳ 正在上傳「${file.name}」…每一列都會做完整分析
     （ML + DeepSeek 投票＋解釋），資料越多越久，請稍候不要關閉視窗。</p>`;
  openExcelModal();

  try {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch(API + "/predict/excel", { method: "POST", body: fd });
    const d = await res.json();
    if (!res.ok) throw new Error(d.detail || "上傳失敗");
    excelData = d;
    renderExcelResults(d);
    // 上傳完直接套用到主頁面：自動播放每一列，並在結果視窗顯示原始數據預覽
    startExcelPlay();
  } catch (e) {
    body.innerHTML = `<p class="excel-warn">❌ ${e.message || e}</p>`;
  }
});

function renderExcelResults(d) {
  const body = document.getElementById("excel-body");
  const dist = d.emotion_distribution || {};
  const total = d.count;

  const name = { positive: "正面", neutral: "中性", negative: "負面" };
  const stats = ["positive", "neutral", "negative"].map(k => `
    <div class="excel-stat ${k}">
      <div class="n">${dist[k] ?? 0}</div>
      <div class="l">${name[k]}</div>
    </div>`).join("");

  let warn = "";
  if ((d.unknown_columns || []).length) {
    warn = `<p class="excel-warn">⚠ 以下欄位沒認得、已跳過：${d.unknown_columns.join("、")}</p>`;
  }

  const rowsHtml = (d.summary || []).map(s => {
    const clickable = s.index < (d.results || []).length;
    // 每一列都做完整分析，DeepSeek 那一票直接顯示
    let ai;
    if (s.llm_emotion) {
      const agree = s.model_agreement;
      const icon = agree === true ? "✔" : (agree === false ? "✘" : "•");
      const cls = agree === true ? "agree" : (agree === false ? "disagree" : "");
      ai = `<span class="excel-ai ${cls}" title="AI 投 ${s.llm_emotion}">🤖 ${icon} ${name[s.llm_emotion] || s.llm_emotion}</span>`;
    } else {
      ai = `<span class="excel-ai muted" title="AI 棄權或呼叫失敗">🤖 棄權</span>`;
    }
    return `<tr class="${clickable ? "clickable" : ""}" data-idx="${s.index}">
      <td>#${s.index + 1}</td>
      <td>${s.sample_id ?? "—"}</td>
      <td><span class="excel-badge ${s.emotion}">${name[s.emotion] || s.emotion}</span></td>
      <td>${Math.round(s.confidence * 100)}%</td>
      <td>${ai}</td>
      <td>${s.quality}</td>
    </tr>`;
  }).join("");

  const ag = d.model_agreement || {};
  const aiLine = `<p class="note">🤖 全部 <strong>${total}</strong> 列都做了完整分析（與即時接收資料相同）：
       ML 與 DeepSeek 一致 <strong>${ag.agree ?? 0}</strong> 列、
       分歧 <strong>${ag.disagree ?? 0}</strong> 列
       （一致率 ${ag.rate == null ? "—" : Math.round(ag.rate * 100) + "%"}）。
       <br>注意：融合規則下 DeepSeek <strong>不會改變預測標籤</strong>，
       它只給第二意見並調整信心度，最終情緒一律以 ML 為準。</p>`;

  body.innerHTML = `
    <div class="excel-stats">
      <div class="excel-stat"><div class="n">${total}</div><div class="l">總列數</div></div>
      ${stats}
    </div>
    <div class="table-wrap" style="max-height:240px; overflow:auto;">
      <table class="excel-table">
        <thead><tr><th>#</th><th>sample_id</th><th>ML 預測</th><th>信心度</th><th>🤖 DeepSeek</th><th>品質</th></tr></thead>
        <tbody>${rowsHtml}</tbody>
      </table>
    </div>
    <p class="note">點表格列可把該列套用到整個網頁（前 ${Math.min(total, (d.results || []).length)} 列）。已自動開始逐筆套用主頁面。</p>
    <div class="excel-features" id="excel-features"></div>
    ${aiLine}
    ${warn}`;

  document.getElementById("excel-download").disabled = false;
  document.getElementById("excel-play").disabled = (d.results || []).length === 0;

  body.querySelectorAll("tr.clickable").forEach(tr => {
    tr.addEventListener("click", () => applyExcelRow(parseInt(tr.dataset.idx)));
  });

  renderExcelFeatures(0);   // 預設顯示第一列的原始數據
}

function applyExcelRow(idx) {
  const r = excelData.results[idx];
  if (!r) return;
  lastResult = r;
  sessionHistory.push({
    time: new Date().toLocaleTimeString("zh-Hant", { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
    emotion: r.prediction.emotion, preset: `Excel #${idx + 1}`,
  });
  if (sessionHistory.length > 50) sessionHistory.shift();
  renderMonitor(r); renderSignals(r); renderPersonal(r); renderAI(r); renderHistory();
  setSource("excel");
  const s = (excelData.summary || [])[idx];
  const note = document.getElementById("source-note");
  if (s && s.llm_emotion) {
    note.textContent = `📂 Excel 第 ${idx + 1} 列（ML + DeepSeek 完整分析：AI 投 ${s.llm_emotion}，`
      + `${s.model_agreement === true ? "與 ML 一致" : s.model_agreement === false ? "與 ML 分歧，採 ML" : "未表態"}）`;
  } else {
    note.textContent = `📂 Excel 第 ${idx + 1} 列（AI 棄權或呼叫失敗，僅 ML）`;
  }
  document.querySelector('.nav-btn[data-tab="monitor"]').click();
  highlightExcelRow(idx);
  renderExcelFeatures(idx);
}

function highlightExcelRow(idx) {
  document.querySelectorAll(".excel-table tr").forEach(tr => tr.classList.remove("playing"));
  const tr = document.querySelector(`.excel-table tr[data-idx="${idx}"]`);
  if (tr) tr.classList.add("playing");
}

/* 原始數據預覽：把選中列的 29 個特徵，依 EEG / HRV / EDA 分組顯示 */
function renderExcelFeatures(idx) {
  const box = document.getElementById("excel-features");
  if (!box || !excelData) return;
  const r = (excelData.results || [])[idx];
  const features = r && r._excel ? r._excel.features : null;
  const groups = excelData.feature_groups || [
    { group: "EEG", features: ["delta", "theta", "alpha", "beta", "gamma",
      "relative_delta", "relative_theta", "relative_alpha", "relative_beta", "relative_gamma",
      "theta_beta_ratio", "alpha_beta_ratio", "alpha_theta_ratio"] },
    { group: "HRV", features: ["heart_rate", "mean_rr", "rmssd", "sdnn", "pnn50",
      "lf", "hf", "lf_hf_ratio", "total_power"] },
    { group: "EDA", features: ["scl", "scr_amplitude", "scr_frequency", "tonic",
      "phasic", "eda_variability", "eda_latency"] },
  ];

  if (!features) {
    box.innerHTML = `<p class="note">此列無原始特徵。</p>`;
    return;
  }

  const fmt = v => (typeof v === "number" ? (Math.round(v * 1000) / 1000).toString() : (v ?? "—"));
  const panels = groups.map(g => {
    const cells = g.features
      .filter(k => k in features)
      .map(k => `<div class="excel-fv"><span class="k">${k}</span><span class="v">${fmt(features[k])}</span></div>`)
      .join("");
    return `<div class="excel-fgroup">
      <div class="excel-fg-head">${g.group}</div>
      <div class="excel-fgrid">${cells}</div>
    </div>`;
  }).join("");

  box.innerHTML = `
    <div class="excel-features-head">📋 原始數據預覽 —— 第 ${idx + 1} 列
      <span class="excel-features-sub">（sample_id: ${(r && r._excel.sample_id) ?? "—"}）</span>
    </div>
    ${panels}`;
}

function downloadExcelCSV() {
  if (!excelData) return;
  const header = "sample_id,person_id,predicted_emotion,confidence,llm_emotion,model_agreement,quality";
  const lines = excelData.summary.map(s =>
    [s.sample_id ?? "", s.person_id ?? "", s.emotion, s.confidence,
     s.llm_emotion ?? "", s.model_agreement ?? "", s.quality].join(","));
  const csv = "﻿" + [header, ...lines].join("\n");   // BOM：Excel 開 CSV 才不會亂碼
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "predictions.csv";
  a.click();
  URL.revokeObjectURL(a.href);
}

const btnExcelDownload = document.getElementById("excel-download");
const btnExcelPlay = document.getElementById("excel-play");
btnExcelDownload.addEventListener("click", downloadExcelCSV);
btnExcelPlay.addEventListener("click", () => {
  if (excelPlayTimer) stopExcelPlay(); else startExcelPlay();
});

function startExcelPlay() {
  if (!excelData) return;
  stopLivePolling();   // Excel 播放與即時輪詢互斥，否則每 2 秒會被模擬資料蓋掉
  excelPlayIdx = 0;
  btnExcelPlay.textContent = "⏸ 停止播放";
  applyExcelRow(0);
  excelPlayTimer = setInterval(() => {
    excelPlayIdx = (excelPlayIdx + 1) % excelData.results.length;
    applyExcelRow(excelPlayIdx);
  }, 1500);
}
function stopExcelPlay() {
  clearInterval(excelPlayTimer);
  excelPlayTimer = null;
  btnExcelPlay.textContent = "▶ 逐筆套用到網頁";
}
