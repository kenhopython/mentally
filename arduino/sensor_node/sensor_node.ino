/*
 * sensor_node.ino — 感測節點（Arduino Nano / Uno）
 * ===================================================
 * 讀取三種生理訊號，每秒用 Serial 輸出一行 JSON：
 *
 *   Pulse Sensor   → A0  心率 (bpm) + 心跳間隔 IBI + HRV (RMSSD)
 *   Grove GSR      → A1  皮膚電 (eda_raw + eda μS 近似值)
 *   NeuroSky TGAM  → D10(RX)  EEG 專注度 / 放鬆度 (0-100)
 *
 * 兩種使用方式：
 *   1. 有線：把這個板子用 USB 接到電腦，Python 端 (serial_reader.py) 讀 Serial。
 *   2. 無線：把這塊板子的 TX 接到 ESP8266 的 RX，由 wifi_bridge.ino 轉發上網。
 *
 * 注意：Serial 波特率必須是 115200（與 config.py 的 SERIAL_BAUD 一致）。
 */

// ------------------------- 腳位定義 -------------------------
const int PULSE_PIN = A0;    // Pulse Sensor 訊號腳
const int GSR_PIN   = A1;    // Grove GSR 訊號腳

// ------------------------- Pulse Sensor 心跳偵測 -------------------------
const int PULSE_THRESHOLD = 512;  // 心跳上升沿的判定閾值（訊號中點）
const int IBI_WINDOW      = 10;   // 存最近 10 個 IBI 計算 HRV (RMSSD)

unsigned long lastBeatTime = 0;    // 上一次心跳的時間 (ms)
unsigned long lastSampleMs = 0;    // 上一次取樣時間
int   signal;                      // 目前訊號值
bool  aboveThreshold = false;      // 訊號是否已在閾值之上
float ibiHistory[IBI_WINDOW];      // IBI 環形緩衝
int   ibiIndex = 0;

float bpm = 0.0;
float ibi = 0.0;                   // 心跳間隔 (ms)
float hrv = 0.0;                   // HRV = RMSSD (ms)

// ------------------------- NeuroSky TGAM 解析 -------------------------
#include <SoftwareSerial.h>
SoftwareSerial tgam(10, 11);       // RX=D10 接 TGAM 的 TX；TX=D11 接 TGAM 的 RX

int attention  = 0;                // 專注度 0-100
int meditation = 0;                // 放鬆度 0-100
int signalQuality = 200;           // 0 = 訊號良好，越大越差

// TGAM 封包解析狀態機
enum TGAMState { SYNC1, SYNC2, PLENGTH, PAYLOAD, CHECKSUM };
TGAMState tgState = SYNC1;
int  plength = 0;
int  payloadIndex = 0;
byte payload[32];
byte checksum = 0;

// ------------------------- 輸出計時 -------------------------
unsigned long lastPrintMs = 0;

void setup() {
  Serial.begin(115200);            // 與 config.py 一致
  tgam.begin(57600);               // TGAM 固定 57600 baud
  pinMode(PULSE_PIN, INPUT);
  pinMode(GSR_PIN, INPUT);
  // 送出啟動提示（方便在 Serial Monitor 看到）
  Serial.println("{\"info\":\"sensor_node ready\"}");
}

void loop() {
  unsigned long now = millis();

  readPulse(now);   // 心跳 / HRV
  readGSR();        // 皮膚電
  readTGAM();       // EEG

  // 每秒輸出一行 JSON
  if (now - lastPrintMs >= 1000) {
    lastPrintMs = now;
    printJSON();
  }
}

// ---------------------------------------------------------------------------
// 讀取 Pulse Sensor：上升沿觸發心跳，計算 BPM / IBI / HRV(RMSSD)
// ---------------------------------------------------------------------------
void readPulse(unsigned long now) {
  signal = analogRead(PULSE_PIN);

  if (signal > PULSE_THRESHOLD && !aboveThreshold) {
    // 上升沿：偵測到一次心跳
    aboveThreshold = true;
    if (lastBeatTime > 0) {
      ibi = now - lastBeatTime;              // 這次與上次心跳的間隔 (ms)
      if (ibi > 300 && ibi < 2000) {         // 過濾明顯的雜訊
        bpm = 60000.0 / ibi;                 // 每分鐘心跳數
        ibiHistory[ibiIndex] = ibi;
        ibiIndex = (ibiIndex + 1) % IBI_WINDOW;
        hrv = computeRMSSD();                // 心率變異性
      }
    }
    lastBeatTime = now;
  } else if (signal < PULSE_THRESHOLD - 20) {
    aboveThreshold = false;                  // 加遲滯，避免雜訊重複觸發
  }
}

// RMSSD：相鄰 IBI 差值的均方根，單位 ms，越大代表越放鬆/自律神經較平衡
float computeRMSSD() {
  float sum = 0;
  int   n = 0;
  for (int i = 1; i < IBI_WINDOW; i++) {
    int a = (ibiIndex - i + IBI_WINDOW) % IBI_WINDOW;
    int b = (ibiIndex - i - 1 + IBI_WINDOW) % IBI_WINDOW;
    if (ibiHistory[a] > 0 && ibiHistory[b] > 0) {
      float diff = ibiHistory[a] - ibiHistory[b];
      sum += diff * diff;
      n++;
    }
  }
  return n ? sqrt(sum / n) : 0.0;
}

// ---------------------------------------------------------------------------
// 讀取 Grove GSR（皮膚電導）
// ---------------------------------------------------------------------------
float edaRaw = 0.0;
float eda = 0.0;

void readGSR() {
  edaRaw = analogRead(GSR_PIN);   // 0-1023

  // 粗略把 ADC 映射到 0-10 μS。
  // 注意：這是線性近似值，正式測量前請用 GSR 模組上的可調電阻校準。
  // 皮膚電導的典型範圍約 0.1-10 μS。
  eda = edaRaw / 1023.0 * 10.0;
}

// ---------------------------------------------------------------------------
// 讀取 NeuroSky TGAM（57600 baud，封包格式：
//   AA AA <plength> <payload[plength]> <checksum>
//   checksum = ~(payload 各 byte 之和) & 0xFF
// plength 意義：0x02=訊號品質  0x04=專注度  0x05=放鬆度
//               0x80=原始波  0x83=腦波能量(24 bytes)
// ---------------------------------------------------------------------------
void readTGAM() {
  while (tgam.available()) {
    byte b = tgam.read();

    switch (tgState) {
      case SYNC1:
        if (b == 0xAA) tgState = SYNC2;
        break;
      case SYNC2:
        if (b == 0xAA) tgState = PLENGTH;
        else tgState = SYNC1;             // 第二個同步位元不對，重來
        break;
      case PLENGTH:
        plength = b;
        payloadIndex = 0;
        checksum = 0;
        if (plength > 32 || plength == 0) tgState = SYNC1;  // 不合法的長度
        else tgState = PAYLOAD;
        break;
      case PAYLOAD:
        payload[payloadIndex++] = b;
        checksum += b;
        if (payloadIndex >= plength) tgState = CHECKSUM;
        break;
      case CHECKSUM:
        if ((byte)(~checksum & 0xFF) == b) {
          parsePayload(plength);          // 校驗通過才解析
        }
        tgState = SYNC1;
        break;
    }
  }
}

void parsePayload(int len) {
  switch (len) {
    case 0x02:  // 訊號品質 0-200（0 最好）
      signalQuality = payload[0];
      break;
    case 0x04:  // 專注度 0-100
      attention = payload[0];
      break;
    case 0x05:  // 放鬆度 0-100
      meditation = payload[0];
      break;
    // 0x83 = 腦波能量（delta/theta/alpha/beta/gamma，各 3 bytes 大端），
    // V1 暫不使用，未來可在此解析各頻段功率。
  }
}

// ---------------------------------------------------------------------------
// 輸出 JSON（一筆一列，Python / ESP8266 直接解析）
// ---------------------------------------------------------------------------
void printJSON() {
  Serial.print("{\"hr\":");
  Serial.print(bpm, 1);
  Serial.print(",\"ibi\":");
  Serial.print(ibi, 1);
  Serial.print(",\"hrv\":");
  Serial.print(hrv, 1);
  Serial.print(",\"eda\":");
  Serial.print(eda, 2);
  Serial.print(",\"eda_raw\":");
  Serial.print(edaRaw, 0);
  Serial.print(",\"attention\":");
  Serial.print(signalQuality == 0 ? attention : 0);   // 訊號差時回報 0
  Serial.print(",\"meditation\":");
  Serial.print(signalQuality == 0 ? meditation : 0);
  Serial.println("}");
}
