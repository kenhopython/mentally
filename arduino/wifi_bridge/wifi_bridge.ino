/*
 * wifi_bridge.ino — 無線傳輸橋接（ESP8266 或 ESP32）+ 手機一鍵設定
 * ================================================================
 * 接線：
 *   感測節點 Arduino 的 TX  →  ESP8266 的 RX (GPIO3)
 *   兩塊板子 GND 接 GND（ESP 只需供電，USB 或 5V/3.3V）
 *
 * 第一次使用（或 WiFi 連不上）時，ESP 會自動開一個設定熱點：
 *   SSID: MoodSense-Setup    密碼: 無
 * 用手機連上這個熱點，瀏覽器會自動跳出設定頁，填入：
 *   1. 你要使用的 WiFi 名稱與密碼
 *   2. 伺服器位址（IP:Port）—— 網頁按「無線連接」按鈕可看到，直接複製
 * 儲存後 ESP 會自動連上 WiFi 並開始上傳資料，設定會記住，之後不用再填。
 *
 * 依賴函式庫：WiFiManager
 *   Arduino IDE → 工具 → 管理程式庫 → 搜尋「WiFiManager」→ 安裝
 *
 * 想重新設定：把下方「wm.resetSettings();」取消註解燒一次，再開機即重新進入熱點。
 */

// =========================== 設定區 ===========================
// 伺服器預設位址（首次設定時的預設值；之後以手機輸入的為準）
// 埠要跟 api/main.py 的 8001 一致；IP 改成你電腦在區網的位址
const char* DEFAULT_SERVER = "192.168.1.100:8001";

// 認證 Token（會放在 X-Ingest-Token 標頭送出）
// ⚠ 後端 /api/ingest 目前「沒有」驗證這個值，所以改了不會有任何效果 ——
//    它只是預留欄位。要真的生效，得先在後端讀環境變數 INGEST_TOKEN 再比對標頭。
//    這裡刻意留佔位符：不要把真實 token 寫進版控。
const char* INGEST_TOKEN   = "CHANGE_ME";
// ==============================================================

#ifdef ESP8266
  #include <ESP8266WiFi.h>
  #include <ESP8266HTTPClient.h>
  #include <WiFiManager.h>
#elif defined(ESP32)
  #include <WiFi.h>
  #include <HTTPClient.h>
  #include <WiFiManager.h>
#endif

WiFiManager wm;
// 自訂參數：伺服器位址（IP:Port），儲存在 flash 中，設定一次即可
WiFiManagerParameter serverParam("server", "Server IP:Port", DEFAULT_SERVER, 24);

void setup() {
  Serial.begin(115200);            // 與 sensor_node 一致
  delay(200);

  // 如要強制重新設定，取消下一行註解再燒錄一次：
  // wm.resetSettings();

  wm.addParameter(&serverParam);   // 加入自訂欄位
  wm.setConfigPortalTimeout(300);  // 設定熱點最多開 5 分鐘

  // 嘗試連線；失敗會自動開熱點「MoodSense-Setup」等手機設定
  if (!wm.autoConnect("MoodSense-Setup")) {
    delay(3000);
    ESP.restart();                 // 重啟後可再次進入設定
  }

  // 已連上 WiFi（順便告訴 sensor_node，它不會解析，無害）
  Serial.println("{\"info\":\"wifi_connected\"}");
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    WiFi.reconnect();
    delay(1000);
    return;
  }

  // 讀一行感測節點送來的 JSON
  if (!Serial.available()) return;
  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.length() == 0 || !line.startsWith("{")) return;

  sendToServer(line);
}

// 把使用者輸入的「IP」或「IP:Port」補成完整網址
String buildHost() {
  String h = String(serverParam.getValue());
  h.trim();
  if (h.indexOf(':') < 0) h += ":5000";   // 沒填 port 就用預設 5000
  return h;
}

void sendToServer(String json) {
#ifdef ESP8266
  WiFiClient client;
  HTTPClient http;
  http.begin(client, "http://" + buildHost() + "/api/data");
#elif defined(ESP32)
  HTTPClient http;
  http.begin("http://" + buildHost() + "/api/data");
#endif

  http.addHeader("Content-Type", "application/json");
  http.addHeader("X-Ingest-Token", INGEST_TOKEN);

  int code = http.POST(json);
  // code > 0 表示伺服器有回應；非 200 可能是 token 錯或位址錯
  http.end();
}
