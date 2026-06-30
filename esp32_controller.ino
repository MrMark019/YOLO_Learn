/*
  ESP32 控制器 - 向树莓派发送 START/STOP 指令

  接线：
    ESP32 TX (GPIO17) -> 树莓派 RX (GPIO15, Pin10)
    ESP32 RX (GPIO16) -> 树莓派 TX (GPIO14, Pin8)
    GND -> GND

  功能：
    - bool runDetection 置 true  -> 发送 "START\n"
    - bool runDetection 置 false -> 发送 "STOP\n"
    - 同时接收树莓派发来的检测数据并打印

  触发方式（三选一）：
    1. 按钮触发（GPIO4 接按钮到 GND）
    2. 定时器自动切换
    3. 串口命令控制（USB 串口输入 1/0）
*/

#include <Arduino.h>

#define RX_PIN 16           // ESP32 UART2 RX
#define TX_PIN 17           // ESP32 UART2 TX
#define BAUD_RATE 115200
#define BUTTON_PIN 4        // 按钮引脚（接 GND，内部上拉）

// ========== 控制变量 ==========
bool runDetection = false;   // 置 1 发送 START，置 0 发送 STOP
bool prevState = false;      // 用于检测状态变化

String piBuffer = "";        // 接收树莓派数据的缓冲区

void setup() {
  Serial.begin(115200);                        // USB 调试串口
  Serial2.begin(BAUD_RATE, SERIAL_8N1, RX_PIN, TX_PIN);  // 树莓派连接串口

  pinMode(BUTTON_PIN, INPUT_PULLUP);           // 按钮内部上拉

  Serial.println("========================================");
  Serial.println("ESP32 Controller Started");
  Serial.println("控制方式:");
  Serial.println("  - 按 GPIO4 按钮切换 START/STOP");
  Serial.println("  - 或在 USB 串口输入 1/0 切换");
  Serial.println("  - 或在代码中修改 runDetection 变量");
  Serial.println("========================================");
  Serial.print("当前状态: ");
  Serial.println(runDetection ? "START (运行中)" : "STOP (已停止)");
}

void loop() {
  // ---- 方式1: 按钮触发（按下切换状态） ----
  static bool lastButtonState = HIGH;
  bool buttonState = digitalRead(BUTTON_PIN);
  if (lastButtonState == HIGH && buttonState == LOW) {
    delay(50);  // 消抖
    if (digitalRead(BUTTON_PIN) == LOW) {
      runDetection = !runDetection;
      Serial.print("[按钮] 状态切换为: ");
      Serial.println(runDetection ? "START" : "STOP");
    }
  }
  lastButtonState = buttonState;

  // ---- 方式2: USB 串口命令控制 ----
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '1') {
      runDetection = true;
      Serial.println("[USB命令] 设置为 START");
    } else if (c == '0') {
      runDetection = false;
      Serial.println("[USB命令] 设置为 STOP");
    }
  }

  // ---- 检测状态变化，发送指令到树莓派 ----
  if (runDetection != prevState) {
    prevState = runDetection;
    if (runDetection) {
      Serial2.print("START\n");
      Serial.println("[->Pi] 发送 START");
    } else {
      Serial2.print("STOP\n");
      Serial.println("[->Pi] 发送 STOP");
    }
  }

  // ---- 接收树莓派数据 ----
  while (Serial2.available()) {
    char c = Serial2.read();
    if (c == '\n') {
      Serial.print("[Pi->] ");
      Serial.println(piBuffer);
      piBuffer = "";
    } else {
      piBuffer += c;
    }
  }
}
