/*
  ESP32 串口接收器 - 接收树莓派 YOLO 检测数据
  接线：
    树莓派 GPIO14(TXD, Pin8)  ->  ESP32 RX (GPIO3)
    树莓派 GPIO15(RXD, Pin10) ->  ESP32 TX (GPIO1)
    GND -> GND (必须共地！)

  波特率: 115200
  数据格式: F=帧号,P=盆栽数,B=x1,y1,x2,y2,conf;x1,y1,x2,y2,conf...\n
*/

#include <Arduino.h>

#define RX_PIN 16   // ESP32 UART2 RX (可自定义)
#define TX_PIN 17   // ESP32 UART2 TX (可自定义)
#define BAUD_RATE 115200

String serialBuffer = "";

void setup() {
  Serial.begin(115200);           // USB 调试串口
  Serial2.begin(BAUD_RATE, SERIAL_8N1, RX_PIN, TX_PIN);  // 树莓派连接串口

  Serial.println("ESP32 Serial Receiver Started");
  Serial.println("Waiting for Pi data...");
}

void loop() {
  // 读取树莓派串口数据
  while (Serial2.available()) {
    char c = Serial2.read();
    if (c == '\n') {
      parseData(serialBuffer);
      serialBuffer = "";
    } else {
      serialBuffer += c;
    }
  }
}

void parseData(String data) {
  data.trim();
  if (data.length() == 0) return;

  Serial.print("[Pi->ESP32] ");
  Serial.println(data);

  // 解析 F=xxx
  int fIdx = data.indexOf("F=");
  int pIdx = data.indexOf("P=");
  int bIdx = data.indexOf("B=");

  if (fIdx == -1 || pIdx == -1) {
    Serial.println("  [WARN] Invalid format");
    return;
  }

  // 提取帧号
  int fEnd = data.indexOf(",", fIdx);
  if (fEnd == -1) fEnd = data.length();
  int frameNum = data.substring(fIdx + 2, fEnd).toInt();

  // 提取盆栽数
  int pEnd = data.indexOf(",", pIdx);
  if (pEnd == -1) pEnd = data.length();
  int plantCount = data.substring(pIdx + 2, pEnd).toInt();

  Serial.print("  Frame=");
  Serial.print(frameNum);
  Serial.print(" Plants=");
  Serial.println(plantCount);

  // 提取框数据
  if (bIdx != -1 && plantCount > 0) {
    String boxData = data.substring(bIdx + 2);
    int boxIdx = 1;
    int start = 0;
    while (start < boxData.length()) {
      int end = boxData.indexOf(";", start);
      if (end == -1) end = boxData.length();
      String oneBox = boxData.substring(start, end);

      // 解析 x1,y1,x2,y2,conf
      int c1 = oneBox.indexOf(",");
      int c2 = oneBox.indexOf(",", c1 + 1);
      int c3 = oneBox.indexOf(",", c2 + 1);
      int c4 = oneBox.indexOf(",", c3 + 1);
      if (c1 != -1 && c2 != -1 && c3 != -1 && c4 != -1) {
        float x1 = oneBox.substring(0, c1).toFloat();
        float y1 = oneBox.substring(c1 + 1, c2).toFloat();
        float x2 = oneBox.substring(c2 + 1, c3).toFloat();
        float y2 = oneBox.substring(c3 + 1, c4).toFloat();
        float conf = oneBox.substring(c4 + 1).toFloat();

        float cx = (x1 + x2) / 2.0;
        float cy = (y1 + y2) / 2.0;

        Serial.print("  Box");
        Serial.print(boxIdx);
        Serial.print(": x1=");
        Serial.print(x1);
        Serial.print(" y1=");
        Serial.print(y1);
        Serial.print(" x2=");
        Serial.print(x2);
        Serial.print(" y2=");
        Serial.print(y2);
        Serial.print(" cx=");
        Serial.print(cx);
        Serial.print(" cy=");
        Serial.print(cy);
        Serial.print(" conf=");
        Serial.println(conf);

        // TODO: 在这里添加小车控制逻辑
        // 例如：根据 cx 判断盆栽在左/右，控制电机转向
        // if (cx < 320) turnLeft(); else turnRight();
      }
      start = end + 1;
      boxIdx++;
    }
  }

  // 示例：根据检测结果控制小车（伪代码）
  if (plantCount > 0) {
    // 有盆栽 -> 可以在这里触发电机控制
    // digitalWrite(LED_PIN, HIGH);
  } else {
    // 无盆栽 -> 原地旋转扫描
    // digitalWrite(LED_PIN, LOW);
  }
}
