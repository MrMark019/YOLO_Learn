"""
ONNX Runtime 摄像头实时检测脚本 (无GUI头less模式)
专为树莓派优化，不依赖 torch，支持ESP32 START/STOP控制

功能：
  - 接收ESP32串口命令 START/STOP 控制检测启停
  - 无屏幕时自动跳过GUI显示，仅通过串口和终端输出
  - 检测到目标时自动保存截图

用法：
  python3 test_camera_onnx_headless.py [camera_index]
"""

import cv2
import numpy as np
import time
import os
import sys
import threading
import queue
import onnxruntime as ort
import serial

# ========== 配置 ==========
SERIAL_PORT = "/dev/serial0"
SERIAL_BAUD = 115200

MODEL_PATH = "runs/detect/indoor_potted_plant_pi/weights/best.onnx"
IMSZ = 416
CONF = 0.25
NMS_IOU = 0.45
SAVE_INTERVAL = 30  # 检测到盆栽时每隔N帧保存一次


def check_display():
    """检查是否有可用的图形显示环境"""
    display = os.environ.get("DISPLAY", "")
    if not display:
        return False
    # 尝试连接X11 (仅判断环境变量不够，但headless场景下够用)
    return True


class PlantDetector:
    def __init__(self, model_path):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4

        self.sess = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        self._orig_w = 1
        self._orig_h = 1

    def detect(self, frame):
        h, w = frame.shape[:2]
        self._orig_w, self._orig_h = w, h

        img = cv2.resize(frame, (IMSZ, IMSZ))
        img = img[:, :, ::-1].transpose(2, 0, 1) / 255.0
        img = np.ascontiguousarray(img, dtype=np.float32)[np.newaxis]

        out = self.sess.run([self.out_name], {self.in_name: img})[0]
        return self._postprocess(out)

    def _postprocess(self, out):
        preds = out[0].T
        mask = preds[:, 4] > CONF
        if not np.any(mask):
            return []

        cx, cy, w, h = preds[mask, 0], preds[mask, 1], preds[mask, 2], preds[mask, 3]
        scores = preds[mask, 4]

        sx = self._orig_w / IMSZ
        sy = self._orig_h / IMSZ
        x1 = (cx - w / 2) * sx
        y1 = (cy - h / 2) * sy
        x2 = (cx + w / 2) * sx
        y2 = (cy + h / 2) * sy

        boxes = np.stack([x1, y1, x2, y2], axis=1)
        idxs = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), CONF, NMS_IOU)
        if len(idxs) == 0:
            return []

        idxs = idxs.flatten()
        return [(boxes[i].astype(int), float(scores[i])) for i in idxs]

    def draw(self, frame, detections):
        annotated = frame.copy()
        for box, conf in detections:
            cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            label = f"Plant {conf:.0%}"
            cv2.putText(
                annotated, label, (box[0], box[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
            )
        return annotated


def format_serial_data(frame_count, detections):
    """格式化串口数据: F=帧号,P=盆栽数,B=x1,y1,x2,y2,conf;...\n"""
    parts = [f"F={frame_count}", f"P={len(detections)}"]
    if detections:
        box_strs = []
        for box, conf in detections:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            box_strs.append(f"{x1},{y1},{x2},{y2},{conf:.2f}")
        parts.append("B=" + ";".join(box_strs))
    return ",".join(parts) + "\n"


def read_serial_command(ser):
    """非阻塞读取串口命令，返回 'START', 'STOP' 或 None"""
    if not ser or not ser.is_open:
        return None
    cmd = None
    while ser.in_waiting > 0:
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line == "START":
                cmd = "START"
            elif line == "STOP":
                cmd = "STOP"
        except Exception:
            break
    return cmd


def main(camera_index: int = 0):
    has_display = check_display()
    if has_display:
        os.environ.setdefault("DISPLAY", ":0")
    else:
        print("[INFO] 未检测到显示环境，以headless模式运行（不显示窗口）")

    print("=" * 60)
    print(f"ONNX 摄像头实时检测 (索引 {camera_index})")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] 模型不存在: {MODEL_PATH}")
        return

    # 初始化串口
    ser = None
    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.1)
        print(f"[OK] 串口已打开: {SERIAL_PORT} @ {SERIAL_BAUD}")
    except Exception as e:
        print(f"[WARN] 串口打开失败 ({e}), 继续运行但不发送串口数据")

    # ========== 新增：Web 控制器方向键命令接收 ==========
    # 命令接收队列（Web服务器通过stdin发送方向键命令）
    cmd_queue = queue.Queue()

    def stdin_reader():
        """后台线程读取stdin命令，不阻塞主循环"""
        while True:
            try:
                line = sys.stdin.readline()
                if line:
                    # 二进制模式下解码为字符串
                    if isinstance(line, bytes):
                        line = line.decode("utf-8", errors="ignore")
                    cmd = line.strip()
                    if cmd:
                        cmd_queue.put(cmd)
            except Exception:
                break

    # 启动stdin读取线程（daemon=True，主程序退出时自动结束）
    threading.Thread(target=stdin_reader, daemon=True).start()
    # ========== 新增结束 ==========

    print("加载模型...")
    detector = PlantDetector(MODEL_PATH)
    print("[OK] 模型加载完成")

    backend = cv2.CAP_V4L2
    cap = cv2.VideoCapture(camera_index, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print(f"[ERROR] 摄像头 {camera_index} 无法打开")
        if ser:
            ser.close()
        return

    save_dir = os.path.join("camera_captures", f"onnx_cam_{camera_index}")
    os.makedirs(save_dir, exist_ok=True)

    # 运行状态: True=检测中, False=暂停
    # 默认False: 启动后等待ESP32发送START命令
    running = False
    prev_running = False

    frame_count = 0
    detect_count = 0
    fps = 0.0
    prev_time = time.time()
    last_log = prev_time

    print(f"\n截图目录: {save_dir}/")
    print(f"串口: {SERIAL_PORT} @ {SERIAL_BAUD}")
    print("[INFO] stdin 命令接收线程已启动")
    print("控制方式:")
    print("  - ESP32 发送 START/STOP 控制检测启停")
    print("  - Web控制器 发送 w/a/s/d/8/2/4/6 控制方向")
    print("  - Ctrl+C 退出程序")
    print("=" * 60)

    try:
        while True:
            # ---- 读取 stdin 命令 (Web控制器发来的方向键) ----
            while not cmd_queue.empty():
                cmd = cmd_queue.get()
                if cmd == "START":
                    if not running:
                        running = True
                        print("[CMD] 收到 START -> 开始检测")
                elif cmd == "STOP":
                    if running:
                        running = False
                        print("[CMD] 收到 STOP -> 暂停检测")
                elif ser and ser.is_open:
                    try:
                        ser.write((cmd + "\n").encode("utf-8"))
                        print(f"[CMD] 串口发送(Web): {cmd}")
                    except Exception as e:
                        print(f"[WARN] 串口命令发送失败: {e}")

            # ---- 读取串口命令 (非阻塞) ----
            cmd = read_serial_command(ser)
            if cmd == "START":
                if not running:
                    running = True
                    print("[CMD] 收到 START -> 开始检测")
            elif cmd == "STOP":
                if running:
                    running = False
                    print("[CMD] 收到 STOP -> 暂停检测")

            # 状态变化时发送确认
            if running != prev_running and ser and ser.is_open:
                prev_running = running
                ack = "RUNNING\n" if running else "PAUSED\n"
                try:
                    ser.write(ack.encode("utf-8"))
                except Exception:
                    pass

            # ---- 读取摄像头 ----
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] 无法读取画面")
                time.sleep(0.5)
                continue

            frame_count += 1

            # ---- 如果处于STOP状态，跳过推理 ----
            if not running:
                # 每隔5秒打印一次心跳
                if time.time() - last_log >= 5.0:
                    last_log = time.time()
                    print(f"[{frame_count:5d}] PAUSED (等待 START 命令)")
                continue

            # ---- 推理 ----
            t0 = time.time()
            detections = detector.detect(frame)
            t1 = time.time()
            inf_time = t1 - t0

            # ---- 串口发送检测数据 ----
            if ser and ser.is_open:
                try:
                    msg = format_serial_data(frame_count, detections)
                    ser.write(msg.encode("utf-8"))
                except Exception as e:
                    print(f"  [WARN] 串口发送失败: {e}")

            # ---- FPS 计算 ----
            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 / dt

            num = len(detections)

            # ---- 终端输出（每秒一次） ----
            if now - last_log >= 1.0:
                last_log = now
                best = f"{max(d[1] for d in detections)*100:.0f}%" if detections else "N/A"
                status = "** DETECTED **" if num else "scanning"
                print(
                    f"[{frame_count:5d}] FPS={fps:4.1f}  "
                    f"Inf={inf_time*1000:4.0f}ms  "
                    f"Plants={num}  Best={best}  {status}"
                )

            # ---- 自动保存 ----
            if num > 0:
                detect_count += 1
                if detect_count % SAVE_INTERVAL == 1:
                    annotated = detector.draw(frame, detections)
                    path = os.path.join(save_dir, f"auto_{frame_count:06d}.jpg")
                    cv2.imwrite(path, annotated)
                    print(f"  [SAVE] {path}")

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C 退出")

    cap.release()
    cv2.destroyAllWindows()
    if ser and ser.is_open:
        ser.close()
        print("[OK] 串口已关闭")
    print(f"\n共处理 {frame_count} 帧")
    print(f"截图保存在: {save_dir}/")


if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(idx)
