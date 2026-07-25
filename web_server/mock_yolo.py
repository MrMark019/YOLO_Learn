"""
Mock YOLO 检测脚本（本地测试用）

模拟树莓派上 test_camera_onnx_headless.py 的输出行为，
用于在 Windows 本地开发环境中测试 Web 控制面板。

模拟输出：
  - 模拟摄像头读取和 ONNX 推理
  - 模拟串口数据发送
  - 从 stdin 读取方向键命令并打印回显
  - 输出与真实 YOLO 程序一致的日志格式

用法：
    python mock_yolo.py 0

预期输入（通过 stdin）：
    w, a, s, d        -> 小车方向键
    8, 2, 4, 6        -> 摄像头方向键
"""

import sys
import time
import random
import threading
import queue


def mock_yolo_headless(camera_index=0):
    """模拟 YOLO headless 模式运行"""

    print("=" * 60)
    print(f"ONNX 摄像头实时检测 (索引 {camera_index}) - [MOCK MODE]")
    print("=" * 60)

    # 模拟状态
    running = False
    prev_running = False
    frame_count = 0
    detect_count = 0
    fps = 0.0
    prev_time = time.time()
    last_log = prev_time

    # 命令接收队列
    cmd_queue = queue.Queue()

    def stdin_reader():
        """后台线程读取 stdin 命令"""
        while True:
            try:
                line = sys.stdin.readline()
                if line:
                    cmd = line.strip()
                    if cmd:
                        cmd_queue.put(cmd)
            except Exception:
                break

    threading.Thread(target=stdin_reader, daemon=True).start()

    print(f"\n截图目录: camera_captures/mock_cam_{camera_index}/")
    print("串口: /dev/serial0 @ 115200")
    print("控制方式:")
    print("  - 从 stdin 接收方向键命令")
    print("  - Ctrl+C 退出程序")
    print("=" * 60)

    try:
        while True:
            # ---- 读取 stdin 命令 ----
            while not cmd_queue.empty():
                cmd = cmd_queue.get()
                if cmd.upper() == "START":
                    running = True
                    print("[CMD] 收到 START -> 开始检测")
                elif cmd.upper() == "STOP":
                    running = False
                    print("[CMD] 收到 STOP -> 暂停检测")
                else:
                    # 方向键命令：w/a/s/d 或 8/2/4/6
                    print(f"[CMD] 串口发送: {cmd}")

            # 状态变化确认
            if running != prev_running:
                prev_running = running
                ack = "RUNNING\n" if running else "PAUSED\n"
                print(f"[INFO] 状态确认: {ack.strip()}")

            # 模拟帧读取
            time.sleep(0.23)  # 模拟 ~230ms 推理时间
            frame_count += 1

            # 如果 STOP 状态，仅打印心跳
            if not running:
                now = time.time()
                if now - last_log >= 5.0:
                    last_log = now
                    print(f"[{frame_count:5d}] PAUSED (等待 START 命令)")
                continue

            # ---- 模拟推理 ----
            t0 = time.time()
            # 随机模拟检测结果 (70% 概率检测到)
            has_plant = random.random() > 0.3
            num = 1 if has_plant else 0
            t1 = time.time()
            inf_time = t1 - t0

            # ---- 模拟串口发送检测数据 ----
            if has_plant:
                x1 = random.randint(50, 200)
                y1 = random.randint(20, 100)
                x2 = x1 + random.randint(100, 300)
                y2 = y1 + random.randint(150, 300)
                conf = random.uniform(0.25, 0.75)
                msg = f"F={frame_count},P=1,B={x1},{y1},{x2},{y2},{conf:.2f}\n"
                print(f"[INFO] 串口发送: {msg.strip()}")
            else:
                msg = f"F={frame_count},P=0\n"

            # ---- 模拟 FPS ----
            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 / dt

            # ---- 模拟终端输出（每秒一次）----
            if now - last_log >= 1.0:
                last_log = now
                best = f"{random.uniform(0.25, 0.75)*100:.0f}%" if has_plant else "N/A"
                status = "** DETECTED **" if num else "scanning"
                print(
                    f"[{frame_count:5d}] FPS={fps:4.1f}  "
                    f"Inf={inf_time*1000:4.0f}ms  "
                    f"Plants={num}  Best={best}  {status}"
                )

            # ---- 模拟自动保存 ----
            if num > 0:
                detect_count += 1
                if detect_count % 30 == 1:
                    print(f"  [SAVE] camera_captures/mock_cam_{camera_index}/auto_{frame_count:06d}.jpg")

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C 退出")

    print(f"\n共处理 {frame_count} 帧")


if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    mock_yolo_headless(idx)
