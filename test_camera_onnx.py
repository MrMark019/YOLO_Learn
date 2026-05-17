"""
ONNX Runtime 摄像头实时检测脚本
专为树莓派优化，不依赖 torch
"""

import cv2
import numpy as np
import time
import os
import onnxruntime as ort

MODEL_PATH = "runs/detect/indoor_potted_plant_pi/weights/best.onnx"
IMSZ = 416      # 模型固定输入尺寸
CONF = 0.25
NMS_IOU = 0.45
SAVE_INTERVAL = 30  # 检测到盆栽时每隔 N 个检测帧保存一次


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

        # 预处理
        img = cv2.resize(frame, (IMSZ, IMSZ))
        img = img[:, :, ::-1].transpose(2, 0, 1) / 255.0  # BGR→RGB, HWC→CHW
        img = np.ascontiguousarray(img, dtype=np.float32)[np.newaxis]

        # 推理
        out = self.sess.run([self.out_name], {self.in_name: img})[0]

        # 后处理
        return self._postprocess(out)

    def _postprocess(self, out):
        # out: (1, 5, N) -> (N, 5)  cx,cy,w,h,conf (absolute coords on 416x416)
        preds = out[0].T

        mask = preds[:, 4] > CONF
        if not np.any(mask):
            return []

        cx, cy, w, h = preds[mask, 0], preds[mask, 1], preds[mask, 2], preds[mask, 3]
        scores = preds[mask, 4]

        # Scale from model input (416) to original frame size
        sx = self._orig_w / IMSZ
        sy = self._orig_h / IMSZ
        x1 = (cx - w / 2) * sx
        y1 = (cy - h / 2) * sy
        x2 = (cx + w / 2) * sx
        y2 = (cy + h / 2) * sy

        boxes = np.stack([x1, y1, x2, y2], axis=1)

        idxs = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(), CONF, NMS_IOU
        )
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


def main(camera_index: int = 0):
    os.environ.setdefault("DISPLAY", ":0")

    print("=" * 60)
    print(f"ONNX 摄像头实时检测 (索引 {camera_index})")
    print("=" * 60)
    print("按键: q/ESC=退出  s=保存截图")

    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型不存在 {MODEL_PATH}")
        return

    print("加载模型...")
    detector = PlantDetector(MODEL_PATH)
    print("模型加载完成")

    backend = cv2.CAP_V4L2
    cap = cv2.VideoCapture(camera_index, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print(f"错误: 摄像头 {camera_index} 无法打开")
        return

    save_dir = os.path.join("camera_captures", f"onnx_cam_{camera_index}")
    os.makedirs(save_dir, exist_ok=True)

    win_name = "Plant Detection (ONNX) - q to quit"
    frame_count = 0
    detect_count = 0
    fps = 0.0
    prev_time = time.time()
    last_log = prev_time

    print(f"\n截图目录: {save_dir}/")
    print("摄像头已启动！请对准盆栽。\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("错误: 无法读取画面")
                break

            frame_count += 1
            t0 = time.time()

            detections = detector.detect(frame)
            annotated = detector.draw(frame, detections)

            t1 = time.time()
            inf_time = t1 - t0

            # FPS
            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 / dt

            num = len(detections)

            # 窗口叠加信息
            lines = [
                f"FPS: {fps:.1f}  |  Infer: {inf_time*1000:.0f}ms",
                f"Plants: {num}",
            ]
            for i, line in enumerate(lines):
                cv2.putText(
                    annotated, line, (10, 30 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )

            # 终端输出（每秒一次）
            if now - last_log >= 1.0:
                last_log = now
                best = f"{max(d[1] for d in detections)*100:.0f}%" if detections else "N/A"
                status = "** DETECTED **" if num else "scanning"
                print(
                    f"[{frame_count:5d}] FPS={fps:4.1f}  "
                    f"Inf={inf_time*1000:4.0f}ms  "
                    f"Plants={num}  Best={best}  {status}"
                )

            # 自动保存
            if num > 0:
                detect_count += 1
                if detect_count % SAVE_INTERVAL == 1:
                    path = os.path.join(save_dir, f"auto_{frame_count:06d}.jpg")
                    cv2.imwrite(path, annotated)
                    print(f"  >> 保存: {path}")

            # 显示窗口
            cv2.imshow(win_name, annotated)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                print("\n用户退出")
                break

            if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                print("\n窗口关闭")
                break

            if key == ord("s"):
                path = os.path.join(save_dir, f"manual_{frame_count:06d}.jpg")
                cv2.imwrite(path, annotated)
                print(f"  >> 手动保存: {path}")

    except KeyboardInterrupt:
        print("\nCtrl+C 退出")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n共处理 {frame_count} 帧")
    print(f"截图在: {save_dir}/")


if __name__ == "__main__":
    import sys

    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(idx)
