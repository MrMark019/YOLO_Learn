"""
YOLO Web 控制器 - Flask + SocketIO 主程序

功能：
  - 提供 HTTP 页面（控制面板）
  - 通过 WebSocket 实时通信
  - 管理 YOLO 子进程的启动/停止
  - 接收前端命令并转发到 YOLO 进程

用法：
    cd web_server && python app.py
    # 浏览器访问 http://localhost:5000

树莓派部署：
    python3 app.py
    # 同一 WiFi 下的 PC/手机访问 http://树莓派IP:5000
"""

import sys
import os
from pathlib import Path

from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit

# yolo_manager 在同级目录，直接导入即可
from yolo_manager import YoloManager
from diagnostics import check_all

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = "yolo-web-controller-secret-2024"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# 全局 YOLO 管理器实例
yolo_mgr = None


def create_manager():
    """创建/重建 YoloManager 实例"""
    global yolo_mgr
    yolo_mgr = YoloManager(
        on_log=_on_log,
        on_status=_on_status,
        use_mock=False,  # 树莓派部署：使用真实 YOLO 脚本
        script_path="/home/mark/YOLO_Learn/test_camera_onnx_headless.py",
    )


def _on_log(msg):
    """日志回调 → 推送给所有连接的客户端"""
    socketio.emit("log", msg)


def _on_status(status):
    """状态回调 → 推送给所有连接的客户端"""
    socketio.emit("status", status)


# ==================== HTTP 路由 ====================

@app.route("/")
def index():
    return render_template("index.html")


# ==================== WebSocket 事件 ====================

@socketio.on("connect")
def handle_connect():
    """客户端连接"""
    emit("connected", {"message": "已连接到 YOLO Web 控制器"})
    # 发送当前 YOLO 状态
    emit("status", {
        "running": yolo_mgr.is_running() if yolo_mgr else False,
        "fps": 0.0,
        "plants": 0,
        "frame": 0,
    })
    print(f"[WS] 客户端连接: {request.sid if 'request' in dir() else 'unknown'}")


@socketio.on("disconnect")
def handle_disconnect():
    """客户端断开"""
    print(f"[WS] 客户端断开")


@socketio.on("yolo_control")
def handle_yolo_control(data):
    """
    接收启动/停止 YOLO 的命令
    data: {"action": "start"} 或 {"action": "stop"}
    """
    action = data.get("action", "")
    if action == "start":
        if yolo_mgr and yolo_mgr.is_running():
            emit("ack", {
                "action": "start",
                "success": False,
                "error": "YOLO 进程已在运行，请先停止后再启动",
            })
            return
        if not yolo_mgr:
            create_manager()

        # ---- 硬件自检 ----
        diag = check_all()
        socketio.emit("diagnostics", diag)
        # ---- 自检结束 ----

        success = yolo_mgr.start()
        if success:
            # 启动进程后，自动发送 START 命令开始检测
            yolo_mgr.send_command("START")
            emit("ack", {"action": "start", "success": True})
        else:
            emit("ack", {
                "action": "start",
                "success": False,
                "error": "启动失败，请查看日志",
            })
    elif action == "stop":
        if yolo_mgr and yolo_mgr.is_running():
            # 先发送 STOP 暂停检测，再终止进程
            yolo_mgr.send_command("STOP")
            success = yolo_mgr.stop()
            emit("ack", {"action": "stop", "success": success})
        else:
            emit("ack", {
                "action": "stop",
                "success": False,
                "error": "YOLO 未启动",
            })
    else:
        emit("ack", {
            "action": action,
            "success": False,
            "error": "未知命令",
        })


@socketio.on("diagnostics")
def handle_diagnostics():
    """手动触发硬件自检"""
    diag = check_all()
    socketio.emit("diagnostics", diag)


@socketio.on("car_command")
def handle_car_command(data):
    """
    接收小车方向键命令
    data: {"key": "w"}
    有效键: w, a, s, d
    """
    key = data.get("key", "")
    if key in ("w", "a", "s", "d", "q", "e", "x"):
        if yolo_mgr and yolo_mgr.is_running():
            yolo_mgr.send_command(key)
            emit("ack", {"command": key, "target": "car", "success": True})
        else:
            emit("ack", {
                "command": key,
                "target": "car",
                "success": False,
                "error": "YOLO 未运行",
            })
    else:
        emit("ack", {
            "command": key,
            "target": "car",
            "success": False,
            "error": "无效命令",
        })


@socketio.on("camera_command")
def handle_camera_command(data):
    """
    接收摄像头方向键命令
    data: {"key": "8"}
    有效键: 8, 2, 4, 6
    """
    key = data.get("key", "")
    if key in ("8", "2", "4", "6"):
        if yolo_mgr and yolo_mgr.is_running():
            yolo_mgr.send_command(key)
            emit("ack", {"command": key, "target": "camera", "success": True})
        else:
            emit("ack", {
                "command": key,
                "target": "camera",
                "success": False,
                "error": "YOLO 未运行",
            })
    else:
        emit("ack", {
            "command": key,
            "target": "camera",
            "success": False,
            "error": "无效命令",
        })


# ==================== 主入口 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("YOLO Web 控制器 - Flask + SocketIO")
    print("=" * 60)
    print("Flask 服务器将启动在 http://0.0.0.0:5000")
    print("请在浏览器中访问上述地址")
    print("按 Ctrl+C 停止服务器")
    print("=" * 60)

    create_manager()

    # 0.0.0.0 允许同一网络中的其他设备访问（包括树莓派部署时）
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)
