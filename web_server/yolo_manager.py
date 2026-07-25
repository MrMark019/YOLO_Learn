"""
YOLO 子进程管理器

功能：
  - 启动/停止 YOLO 检测进程（子进程）
  - 实时捕获 stdout/stderr 日志
  - 解析日志中的状态信息（FPS、Plants、Frame 等）
  - 通过 stdin 向 YOLO 进程发送方向键命令
  - 通过回调将日志和状态推送给 SocketIO 服务器

用法（在 Flask-SocketIO 上下文中）：
    from yolo_manager import YoloManager
    manager = YoloManager(
        on_log=lambda msg: socketio.emit("log", msg),
        on_status=lambda st: socketio.emit("status", st),
    )
    manager.start()
    manager.send_command("w")      # 方向键
    manager.stop()
"""

import subprocess
import threading
import time
import re
from pathlib import Path


class YoloManager:
    """
    YOLO 检测进程管理器
    """

    def __init__(self, script_path=None, on_log=None, on_status=None, use_mock=False):
        """
        :param script_path: YOLO 脚本路径（默认使用本地 mock 或项目根目录下的脚本）
        :param on_log: 回调函数，接收日志消息 dict {"message": str, "type": str}
        :param on_status: 回调函数，接收状态 dict {"running": bool, "fps": float, "plants": int, "frame": int}
        :param use_mock: 是否使用本地 mock 脚本（不依赖树莓派环境）
        """
        self.script_path = script_path
        self.on_log = on_log or (lambda x: None)
        self.on_status = on_status or (lambda x: None)
        self.use_mock = use_mock

        self.process = None
        self._read_thread = None
        self._stop_event = threading.Event()

        # 解析用的正则
        self._log_patterns = [
            # 匹配标准行: [ 676] FPS=4.1  Inf=230ms  Plants=1  Best=37%  ** DETECTED **
            re.compile(
                r"\[\s*(\d+)\]\s+FPS=([\d.]+)\s+Inf=([\d]+)ms\s+Plants=(\d+)\s+Best=([^\s]+)\s+(.*)"
            ),
            # 匹配 [CMD] 串口发送: w
            re.compile(r"\[CMD\].*串口发送:\s*(\w)"),
            # 匹配 [INFO]/[WARN]/[ERROR]
            re.compile(r"\[(INFO|WARN|ERROR)\]\s*(.*)"),
        ]

    def _get_script_path(self):
        """确定要运行的脚本路径"""
        if self.script_path:
            return self.script_path

        repo_root = Path(__file__).resolve().parent.parent  # web_server/ 的父目录

        if self.use_mock:
            # 本地测试：使用 mock_yolo.py
            mock_path = Path(__file__).resolve().parent / "mock_yolo.py"
            if mock_path.exists():
                return str(mock_path)
            # fallback 到项目根目录下的 mock_yolo.py
            fallback = repo_root / "mock_yolo.py"
            if fallback.exists():
                return str(fallback)
            raise FileNotFoundError("未找到 mock_yolo.py，请先创建")

        # 树莓派部署：使用项目根目录下的 headless 脚本
        headless = repo_root / "test_camera_onnx_headless.py"
        if headless.exists():
            return str(headless)
        raise FileNotFoundError(f"未找到 YOLO 脚本: {headless}")

    def start(self):
        """启动 YOLO 检测进程"""
        if self.process is not None:
            if self.process.poll() is None:
                self._emit_log("[WARN] YOLO 进程已在运行", "warning")
                return False
            else:
                # 进程已退出但引用还在，清理旧引用
                self._emit_log("[INFO] 清理旧进程引用", "info")
                self.process = None

        script = self._get_script_path()
        self._emit_log(f"[INFO] 启动 YOLO 进程: {script}", "info")

        # 检测 venv 中的 Python（树莓派部署时使用）
        import shutil
        script_dir = Path(script).parent
        venv_python = script_dir / "venv" / "bin" / "python3"
        if not venv_python.exists():
            venv_python = script_dir.parent / "venv" / "bin" / "python3"
        python_cmd = str(venv_python) if venv_python.exists() else "python3"
        self._emit_log(f"[INFO] 使用 Python: {python_cmd}", "info")

        try:
            # 创建子进程，捕获 stdout/stderr，并通过 stdin 输入命令
            # 使用二进制模式避免缓冲问题，然后手动解码
            # cwd 设置为脚本所在目录，确保相对路径（如 runs/detect/...）正确
            self.process = subprocess.Popen(
                [python_cmd, "-u", script, "0"],  # -u 无缓冲模式，确保实时输出
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # 合并 stderr 到 stdout
                stdin=subprocess.PIPE,     # 用于发送命令
                text=False,                # 二进制模式，避免缓冲问题
                bufsize=0,                 # 无缓冲
                cwd=str(script_dir),       # 工作目录设为脚本所在目录
            )
        except FileNotFoundError:
            # 树莓派上 python3 可能叫 python
            self._emit_log("[WARN] python3 未找到，尝试 python", "warning")
            self.process = subprocess.Popen(
                ["python", "-u", script, "0"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE,
                text=False,
                bufsize=0,
                cwd=str(script_dir),
            )

        self._stop_event.clear()
        self._read_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._read_thread.start()

        # 通知前端状态
        self._emit_status({"running": True, "fps": 0.0, "plants": 0, "frame": 0})
        self._emit_log("[INFO] YOLO 进程已启动", "success")
        return True

    def stop(self):
        """停止 YOLO 检测进程"""
        if self.process is None:
            self._emit_log("[WARN] YOLO 进程未运行", "warning")
            return False

        self._emit_log("[INFO] 正在停止 YOLO 进程...", "info")
        self._stop_event.set()

        # 先尝试优雅终止（发送 STOP 命令，让 YOLO 自行退出）
        try:
            if self.process.stdin and not self.process.stdin.closed:
                self.process.stdin.write("\n")
                self.process.stdin.flush()
        except Exception:
            pass

        # 给进程 2 秒时间自行退出
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._emit_log("[WARN] 进程未响应，强制终止", "warning")
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()

        self.process = None
        self._emit_status({"running": False, "fps": 0.0, "plants": 0, "frame": 0})
        self._emit_log("[INFO] YOLO 进程已停止", "success")
        return True

    def send_command(self, cmd):
        """向 YOLO 进程发送方向键命令（通过 stdin）"""
        if self.process is None or self.process.poll() is not None:
            self._emit_log(f"[WARN] 无法发送命令 '{cmd}'，YOLO 进程未运行", "warning")
            return False

        if self.process.stdin and not self.process.stdin.closed:
            try:
                self.process.stdin.write((cmd + "\n").encode("utf-8"))
                self.process.stdin.flush()
                self._emit_log(f"[CMD] 已发送命令: {cmd}", "command")
                return True
            except Exception as e:
                self._emit_log(f"[ERROR] 发送命令失败: {e}", "error")
                return False
        else:
            self._emit_log("[WARN] stdin 已关闭，无法发送命令", "warning")
            return False

    def _reader_loop(self):
        """后台线程：持续读取子进程 stdout"""
        if self.process is None or self.process.stdout is None:
            return

        for line in self.process.stdout:
            if self._stop_event.is_set():
                break
            if not line:
                continue

            # 二进制模式解码
            if isinstance(line, bytes):
                try:
                    line = line.decode("utf-8", errors="replace")
                except Exception:
                    continue

            line = line.rstrip("\n\r")
            if not line:
                continue

            # 解析日志行，提取状态信息
            parsed = self._parse_line(line)
            if parsed:
                self._emit_status(parsed)

            # 推送给前端
            self._emit_log(line, "info")

        # 读取结束，进程可能已退出
        if self.process is not None:
            return_code = self.process.poll()
            if return_code is not None and not self._stop_event.is_set():
                self._emit_log(f"[WARN] YOLO 进程异常退出 (code={return_code})", "error")
                self._emit_status({"running": False, "fps": 0.0, "plants": 0, "frame": 0})
                self.process = None

    def _parse_line(self, line):
        """解析日志行，提取状态信息，返回 dict 或 None"""
        # 匹配标准检测行: [ 676] FPS=4.1  Inf=230ms  Plants=1  Best=37%  ** DETECTED **
        m = re.search(
            r"\[\s*(\d+)\]\s+FPS=([\d.]+)\s+Inf=([\d]+)ms\s+Plants=(\d+)\s+Best=([^\s]+)",
            line,
        )
        if m:
            return {
                "running": True,
                "frame": int(m.group(1)),
                "fps": float(m.group(2)),
                "inf_ms": int(m.group(3)),
                "plants": int(m.group(4)),
                "best": m.group(5),
            }

        # 匹配 [INFO] 或 [CMD] 等标记行
        if "收到 START" in line or "RUNNING" in line:
            return {"running": True}
        if "收到 STOP" in line or "PAUSED" in line:
            return {"running": False}

        return None

    def _emit_log(self, message, msg_type="info"):
        """通过回调推送日志到前端"""
        self.on_log({"message": message, "type": msg_type})

    def _emit_status(self, status):
        """通过回调推送状态到前端"""
        self.on_status(status)

    def is_running(self):
        """返回 YOLO 进程是否正在运行"""
        return self.process is not None and self.process.poll() is None


if __name__ == "__main__":
    # 独立测试
    def on_log(msg):
        print(f"[LOG] {msg}")

    def on_status(st):
        print(f"[STATUS] {st}")

    manager = YoloManager(on_log=on_log, on_status=on_status, use_mock=True)
    manager.start()

    time.sleep(1)
    manager.send_command("w")
    time.sleep(0.5)
    manager.send_command("a")
    time.sleep(2)
    manager.stop()
