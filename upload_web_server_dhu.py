#!/usr/bin/env python3
"""上传 web_server 到树莓派 (DHU-1X 网络)"""
import paramiko
import sys
from pathlib import Path

HOST = "10.206.190.162"
USER = "mark"
PASSWORD = "Mark0602"
PORT = 22

LOCAL_DIR = Path("D:/MarkLab/YOLO_Learn/web_server")
REMOTE_BASE = "/home/mark/YOLO_Learn"

def upload_file(sftp, local_path, remote_path):
    sftp.put(str(local_path), remote_path)
    print(f"[OK] {local_path.name} -> {remote_path}")

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, PORT, USER, PASSWORD, timeout=15)
    sftp = client.open_sftp()

    # 创建远程目录
    dirs = [
        f"{REMOTE_BASE}/web_server",
        f"{REMOTE_BASE}/web_server/templates",
        f"{REMOTE_BASE}/web_server/static/css",
        f"{REMOTE_BASE}/web_server/static/js",
    ]
    for d in dirs:
        try:
            sftp.mkdir(d)
        except IOError:
            pass  # 已存在

    files = [
        (LOCAL_DIR / "app.py", f"{REMOTE_BASE}/web_server/app.py"),
        (LOCAL_DIR / "yolo_manager.py", f"{REMOTE_BASE}/web_server/yolo_manager.py"),
        (LOCAL_DIR / "templates" / "index.html", f"{REMOTE_BASE}/web_server/templates/index.html"),
        (LOCAL_DIR / "static" / "css" / "style.css", f"{REMOTE_BASE}/web_server/static/css/style.css"),
        (LOCAL_DIR / "static" / "js" / "controller.js", f"{REMOTE_BASE}/web_server/static/js/controller.js"),
    ]

    for local, remote in files:
        upload_file(sftp, local, remote)

    sftp.close()
    client.close()
    print("[DONE] 所有文件上传完成!")

if __name__ == "__main__":
    main()
