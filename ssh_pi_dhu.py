"""
树莓派 SSH 连接脚本 - DHU-1X 校园网
用法:
  python ssh_pi_dhu.py            # 交互式 shell
  python ssh_pi_dhu.py "ls -la"   # 执行单条命令
"""

import sys
import paramiko

HOST = "10.206.190.162"
USER = "mark"
PASSWORD = "Mark0602"
PORT = 22


def run_command(cmd=None):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, PORT, USER, PASSWORD, timeout=10)

    if cmd:
        stdin, stdout, stderr = client.exec_command(cmd)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        client.close()
        if err.strip():
            return err.strip()
        return out
    else:
        channel = client.invoke_shell()
        print(f"[已连接 {USER}@{HOST}]")
        import threading

        def writeall():
            while True:
                data = channel.recv(1024)
                if not data:
                    break
                print(data.decode("utf-8", errors="replace"), end="", flush=True)

        writer = threading.Thread(target=writeall)
        writer.daemon = True
        writer.start()

        try:
            while True:
                cmd = input()
                channel.send(cmd + "\n")
        except (EOFError, KeyboardInterrupt):
            print("\n[断开连接]")
        finally:
            channel.close()
            client.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = " ".join(sys.argv[1:])
        print(run_command(cmd))
    else:
        run_command()
