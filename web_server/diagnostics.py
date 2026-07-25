"""
硬件自检模块
通过 vcgencmd 和 sysfs 检测树莓派硬件状态
"""

import subprocess
import os


def _vcgencmd(param):
    """执行 vcgencmd 命令，返回输出"""
    try:
        r = subprocess.run(["vcgencmd", param], capture_output=True, text=True, timeout=5)
        return r.stdout.strip()
    except Exception:
        return ""


def _read_file(path):
    """安全读取 sysfs 文件"""
    try:
        with open(path, "r") as f:
            return f.read().strip()
    except Exception:
        return ""


def check_throttled():
    """检测 throttled 状态位"""
    out = _vcgencmd("get_throttled")
    # throttled=0x50005 -> 0x50005
    if "=" in out:
        val = int(out.split("=")[1], 16)
    else:
        return {"error": "无法读取 throttled", "issues": []}

    issues = []
    # bit 0: 当前欠压
    undervolt_now = bool(val & 0x1)
    # bit 2: 当前降频
    freq_cap_now = bool(val & 0x4)
    # bit 16: 历史欠压
    undervolt_past = bool(val & 0x10000)
    # bit 18: 历史降频
    freq_cap_past = bool(val & 0x40000)

    if undervolt_now:
        issues.append({"level": "danger", "msg": "CPU 欠压中！请检查电源/USB线"})
    else:
        if undervolt_past:
            issues.append({"level": "warning", "msg": "曾发生 CPU 欠压，建议检查电源"})

    if freq_cap_now:
        issues.append({"level": "danger", "msg": "CPU 频率被强制限制！"})
    else:
        if freq_cap_past:
            issues.append({"level": "warning", "msg": "曾发生 CPU 频率限制"})

    return {
        "throttled_raw": out,
        "undervolt_now": undervolt_now,
        "freq_cap_now": freq_cap_now,
        "undervolt_past": undervolt_past,
        "freq_cap_past": freq_cap_past,
        "issues": issues,
    }


def check_clock():
    """检测 ARM 频率"""
    out = _vcgencmd("measure_clock arm")
    if "=" in out:
        hz = int(out.split("=")[1])
        mhz = hz / 1_000_000
    else:
        return {"error": "无法读取频率", "issues": []}

    issues = []
    level = "ok"
    if mhz < 1000:
        issues.append({"level": "danger", "msg": f"CPU 频率极低 ({mhz:.0f}MHz)"})
        level = "danger"
    elif mhz < 1600:
        issues.append({"level": "warning", "msg": f"CPU 频率偏低 ({mhz:.0f}MHz)"})
        level = "warning"

    return {"freq_mhz": round(mhz), "level": level, "issues": issues}


def check_temp():
    """检测 CPU 温度"""
    out = _vcgencmd("measure_temp")
    if "=" in out:
        temp_str = out.split("=")[1].replace("'C", "").strip()
        try:
            temp = float(temp_str)
        except ValueError:
            return {"error": "无法解析温度", "issues": []}
    else:
        return {"error": "无法读取温度", "issues": []}

    issues = []
    level = "ok"
    if temp > 85:
        issues.append({"level": "danger", "msg": f"CPU 温度过高 ({temp:.0f}°C)，可能降频"})
        level = "danger"
    elif temp > 80:
        issues.append({"level": "warning", "msg": f"CPU 温度偏高 ({temp:.0f}°C)"})
        level = "warning"

    return {"temp_c": round(temp, 1), "level": level, "issues": issues}


def check_governor():
    """检测 CPU 调度器"""
    gov = _read_file("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if not gov:
        return {"error": "无法读取 governor", "issues": []}

    issues = []
    if gov != "performance":
        issues.append({"level": "warning", "msg": f"CPU 调度器为 {gov}（建议 performance）"})

    return {"governor": gov, "issues": issues}


def check_camera():
    """检测摄像头"""
    cam = "/dev/video0"
    issues = []
    if os.path.exists(cam):
        return {"camera_ok": True, "issues": []}
    else:
        issues.append({"level": "danger", "msg": "摄像头未检测到 (无 /dev/video0)"})
        return {"camera_ok": False, "issues": issues}


def check_all():
    """执行全部自检，返回汇总结果"""
    results = {
        "throttled": check_throttled(),
        "clock": check_clock(),
        "temp": check_temp(),
        "governor": check_governor(),
        "camera": check_camera(),
    }

    # 收集所有 issues
    all_issues = []
    has_danger = False
    has_warning = False
    for key, result in results.items():
        for issue in result.get("issues", []):
            if issue["level"] == "danger":
                has_danger = True
            elif issue["level"] == "warning":
                has_warning = True
            all_issues.append(issue)

    # 摘要
    if has_danger:
        summary = "硬件存在严重问题，可能影响检测性能"
    elif has_warning:
        summary = "硬件存在警告，建议检查"
    else:
        summary = "硬件状态正常"

    results["issues"] = all_issues
    results["summary"] = summary
    results["all_ok"] = not has_danger and not has_warning
    results["has_danger"] = has_danger
    results["has_warning"] = has_warning

    return results


if __name__ == "__main__":
    import json
    print(json.dumps(check_all(), indent=2, ensure_ascii=False))
