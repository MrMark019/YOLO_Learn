/**
 * YOLO Web 控制器 - 前端交互逻辑
 *
 * 功能：
 *   - SocketIO 连接管理
 *   - 方向键按钮事件（鼠标/触摸）
 *   - 键盘快捷键（WASD、方向键）
 *   - YOLO 启动/停止控制
 *   - 实时日志接收与显示
 *   - 状态栏更新
 */

(function() {
    "use strict";

    // ==================== DOM 引用 ====================
    const els = {
        connStatus: document.getElementById("conn-status"),
        yoloStatus: document.getElementById("yolo-status"),
        liveStats: document.getElementById("live-stats"),
        btnStart: document.getElementById("btn-start"),
        btnStop: document.getElementById("btn-stop"),
        btnClearLog: document.getElementById("btn-clear-log"),
        btnPauseLog: document.getElementById("btn-pause-log"),
        logContainer: document.getElementById("log-container"),
        footerInfo: document.getElementById("footer-info"),
        videoContainer: document.getElementById("video-container"),
        videoPlaceholder: document.getElementById("video-placeholder"),
        videoStream: document.getElementById("video-stream"),
        diagBanner: document.getElementById("diag-banner"),
        diagInfo: document.getElementById("diag-info"),
        diagTemp: document.getElementById("diag-temp"),
        diagFreq: document.getElementById("diag-freq"),
        diagGov: document.getElementById("diag-gov"),
        diagCam: document.getElementById("diag-cam"),
        btnChecks: document.getElementById("btn-checks"),
        infoPlaceholder: document.getElementById("info-placeholder"),
        coordData: document.getElementById("coord-data"),
    };

    let mjpegTimer = null;
    const MJPEG_URL = "http://" + window.location.hostname + ":5001/stream";

    let infoToggle = false;
    const INFO_DEFAULT = '<p>植物类型: --</p><p>浇水量: --</p><p>其他信息: --</p>';
    const INFO_PLANT1 = '<div class="plant-card"><h4>绿萝</h4><p>浇水量: 300ml</p><p>喜温暖湿润、通风良好环境，耐阴忌强光；耐寒性弱（10℃以上可正常生长）；观叶植物，极少开花；汁液有毒，勿误食；及时剪除枯黄藤蔓，保持株型；适合水培或疏松肥沃、排水良好的微酸性土壤种植</p></div>';
    const INFO_PLANT2 = '<div class="plant-card"><h4>多肉植物</h4><p>浇水量: 150ml</p><p>喜阳光充足、通风良好环境，耐干旱忌潮湿积水；耐寒性弱（5℃以上可安全越冬，10℃以上正常生长）；花期因品种而异，多集中在春夏季；部分品种汁液有毒，勿误食；花后剪去残花茎，避免消耗养分；适合疏松透气、排水良好的颗粒土种植</p></div>';
    const INFO_PLANTS = INFO_PLANT1 + INFO_PLANT2;

    // ==================== 状态 ====================
    let socket = null;
    let yoloRunning = false;
    let logPaused = false;
    let logLines = [];
    const MAX_LOG_LINES = 300;

    // 按钮状态映射（防止重复触发）
    const activeKeys = new Set();

    // ==================== SocketIO 初始化 ====================
    function initSocket() {
        socket = io({
            transports: ["websocket", "polling"],
            reconnection: true,
            reconnectionAttempts: 10,
            reconnectionDelay: 2000,
        });

        socket.on("connect", () => {
            updateConnStatus(true);
            updateFooter("SocketIO 已连接");
        });

        socket.on("disconnect", () => {
            updateConnStatus(false);
            updateFooter("SocketIO 已断开，尝试重连...");
        });

        socket.on("connect_error", (err) => {
            updateConnStatus(false);
            updateFooter(`连接错误: ${err.message}`);
        });

        socket.on("connected", (data) => {
            appendLog(data.message, "success");
        });

        socket.on("log", (data) => {
            if (!logPaused) {
                appendLog(data.message, data.type || "info");
            }
            // 解析 [DATA] 消息更新坐标显示
            var msg = data.message || "";
            if (msg.startsWith("[DATA] ")) {
                updateCoords(msg.substring(7));
            }
        });

        socket.on("status", (data) => {
            updateStatus(data);
        });

        socket.on("ack", (data) => {
            if (data.success) {
                appendLog(`[ACK] ${data.target || ""} ${data.command || data.action} 已执行`, "success");
            } else {
                appendLog(`[ERR] ${data.error || "命令失败"}`, "error");
            }
        });

        socket.on("diagnostics", (data) => {
            updateDiagnostics(data);
        });
    }

    // ==================== UI 更新 ====================
    function updateConnStatus(connected) {
        if (connected) {
            els.connStatus.textContent = "已连接";
            els.connStatus.className = "badge connected";
        } else {
            els.connStatus.textContent = "未连接";
            els.connStatus.className = "badge disconnected";
        }
    }

    function updateStatus(data) {
        yoloRunning = data.running || false;

        if (yoloRunning) {
            els.yoloStatus.textContent = "识别运行中";
            els.yoloStatus.className = "badge running";
            els.btnStart.disabled = true;
            els.btnStop.disabled = false;
        } else {
            els.yoloStatus.textContent = "识别已停止";
            els.yoloStatus.className = "badge stopped";
            els.btnStart.disabled = false;
            els.btnStop.disabled = true;
        }

        const fps = data.fps !== undefined ? data.fps.toFixed(1) : "--";
        const plants = data.plants !== undefined ? data.plants : "--";
        const frame = data.frame !== undefined ? data.frame : "--";
        els.liveStats.textContent = `FPS: ${fps} | Plants: ${plants} | Frame: ${frame}`;

        // MJPEG 视频流：识别运行时 <img> 标签加载视频，停止时显示占位
        if (yoloRunning) {
            if (els.videoStream.src !== MJPEG_URL) {
                els.videoStream.src = MJPEG_URL;
            }
            els.videoStream.style.display = "block";
            els.videoPlaceholder.style.display = "none";
        } else {
            els.videoStream.style.display = "none";
            els.videoPlaceholder.style.display = "flex";
            // 延迟一下再清空 src，避免立即停止流时闪烁
            clearTimeout(mjpegTimer);
            mjpegTimer = setTimeout(function() {
                els.videoStream.src = "";
            }, 500);
        }
    }

    function updateCoords(msg) {
        // msg 格式: F=6,P=1,B=94,116,335,359,0.73 或 F=6,P=0
        var match = msg.match(/B=([\d,.-]+)/);
        if (match) {
            var parts = match[1].split(",");
            if (parts.length >= 5) {
                els.coordData.textContent = "x1=" + parts[0] + " y1=" + parts[1] + " x2=" + parts[2] + " y2=" + parts[3] + " conf=" + parts[4];
            }
        } else {
            els.coordData.textContent = "无检测目标";
        }
    }

    function updateDiagnostics(data) {
        // 更新硬件信息条
        var tb = data.throttled || {};
        var cl = data.clock || {};
        var tm = data.temp || {};
        var gv = data.governor || {};
        var cm = data.camera || {};

        els.diagTemp.textContent = "T: " + (tm.temp_c || "--") + "\u00B0C";
        els.diagFreq.textContent = "F: " + (cl.freq_mhz || "--") + "MHz";
        els.diagGov.textContent = "G: " + (gv.governor || "--");
        els.diagCam.textContent = "C: " + (cm.camera_ok ? "OK" : "NG");

        // 更新告警横幅
        var issues = data.issues || [];
        if (issues.length === 0) {
            els.diagBanner.style.display = "none";
            return;
        }

        var html = "";
        for (var i = 0; i < issues.length; i++) {
            var cls = "diag-alert " + (issues[i].level === "danger" ? "diag-danger" : "diag-warning");
            html += '<span class="' + cls + '">' + issues[i].msg + "</span>";
        }
        els.diagBanner.innerHTML = html;
        els.diagBanner.className = "diag-banner " + (data.has_danger ? "diag-banner-danger" : "diag-banner-warning");
        els.diagBanner.style.display = "flex";
    }

    function updateFooter(text) {
        els.footerInfo.textContent = text;
    }

    // ==================== 日志管理 ====================
    function appendLog(message, type) {
        // 移除占位符
        const placeholder = els.logContainer.querySelector(".log-placeholder");
        if (placeholder) {
            placeholder.remove();
        }

        // 创建日志条目
        const entry = document.createElement("div");
        entry.className = `log-entry type-${type}`;
        entry.textContent = message;

        els.logContainer.appendChild(entry);
        logLines.push(entry);

        // 限制最大行数
        if (logLines.length > MAX_LOG_LINES) {
            const old = logLines.shift();
            if (old && old.parentNode) {
                old.remove();
            }
        }

        // 自动滚动到底部
        if (!logPaused) {
            els.logContainer.scrollTop = els.logContainer.scrollHeight;
        }
    }

    function clearLog() {
        els.logContainer.innerHTML = "";
        logLines = [];
        const placeholder = document.createElement("div");
        placeholder.className = "log-placeholder";
        placeholder.textContent = "日志已清空";
        els.logContainer.appendChild(placeholder);
    }

    function togglePauseLog() {
        logPaused = !logPaused;
        els.btnPauseLog.textContent = logPaused ? "继续" : "暂停";
        els.btnPauseLog.style.background = logPaused ? "var(--accent-yellow)" : "";
        els.btnPauseLog.style.color = logPaused ? "#000" : "";
    }

    // ==================== 命令发送 ====================
    function sendCarCommand(key) {
        if (!socket || !socket.connected) {
            appendLog("[ERR] 未连接到服务器，无法发送命令", "error");
            return;
        }
        socket.emit("car_command", { key: key });
    }

    function sendCameraCommand(key) {
        if (!socket || !socket.connected) {
            appendLog("[ERR] 未连接到服务器，无法发送命令", "error");
            return;
        }
        socket.emit("camera_command", { key: key });
    }

    function sendStop() {
        if (!socket || !socket.connected) return;
        socket.emit("car_command", { key: "x" });
    }

    function startYolo() {
        if (!socket || !socket.connected) {
            appendLog("[ERR] 未连接到服务器", "error");
            return;
        }
        socket.emit("yolo_control", { action: "start" });
        appendLog("[INFO] 正在启动 YOLO 识别...", "info");
    }

    function stopYolo() {
        if (!socket || !socket.connected) {
            appendLog("[ERR] 未连接到服务器", "error");
            return;
        }
        socket.emit("yolo_control", { action: "stop" });
        appendLog("[INFO] 正在停止 YOLO 识别...", "info");
    }

    // ==================== 按钮事件绑定 ====================
    function bindButtons() {
        // 左方向键：小车控制
        document.querySelectorAll('[data-target="car"]').forEach((btn) => {
            const key = btn.dataset.key;
            if (!key) return;

            btn.addEventListener("mousedown", (e) => {
                e.preventDefault();
                btn.classList.add("active");
                sendCarCommand(key);
            });
            btn.addEventListener("mouseup", () => {
                btn.classList.remove("active");
                sendStop();
            });
            btn.addEventListener("mouseleave", () => {
                btn.classList.remove("active");
            });
            btn.addEventListener("touchstart", (e) => {
                e.preventDefault();
                btn.classList.add("active");
                sendCarCommand(key);
            }, { passive: false });
            btn.addEventListener("touchend", () => {
                btn.classList.remove("active");
                sendStop();
            });
        });

        // 右方向键：摄像头控制
        document.querySelectorAll('[data-target="camera"]').forEach((btn) => {
            const key = btn.dataset.key;
            if (!key) return;

            btn.addEventListener("mousedown", (e) => {
                e.preventDefault();
                btn.classList.add("active");
                sendCameraCommand(key);
            });
            btn.addEventListener("mouseup", () => {
                btn.classList.remove("active");
                sendStop();
            });
            btn.addEventListener("mouseleave", () => {
                btn.classList.remove("active");
            });
            btn.addEventListener("touchstart", (e) => {
                e.preventDefault();
                btn.classList.add("active");
                sendCameraCommand(key);
            }, { passive: false });
            btn.addEventListener("touchend", () => {
                btn.classList.remove("active");
                sendStop();
            });
        });

        // 启动/停止
        els.btnStart.addEventListener("click", startYolo);
        els.btnStop.addEventListener("click", stopYolo);

        // 日志控制
        els.btnClearLog.addEventListener("click", clearLog);
        els.btnPauseLog.addEventListener("click", togglePauseLog);

        // 硬件自检
        els.btnChecks.addEventListener("click", function() {
            if (socket && socket.connected) {
                socket.emit("diagnostics");
                appendLog("[INFO] 正在执行硬件自检...", "info");
            }
        });
    }

    // ==================== 键盘快捷键 ====================
    function bindKeyboard() {
        document.addEventListener("keydown", (e) => {
            // 忽略输入框中的按键
            if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") {
                return;
            }

            // 防止重复触发
            if (activeKeys.has(e.key)) return;
            activeKeys.add(e.key);

            // 1键：切换植物信息显示
            if (e.key === "1") {
                e.preventDefault();
                infoToggle = !infoToggle;
                els.infoPlaceholder.innerHTML = infoToggle ? INFO_PLANTS : INFO_DEFAULT;
                return;
            }

            // 空格：启动/停止 YOLO
            if (e.key === " " || e.key === "Spacebar") {
                e.preventDefault();
                if (yoloRunning) {
                    stopYolo();
                } else {
                    startYolo();
                }
                return;
            }

            // 小车控制：WASD + Q/E 旋转
            const carMap = {
                "q": "q", "Q": "q",
                "w": "w", "W": "w",
                "e": "e", "E": "e",
                "a": "a", "A": "a",
                "s": "s", "S": "s",
                "d": "d", "D": "d",
            };
            if (carMap[e.key]) {
                e.preventDefault();
                sendCarCommand(carMap[e.key]);
                highlightBtn("car", carMap[e.key], true);
                return;
            }

            // 摄像头控制：方向键 / 小键盘
            const cameraMap = {
                "ArrowUp": "8",
                "ArrowDown": "2",
                "ArrowLeft": "4",
                "ArrowRight": "6",
                "8": "8", "2": "2", "4": "4", "6": "6",
            };
            if (cameraMap[e.key]) {
                e.preventDefault();
                sendCameraCommand(cameraMap[e.key]);
                highlightBtn("camera", cameraMap[e.key], true);
                return;
            }
        });

        document.addEventListener("keyup", (e) => {
            activeKeys.delete(e.key);

            const carMap = {
                "q": "q", "Q": "q",
                "w": "w", "W": "w",
                "e": "e", "E": "e",
                "a": "a", "A": "a",
                "s": "s", "S": "s",
                "d": "d", "D": "d",
            };
            if (carMap[e.key]) {
                highlightBtn("car", carMap[e.key], false);
                sendStop();
            }

            const cameraMap = {
                "ArrowUp": "8", "ArrowDown": "2", "ArrowLeft": "4", "ArrowRight": "6",
                "8": "8", "2": "2", "4": "4", "6": "6",
            };
            if (cameraMap[e.key]) {
                highlightBtn("camera", cameraMap[e.key], false);
                sendStop();
            }
        });
    }

    function highlightBtn(target, key, active) {
        const btn = document.querySelector(`[data-target="${target}"][data-key="${key}"]`);
        if (btn) {
            if (active) {
                btn.classList.add("active");
            } else {
                btn.classList.remove("active");
            }
        }
    }

    // ==================== 初始化 ====================
    function init() {
        initSocket();
        bindButtons();
        bindKeyboard();

        // 初始状态
        updateConnStatus(false);
        updateStatus({ running: false });
        updateFooter("SocketIO 初始化中...");
    }

    document.addEventListener("keydown", function(e) {
        if (e.key === "2") {
            window.open("http://192.168.71.74/");
        }
    });

    // DOM 加载完成后启动
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
