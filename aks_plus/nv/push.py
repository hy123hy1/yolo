import psutil
import numpy as np
import cv2
import subprocess
import threading
from detectors.acl_runner import ACLRunner
import time
import os
import traceback
import queue
import json

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = \
    "rtsp_transport;tcp|stimeout;5000000|max_delay;500000"



# 存储当前运行的摄像头线程 & process
active_streams = {}
stream_stop_flags = {}

MODEL_PATH = "/jq/yolo/app/models/yolov8n.om"
# session = InferSession(device_id=0, model_path=MODEL_PATH)
acl_runner = ACLRunner(MODEL_PATH)
acl_runner.init()
acl_runner.load_model()

# =========================
# 全局状态
# =========================
active_streams = {}   # camera_id -> {"threads": [...], "process":..., "stop_flag":...}
detect_results = {}   # camera_id -> {"boxes": [], "last_update": time}
detect_lock = threading.Lock()

# 推理队列，多个摄像头共用
infer_queue = queue.Queue(maxsize=50)

# 跳帧检测参数
SKIP_DETECT_FRAME = 5

class FFmpegCapture:
    def __init__(self, rtsp_url, width, height):
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.process = None
        self.start()

    def start(self):
        cmd = [
            "ffmpeg",
            "-loglevel", "error",

            "-rtsp_transport", "tcp",
            "-fflags", "nobuffer+discardcorrupt",
            "-flags", "low_delay",
            "-max_delay", "500000",

            "-i", self.rtsp_url,

            "-an",
            "-c:v", "rawvideo",
            "-pix_fmt", "bgr24",

            "-f", "rawvideo",
            "pipe:1"
        ]

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=10**8
        )

    def read(self):
        frame_size = self.width * self.height * 3
        raw = self.process.stdout.read(frame_size)

        if len(raw) != frame_size:
            return False, None

        frame = np.frombuffer(raw, dtype=np.uint8).reshape(
            (self.height, self.width, 3)
        ).copy()
        return True, frame

    def release(self):
        if self.process:
            self.process.kill()

def is_bad_frame(frame):
    if frame is None:
        return True

    # 1. 全黑 / 全灰检测
    if np.mean(frame) < 5:
        return True

    # 2. 方差过低（马赛克/花屏）
    if np.std(frame) < 3:
        return True

    return False

def get_stream_size(rtsp_url):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        rtsp_url
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)

    w = data["streams"][0]["width"]
    h = data["streams"][0]["height"]

    return w, h

# =========================
# YOLO 输出解析
# =========================
def parse_yolo_outputs(outs, img_shape):
    """
    解析 YOLOv8 OM 输出为 [x1,y1,x2,y2,conf,cls_id] 格式
    """
    if not outs or len(outs) == 0:
        return np.zeros((0,6), dtype=np.float32)

    arr = outs[0]
    if arr.ndim == 3:  # (1,84,8400)
        arr = arr.reshape(arr.shape[1], -1).T  # (8400,84)
    dets = []
    h, w = img_shape[:2]
    for i in range(arr.shape[0]):
        cls_scores = arr[i, 4:]
        cls_id = int(np.argmax(cls_scores))
        conf = float(cls_scores[cls_id])
        if conf < CONFIDENCE:
            continue
        cx, cy, bw, bh = arr[i, :4]
        x1 = int(max(0, cx - bw/2))
        y1 = int(max(0, cy - bh/2))
        x2 = int(min(w-1, cx + bw/2))
        y2 = int(min(h-1, cy + bh/2))
        dets.append([x1, y1, x2, y2, conf, cls_id])
    return np.array(dets)

# =========================
# 推理线程（全局共享）
# =========================
def infer_worker():
    while True:
        try:
            camera_id, frame, fence_area = infer_queue.get(timeout=1)
        except queue.Empty:
            continue

        try:
            # ROI 裁剪
            pts = np.array(fence_area, dtype=np.int32)
            x, y, w, h = cv2.boundingRect(pts)
            crop = frame[y:y+h, x:x+w]
            if crop.size == 0:
                continue

            # ======================
            # YOLOv8 OM 推理前处理
            # ======================
            ori_h, ori_w = crop.shape[:2]
            input_size = 640
            scale = min(input_size / ori_w, input_size / ori_h)
            new_w, new_h = int(ori_w * scale), int(ori_h * scale)
            resized = cv2.resize(crop, (new_w, new_h))

            # letterbox padding
            padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
            pad_x = (input_size - new_w) // 2
            pad_y = (input_size - new_h) // 2
            padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized

            # blob
            blob = cv2.dnn.blobFromImage(padded, scalefactor=1/255, size=(640,640), swapRB=True)

            # 推理
            # outs = session.infer(feeds=[blob], mode="static")
            outs = acl_runner.session.infer(feeds=[blob], mode="static")

            # 解析输出
            boxes = parse_yolo_outputs(outs, crop.shape)

            # 坐标还原到原图
            scale_x, scale_y = w / 640, h / 640
            final_boxes = []
            for b in boxes:
                x1, y1, x2, y2, conf, cls = b
                final_boxes.append([
                    int(x + x1 * scale_x),
                    int(y + y1 * scale_y),
                    int(x + x2 * scale_x),
                    int(y + y2 * scale_y),
                    float(conf),
                    int(cls)
                ])

            print(f"[DEBUG] camera_id={camera_id}, boxes={final_boxes}")

            # 更新全局检测结果
            with detect_lock:
                detect_results[camera_id] = {"boxes": final_boxes, "last_update": time.time()}

        except Exception as e:
            print(f"[ERROR] infer_worker camera_id={camera_id} 异常: {e}")
            traceback.print_exc()
            continue


# 启动全局推理线程
# t_infer = threading.Thread(target=infer_worker, daemon=True)
# t_infer.start()


# ffmpeg拉流
def stream_worker3(camera_id, rtsp_url, fence_area):
    global active_streams

    if camera_id in active_streams:
        print(f"[WARN] {camera_id} 已运行")
        return

    stop_flag = threading.Event()
    frame_queue = queue.Queue(maxsize=50)

    print(f"[INFO] 启动摄像头 {camera_id}")

    #  必须固定（可改成配置）
    width, height = get_stream_size(rtsp_url)
    cap = FFmpegCapture(rtsp_url, width, height)
    fps = 25

    rtsp_push = f"rtsp://10.168.60.52:8553/Streaming/Channels/{camera_id}"

    # =========================
    # FFmpeg 推流
    # =========================
    def start_ffmpeg():
        cmd = [
            "ffmpeg",
            "-loglevel", "error",

            "-fflags", "+genpts+discardcorrupt",

            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "pipe:0",

            "-an",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",

            "-g", str(fps),
            "-keyint_min", str(fps),
            "-sc_threshold", "0",

            "-pix_fmt", "yuv420p",

            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            rtsp_push
        ]

        print(f"[DEBUG] 启动推流 FFmpeg {camera_id}")
        return subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0
        )

    push_proc = start_ffmpeg()

    # =========================
    # 拉流线程（FFmpeg版）
    # =========================
    def read_loop():
        nonlocal cap

        bad_count = 0
        reconnect_count = 0

        while not stop_flag.is_set():
            ret, frame = cap.read()

            # if ret:
            #     frame = frame.copy()

            if not ret or frame is None:
                bad_count += 1

                if bad_count > 10:
                    print(f"[WARN] {camera_id} 拉流失败，重启FFmpeg拉流")

                    cap.release()
                    time.sleep(min(2 ** reconnect_count, 10))
                    reconnect_count += 1

                    cap = FFmpegCapture(rtsp_url, width, height)
                    bad_count = 0

                continue

            bad_count = 0
            reconnect_count = 0

            # ===== 围栏
            pts = np.array(fence_area, np.int32)
            cv2.polylines(frame, [pts], True, (0, 0, 255), 2)

            # ===== 推理
            # if infer_queue.full():
            #     infer_queue.get_nowait()
            # infer_queue.put((camera_id, frame.copy(), fence_area))
            #
            # # ===== 检测结果
            # with detect_lock:
            #     boxes = detect_results.get(camera_id, {}).get("boxes", [])
            #
            # for x1, y1, x2, y2, conf, cls in boxes:
            #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            #     cv2.putText(frame, f"{cls}:{conf:.2f}",
            #                 (x1, y1 - 5),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #                 (0, 255, 255), 1)

            # ===== 推流队列（丢帧）
            if frame_queue.full():
                frame_queue.get_nowait()
            frame_queue.put_nowait(frame)

    # =========================
    # 推流线程
    # =========================
    def push_loop():
        nonlocal push_proc

        while not stop_flag.is_set():
            try:
                frame = frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            if push_proc.poll() is not None:
                print(f"[WARN] 推流FFmpeg挂了，重启 {camera_id}")
                push_proc = start_ffmpeg()
                continue

            try:
                push_proc.stdin.write(frame.tobytes())
            except BrokenPipeError:
                print(f"[ERROR] BrokenPipe {camera_id}")
                push_proc = start_ffmpeg()
            except Exception as e:
                print(f"[ERROR] 推流异常 {camera_id}: {e}")
                push_proc = start_ffmpeg()

    # =========================
    # 启动线程
    # =========================
    t1 = threading.Thread(target=read_loop, daemon=True)
    t2 = threading.Thread(target=push_loop, daemon=True)

    t1.start()
    t2.start()

    active_streams[camera_id] = {
        "threads": [t1, t2],
        "process": push_proc,
        "stop_flag": stop_flag
    }

    print(f"[INFO] {camera_id} 启动完成")

    try:
        while not stop_flag.is_set():
            time.sleep(5)
    finally:
        stop_flag.set()
        cap.release()
        try:
            push_proc.kill()
        except:
            pass
        active_streams.pop(camera_id, None)

def stop_stream(camera_id, fence_dict):
    """
    简化版流停止函数：安全关闭线程与推流进程。
    """
    if camera_id not in active_streams:
        print(f"[INFO] 摄像头 {camera_id} 未在运行")
        return

    info = active_streams[camera_id]
    print(f"[INFO] 正在停止摄像头 {camera_id}...")

    # === 停止 ffmpeg 推流 ===
    proc = info.get("process")
    if proc:
        try:
            pid = proc.pid
            proc.kill()
            print(f"[INFO] 推流进程 (PID={pid}) 已终止")
        except Exception as e:
            print(f"[WARN] 终止 ffmpeg 进程失败: {e}")

    # === 检查并清理残留 ffmpeg 子进程 ===
    try:
        for p in psutil.process_iter(["pid", "name", "cmdline"]):
            if "ffmpeg" in p.info["name"] and str(camera_id) in " ".join(p.info["cmdline"]):
                print(f"[INFO] 检测到残留进程 PID={p.info['pid']}，强制结束")
                p.kill()
    except Exception as e:
        print(f"[WARN] 清理残留进程出错: {e}")

    # === 清除缓存 ===
    active_streams.pop(camera_id, None)
    fence_dict.pop(camera_id, None)
    print(f"[INFO] 摄像头 {camera_id} 已完全停止")

