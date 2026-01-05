import psutil
import numpy as np
import cv2
import subprocess
from utils.common import is_intersect
import threading
from detectors.acl_runner import ACLRunner
from detectors.person import _decode_yolo_outputs
import time
from multiprocessing import Process
import queue
import os
import traceback
from datetime import datetime


# 存储当前运行的摄像头线程 & process
active_streams = {}
stream_stop_flags = {}

MODEL_PATH = "/jq/yolo/app/models/yolov8n.om"
acl_runner = ACLRunner(MODEL_PATH)
acl_runner.init()
acl_runner.load_model()


def log_reader(stream, prefix):
    for line in iter(stream.readline, b''):
        # print(f"{prefix} {line.decode(errors='ignore').strip()}", flush=True)
        print("112235456565")

# opencv 拉流
def stream_worker(camera_id, rtsp_url, fence_area):
    global active_streams

    print(f"[DEBUG] stream_worker 启动 camera_id={camera_id}", flush=True)
    print(f"[DEBUG] 输入流: {rtsp_url}", flush=True)
    print(f"[DEBUG] 推流地址: rtsp://10.168.60.52:8553/Streaming/Channels/{camera_id}", flush=True)

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = \
        "rtsp_transport;tcp|stimeout;5000000|err_detect;ignore_err|max_delay;500000"

    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print(f"[ERROR] 无法打开摄像头 {camera_id}", flush=True)
        active_streams.pop(camera_id, None)
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    # print(f"[DEBUG] 摄像头已打开: {width}x{height}@{fps}fps", flush=True)





    if not cap.isOpened():
        print(f"[FATAL] 摄像头 {camera_id} 打开失败", flush=True)
        return
    print(f"[OK] 摄像头 {camera_id} 已打开", flush=True)

    # FFmpeg 推流命令
    rtsp_push = f"rtsp://10.168.60.52:8553/Streaming/Channels/{camera_id}"
    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-fflags", "nobuffer",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",

        "-an",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-pix_fmt", "yuv420p",
        "-f", "rtsp",
        "-rtsp_transport", "tcp",
        rtsp_push
    ]
    print("[DEBUG] 启动 FFmpeg:", " ".join(cmd), flush=True)
    push_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    send_count = 0
    last_log = time.time()

    print("[OK] 开始推流循环", flush=True)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"[WARN] 摄像头 {camera_id} 读取失败，跳过", flush=True)
            time.sleep(0.1)
            continue
        fence_area = np.array(fence_area, dtype=np.int32)
        cv2.polylines(frame, [fence_area], isClosed=True, color=(0, 0, 255), thickness=2)

        # 推流到 FFmpeg
        try:
            push_proc.stdin.write(frame.tobytes())
            push_proc.stdin.flush()
            send_count += 1
        except BrokenPipeError:
            print(f"[FATAL] FFmpeg stdin 断开（BrokenPipe） camera_id={camera_id}", flush=True)
            break
        except Exception as e:
            print(f"[FATAL] 写入 FFmpeg stdin 异常: {e}", flush=True)
            break

        # 每 2 秒打印推送帧数
        if time.time() - last_log > 2:
            print(f"[DEBUG][{camera_id}] 已推送帧: {send_count}", flush=True)
            last_log = time.time()

        time.sleep(1 / fps)

    cap.release()
    push_proc.kill()
    print(f"[EXIT] stream_worker_fence 退出 camera_id={camera_id}", flush=True)

# ffmpeg拉流 确保持续写入
def stream_worker3(camera_id, rtsp_url, fence_area, detect_interval=25, max_black_count=10):
    global active_streams

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = \
        "rtsp_transport;tcp|stimeout;5000000|err_detect;ignore_err|max_delay;500000"

    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开摄像头 {camera_id}", flush=True)
        active_streams.pop(camera_id, None)
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fps = int(cap.get(cv2.CAP_PROP_FPS) or 25)
    if fps <= 0 or fps > 60:
        fps = 25
    print(f"[DEBUG] 摄像头已打开: {width}x{height}@{fps}fps", flush=True)

    output_url = f"rtsp://10.168.60.52:8553/Streaming/Channels/{camera_id}"
    frame_queue = queue.Queue(maxsize=120)
    black_frame = np.zeros((height, width, 3), dtype=np.uint8)

    # 启动 FFmpeg 推流
    def start_ffmpeg():
        cmd = [
            "ffmpeg",
            "-loglevel", "error",
            "-fflags", "nobuffer",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "pipe:0",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-pix_fmt", "yuv420p",
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            output_url
        ]
        print(f"[DEBUG] 启动 FFmpeg: {cmd}", flush=True)
        return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    push_proc = start_ffmpeg()

    # FFmpeg stderr 读取线程
    def ffmpeg_log_reader(proc):
        try:
            while True:
                line = proc.stderr.readline()
                if not line:
                    break
                print(f"[FFMPEG-{camera_id}] {line.decode(errors='ignore').strip()}", flush=True)
        except Exception as e:
            print(f"[ERROR] FFMPEG log reader异常: {e}", flush=True)

    threading.Thread(target=ffmpeg_log_reader, args=(push_proc,), daemon=True).start()

    # 推流线程
    def push_worker():
        frame_count = 0
        while not stream_stop_flags.get(camera_id):
            try:
                try:
                    frame = frame_queue.get(timeout=1)
                except queue.Empty:
                    frame = black_frame.copy()

                if push_proc.poll() is not None:
                    print(f"[WARN] FFmpeg 已退出或崩溃，停止推流", flush=True)
                    push_proc = start_ffmpeg()
                    threading.Thread(target=ffmpeg_log_reader, args=(push_proc,), daemon=True).start()
                    continue

                try:
                    push_proc.stdin.write(frame.tobytes())
                    frame_count += 1
                    if frame_count % 50 == 0:
                        print(f"[DEBUG][{camera_id}] 已推送帧数量: {frame_count}, shape={frame.shape}", flush=True)
                except BrokenPipeError:
                    print(f"[FATAL][{camera_id}] FFmpeg BrokenPipe，可能推流断开", flush=True)
                    sys.stdout.flush()
                    break

                time.sleep(1.0 / fps)  # 控制帧率，避免阻塞

            except Exception as e:
                print(f"[ERROR][{camera_id}] 推流异常: {e}\n{traceback.format_exc()}", flush=True)
                break

    threading.Thread(target=push_worker, daemon=True).start()

    frame_counter = 0
    last_results = []

    try:
        while not stream_stop_flags.get(camera_id):
            ret, frame = cap.read()
            frame_counter += 1
            if not ret or frame is None:
                frame = black_frame.copy()
                print(f"[WARN] 读取帧失败，使用黑帧兜底", flush=True)

            # ACL 推理
            if acl_runner and frame_counter % detect_interval == 0:
                try:
                    outs = acl_runner.run(frame, output_dtype=np.float32)
                    last_results = _decode_yolo_outputs(outs, frame.shape)
                    print(f"[DEBUG] ACL 推理完成，检测到 {len(last_results)} 个目标", flush=True)
                except Exception as e:
                    print(f"[WARN] 推理失败: {e}", flush=True)

            # 绘制检测框
            for det in last_results:
                try:
                    x1, y1, x2, y2, conf, cls_id = det.astype(np.float32)
                    if int(cls_id) == 0 and conf > 0.5:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                except:
                    continue

            # 绘制围栏
            if fence_area:
                try:
                    pts = np.array(fence_area, np.int32).reshape(-1, 1, 2)
                    cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
                except:
                    pass

            # 推流队列
            try:
                frame_queue.put(frame, timeout=0.02)
            except queue.Full:
                _ = frame_queue.get_nowait()
                frame_queue.put(frame, timeout=0.02)

    finally:
        cap.release()
        if push_proc:
            push_proc.kill()
        print(f"[DEBUG] 摄像头 {camera_id} 已停止", flush=True)


def stream_worker2(camera_id, rtsp_url, fence_area, detect_interval=25):
    global active_streams

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = \
        "rtsp_transport;tcp|stimeout;5000000|err_detect;ignore_err|max_delay;500000"

    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开摄像头 {camera_id}", flush=True)
        active_streams.pop(camera_id, None)
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    fps = int(cap.get(cv2.CAP_PROP_FPS) or 25)
    if fps <= 0 or fps > 60:
        fps = 25
    print(f"[DEBUG] 摄像头已打开: {width}x{height}@{fps}fps", flush=True)

    output_url = f"rtsp://10.168.60.52:8553/Streaming/Channels/{camera_id}"
    frame_queue = queue.Queue(maxsize=120)
    black_frame = np.zeros((height, width, 3), dtype=np.uint8)

    # 启动 FFmpeg 推流
    def start_ffmpeg():
        cmd = [
            "ffmpeg",
            "-loglevel", "info",
            "-fflags", "nobuffer",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "pipe:0",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-pix_fmt", "yuv420p",
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            output_url
        ]
        print(f"[DEBUG] 启动 FFmpeg: {cmd}", flush=True)
        return subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,   # 捕获标准输出
            stderr=subprocess.PIPE    # 捕获错误输出
        )

    push_proc = start_ffmpeg()

    # FFmpeg stderr 读取线程
    def ffmpeg_log_reader(proc):
        try:
            for line in iter(proc.stderr.readline, b''):
                if not line:
                    break
                print(f"[FFMPEG-{camera_id}] {line.decode(errors='ignore').strip()}", flush=True)
        except Exception as e:
            print(f"[ERROR] FFMPEG log reader异常: {e}", flush=True)

    threading.Thread(target=ffmpeg_log_reader, args=(push_proc,), daemon=True).start()

    # 推流线程
    def push_worker():
        frame_count = 0
        expected_size = width * height * 3
        while not stream_stop_flags.get(camera_id):
            try:
                try:
                    frame = frame_queue.get(timeout=1)
                    source = "camera"
                except queue.Empty:
                    frame = black_frame.copy()
                    source = "black"

                if push_proc.poll() is not None:
                    print(f"[WARN] FFmpeg 已退出或崩溃，停止推流", flush=True)
                    break

                try:
                    data = frame.tobytes()
                    size = len(data)
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    if size != expected_size:
                        print(
                            f"[ERROR][{camera_id}] {timestamp} 写入帧大小异常: {size} vs {expected_size}, 来源={source}",
                            flush=True)
                    else:
                        print(f"[DEBUG][{camera_id}] {timestamp} 写入帧成功: {size} bytes, 来源={source}", flush=True)

                    push_proc.stdin.write(data)
                    frame_count += 1
                    if frame_count % 50 == 0:
                        print(f"[INFO][{camera_id}] 已推送帧数量: {frame_count}", flush=True)
                except BrokenPipeError:
                    print(f"[FATAL][{camera_id}] FFmpeg BrokenPipe，可能推流断开", flush=True)
                    sys.stdout.flush()
                    break

                time.sleep(1.0 / fps)  # 控制帧率，避免阻塞

            except Exception as e:
                print(f"[ERROR][{camera_id}] 推流异常: {e}\n{traceback.format_exc()}", flush=True)
                break

    threading.Thread(target=push_worker, daemon=True).start()

    frame_counter = 0
    last_results = []

    try:
        while not stream_stop_flags.get(camera_id):
            ret, frame = cap.read()
            frame_counter += 1
            if not ret or frame is None:
                frame = black_frame.copy()
                print(f"[WARN] 读取帧失败，使用黑帧兜底", flush=True)

            # ACL 推理
            if acl_runner and frame_counter % detect_interval == 0:
                try:
                    outs = acl_runner.run(frame, output_dtype=np.float32)
                    last_results = _decode_yolo_outputs(outs, frame.shape)
                    print(f"[DEBUG] ACL 推理完成，检测到 {len(last_results)} 个目标", flush=True)
                except Exception as e:
                    print(f"[WARN] 推理失败: {e}", flush=True)

            # 绘制检测框
            for det in last_results:
                try:
                    x1, y1, x2, y2, conf, cls_id = det.astype(np.float32)
                    if int(cls_id) == 0 and conf > 0.5:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                except:
                    continue

            # 绘制围栏
            if fence_area:
                try:
                    pts = np.array(fence_area, np.int32).reshape(-1, 1, 2)
                    cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
                except:
                    pass

            # 推流队列
            try:
                frame_queue.put(frame, timeout=0.02)
            except queue.Full:
                _ = frame_queue.get_nowait()
                frame_queue.put(frame, timeout=0.02)

    finally:
        cap.release()
        if push_proc:
            push_proc.kill()
        print(f"[DEBUG] 摄像头 {camera_id} 已停止", flush=True)


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