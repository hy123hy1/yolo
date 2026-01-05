import os
import sys
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000|err_detect;ignore_err|max_delay;500000"
sys.stdout.reconfigure(line_buffering=True)

import cv2
import numpy as np
import datetime
import json
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from minio import Minio
import io
import subprocess
import tempfile
import queue
import time
from collections import deque
import os
from collections import defaultdict
import psutil
import re
from ais_bench.infer.interface import InferSession
from collections import defaultdict
import threading

# ===== 新增：闯入事件状态机 =====
intrusion_state = defaultdict(lambda: {
    "active": False,          # 是否在闯入事件中
    "start_time": None,       # 事件开始时间
    "last_upload_time": None, # 上次上传图片时间
    "image_url": None         # 本次事件使用的图片
})

# 连续帧计数（防抖）
intrusion_frame_counter = defaultdict(int)
# # 连续“无人的帧”计数（用于事件结束）
intrusion_disappear_counter = defaultdict(int)
INTRUSION_END_FRAMES = 25   # 连续 10 帧没人，判定事件结束


# ===== 安全帽事件状态 =====
helmet_state = defaultdict(lambda: {
    "active": False,
    "start_time": None,
    "image_url": None
})

# 连续帧计数（防抖）
helmet_frame_counter = defaultdict(int)
helmet_disappear_counter = defaultdict(int)

MIN_HELMET_FRAMES = 25      # 连续 5 帧未戴才算违规
HELMET_END_FRAMES = 25    # 连续 10 帧合规，事件结束


# ===== 人群聚集事件状态 =====
# 人员聚集状态
overcrowd_counter = defaultdict(int)
overcrowd_active = defaultdict(bool)

"""
常量区
"""
# 电子围栏区域（动态更新）
fence_dict = {}
algorithm_dict = {}

fence_lock = threading.Lock()
lock = threading.Lock()
algorithm_lock = threading.Lock()
active_streams = {}

# 全局状态变量
MAX_PEOPLE = 15       # 最大允许人数
TIME_THRESHOLD = 30   # 超员持续秒数

MAX_READ_FAIL = 30        # 连续读取失败多少次认为断流
RECONNECT_INTERVAL = 5   # 重连间隔（秒）
MAX_RECONNECT_TIMES = 0  # 0 表示无限重连，>0 表示限制次数

# 全局缓存：记录每个摄像头最近一次报警时间
last_alarm_time = {}
# 记录窗口内报警次数
alarm_counter = {}
# 记录摄像头最近一次视频时间
last_video_time = {}

gathering_state = {
    "count": 0,
    "over_time": 0
}

# 告警缓存机制  每个算法类型对应1个队列
alarm_queue = queue.Queue()
alarm_queue_video = queue.Queue()
safehat_queue = queue.Queue()
safehat_queue_video = queue.Queue()
personcrowd_queue = queue.Queue()
personcrowd_queue_video = queue.Queue()

FPS = 25
PRE_SECONDS = 3
SKIP_INTERVAL = 25  # 跳帧间隔

# 全局缓存（按摄像头）
latest_pre_frames = defaultdict(list)

# 记录每个摄像头的上次播报时间
last_alert_time = {}

# 初始化模型（进程级）
MODEL_PATH = "yolov8n.om"
session = InferSession(device_id=0, model_path=MODEL_PATH)
session2 = InferSession(device_id=0, model_path=MODEL_PATH)

MODEL_PATH1 = "safehat.om"
session1 = InferSession(device_id=0, model_path=MODEL_PATH1)



CONFIDENCE = 0.4
IOU = 0.45
CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',
    8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
    29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
    35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple',
    48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
    55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet',
    62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
    69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase',
    76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}
CLASSES2 = {0: 'helmet', 1: 'head'}

colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

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


# 初始化minio客户端
minio_client = Minio(
    "172.21.3.141:8084",
    access_key="root",
    secret_key="12345678",
    secure=False
)

"""
函数功能区
"""
# 获取摄像头列表
def get_cameras():
    url = "http://172.21.3.141:8080/aks-mkaqjcyj/cameraMonitoring/selectSxtInfo"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers)

    if resp.status_code != 200:
        print("请求失败:", resp.text[:200])
        return []

    data = resp.json()
    cameras_data = data.get("data", [])
    print(f"发现 {len(cameras_data)} 个摄像头配置。")

    # originalRtsp
    # 过滤掉 algorithmtypes 为空的摄像头
    valid_cameras = []
    for item in cameras_data:
        algorithmtypes = item.get("algorithmtypes")
        if algorithmtypes:  # 非空才加入
            valid_cameras.append(
                (item.get("id"), item.get("originalRtsp"), item.get("ipAddress"), algorithmtypes)
            )
        else:
            print(f"[跳过] 摄像头 {item.get('id')} 的 algorithmtypes 为空。")

    return valid_cameras

def load_config(path="./cfg/config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# 上传图片
def upload_warning_image(image, camera_id, label):
    success, buffer = cv2.imencode('.jpg', image)
    if not success:
        raise ValueError("图像编码失败")

    image_bytes = io.BytesIO(buffer.tobytes())
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    object_name = f"{camera_id}/{label}/warning_{timestamp}.jpg"

    minio_client.put_object(
        bucket_name="yolo",
        object_name=object_name,
        data=image_bytes,
        length=len(image_bytes.getvalue()),
        content_type="image/jpeg"
    )
    print(f"[INFO] 报警图片上传成功: {object_name}")

    url = f"http://172.21.3.141:8084/yolo/{object_name}"
    # print(f"[INFO] 报警图片 URL: {url}")
    return url

# 上传视频
def upload_warning_video(video_bytes, camera_id):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    object_name = f"{camera_id}/{timestamp}.mp4"

    video_stream = io.BytesIO(video_bytes)

    minio_client.put_object(
        bucket_name="yolo",
        object_name=object_name,
        data=video_stream,
        length=len(video_bytes),
        content_type="video/mp4"
    )

    url = f"http://172.21.3.141:8084/yolo/{object_name}"
    return url

# 语音播报
def alert_intrusion(camera_id):
    current_time = time.time()
    last_time = last_alert_time.get(camera_id, 0)

    # 如果距离上次播报超过 60 秒，则播放语音
    if current_time - last_time >= 60:
        pygame.mixer.init()
        pygame.mixer.music.load("E:\project\yw\ef\cfg\warning.mp3")
        pygame.mixer.music.play()
        last_alert_time[camera_id] = current_time
    else:
        print(f"摄像头 {camera_id} 在冷却期内，跳过播报")

# 生成报警视频
def generate_warning_video_memory(frame_buffer, video_frames, FPS, camera_id):
    try:
        if not frame_buffer and not video_frames:
            print("[WARN] 无视频帧可写入")
            return None

        sample_frame = frame_buffer[0] if frame_buffer else video_frames[0]
        h, w = sample_frame.shape[:2]

        # === 安全创建临时文件 ===
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        # === 写入视频 ===
        # 格式编码
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_path, fourcc, FPS, (w, h))
        if not writer.isOpened():
            print("[ERROR] 无法创建视频写入对象")
            return None

        for f in list(frame_buffer) + list(video_frames):
            writer.write(f)
        writer.release()

        # === 读取视频字节流 ===
        with open(tmp_path, "rb") as f:
            video_bytes = f.read()

        # === 上传到 MinIO ===
        video_url = upload_warning_video(video_bytes, camera_id)
        print("[INFO] 报警视频上传成功:", video_url)

        # === 上传后安全删除 ===
        os.remove(tmp_path)

        return video_url

    except Exception as e:
        print(f"[ERROR] 视频生成失败: {e}")
        return None

# 异步生成预警视频
def _async_generate_video_and_notify(pre_frames, rtsp_url, FPS, camera_id, alarm_queue_video, image_url):
    try:
        post_frames = record_post_frames(rtsp_url, seconds=3)
        video_url = generate_warning_video_memory(pre_frames, post_frames, FPS, camera_id)
        print("生成视频链接为：", video_url)

        update = {
            "timestamp": datetime.datetime.now().isoformat(),
            "image_url": str(image_url),
            "video_url": str(video_url),
            "algorithmtypes": str(1)
        }
        # batch = alarm_queue_video.put(update)
        send_alarm_to_java(update, config["server_endpoints4"])
        # print("发送地址为:", config["server_endpoints4"])
    except Exception as e:
        print(f"[ERROR] 异步生成预警视频失败: {e}", flush=True)

def _async_generate_video_and_notify2(pre_frames, rtsp_url, FPS, camera_id, safehat_queue, image_url):
    try:
        post_frames = record_post_frames(rtsp_url, seconds=3)
        video_url = generate_warning_video_memory(pre_frames, post_frames, FPS, camera_id)
        # 生成完成后把一个更新消息放入队列，供 alarm_dispatcher 或后续处理发送或更新
        update = {
            "timestamp": datetime.datetime.now().isoformat(),
            "image_url": str(image_url),
            "video_url": str(video_url),
            "algorithmtypes": str(2)
        }
        # safehat_queue.put(update)
        send_alarm_to_java(update, config["server_endpoints4"])
        print(f"[INFO] 异步预警视频生成完成: {video_url}", flush=True)
    except Exception as e:
        print(f"[ERROR] 异步生成预警视频失败: {e}", flush=True)


# 供人员聚集使用
def _async_generate_video_and_notify3(pre_frames, rtsp_url, FPS, camera_id, personcrowd_queue_video, image_url, config):
    try:
        post_frames = record_post_frames(rtsp_url, seconds=3)
        video_url = generate_warning_video_memory(pre_frames, post_frames, FPS, camera_id)
        batch = []
        # 生成完成后把一个更新消息放入队列，供 alarm_dispatcher 后续处理发送或更新
        update = {
            "timestamp": datetime.datetime.now().isoformat(),
            "image_url": str(image_url),
            "video_url": str(video_url),
            "algorithmtypes": str(3)
        }
        batch = personcrowd_queue_video.put(update)
        send_alarm_to_java(batch, config["server_endpoints3"])
    except Exception as e:
        print(f"[ERROR] 异步生成预警视频失败: {e}", flush=True)


# 判断两个矩形是否相交
def is_intersect(box_rect, fence_poly, frame_shape):
    box_poly = np.array(box_rect, dtype=np.int32)
    fence_poly = np.array(fence_poly, dtype=np.int32)
    mask1 = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.uint8)
    mask2 = np.zeros_like(mask1)
    cv2.fillPoly(mask1, [box_poly], 255)
    cv2.fillPoly(mask2, [fence_poly], 255)
    intersection = cv2.bitwise_and(mask1, mask2)
    return np.any(intersection > 0)

# 发送报警信息到Java端
def send_alarm_to_java(alarm_data, endpoints):
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "curl/7.79.1",
        "Accept": "*/*"
    }
    for url in endpoints:
        try:
            response = requests.post(url, json=alarm_data, headers=headers)
            print("发送内容:", alarm_data)
            # print("返回内容:", response.text)
            print(f"告警发送至 {url} 状态码: {response.status_code}")
        except Exception as e:
            print(f"告警发送失败: {e}")

def alarm_dispatcher(config, alarm_queue):
    try:
        batch = alarm_queue.get(timeout=1)
    except queue.Empty:
        return
    print(f"[INFO] 发送 {len(batch)} 条报警信息到Java端", flush=True)
    send_alarm_to_java(batch, config["server_endpoints"])

def safehat_dispatcher(config, safehat_queue):
    try:
        batch = safehat_queue.get(timeout=1)
    except queue.Empty:
        return
    print(f"[INFO] 发送 {len(batch)} 条报警信息到Java端", flush=True)
    send_alarm_to_java(batch, config["server_endpoints2"])

def person_count_dispatcher(config, personcrowd_queue):
    while True:
        time.sleep(10)
        batch = []
        while not personcrowd_queue.empty():
            batch.append(personcrowd_queue.get())

        if batch:
            print(f"[INFO] 发送 {len(batch)} 条报警信息到Java端")
            send_alarm_to_java(batch, config["server_endpoints3"])

# 从 RTSP 读取指定秒数的帧（用于安全帽视频存证）
def record_post_frames(rtsp_url, seconds=3):
    """从 RTSP 读取指定秒数的帧 """
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    frames = []
    fps = 25
    total_frames = fps * seconds

    while len(frames) < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame.copy())

    cap.release()
    return frames

# 解码 YOLO 输出
def _decode_yolo_outputs(outs, img_shape):
    """
    安全解析 YOLO 输出的占位函数。
    - outs: list[np.ndarray] from acl_runner.run()
    - 返回: list[[x1,y1,x2,y2,conf,cls_id]]
    注意：不同 OM 导出格式差异很大，这里仅示范。
    """
    if not outs:
        return np.zeros((0, 6), dtype=np.float32)

    # 典型 YOLOv8 export 场景之一：单输出且为 [num, 6] 或一维向量长度可整除 6
    arr = outs[0]
    if arr.ndim == 1 and arr.size % 6 == 0:
        dets = arr.reshape(-1, 6)
    elif arr.ndim == 2 and arr.shape[1] == 6:
        dets = arr
    else:
        # 如果是特征图格式，需根据你的 OM 解码逻辑实现，这里先返回空
        dets = np.zeros((0, 6), dtype=np.float32)

    # 可选：裁剪到图像边界
    h, w = img_shape[:2]
    dets[:, 0] = np.clip(dets[:, 0], 0, w - 1)
    dets[:, 1] = np.clip(dets[:, 1], 0, h - 1)
    dets[:, 2] = np.clip(dets[:, 2], 0, w - 1)
    dets[:, 3] = np.clip(dets[:, 3], 0, h - 1)
    return dets

# =====================
#  检测逻辑
# =====================
# 人员闯入电子围栏
def detect_intrusion(frame, camera_id, rtsp_url, config, frame_buffer, FPS, ip_address):

    # print("开始执行人员闯入检测--------------", flush=True)
    confidence_threshold = config.get("confidence_threshold", 0.5)

    # YOLO letterbox 预处理 ====
    ori_h, ori_w = frame.shape[:2]
    img = frame.copy()
    # 目标输入大小
    input_size = 640

    # 等比例缩放
    scale = min(input_size / ori_w, input_size / ori_h)
    new_w = int(ori_w * scale)
    new_h = int(ori_h * scale)
    resized = cv2.resize(img, (new_w, new_h))

    # Letterbox padding
    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    pad_x = (input_size - new_w) // 2
    pad_y = (input_size - new_h) // 2
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    # blob 预处理
    blob = cv2.dnn.blobFromImage(padded, scalefactor=1.0 / 255, size=(640, 640), swapRB=True)

    # 推理
    outs = session.infer(feeds=[blob], mode="static")

    # 解析为 [x1,y1,x2,y2,conf,cls_id]
    detections = parse_yolo_outputs(outs, frame.shape)

    detected_objects = []

    # ==== 2) 过滤并把坐标从 padded 映射回原图 ====
    # 如果检测到人，则记录
    valid_dets = []
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det  # parse_yolo_outputs 返回 float 型
        if cls_id != 0 or conf < confidence_threshold:
        # if conf < confidence_threshold:
            continue

        # 去 padding 并反缩放到原图坐标
        rx1 = int(max(0, min(ori_w - 1, round((x1 - pad_x) / scale))))
        ry1 = int(max(0, min(ori_h - 1, round((y1 - pad_y) / scale))))
        rx2 = int(max(0, min(ori_w - 1, round((x2 - pad_x) / scale))))
        ry2 = int(max(0, min(ori_h - 1, round((y2 - pad_y) / scale))))

        valid_dets.append((rx1, ry1, rx2, ry2, float(conf), 0, "person"))

        cls_id = int(cls_id)
        conf = float(conf)

        label = CLASSES.get(cls_id, "unknown")
        valid_dets.append((rx1, ry1, rx2, ry2, conf, cls_id, label))

        # 连续帧确认
        MIN_FRAMES = config.get("min_intrusion_frames", 25)

        if valid_dets:
            intrusion_frame_counter[camera_id] += 1
        else:
            intrusion_frame_counter[camera_id] = 0

        if intrusion_frame_counter[camera_id] < MIN_FRAMES:
            return False, []
        # ================== 事件状态机 ==================
        now = datetime.datetime.now()
        state = intrusion_state[camera_id]

        image_url = None

        frame_draw = frame.copy()
        # 绘制检测框 frame.copy()的方式，防止原图被修改
        for (rx1, ry1, rx2, ry2, conf, cls_id, label) in valid_dets:
            cv2.rectangle(frame_draw, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
            cv2.putText(
                frame_draw,
                f"{label} {conf:.2f}",
                (rx1, max(ry1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )

        # ===== 事件开始（只触发一次）=====
        if not state["active"]:
            state["active"] = True
            state["start_time"] = now
            state["last_upload_time"] = now

            try:
                image_url = upload_warning_image(frame_draw, camera_id, "person")
                state["image_url"] = image_url
            except Exception as e:
                print(f"[WARN] 上传报警图片失败: {e}", flush=True)

            # ===== 只在事件开始时生成视频 =====
            try:
                pre_frames = list(frame_buffer)
                threading.Thread(
                    target=_async_generate_video_and_notify,
                    args=(pre_frames, rtsp_url, FPS, camera_id, alarm_queue_video, image_url),
                    daemon=True
                ).start()
            except Exception as e:
                print(f"[WARN] 启动视频线程失败: {e}", flush=True)

        else:
            # 事件进行中，不重复上传
            image_url = state.get("image_url")

        # ================== 事件级报警（只入队一次） ==================
        detected_objects = [{
            "label": "person",
            "confidence": round(conf, 2),
            "bbox": [rx1, ry1, rx2, ry2],
            "status": "warning",
            "warning_image": image_url,
            "warning_video": ""
        } for (rx1, ry1, rx2, ry2, conf, _, _) in valid_dets]

        if state["start_time"] == now:  # 只在事件开始那一刻入队
            alarm_queue.put({
                "timestamp": now.isoformat(),
                "camera_id": str(camera_id),
                "count": len(valid_dets),
                "objects": detected_objects,
                "sxt_ip": str(ip_address)
            })
            threading.Thread(
                target=alarm_dispatcher,
                args=(config, alarm_queue),
                daemon=True
            ).start()

    return True, detected_objects


# 安全帽检测
def check_helmet_and_alert(frame, camera_id, rtsp_url, config, frame_buffer, FPS, ip_address):
    """
    安全帽检测：检测未佩戴安全帽的人员，并触发报警（含图片 + 视频存证）
    """
    confidence_threshold = config.get("confidence_threshold", 0.5)

    # YOLO letterbox 预处理 ====
    ori_h, ori_w = frame.shape[:2]
    img = frame.copy()
    # 目标输入大小
    input_size = 640

    # 等比例缩放
    scale = min(input_size / ori_w, input_size / ori_h)
    new_w = int(ori_w * scale)
    new_h = int(ori_h * scale)
    resized = cv2.resize(img, (new_w, new_h))
    # Letterbox padding
    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    pad_x = (input_size - new_w) // 2
    pad_y = (input_size - new_h) // 2
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    # blob 预处理
    blob = cv2.dnn.blobFromImage(padded, scalefactor=1.0 / 255, size=(640, 640), swapRB=True)

    # 推理
    outs = session1.infer(feeds=[blob], mode="static")

    # 解析为 [x1,y1,x2,y2,conf,cls_id]
    detections = parse_yolo_outputs(outs, frame.shape)

    no_helmet_dets = []

    valid_dets = []
    # 遍历检测结果
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det  # parse_yolo_outputs 返回 float 型
        if conf <= confidence_threshold:
            continue

        # 去 padding 并反缩放到原图坐标
        rx1 = int(max(0, min(ori_w - 1, round((x1 - pad_x) / scale))))
        ry1 = int(max(0, min(ori_h - 1, round((y1 - pad_y) / scale))))
        rx2 = int(max(0, min(ori_w - 1, round((x2 - pad_x) / scale))))
        ry2 = int(max(0, min(ori_h - 1, round((y2 - pad_y) / scale))))

        cls_id = int(cls_id)
        conf = float(conf)

        label = CLASSES2.get(cls_id, "unknown")

        # 违规判定
        if label == "head":
            no_helmet_dets.append((rx1, ry1, rx2, ry2, float(conf)))
    # ================== 连续帧防抖 ==================
    if no_helmet_dets:
        helmet_frame_counter[camera_id] += 1
        helmet_disappear_counter[camera_id] = 0
    else:
        helmet_frame_counter[camera_id] = 0
        helmet_disappear_counter[camera_id] += 1

    if helmet_frame_counter[camera_id] < MIN_HELMET_FRAMES:
        return False

    state = helmet_state[camera_id]
    now = datetime.datetime.now()
    image_url = state.get("image_url")

    frame_draw = frame.copy()
    # 绘制检测框 frame.copy()的方式
    for (rx1, ry1, rx2, ry2, conf, cls_id, label) in valid_dets:
        cv2.rectangle(frame_draw, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
        cv2.putText(
            frame_draw,
            f"{label} {conf:.2f}",
            (rx1, max(ry1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

    # ================== 事件开始 ==================
    if not state["active"]:
        state["active"] = True
        state["start_time"] = now

        try:
            image_url = upload_warning_image(frame_draw, camera_id, "no_helmet")
            state["image_url"] = image_url
        except Exception as e:
            print(f"[WARN] 上传安全帽图片失败: {e}", flush=True)

        try:
            pre_frames = list(frame_buffer)
            threading.Thread(
                target=_async_generate_video_and_notify2,
                args=(pre_frames, rtsp_url, FPS, camera_id, safehat_queue_video, image_url),
                daemon=True
            ).start()
        except Exception as e:
            print(f"[WARN] 启动安全帽视频线程失败: {e}", flush=True)

        # ===== 事件级告警（只一次）=====
        objects = [{
            "helmet_status": "no_helmet",
            "confidence": round(conf, 2),
            "bbox": [x1, y1, x2, y2],
            "status": "warning",
            "warning_image": image_url,
            "warning_video": ""
        } for (x1, y1, x2, y2, conf) in no_helmet_dets]

        safehat_queue.put({
            "timestamp": now.isoformat(),
            "camera_id": camera_id,
            "count": len(no_helmet_dets),
            "objects": objects,
            "sxt_ip": ip_address
        })

        threading.Thread(
            target=safehat_dispatcher,
            args=(config, safehat_queue),
            daemon=True
        ).start()

    # ================== 事件结束 ==================
    if helmet_disappear_counter[camera_id] >= HELMET_END_FRAMES:
        helmet_state[camera_id] = {
            "active": False,
            "start_time": None,
            "image_url": None
        }
        helmet_frame_counter[camera_id] = 0
        helmet_disappear_counter[camera_id] = 0

    return True

# 人员聚集检测
def detect_overcrowding(frame, camera_id, rtsp_url, config, frame_buffer, FPS, ip_address):
    """
    人员聚集：检测人员超员，并触发报警（含图片 + 视频存证）
    """
    overcrowd_counter = defaultdict(int)

    confidence_threshold = config.get("confidence_threshold", 0.5)
    max_people = config.get("max_people", 0)
    duration_threshold = config.get("duration_threshold", 1)
    need_frames = max(1, int(duration_threshold * FPS))
    fps = FPS

    # === 1) Letterbox + blob 预处理 ===
    ori_h, ori_w = frame.shape[:2]
    input_size = 640

    scale = min(input_size / ori_w, input_size / ori_h)
    new_w = int(ori_w * scale)
    new_h = int(ori_h * scale)
    resized = cv2.resize(frame, (new_w, new_h))

    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    pad_x = (input_size - new_w) // 2
    pad_y = (input_size - new_h) // 2
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    blob = cv2.dnn.blobFromImage(
        padded,
        scalefactor=1.0 / 255,
        size=(640, 640),
        swapRB=True
    )

    # === 2) 推理 ===
    outs = session2.infer(feeds=[blob], mode="static")
    detections = parse_yolo_outputs(outs, frame.shape)

    # === 3) 框映射回原图 ===
    valid_dets = []
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        if cls_id != 0 or conf < confidence_threshold:
            continue

        label = CLASSES.get(int(cls_id), "unknown")
        if label != "person":
            continue

        # 坐标反映射
        rx1 = int((x1 - pad_x) / scale)
        ry1 = int((y1 - pad_y) / scale)
        rx2 = int((x2 - pad_x) / scale)
        ry2 = int((y2 - pad_y) / scale)
        rx1 = max(0, min(ori_w - 1, rx1))
        ry1 = max(0, min(ori_h - 1, ry1))
        rx2 = max(0, min(ori_w - 1, rx2))
        ry2 = max(0, min(ori_h - 1, ry2))

        valid_dets.append((rx1, ry1, rx2, ry2, float(conf), int(cls_id), "person"))

    # 没检测到合理目标
    if not valid_dets:
        return False, []

    # === 4) 人数统计 ===
    people_count = len(valid_dets)

    # === 5) 绘制结果 ===
    for (rx1, ry1, rx2, ry2, conf, cls_id, label) in valid_dets:
        color = (0, 255, 255)
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}",
                    (rx1, max(ry1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.putText(frame, f"区域人数: {people_count}",
                (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # === 6) 状态计数器（持续 N 帧才报警） ===
    if people_count == 0:
        overcrowd_counter[camera_id] = 0
        overcrowd_active[camera_id] = False
        return False

    # ===== 状态累计 =====
    if people_count > max_people:
        overcrowd_counter[camera_id] += 1
    else:
        overcrowd_counter[camera_id] = max(0, overcrowd_counter[camera_id] - 1)
        overcrowd_active[camera_id] = False
        return False

    print(
        f"[DEBUG] overcrowd cam={camera_id} "
        f"{people_count}/{max_people} "
        f"{overcrowd_counter[camera_id]}/{need_frames}",
        flush=True
    )

    # ===== 未达到持续帧数，不报警 =====
    if overcrowd_counter[camera_id] < need_frames:
        return False

    # ===== 已经处于报警状态，避免重复 =====
    if overcrowd_active[camera_id]:
        return True

    overcrowd_active[camera_id] = True
    overcrowd_counter[camera_id] = 0

    # ===== 事件开始：上传图片 =====
    try:
        image_url = upload_warning_image(frame, camera_id, "overcrowding")
    except Exception as e:
        print(f"[WARN] overcrowding image upload failed: {e}", flush=True)
        image_url = None

    # ===== 只在事件开始时生成视频 =====
    try:
        pre_frames = list(frame_buffer)
        threading.Thread(
            target=_async_generate_video_and_notify3,
            args=(
                pre_frames,
                rtsp_url,
                FPS,
                camera_id,
                personcrowd_queue_video,
                image_url,
                config,
            ),
            daemon=True
        ).start()
    except Exception as e:
        print(f"[WARN] 启动聚集视频线程失败: {e}", flush=True)

    # ===== 入队（只入队，不起 dispatcher）=====
    personcrowd_queue.put({
        "timestamp": datetime.datetime.now().isoformat(),
        "camera_id": str(camera_id),
        "count": people_count,
        "max_people": max_people,
        "warning_image": image_url,
        "warning_video": "",
        "ip": str(ip_address)
    })

    print(f"[WARN] 人员聚集报警触发: cam={camera_id}, count={people_count}", flush=True)

    return True

def start_all_streams(config, fence_lock, algorithm_lock, fence_dict, algorithm_dict, latest_pre_frames, cams):
    for cam_id, rtsp_url, ip_address, algorithmtypes in cams:
        thread = threading.Thread(
            target=process_stream,
            args=(cam_id, rtsp_url, config, fence_lock, algorithm_lock, fence_dict, algorithm_dict, latest_pre_frames, ip_address, algorithmtypes),
            daemon=True
        )
        thread.start()
        # print(f"[INFO] 启动摄像头 {cam_id} 的流")

def process_stream(camera_id, rtsp_url, config, fence_lock, algorithm_lock, fence_dict, algorithm_dict, latest_pre_frames, ip_address, algorithmtypes):

    def open_capture():
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            return None
        return cap

    cap = open_capture()

    if cap is None:
        print(f"[ERROR] 初始无法打开摄像头: {rtsp_url}", flush=True)

    frame_buffer = deque(maxlen=FPS * PRE_SECONDS)
    frame_counter = 0
    read_fail_count = 0
    reconnect_count = 0

    while True:
        # ====== cap 不存在，尝试重连 ======
        if cap is None or not cap.isOpened():
            if MAX_RECONNECT_TIMES > 0 and reconnect_count >= MAX_RECONNECT_TIMES:
                print(f"[ERROR] 摄像头 {camera_id} 超过最大重连次数，终止线程", flush=True)
                break

            reconnect_count += 1
            print(f"[INFO] 摄像头 {camera_id} 尝试第 {reconnect_count} 次重连", flush=True)
            time.sleep(RECONNECT_INTERVAL)
            cap = open_capture()
            read_fail_count = 0
            continue

        ret, frame = cap.read()

        if not ret:
            read_fail_count += 1
            print(f"[WARNING] 摄像头 {camera_id} 读取失败 ({read_fail_count}/{MAX_READ_FAIL})", flush=True)

            if read_fail_count >= MAX_READ_FAIL:
                print(f"[ERROR] 摄像头 {camera_id} 判定断流，重新拉流", flush=True)
                cap.release()
                cap = None
            time.sleep(0.05)
            continue

        # ====== 正常读到帧 ======
        read_fail_count = 0
        frame_counter += 1
        frame_buffer.append(frame.copy())

        # # ==== 跳帧检测 ====
        if frame_counter % SKIP_INTERVAL != 0:
            continue

        # 获取电子围栏区域
        with fence_lock:
            fence_area = fence_dict.get(int(camera_id))

        with algorithm_lock:
            algorithm_type = int(algorithm_dict.get(int(camera_id), 1))

        parts = algorithmtypes.split(",")

        if "1" in parts and "2" in parts:
            # print("同时包含 1 和 2")
            intrusion_active, _ = detect_intrusion(frame, camera_id, rtsp_url, config, frame_buffer, FPS, ip_address)

            if intrusion_active:
                check_helmet_and_alert(frame, camera_id, rtsp_url, config, frame_buffer, FPS, ip_address)
        elif "1" in parts:
            # print("只包含 1")
            intrusion_active, _ = detect_intrusion(frame, camera_id, rtsp_url, config, frame_buffer, FPS, ip_address)
        elif "2" in parts:
            # print("只包含 2")
            check_helmet_and_alert(frame, camera_id, rtsp_url, config, frame_buffer, FPS, ip_address)
        elif "3" in parts:
            detect_overcrowding(frame, camera_id, rtsp_url, config, frame_buffer, FPS, ip_address)
        else:
            print("没有包含任何算法")
    if cap is not None:
        cap.release()
    print(f"[INFO] 摄像头 {camera_id} 流结束", flush=True)


if __name__ == "__main__":
    config = load_config()
    cams = get_cameras()
    # cams = cams[:1]   # 只取一路
    # 取最后一路
    # cams = [cams[-1]]
    print(cams)
    # cams = get_camera_by_id(1997855393744818240)
    # cams = [('1997855393744818240', 'rtsp://admin:zh5555002@10.164.60.4:554/Streaming/Channels/6701', '172.16.18.77')]
    # cams = [('1997855044199911425', 'rtsp://admin:ad123456@10.161.60.101:554/Streaming/Channels/1601', '10.209.151.115')]
    #1997855393744818240 172.16.18.77
    start_all_streams(config, fence_lock, algorithm_lock, fence_dict, algorithm_dict, latest_pre_frames, cams)

    # 阻塞主线程，避免退出
    while True:
        time.sleep(100)

