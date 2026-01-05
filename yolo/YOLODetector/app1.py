from flask import Flask, request, jsonify
from flask_cors import CORS
from multiprocessing import Process
from config_loader import load_config
import threading
from common import rect_to_polygon
from collections import defaultdict
from push import stop_stream, stream_worker, stream_worker2
import cv2


"""
常量区
"""
# 电子围栏区域（动态更新）
fence_dict = {}
algorithm_dict = {}

fence_lock = threading.Lock()
algorithm_lock = threading.Lock()

# 存储当前运行的摄像头线程 & process
active_streams = {}
stream_stop_flags = {}

# 导入模型
# model = YOLO('./models/yolov8n.pt')

# 全局缓存（按摄像头）
latest_pre_frames = defaultdict(list)

"""
启动服务
"""

app = Flask(__name__)
CORS(app)
lock = threading.Lock()
config = load_config()

"""
API 接口
"""
# 根路由
@app.route("/")
def index():
    return "Service is running", 200

# 接收围栏设置请求
@app.route("/set_fence", methods=["POST"])
def set_fence():
    data = request.get_json() or {}
    algorithmType = data.get("algorithmType")
    rtsp_url = data.get("url")
    rect = data.get("fence_area")
    default_area = data.get("default_area")
    camera_id = data.get("cam_id")

    cap = cv2.VideoCapture(rtsp_url)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25

    print(f"[DEBUG] 摄像头已打开: {width}x{height}@{fps}fps", flush=True)

    try:
        if rect:
            fence_area = rect_to_polygon(rect, width, height)
    except Exception as e:
        return jsonify({"status": "error", "message": f"fence_area 格式错误: {e}"}), 400

    print(f"[INFO] 请求启动 摄像头={camera_id} rtsp={rtsp_url} fence={fence_area} algorithmType={algorithmType} default_area={default_area}", flush=True)

    # ==== 若已存在旧流，先彻底停止 ====
    if camera_id in active_streams:
        print(f"[INFO] 摄像头 {camera_id} 已在运行，尝试停止旧流")
        try:
            stop_stream(camera_id, fence_dict)  # 停止旧线程
            del active_streams[camera_id]
        except Exception as e:
            print(f"[WARN] 停止旧流失败: {e}")

    # ==== 启动独立进程运行 FFmpeg ====
    p = Process(target=stream_worker, args=(camera_id, rtsp_url, fence_area))
    p.daemon = True
    p.start()
    active_streams[camera_id] = p
    # print(f"[DEBUG] 已启动 FFmpeg 进程 ........................................", flush=True)

    # ==== 设置围栏 ====
    with fence_lock:
        fence_dict[camera_id] = fence_area

    with algorithm_lock:
        algorithm_dict[int(camera_id)] = algorithmType

    output_url = f"rtsp://10.168.60.52:8553/Streaming/Channels/{camera_id}"
    print("url", output_url)

    return jsonify({
        "status": "success",
        "camera_id": camera_id,
        "output_url": output_url
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005, debug=False)
