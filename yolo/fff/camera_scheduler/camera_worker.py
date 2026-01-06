# camera_worker.py
import time
import cv2

def process_stream(cam, config, stop_event):
    camera_id = cam["id"]
    rtsp = cam["rtsp"]

    cap = None
    backoff = 1

    while not stop_event.is_set():

        # ===== 连接 / 重连 =====
        if cap is None or not cap.isOpened():
            time.sleep(min(backoff, 30))
            backoff *= 2

            if cap:
                cap.release()

            cap = cv2.VideoCapture(rtsp)
            if not cap.isOpened():
                cap = None
                continue

            backoff = 1
            print(f"[INFO] 摄像头 {camera_id} 已连接")

        ret, frame = cap.read()
        if not ret:
            cap.release()
            cap = None
            continue

        # ===== 你的算法逻辑 =====
        # detect_intrusion(...)
        # check_helmet(...)
        # ========================

    if cap:
        cap.release()

    print(f"[INFO] 摄像头 {camera_id} 已退出")
