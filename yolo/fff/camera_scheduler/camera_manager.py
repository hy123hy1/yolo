# camera_manager.py
import threading
import time
from camera_worker import process_stream

RUNNING = {}
# cam_id -> { thread, stop_event, cam }

def cam_changed(old, new):
    return old != new

def start_camera(cam, config):
    stop_event = threading.Event()

    t = threading.Thread(
        target=process_stream,
        args=(cam, config, stop_event),
        daemon=True
    )
    t.start()

    RUNNING[cam["id"]] = {
        "thread": t,
        "stop_event": stop_event,
        "cam": cam
    }

    print(f"[MANAGER] 启动摄像头 {cam['id']}")

def stop_camera(cam_id):
    info = RUNNING.get(cam_id)
    if not info:
        return

    info["stop_event"].set()
    info["thread"].join(timeout=5)
    RUNNING.pop(cam_id, None)

    print(f"[MANAGER] 停止摄像头 {cam_id}")

def poll_loop(get_cameras, config, interval=5):
    while True:
        cams = get_cameras()
        cam_map = {c["id"]: c for c in cams}

        # 新增
        for cam_id, cam in cam_map.items():
            if cam_id not in RUNNING:
                start_camera(cam, config)

        # 删除
        for cam_id in list(RUNNING.keys()):
            if cam_id not in cam_map:
                stop_camera(cam_id)

        # 修改
        for cam_id, info in list(RUNNING.items()):
            new_cam = cam_map.get(cam_id)
            if new_cam and cam_changed(info["cam"], new_cam):
                stop_camera(cam_id)
                start_camera(new_cam, config)

        time.sleep(interval)
