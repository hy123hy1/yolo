from detector import YOLODetector
from collections import defaultdict
# 连续帧计数（防抖）
intrusion_frame_counter = defaultdict(int)

def detect_intrusion(frame, camera_id, rtsp_url, config,
                     frame_buffer, FPS, ip_address, yolo: YOLODetector):

    MIN_FRAMES = config.get("min_intrusion_frames", 25)

    # ===== YOLO 推理 =====
    dets = yolo.detect(frame)

    # 只保留 person
    valid_dets = [
        d for d in dets
        if d[5] == 0   # cls_id == person
    ]

    # ===== 连续帧确认 =====
    if valid_dets:
        intrusion_frame_counter[camera_id] += 1
    else:
        intrusion_frame_counter[camera_id] = 0

    if intrusion_frame_counter[camera_id] < MIN_FRAMES:
        return False, []

    # ===== 事件状态机 =====
    now = datetime.datetime.now()
    state = intrusion_state[camera_id]

    frame_draw = frame.copy()
    for x1, y1, x2, y2, conf, _, label in valid_dets:
        cv2.rectangle(frame_draw, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            frame_draw,
            f"{label} {conf:.2f}",
            (x1, max(y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

    image_url = state.get("image_url")

    # ===== 事件开始 =====
    if not state["active"]:
        state["active"] = True
        state["start_time"] = now

        try:
            image_url = upload_warning_image(frame_draw, camera_id, "person")
            state["image_url"] = image_url
        except Exception as e:
            print(f"[WARN] 上传报警图片失败: {e}", flush=True)

        try:
            pre_frames = list(frame_buffer)
            threading.Thread(
                target=_async_generate_video_and_notify,
                args=(pre_frames, rtsp_url, FPS, camera_id, alarm_queue_video, image_url),
                daemon=True
            ).start()
        except Exception as e:
            print(f"[WARN] 视频线程失败: {e}", flush=True)

        # ===== 事件级报警（只一次）=====
        alarm_queue.put({
            "timestamp": now.isoformat(),
            "camera_id": str(camera_id),
            "count": len(valid_dets),
            "objects": [{
                "label": "person",
                "confidence": round(conf, 2),
                "bbox": [x1, y1, x2, y2],
                "status": "warning",
                "warning_image": image_url,
                "warning_video": ""
            } for x1, y1, x2, y2, conf, _, _ in valid_dets],
            "sxt_ip": str(ip_address)
        })

        threading.Thread(
            target=alarm_dispatcher,
            args=(config, alarm_queue),
            daemon=True
        ).start()

    return True, valid_dets
