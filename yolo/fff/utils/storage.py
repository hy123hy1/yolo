import cv2
import time
import io
import uuid
from minio import Minio
import tempfile
import os
import logging



logger = logging.getLogger(__name__)


_minio_client = None



def _get_minio_client(config):
    global _minio_client
    if _minio_client:
        return _minio_client

    cfg = config["MINIO"]
    _minio_client = Minio(
        cfg["endpoint"],
        access_key=cfg["access_key"],
        secret_key=cfg["secret_key"],
        secure=cfg.getboolean("secure")
    )
    return _minio_client

def save_image(frame, config, prefix):
    if not config["STORAGE"].getboolean("save_image"):
        return None

    quality = config["STORAGE"].getint("image_quality", 90)

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return None

    ts = time.strftime("%Y%m%d/%H")
    object_name = f"{prefix}/image/{ts}/{uuid.uuid4().hex}.jpg"

    _upload_minio(buf.tobytes(), object_name, "image/jpeg", config)
    print(f"Uploaded: {object_name}", flush=True)
    return object_name


def save_video(pre_frames, post_frames, config, prefix, frame_size):
    """
    使用临时文件生成视频，然后上传 MinIO
    """
    if not config["STORAGE"].getboolean("save_video"):
        return None

    fps = config["STORAGE"].getint("video_fps", 15)
    codec = config["STORAGE"].get("video_codec", "avc1")
    fourcc = cv2.VideoWriter_fourcc(*codec)

    # === 临时文件 ===
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        video_path = tmp.name

    writer = cv2.VideoWriter(
        video_path,
        fourcc,
        fps,
        frame_size
    )

    for f in list(pre_frames) + list(post_frames):
        writer.write(f)

    writer.release()

    # === 上传 MinIO ===
    ts = time.strftime("%Y%m%d/%H")
    object_name = f"{prefix}/video/{ts}/{uuid.uuid4().hex}.mp4"

    with open(video_path, "rb") as f:
        data = f.read()
        _upload_minio(data, object_name, "video/mp4", config)

    os.remove(video_path)

    logger.info(f"Video uploaded: {object_name}")
    return object_name


def _upload_minio(data, object_name, content_type, config):
    client = _get_minio_client(config)
    bucket = config["MINIO"]["bucket"]

    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)

    client.put_object(
        bucket,
        object_name,
        io.BytesIO(data),
        length=len(data),
        content_type=content_type
    )
    print(f"Uploaded: {bucket}/{object_name}", flush=True)

    # logger.info(f"Uploaded: {bucket}/{object_name}")

