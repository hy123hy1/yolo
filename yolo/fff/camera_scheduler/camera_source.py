# camera_source.py
import requests

def get_cameras():
    url = "http://xxx/selectSxtInfo"
    resp = requests.get(url, timeout=5)

    if resp.status_code != 200:
        return []

    data = resp.json().get("data", [])

    cams = []
    for item in data:
        if not item.get("algorithmtypes"):
            continue

        cams.append({
            "id": item["id"],
            "rtsp": item["originalRtsp"],
            "ip": item["ipAddress"],
            "algorithmtypes": item["algorithmtypes"]
        })

    return cams


# def get_cameras():
#     url = "http://172.21.3.141:8080/aks-mkaqjcyj/cameraMonitoring/selectSxtInfo"
#     headers = {"User-Agent": "Mozilla/5.0"}
#     resp = requests.get(url, headers=headers)
#
#     if resp.status_code != 200:
#         print("请求失败:", resp.text[:200])
#         return []
#
#     data = resp.json()
#     cameras_data = data.get("data", [])
#     print(f"发现 {len(cameras_data)} 个摄像头配置。")
#
#     # originalRtsp
#     # 过滤掉 algorithmtypes 为空的摄像头
#     valid_cameras = []
#     for item in cameras_data:
#         algorithmtypes = item.get("algorithmtypes")
#         if algorithmtypes:  # 非空才加入
#             valid_cameras.append(
#                 (item.get("id"), item.get("originalRtsp"), item.get("ipAddress"), algorithmtypes)
#             )
#         else:
#             print(f"[跳过] 摄像头 {item.get('id')} 的 algorithmtypes 为空。")
#
#     return valid_cameras