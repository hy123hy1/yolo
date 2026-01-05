import cv2

class VideoStream:
    def __init__(self, config):
        vcfg = config["VIDEO"]
        gcfg = config["GLOBAL"]

        self.fps = int(gcfg["fps"])
        self.pre_seconds = int(gcfg["pre_seconds"])
        self.show = vcfg.getboolean("show_window")

        self.cap = cv2.VideoCapture(vcfg["rtsp_url"], cv2.CAP_FFMPEG)
        print("VideoStream initialized", flush=True)

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame
