from picamera2 import MappedArray, Picamera2
import cv2
import numpy as np
import time
from time import perf_counter
import matplotlib.pyplot as plt
from picamera2.encoders import H264Encoder
from picamera2.outputs import PyavOutput

def apply_timestamp(request):
    timestamp = time.strftime("%Y-%m-%d %X")
    with MappedArray(request, "main") as m:
        cv2.putText(m.array, timestamp, (0,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0,255,0), 2)

class Camera:
    def __init__(self, main_size: tuple, main_format: str, lores_size: tuple, lores_format: str):
        self.picam2 = Picamera2()
        self.main_size = main_size
        self.main_format = main_format
        self.lores_size = lores_size
        self.lores_format = lores_format
        self.config = None
        self._configure()

    def _configure(self):
        # Configure main and lores
        self.config = self.picam2.create_video_configuration(
            main={"size": self.main_size, "format": self.main_format},
            lores={"size": self.lores_size, "format": self.lores_format}
        )
        self.picam2.configure(self.config)

    def set_pre_callback(self, fn):
        self.picam2.pre_callback = fn

    def start(self):
        self.picam2.start()

    def capture_array(self, name="main"):
        return self.picam2.capture_array(name=name)
    
    def start_encoder(self, name='main'):
        return self.picam2.start_encoder

    def stop(self):
        self.picam2.stop()

    def close(self):
        self.picam2.close()


# Initiatlize and Config the Camera
main_size=(1280,720)
main_format="RGB888"
lores_size=(640,360)
lores_format="YUV420"

camera = Camera(main_size, main_format, lores_size, lores_format)

# Immediatly on frame capture, add timestamp
camera.set_pre_callback = apply_timestamp
camera.start()

# Prepare to output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("out/output.mp4", fourcc, 20, (1280, 720))

start = perf_counter()

frame1 = camera.capture_array("lores")
frame1 = frame1[:lores_size[1], :]   # Y channel only

while (perf_counter() - start) < 15:
    frame2 = camera.capture_array("lores")
    frame2 = frame2[:lores_size[1], :]

    diff = cv2.absdiff(frame1, frame2)
    blur = cv2.GaussianBlur(diff, (5,5), 0)

    _, thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if cv2.contourArea(c) > 900:
            print("motion")
    out.write(cv2.cvtColor(frame2, cv2.COLOR_YUV420p2RGB))
    frame1 = frame2

out.release()
camera.stop()
camera.close()
