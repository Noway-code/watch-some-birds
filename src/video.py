from picamera2 import MappedArray, Picamera2
import cv2
import numpy as np
import time
from time import perf_counter
import matplotlib.pyplot as plt
from picamera2.encoders import H264Encoder
from picamera2.outputs import PyavOutput
import requests
import os
import traceback

CLIP_DURATION = 20
MOTION_AREA_THRESHOLD = 900
OUTPUT_DIR = "out"
ENDPOINT = "http://localhost:8000/uploadfile/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

recording = False
record_start = None
writer = None


class Camera:
    def __init__(
        self, main_size: tuple, main_format: str, lores_size: tuple, lores_format: str
    ):
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
            lores={"size": self.lores_size, "format": self.lores_format},
        )
        self.picam2.configure(self.config)

    def set_pre_callback(self, fn):
        self.picam2.pre_callback = fn

    def start(self):
        self.picam2.start()

    def capture_array(self, name="main"):
        return self.picam2.capture_array(name=name)

    def start_encoder(self, name="main"):
        return self.picam2.start_encoder

    def stop(self):
        self.picam2.stop()

    def close(self):
        self.picam2.close()


# Initiatlize and Config the Camera
main_size = (1280, 720)
main_format = "RGB888"
lores_size = (640, 360)
lores_format = "YUV420"
threshold = 900

camera = Camera(main_size, main_format, lores_size, lores_format)
camera.start()

frame1 = camera.capture_array("lores")
frame1 = frame1[: lores_size[1], :]  # Y channel only
filepath = None
try:
    while True:
        frame2 = camera.capture_array("lores")
        frame2 = frame2[: lores_size[1], :]

        diff = cv2.absdiff(frame1, frame2)
        blur = cv2.GaussianBlur(diff, (5, 5), 0)

        _, thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Extremly sensitive
        motion_detected = any(
            cv2.contourArea(c) > MOTION_AREA_THRESHOLD for c in contours
        )

        if motion_detected and not recording:
            print("motion detected -> start recording")

            recording = True
            record_start = perf_counter()

            filename = time.strftime("%Y%m%d_%H%M%S") + ".mp4"
            # store it to out/<currenttime>.mp4
            filepath = os.path.join(OUTPUT_DIR, filename)

            # Setup writer for this clip and will terminate afterwards
            writer = cv2.VideoWriter(
                filepath,
                cv2.VideoWriter_fourcc(*"mp4v"),  # Fourcc - ASCII code for format
                20,  # FPS
                (main_size[0], main_size[1]),  # Width/Height
            )

        # RECORDING IN PROGRESS
        if recording:
            if record_start is None or writer is None or filepath is None:
                break
            frame_main = camera.capture_array("main")
            print(frame_main.shape, frame_main.dtype)
            frame_main = cv2.cvtColor(frame_main, cv2.COLOR_RGB2BGR)
            print(frame_main.shape, frame_main.dtype)
            writer.write(frame_main)

            # If video max time is reached
            if (perf_counter() - record_start) >= CLIP_DURATION:
                print("recording finished")

                writer.release()
                recording = False

                # SEND FILE
                try:
                    with open(filepath, "rb") as f:
                        response = requests.post(
                            ENDPOINT,
                            data=f,
                            headers={"Content-Type": "video/mp4"},
                            stream=True,
                        )
                        print("upload status:", response.status_code)
                except Exception as e:
                    print("upload failed:", e)
        frame1 = frame2
except Exception as e:
    traceback.print_exc()
finally:
    if recording is True and writer is not None:
        writer.release()
        recording = False
    camera.stop()
    camera.close()
