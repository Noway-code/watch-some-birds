import os
import cv2
import time
import asyncio
import logging
import traceback
import aiohttp
from time import perf_counter
from dotenv import load_dotenv
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput

# ==============================
# Config
# ==============================

CLIP_DURATION = 5
MOTION_AREA_THRESHOLD = 900
OUTPUT_DIR = "out"

MAIN_SIZE = (1280, 720)
LORES_SIZE = (640, 360)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ==============================
# Environment
# ==============================

load_dotenv()
ENDPOINT = os.getenv("UPLOAD_ENDPOINT")

if not ENDPOINT:
    raise RuntimeError("UPLOAD_ENDPOINT not set")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# Camera Wrapper
# ==============================


class Camera:
    def __init__(self):
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": MAIN_SIZE, "format": "YUV420"},
            lores={"size": LORES_SIZE, "format": "YUV420"},
        )
        self.picam2.configure(config)

    def start(self):
        self.picam2.start()

    def stop(self):
        self.picam2.stop()
        self.picam2.close()

    def get_lores(self):
        frame = self.picam2.capture_array("lores")
        return frame[: LORES_SIZE[1], :]


# ==============================
# Upload Logic
# ==============================


async def upload_file(filepath: str, filename: str) -> bool:
    attempts = 5
    backoff = 3

    timeout = aiohttp.ClientTimeout(total=15)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for attempt in range(1, attempts + 1):
            try:
                with open(filepath, "rb") as f:
                    data = aiohttp.FormData()
                    data.add_field(
                        "file", f, filename=filename, content_type="video/mp4"
                    )

                    async with session.post(ENDPOINT, data=data) as resp:
                        if resp.status == 200:
                            logger.info(f"Upload success: {filename}")
                            return True
                        raise Exception(f"Bad status: {resp.status}")

            except Exception as e:
                logger.warning(f"Upload failed (attempt {attempt}): {e}")

            delay = backoff * (2 ** (attempt - 1))
            await asyncio.sleep(delay)

    return False


def cleanup(filepath: str):
    try:
        os.remove(filepath)
        logger.info(f"Successfully cleaned up {filepath}")
    except Exception:
        logger.exception("Failed to delete file")


# ==============================
# Motion Detection
# ==============================


def detect_motion(frame1, frame2) -> bool:
    diff = cv2.absdiff(frame1, frame2)
    blur = cv2.GaussianBlur(diff, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return any(cv2.contourArea(c) > MOTION_AREA_THRESHOLD for c in contours)


# ==============================
# Recorder
# ==============================


class Recorder:
    def __init__(self, picam2):
        self.picam2 = picam2
        self.recording = False
        self.writer = None
        self.start_time = perf_counter()
        self.filepath = None
        self.filename = None
        self.encoder = H264Encoder(bitrate=10_000_000)

    def start(self):
        self.recording = True
        self.start_time = perf_counter()
        self.filename = time.strftime("%Y%m%d_%H%M%S") + ".mp4"
        self.filepath = os.path.join(OUTPUT_DIR, self.filename)

        output = FfmpegOutput(self.filepath)
        self.picam2.start_recording(self.encoder, output)

        logger.info(f"Recording started: {self.filename}")

    def should_stop(self) -> bool:
        return (perf_counter() - self.start_time) >= CLIP_DURATION

    def stop(self):
        self.picam2.stop_recording()
        self.recording = False

        logger.info(f"Recording finished: {self.filename}")
        return self.filepath, self.filename


# ==============================
# Main Loop
# ==============================


def main():
    camera = Camera()
    camera.start()

    recorder = Recorder(camera.picam2)

    logger.info("Camera started")

    frame1 = camera.get_lores()

    try:
        while True:
            frame2 = camera.get_lores()

            if detect_motion(frame1, frame2) and not recorder.recording:
                recorder.start()

            if recorder.recording:
                if recorder.should_stop():
                    filepath, filename = recorder.stop()
                    if not filepath or not filename:
                        logger.error("Upload failed permanently")
                        raise Exception("File not created")

                    success = asyncio.run(upload_file(filepath, filename))

                    if success:
                        cleanup(filepath)
                    else:
                        logger.error("Upload failed permanently")

            frame1 = frame2

    except Exception:
        logger.exception("Fatal error in main loop")

    finally:
        if recorder.recording:
            recorder.stop()

        camera.stop()
        logger.info("Camera stopped")


# ==============================
# Entry
# ==============================

if __name__ == "__main__":
    main()
