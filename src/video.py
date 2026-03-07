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
        cv2.putText(m.array, timestamp, origin, font, scale, colour, thickness)

# Initiatlize and Config the Camera
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (1280, 720), "format": "RGB888"}
)
picam2.configure(config)
colour = (0, 255, 0)
origin = (0, 30)
font = cv2.FONT_HERSHEY_SIMPLEX
scale = 1
thickness = 2
        
# Immediatly on frame capture, add timestamp
picam2.pre_callback = apply_timestamp
picam2.start()

# Prepare to output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("out/output.mp4", fourcc, 20, (1280, 720))

start = perf_counter()

frame1 = picam2.capture_array()
frame2 = picam2.capture_array()
while (perf_counter() - start ) < 15:
    # Frame for displaying with grids
    frame_display = frame2.copy()

    # Diff and smear color on motion
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0) 
    _, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if cv2.contourArea(c) > 900:
            print("Movement detected")
            # get the bounding box coordinates
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame_display, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # Add bounded box frame to video output
    out.write(frame_display)

    # Prep next loop
    frame1 = frame2
    frame2 = picam2.capture_array()

out.release()
picam2.stop()
picam2.close()
