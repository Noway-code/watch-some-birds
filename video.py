# import the necessary packages
from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv1
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=499, help="minimum area size")
args = vars(ap.parse_args())
# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
    vs = VideoStream(src=-1).start()
    time.sleep(1.0)
# otherwise, we are reading from a video file
else:
    vs = cv1.VideoCapture(args["video"])
# initialize the first frame in the video stream
firstFrame = None