#!/usr/bin/env python3

import cv2
import depthai as dai
import time

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
xoutLeft = pipeline.create(dai.node.XLinkOut)
xoutRight = pipeline.create(dai.node.XLinkOut)

xoutLeft.setStreamName('left')
xoutRight.setStreamName('right')

# Properties
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

# Linking
monoRight.out.link(xoutRight.input)
monoLeft.out.link(xoutLeft.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queues will be used to get the grayscale frames from the outputs defined above
    qLeft = device.getOutputQueue(name="left", maxSize=4, blocking=False)
    qRight = device.getOutputQueue(name="right", maxSize=4, blocking=False)

    while True:
        # Instead of get (blocking), we use tryGet (non-blocking) which will return the available data or None otherwise
        inLeft = qLeft.get()
        inRight = qRight.get()

        timestamp = time.strftime('%Y_%m_%d__%H_%M_%S')

        if inLeft is not None and inRight is not None:
            leftFrame = inLeft.getCvFrame()
            cv2.imshow("left", leftFrame)

            rightFrame = inRight.getCvFrame()
            cv2.imshow("right", rightFrame)

            cv2.imwrite(f"../_gendir/mono_left/{timestamp}.png", leftFrame)
            cv2.imwrite(f"../_gendir/mono_right/{timestamp}.png", rightFrame)

        if cv2.waitKey(1) == ord('q'):
            break