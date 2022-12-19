import depthai as dai
import numpy as np
import matplotlib.pyplot as plt
import time

height = 100
width = 180

input_height = 100
input_width = 180

fps = 10

# bbBlobPath = "/home/bz/hitnet_out/onnxs/HITNet_SF_oak_sized_model_simp.blob"
# bbBlobPath = "/home/bz/hitnet_out/onnxs/HITNet_SF_oak_sized_model.blob"
bbBlobPath = "/home/bz/hitnet_out/onnxs/HITNet_SF_Gray_concat.blob"

pipeline = dai.Pipeline()

camLeft = pipeline.create(dai.node.MonoCamera)
camLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
camLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
camLeft.setFps(fps)

leftManip = pipeline.create(dai.node.ImageManip)
leftManip.setNumFramesPool(2)
leftManip.initialConfig.setResize(input_width, input_height)
camLeft.out.link(leftManip.inputImage)

camRight = pipeline.create(dai.node.MonoCamera)
camRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
camRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
camRight.setFps(fps)

rightManip = pipeline.create(dai.node.ImageManip)
rightManip.setNumFramesPool(2)
rightManip.initialConfig.setResize(input_width, input_height)
camRight.out.link(rightManip.inputImage)

nn = pipeline.create(dai.node.NeuralNetwork)
nn.setBlobPath(bbBlobPath)
nn.setNumInferenceThreads(2)

leftManip.out.link(nn.inputs['left'])
rightManip.out.link(nn.inputs['right'])

# Send NN out to the host via XLink
nnXout = pipeline.create(dai.node.XLinkOut)
leftOut = pipeline.create(dai.node.XLinkOut)
nnXout.setStreamName("nn")
leftOut.setStreamName("left")
nn.out.link(nnXout.input)
rightManip.out.link(leftOut.input)

with dai.Device(pipeline) as device:
  device.setLogLevel(dai.LogLevel.DEBUG)
  device.setLogOutputLevel(dai.LogLevel.DEBUG)

  start_time = int(time.time())
  counter = 0
  while True:
    qNn = device.getOutputQueue("nn")
    qLeft = device.getOutputQueue("left")

    nnData = qNn.get() # Blocking
    leftData = qLeft.get()

    # NN can output from multiple layers. Print all layer names:
    # print(nnData.getAllLayerNames())

    # Get layer named "4078" as FP16
    layer1Data = nnData.getLayerFp16("1622")
    # layer1Data = nnData.getLayerFp16("1614")

    if layer1Data:
      depth_map = np.reshape(np.array(layer1Data) - min(layer1Data), (height, width))
      leftImage = leftData.getCvFrame()

      plt.imshow(np.vstack((leftImage, depth_map)))
      plt.show(block=False)
      plt.pause(0.05)
      plt.title("TOP: Mono image, BOTTOM: Depth estimation")

      # plt.imshow(depth_map)
      # plt.show(block=False)
      # plt.pause(1)

      counter += 1
      # You can now decode the output of your NN
      current_time = int(time.time())
      diff_time = (current_time - start_time)

      if diff_time % 10 == 0:
        print(counter/diff_time) 