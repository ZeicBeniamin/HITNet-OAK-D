import depthai as dai
import numpy as np
import matplotlib.pyplot as plt
import time

bbBlobPath = "/home/bz/hitnet_out/onnxs/oak_sized_model.blob"
# bbBlobPath = "/home/bz/hitnet_out/onnxs/HITNet_SF_oak_sized_model.blob"

pipeline = dai.Pipeline()

camLeft = pipeline.create(dai.node.MonoCamera)
camLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
camLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

leftManip = pipeline.create(dai.node.ImageManip)
leftManip.initialConfig.setResize(100, 180)
camLeft.out.link(leftManip.inputImage)

camRight = pipeline.create(dai.node.MonoCamera)
camRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
camRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

rightManip = pipeline.create(dai.node.ImageManip)
rightManip.initialConfig.setResize(100, 180)
camRight.out.link(rightManip.inputImage)

cam = pipeline.create(dai.node.ColorCamera)

nn = pipeline.create(dai.node.NeuralNetwork)
nn.setBlobPath(bbBlobPath)
nn.setNumInferenceThreads(2)

leftManip.out.link(nn.inputs['left'])
leftManip.out.link(nn.inputs['right'])

# Send NN out to the host via XLink
nnXout = pipeline.create(dai.node.XLinkOut)
nnXout.setStreamName("nn")
nn.out.link(nnXout.input)

with dai.Device(pipeline) as device:
  start_time = int(time.time())
  counter = 0
  while True:
    qNn = device.getOutputQueue("nn")

    nnData = qNn.get() # Blocking

    # NN can output from multiple layers. Print all layer names:
    # print(nnData.getAllLayerNames())

    # Get layer named "4078" as FP16
    layer1Data = nnData.getLayerFp16("4078")
    # layer1Data = nnData.getLayerFp16("1614")

    depth_map = np.reshape(np.array(layer1Data) - min(layer1Data), (100, 180))

    # plt.imshow(depth_map)
    # plt.show()

    counter += 1
    # You can now decode the output of your NN
    current_time = int(time.time())
    diff_time = (current_time - start_time)

    if diff_time % 10 == 0:
      print(counter/diff_time) 