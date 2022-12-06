import depthai as dai
import typing

bbBlobPath = "/home/bz/hitnet_out/onnxs/disparity_only_output.blob"

pipeline = dai.Pipeline()

camLeft = pipeline.create(dai.node.MonoCamera)
camLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
camLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

camRight = pipeline.create(dai.node.MonoCamera)
camRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
camRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

cam = pipeline.create(dai.node.ColorCamera)

nn = pipeline.create(dai.node.NeuralNetwork)
nn.setBlobPath(bbBlobPath)

inputs: (dai.Node.Input) = nn.getInputs()

print(inputs.)
print(len(inputs))

# Send NN out to the host via XLink
nnXout = pipeline.create(dai.node.XLinkOut)
nnXout.setStreamName("nn")
nn.out.link(nnXout.input)

# with dai.Device(pipeline) as device:
#   qNn = device.getOutputQueue("nn")

#   nnData = qNn.get() # Blocking

#   # NN can output from multiple layers. Print all layer names:
#   print(nnData.getAllLayerNames())

#   # Get layer named "Layer1_FP16" as FP16
#   layer1Data = nnData.getLayerFp16("Layer1_FP16")

#   # You can now decode the output of your NN