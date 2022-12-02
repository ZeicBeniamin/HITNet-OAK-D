import openvino.runtime as ov
import cv2 as cv
import numpy as np

left = cv.imread("/home/bz/Documents/SpatialAI/kitti_2012/testing/colored_0/000000_10.png")
right = cv.imread("/home/bz/Documents/SpatialAI/kitti_2012/testing/colored_1/000000_10.png")

# left = cv.convertScaleAbs(left)
# right = cv.convertScaleAbs(right)

left = left/255
right = right/255

left = np.transpose(left, axes=(2, 0, 1))
right = np.transpose(right, axes=(2, 0, 1))

left = np.expand_dims(left, 0)
right = np.expand_dims(right, 0)

left = left.astype(np.float32)
right = right.astype(np.float32)

print(left.dtype)


core = ov.Core()
compiled_model = core.compile_model("/home/bz/hitnet_out/onnxs/short_clip_tensors.onnx", "CPU")
infer_request = compiled_model.create_infer_request()


left = ov.Tensor(array=left, shared_memory=False)
right = ov.Tensor(array=right, shared_memory=False)

infer_request.set_input_tensor(0, left)
infer_request.set_input_tensor(1, right)

infer_request.infer()

output = infer_request.get_output_tensor()
