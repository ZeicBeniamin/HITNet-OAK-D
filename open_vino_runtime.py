import openvino.runtime as ov
import cv2 as cv
import numpy as np
import torchvision
import torch

def np2torch(x, t=True, bgr=False):
    if len(x.shape) == 2:
        x = x[..., None]
    if bgr:
        x = x[..., [2, 1, 0]]
    if t:
        x = np.transpose(x, (2, 0, 1))
    if x.dtype == np.uint8:
        x = x.astype(np.float32) / 255
    x = torch.from_numpy(x.copy())
    return x



left = cv.imread("/home/bz/Documents/SpatialAI/kitti_2012/testing/colored_0/000000_10.png", cv.IMREAD_COLOR)
right = cv.imread("/home/bz/Documents/SpatialAI/kitti_2012/testing/colored_1/000000_10.png", cv.IMREAD_COLOR)

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
compiled_model = core.compile_model("/home/bz/hitnet_out/onnxs/disparity_only_output.onnx", "MYRIAD")
infer_request = compiled_model.create_infer_request()


left_tensor = ov.Tensor(array=left, shared_memory=False)
right_tensor = ov.Tensor(array=right, shared_memory=False)

infer_request.set_input_tensor(0, left_tensor)
infer_request.set_input_tensor(1, right_tensor)

infer_request.infer()

output = infer_request.get_output_tensor()
output_buffer: np.ndarray = output.data

from colormap import apply_colormap, dxy_colormap

print("left.shape", left.shape)
left = left.squeeze(0)

left = torch.from_numpy(left).unsqueeze(0)
print("left.shape", left.shape)

disp = torch.from_numpy(output_buffer)
disp = torch.clip(disp / 192 * 255, 0, 255).long()
disp = apply_colormap(disp)

print("disp.shape", disp.shape)
print("left.shape", left.shape)

output = [left, disp]

output = torch.cat(output, dim=0)
torchvision.utils.save_image(output, "disparity.png", nrow=1)

print(output_buffer.shape)