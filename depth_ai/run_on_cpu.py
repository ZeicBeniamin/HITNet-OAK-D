import openvino.runtime as ov
import cv2 as cv
import numpy as np

left = cv.imread("/home/bz/Documents/SpatialAI/TinyHITNet/_gendir/mono_left/2022_12_07__00_00_24.png")
right = cv.imread("/home/bz/Documents/SpatialAI/TinyHITNet/_gendir/mono_right/2022_12_07__00_00_24.png")

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
compiled_model = core.compile_model("/home/bz/hitnet_out/onnxs/HITNet_SF_oak_sized_model_cpu.blob", "CPU")
infer_request = compiled_model.create_infer_request()


left_tensor = ov.Tensor(array=left, shared_memory=False)
right_tensor = ov.Tensor(array=right, shared_memory=False)

infer_request.set_input_tensor(0, left_tensor)
infer_request.set_input_tensor(1, right_tensor)

infer_request.infer()

output = infer_request.get_output_tensor()
output_buffer: np.ndarray = output.data

from colormap import apply_colormap, dxy_colormap

# print("left.shape", left.shape)
# left = left.squeeze(0)

# left = torch.from_numpy(left).unsqueeze(0)
# print("left.shape", left.shape)

# disp = torch.from_numpy(output_buffer)
# disp = torch.clip(disp / 192 * 255, 0, 255).long()
# disp = apply_colormap(disp)

# print("disp.shape", disp.shape)
# print("left.shape", left.shape)

# output = [left, disp]

# output = torch.cat(output, dim=0)
# torchvision.utils.save_image(output, "disparity_cpu.png", nrow=1)

# print(output_buffer.shape)