import onnx
from onnxsim import simplify

onnx_model = onnx.load("/home/bz/hitnet_out/onnxs/HITNet_SF_oak_sized_model.onnx")
model, check = simplify(onnx_model)
onnx.save(model, "/home/bz/hitnet_out/onnxs/HITNet_SF_oak_sized_model_simp.onnx")