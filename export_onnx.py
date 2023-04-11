from models.mobilefacenet import MobileFaceNet
from torchsummary import summary

import onnxruntime as ort
import onnx
import torch

model = MobileFaceNet()
# summary(model, (3, 100, 100))
model.eval()
dummy_input = torch.rand((1, 3, 100, 100))

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "N"}, "output": {0: "N"}},
    opset_version=13
)