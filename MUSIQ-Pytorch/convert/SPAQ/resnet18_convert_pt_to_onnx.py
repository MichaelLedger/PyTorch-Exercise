import torch
import torchvision

# Load the PyTorch model
model = torchvision.models.resnet18(pretrained=True)

# Create an example input tensor
example_input = torch.randn(1, 3, 224, 224)

# Export the model to ONNX
torch.onnx.export(model, example_input, "resnet18.onnx")
