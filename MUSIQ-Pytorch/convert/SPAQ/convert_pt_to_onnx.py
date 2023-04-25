import torch
import torchvision
import argparse
import os

# Define the device to load the model onto
device = torch.device('cpu')

# Load the PyTorch model
#model = torchvision.models.resnet18(pretrained=True)

# Load the model from file
model = torch.load("BL_release.pt", map_location=device)

# Print the input tensor
# the keys of the model dictionary
print(model.keys())
#print(model['input'])

# Create an example input tensor
#input_tensor = torch.randn(1, 3, 224, 224)

#input_str = "Hello, world!"
#input_tensor = torch.as_tensor(input_str)

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_1', type=str, default='./images/05293.png')
    parser.add_argument('--image_2', type=str, default='./images/00914.png')
    return parser.parse_args()

#parser = argparse.ArgumentParser()
#parser.add_argument('--input', type=str, default='./images/05293.png', help='path to input image')
#args = parser.parse_args()

#cfg = parse_config()
#input_tensor = torch.load(cfg.input_tensor)

# Running Error
# TypeError: new(): invalid data type 'str'
input_str = "./images/05293.png"
#input_tensor = torch.Tensor([input_str], dtype=torch.str)
input_tensor = torch.tensor(input_str)

# Export the model to ONNX
torch.onnx.export(model, input_tensor, "BL_release.onnx")
