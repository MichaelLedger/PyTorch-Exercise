# https://github.com/ducky-hong/pytorch-dcase-task3/blob/9272ce7b5ac2f4838908704f38d0392d519262c2/export_to_onnx.py
import io
import numpy as np

import torch
import torch.onnx

import model.model as model_arch

#checkpoint = 'saved/Bulbul/1219_202445/checkpoint-epoch100.pth'
#checkpoint = 'saved/BAD_MobileNetV2/1219_130017/checkpoint-epoch100.pth'
checkpoint = 'saved/BAD_MobileNetV2Reduced/1219_100213/checkpoint-epoch100.pth'
out_name = checkpoint.split('/')
out_name = out_name[1] + out_name[2] + '.onnx'

batch_size = 1

#model = model_arch.BulbulModel()
model = model_arch.MobileNetV2Reduced(n_class=2)
model.load_state_dict(torch.load(checkpoint)['state_dict'])
model.cpu()
model.train(False)

#x = torch.randn(batch_size, 1, 80, 716, requires_grad=True)
x = torch.randn(batch_size, 1, 64, 501, requires_grad=True)

torch_out = torch.onnx._export(model, x, out_name, export_params=True)