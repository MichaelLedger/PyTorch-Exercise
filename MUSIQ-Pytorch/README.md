# MUSIQ: Multi-Scale Image Quality Transformer
Unofficial pytorch implementation of the paper "MUSIQ: Multi-Scale Image Quality Transformer"
(paper link: https://arxiv.org/abs/2108.05997)

This code doesn't exactly match what the paper describes.
- It only works on the KonIQ-10k dataset. Or it works on the database which resolution is 1024(witdh) x 768(height).
- Instead of using 5-layer Resnet as a backbone network, we use ResNet50 pretrained on ImageNet database.
- We need to implement Earth Mover Distance (EMD) loss to train on other databases.
- We additionally use ranking loss to improve the performance (we will upload the training code including ranking loss later)

The environmental settings are described below. (I cannot gaurantee if it works on other environments)
- Pytorch=1.7.1 (with cuda 11.0)
- einops=0.3.0
- numpy=1.18.3
- cv2=4.2.0
- scipy=1.4.1
- json=2.0.9
- tqdm=4.45.0

# Train & Validation
First, you need to download weights of ResNet50 pretrained on ImageNet database.
- Downlod the weights from this website (https://download.pytorch.org/models/resnet50-0676ba61.pth)
- rename the .pth file as "resnet50.pth" and put it in the "model" folder

Second, you need to download the KonIQ-10k dataset.
- Download the database from this website (http://database.mmsp-kn.de/koniq-10k-database.html)
- set the database path in "train.py" (It is represented as "db_path" in "train.py")
- Please check "koniq-10k.txt" is in "IQA_list" folder
- "koniq-10k.txt" file includes [scene number / image name / ground truth score] information

After those settings, you can run the train & validation code by running "train.py"
- python3 train.py (execution code)
- This code works on single GPU. If you want to train this code in muti-gpu, you need to change this code
- Options are all included in "train.py". So you should change the variable "config" in "train.py"
![image](https://user-images.githubusercontent.com/77471764/138195607-cf7165a1-dd64-4031-b1ab-872012f7046a.png)

Belows are the validation performance on KonIQ-10k database (I'm still training the code, so the results will be updated later)
- SRCC: 0.9023 / PLCC: 0.9232 (after training 105 epochs)
- If the codes are implemented exactly the same as the paper, the performance can be further improved

# Inference
First, you need to specify variables in "inference.py"
- dirname: root folder of test images
- checkpoint: checkpoint file (trained on KonIQ-10k dataset)
- result_score_txt: inference score will be saved on this txt file
![image](https://user-images.githubusercontent.com/77471764/138195041-3176224f-6ab6-42b1-aa61-f9ec8a1ffa96.png)

After those settings, you can run the inference code by running "inference.py"
- python3 inference.py (execution code)

# Acknolwdgements
We refer to the following website to implement the transformer (https://paul-hyun.github.io/transformer-01/)

## 

https://github.com/chaofengc/IQA-PyTorch

`pip3 install pyiqa`

## pytorch trained file .pth transfer to tflite

To transfer a PyTorch trained file with a .pth extension to a TFLite format, you will need to follow a few steps.

First, you will need to convert the PyTorch model to an ONNX format using the torch.onnx.export function. This function takes in the PyTorch model, an input tensor, and the path to save the ONNX model. Here is an example of how to use this function:
```
import torch
import torchvision

# Load the PyTorch model
model = torchvision.models.resnet18(pretrained=True)

# Create an example input tensor
example_input = torch.randn(1, 3, 224, 224)

# Export the model to ONNX format
torch.onnx.export(model, example_input, "resnet18.onnx")
```
Once you have the ONNX model, you can use the TensorFlow Lite converter to convert it to a TFLite format. Here is an example of how to use the converter:
```
import tensorflow as tf

# Convert the ONNX model to a TFLite model
converter = tf.lite.TFLiteConverter.from_onnx_file("resnet18.onnx")
tflite_model = converter.convert()

# Save the TFLite model to a file
with open("resnet18.tflite", "wb") as f:
    f.write(tflite_model)
```
This should allow you to successfully transfer your PyTorch trained model to a TFLite format.


## PyTorch mobile solution
There’s PyTorch mobile solution, it can run PyTorch models on iOS/Android by Mobile libs. Can we try their solution to optimize the PyTorch model to mobile one and try?
https://pytorch.org/mobile/home/
https://pytorch.org/mobile/home/#deployment-workflow
https://github.com/pytorch/ios-demo-app
https://github.com/pytorch/android-demo-app

## Why `torch.cuda.is_available()` returns False even after installing pytorch with cuda?

https://stackoverflow.com/questions/60987997/why-torch-cuda-is-available-returns-false-even-after-installing-pytorch-with

The easiest way to check if PyTorch supports your compute capability is to install the desired version of PyTorch with CUDA support and run the following from a python interpreter

```
>>> import torch
>>> torch.zeros(1).cuda()
```

## Introducing Accelerated PyTorch Training on Mac

https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/

https://pytorch.org/get-started/locally/

```
# MPS acceleration is available on MacOS 12.3+
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```

https://towardsdatascience.com/installing-pytorch-on-apple-m1-chip-with-gpu-acceleration-3351dc44d67c
```
$ conda create -n torch-gpu python=3.8
$ conda inactivate
$ conda activate torch-gpu
$ conda install pytorch torchvision torchaudio -c pytorch-nightly
```

```
$ conda info --envs
# conda environments:
#
base                     /Users/gavinxiang/miniconda3
torch-gpu                /Users/gavinxiang/miniconda3/envs/torch-gpu
                         /Users/gavinxiang/miniforge3/envs/test
                         /usr/local/Caskroom/miniforge/base
                         /usr/local/Caskroom/miniforge/base/envs/mlp
                         /usr/local/Caskroom/miniforge/base/envs/tf
```

https://github.com/chaofengc/IQA-PyTorch

`$ pip3 install pyiqa`

Sanity Check

Next, let’s make sure everything went as expected. That is:

PyTorch was installed successfully.
PyTorch can use the GPU successfully.
To make things easy, install the Jupyter notebook and/or Jupyter lab:

`$ conda install -c conda-forge jupyter jupyterlab`

Now, we will check if PyTorch can find the Metal Performance Shaders plugin. Open the Jupiter notebook and run the following:
```
import torch
import math

# this ensures that the current MacOS version is at least 12.3+
print(torch.backends.mps.is_available())

# this ensures that the current current PyTorch installation was built with MPS activated.
print(torch.backends.mps.is_built())
```

If both commands return True, then PyTorch has access to the GPU!

**To run PyTorch code on the GPU, use `torch.device("mps")` analogous to `torch.device("cuda")` on an Nvidia GPU.**

 Hence, in this example, we move all computations to the GPU.

```
dtype = torch.float
device = torch.device("mps")

# Create random input and output data
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# Randomly initialize weights
a = torch.randn((), device=device, dtype=dtype)
b = torch.randn((), device=device, dtype=dtype)
c = torch.randn((), device=device, dtype=dtype)
d = torch.randn((), device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(2000):
    # Forward pass: compute predicted y
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)

# Backprop to compute gradients of a, b, c, d with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # Update weights using gradient descent
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d


print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')
```

## Warning: Cannot set number of intraop threads

https://blog.csdn.net/Castlehe/article/details/120498239

```
number of train scenes: 8058
number of test scenes: 2015
[W ParallelNative.cpp:230] Warning: Cannot set number of intraop threads after parallel work has started or after set_num_threads call when using native parallel backend (function set_num_threads)
```

无法在并行任务启动后设置线程数量，也无法在使用原生的并行后端时调用set_num_threds。

总的来说，就是这个问题还没解决，只能取消并行，用单线程去运行。
```
import os
os.environ["OMP_NUM_THREADS"] = "1"
```

```
(torch-gpu) macstudio@MacStudios-Mac-Studio MUSIQ-Pytorch % python3 train.py
Using GPU 1
number of train scenes: 8058
number of test scenes: 2015
  0%|                                                                 | 0/1007 [00:00<?, ?it/s]Using GPU 1
number of train scenes: 8058
number of test scenes: 2015
Using GPU 1
number of train scenes: 8058
number of test scenes: 2015
Using GPU 1
number of train scenes: 8058
number of test scenes: 2015
Using GPU 1
number of train scenes: 8058
number of test scenes: 2015
Using GPU 1
number of train scenes: 8058
number of test scenes: 2015
Using GPU 1
number of train scenes: 8058
number of test scenes: 2015
Using GPU 1
number of train scenes: 8058
number of test scenes: 2015
Using GPU 1
number of train scenes: 8058
number of test scenes: 2015
/Users/macstudio/miniconda3/envs/torch-gpu/lib/python3.8/site-packages/torch/autograd/__init__.py:204: UserWarning: The operator 'aten::sgn.out' is not currently supported on the MPS backend and will fall back to run on the CPU. This may have performance implications. (Triggered internally at /Users/runner/work/_temp/anaconda/conda-bld/pytorch_1682147327173/work/aten/src/ATen/mps/MPSFallback.mm:12.)
  Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
  0%|▏                                                     | 3/1007 [02:36<13:36:23, 48.79s/it]
```

## TRAINING WITH PYTORCH

https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

```
    return _VF.dropout_(input, p, training) if inplace else _VF.dropout(input, p, training)
RuntimeError: MPS backend out of memory (MPS allocated: 17.01 GB, other allocations: 1.07 GB, max allowed: 18.13 GB). Tried to allocate 152.30 MB on private pool. Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable upper limit for memory allocations (may cause system failure).
```
## CPU only training
```
[W ParallelNative.cpp:230] Warning: Cannot set number of intraop threads after parallel work has started or after set_num_threads call when using native parallel backend (function set_num_threads)
100%|███████████████████████████████████████████████████████████████| 1007/1007 [6:07:09<00:00, 21.88s/it]
[train] epoch:1 / loss:0.304822 / SROCC:0.516225 / PLCC:0.430035

100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1007/1007 [9:30:33<00:00, 34.00s/it]
[train] epoch:3 / loss:0.222481 / SROCC:0.759811 / PLCC:0.795320
start saving weights of epoch 3
save weights of epoch 3
  0%|                                                                                                                                                          | 0/251 [00:00<?, ?it/s]will save weights of epoch 100 to path: ./weights/epoch100.pth
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 251/251 [40:38<00:00,  9.71s/it]
test epoch:3 / loss:0.308393 /SROCC:0.808958 / PLCC:0.839890

100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 251/251 [40:30<00:00,  9.68s/it]
test epoch:4 / loss:0.170981 /SROCC:0.820304 / PLCC:0.851920  

100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1007/1007 [9:48:05<00:00, 35.04s/it]
[train] epoch:12 / loss:0.163302 / SROCC:0.860065 / PLCC:0.885368
start saving weights of epoch 12
save weights of epoch 12
  0%|                                                                                                                                                          | 0/251 [00:00<?, ?it/s]
Traceback (most recent call last):
  File "train.py", line 164, in <module>
    training()
  File "train.py", line 161, in training
    loss, rho_s, rho_p = eval_epoch(config, epoch, model_transformer, model_backbone, criterion, test_loader)
  File "/Users/macstudio/Documents/MUSIQ-Pytorch/trainer.py", line 103, in eval_epoch
    for data in tqdm(test_loader):
  File "/Users/macstudio/miniconda3/envs/torch-gpu/lib/python3.8/site-packages/tqdm/std.py", line 1178, in __iter__
    for obj in iterable:
  File "/Users/macstudio/miniconda3/envs/torch-gpu/lib/python3.8/site-packages/torch/utils/data/dataloader.py", line 441, in __iter__
    return self._get_iterator()
  File "/Users/macstudio/miniconda3/envs/torch-gpu/lib/python3.8/site-packages/torch/utils/data/dataloader.py", line 388, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
  File "/Users/macstudio/miniconda3/envs/torch-gpu/lib/python3.8/site-packages/torch/utils/data/dataloader.py", line 1042, in __init__
    w.start()
  File "/Users/macstudio/miniconda3/envs/torch-gpu/lib/python3.8/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
  File "/Users/macstudio/miniconda3/envs/torch-gpu/lib/python3.8/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
  File "/Users/macstudio/miniconda3/envs/torch-gpu/lib/python3.8/multiprocessing/context.py", line 284, in _Popen
    return Popen(process_obj)
  File "/Users/macstudio/miniconda3/envs/torch-gpu/lib/python3.8/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/Users/macstudio/miniconda3/envs/torch-gpu/lib/python3.8/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/Users/macstudio/miniconda3/envs/torch-gpu/lib/python3.8/multiprocessing/popen_spawn_posix.py", line 54, in _launch
    child_r, parent_w = os.pipe()
OSError: [Errno 24] Too many open files
```
## GPU training log 2023/4/25 13:17
```
(torch-gpu) macstudio@MacStudios-Mac-Studio MUSIQ-Pytorch % python3 train.py
will save weights of epoch 105 to path: ./weights/epoch105.pth
Using GPU 1
number of train scenes: 8058
number of test scenes: 2015
/Users/macstudio/miniconda3/envs/torch-gpu/lib/python3.8/site-packages/torch/autograd/__init__.py:204: UserWarning: The operator 'aten::sgn.out' is not currently supported on the MPS backend and will fall back to run on the CPU. This may have performance implications. (Triggered internally at /Users/runner/work/_temp/anaconda/conda-bld/pytorch_1682147327173/work/aten/src/ATen/mps/MPSFallback.mm:12.)
  Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
100%|███████████████████████████████████████████████████████████████████████████████████████████████| 1007/1007 [14:19:07<00:00, 51.19s/it]
[train] epoch:1 / loss:0.328861 / SROCC:0.451107 / PLCC:0.348204
start saving weights of epoch 1
save weights of epoch 1
```

https://github.com/pytorch/pytorch/issues/80278

This is indeed expected. Currently we don't have support for bitwise ops which is used while print is applied on a tensor on MPS device. For formatting the output those ops are used and currently we fallback to CPU hence the warning. We are looking into adding it as discussed here.

## Print multiple variables in Python

https://www.includehelp.com/python/print-multiple-variables.aspx

```
# printing variables one by one
print(name)
print(age)
print(country)
print() # prints a newline

# printing variables one by one
# with messages
print("Name:", name)
print("Age:", age)
print("Country:", country)

# printing the multiple variables separated by the commas
print("{0} {1} {2}".format(variable1, variable2, variable2)
```

## Check multiple conditions in if statement – Python

https://www.geeksforgeeks.org/check-multiple-conditions-in-if-statement-python/

```
if (cond1 AND/OR COND2) AND/OR (cond3 AND/OR cond4):
    code1
else:
    code2
```
```
age = 18

if ((age>= 8) and (age<= 12)):
    print("YOU ARE ALLOWED. WELCOME !")
else:
    print("SORRY ! YOU ARE NOT ALLOWED. BYE !")
```
```
var = 'N'

if (var =='Y' or var =='y'):
    print("YOU SAID YES")
elif(var =='N' or var =='n'):
    print("YOU SAID NO")
else:
    print("INVALID INPUT")

```

## What is the difference between .pt, .pth and .pwf extentions in PyTorch?

https://stackoverflow.com/questions/59095824/what-is-the-difference-between-pt-pth-and-pwf-extentions-in-pytorch

There are no differences between the extensions that were listed: .pt, .pth, .pwf. One can use whatever extension (s)he wants. So, if you're using torch.save() for saving models, then it by default uses python pickle (pickle_module=pickle) to save the objects and some metadata. Thus, you have the liberty to choose the extension you want, as long as it doesn't cause collisions with any other standardized extensions.

Having said that, it is however not recommended to use .pth extension when checkpointing models because it collides with Python path (.pth) configuration files. Because of this, I myself use .pth.tar or .pt but not .pth, or any other extensions.


## Exit codes in Python

https://stackoverflow.com/questions/285289/exit-codes-in-python

https://docs.python.org/2/library/sys.html#sys.exit

`sys.exit()`

```
import sys, os

try:
    config()
except:
    sys.exit(os.EX_CONFIG) 
try:
    do_stuff()
except:
    sys.exit(os.EX_SOFTWARE)
sys.exit(os.EX_OK) # code 0, all ok
```

## TORCH.JIT.SAVE
https://pytorch.org/docs/stable/generated/torch.jit.save.html

```
import torch
import io

class MyModule(torch.nn.Module):
    def forward(self, x):
        return x + 10

m = torch.jit.script(MyModule())

# Save to file
torch.jit.save(m, 'scriptmodule.pt')
# This line is equivalent to the previous
m.save("scriptmodule.pt")

# Save to io.BytesIO buffer
buffer = io.BytesIO()
torch.jit.save(m, buffer)

# Save with extra files
extra_files = {'foo.txt': b'bar'}
torch.jit.save(m, 'scriptmodule.pt', _extra_files=extra_files)
```

## Optimize model for Mobile

LibTorch_Lite with .ptl

https://pytorch.org/mobile/home/


## TORCH.TENSOR.NEW_ZEROS
https://pytorch.org/docs/stable/generated/torch.Tensor.new_zeros.html#torch.Tensor.new_zeros

## TORCH.JIT.TRACE

https://pytorch.org/docs/stable/generated/torch.jit.trace.html?highlight=torch+jit+trace#torch.jit.trace

## Cursor Question: python sample[:, :, 0] = (sample[:, :, 0] - self.mean[0]) / self.var[0] meanings 

Cursor Answer:

Based on the provided code, it seems like you are trying to normalize the first channel of an image using the mean and variance values stored in the self.mean and self.var arrays.

The code you provided is indexing the first channel of the sample array using sample[:, :, 0] and then subtracting the mean value of the first channel (self.mean[0]) from it. Finally, it divides the result by the variance value of the first channel (self.var[0]). This operation will normalize the first channel of the image.

However, it's important to note that this code only normalizes the first channel of the image. If you want to normalize all channels, you will need to perform this operation for each channel separately.

Here's an example of how you can normalize all channels of an image using the mean and variance values stored in self.mean and self.var:

```
# Assuming sample is a 3-channel image
for i in range(3):
    sample[:, :, i] = (sample[:, :, i] - self.mean[i]) / self.var[i]
```

This code will loop over each channel of the image and normalize it using the corresponding mean and variance values stored in self.mean and self.var.

I hope this helps! Let me know if you have any further questions.

## at::IntArrayRef usage in LibTorch C++

Usage of at::IntArrayRef in LibTorch C++

at::IntArrayRef is a lightweight class in LibTorch C++ that represents a reference to an array of integers. It is commonly used in functions that take a variable number of integer arguments, such as tensor indexing operations.

Here is an example of how at::IntArrayRef is used in the LibTorch C++ codebase:

```
#include <ATen/ATen.h>

void foo(at::Tensor tensor, at::IntArrayRef sizes) {
  // Resize the tensor to the given sizes
  tensor.resize_(sizes);
}

int main() {
  // Create a tensor of size (2, 3)
  auto tensor = at::ones({2, 3});

  // Resize the tensor to size (4, 5)
  foo(tensor, {4, 5});

  return 0;
}
```

In this example, the foo function takes a at::Tensor object and an at::IntArrayRef object as arguments. The sizes argument is used to resize the tensor to the given sizes.

To create an at::IntArrayRef object, you can pass a pointer to an array of integers and the length of the array to the constructor. Alternatively, you can use the at::ArrayRef> constructor to create an at::IntArrayRef object from a std::vector>.

Here is an example of how to create an at::IntArrayRef object:
```
#include <ATen/ATen.h>

void bar(at::IntArrayRef sizes) {
  // Print the sizes of the tensor
  for (int size : sizes) {
    std::cout << size << " ";
  }
  std::cout << std::endl;
}

int main() {
  // Create an IntArrayRef object from an array of integers
  int sizes[] = {2, 3};
  auto int_array_ref = at::IntArrayRef(sizes, 2);

  // Create an IntArrayRef object from a vector of integers
  std::vector<int> vec_sizes = {4, 5};
  auto vec_array_ref = at::ArrayRef<int>(vec_sizes);

  // Call the bar function with the IntArrayRef objects
  bar(int_array_ref);
  bar(vec_array_ref);

  return 0;
}
```
In this example, the bar function takes an at::IntArrayRef object as an argument and prints the sizes of the tensor. The int_array_ref object is created from an array of integers, while the vec_array_ref object is created from a vector of integers using the at::ArrayRef> constructor.

I hope this helps! Let me know if you have any further questions.

## convert pred = model_transformer(mask_inputs, feat_dis_org, feat_dis_scale_1, feat_dis_scale_2) to  LibTorch C++

To convert the line pred = model_transformer(mask_inputs, feat_dis_org, feat_dis_scale_1, feat_dis_scale_2) to LibTorch C++, you will need to first define the model_transformer function in C++.

Assuming that model_transformer is a PyTorch model defined in Python, you will need to export the model to a C++ script module using the torch.jit.script function. Here is an example of how to do this:

```
import torch

class ModelTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define your PyTorch model here

    def forward(self, mask_inputs, feat_dis_org, feat_dis_scale_1, feat_dis_scale_2):
        # Define the forward pass of your PyTorch model here
        return output

# Create an instance of the PyTorch model
model = ModelTransformer()

# Export the PyTorch model to a C++ script module
script_module = torch.jit.script(model)
script_module.save("model.pt")
```

To convert the line pred = model_transformer(mask_inputs, feat_dis_org, feat_dis_scale_1, feat_dis_scale_2) to LibTorch C++, you will need to first define the model_transformer function in C++.

Assuming that model_transformer is a PyTorch model defined in Python, you will need to export the model to a C++ script module using the torch.jit.script function. Here is an example of how to do this:



This will save the exported C++ script module to a file named model.pt.

Once you have the C++ script module, you can load it in your C++ code using the torch::jit::load function and call its forward method to make predictions. Here is an example of how to do this:
```
#include <torch/script.h>

int main() {
    // Load the C++ script module
    torch::jit::script::Module module = torch::jit::load("model.pt");

    // Create input tensors
    torch::Tensor mask_inputs = ...;
    torch::Tensor feat_dis_org = ...;
    torch::Tensor feat_dis_scale_1 = ...;
    torch::Tensor feat_dis_scale_2 = ...;

    // Call the forward method of the C++ script module to make predictions
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(mask_inputs);
    inputs.push_back(feat_dis_org);
    inputs.push_back(feat_dis_scale_1);
    inputs.push_back(feat_dis_scale_2);
    torch::Tensor output = module.forward(inputs).toTensor();
}
```

Replace the ... with your actual input tensors. Note that you will need to include the appropriate headers and link against the LibTorch library in your C++ code.

## from model.model_main import IQARegression

https://yilinwang.org/papers/271.pdf

STRUCTURE-PRESERVING IMAGE QUALITY ASSESSMENT

https://eudl.eu/pdf/10.4108/eai.13-7-2017.2269980

A No-reference IQA Metric Based on BEMD and Riesz Tansform

## use IQA in libpytorch c++

Based on my understanding, IQA stands for Image Quality Assessment. If you are looking to use IQA in libtorch (a C++ library for PyTorch), then you can try using the TorchVision package which provides several pre-trained models to perform various image tasks including IQA.

To use the pre-trained IQA models in TorchVision, you can follow these steps:

1. Install TorchVision:
`pip install torchvision`
2. Load the pre-trained model:
```
   #include <torch/torch.h>
   #include <torch/script.h>
   
   // Load the model
   torch::jit::script::Module module = torch::jit::load("path/to/IQA/model.pt");
```
3. Preprocess the input image:
```
   // Load the input image
   cv::Mat img = cv::imread("path/to/image.jpg");
   
   // Convert the image to a tensor
   torch::Tensor tensor_image = torch::from_blob(img.data, {1, img.rows, img.cols, 3}, torch::kByte);
   
   // Preprocess the image tensor
   tensor_image = tensor_image.permute({0, 3, 1, 2}).to(torch::kFloat);
   tensor_image = tensor_image.div(255).sub(0.5).div(0.5);
```
4. Evaluate the input image using the pre-trained model:
```
   // Evaluate the input tensor
   at::Tensor output = module.forward({tensor_image}).toTensor();
   
   // Get the predicted IQA score
   float iqa_score = output.item<float>();
```
By following these steps, you should be able to use the pre-trained IQA models in TorchVision for C++. Let me know if this helps or if you have any further questions.

## we wanna to use CVPixelBuffer to normalized image instead of opencv
CVPixelBuffer is a Core Video pixel buffer and is commonly used on iOS and macOS platforms. If you want to use CVPixelBuffer to normalize an image instead of OpenCV, you can use the following code:

1. Create a CVPixelBuffer from a UIImage:
```
   - (CVPixelBufferRef) pixelBufferFromUIImage:(UIImage *)image
   {
       CGSize frameSize = CGSizeMake(CGImageGetWidth(image.CGImage), CGImageGetHeight(image.CGImage));
       NSDictionary *options = @{(id) kCVPixelBufferCGImageCompatibilityKey: @(YES),
                                 (id) kCVPixelBufferCGBitmapContextCompatibilityKey: @(YES)};

       CVPixelBufferRef pixelBuffer = NULL;
       CVReturn result = CVPixelBufferCreate(kCFAllocatorDefault,
                                             frameSize.width,
                                             frameSize.height,
                                             kCVPixelFormatType_32BGRA,
                                             (__bridge CFDictionaryRef) options,
                                             &pixelBuffer);
       
       NSParameterAssert(result == kCVReturnSuccess && pixelBuffer != NULL);
       
       CVPixelBufferLockBaseAddress(pixelBuffer, 0);
       void *data = CVPixelBufferGetBaseAddress(pixelBuffer);
       CGColorSpaceRef rgbColorSpace = CGColorSpaceCreateDeviceRGB();
       CGContextRef context = CGBitmapContextCreate(data,
                                                    frameSize.width,
                                                    frameSize.height,
                                                    8,
                                                    kCVPixelFormatType_32BGRA,
                                                    rgbColorSpace,
                                                    kCGImageAlphaNoneSkipFirst);
       
       NSParameterAssert(context);
       CGContextConcatCTM(context, CGAffineTransformIdentity);
       CGContextDrawImage(context, CGRectMake(0, 0, CGImageGetWidth(image.CGImage), CGImageGetHeight(image.CGImage)), image.CGImage);
       CGColorSpaceRelease(rgbColorSpace);
       CGContextRelease(context);
       
       CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
       return pixelBuffer;
   }
```
2. Normalize the pixel buffer data and convert it to a tensor:
```
   // Convert the pixel buffer to a tensor
   CVPixelBufferRef pixelBuffer = [self pixelBufferFromUIImage:self.originalImage];
   int width = (int) CVPixelBufferGetWidth(pixelBuffer);
   int height = (int) CVPixelBufferGetHeight(pixelBuffer);
   float *pixelBufferData = (float *) CVPixelBufferGetBaseAddress(pixelBuffer);
   torch::Tensor tensor_image = torch::from_blob(pixelBufferData, {1, height, width, 4}, at::kFloat32);
   
   // Release the pixel buffer
   CVPixelBufferRelease(pixelBuffer);
   
   // Convert from RGBA to RGB and normalize
   tensor_image = tensor_image.permute({0, 3, 1, 2}).to(torch::kFloat);
   tensor_image = tensor_image.slice(1, 0, 3);
   tensor_image = tensor_image.div(255).sub(0.5).div(0.5);
```
By using CVPixelBuffer to normalize the image data, you should be able to achieve the same result as using OpenCV. Let me know if this helps or if you have any further questions.

## Different output from Libtorch C++ and pytorch

https://stackoverflow.com/questions/63502473/different-output-from-libtorch-c-and-pytorch

before the final normalization, you need to scale your input to the range 0-1 and then carry on the normalization you are doing. convert to float and then divide by 255 should get you there. Here is the snippet I wrote, there might be some syntaax errors, that should be visible.
Try this :

```
#include <opencv2/opencv.hpp>
#include <torch/script.h>


int main()
{
    try
    {
        torch::jit::script::Module model = torch::jit::load("traced_facelearner_model_new.pt");
        model.to(torch::kCUDA);
        
        cv::Mat visibleFrame = cv::imread("example.jpg");

        cv::resize(visibleFrame, visibleFrame, cv::Size(112, 112));
        at::Tensor tensor_image = torch::from_blob(visibleFrame.data, {  visibleFrame.rows, 
                                                    visibleFrame.cols, 3 }, at::kByte);
        
        tensor_image = tensor_image.to(at::kFloat).div(255).unsqueeze(0);
        tensor_image = tensor_image.permute({ 0, 3, 1, 2 });
        ensor_image.sub_(0.5).div_(0.5);

        tensor_image = tensor_image.to(torch::kCUDA);
        // Execute the model and turn its output into a tensor.
        auto output = model.forward({tensor_image}).toTensor();
        output = output.cpu();
        std::cout << "Embds: " << output << std::endl;

        std::cout << "Done!\n";
    }
    catch (std::exception e)
    {
        std::cout << "exception" << e.what() << std::endl;
    }
}
```

## Linear Regression using PyTorch C++ API (Libtorch) on CSV Files: Code Review and Demo Run!

https://www.youtube.com/watch?v=6raFznPFy2Y

https://github.com/krshrimali/Linear-Regression-using-PyTorch-CPP

Implementing Linear Regression on a CSV file using PyTorch C++ Frontend API.

This video contains the demo run and code review of the project. The project has been open sourced here: https://github.com/BuffetCodes/Linear....

If you are using CMake for the first time, please follow this tutorial: http://derekmolloy.ie/hello-world-int.... The official documentation is always the best though: https://cmake.org/cmake/help/latest/g....

Find me here:
Twitter: http://twitter.com/kushashwa
Facebook: https://www.facebook.com/v.krshrimali
GitHub: https://github.com/krshrimali
Reddit: https://www.reddit.com/user/Kushashwa/

## Building a Regression Model in PyTorch

https://machinelearningmastery.com/building-a-regression-model-in-pytorch/

Summary

In this post, you discovered the use of PyTorch to build a regression model.

You learned how you can work through a regression problem step-by-step with PyTorch, specifically:

How to load and prepare data for use in PyTorch

How to create neural network models and choose a loss function for regression

How to improve model accuracy by applying standard scaler

## LOADING A TORCHSCRIPT MODEL IN C++

https://pytorch.org/tutorials/advanced/cpp_export.html

/Users/gavinxiang/Downloads/PyTorch-Exercise/MUSIQ-Pytorch/model/backbone.py

/Users/gavinxiang/Downloads/PyTorch-Exercise/MUSIQ-Pytorch/model/model_main.py

/Users/gavinxiang/Downloads/PyTorch-Exercise/MUSIQ-Pytorch/model/resnet50.pth

## Attempted to read a PyTorch file with version 10, but the maximum supported version for reading is 6.
```
2023-05-07 14:36:17.669559+0800 PyTorchDemo[1139:235188] version_ <= kMaxSupportedFileFormatVersionINTERNAL ASSERT FAILED at "/Users/distiller/project/caffe2/serialize/inline_container.cc":146, please report a bug to PyTorch. Attempted to read a PyTorch file with version 10, but the maximum supported version for reading is 6. Your PyTorch installation may be too old.
Exception raised from init at /Users/distiller/project/caffe2/serialize/inline_container.cc:146 (most recent call first):
```

`torch.jit.save(traced_script_module, 'IQA_scriptmodule.pt', _use_new_zipfile_serialization=False)`

`Traceback (most recent call last):
  File "convert_IQA_torch_script.py", line 133, in <module>
    torch.jit.save(traced_script_module, 'IQA_scriptmodule.pt', _use_new_zipfile_serialization=False)
TypeError: save() got an unexpected keyword argument '_use_new_zipfile_serialization'`

```
platform :ios, '14.0'
target 'PyTorchDemo' do
  # pod 'LibTorch', '~>1.9.0'
  pod 'LibTorch', '~> 1.13.0.1'
end
```


## TORCH.CAT

https://pytorch.org/docs/stable/generated/torch.cat.html

tensors (sequence of Tensors) – any python sequence of tensors of the same type. Non-empty tensors provided must have the same shape, except in the cat dimension.

dim (int, optional) – the dimension over which the tensors are concatenated

```
>>> x = torch.randn(2, 3)
>>> x
tensor([[ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497]])
>>> torch.cat((x, x, x), 0)
tensor([[ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497],
        [ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497],
        [ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497]])
>>> torch.cat((x, x, x), 1)
tensor([[ 0.6580, -1.0969, -0.4614,  0.6580, -1.0969, -0.4614,  0.6580,
         -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497, -0.1034, -0.5790,  0.1497, -0.1034,
         -0.5790,  0.1497]])
```

## TORCH.MATMUL

https://pytorch.org/docs/stable/generated/torch.matmul.html?highlight=torch+matmul#torch.matmul

Matrix product of two tensors.

The behavior depends on the dimensionality of the tensors as follows:

If both tensors are 1-dimensional, the dot product (scalar) is returned.
If both arguments are 2-dimensional, the matrix-matrix product is returned.
If the first argument is 1-dimensional and the second argument is 2-dimensional, a 1 is prepended to its dimension for the purpose of the matrix multiply. After the matrix multiply, the prepended dimension is removed.
If the first argument is 2-dimensional and the second argument is 1-dimensional, the matrix-vector product is returned.

This operation has support for arguments with sparse layouts. In particular the matrix-matrix (both arguments 2-dimensional) supports sparse arguments with the same restrictions as torch.mm()


## Answering the Query: "koniq-10k score range"

The Koniq-10k dataset is a large-scale image quality assessment dataset that contains 10,073 distorted images. Each image is rated by at least 20 human observers on a scale from 1 to 5, where 1 indicates the lowest quality and 5 indicates the highest quality.

The score range for the Koniq-10k dataset is from 1 to 5, with 1 being the lowest quality and 5 being the highest quality. The scores are provided in the file "koniq10kscoresanddistributions.csv" which is located in the "koniq10k" folder of the dataset.

To access the scores in Python, you can use the Pandas library to read the CSV file and extract the scores. Here is an example code snippet:
```
import pandas as pd

# Load the scores CSV file
scores_df = pd.read_csv("koniq10k/koniq10k_scores_and_distributions.csv")

# Extract the scores column
scores = scores_df["MOS"]

# Print the score range
print("Score range: [{}, {}]".format(scores.min(), scores.max()))
```

This code will load the scores CSV file, extract the MOS (Mean Opinion Score) column, and print the score range. The output will be in the form of "[minscore, maxscore]".

I hope this helps answer your query. Let me know if you have any further questions.


## Convert Pytorch Model to Core Machine Learning Model

https://developer.apple.com/documentation/coreml?language=objc

https://coremltools.readme.io/docs/pytorch-conversion-examples

https://github.com/yulingtianxia/Core-ML-Sample

https://coremltools.readme.io/docs/image-inputs

https://coremltools.readme.io/docs/flexible-inputs

Enumerated Shapes with Multi-input Models

For a multi-input model, only one of the inputs can be marked with EnumeratedShapes; the rest must have fixed single shapes. If you require multiple inputs to be flexible, set the range for each dimension.

```
$ conda activate torch-gpu   
$ pip3 install coremltools
$ python3 convert_IQA_ML_script.py
$ python3 convert_resnet50_ML_script.py
```

**Cursor QA**
Q:

**convert pytorch model which input like (mask_inputs, feat_dis_org, feat_dis_scale_1, feat_dis_scale_2) to coreml**

A:

To convert a PyTorch model with multiple inputs to Core ML, you can use the coremltools library. First, you need to install the library if you haven't already:

`pip install coremltools`

Next, you can follow these steps to convert your PyTorch model to Core ML:

1. Make sure your PyTorch model is in evaluation mode and trace it using torch.jit.trace:

```
import torch

# Assuming your model is called `model`
model.eval()

# Create dummy inputs with the same shape as your actual inputs
mask_inputs = torch.randn(1, 3, 224, 224)  # Replace with the correct shape
feat_dis_org = torch.randn(1, 10)  # Replace with the correct shape
feat_dis_scale_1 = torch.randn(1, 10)  # Replace with the correct shape
feat_dis_scale_2 = torch.randn(1, 10)  # Replace with the correct shape

# Trace the model
traced_model = torch.jit.trace(model, (mask_inputs, feat_dis_org, feat_dis_scale_1, feat_dis_scale_2))
```

2. Convert the traced model to Core ML using coremltools:

```
import coremltools as ct

# Convert the traced model to Core ML
coreml_model = ct.convert(
    traced_model,
    inputs=[
        ct.TensorType(name="mask_inputs", shape=(1, 3, 224, 224)),  # Replace with the correct shape
        ct.TensorType(name="feat_dis_org", shape=(1, 10)),  # Replace with the correct shape
        ct.TensorType(name="feat_dis_scale_1", shape=(1, 10)),  # Replace with the correct shape
        ct.TensorType(name="feat_dis_scale_2", shape=(1, 10)),  # Replace with the correct shape
    ],
)
```

3. Save the Core ML model to a file:

`coreml_model.save("MyModel.mlmodel")`

Now you have successfully converted your PyTorch model with multiple inputs to Core ML. You can use the generated MyModel.mlmodel file in your iOS or macOS applications 

Q:

**how to add description and author for mlmodel**

A:

To add a description and author information to your Core ML model, you can set the short_description, author, and other metadata properties of the coreml_model object. Here's how you can do it:

```
# Set the metadata properties
coreml_model.short_description = "Your model description goes here"
coreml_model.author = "Your name or organization goes here"
coreml_model.license = "Your license information goes here"
coreml_model.version = "1.0"  # You can set the version number as a string

# Save the Core ML model to a file with the updated metadata
coreml_model.save("MyModel.mlmodel")
```

Now, when you save your Core ML model to a file, it will include the description, author, license, and version information you provided. This metadata can be useful for documentation purposes and for identifying the model's purpose and origin when used in iOS or macOS applications 

## ResNet

The ResNet model is based on the Deep Residual Learning for Image Recognition paper.

https://pytorch.org/vision/main/models/resnet.html

https://arxiv.org/abs/1512.03385

ResNet-50 from Deep Residual Learning for Image Recognition.

https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html

Deep Residual Learning for Image Recognition

https://arxiv.org/pdf/1512.03385.pdf

## ImageNet pretrained ResNet50 backbones are different between Pytorch and TensorFlow

https://stackoverflow.com/questions/67365237/imagenet-pretrained-resnet50-backbones-are-different-between-pytorch-and-tensorf

There are 2 things that differ in the implementations of ResNet50 in TensorFlow and PyTorch that I could notice and might explain your observation.

The batch normalization does not have the same momentum in both. It's 0.1 in PyTorch and 0.01 in TensorFlow (although it is reported as 0.99 I am writing it down in PyTorch's convention for comparison here). This might affect training and therefore the weights.
TensorFlow's implementation uses biases in convolutions while PyTorch's one doesn't (as can be seen in the conv3x3 and conv1x1 definitions). Because the batch normalization layers are affine, the biases are not needed, and are spurious. I think this is truly what explains the difference in your case since they can be compensated by the batch norm, and therefore be arbitrarily large, which would be why you observe a bigger range for TF. Another way to see this is to compare the summaries as I did in this colab.
I currently have a PR that should fix the bias part (at least provide the possibility to train a resnet without conv bias in TF), and plan on submitting one for BN soon.

EDIT

I have actually found out more differences, that I listed in a paper I recently wrote. You can check them in Table 3 of the F appendix.

https://ar5iv.labs.arxiv.org/html/2206.13424#A6.T3

I list here for completeness of the answer, those that might have an impact on the output features statistics:

the variance estimation in the batch norm is different
the convolution weights and classification head weights and bias initialization are not the same

## swift use coreML model with custom input params named param1 and param2

To use a Core ML model with custom input parameters named param1 and param2 in Swift, you'll need to follow these steps:

1. Import the necessary frameworks.
2. Create a class for your Core ML model.
3. Create a function to handle the input parameters and make a prediction using the model.

Here's an example of how you can achieve this:

1. Import the necessary frameworks:
```
import UIKit
import CoreML
```
2. Create a class for your Core ML model. Assuming you have a model named MyModel, you can create a class like this:
```
class MyModel {
    private let model: MLModel
    
    init() {
        guard let modelURL = Bundle.main.url(forResource: "MyModel", withExtension: "mlmodelc") else {
            fatalError("Failed to find model URL.")
        }
        
        guard let model = try? MLModel(contentsOf: modelURL) else {
            fatalError("Failed to load model.")
        }
        
        self.model = model
    }
}
```
3. Create a function to handle the input parameters and make a prediction using the model. In this example, we'll create a function named predict that takes two parameters, param1 and param2, and returns the prediction result:
```
extension MyModel {
    func predict(param1: Double, param2: Double) -> String {
        // Create a dictionary with the input parameters
        let inputParameters: [String: Any] = [
            "param1": param1,
            "param2": param2
        ]
        
        // Create a prediction input using the input parameters
        guard let input = try? MLDictionaryFeatureProvider(dictionary: inputParameters) else {
            fatalError("Failed to create input.")
        }
        
        // Make a prediction using the model
        guard let output = try? model.prediction(from: input) else {
            fatalError("Failed to make prediction.")
        }
        
        // Extract the prediction result from the output
        guard let result = output.featureValue(for: "result")?.stringValue else {
            fatalError("Failed to get prediction result.")
        }
        
        return result
    }
}
```
Now you can use the MyModel class to make predictions with custom input parameters param1 and param2:
```
let myModel = MyModel()
let predictionResult = myModel.predict(param1: 1.0, param2: 2.0)
print("Prediction result: \(predictionResult)")
```
Please note that you'll need to replace "MyModel" with the actual name of your Core ML model, and adjust the input and output parameter names and types according to your specific model 


## swift use coreML model with input shape MultiArray (Float32 1 × 3 × 224 × 224)

To use a Core ML model with an input shape of MultiArray (Float32 1 × 3 × 224 × 224) in Swift, you need to follow these steps:

1. Import the necessary frameworks:
```
import UIKit
import CoreML
import Vision
```
2. Load the Core ML model:
```
guard let modelURL = Bundle.main.url(forResource: "YourModel", withExtension: "mlmodelc") else {
    fatalError("Failed to find the model file.")
}
guard let model = try? VNCoreMLModel(for: MLModel(contentsOf: modelURL)) else {
    fatalError("Failed to load the Core ML model.")
}
```
Replace "YourModel" with the name of your Core ML model file.
3. Create a VNCoreMLRequest object and set the completion handler:
```
let request = VNCoreMLRequest(model: model) { (request, error) in
    guard let results = request.results as? [VNClassificationObservation] else {
        fatalError("Failed to process the request.")
    }
    
    // Handle the results
    print(results)
}
```
4. Prepare the input image:
```
guard let image = UIImage(named: "yourImageName") else {
    fatalError("Failed to load the input image.")
}
```
5. Resize the input image to match the model's input shape (224 × 224):
```
func resizeImage(_ image: UIImage, to newSize: CGSize) -> UIImage? {
    UIGraphicsBeginImageContextWithOptions(newSize, false, 0.0)
    image.draw(in: CGRect(origin: .zero, size: newSize))
    let resizedImage = UIGraphicsGetImageFromCurrentImageContext()
    UIGraphicsEndImageContext()
    return resizedImage
}

let resizedImage = resizeImage(image, to: CGSize(width: 224, height: 224))
```
6. Convert the resized image to a CVPixelBuffer:
```
func buffer(from image: UIImage) -> CVPixelBuffer? {
    let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue, kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
    var pixelBuffer: CVPixelBuffer?
    let status = CVPixelBufferCreate(kCFAllocatorDefault, Int(image.size.width), Int(image.size.height), kCVPixelFormatType_32ARGB, attrs, &pixelBuffer)
    guard status == kCVReturnSuccess else {
        return nil
    }
    
    CVPixelBufferLockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
    let pixelData = CVPixelBufferGetBaseAddress(pixelBuffer!)
    
    let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
    let context = CGContext(data: pixelData, width: Int(image.size.width), height: Int(image.size.height), bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer!), space: rgbColorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)
    
    context?.translateBy(x: 0, y: image.size.height)
    context?.scaleBy(x: 1.0, y: -1.0)
    
    UIGraphicsPushContext(context!)
    image.draw(in: CGRect(x: 0, y: 0, width: image.size.width, height: image.size.height))
    UIGraphicsPopContext()
    
    CVPixelBufferUnlockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
    
    return pixelBuffer
}

guard let pixelBuffer = buffer(from: resizedImage!) else {
    fatalError("Failed to convert the input image to a CVPixelBuffer.")
}
```
7. Create a VNImageRequestHandler object and perform the request:
```
let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
do {
    try handler.perform([request])
} catch {
    print("Failed to perform the request: \(error.localizedDescription)")
}
```
Now, the Core ML model will process the input image, and the results will be printed in the completion handler 

## How to reduce a Core ML Model for an iOS app?

https://stackoverflow.com/questions/52998909/how-to-reduce-a-core-ml-model-for-an-ios-app

 have a model that's almost 500mb that a client wants to implement into their iOS app. HOW IN THE WORLD is a model like this implemented, with it being the size it is? It seems to be a fairly popular model, but there is no documentation or posts anywhere of people actually using the model in a mobile app. Is there such thing as hosting the model somewhere and pointing the client to the server hosting the model? Any options to compress the model? Would love to hear what you've done to solve this issue.

The model is found here: https://coreml.store/cnnemotions

People doing academic research on deep learning / computer vision often use VGG16 or VGG19 as their base model because it's simple and works well. Unfortunately it also has over 100 million parameters, making it unsuitable for use on mobile.

The solution here is to re-architect the model using a smaller feature extractor such as MobileNet or SqueezeNet, and train it again. Now you'll end up with a model that is much smaller, between 10 and 20 MB.

This does mean you can't just grab models off of the Internet and convert them to Core ML and expect them to work well. It requires a bit more work than that!

Reply:

I appreciate your time and your answer. You're right. In order to get a model "onboard" a mobile device, using SqueezeNet ended up being a good option. I had to abandon the original model I mentioned in my question, and train my own model, but it works (generally speaking).

## MLShapedArray

/// MLShapedArray is a N-dimensional array with non-contiguous strides.
@available(macOS 12.0, iOS 15.0, watchOS 8.0, tvOS 15.0, *)

UIImage to MLMultiArray ?

https://github.com/hollance/CoreMLHelpers/issues/5

## If a large core ML model is loaded in the main thread by iPhone not Mac, it will definitely crash!

**[DEBUG] Core ML Models load success, cost 280560.92229166825 ms in Mac**

```
2023-06-02 11:02:34.984912+0800 ScoreImage[99666:1229249] [default] LSPrefs: could not find untranslocated node for <FSNode 0x600002364380> { isDir = ?, path = '/private/var/folders/wk/frkkcch539lc6s2dk6dw9dy80000gn/X/C0A25DDE-C009-5BD1-B0BB-AB0B8778FD54/d/Wrapper/ScoreImage.app' }, proceeding on the assumption it is not translocated: Error Domain=NSPOSIXErrorDomain Code=1 "Operation not permitted"
2023-06-02 11:02:34.988683+0800 ScoreImage[99666:1229249] [default] LSPrefs: could not find untranslocated node for <FSNode 0x600002364380> { isDir = ?, path = '/private/var/folders/wk/frkkcch539lc6s2dk6dw9dy80000gn/X/C0A25DDE-C009-5BD1-B0BB-AB0B8778FD54/d/Wrapper/ScoreImage.app' }, proceeding on the assumption it is not translocated: Error Domain=NSPOSIXErrorDomain Code=1 "Operation not permitted"
2023-06-02 11:02:34.992023+0800 ScoreImage[99666:1229249] [default] LSPrefs: could not find untranslocated node for <FSNode 0x600002364380> { isDir = ?, path = '/private/var/folders/wk/frkkcch539lc6s2dk6dw9dy80000gn/X/C0A25DDE-C009-5BD1-B0BB-AB0B8778FD54/d/Wrapper/ScoreImage.app' }, proceeding on the assumption it is not translocated: Error Domain=NSPOSIXErrorDomain Code=1 "Operation not permitted"
2023-06-02 11:02:35.131297+0800 ScoreImage[99666:1229249] [SceneConfiguration] Info.plist contained no UIScene configuration dictionary (looking for configuration named "(no name)")
2023-06-02 11:02:35.131360+0800 ScoreImage[99666:1229249] [SceneConfiguration] Info.plist contained no UIScene configuration dictionary (looking for configuration named "(no name)")
2023-06-02 11:02:35.131405+0800 ScoreImage[99666:1229249] [SceneConfiguration] Info.plist contained no UIScene configuration dictionary (looking for configuration named "(no name)")
[DEBUG] Start to load Core ML Models...
2023-06-02 11:02:35.300707+0800 ScoreImage[99666:1229249] [default] LSPrefs: could not find untranslocated node for <FSNode 0x600002364380> { isDir = ?, path = '/private/var/folders/wk/frkkcch539lc6s2dk6dw9dy80000gn/X/C0A25DDE-C009-5BD1-B0BB-AB0B8778FD54/d/Wrapper/ScoreImage.app' }, proceeding on the assumption it is not translocated: Error Domain=NSPOSIXErrorDomain Code=1 "Operation not permitted"
2023-06-02 11:02:35.529017+0800 ScoreImage[99666:1229249] [SceneConfiguration] Info.plist contained no UIScene configuration dictionary (looking for configuration named "(no name)")
2023-06-02 11:02:36.394397+0800 ScoreImage[99666:1229326] Metal API Validation Enabled
2023-06-02 11:02:37.480074+0800 ScoreImage[99666:1229249] [plugin] AddInstanceForFactory: No factory registered for id <CFUUID 0x600002275540> 30010C1C-93BF-11D8-8B5B-000A95AF9C6A
2023-06-02 11:02:38.571625+0800 ScoreImage[99666:1229249] [] CMIO_DAL_CMIOExtension_Device.mm:347:Device legacy uuid isn't present, using new style uuid instead
2023-06-02 11:02:38.643543+0800 ScoreImage[99666:1229334] IOMFB: FigInstallVirtualDisplay not found
2023-06-02 11:02:38.643635+0800 ScoreImage[99666:1229334] iomfb_populate_display_infos: Local call to iomfb_match_callback
2023-06-02 11:02:38.643720+0800 ScoreImage[99666:1229334] iomfb_match_callback: primary
2023-06-02 11:02:38.643788+0800 ScoreImage[99666:1229334] iomfb_match_callback: external-0
2023-06-02 11:02:38.643840+0800 ScoreImage[99666:1229334] iomfb_match_callback: external-1
2023-06-02 11:02:38.660397+0800 ScoreImage[99666:1229249] [plugin] AddInstanceForFactory: No factory registered for id <CFUUID 0x6000023642e0> F8BB1C28-BAE8-11D6-9C31-00039315CD46
2023-06-02 11:02:45.167332+0800 ScoreImage[99666:1229445] [] CMIO_DP_HardwarePlugInInterface.cpp:791:HardwarePlugIn_StreamCopyBufferQueue CMIO::DP::HardwarePlugIn_StreamCopyBufferQueue: no queue altered proc
2023-06-02 11:02:45.167619+0800 ScoreImage[99666:1229445] [] CMIO_DAL_PlugIn.cpp:371:StreamCopyBufferQueue Error: 1852797029, got an error from the plug-in routine
2023-06-02 11:02:45.167750+0800 ScoreImage[99666:1229445] [] CMIOHardware.cpp:1222:CMIOStreamCopyBufferQueue Error: 1852797029, failed
2023-06-02 11:02:45.168252+0800 ScoreImage[99666:1229445] [] CMIOHardware.cpp:1388:CMIOStreamRegisterAsyncStillCaptureCallback stream doesn't support async still capture
2023-06-02 11:02:45.168375+0800 ScoreImage[99666:1229445] [] CMIOHardware.cpp:1412:CMIOStreamRegisterAsyncStillCaptureCallback Error: 1970171760, failed
2023-06-02 11:02:45.191872+0800 ScoreImage[99666:1229249] [API] cannot add handler to 3 from 3 - dropping
2023-06-02 11:02:45.507141+0800 ScoreImage[99666:1229249] [API] cannot add handler to 3 from 3 - dropping
2023-06-02 11:02:45.678501+0800 ScoreImage[99666:1229249] +[CATransaction synchronize] called within transaction
2023-06-02 11:02:48.520380+0800 ScoreImage[99666:1229249] [API] cannot add handler to 3 from 3 - dropping
2023-06-02 11:06:13.493706+0800 ScoreImage[99666:1229249] [API] cannot add handler to 3 from 3 - dropping
[DEBUG] Core ML Models load success, cost 280560.92229166825 ms
```

```
Details

The app “ScoreImage” on Gavin’s iPhone XS Max quit unexpectedly.
Domain: IDEDebugSessionErrorDomain
Code: 11
Failure Reason: Message from debugger: Terminated due to memory issue
User Info: {
    DVTErrorCreationDateKey = "2023-06-02 02:16:43 +0000";
    IDERunOperationFailingWorker = DBGLLDBLauncher;
}
--

Analytics Event: com.apple.dt.IDERunOperationWorkerFinished : {
    "device_model" = "iPhone11,6";
    "device_osBuild" = "16.0.3 (20A392)";
    "device_platform" = "com.apple.platform.iphoneos";
    "launchSession_schemeCommand" = Run;
    "launchSession_state" = 2;
    "launchSession_targetArch" = arm64;
    "operation_duration_ms" = 11527;
    "operation_errorCode" = 11;
    "operation_errorDomain" = IDEDebugSessionErrorDomain;
    "operation_errorWorker" = DBGLLDBLauncher;
    "operation_name" = IDEiPhoneRunOperationWorkerGroup;
    "param_consoleMode" = 0;
    "param_debugger_attachToExtensions" = 0;
    "param_debugger_attachToXPC" = 1;
    "param_debugger_type" = 5;
    "param_destination_isProxy" = 0;
    "param_destination_platform" = "com.apple.platform.iphoneos";
    "param_diag_MainThreadChecker_stopOnIssue" = 0;
    "param_diag_MallocStackLogging_enableDuringAttach" = 0;
    "param_diag_MallocStackLogging_enableForXPC" = 1;
    "param_diag_allowLocationSimulation" = 1;
    "param_diag_checker_tpc_enable" = 1;
    "param_diag_gpu_frameCapture_enable" = 0;
    "param_diag_gpu_shaderValidation_enable" = 0;
    "param_diag_gpu_validation_enable" = 0;
    "param_diag_memoryGraphOnResourceException" = 0;
    "param_diag_queueDebugging_enable" = 1;
    "param_diag_runtimeProfile_generate" = 0;
    "param_diag_sanitizer_asan_enable" = 0;
    "param_diag_sanitizer_tsan_enable" = 0;
    "param_diag_sanitizer_tsan_stopOnIssue" = 0;
    "param_diag_sanitizer_ubsan_stopOnIssue" = 0;
    "param_diag_showNonLocalizedStrings" = 0;
    "param_diag_viewDebugging_enabled" = 1;
    "param_diag_viewDebugging_insertDylibOnLaunch" = 1;
    "param_install_style" = 0;
    "param_launcher_UID" = 2;
    "param_launcher_allowDeviceSensorReplayData" = 0;
    "param_launcher_kind" = 0;
    "param_launcher_style" = 0;
    "param_launcher_substyle" = 0;
    "param_runnable_appExtensionHostRunMode" = 0;
    "param_runnable_productType" = "com.apple.product-type.application";
    "param_testing_launchedForTesting" = 0;
    "param_testing_suppressSimulatorApp" = 0;
    "param_testing_usingCLI" = 0;
    "sdk_canonicalName" = "iphoneos16.4";
    "sdk_osVersion" = "16.4";
    "sdk_variant" = iphoneos;
}
--


System Information

macOS Version 13.2 (Build 22D49)
Xcode 14.3 (21812) (Build 14E222b)
Timestamp: 2023-06-02T10:16:43+08:00
```

MLModel crash app on init due to memory issue

https://stackoverflow.com/questions/67968988/mlmodel-crash-app-on-init-due-to-memory-issue

Execution speed. You are dispatching threads faster than they can be processed.

In my experience, not on this platform, object detection using a cnn is not fast enough to process every frame from the camera in real-time at 30 fps.

With hardware acceleration, like the "Apple Neural Engine", it is possible (I have an FPGA on my desk that does this task in real time in "hardware" using 15 watts).

I would suggest processing every 50th frame and speed it up until it fails.

The other issue is image size. To be performant the image must be as small as possible and still detect the feature.

**The larger the input image, the more convolution layers are required. Most models are in the smaller ranges like 200x200 pixels.**

## swift core ml decale input as 1 by 912 matrix of floats, same as pytorch at::ones

To declare an input as a 1 by 912 matrix of floats in Swift Core ML, you can use the following code:

`let input = try! MLMultiArray(shape: [1, 912], dataType: .float32)`

This creates a multi-array with a shape of 1 by 912 and a data type of float32. As for setting the values of the array to all ones, you can use the fill method:

`input.fill(1.0)`

This will set all values in the array to 1.0. This is equivalent to the PyTorch at::ones function 

**Value of type 'MLMultiArray' has no member 'fill'**

It seems that you are encountering an error in Swift Core ML where the value of type 'MLMultiArray' has no member 'fill'. This error occurs because the fill method is not a member of the MLMultiArray class in Swift Core ML.

If you are trying to set all values in the array to a specific value, you can use a loop to iterate through each element in the array and set it to the desired value. Here is an example code snippet:

```
let input = try! MLMultiArray(shape: [1, 912], dataType: .float32)
let value: Float = 1.0

for i in 0..<input.count {
    input[i] = value
}
```

This code creates a multi-array with a shape of 1 by 912 and a data type of float32, and then uses a loop to set each element in the array to the value of 1.0.

I hope this helps you resolve your issue! Let me know if you have any further questions 

## Pytorch numpy.transpose(sample, (2, 0, 1)) means

```
// inference.py
# numpy array -> torch tensor
class ToTensor(object):
    def __call__(self, sample):
        sample = np.transpose(sample, (2, 0, 1))
        sample = torch.from_numpy(sample)
        return sample
```

The PyTorch function numpy.transpose(sample, (2, 0, 1)) is used to transpose a tensor. The first argument sample is the tensor to be transposed, and the second argument (2, 0, 1) specifies the order of the dimensions after the transpose operation. In this case, the dimensions are being reordered so that the third dimension becomes the first, the first dimension becomes the second, and the second dimension becomes the third.

For example, if sample is a tensor with shape (3, 4, 5), then numpy.transpose(sample, (2, 0, 1)) will return a tensor with shape (5, 3, 4).

It's worth noting that the transpose operation does not change the underlying data in the tensor, but rather changes the way the data is accessed 

## why need np.transpose(image,(2,0,1)) before feed image to model

https://github.com/isl-org/MiDaS/issues/79

**OpenCV img = cv2.imread(path) loads an image with HWC-layout (height, width, channels), while Pytorch requires CHW-layout. So we have to do np.transpose(image,(2,0,1)) for HWC->CHW transformation.**

## core ml declare image input as shape MultiArray (Float32 1 × 3 × 224 × 224)

To declare an image input as a MultiArray with shape (Float32 1 × 3 × 224 × 224) in Core ML, you can use the following code:
```
import coremltools as ct
import numpy as np

# Define input shape
input_shape = (1, 3, 224, 224)

# Create input feature description
input_description = ct.ImageType(shape=input_shape, bias=[-1,-1,-1], scale=1/127.5)

# Create model input
model_input = ct.TensorType(shape=input_shape)

# Create Core ML model
model = ct.models.MLModel('model.mlmodel')

# Set input description
model.input_description['input'] = input_description

# Set model input
model.input[0] = model_input
```
This code imports the necessary libraries, defines the input shape, creates an input feature description with the specified shape, creates a model input with the same shape, loads the Core ML model, sets the input description to the created input feature description, and sets the model input to the created model input.

Note that the bias and scale parameters in the input feature description are used to preprocess the input image before it is fed into the model. The bias subtracts the mean pixel value from each channel, and the scale divides the pixel values by a constant factor. These values are specific to the model being used and may need to be adjusted accordingly 

## how to define a set of image sizes for ml model input image

To define a set of image sizes for an ML model input image, you can use the ImageType class from the coremltools library. This class allows you to specify the shape of the input image tensor, including the number of channels, height, and width.

Here's an example code snippet:

```
import coremltools as ct

# Define the set of image sizes
input_sizes = [(3, 224, 224), (3, 256, 256), (3, 384, 384)]

# Create a list of ImageType objects
input_types = [ct.ImageType(shape=input_size) for input_size in input_sizes]

# Create the Core ML model specification
model_spec = ct.ModelSpecification(
    input_types=input_types,
    # Add other model specification details here
)

# Create the Core ML model
mlmodel = ct.models.MLModel(model_spec)

# Save the Core ML model to a file
mlmodel.save('model.mlmodel')
```

In this example, input_sizes is a list of tuples, where each tuple represents the desired shape of the input image tensor. You can adjust this list to include the specific image sizes you need for your use case.

The input_types list is created by iterating over input_sizes and creating an ImageType object for each size. This list is then passed to the ModelSpecification constructor, along with any other details about the model specification.

Finally, the MLModel constructor is used to create the Core ML model from the model specification. The resulting mlmodel object can then be saved to a file using the save() method.

Note that this is just an example, and you may need to modify the code to fit your specific use case 

## pip3 show coremltools

Note that you may need to modify the code to fit your specific use case, and that the input_description parameter is only available in coremltools version 4.0 or later 

```
(torch-gpu) ➜  MUSIQ-Pytorch git:(main) ✗ pip3 show coremltools
Name: coremltools
Version: 6.3.0
Summary: Community Tools for Core ML
Home-page: https://github.com/apple/coremltools
Author: Apple Inc.
Author-email: coremltools@apple.com
License: BSD
Location: /Users/gavinxiang/miniconda3/envs/torch-gpu/lib/python3.8/site-packages
Requires: numpy, packaging, protobuf, sympy, tqdm
Required-by: 
```

## How to Seamlessly Convert Your PyTorch Model to Core ML

https://deci.ai/blog/pytorch-to-coreml/

If direct conversion from the PyTorch model to the Core ML model is not supported due to older platform deployment, you can first convert your PyTorch model to ONNX format and then convert it to Core ML. 

## Apple - coremltools API Reference

Converters / Unified (TensorFlow and Pytorch)

https://apple.github.io/coremltools/source/coremltools.converters.convert.html

Conversion Options / Image Input and Output

https://coremltools.readme.io/docs/image-inputs

**Preprocessing for Torch**

https://coremltools.readme.io/docs/image-inputs#preprocessing-for-torch

`output[channel] = (input[channel] - mean [channel]) / std [channel]`

For all pre-trained torchvision models, including MobileNetV2, the values are as follows:

mean is `[0.485, 0.456, 0.406]`.
std (standard deviation) is `[0.229, 0.224, 0.225]` .

```
import coremltools as ct
# Set the image scale and bias for input image preprocessing
scale = 1/(0.226*255.0)
bias = [- 0.485/(0.229) , - 0.456/(0.224), - 0.406/(0.225)]

image_input = ct.ImageType(name="input_1",
                           shape=example_input.shape,
                           scale=scale, bias=bias)
```

**Flexible Input Shapes**

https://coremltools.readme.io/docs/flexible-inputs

```
import coremltools as ct
import torch
import numpy as np

# Define a model for this example.
class TestConvModule(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=10, kernel_size=3):
        super(TestConvModule, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels,
                                    kernel_size)

    def forward(self, x):
        return self.conv(x)

# Trace the model with random input.
example_input = torch.rand(1, 3, 50, 50)
traced_model = torch.jit.trace(TestConvModule().eval(), example_input)

# Set the input_shape to use EnumeratedShapes.
input_shape = ct.EnumeratedShapes(shapes=[[1, 3, 25, 25],
                                          [1, 3, 50, 50],
                                          [1, 3, 67, 67]],
                                          default=[1, 3, 67, 67])

# Convert the model with input_shape.
model = ct.convert(traced_model,
                   inputs=[ct.TensorType(shape=input_shape, name="input")],
                   outputs=[ct.TensorType(name="output")],
                   convert_to="mlprogram",
                   )

model.save("enumerated_shapes.mlpackage")

# Test the model with predictions.
input_1 = np.random.rand(1, 3, 25, 25)
input_2 = np.random.rand(1, 3, 67, 67)

output_1 = model.predict({"input": input_1})["output"]
print("output shape {} for input shape {}".format(output_1.shape, input_1.shape))
output_2 = model.predict({"input": input_2})["output"]
print("output shape {} for input shape {}".format(output_2.shape, input_2.shape))
```

**Set feature descriptions manually**

https://apple.github.io/coremltools/source/coremltools.models.html

```
# Load the model
model = MLModel("HousePricer.mlmodel")

# Set the model metadata
model.author = "Author"
model.license = "BSD"
model.short_description = "Predicts the price of a house in the Seattle area."

# Get the interface to the model
model.input_description
model.output_description

# Set feature descriptions manually
model.input_description["bedroom"] = "Number of bedrooms"
model.input_description["bathrooms"] = "Number of bathrooms"
model.input_description["size"] = "Size (in square feet)"

# Set
model.output_description["price"] = "Price of the house"

# Make predictions
predictions = model.predict({"bedroom": 1.0, "bath": 1.0, "size": 1240})

# Get the spec of the model
spec = model.get_spec()

# Save the model
model.save("HousePricer.mlpackage")

# Load the model from the spec object
spec = model.get_spec()
# modify spec (e.g. rename inputs/ouputs etc)
model = MLModel(spec)
# if model type is mlprogram, i.e. spec.WhichOneof('Type') == "mlProgram", then:
model = MLModel(spec, weights_dir=model.weights_dir)
```


## PYTORCH MOBILE PERFORMANCE RECIPES

https://pytorch.org/tutorials/recipes/mobile_perf.html?highlight=mobile

1. Fuse operators using torch.quantization.fuse_modules

`torch.quantization.fuse_modules(model, [['conv', 'bn', 'relu']], inplace=True)`

2. Quantize your model

Quantization of the model not only moves computation to int8, but also reduces the size of your model on a disk.

That size reduction helps to reduce disk read operations during the first load of the model and decreases the amount of RAM.

**Both of those resources can be crucial for the performance of mobile applications.**

Introduction to Quantization on PyTorch

https://pytorch.org/blog/introduction-to-quantization-on-pytorch/

```
import torch.quantization
quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
```

```
model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
torch.quantization.prepare(model, inplace=True)
# Calibrate your model
def calibrate(model, calibration_data):
    # Your calibration code here
    return
calibrate(model, [])
torch.quantization.convert(model, inplace=True)
```


3. Use torch.utils.mobile_optimizer
```
torchscript_model_optimized = optimize_for_mobile(torchscript_model)
torch.jit.save(torchscript_model_optimized, "model.pt")
```

4. Prefer Using Channels Last Tensor memory format
```
def forward(self, x):
    x = x.contiguous(memory_format=torch.channels_last)
    ...
```

5. Android - Reusing tensors for forward

6. Load time optimization

`model._save_for_lite_interpreter('path/to/file.ptl', _use_flatbuffer=True)`

`2023-07-10 09:41:04.924323+0800 ScoreImage[2338:910900] Flatbuffer input file but the build hasn't enable flatbuffer`
