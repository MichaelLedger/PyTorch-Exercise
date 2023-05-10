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
