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
```
## CPU & GPU training log 2023/4/25 13:17
```
(torch-gpu) macstudio@MacStudios-Mac-Studio MUSIQ-Pytorch % python3 train.py
will save weights of epoch 105 to path: ./weights/epoch105.pth
Using GPU 1
number of train scenes: 8058
number of test scenes: 2015
  0%|                                                                                                             | 0/1007 [00:00<?, ?it/s]will save weights of epoch 105 to path: ./weights/epoch105.pth
Using GPU 1
number of train scenes: 8058
number of test scenes: 2015
will save weights of epoch 105 to path: ./weights/epoch105.pth
Using GPU 1
number of train scenes: 8058
number of test scenes: 2015
will save weights of epoch 105 to path: ./weights/epoch105.pth
Using GPU 1
number of train scenes: 8058
number of test scenes: 2015
will save weights of epoch 105 to path: ./weights/epoch105.pth
Using GPU 1
number of train scenes: 8058
number of test scenes: 2015
will save weights of epoch 105 to path: ./weights/epoch105.pth
Using GPU 1
number of train scenes: 8058
number of test scenes: 2015
will save weights of epoch 105 to path: ./weights/epoch105.pth
Using GPU 1
number of train scenes: 8058
number of test scenes: 2015
will save weights of epoch 105 to path: ./weights/epoch105.pth
Using GPU 1
number of train scenes: 8058
number of test scenes: 2015
will save weights of epoch 105 to path: ./weights/epoch105.pth
Using GPU 1
number of train scenes: 8058
number of test scenes: 2015
/Users/macstudio/miniconda3/envs/torch-gpu/lib/python3.8/site-packages/torch/autograd/__init__.py:204: UserWarning: The operator 'aten::sgn.out' is not currently supported on the MPS backend and will fall back to run on the CPU. This may have performance implications. (Triggered internally at /Users/runner/work/_temp/anaconda/conda-bld/pytorch_1682147327173/work/aten/src/ATen/mps/MPSFallback.mm:12.)
  Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
  6%|██████▎                                                                                          | 65/1007 [53:12<14:35:01, 55.73s/it]
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
