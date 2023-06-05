import sys, os

import torch
import torchvision

import numpy as np
import cv2
from tqdm import tqdm

from option.config import Config
from model.backbone import resnet50_backbone
from model.model_main import IQARegression

import torch.utils.mobile_optimizer as mobile_optimizer

import coremltools as ct

# configuration
config = Config({
    'gpu_id': 0,                                                        # specify gpu number to use
    'dirname': '/Users/gavinxiang/Downloads/PyTorch-Exercise/MUSIQ-Pytorch/sample/1024x768',     # directory of data root
    'checkpoint': './weights/epoch12.pth',                              # weights of trained model
    'result_score_txt': 'test_score.txt',                               # file for saving inference results
    'batch_size': 1,                                                    # fix the value as 1 (for inference)

    # ViT structure
    'n_enc_seq': 32*24 + 12*9 + 7*5,        # input feature map dimension (N = H*W) from backbone
    'n_layer': 14,                          # number of encoder layers
    'd_hidn': 384,                          # input channel of encoder (input: C x N)
    'i_pad': 0,
    'd_ff': 384,                            # feed forward hidden layer dimension
    'd_MLP_head': 1152,                     # hidden layer of final MLP
    'n_head': 6,                            # number of head (in multi-head attention)
    'd_head': 384,                          # channel of each head -> same as d_hidn
    'dropout': 0.1,                         # dropout ratio
    'emb_dropout': 0.1,                     # dropout ratio of input embedding
    'layer_norm_epsilon': 1e-12,
    'n_output': 1,                          # dimension of output
    'Grid': 10,                             # grid of 2D spatial embedding
    'scale_1': 384,                         # multi-scale
    'scale_2': 224,                         # multi-scale
})


# device setting
config.device = torch.device('cuda:%s' % config.gpu_id if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('Using GPU %s' % config.gpu_id)
else:
    print('Using CPU')


# input normalize
class Normalize(object):
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var
    def __call__(self, sample):
        sample[:, :, 0] = (sample[:, :, 0] - self.mean[0]) / self.var[0]
        sample[:, :, 1] = (sample[:, :, 1] - self.mean[1]) / self.var[1]
        sample[:, :, 2] = (sample[:, :, 2] - self.mean[2]) / self.var[2]
        return sample

# numpy array -> torch tensor
class ToTensor(object):
    def __call__(self, sample):
        sample = np.transpose(sample, (2, 0, 1))
        sample = torch.from_numpy(sample)
        return sample


# create model
model_backbone = resnet50_backbone().to(config.device)
model_transformer = IQARegression(config).to(config.device)

# load weights
checkpoint = torch.load(config.checkpoint)
model_backbone.load_state_dict(checkpoint['model_backbone_state_dict'])
model_transformer.load_state_dict(checkpoint['model_transformer_state_dict'])
model_backbone.eval()
model_transformer.eval()

# input transform
transforms = torchvision.transforms.Compose([Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ToTensor()])

# An example input you would normally provide to your model's forward() method.
example_input = torch.rand(1, 3, 224, 224)
#example_input = torch.tensor((), dtype=torch.float64)
#example_input = torch.rand(1)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model_backbone, example_input)

# Save to file
#torch.jit.save(traced_script_module, 'resnet50_script_module.pt')
# This line is equivalent to the previous
#traced_script_module.save("scriptmodule.pt")

# model optimization (optional)
#opt_model = mobile_optimizer.optimize_for_mobile(traced_script_module)

# save optimized model for mobile
#opt_model._save_for_lite_interpreter("resnet50_mobile_model.ptl")
#traced_script_module._save_for_lite_interpreter("resnet50_mobile_model.ptl")

# Define the set of image sizes (1, channel, height, width)
#input_names = ['feat_dis_org', 'feat_dis_scale_1', 'feat_dis_scale_2']
#input_sizes = [(1, 3, 768, 1024), (1, 3, 288, 384), (1, 3, 160, 224)]
#input_descs = ['color (kCVPixelFormatType_32BGRA) image buffer, 1024 pixels wide by 768 pixels high',
#               'color (kCVPixelFormatType_32BGRA) image buffer, 384 pixels wide by 288 pixels high',
#               'color (kCVPixelFormatType_32BGRA) image buffer, 224 pixels wide by 160 pixels high']

# Set the image scale and bias for input image preprocessing
image_scale = 1/(0.226*255.0)
image_bias = [- 0.485/(0.229) , - 0.456/(0.224), - 0.406/(0.225)]

# Set the input_shape to use EnumeratedShapes.
input_shape = ct.EnumeratedShapes(shapes=[[1, 3, 768, 1024],
                                          [1, 3, 288, 384],
                                          [1, 3, 160, 224]],
                                          default=[1, 3, 768, 1024])

image_input = ct.ImageType(name="input_image",
                           shape=input_shape,
                           scale=image_scale, bias=image_bias)

# Create a list of ImageType objects
#input_types = [ct.ImageType(shape=input_size) for input_size in input_sizes]
#input_types = [ct.ImageType(name=input_names[0], shape=input_sizes[0]),
#               ct.ImageType(name=input_names[1], shape=input_sizes[1]),
#               ct.ImageType(name=input_names[2], shape=input_sizes[2])]
#print('input_types: %s' % input_types)

# Convert the PyTorch model to Core ML with input descriptions
# As an alternative, you can convert the model to a neural network by eliminating the convert_to parameter:
# if we define input as image type, it always failed with the following error message:
#2023-06-02 15:47:44.435208+0800 ScoreImage[17905:1504044] [espresso] [Espresso::handle_ex_plan] exception=Espresso exception: "Invalid state": reshape mismatching size: 2147483647 1 1 1 1 -> 32 24 384 1 1 status=-5
#2023-06-02 15:47:44.435336+0800 ScoreImage[17905:1504044] [coreml] Error computing NN outputs -5
core_ml_neural_network_model = ct.convert(
                                          traced_script_module,
                                          #inputs=[ct.TensorType(shape=example_input.shape)]
                                          inputs=[ct.TensorType(name="input_image", shape=input_shape)]
                                          #inputs=[ct.ImageType(shape=input_shape, name="input_image")]
#                                          inputs=[image_input]
                                          #input_names=input_names,
                                          #input_description=input_descs
                                          )
# Set the metadata properties
core_ml_neural_network_model.short_description = "ResNet-50 from Deep Residual Learning for Image Recognition (paper link: https://arxiv.org/abs/1512.03385)."
core_ml_neural_network_model.author = "PlanetArt: GavinXiang"
core_ml_neural_network_model.license = "MIT License."
core_ml_neural_network_model.version = "1.0.0"  # You can set the version number as a string

# Set feature descriptions manually
core_ml_neural_network_model.input_description["input_image"] = "input imgage as color (kCVPixelFormatType_32BGRA) image buffer, 1024 pixels wide by 768 pixels high, 384 pixels wide by 288 pixels high, 224 pixels wide by 160 pixels high."

print('==start to save core ML neural network model for resnet50!==')
# Save the Core ML model to a file with the updated metadata
core_ml_neural_network_model.save("resnet50_ML_Neural_Network.mlmodel")
print('==success saved core ML neural network model for resnet50!==')

# Exit
sys.exit(os.EX_OK)

print('should not run to here!')

# save results
pred_total = []

filenames = os.listdir(config.dirname)
filenames.sort()
f = open(config.result_score_txt, 'w')

# input mask (batch_size x len_sqe+1)
mask_inputs = torch.ones(config.batch_size, config.n_enc_seq+1).to(config.device)

# inference
for filename in tqdm(filenames):
    d_img_name = os.path.join(config.dirname, filename)
    ext = os.path.splitext(d_img_name)[-1]
    if ext == '.jpg':
        # multi-scale feature extraction
        d_img_org = cv2.imread(d_img_name)
        d_img_org = cv2.cvtColor(d_img_org, cv2.COLOR_BGR2RGB)
        d_img_org = np.array(d_img_org).astype('float32') / 255

        h, w, c = d_img_org.shape
        d_img_scale_1 = cv2.resize(d_img_org, dsize=(config.scale_1, int(h*(config.scale_1/w))), interpolation=cv2.INTER_CUBIC)
        d_img_scale_2 = cv2.resize(d_img_org, dsize=(config.scale_2, int(h*(config.scale_2/w))), interpolation=cv2.INTER_CUBIC)
        d_img_scale_2 = d_img_scale_2[:160, :, :]

        d_img_org = transforms(d_img_org)
        d_img_org = torch.tensor(d_img_org.to(config.device)).unsqueeze(0)
        d_img_scale_1 = transforms(d_img_scale_1)
        d_img_scale_1 = torch.tensor(d_img_scale_1.to(config.device)).unsqueeze(0)
        d_img_scale_2 = transforms(d_img_scale_2)
        d_img_scale_2 = torch.tensor(d_img_scale_2.to(config.device)).unsqueeze(0)

        feat_dis_org = model_backbone(d_img_org)
        feat_dis_scale_1 = model_backbone(d_img_scale_1)
        feat_dis_scale_2 = model_backbone(d_img_scale_2)

        # quality prediction
        pred = model_transformer(mask_inputs, feat_dis_org, feat_dis_scale_1, feat_dis_scale_2)
        pred_total = np.append(pred_total, float(pred.item()))

        # result save
        line = '%s\t%f\n' % (filename, float(pred.item()))
        f.write(line)
f.close()
        






