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

print('torch.__version__ %s' % torch.__version__)

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
        
        # An example input you would normally provide to your model's forward() method.
        #example = torch.rand(1, 3, 224, 224)
        #example = torch.tensor((), dtype=torch.float64)
        #example = torch.rand(1)
        example = (mask_inputs, feat_dis_org, feat_dis_scale_1, feat_dis_scale_2)

        # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
        traced_script_module = torch.jit.trace(model_transformer, example)
        
        # Fuse operators using torch.quantization.fuse_modules
        # The inplace=True argument tells PyTorch to modify the model in place, rather than returning a new model. If you don't want to modify the original model, you can set inplace=False and assign the result to a new variable.
        # AttributeError: 'RecursiveScriptModule' object has no attribute 'conv'
        #torch.quantization.fuse_modules(traced_script_module, [['conv', 'bn', 'relu']], inplace=True)
        
        # Quantization of the model not only moves computation to int8, but also reduces the size of your model on a disk.
        quantized_model = torch.quantization.quantize_dynamic(traced_script_module, {torch.nn.Linear}, dtype=torch.qint8)

        # Save to file
        torch.jit.save(quantized_model, 'IQA_script_module.pt')
        # This line is equivalent to the previous
        #traced_script_module.save("scriptmodule.pt")
        
        # model optimization (optional)
        opt_model = mobile_optimizer.optimize_for_mobile(quantized_model)
        
        # save optimized model for mobile
        # The extra argument _use_flatbuffer makes a FlatBuffer file instead of a zip file. The created file will be faster to load.
        opt_model._save_for_lite_interpreter("IQA_mobile_model.ptl", _use_flatbuffer=True)
        #traced_script_module._save_for_lite_interpreter("IQA_mobile_model.ptl")

        # Exit
        sys.exit(os.EX_OK)

        print('should not run to here!')
        
        pred_total = np.append(pred_total, float(pred.item()))

        # result save
        line = '%s\t%f\n' % (filename, float(pred.item()))
        f.write(line)
f.close()
        






