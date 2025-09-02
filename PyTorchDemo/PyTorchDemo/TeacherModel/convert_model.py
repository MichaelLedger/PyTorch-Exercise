import torch
import sys
import os
import torch.utils.mobile_optimizer as mobile_optimizer
import torch.nn as nn
import timm
from einops import rearrange

class Attention_Block(nn.Module):
    def __init__(self, dim, drop=0.1):
        super().__init__()
        self.c_q = nn.Linear(dim, dim)
        self.c_k = nn.Linear(dim, dim)
        self.c_v = nn.Linear(dim, dim)
        self.norm_fact = dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x):
        _x = x
        B, C, N = x.shape
        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)

        attn = q @ k.transpose(-2, -1) * self.norm_fact
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, C, N)
        x = self.proj_drop(x)
        x = x + _x
        return x

class Self_Attention(nn.Module):
    def __init__(self, in_dim):
        super(Self_Attention, self).__init__()
        self.qConv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.kConv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.vConv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inFeature):
        bs, C, w, h = inFeature.size()
        proj_query = self.qConv(inFeature).view(bs, -1, w * h).permute(0, 2, 1)
        proj_key = self.kConv(inFeature).view(bs, -1, w * h)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.vConv(inFeature).view(bs, -1, w * h)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(bs, C, w, h)
        out = self.gamma * out + inFeature
        return out

class MAL(nn.Module):
    def __init__(self, in_dim=768, feature_num=4, feature_size=28):
        super().__init__()
        self.channel_attention = Attention_Block(in_dim * feature_num)
        self.feature_attention = Attention_Block(feature_size ** 2 * feature_num)
        self.attention_module = nn.ModuleList()
        for _ in range(feature_num):
            self.attention_module.append(Self_Attention(in_dim))
        self.feature_num = feature_num
        self.in_dim = in_dim

    def forward(self, features):
        device = features[0].device if len(features) > 0 else torch.device('cpu')
        feature = torch.tensor([]).to(device)
        for index, _ in enumerate(features):
            feature = torch.cat((feature, self.attention_module[index](features[index]).unsqueeze(0)), dim=0)
        features = feature
        input_tensor = rearrange(features, 'n b c w h -> b (n c) (w h)')
        bs, _, _ = input_tensor.shape
        in_feature = rearrange(input_tensor, 'b (w c) h -> b w (c h)', w=self.in_dim, c=self.feature_num)
        feature_weight_sum = self.feature_attention(in_feature)
        in_channel = input_tensor.permute(0, 2, 1)
        channel_weight_sum = self.channel_attention(in_channel)
        weight_sum_res = (rearrange(feature_weight_sum, 'b w (c h) -> b (w c) h', w=self.in_dim,
                                    c=self.feature_num) + channel_weight_sum.permute(0, 2, 1)) / 2
        weight_sum_res = torch.mean(weight_sum_res.view(bs, self.feature_num, self.in_dim, -1), dim=1)
        return weight_sum_res

class Local_Distortion_Aware(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.cnn1 = nn.Conv2d(in_dim, out_dim * 2, 1, 1)
        self.grelu = nn.GELU()
        self.avg2 = nn.AvgPool2d(4, 4)
        self.cnn2 = nn.Conv2d(out_dim * 2, out_dim, 1, 1)
        self.avg = nn.AdaptiveAvgPool2d((22, 22))

    def forward(self, features):
        local_1 = self.avg(self.grelu(self.cnn1(features)))
        local_2 = self.cnn2(local_1)
        return local_2.unsqueeze(1)

class MobileIQA(nn.Module):
    def __init__(self, drop=0.1, dim_mlp=768):
        super().__init__()
        self.input_size = 16
        self.dim_mlp = dim_mlp

        out_indices = [0, 1, 2, 3, 4]
        self.global_vit = timm.create_model('mobilevitv2_200', features_only=True, out_indices=out_indices, pretrained=True)
        
        self.LGF1 = Local_Distortion_Aware(128, 256)
        self.LGF2 = Local_Distortion_Aware(256, 256)
        self.LGF3 = Local_Distortion_Aware(512, 256)
        self.LGF4 = Local_Distortion_Aware(768, 256)
        self.LGF5 = Local_Distortion_Aware(1024, 256)
        
        self.MALs = nn.ModuleList()
        for _ in range(3):
            self.MALs.append(MAL(256, feature_num=5, feature_size=22))

        self.fusion_mal = MAL(256, feature_num=3, feature_size=22)
        self.cnn = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((4, 4)),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc_score = nn.Sequential(
            nn.Linear(128, 128 // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(128 // 2, 1),
            nn.Sigmoid()
        )
        self.input_size = 22

    def forward(self, full_img):
        global_features = self.global_vit(full_img)
        global_feature_list = None
        for idx, _ in enumerate(global_features):
            if global_feature_list is None:
                global_feature_list = getattr(self, f'LGF{idx + 1}')(global_features[idx])
            else:
                global_feature_list = torch.cat((global_feature_list, getattr(self, f'LGF{idx + 1}')(global_features[idx])), dim=1)

        x = global_feature_list
        x = x.permute(1, 0, 2, 3, 4)

        device = x.device
        DOF = torch.tensor([]).to(device)
        for index, _ in enumerate(self.MALs):
            DOF = torch.cat((DOF, self.MALs[index](x).unsqueeze(0)), dim=0)
        DOF = rearrange(DOF, 'n c d (w h) -> n c d w h', w=self.input_size, h=self.input_size)

        fusion_mal = self.fusion_mal(DOF).permute(0, 2, 1)
        IQ_feature = fusion_mal.permute(0, 2, 1)
        IQ_feature = rearrange(IQ_feature, 'c d (w h) -> c d w h', w=self.input_size, h=self.input_size)
        score = self.cnn(IQ_feature).squeeze(-1).squeeze(-1)
        score = self.fc_score(score).view(-1)
        
        return score

def convert_to_torchscript():
    print("Loading model...")
    # Load the model state dictionary
    state_dict = torch.load('teacher_model.pkl', map_location=torch.device('cpu'))
    
    # Create model with mobile architecture
    model = MobileIQA()
    
    # Map the state dict keys
    mapped_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('global_vit.'):
            mapped_state_dict[key] = value
        elif key.startswith('LGF'):
            # LGF1, LGF2, etc. are already correctly named
            mapped_state_dict[key] = value
        elif key.startswith('MALs.'):
            # MALs are already correctly named
            mapped_state_dict[key] = value
        elif key.startswith('fusion_mal.'):
            # fusion_mal is already correctly named
            mapped_state_dict[key] = value
        elif key.startswith('cnn.'):
            mapped_state_dict[key] = value
        elif key.startswith('fc_score.'):
            mapped_state_dict[key] = value
    
    # Load the state dict
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    print("Creating example inputs...")
    # Create example input matching the size from the log
    example_input = torch.randn(1, 3, 1188, 1914)  # Size from the log
    
    print("Converting to TorchScript...")
    # Convert to TorchScript via tracing
    traced_script_module = torch.jit.trace(model, example_input)
    
    print("Optimizing for mobile...")
    # Optimize the model for mobile
    traced_script_module_optimized = mobile_optimizer.optimize_for_mobile(
        traced_script_module,
        backend='CPU',  # Use CPU backend for iOS
        optimization_blocklist=None  # Apply all optimizations
    )
    
    print("Saving optimized model...")
    # Save the optimized TorchScript model with .ptl extension
    traced_script_module_optimized.save("teacher_model.ptl")
    print("Model converted and optimized successfully!")

if __name__ == '__main__':
    convert_to_torchscript()
