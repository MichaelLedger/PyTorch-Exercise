import torch
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from models.MobileNet_IQA import MoNet
import numpy as np

def apply_weight_compression(model):
    """Apply weight value compression"""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name:
                # Round small values to zero
                mask = torch.abs(param.data) < 0.01
                param.data[mask] = 0
                
                # Quantize remaining values to 16-bit
                param.data = param.data.half().float()

def compress_model():
    print('Loading model for compression...')
    
    # Create and load model
    model = MoNet(drop=0.1, dim_mlp=768, img_size=224)
    checkpoint = torch.load('student_model.pkl', map_location=torch.device('cpu'))
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    print('Applying weight pruning...')
    
    # Apply unstructured pruning to all layers
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    
    # Global pruning of 90% of connections
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.9,  # Remove 90% of connections
    )
    
    # Make pruning permanent
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.remove(module, 'weight')
    
    print('Applying weight compression...')
    # Apply weight compression
    apply_weight_compression(model)
    
    # Save compressed model
    print('Saving compressed model...')
    torch.save(model.state_dict(), 'student_model_compressed_v2.pkl')
    print('Compressed model saved successfully!')

if __name__ == '__main__':
    compress_model()