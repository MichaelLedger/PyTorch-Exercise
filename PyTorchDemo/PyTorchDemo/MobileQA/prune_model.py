import torch
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from models.MobileNet_IQA import MoNet

def apply_channel_pruning(model, amount=0.3):
    """Apply channel-wise pruning to convolutional layers"""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            # Get output channels importance
            out_channels = module.weight.shape[0]
            l1_norm = torch.sum(torch.abs(module.weight.data.reshape(out_channels, -1)), dim=1)
            num_keep = int(out_channels * (1 - amount))
            
            # Get indices of channels to keep
            _, indices = torch.topk(l1_norm, num_keep)
            mask = torch.zeros(out_channels)
            mask[indices] = 1
            
            # Apply mask
            module.weight.data = module.weight.data[mask == 1]
            if module.bias is not None:
                module.bias.data = module.bias.data[mask == 1]
            module.out_channels = num_keep

def apply_weight_compression(model):
    """Apply weight value compression"""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name:
                # Round small values to zero
                mask = torch.abs(param.data) < 0.01
                param.data[mask] = 0
                
                # Quantize remaining values to 16-bit
                param.data = param.data.half()

def prune_model():
    print('Loading model for pruning...')
    
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
    
    print('Applying aggressive pruning...')
    
    # 1. Apply global unstructured pruning (60% instead of 30%)
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    
    # Global pruning of 60% of connections
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.6,  # Remove 60% of connections
    )
    
    # Make pruning permanent
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.remove(module, 'weight')
    
    print('Applying channel pruning...')
    # 2. Apply channel pruning
    apply_channel_pruning(model, amount=0.4)  # Remove 40% of channels
    
    print('Applying weight compression...')
    # 3. Apply weight compression
    apply_weight_compression(model)
    
    # Save pruned and compressed model
    print('Saving optimized model...')
    torch.save(model.state_dict(), 'student_model_optimized.pkl')
    print('Optimized model saved successfully!')

if __name__ == '__main__':
    prune_model()