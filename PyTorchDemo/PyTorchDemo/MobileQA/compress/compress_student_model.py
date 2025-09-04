import sys
import os
import torch
import torchvision
import numpy as np
import coremltools as ct
from models.MobileNet_IQA import MoNet
from PIL import Image

def get_model_size(model_path):
    """Get the size of the model file in MB"""
    try:
        if os.path.isdir(model_path):
            total_size = 0
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)
            return total_size / (1024 * 1024)
        else:
            return os.path.getsize(model_path) / (1024 * 1024)
    except Exception as e:
        print(f"Error calculating size for {model_path}: {str(e)}")
        return 0

def prune_weights(model, threshold=0.1):
    """Prune model weights below threshold"""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name:
                mask = torch.abs(param.data) > threshold
                param.data *= mask.float()
    return model

def optimize_model(model):
    """Apply joint compression (pruning + quantization) to the model"""
    model.eval()
    
    # Register compression metadata version
    model.register_buffer("_COREML_/metadata_version", torch.tensor(1))
    
    # Apply joint compression to each layer with more aggressive parameters
    n_bits = 3  # Use 3-bit quantization for even more compression
    pruning_threshold = 0.15  # Increased pruning threshold
    
    with torch.no_grad():
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                # Get weight tensor
                weight_tensor = module.weight.data
                
                # Skip if tensor is empty
                if weight_tensor.numel() == 0:
                    continue
                
                # Adaptive pruning threshold based on weight statistics
                weight_abs = torch.abs(weight_tensor)
                mean_weight = torch.mean(weight_abs)
                std_weight = torch.std(weight_abs)
                adaptive_threshold = mean_weight + pruning_threshold * std_weight
                
                # Calculate per-channel scale for quantization
                weight_shape = weight_tensor.shape
                if len(weight_shape) >= 2:
                    # Reshape for per-channel quantization
                    weight_flat = weight_tensor.reshape(weight_shape[0], -1)
                    
                    # Calculate per-channel statistics
                    max_abs_val = torch.amax(torch.abs(weight_flat), dim=1)
                    min_abs_val = torch.amin(torch.abs(weight_flat), dim=1)
                    
                    # Optimize scale calculation
                    scale = (max_abs_val - min_abs_val) / (2**n_bits - 1)
                    scale = scale.reshape(-1, 1)  # Shape: [out_channels, 1]
                else:
                    # Handle 1D tensors with optimized range
                    max_abs_val = torch.max(torch.abs(weight_tensor))
                    min_abs_val = torch.min(torch.abs(weight_tensor))
                    scale = (max_abs_val - min_abs_val) / (2**n_bits - 1)
                    scale = scale.reshape(1, 1)
                
                # Ensure scale is not zero with smaller epsilon
                scale = torch.maximum(scale, torch.tensor(1e-10))
                
                # Register compression info
                module.register_buffer("_COREML_/weight/compression_type", torch.tensor([1, 3]))
                module.register_buffer("_COREML_/weight/quantization_n_bits", torch.tensor(n_bits))
                module.register_buffer("_COREML_/weight/quantization_scale", scale)
                
                # Apply adaptive pruning
                pruning_mask = (torch.abs(weight_tensor) > adaptive_threshold).float()
                weight_tensor_pruned = weight_tensor * pruning_mask
                
                # Quantize weights with improved precision
                if len(weight_shape) >= 2:
                    # Reshape for per-channel quantization
                    weight_flat = weight_tensor_pruned.reshape(weight_shape[0], -1)
                    # Center the weights around zero before quantization
                    weight_centered = weight_flat - torch.mean(weight_flat, dim=1, keepdim=True)
                    weight_q = torch.round(weight_centered / scale) * scale
                    weight_q = weight_q.reshape(weight_shape)
                else:
                    # Handle 1D tensors with centering
                    weight_centered = weight_tensor_pruned - torch.mean(weight_tensor_pruned)
                    weight_q = torch.round(weight_centered / scale[0,0]) * scale[0,0]
                
                # Update weights
                module.weight.data.copy_(weight_q)
                print(f"Applied joint compression to {name}")
    
    return model

def convert_student_model_to_coreml():
    print('Starting student model conversion to Core ML with aggressive compression...')
    
    # Create model
    model = MoNet(drop=0.1, dim_mlp=768, img_size=224)
    model = model.eval()
    
    try:
        # Load the model weights
        checkpoint = torch.load('student_model.pkl', map_location=torch.device('cpu'))
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
            
        print('Successfully loaded model weights')
        
        # Get original model size
        original_size = get_model_size('student_model.pkl')
        print(f'Original model size: {original_size:.2f} MB')
        
        # Apply weight pruning
        print('Applying weight pruning...')
        model = prune_weights(model, threshold=0.2)  # More aggressive pruning threshold
        
        # Apply model optimization
        print('Applying model optimization...')
        model = optimize_model(model)
        
    except Exception as e:
        print(f'Error in model preparation: {str(e)}')
        sys.exit(1)
    
    # Create example input
    example_input = torch.randn(1, 3, 224, 224)
    
    # Trace the model
    print('Tracing the model...')
    try:
        traced_model = torch.jit.trace(model, example_input)
        print('Successfully traced the model')
    except Exception as e:
        print(f'Error tracing model: {str(e)}')
        sys.exit(1)
    
    # Convert to Core ML with compression
    print('Converting to Core ML with compression...')
    try:
        # Define input tensor format
        image_input = ct.TensorType(
            name="input_image",
            shape=example_input.shape,
        )
        
        # Convert the model with aggressive compression options
        # Convert to Core ML with aggressive compression
        config = ct.ComputeUnit.CPU_AND_NE  # Use CPU and Neural Engine for better optimization
        
        # Convert with aggressive compression
        coreml_model = ct.convert(
            traced_model,
            inputs=[image_input],
            compute_units=config,
            minimum_deployment_target=ct.target.iOS15,  # Target older iOS for better compatibility
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16,  # Use FP16 for base precision
            skip_model_load=True  # Skip model reloading to preserve quantization
        )
        
        # Add metadata
        coreml_model.author = "PlanetArt: GavinXiang"
        coreml_model.license = "MIT License"
        coreml_model.short_description = "Student model for image quality assessment using MobileNet"
        coreml_model.version = "1.0.0"
        
        # Save intermediate model for size comparison
        torch.save(model.state_dict(), 'intermediate_model.pkl')
        intermediate_size = get_model_size('intermediate_model.pkl')
        
        # Save the final CoreML model
        print('Saving Core ML model...')
        output_path = "StudentModel.mlpackage"
        coreml_model.save(output_path)
        print('Successfully saved StudentModel.mlpackage')
        
        # Calculate detailed compression metrics
        compressed_size = get_model_size(output_path)
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
        size_reduction = (original_size - compressed_size) / original_size * 100 if original_size > 0 else 0
        
        # Calculate intermediate metrics
        pruning_reduction = (original_size - intermediate_size) / original_size * 100
        quantization_reduction = (intermediate_size - compressed_size) / intermediate_size * 100
        
        print('\nDetailed Compression Results:')
        print(f'Original model size: {original_size:.2f} MB')
        print(f'Size after pruning: {intermediate_size:.2f} MB (reduced by {pruning_reduction:.1f}%)')
        print(f'Final compressed size: {compressed_size:.2f} MB (reduced by {quantization_reduction:.1f}% from pruned model)')
        print(f'Overall compression ratio: {compression_ratio:.2f}x')
        print(f'Total size reduction: {size_reduction:.1f}%')
        print('\nCompression techniques applied:')
        print('1. Joint pruning + quantization:')
        print('   - Adaptive weight pruning (base threshold: 0.15)')
        print('   - 3-bit quantization with weight centering')
        print('   - Per-channel dynamic range optimization')
        print('2. FP16 compute precision')
        print('3. CPU and Neural Engine optimization')
        print('4. MLProgram format optimization')
        
        # Clean up intermediate file
        if os.path.exists('intermediate_model.pkl'):
            os.remove('intermediate_model.pkl')
        
    except Exception as e:
        print(f'Error converting to Core ML: {str(e)}')
        if os.path.exists('intermediate_model.pkl'):
            os.remove('intermediate_model.pkl')
        sys.exit(1)
    
    print('Conversion and compression completed successfully!')

if __name__ == "__main__":
    convert_student_model_to_coreml()