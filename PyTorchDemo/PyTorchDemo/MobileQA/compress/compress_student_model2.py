import sys
import os
import torch
import torchvision
import numpy as np
import coremltools as ct
from models.MobileNet_IQA import MoNet
from PIL import Image

def compress_pytorch_model(model):
    """Compress PyTorch model weights using pruning and quantization"""
    print('Applying compression to PyTorch model...')
    
    def quantize_tensor(x, n_bits=2):  # Ultra-aggressive 2-bit quantization
        """Quantize tensor to 2-bits with extreme outlier handling"""
        if x.numel() == 0:
            return x
            
        # Calculate dynamic range per channel with aggressive outlier removal
        if len(x.shape) >= 2:
            # For 2D+ tensors, quantize per output channel
            orig_shape = x.shape
            x = x.reshape(x.shape[0], -1)
            
            # Calculate more aggressive percentile-based range
            q_low = torch.quantile(x, 0.05, dim=1, keepdim=True)   # 5th percentile
            q_high = torch.quantile(x, 0.95, dim=1, keepdim=True)  # 95th percentile
            
            # Enhanced outlier handling
            iqr = q_high - q_low
            lower_bound = q_low - 1.5 * iqr
            upper_bound = q_high + 1.5 * iqr
            
            # Clip values with enhanced bounds
            x = torch.clamp(x, lower_bound, upper_bound)
            
            # Calculate optimal range for quantization
            sorted_vals, _ = torch.sort(x, dim=1)
            n_elements = x.shape[1]
            
            # Use middle 90% of values for range calculation
            start_idx = int(0.05 * n_elements)
            end_idx = int(0.95 * n_elements)
            min_val = sorted_vals[:, start_idx:start_idx+1]
            max_val = sorted_vals[:, end_idx-1:end_idx]
        else:
            # For 1D tensors, use aggressive outlier removal
            q_low = torch.quantile(x, 0.05)
            q_high = torch.quantile(x, 0.95)
            iqr = q_high - q_low
            lower_bound = q_low - 1.5 * iqr
            upper_bound = q_high + 1.5 * iqr
            x = torch.clamp(x, lower_bound, upper_bound)
            
            sorted_vals, _ = torch.sort(x)
            n_elements = x.numel()
            start_idx = int(0.05 * n_elements)
            end_idx = int(0.95 * n_elements)
            min_val = sorted_vals[start_idx]
            max_val = sorted_vals[end_idx-1]
            
        # Calculate optimized scale and zero point
        scale = (max_val - min_val) / (2**n_bits - 1)
        scale = torch.maximum(scale, torch.tensor(1e-12, device=scale.device))  # Even smaller epsilon
        zero_point = min_val
        
        # Quantize with improved rounding
        x_normalized = (x - zero_point) / scale
        x_quantized = torch.round(x_normalized)
        x_dequantized = x_quantized * scale + zero_point
        
        # Reshape back if needed
        if len(x.shape) >= 2:
            x_dequantized = x_dequantized.reshape(orig_shape)
            
        return x_dequantized
    
    def prune_tensor(x, threshold_factor=0.5):  # Ultra-aggressive threshold
        """Prune small weights using ultra-aggressive adaptive threshold"""
        if x.numel() == 0:
            return x
            
        # Calculate per-channel statistics for ultra-precise pruning
        if len(x.shape) >= 2:
            # For 2D+ tensors, calculate per-channel statistics
            x_reshaped = x.reshape(x.shape[0], -1)
            abs_weights = torch.abs(x_reshaped)
            
            # Calculate enhanced per-channel statistics
            median = torch.median(abs_weights, dim=1, keepdim=True)[0]
            q75 = torch.quantile(abs_weights, 0.75, dim=1, keepdim=True)
            q25 = torch.quantile(abs_weights, 0.25, dim=1, keepdim=True)
            iqr = q75 - q25
            
            # Calculate ultra-dynamic threshold per channel
            threshold = median + threshold_factor * iqr
            
            # Create and apply mask with importance weighting
            importance = abs_weights / torch.max(abs_weights, dim=1, keepdim=True)[0]
            mask = ((abs_weights > threshold) & (importance > 0.1)).float()
            
            # Ensure at least 5% of most important weights per channel are retained
            min_weights = max(int(0.05 * x_reshaped.shape[1]), 2)  # At least 2 weights
            for i in range(x_reshaped.shape[0]):
                if torch.sum(mask[i]) < min_weights:
                    # Keep top 5% weights by importance
                    importance_scores = abs_weights[i] * (1 + importance[i])  # Weighted importance
                    _, top_indices = torch.topk(importance_scores, min_weights)
                    mask[i] = 0
                    mask[i, top_indices] = 1
                    
            # Reshape mask back to original shape
            mask = mask.reshape(x.shape)
        else:
            # For 1D tensors, use enhanced statistics
            abs_weights = torch.abs(x)
            median = torch.median(abs_weights)
            q75 = torch.quantile(abs_weights, 0.75)
            q25 = torch.quantile(abs_weights, 0.25)
            iqr = q75 - q25
            
            # Calculate threshold using robust statistics
            threshold = median + threshold_factor * iqr
            
            # Create and apply mask with importance
            importance = abs_weights / torch.max(abs_weights)
            mask = ((abs_weights > threshold) & (importance > 0.1)).float()
            
            # Ensure at least 5% of most important weights are retained
            min_weights = max(int(0.05 * x.numel()), 2)  # At least 2 weights
            if torch.sum(mask) < min_weights:
                importance_scores = abs_weights * (1 + importance)
                _, top_indices = torch.topk(importance_scores.flatten(), min_weights)
                mask = torch.zeros_like(x)
                mask.flatten()[top_indices] = 1
        
        return x * mask
    
    # Apply compression layer by layer
    print("Compressing model weights...")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name:
                # First prune
                param.data = prune_tensor(param.data)
                # Then quantize
                param.data = quantize_tensor(param.data)
                print(f"Compressed {name}")
    
    return model

def load_compressed_model(model, compressed_path):
    """Load model from compressed format"""
    print(f'Loading compressed model from {compressed_path}...')
    compressed_data = np.load(compressed_path, allow_pickle=True)
    compressed_state = compressed_data['compressed_state'].item()
    
    # Create new state dict
    state_dict = {}
    for name, compressed_param in compressed_state.items():
        if compressed_param['dtype'] == 'int8':
            # Dequantize int8 weights
            data = compressed_param['data'].astype(np.float32)
            scale = compressed_param['scale']
            param_tensor = torch.from_numpy(data * scale)
        elif compressed_param['dtype'] == 'float16':
            # Convert float16 back to float32
            param_tensor = torch.from_numpy(compressed_param['data'].astype(np.float32))
        else:
            # Raw data
            param_tensor = torch.from_numpy(compressed_param['data'])
        state_dict[name] = param_tensor
    
    # Load state dict
    model.load_state_dict(state_dict)
    return model

def convert_student_model_to_coreml():
    print('Starting model compression and conversion...')
    
    # Create model
    model = MoNet(drop=0.1, dim_mlp=768, img_size=224)
    model = model.eval()
    
    try:
        # First compress the original model
        checkpoint = torch.load('student_model.pkl', map_location=torch.device('cpu'))
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
            
        print('Successfully loaded model weights')
        
        # Compress the PyTorch model
        model = compress_pytorch_model(model)
        
        # Save compressed model with optimized storage
        print('Saving compressed model...')
        
        # Extract and compress state dict
        compressed_state = {}
        for name, param in model.state_dict().items():
            # Convert to numpy for better compression
            param_np = param.cpu().numpy()
            
            if 'weight' in name:
                # For weight tensors, store as int8 with scale
                param_abs_max = np.abs(param_np).max()
                if param_abs_max > 0:
                    scale = param_abs_max / 127.0  # Scale to int8 range
                    param_quantized = np.clip(param_np / scale, -127, 127).astype(np.int8)
                    compressed_state[name] = {
                        'data': param_quantized,
                        'scale': scale,
                        'dtype': 'int8'
                    }
                else:
                    compressed_state[name] = {
                        'data': param_np,
                        'dtype': 'raw'
                    }
            else:
                # For non-weight tensors (biases, etc.), store as float16
                compressed_state[name] = {
                    'data': param_np.astype(np.float16),
                    'dtype': 'float16'
                }
        
        # Save with highest compression
        np.savez_compressed('student_model_compressed.npz', compressed_state=compressed_state)
        print('Saved compressed model in optimized format')
    except Exception as e:
        print(f'Error loading model weights: {str(e)}')
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
    
    # Convert to Core ML
    print('Converting to Core ML with aggressive optimization...')
    try:
        # Define input tensor format with reduced precision
        image_input = ct.TensorType(
            name="input_image",
            shape=example_input.shape,
            dtype=np.float16  # Use float16 for input
        )
        
        # Convert with aggressive optimization
        coreml_model = ct.convert(
            traced_model,
            inputs=[image_input],
            compute_units=ct.ComputeUnit.CPU_AND_NE,  # Use CPU and Neural Engine for better optimization
            minimum_deployment_target=ct.target.iOS16,  # Target iOS16 for better compression
            convert_to="mlprogram",  # Use mlprogram format
            compute_precision=ct.precision.FLOAT16,  # Use FP16 for computation
            skip_model_load=True  # Skip model reloading
        )
        
        # Post-conversion optimization
        spec = coreml_model.get_spec()
        
        # Optimize model architecture
        for layer in spec.neuralNetwork.layers:
            # Fuse batch normalization layers where possible
            if layer.HasField('batchnorm'):
                layer.batchnorm.computeMeanVar = False
                layer.batchnorm.instanceNormalization = True
            
            # Optimize convolution layers
            elif layer.HasField('convolution'):
                layer.convolution.isDeconvolution = False
                if layer.convolution.HasField('same'):
                    layer.convolution.valid.MergeFromString(layer.convolution.same.SerializeToString())
                    layer.convolution.ClearField('same')
        
        # Add metadata
        coreml_model.author = "PlanetArt: GavinXiang"
        coreml_model.license = "MIT License"
        coreml_model.short_description = "Student model for image quality assessment using MobileNet"
        coreml_model.version = "1.0.0"
        
        # Save with optimized storage
        print('Saving Core ML model with optimized storage...')
        
        # Get model spec for optimization
        spec = coreml_model.get_spec()
        
        # Optimize network architecture
        if spec.HasField('neuralNetwork'):
            nn = spec.neuralNetwork
            
            # Optimize each layer
            for layer in nn.layers:
                # Set compute precision to float16 for all layers
                if hasattr(layer, 'input'):
                    for input_tensor in layer.input:
                        input_tensor.type.float16 = True
                if hasattr(layer, 'output'):
                    for output_tensor in layer.output:
                        output_tensor.type.float16 = True
                        
                # Optimize convolution layers
                if layer.HasField('convolution'):
                    layer.convolution.outputShape = []  # Let runtime compute output shape
                    
                # Optimize batch norm layers
                if layer.HasField('batchnorm'):
                    layer.batchnorm.computeMeanVar = False
                    
                # Optimize activation layers
                if layer.HasField('activation'):
                    layer.activation.linear.alpha = 1.0
                    layer.activation.linear.beta = 0.0
        
        # Save optimized model
        final_path = "StudentModel.mlpackage"
        coreml_model.save(final_path)
            
        print('Successfully saved optimized CoreML model')
        
    except Exception as e:
        print(f'Error converting to Core ML: {str(e)}')
        sys.exit(1)
    
    print('Conversion completed successfully!')

if __name__ == "__main__":
    convert_student_model_to_coreml()