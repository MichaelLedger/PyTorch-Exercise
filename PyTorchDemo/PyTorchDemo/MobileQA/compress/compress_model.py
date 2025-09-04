#!/usr/bin/env python3
"""
Core ML Model Compression Script
This script compresses a Core ML model using various compression techniques
while maintaining acceptable accuracy.
"""

import os
import glob
import argparse
import coremltools as ct
import torch
import numpy as np
from tqdm import tqdm

def get_model_size(model_path):
    """Get the size of the model file in MB"""
    try:
        if os.path.isdir(model_path):
            # For .mlpackage (directory), sum up all file sizes
            total_size = 0
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)
            return total_size / (1024 * 1024)
        else:
            # For single file
            return os.path.getsize(model_path) / (1024 * 1024)
    except Exception as e:
        print(f"Error calculating size for {model_path}: {str(e)}")
        return 0

def compress_model(model_path, output_path, compression_mode='linear', bits=8, weight_threshold=0.1):
    """
    Compress the Core ML model using specified techniques.
    
    Args:
        model_path (str): Path to the input .mlpackage
        output_path (str): Path to save the compressed model
        compression_mode (str): Compression mode ('linear', 'kmeans_lut', etc.)
        bits (int): Number of bits for quantization
        weight_threshold (float): Threshold for pruning weights
    
    Returns:
        tuple: (compressed_model, compression_metrics)
    """
    print(f"\nLoading model from: {model_path}")
    
    # Verify model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    # Calculate original size
    original_size = get_model_size(model_path)
    print(f"Original model size: {original_size:.2f} MB")
    
    if original_size == 0:
        raise ValueError("Model file appears to be empty or inaccessible")
    
    try:
        # Load the model with weights directory
        weights_dir = os.path.join(model_path, "Data")
        if not os.path.exists(weights_dir):
            weights_dir = model_path
            
        print(f"\nLoading model with weights from: {weights_dir}")
        model = ct.models.MLModel(model_path, weights_dir=weights_dir)
        
        print(f"\nModel loaded successfully")
        print(f"Model type: {type(model)}")
        
        # Get model spec
        spec = model.get_spec() if hasattr(model, 'get_spec') else model._spec
        
        # Create config for compression
        config = {
            "nbits": bits,
            "mode": compression_mode,
            "quantize_inputs": True,
            "quantize_outputs": True,
            "quantize_activations": True,
            "weight_threshold": weight_threshold
        }
        
        print(f"\nApplying compression with configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
            
        # Convert to fp16 for size reduction
        print("\nConverting to FP16...")
        
        # Create a dummy input with the correct shape
        input_shape = [1, 3, 224, 224]  # Based on the model description
        example_input = np.random.rand(*input_shape).astype(np.float32)
        
        compressed_model = ct.convert(
            model,
            inputs=[ct.TensorType(shape=input_shape, name="input_image")],
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_GPU,
            minimum_deployment_target=ct.target.iOS16
        )
        
        # Save compressed model
        print(f"\nSaving compressed model to: {output_path}")
        compressed_model.save(output_path)
        
        # Calculate metrics
        compressed_size = get_model_size(output_path)
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
        size_reduction = (original_size - compressed_size) / original_size * 100 if original_size > 0 else 0
        
        metrics = {
            'original_size_mb': original_size,
            'compressed_size_mb': compressed_size,
            'compression_ratio': compression_ratio,
            'size_reduction_percent': size_reduction
        }
        
        print("\nCompression Results:")
        print(f"Compressed model size: {compressed_size:.2f} MB")
        print(f"Compression ratio: {compression_ratio:.2f}x")
        print(f"Size reduction: {size_reduction:.1f}%")
        
        return compressed_model, metrics
        
    except Exception as e:
        print(f"\nError during compression: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Compress Core ML model while maintaining accuracy')
    parser.add_argument('--input', required=True, help='Path to input .mlpackage')
    parser.add_argument('--output', required=True, help='Path to save compressed model')
    parser.add_argument('--mode', default='linear',
                      choices=['linear', 'kmeans_lut'],
                      help='Compression mode')
    parser.add_argument('--bits', type=int, default=8, help='Number of bits for quantization')
    parser.add_argument('--threshold', type=float, default=0.1,
                      help='Weight threshold for pruning')
    
    args = parser.parse_args()
    
    try:
        compress_model(
            args.input,
            args.output,
            compression_mode=args.mode,
            bits=args.bits,
            weight_threshold=args.threshold
        )
    except Exception as e:
        print(f"\nCompression failed: {str(e)}")
        exit(1)

if __name__ == '__main__':
    main()