import sys
import os
import torch
import torchvision
import numpy as np
import coremltools as ct
from models.MobileNet_IQA import MoNet
from PIL import Image

def convert_student_model_to_coreml():
    print('Starting student model conversion to Core ML...')
    
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
    print('Converting to Core ML...')
    try:
        # Define input tensor format
        image_input = ct.TensorType(
            name="input_image",
            shape=example_input.shape,
        )
        
        # Convert the model
        coreml_model = ct.convert(
            traced_model,
            inputs=[image_input],
            compute_units=ct.ComputeUnit.ALL,  # Use all available compute units
            minimum_deployment_target=ct.target.iOS16,  # Set minimum iOS version
            convert_to="mlprogram"  # Explicitly set to mlprogram format
        )
        
        # Add metadata
        coreml_model.author = "PlanetArt: GavinXiang"
        coreml_model.license = "MIT License"
        coreml_model.short_description = "Student model for image quality assessment using MobileNet"
        coreml_model.version = "1.0.0"
        
        # Save the model
        print('Saving Core ML model...')
        coreml_model.save("StudentModel.mlpackage")
        print('Successfully saved StudentModel.mlpackage')
        
    except Exception as e:
        print(f'Error converting to Core ML: {str(e)}')
        sys.exit(1)
    
    print('Conversion completed successfully!')

if __name__ == "__main__":
    convert_student_model_to_coreml()