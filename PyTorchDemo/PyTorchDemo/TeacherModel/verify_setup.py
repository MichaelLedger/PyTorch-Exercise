#!/usr/bin/env python3
"""
MobileIQA Environment Verification Script
"""

import sys

def verify_setup():
    print("=" * 50)
    print("MobileIQA Environment Verification")
    print("=" * 50)
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check core dependencies
    dependencies = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('torchaudio', 'TorchAudio'),
        ('numpy', 'NumPy'),
        ('cv2', 'OpenCV'),
        ('timm', 'TIMM'),
        ('einops', 'Einops'),
        ('scipy', 'SciPy'),
        ('PIL', 'Pillow'),
        ('tqdm', 'TQDM'),
        ('openpyxl', 'OpenPyXL')
    ]
    
    print("\nChecking dependencies:")
    print("-" * 30)
    
    all_good = True
    for module, name in dependencies:
        try:
            if module == 'cv2':
                import cv2
                version = cv2.__version__
            else:
                imported = __import__(module)
                version = getattr(imported, '__version__', 'Unknown')
            print(f"✓ {name}: {version}")
        except ImportError as e:
            print(f"✗ {name}: Not installed or import error")
            all_good = False
    
    # Check PyTorch specific info
    try:
        import torch
        print(f"\nPyTorch Configuration:")
        print(f"  - Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        print(f"  - CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  - CUDA version: {torch.version.cuda}")
            print(f"  - GPU count: {torch.cuda.device_count()}")
    except ImportError:
        print("PyTorch not available")
        all_good = False
    
    print("\n" + "=" * 50)
    if all_good:
        print("✓ Environment setup successful! Ready to use MobileIQA.")
    else:
        print("✗ Some dependencies are missing. Please check the errors above.")
    print("=" * 50)

if __name__ == "__main__":
    verify_setup()

