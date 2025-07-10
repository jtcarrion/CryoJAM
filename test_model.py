#!/usr/bin/env python3
"""
Test script to verify CryoJAM model loading and inference
"""

import torch
import numpy as np
from unet_model import UNet
from cryodataset import CryoDataNew

def test_model_loading(checkpoint_path):
    """Test if a saved model can be loaded successfully"""
    try:
        # Load model
        model = UNet()
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✓ Model loaded successfully from {checkpoint_path}")
        print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  - Loss: {checkpoint.get('loss', 'N/A')}")
        
        return model
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return None

def test_inference(model, dataset_path, num_samples=3):
    """Test model inference on a few samples"""
    try:
        # Load dataset
        dataset = CryoDataNew(dataset_path)
        
        print(f"Testing inference on {min(num_samples, len(dataset))} samples...")
        
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            
            # Prepare input
            homolog_ca = sample['homolog_ca'].unsqueeze(0)  # Add batch dimension
            true_vol = sample['true_vol'].unsqueeze(0)
            inputs = torch.stack((homolog_ca, true_vol), dim=1)
            
            # Run inference
            with torch.no_grad():
                outputs = model(inputs)
                predictions = outputs[:, :1, :, :, :].squeeze()
            
            print(f"  Sample {i+1}: {sample['name']}")
            print(f"    - Input shape: {inputs.shape}")
            print(f"    - Output shape: {predictions.shape}")
            print(f"    - Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
            
        print("✓ Inference test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Inference test failed: {e}")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test CryoJAM model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='./data/training_data.h5',
                       help='Path to dataset for testing')
    parser.add_argument('--num-samples', type=int, default=3,
                       help='Number of samples to test')
    
    args = parser.parse_args()
    
    print("Testing CryoJAM model...")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")
    
    # Test model loading
    model = test_model_loading(args.checkpoint)
    
    if model is not None:
        # Test inference
        test_inference(model, args.dataset, args.num_samples)
    
    print("\nTest completed!")

if __name__ == "__main__":
    main() 