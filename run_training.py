#!/usr/bin/env python3
"""
CryoJAM Training Script
Run this script to train the CryoJAM model and save weights.
"""

import torch
import numpy as np
import argparse
import os
from train import train
from unet_model import UNet

def main():
    parser = argparse.ArgumentParser(description='Train CryoJAM model')
    parser.add_argument('--dataset-path', type=str, default='./data/training_data.h5',
                       help='Path to the H5 dataset file')
    parser.add_argument('--checkpoint-dir', type=str, default='./ckpt',
                       help='Directory to save checkpoints')
    parser.add_argument('--num-epochs', type=int, default=25,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--shells', type=int, default=20,
                       help='Number of FSC shells')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Generate checkpoint filename
    checkpoint_file = os.path.join(args.checkpoint_dir, 
                                  f"cryojam_checkpoint_{args.num_epochs}epochs_{args.shells}shells.pth")
    
    print(f"Training for {args.num_epochs} epochs...")
    print(f"Checkpoint will be saved to: {checkpoint_file}")
    
    try:
        # Run training
        train(
            dataset_path=args.dataset_path,
            seed=args.seed,
            device=device,
            checkpoint_file=checkpoint_file,
            shells=args.shells,
            num_epochs=args.num_epochs,
            lr=args.lr
        )
        
        print(f"\nTraining completed successfully!")
        print(f"Model weights saved to: {checkpoint_file}")
        
        # Test loading the model
        print("Testing model loading...")
        model = UNet()
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print("✓ Model loaded successfully!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    main() 