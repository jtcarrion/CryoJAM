#!/usr/bin/env python3
"""
CryoJAM Training Script
Run this script to train the CryoJAM model and save weights.
"""

import torch
import numpy as np
import argparse
import os
import json
from train import train
from unet_model import UNet
from cryojam.utils.loss_utils import combined_loss_function

def save_checkpoint(model, optimizer, epoch, combo_l, fsc_l, rmse_l, metrics, filepath):
    """Save model checkpoint with comprehensive metadata."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'combo_loss': combo_l,
        'fsc_loss': fsc_l,
        'rmse_loss': rmse_l,
        'metrics': metrics,
        'training_config': {
            'learning_rate': optimizer.param_groups[0]['lr'],
            'epoch': epoch
        }
    }
    torch.save(checkpoint, filepath)
    print(f"✓ Checkpoint saved: {filepath}")

def load_checkpoint(filepath, model, optimizer=None, device='cuda'):
    """Load model checkpoint with metadata."""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"✓ Checkpoint loaded from epoch {checkpoint['epoch']}")
    if 'combo_loss' in checkpoint:
        print(f"  Combo Loss: {checkpoint['combo_loss']:.4f}")
        print(f"  FSC Loss: {checkpoint['fsc_loss']:.4f}")
        print(f"  RMSE Loss: {checkpoint['rmse_loss']:.4f}")
    elif 'loss' in checkpoint:
        print(f"  Loss: {checkpoint['loss']:.4f}")
    if 'timestamp' in checkpoint:
        print(f"  Timestamp: {checkpoint['timestamp']}")
    
    return checkpoint

def save_training_history(history, filepath):
    """Save training history to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✓ Training history saved: {filepath}")

def main():
    parser = argparse.ArgumentParser(description='Train CryoJAM model')
    parser.add_argument('--dataset-path', type=str, default='./data/training_data.h5', help='Path to the H5 dataset file')
    parser.add_argument('--checkpoint-dir', type=str, default='./ckpt', help='Directory to save checkpoints')
    parser.add_argument('--num-epochs', type=int, default=25, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--shells', type=int, default=20, help='Number of FSC shells')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--save-every', type=int, default=5, help='Save checkpoint every N epochs')
    parser.add_argument('--save-best', action='store_true', help='Save best model based on validation loss')
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
    
    # Generate checkpoint filenames
    base_checkpoint = os.path.join(args.checkpoint_dir, 
                                   f"cryojam_checkpoint_{args.num_epochs}epochs")
    best_checkpoint = os.path.join(args.checkpoint_dir, 
                                   f"cryojam_best_{args.num_epochs}epochs")
    history_file = os.path.join(args.checkpoint_dir, 
                                f"training_history_{args.num_epochs}epochs.json")
    
    print(f"Training for {args.num_epochs} epochs...")
    print(f"Checkpoints will be saved to: {args.checkpoint_dir}")
    print(f"Save every {args.save_every} epochs")
    if args.save_best:
        print(f"Best model will be saved to: {best_checkpoint}")
    
    try:
        # Initialize training history
        training_history = {
            'epochs': [],
            'train_loss': [],
            'fsc_loss': [],
            'rmse_loss': [],
            'best_epoch': 0,
            'best_loss': float('inf'),
            'config': vars(args)
        }
        
        # Run training with checkpointing
        test_loader = train(
            dataset_path=args.dataset_path,
            seed=args.seed,
            device=device,
            checkpoint_file=base_checkpoint,
            shells=args.shells,
            num_epochs=args.num_epochs,
            lr=args.lr,
            save_every=args.save_every,
            save_best=args.save_best,
            training_history=training_history,
            best_checkpoint_path=best_checkpoint
        )
        
        # Save final training history
        save_training_history(training_history, history_file)
        
        print(f"\nTraining completed successfully!")
        print(f"Final model saved to: {base_checkpoint}_epoch_{args.num_epochs}.pth")
        if args.save_best:
            print(f"Best model saved to: {best_checkpoint}")
        print(f"Training history saved to: {history_file}")
        
        # Test loading the best model
        if args.save_best and os.path.exists(best_checkpoint):
            print("\nTesting best model loading...")
            model = UNet()
            load_checkpoint(best_checkpoint, model, device=device)
            model.to(device)
            model.eval()
            print("✓ Best model loaded successfully!")
            
            # Test the model on a sample from test_loader
            with torch.no_grad():
                for batch in test_loader:
                    homolog_1 = batch['homolog_1'].to(device)
                    homolog_2 = batch['homolog_2'].to(device)
                    homolog_3 = batch['homolog_3'].to(device)
                    true_vol = batch['syn_density'].to(device)
                    true_ca = batch['gt_voxel'].to(device)
                    
                    # Stack inputs
                    inputs = torch.stack((homolog_1, homolog_2, homolog_3, true_vol), dim=1)
                    
                    # Get model output
                    output = model(inputs)
                    homolog_ca_predictions = output[:, :1, :, :, :].squeeze()
                    
                    # Calculate losses
                    combo_loss, fsc_loss, rmse_loss, _ = combined_loss_function(
                        homolog_ca_predictions, true_ca.squeeze(), args.shells, g=0
                    )
                    
                    print(f"Test sample - Combo loss: {combo_loss.item():.4f}")
                    print(f"Test sample - FSC loss: {fsc_loss.item():.4f}")
                    print(f"Test sample - RMSE loss: {rmse_loss.item():.4f}")
                    break  # Only test on first sample

            
    except Exception as e:
        print(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    main() 