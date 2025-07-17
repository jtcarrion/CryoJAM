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
from CryoNET import UNet
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
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
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

def test_model_on_samples(model, test_loader, device, shells=20, num_samples=3):
    """Test model on a few samples from test_loader with comprehensive metrics."""
    print(f"\n=== Testing Model on {num_samples} Samples ===")
    
    model.eval()
    total_combo_loss = 0
    total_fsc_loss = 0
    total_rmse_loss = 0
    sample_count = 0
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_samples:
                break
                
            # Handle both data formats
            if 'homolog_ca' in batch and 'true_vol' in batch:
                # Old format
                homolog_ca = batch['homolog_ca'].to(device)
                true_vol = batch['true_vol'].to(device)
                inputs = torch.stack((homolog_ca, true_vol), dim=1)
                true_ca = batch['true_ca'].to(device)
            elif 'homolog_1' in batch and 'syn_density' in batch:
                # New format
                homolog_1 = batch['homolog_1'].to(device)
                homolog_2 = batch['homolog_2'].to(device)
                homolog_3 = batch['homolog_3'].to(device)
                true_vol = batch['syn_density'].to(device)
                inputs = torch.stack((homolog_1, homolog_2, homolog_3, true_vol), dim=1)
                true_ca = batch['gt_voxel'].to(device)
            else:
                print(f"Warning: Unknown data format in batch {i}, skipping...")
                continue
            
            # Get model output
            output = model(inputs)
            homolog_ca_predictions = output[:, :1, :, :, :].squeeze()
            
            # Calculate losses
            combo_loss, fsc_loss, rmse_loss, dice_loss = combined_loss_function(
                homolog_ca_predictions, true_ca.squeeze(), shells
            )
            
            # Accumulate losses
            total_combo_loss += combo_loss.item()
            total_fsc_loss += fsc_loss.item()
            total_rmse_loss += rmse_loss.item()
            sample_count += 1
            
            # Print sample results
            sample_name = batch.get('name', [f'sample_{i}'])[0] if 'name' in batch else f'sample_{i}'
            print(f"  Sample {i+1}: {sample_name}")
            print(f"    - Input shape: {inputs.shape}")
            print(f"    - Output shape: {homolog_ca_predictions.shape}")
            print(f"    - Prediction range: [{homolog_ca_predictions.min():.4f}, {homolog_ca_predictions.max():.4f}]")
            print(f"    - Combo Loss: {combo_loss.item():.4f}")
            print(f"    - FSC Loss: {fsc_loss.item():.4f}")
            print(f"    - RMSE Loss: {rmse_loss.item():.4f}")
            print(f"    - Dice Loss: {dice_loss.item():.4f}")
    
    if sample_count > 0:
        avg_combo_loss = total_combo_loss / sample_count
        avg_fsc_loss = total_fsc_loss / sample_count
        avg_rmse_loss = total_rmse_loss / sample_count
        
        print(f"\n=== Test Summary ===")
        print(f"  - Tested {sample_count} samples")
        print(f"  - Average Combo Loss: {avg_combo_loss:.4f}")
        print(f"  - Average FSC Loss: {avg_fsc_loss:.4f}")
        print(f"  - Average RMSE Loss: {avg_rmse_loss:.4f}")
        
        return {
            'avg_combo_loss': avg_combo_loss,
            'avg_fsc_loss': avg_fsc_loss,
            'avg_rmse_loss': avg_rmse_loss,
            'num_samples': sample_count
        }
    else:
        print("No valid samples found for testing!")
        return {}

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
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to a checkpoint .pth file to resume training from (optional)')
    parser.add_argument('--test-samples', type=int, default=3, help='Number of samples to test after training')
    
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
            best_checkpoint_path=best_checkpoint,
            resume_checkpoint=args.checkpoint
        )
        
        # Save final training history
        save_training_history(training_history, history_file)
        
        print(f"\nTraining completed successfully!")
        print(f"Final model saved to: {base_checkpoint}_epoch_{args.num_epochs}.pth")
        if args.save_best:
            print(f"Best model saved to: {best_checkpoint}")
        print(f"Training history saved to: {history_file}")
        
        # Test loading the best model or a specific checkpoint
        eval_checkpoint = args.checkpoint if args.checkpoint is not None else best_checkpoint
        if os.path.exists(eval_checkpoint):
            print(f"\nTesting model loading from: {eval_checkpoint}")
            model = UNet()
            load_checkpoint(eval_checkpoint, model, device=device)
            model.to(device)
            model.eval()
            print("✓ Model loaded successfully!")
            
            # Test the model on samples from test_loader
            if test_loader is not None:
                test_results = test_model_on_samples(
                    model, test_loader, device, args.shells, args.test_samples
                )
                
                # Save test results
                test_results_file = os.path.join(args.checkpoint_dir, 
                                               f"test_results_{args.num_epochs}epochs.json")
                with open(test_results_file, 'w') as f:
                    json.dump(test_results, f, indent=2)
                print(f"✓ Test results saved to: {test_results_file}")
            else:
                print("Warning: No test_loader available for testing")
        else:
            print(f"\nNo checkpoint found at {eval_checkpoint} for evaluation")

            
    except Exception as e:
        print(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    main() 