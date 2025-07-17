#!/usr/bin/env python3
"""
Comprehensive test script to verify CryoJAM model loading and inference
"""

import torch
import numpy as np
import argparse
import os
import json
from tqdm import tqdm
from CryoNET import UNet
from cryodataset import CryoDataNew
from cryojam.utils.loss_utils import combined_loss_function, calculate_fsc, calculate_rmse, calculate_dice
from torch.utils.data import DataLoader

def load_checkpoint(filepath, model, optimizer=None, device='cuda'):
    """Load model checkpoint with comprehensive metadata."""
    try:
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
    except Exception as e:
        print(f"✗ Failed to load checkpoint: {e}")
        return None

def test_model_loading(checkpoint_path, device='cuda'):
    """Test if a saved model can be loaded successfully"""
    try:
        # Load model
        model = UNet()
        checkpoint = load_checkpoint(checkpoint_path, model, device=device)
        if checkpoint is None:
            return None
            
        model.to(device)
        model.eval()
        
        print(f"✓ Model loaded successfully from {checkpoint_path}")
        print(f"  - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  - Device: {next(model.parameters()).device}")
        
        return model, checkpoint
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return None, None

def calculate_metrics(predictions, targets, shells=20):
    """Calculate comprehensive metrics for model evaluation."""
    metrics = {}
    
    # Basic metrics
    metrics['fsc'] = calculate_fsc(predictions, targets, shells).mean().item()
    metrics['rmse'] = calculate_rmse(predictions, targets).item()
    metrics['dice'] = calculate_dice(predictions, targets).item()
    
    # Additional metrics
    metrics['prediction_mean'] = predictions.mean().item()
    metrics['prediction_std'] = predictions.std().item()
    metrics['prediction_min'] = predictions.min().item()
    metrics['prediction_max'] = predictions.max().item()
    
    return metrics

def test_inference_comprehensive(model, dataset_path, device='cuda', num_samples=None, shells=20):
    """Comprehensive model inference testing with detailed metrics"""
    try:
        # Load dataset
        dataset = CryoDataNew(dataset_path)
        
        if num_samples is None:
            num_samples = len(dataset)
        else:
            num_samples = min(num_samples, len(dataset))
        
        print(f"Testing inference on {num_samples} samples...")
        
        # Initialize metrics storage
        all_metrics = []
        total_combo_loss = 0
        total_fsc_loss = 0
        total_rmse_loss = 0
        
        # Create data loader for batch processing
        test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_loader, desc="Testing samples")):
                if i >= num_samples:
                    break
                    
                # Prepare input (handle both old and new data formats)
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
                    print(f"Warning: Unknown data format in batch {i}")
                    continue
                
                # Run inference
                outputs = model(inputs)
                predictions = outputs[:, :1, :, :, :].squeeze()
                
                # Calculate losses
                combo_loss, fsc_loss, rmse_loss, dice_loss = combined_loss_function(
                    predictions, true_ca.squeeze(), shells
                )
                
                # Calculate additional metrics
                sample_metrics = calculate_metrics(predictions, true_ca.squeeze(), shells)
                sample_metrics['combo_loss'] = combo_loss.item()
                sample_metrics['fsc_loss'] = fsc_loss.item()
                sample_metrics['rmse_loss'] = rmse_loss.item()
                sample_metrics['dice_loss'] = dice_loss.item()
                
                # Store metrics
                all_metrics.append(sample_metrics)
                total_combo_loss += combo_loss.item()
                total_fsc_loss += fsc_loss.item()
                total_rmse_loss += rmse_loss.item()
                
                # Print sample info
                sample_name = batch.get('name', [f'sample_{i}'])[0] if 'name' in batch else f'sample_{i}'
                print(f"\n  Sample {i+1}: {sample_name}")
                print(f"    - Input shape: {inputs.shape}")
                print(f"    - Output shape: {predictions.shape}")
                print(f"    - Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
                print(f"    - Combo Loss: {combo_loss.item():.4f}")
                print(f"    - FSC Loss: {fsc_loss.item():.4f}")
                print(f"    - RMSE Loss: {rmse_loss.item():.4f}")
                print(f"    - Dice Loss: {dice_loss.item():.4f}")
        
        # Calculate average metrics
        avg_metrics = {
            'avg_combo_loss': total_combo_loss / num_samples,
            'avg_fsc_loss': total_fsc_loss / num_samples,
            'avg_rmse_loss': total_rmse_loss / num_samples,
            'num_samples': num_samples
        }
        
        # Calculate standard deviations
        fsc_losses = [m['fsc_loss'] for m in all_metrics]
        rmse_losses = [m['rmse_loss'] for m in all_metrics]
        combo_losses = [m['combo_loss'] for m in all_metrics]
        
        avg_metrics['std_combo_loss'] = np.std(combo_losses)
        avg_metrics['std_fsc_loss'] = np.std(fsc_losses)
        avg_metrics['std_rmse_loss'] = np.std(rmse_losses)
        
        print(f"\n✓ Comprehensive inference test completed!")
        print(f"  - Average Combo Loss: {avg_metrics['avg_combo_loss']:.4f} ± {avg_metrics['std_combo_loss']:.4f}")
        print(f"  - Average FSC Loss: {avg_metrics['avg_fsc_loss']:.4f} ± {avg_metrics['std_fsc_loss']:.4f}")
        print(f"  - Average RMSE Loss: {avg_metrics['avg_rmse_loss']:.4f} ± {avg_metrics['std_rmse_loss']:.4f}")
        
        return True, avg_metrics, all_metrics
        
    except Exception as e:
        print(f"✗ Comprehensive inference test failed: {e}")
        return False, {}, []

def save_test_results(results, output_file):
    """Save test results to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Test results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Comprehensive CryoJAM model testing')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='./data/training_data.h5',
                       help='Path to dataset for testing')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Number of samples to test (default: all)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--shells', type=int, default=20,
                       help='Number of FSC shells')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for test results (JSON)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print("=== CryoJAM Model Testing ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {device}")
    print(f"Shells: {args.shells}")
    
    # Test model loading
    model, checkpoint = test_model_loading(args.checkpoint, device)
    
    if model is not None:
        # Run comprehensive inference testing
        success, avg_metrics, all_metrics = test_inference_comprehensive(
            model, args.dataset, device, args.num_samples, args.shells
        )
        
        if success:
            # Prepare results for saving
            results = {
                'checkpoint_path': args.checkpoint,
                'dataset_path': args.dataset,
                'device': str(device),
                'shells': args.shells,
                'num_samples': avg_metrics['num_samples'],
                'average_metrics': avg_metrics,
                'all_sample_metrics': all_metrics,
                'checkpoint_info': {
                    'epoch': checkpoint.get('epoch', 'N/A'),
                    'combo_loss': checkpoint.get('combo_loss', 'N/A'),
                    'fsc_loss': checkpoint.get('fsc_loss', 'N/A'),
                    'rmse_loss': checkpoint.get('rmse_loss', 'N/A')
                } if checkpoint else {}
            }
            
            # Save results if output file specified
            if args.output:
                save_test_results(results, args.output)
            
            print(f"\n=== Test Summary ===")
            print(f"✓ Model testing completed successfully!")
            print(f"  - Tested {avg_metrics['num_samples']} samples")
            print(f"  - Average Combo Loss: {avg_metrics['avg_combo_loss']:.4f}")
            print(f"  - Average FSC Loss: {avg_metrics['avg_fsc_loss']:.4f}")
            print(f"  - Average RMSE Loss: {avg_metrics['avg_rmse_loss']:.4f}")
        else:
            print("✗ Model testing failed!")
    else:
        print("✗ Model loading failed!")
    
    print("\n=== Test completed! ===")

if __name__ == "__main__":
    main() 