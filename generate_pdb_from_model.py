#!/usr/bin/env python3
"""
Generate PDB files from CryoJAM model predictions
"""

import torch
import numpy as np
import argparse
import os
import json
from pathlib import Path
from CryoNET import UNet
from cryodataset import CryoDataNew
from cryojam.utils.prediction_utils import (
    binarize_predictions, 
    coords_from_scaled_vol, 
    generate_pdb,
    revert_coordinates_using_dict
)
from cryojam.utils.postprocess import save_pdb, save_mrc

def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint."""
    try:
        model = UNet()
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"✓ Model loaded from: {checkpoint_path}")
        print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  - Loss: {checkpoint.get('combo_loss', checkpoint.get('loss', 'N/A')):.4f}")
        
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

def predict_and_binarize(model, sample, device='cuda'):
    """Run model prediction and binarize the output."""
    model.eval()
    
    with torch.no_grad():
        # Prepare input (handle both data formats)
        if 'homolog_ca' in sample and 'true_vol' in sample:
            # Old format
            homolog_ca = sample['homolog_ca'].unsqueeze(0).to(device)
            true_vol = sample['true_vol'].unsqueeze(0).to(device)
            inputs = torch.stack((homolog_ca, true_vol), dim=1)
        elif 'homolog_1' in sample and 'syn_density' in sample:
            # New format
            homolog_1 = sample['homolog_1'].unsqueeze(0).to(device)
            homolog_2 = sample['homolog_2'].unsqueeze(0).to(device)
            homolog_3 = sample['homolog_3'].unsqueeze(0).to(device)
            true_vol = sample['syn_density'].unsqueeze(0).to(device)
            inputs = torch.stack((homolog_1, homolog_2, homolog_3, true_vol), dim=1)
        else:
            raise ValueError("Unknown data format")
        
        # Get model prediction
        output = model(inputs)
        prediction = output[:, :1, :, :, :].squeeze()
        
        # Get true CA count for binarization
        if 'true_ca' in sample:
            true_ca_count = sample['true_ca'].sum().item()
        elif 'gt_voxel' in sample:
            true_ca_count = sample['gt_voxel'].sum().item()
        else:
            # Estimate from scale information
            true_ca_count = 100  # Default estimate
        
        # Binarize prediction
        binarized_prediction = binarize_predictions(prediction, true_ca_count, min_distance=1)
        
        return prediction, binarized_prediction

def generate_pdb_from_prediction(prediction, binarized_prediction, sample, output_dir, sample_name):
    """Generate PDB file from model prediction."""
    
    # Get scale information
    scale_dict = sample.get('true_scale', sample.get('homolog_scale', None))
    
    if scale_dict is None:
        print(f"⚠️  No scale information found for {sample_name}, using default scaling")
        # Create default scale dict
        scale_dict = {
            'norm': torch.tensor([1.0, 1.0, 1.0]),
            'min_coord': torch.tensor([0.0, 0.0, 0.0])
        }
    
    try:
        # Convert binarized prediction to coordinates
        coords = torch.argwhere(binarized_prediction == 1)
        
        if len(coords) == 0:
            print(f"⚠️  No atoms found in prediction for {sample_name}")
            return None
        
        # Scale coordinates back to original space
        scaled_coords = coords_from_scaled_vol(binarized_prediction, scale_dict)
        
        # Convert to PDB format
        atoms = []
        for i, coord in enumerate(scaled_coords):
            atoms.append({
                'type': 'CA',  # Carbon alpha
                'x': coord[0].item(),
                'y': coord[1].item(),
                'z': coord[2].item()
            })
        
        # Save PDB file
        pdb_filename = os.path.join(output_dir, f"{sample_name}_predicted.pdb")
        save_pdb(atoms, pdb_filename)
        print(f"✓ PDB saved: {pdb_filename} ({len(atoms)} atoms)")
        
        # Save MRC file for visualization
        mrc_filename = os.path.join(output_dir, f"{sample_name}_prediction.mrc")
        save_mrc(prediction, mrc_filename)
        print(f"✓ MRC saved: {mrc_filename}")
        
        # Save binarized MRC
        binarized_mrc_filename = os.path.join(output_dir, f"{sample_name}_binarized.mrc")
        save_mrc(binarized_prediction, binarized_mrc_filename)
        print(f"✓ Binarized MRC saved: {binarized_mrc_filename}")
        
        return {
            'sample_name': sample_name,
            'num_atoms': len(atoms),
            'pdb_file': pdb_filename,
            'mrc_file': mrc_filename,
            'binarized_mrc_file': binarized_mrc_filename,
            'coordinates': scaled_coords.cpu().numpy().tolist()
        }
        
    except Exception as e:
        print(f"❌ Error generating PDB for {sample_name}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Generate PDB files from CryoJAM model predictions')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='./data/training_data.h5',
                       help='Path to dataset')
    parser.add_argument('--output-dir', type=str, default='./pdb_outputs',
                       help='Output directory for PDB files')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Number of samples to process (default: all)')
    parser.add_argument('--sample-indices', type=str, default=None,
                       help='Comma-separated list of specific sample indices to process')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print("=== CryoJAM PDB Generation ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = load_model(args.checkpoint, device)
    if model is None:
        return
    
    # Load dataset
    try:
        dataset = CryoDataNew(args.dataset)
        print(f"✓ Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return
    
    # Determine which samples to process
    if args.sample_indices:
        # Process specific samples
        indices = [int(x.strip()) for x in args.sample_indices.split(',')]
        indices = [i for i in indices if 0 <= i < len(dataset)]
        print(f"Processing {len(indices)} specific samples: {indices}")
    elif args.num_samples:
        # Process first N samples
        indices = list(range(min(args.num_samples, len(dataset))))
        print(f"Processing first {len(indices)} samples")
    else:
        # Process all samples
        indices = list(range(len(dataset)))
        print(f"Processing all {len(indices)} samples")
    
    # Process samples
    results = []
    successful = 0
    
    for i, idx in enumerate(indices):
        try:
            sample = dataset[idx]
            sample_name = sample.get('name', f'sample_{idx}')
            
            print(f"\n[{i+1}/{len(indices)}] Processing {sample_name}...")
            
            # Generate prediction
            prediction, binarized_prediction = predict_and_binarize(model, sample, device)
            
            # Generate PDB
            result = generate_pdb_from_prediction(
                prediction, binarized_prediction, sample, args.output_dir, sample_name
            )
            
            if result:
                results.append(result)
                successful += 1
                
                # Print prediction statistics
                print(f"  - Prediction range: [{prediction.min():.4f}, {prediction.max():.4f}]")
                print(f"  - Binarized atoms: {binarized_prediction.sum().item()}")
                print(f"  - Generated atoms: {result['num_atoms']}")
            
        except Exception as e:
            print(f"❌ Error processing sample {idx}: {e}")
            continue
    
    # Save summary
    summary_file = os.path.join(args.output_dir, 'generation_summary.json')
    summary = {
        'checkpoint': args.checkpoint,
        'dataset': args.dataset,
        'total_samples': len(indices),
        'successful': successful,
        'failed': len(indices) - successful,
        'results': results
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ PDB generation complete!")
    print(f"📊 Summary:")
    print(f"  - Total samples: {len(indices)}")
    print(f"  - Successful: {successful}")
    print(f"  - Failed: {len(indices) - successful}")
    print(f"  - Output directory: {args.output_dir}")
    print(f"  - Summary file: {summary_file}")
    
    if successful > 0:
        print(f"\n📁 Generated files:")
        for result in results:
            print(f"  - {result['pdb_file']}")
            print(f"  - {result['mrc_file']}")
            print(f"  - {result['binarized_mrc_file']}")

if __name__ == "__main__":
    main() 