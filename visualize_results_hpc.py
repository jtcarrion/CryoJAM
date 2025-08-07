#!/usr/bin/env python3
"""
HPC-Compatible Visualization script for CryoJAM training and testing results
Optimized for headless environments and cluster computing
"""

import json
import matplotlib
# Set non-interactive backend for HPC clusters
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import sys
from pathlib import Path

def setup_hpc_environment():
    """Setup matplotlib for HPC cluster environment."""
    # Configure matplotlib for headless environment
    plt.ioff()  # Turn off interactive mode
    
    # Set font properties for better rendering
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 10
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    
    print("✓ HPC environment configured for headless plotting")

def load_json_file(filepath):
    """Load and return JSON data from file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON file: {filepath}")
        return None

def plot_training_history(history_data, output_dir="./plots"):
    """Plot training loss curves from history JSON (HPC compatible)."""
    if not history_data:
        print("❌ No training history data provided")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    epochs = history_data.get('epochs', [])
    train_loss = history_data.get('train_loss', [])
    fsc_loss = history_data.get('fsc_loss', [])
    rmse_loss = history_data.get('rmse_loss', [])
    
    if not epochs:
        print("❌ No epoch data found in training history")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('CryoJAM Training History (HPC)', fontsize=16, fontweight='bold')
    
    # Plot 1: Combined Loss
    axes[0, 0].plot(epochs, train_loss, 'b-', linewidth=2, label='Combined Loss')
    axes[0, 0].set_title('Combined Loss Over Time')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: FSC Loss
    axes[0, 1].plot(epochs, fsc_loss, 'r-', linewidth=2, label='FSC Loss')
    axes[0, 1].set_title('FSC Loss Over Time')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot 3: RMSE Loss
    axes[1, 0].plot(epochs, rmse_loss, 'g-', linewidth=2, label='RMSE Loss')
    axes[1, 0].set_title('RMSE Loss Over Time')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Plot 4: All losses together
    axes[1, 1].plot(epochs, train_loss, 'b-', linewidth=2, label='Combined Loss')
    axes[1, 1].plot(epochs, fsc_loss, 'r-', linewidth=2, label='FSC Loss')
    axes[1, 1].plot(epochs, rmse_loss, 'g-', linewidth=2, label='RMSE Loss')
    axes[1, 1].set_title('All Losses Comparison')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    # Save plot (no plt.show() for HPC)
    plot_path = os.path.join(output_dir, 'training_history_hpc.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to free memory
    print(f"✓ Training history plot saved: {plot_path}")
    
    # Print summary statistics
    print(f"\n📊 Training Summary:")
    print(f"  - Total epochs: {len(epochs)}")
    print(f"  - Best epoch: {history_data.get('best_epoch', 'N/A')}")
    print(f"  - Best loss: {history_data.get('best_loss', 'N/A'):.4f}")
    print(f"  - Final combined loss: {train_loss[-1]:.4f}")
    print(f"  - Final FSC loss: {fsc_loss[-1]:.4f}")
    print(f"  - Final RMSE loss: {rmse_loss[-1]:.4f}")

def plot_test_results(test_data, output_dir="./plots"):
    """Plot test results from test JSON (HPC compatible)."""
    if not test_data:
        print("❌ No test data provided")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract metrics
    avg_metrics = test_data.get('average_metrics', {})
    all_sample_metrics = test_data.get('all_sample_metrics', [])
    
    if not avg_metrics:
        print("❌ No average metrics found in test data")
        return
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('CryoJAM Test Results (HPC)', fontsize=16, fontweight='bold')
    
    # Plot 1: Average losses
    losses = ['avg_combo_loss', 'avg_fsc_loss', 'avg_rmse_loss']
    loss_names = ['Combined Loss', 'FSC Loss', 'RMSE Loss']
    loss_values = [avg_metrics.get(loss, 0) for loss in losses]
    
    bars = axes[0, 0].bar(loss_names, loss_values, color=['blue', 'red', 'green'])
    axes[0, 0].set_title('Average Test Losses')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, loss_values):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{value:.4f}', ha='center', va='bottom')
    
    # Plot 2: Individual sample losses (if available)
    if all_sample_metrics:
        sample_indices = list(range(len(all_sample_metrics)))
        combo_losses = [m.get('combo_loss', 0) for m in all_sample_metrics]
        fsc_losses = [m.get('fsc_loss', 0) for m in all_sample_metrics]
        rmse_losses = [m.get('rmse_loss', 0) for m in all_sample_metrics]
        
        axes[0, 1].scatter(sample_indices, combo_losses, c='blue', alpha=0.7, label='Combined Loss')
        axes[0, 1].scatter(sample_indices, fsc_losses, c='red', alpha=0.7, label='FSC Loss')
        axes[0, 1].scatter(sample_indices, rmse_losses, c='green', alpha=0.7, label='RMSE Loss')
        axes[0, 1].set_title('Individual Sample Losses')
        axes[0, 1].set_xlabel('Sample Index')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Prediction statistics
    if all_sample_metrics:
        pred_means = [m.get('prediction_mean', 0) for m in all_sample_metrics]
        pred_stds = [m.get('prediction_std', 0) for m in all_sample_metrics]
        
        axes[1, 0].scatter(sample_indices, pred_means, c='purple', alpha=0.7, label='Mean')
        axes[1, 0].scatter(sample_indices, pred_stds, c='orange', alpha=0.7, label='Std Dev')
        axes[1, 0].set_title('Prediction Statistics')
        axes[1, 0].set_xlabel('Sample Index')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Loss distribution histogram
    if all_sample_metrics:
        axes[1, 1].hist(combo_losses, bins=10, alpha=0.7, color='blue', label='Combined Loss')
        axes[1, 1].hist(fsc_losses, bins=10, alpha=0.7, color='red', label='FSC Loss')
        axes[1, 1].hist(rmse_losses, bins=10, alpha=0.7, color='green', label='RMSE Loss')
        axes[1, 1].set_title('Loss Distribution')
        axes[1, 1].set_xlabel('Loss Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot (no plt.show() for HPC)
    plot_path = os.path.join(output_dir, 'test_results_hpc.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to free memory
    print(f"✓ Test results plot saved: {plot_path}")
    
    # Print summary
    print(f"\n📊 Test Summary:")
    print(f"  - Number of samples: {avg_metrics.get('num_samples', 'N/A')}")
    print(f"  - Average combined loss: {avg_metrics.get('avg_combo_loss', 'N/A'):.4f}")
    print(f"  - Average FSC loss: {avg_metrics.get('avg_fsc_loss', 'N/A'):.4f}")
    print(f"  - Average RMSE loss: {avg_metrics.get('avg_rmse_loss', 'N/A'):.4f}")

def generate_summary_report(history_data, test_data, output_dir="./plots"):
    """Generate a text summary report (useful for HPC)."""
    report_path = os.path.join(output_dir, 'training_summary.txt')
    
    with open(report_path, 'w') as f:
        f.write("=== CryoJAM Training Summary Report ===\n\n")
        
        if history_data:
            f.write("TRAINING HISTORY:\n")
            f.write(f"  - Total epochs: {len(history_data.get('epochs', []))}\n")
            f.write(f"  - Best epoch: {history_data.get('best_epoch', 'N/A')}\n")
            f.write(f"  - Best loss: {history_data.get('best_loss', 'N/A'):.4f}\n")
            f.write(f"  - Final combined loss: {history_data.get('train_loss', [0])[-1]:.4f}\n")
            f.write(f"  - Final FSC loss: {history_data.get('fsc_loss', [0])[-1]:.4f}\n")
            f.write(f"  - Final RMSE loss: {history_data.get('rmse_loss', [0])[-1]:.4f}\n\n")
        
        if test_data:
            avg_metrics = test_data.get('average_metrics', {})
            f.write("TEST RESULTS:\n")
            f.write(f"  - Number of samples: {avg_metrics.get('num_samples', 'N/A')}\n")
            f.write(f"  - Average combined loss: {avg_metrics.get('avg_combo_loss', 'N/A'):.4f}\n")
            f.write(f"  - Average FSC loss: {avg_metrics.get('avg_fsc_loss', 'N/A'):.4f}\n")
            f.write(f"  - Average RMSE loss: {avg_metrics.get('avg_rmse_loss', 'N/A'):.4f}\n")
            f.write(f"  - Std combined loss: {avg_metrics.get('std_combo_loss', 'N/A'):.4f}\n")
            f.write(f"  - Std FSC loss: {avg_metrics.get('std_fsc_loss', 'N/A'):.4f}\n")
            f.write(f"  - Std RMSE loss: {avg_metrics.get('std_rmse_loss', 'N/A'):.4f}\n")
    
    print(f"✓ Summary report saved: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='HPC-Compatible CryoJAM Results Visualizer')
    parser.add_argument('--history-file', type=str, help='Path to training history JSON file')
    parser.add_argument('--test-file', type=str, help='Path to test results JSON file')
    parser.add_argument('--output-dir', type=str, default='./plots', help='Output directory for plots')
    parser.add_argument('--show-structure', action='store_true', help='Show JSON structure')
    parser.add_argument('--generate-report', action='store_true', help='Generate text summary report')
    
    args = parser.parse_args()
    
    # Setup HPC environment
    setup_hpc_environment()
    
    print("=== CryoJAM HPC Results Visualizer ===\n")
    print(f"Output directory: {args.output_dir}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    history_data = None
    test_data = None
    
    # Handle training history
    if args.history_file:
        print(f"📈 Loading training history from: {args.history_file}")
        history_data = load_json_file(args.history_file)
        
        if history_data:
            if args.show_structure:
                print("\n📋 Training History Structure:")
                print(f"  Keys: {list(history_data.keys())}")
                print(f"  Epochs: {len(history_data.get('epochs', []))}")
            
            plot_training_history(history_data, args.output_dir)
        else:
            print("❌ Failed to load training history")
    
    # Handle test results
    if args.test_file:
        print(f"\n🧪 Loading test results from: {args.test_file}")
        test_data = load_json_file(args.test_file)
        
        if test_data:
            if args.show_structure:
                print("\n📋 Test Results Structure:")
                print(f"  Keys: {list(test_data.keys())}")
                print(f"  Average metrics: {list(test_data.get('average_metrics', {}).keys())}")
            
            plot_test_results(test_data, args.output_dir)
        else:
            print("❌ Failed to load test results")
    
    # Auto-discover JSON files if none specified
    if not args.history_file and not args.test_file:
        print("🔍 Searching for JSON files in current directory...")
        
        json_files = list(Path('.').glob('**/*.json'))
        if json_files:
            print(f"Found {len(json_files)} JSON files:")
            for file in json_files:
                print(f"  - {file}")
            
            # Try to identify training history and test results
            for file in json_files:
                if 'history' in file.name.lower():
                    print(f"\n📈 Found training history: {file}")
                    history_data = load_json_file(str(file))
                    if history_data:
                        plot_training_history(history_data, args.output_dir)
                
                elif 'test' in file.name.lower():
                    print(f"\n🧪 Found test results: {file}")
                    test_data = load_json_file(str(file))
                    if test_data:
                        plot_test_results(test_data, args.output_dir)
        else:
            print("❌ No JSON files found in current directory")
    
    # Generate summary report
    if args.generate_report or (history_data or test_data):
        generate_summary_report(history_data, test_data, args.output_dir)
    
    print(f"\n✅ HPC visualization complete!")
    print(f"📁 Outputs saved to: {args.output_dir}")
    print(f"📊 Files generated:")
    print(f"  - training_history_hpc.png")
    print(f"  - test_results_hpc.png")
    print(f"  - training_summary.txt")

if __name__ == "__main__":
    main() 