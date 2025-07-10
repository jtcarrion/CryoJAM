# CryoJAM Training Guide

## Overview
This guide explains how to train the CryoJAM model and save weights for distribution to others.

## Issues Fixed

### 1. **Critical Bugs in `train.py`**
- ❌ **Undefined variable**: `train_path` was used instead of `dataset_path`
- ❌ **Missing import**: `argparse` was not imported
- ❌ **Undefined variable**: `checkpoint_file_name` was commented out but used later
- ❌ **Syntax errors**: Incomplete assert statements
- ❌ **Function signature mismatch**: `main()` expected different parameters

### 2. **Critical Bugs in `loss_utils.py`**
- ❌ **Undefined variable**: `smooth` variable was missing in `calculate_dice()`
- ❌ **Typo**: `scalculate_dice` instead of `calculate_dice`

### 3. **Model Architecture Issues**
- ❌ **Inconsistent parameters**: UNet constructor didn't match usage

## How to Train

### Step 1: Setup Environment
```bash
# Create conda environment
conda create -n cryojam python=3.9
conda activate cryojam

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Prepare Data
Ensure your H5 dataset file is in the correct location:
```bash
# Your dataset should be at:
./data/training_data.h5
```

### Step 3: Run Training
```bash
# Basic training (25 epochs, 20 shells)
python run_training.py

# Custom training
python run_training.py \
    --dataset-path ./data/training_data.h5 \
    --num-epochs 50 \
    --lr 0.0005 \
    --shells 20 \
    --checkpoint-dir ./ckpt
```

### Step 4: Test the Model
```bash
# Test model loading and inference
python test_model.py \
    --checkpoint ./ckpt/cryojam_checkpoint_25epochs_20shells.pth \
    --dataset ./data/training_data.h5
```

## Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset-path` | `./data/training_data.h5` | Path to H5 dataset |
| `--checkpoint-dir` | `./ckpt` | Directory to save checkpoints |
| `--num-epochs` | `25` | Number of training epochs |
| `--lr` | `0.001` | Learning rate |
| `--shells` | `20` | Number of FSC shells |
| `--seed` | `42` | Random seed |
| `--device` | `auto` | Device (auto/cuda/cpu) |

## Model Architecture

The model uses a 3D UNet architecture:
- **Input**: 2-channel 64×64×64 volume (homolog_ca + true_vol)
- **Output**: 1-channel 64×64×64 volume (predicted CA positions)
- **Loss**: Combined FSC + RMSE + Dice loss

## Expected Output

After successful training, you'll have:
```
./ckpt/
└── cryojam_checkpoint_25epochs_20shells.pth
```

The checkpoint contains:
- Model state dict
- Optimizer state dict  
- Training epoch
- Final loss value

## Troubleshooting

### Common Issues:

1. **CUDA out of memory**
   ```bash
   # Reduce batch size or use CPU
   python run_training.py --device cpu
   ```

2. **Dataset not found**
   ```bash
   # Check your dataset path
   ls -la ./data/
   ```

3. **Import errors**
   ```bash
   # Ensure you're in the right environment
   conda activate cryojam
   pip install -r requirements.txt
   ```

## Distribution

Once training is complete, you can distribute:
1. **Model checkpoint**: `./ckpt/cryojam_checkpoint_*.pth`
2. **Requirements**: `requirements.txt`
3. **Test script**: `test_model.py`
4. **Model code**: `unet_model.py`, `cryodataset.py`

## Files Created/Fixed

### New Files:
- `run_training.py` - Clean training interface
- `test_model.py` - Model testing script
- `requirements.txt` - Dependencies
- `TRAINING_GUIDE.md` - This guide

### Fixed Files:
- `train.py` - Fixed all critical bugs
- `cryojam/utils/loss_utils.py` - Fixed undefined variables and typos

## Quick Start for Tonight

```bash
# 1. Setup environment
conda create -n cryojam python=3.9
conda activate cryojam
pip install -r requirements.txt

# 2. Run training (adjust paths as needed)
python run_training.py --dataset-path /path/to/your/data.h5

# 3. Test the model
python test_model.py --checkpoint ./ckpt/cryojam_checkpoint_25epochs_20shells.pth
```

The training will save weights that can be distributed to others for inference! 