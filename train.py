import torch
import numpy as np
import h5py
import argparse
import os
from datetime import datetime

# Set memory management environment variable
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from data.CryoData import CryoDataBackbone 
import torch
from torch.utils.data import DataLoader, random_split
from cryojam.utils.postprocess import save_pdb
from cryojam.utils.loss_utils import check_distributions, combined_loss_function, fsc_loss_function, calculate_subset_fsc_losses, update_fsc_loss_dict, calculate_full_protein_fsc_loss
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  
from CryoNET import UNet
import os


def train(dataset_path: str = './', 
          seed: int = 42, 
          device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
          output_dir: str = "./ckpt/",
          checkpoint_file: str = "./ckpt/sample.pth",
          shells: int = 20,
          num_epochs: int = 5, 
          lr: float = .001,
          save_every: int = 5,
          save_best: bool = True,
          training_history: dict = None,
          best_checkpoint_path: str = None,
          resume_checkpoint: str = None,
         ):
    
    # Clear GPU memory at start
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Check available GPU memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        if gpu_memory < 2e9:  # Less than 2GB available
            print("GPU memory insufficient, falling back to CPU")
            device = torch.device("cpu")
    
    dataset = CryoDataBackbone(dataset_path)
    np.random.seed(seed)
    torch.manual_seed(seed)

                   
    ##### Dataset ########
    dataset_size = len(dataset)
    train_size = int(0.95 * dataset_size)  # 95% for training
    test_size = dataset_size - train_size  # 5% for testing
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # Data loaders for both train and test sets
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=False)

    print(f'Using device: {device}')
    model = UNet().to(device)
    print(f"Number of Epochs to Run: {num_epochs}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Load checkpoint if provided
    start_epoch = 0
    if resume_checkpoint is not None and os.path.exists(resume_checkpoint):
        print(f"Resuming training from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming from epoch {start_epoch}")
        
        # Update training history if available
        if training_history is not None and 'combo_loss' in checkpoint:
            training_history['best_loss'] = checkpoint['combo_loss']
            training_history['best_epoch'] = start_epoch
    else:
        print("Starting training from scratch")
    
    fsc_loss_train_values = []
    rmse_loss_train_values = []
    combined_loss_values = []
    
    fsc_loss_values = {"subset_chain": dict(), 
                       "box_chain": dict(),
                       "box_non_chain": dict()
                      }

    checkpoint_file_name = checkpoint_file
    # Training loop
    best_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_losses = []
        epoch_fsc_losses = []
        epoch_rmse_losses = []
        
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for i, batch in enumerate(train_loader):
                homolog_1 = batch['homolog_1'].to(device)
                homolog_2 = batch['homolog_2'].to(device)
                homolog_3 = batch['homolog_3'].to(device)

                true_vol = batch['syn_density'].to(device)
                true_ca = batch['gt_voxel'].to(device)
                # Chain mask is optional - use synthetic density as natural mask
                voxel_mask = batch.get("chain_voxel_mask", None)
                if voxel_mask is not None:
                    voxel_mask = voxel_mask.to(device)
                
                # Stack the arrays along a new dimension to create a tensor of shape 4x64x64x64
                inputs = torch.stack((homolog_1, homolog_2, homolog_3, true_vol), dim=1)
                
                optimizer.zero_grad()
                outputs = model(inputs)
        
                # Compute the predictions corresponding to the homolog_ca array
                homolog_ca_predictions = outputs[:, :1, :, :, :]
                assert homolog_ca_predictions.squeeze().shape == (64,64,64), "Prediction shape mismatch"
                assert true_ca.squeeze().shape == (64,64,64), "Target shape mismatch"
    
                homolog_ca_predictions = homolog_ca_predictions.squeeze()
                true_ca = true_ca.squeeze()
                
                combined_loss, fsc_loss_value, rmse_loss, _ = combined_loss_function(homolog_ca_predictions.squeeze(), 
                                                                                  true_ca.squeeze(), shells, g=0)
                combined_loss.backward()
                optimizer.step()
                
                # Clear gradients and free memory
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
                # Optional: Calculate subset FSC losses if chain mask is available
                if voxel_mask is not None:
                    voxel_mask = voxel_mask.squeeze()
                    fsc_loss_calcs = calculate_subset_fsc_losses(homolog_ca_predictions, true_ca, voxel_mask, shells)
                    update_fsc_loss_dict(*fsc_loss_calcs, batch['name'], fsc_loss_values)
    
                # Track losses for this epoch
                epoch_losses.append(combined_loss.item())
                epoch_fsc_losses.append(fsc_loss_value.item())
                epoch_rmse_losses.append(rmse_loss.item())
    
                # Update the progress bar
                pbar.set_postfix({'loss': combined_loss.item()})
                pbar.update(1)
        
        # Calculate epoch averages
        avg_loss = np.mean(epoch_losses)
        avg_fsc_loss = np.mean(epoch_fsc_losses)
        avg_rmse_loss = np.mean(epoch_rmse_losses)
        
        # Update training history
        if training_history is not None:
            training_history['epochs'].append(epoch + 1)
            training_history['train_loss'].append(avg_loss)
            training_history['fsc_loss'].append(avg_fsc_loss)
            training_history['rmse_loss'].append(avg_rmse_loss)
            
            # Check if this is the best model so far
            if avg_loss < training_history['best_loss']:
                training_history['best_loss'] = avg_loss
                training_history['best_epoch'] = epoch + 1
                best_loss = avg_loss
                best_epoch = epoch + 1
                
                # Save best model
                if save_best and best_checkpoint_path is not None:
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'combo_loss': avg_loss,
                        'fsc_loss': avg_fsc_loss,
                        'rmse_loss': avg_rmse_loss,
                        'metrics': {
                            'fsc_loss': avg_fsc_loss,
                            'rmse_loss': avg_rmse_loss
                        },
                        'timestamp': datetime.now().isoformat(),
                        'training_config': {
                            'learning_rate': lr,
                            'epoch': epoch + 1
                        }
                    }, best_checkpoint_path)
                    print(f"🏆 New best model saved! Loss: {avg_loss:.4f}")
                    
        # Log for each epoch
        print(f"Finished Epoch #{epoch+1}")
        print(f"Average FSC Loss: {avg_fsc_loss:.4f}")
        print(f"Average RMSE: {avg_rmse_loss:.4f}")
        print(f"Combined Loss: {avg_loss:.4f}")
        if save_best and avg_loss == best_loss:
            print(f"🎯 Best model so far! (Epoch {epoch+1})")
        
        # Save checkpoint every N epochs
        if (epoch + 1) % save_every == 0 or epoch == num_epochs - 1:
            #checkpoint_path = f"{checkpoint_file}_epoch_{epoch+1}.pth"
            checkpoint_path = f"{output_dir}w.{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'combo_loss': avg_loss,
                'fsc_loss': avg_fsc_loss,
                'rmse_loss': avg_rmse_loss,
                'metrics': {
                    'fsc_loss': avg_fsc_loss,
                    'rmse_loss': avg_rmse_loss
                },
                'timestamp': datetime.now().isoformat(),
                'training_config': {
                    'learning_rate': lr,
                    'epoch': epoch + 1
                }
            }, checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")

    return test_loader
            
    
def main(args):
    torch.manual_seed(args.seed)  
    train(args.dataset_path, num_epochs = args.num_epochs, lr = args.lr, 
          device = args.device, seed = args.seed, checkpoint_file = args.checkpoint_file)


    '''
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    model = UNet() 
    if os.path.exists(args.checkpoint_file):
        model.load_state_dict(torch.load(args.checkpoint_file, map_location=args.device))
        print(f"Loaded checkpoint from {args.checkpoint_file}")
    else:
        print(f"No checkpoint found at {args.checkpoint_file}, starting from scratch.")
    model.to(args.device)
    model.eval()

    with torch.no_grad():
        for batch in train_loader:
            # Move all tensors to the correct device
            device = torch.device(args.device)
            
            homolog_1 = batch['homolog_1'].squeeze().to(device)
            homolog_2 = batch['homolog_2'].squeeze().to(device)
            homolog_3 = batch['homolog_3'].squeeze().to(device)
            syn_density = batch['syn_density'].squeeze().to(device)
            gt_voxel = batch['gt_voxel'].squeeze().to(device)
            
            inputs = torch.stack((
                homolog_1, homolog_2, homolog_3, syn_density
            ), dim=0).unsqueeze(0).to(device)  # Add batch dimension back
            
            outputs = model(inputs)
            homolog_ca_predictions = outputs[:, :1, :, :, :].squeeze()
            
            # Calculate FSC loss with GPU tensors
            fsc_loss = fsc_loss_function(homolog_ca_predictions, gt_voxel) 
            print(f"FSC Loss: {fsc_loss.item():.4f}")
            
            # Check distributions (ensure GPU compatibility)
            check_distributions(homolog_ca_predictions)
    '''


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default='./')
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-shells", type=int, default=20)
    parser.add_argument("--num-epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--checkpoint-file", default="./ckpt/sample.pth")

    args = parser.parse_args()
    main(args)
