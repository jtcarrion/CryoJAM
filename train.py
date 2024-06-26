import torch
import numpy as np
import h5py
from .cryodataset import CryoDataNew 
import torch
from torch.utils.data import DataLoader, random_split
from .utils.postprocess import save_pdb
from .utils.loss_utils import check_distributions, combined_loss_function, calculate_subset_fsc_losses, update_fsc_loss_dict
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  
from .unet_model import UNet


def train(dataset_path: str = './', 
          seed: int = 42, 
          device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
          checkpoint_file: str = "./ckpt/sample.pth",
          shells: int = 20,
          num_epochs: int = 25, 
          lr: float = .001,
         ):
    dataset = CryoDataNew(train_path)
    np.random.seed(seed)
    torch.manual_seed(seed)

                   
    ##### Dataset ########
    dataset_size = len(dataset)
    train_size = int(0.9 * dataset_size)  # 90% for training
    test_size = dataset_size - train_size  # 10% for testing
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # Data loaders for both train and test sets
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(f'Using device: {device}')
    model = UNet().to(device)
    print(f"Number of Epochs to Run: {num_epochs}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    fsc_loss_train_values = []
    rmse_loss_train_values = []
    combined_loss_values = []
    
    fsc_loss_values = {"subset_chain": dict(), 
                       "box_chain": dict(),
                       "box_non_chain": dict()
                      }

    # checkpoint_file_name = '20240513_checkpoint_20shells_20epochs_copycat.pth' # change
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for i, batch in enumerate(train_loader):
                homolog_ca = batch['homolog_ca'].to(device)
                true_vol = batch['true_vol'].to(device)
                true_ca = batch['true_ca'].to(device)
                voxel_mask = batch["chain_voxel_mask"].to(device)
                name=batch["name"][0]
                
                # Stack the arrays along a new dimension to create a tensor of shape 2x64x64x64
                inputs = torch.stack((homolog_ca, true_vol), dim=1)
                
                optimizer.zero_grad()
                outputs = model(inputs)
        
                # Compute the predictions corresponding to the homolog_ca array
                homolog_ca_predictions = outputs[:, :1, :, :, :]
                assert homolog_ca_predictions.squeeze().shape == (64,64,64), 
                assert true_ca.squeeze().shape == (64,64,64), 
    
                homolog_ca_predictions = homolog_ca_predictions.squeeze()
                true_ca = true_ca.squeeze()
                voxel_mask = voxel_mask.squeeze()
                combined_loss, fsc_loss_value, rmse_loss = combined_loss_function(homolog_ca_predictions.squeeze(), 
                                                                                  true_ca.squeeze(), shells)
                combined_loss.backward()
                optimizer.step()
    
                fsc_loss_calcs = calculate_subset_fsc_losses(homolog_ca_predictions, true_ca, voxel_mask, shells)
    
                fsc_loss_train_values.append(fsc_loss_value.item())
                rmse_loss_train_values.append(rmse_loss.item())
                combined_loss_values.append(combined_loss.item())
    
                update_fsc_loss_dict(*fsc_loss_calcs, batch['name'], fsc_loss_values)
    
                # Update the progress bar
                pbar.set_postfix({'loss': combined_loss.item()})
                pbar.update(1)
                    
            # Log for each epoch
            print(f"Finished Epoch #{epoch+1}")
            print(f"Average FSC Loss: {np.array(fsc_loss_train_values).mean():.4f}")
            print(f"Average RMSE: {np.array(rmse_loss_train_values).mean():.4f}")
            print(f"Combined Loss: {np.array(combined_loss_values).mean():.4f}")
        
            # Save training data
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': combined_loss,
            }, checkpoint_file_name)
            
    
def main(args):
    torch.manual_seed(args.seed)
    dataset = CryoDataNew(args.dataset_path)  
    model = UNet(shells=args.num_shells) 
    model.load_state_dict(torch.load(args.checkpoint_file, map_location=args.device))
    model.to(args.device)
    model.eval()

    with torch.no_grad():
        for data in dataset:
            inputs = data['inputs'].to(args.device)
            outputs = model(inputs)
            homolog_ca_predictions = outputs[:, :1, :, :, :].squeeze()
            save_pdb(homolog_ca_predictions)
            calculate_subset_fsc_losses(homolog_ca_predictions) 
            check_distributions(homolog_ca_predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default='./')
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42)
    parser.add_argument("--num-shells", default=20)
    parser.add_argument("--num-epochs", default=25)
    parser.add_argument("--lr", default=0.001)
    parser.add_argument("--checkpoint-file", default="./ckpt/sample.pth")

    args = parser.parse_args()
    main(args)

   

    




    
    

