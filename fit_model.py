import torch
from unet_model import UNet
from cryojam.utils.loss_utils import combined_loss_function
import os

# Example inference function
def run_inference(test_loader, checkpoint_path, device='cuda'):
    # Load model
    model = UNet()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Loaded model from {checkpoint_path}")

    results = []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
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
            # Optionally, calculate losses
            combo_loss, fsc_loss, rmse_loss, _ = combined_loss_function(
                homolog_ca_predictions, true_ca.squeeze(), checkpoint.get('training_config', {}).get('shells', 20), g=0
            )
            results.append({
                'prediction': homolog_ca_predictions.cpu(),
                'true_ca': true_ca.cpu(),
                'combo_loss': combo_loss.item(),
                'fsc_loss': fsc_loss.item(),
                'rmse_loss': rmse_loss.item(),
                'meta': {k: batch[k] for k in batch if k not in ['homolog_1','homolog_2','homolog_3','syn_density','gt_voxel']}
            })
            print(f"Sample {i}: Combo loss={combo_loss.item():.4f}, FSC={fsc_loss.item():.4f}, RMSE={rmse_loss.item():.4f}")
    return results

# Example usage (to be replaced with your actual test_loader and checkpoint)
# from data.CryoData import CryoDataBackbone
# test_loader = ...
# checkpoint_path = 'ckpt/cryojam_best_25epochs.pth'
# results = run_inference(test_loader, checkpoint_path, device='cuda') 

if __name__ == "__main__":
    import argparse
    
        

