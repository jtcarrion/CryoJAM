import torch
import torch.nn.functional as F
from .prediction_utils import binarize_predictions, greedy_selection

def calculate_fsc(volume1, volume2, num_shells):
    """
    Calculate FSC between two 3D volumes with full GPU acceleration.
    
    Args:
        volume1: First 3D volume tensor
        volume2: Second 3D volume tensor  
        num_shells: Number of FSC shells
    
    Returns:
        fsc: FSC values tensor on the same device as input
    """
    # Ensure both volumes are on the same device
    device = volume1.device
    volume2 = volume2.to(device)
    
    # Compute the Fourier Transforms (GPU accelerated)
    fft_vol1 = torch.fft.fftn(volume1)
    fft_vol2 = torch.fft.fftn(volume2)

    # Shift zero frequency to the center
    fft_vol1_shifted = torch.fft.fftshift(fft_vol1)
    fft_vol2_shifted = torch.fft.fftshift(fft_vol2)

    # Compute radii of each voxel (GPU accelerated)
    center = torch.tensor(volume1.shape, device=device) // 2
    kx, ky, kz = torch.meshgrid(
        torch.arange(volume1.shape[0], device=device), 
        torch.arange(volume1.shape[1], device=device), 
        torch.arange(volume1.shape[2], device=device), 
        indexing='ij'
    )
    radii = torch.sqrt((kx - center[0])**2 + (ky - center[1])**2 + (kz - center[2])**2)
    max_radius = torch.max(radii)

    # Initialize FSC on the correct device
    fsc = torch.zeros(num_shells, device=device)

    # Calculate FSC for each shell (GPU accelerated)
    shell_indices = torch.round((radii / max_radius) * num_shells).long()
    
    for i in range(num_shells):
        mask = (shell_indices == i)
        num_voxels = mask.sum()

        if num_voxels > 0:
            # GPU-accelerated correlation calculation
            corr = torch.sum(torch.conj(fft_vol1_shifted[mask]) * fft_vol2_shifted[mask])
            vol1_power = torch.sum(torch.abs(fft_vol1_shifted[mask])**2)
            vol2_power = torch.sum(torch.abs(fft_vol2_shifted[mask])**2)
            
            # Avoid division by zero
            denominator = torch.sqrt(vol1_power * vol2_power)
            if denominator > 0:
                fsc[i] = torch.abs(corr) / denominator
            else:
                fsc[i] = torch.tensor(0.0, device=device)

    return fsc


def calculate_rmse(vol1, vol2):
    # calculate root mean squared error between two 3d volume arrays, that may or may not be binarized
    vol1 = vol1.squeeze()
    vol2 = vol2.squeeze()

    rmse = ((vol1 - vol2) ** 2).mean().sqrt()
    return rmse


def calculate_dice(prediction, target):
    smooth = 1e-6  # Small constant to prevent division by zero
    intersection = torch.sum(prediction * target)
    union = torch.sum(prediction) + torch.sum(target)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

def calculate_coord_rmsd(coords1, coords2):
    # coordinates should already be ordered by pairing.
    return (coords1 - coords2).pow(2).mean().sqrt()
    
def coords_from_scaled_vol(vol, scale_dict):
    coords = torch.argwhere(vol.squeeze() == 1)
    scaled_coords = coords * 1 / scale_dict["norm"] + scale_dict["min_coord"]
    return scaled_coords
    

def calculate_rmsd(coords1, coords2):
    """
    Calculate the Root Mean Square Deviation (RMSD) between two 3D arrays.
    
    Parameters:
        array1 (numpy.ndarray): First 3D binary array.
        array2 (numpy.ndarray): Second 3D binary array.
        
    Returns:
        float: The RMSD value.
    """
    # Ensure both coordinate arrays have the same shape
    assert coords1.shape == coords2.shape

    coords1 = coords1[:, :, 0]
    coords2 = coords2[:, :, 0]
    
    # Calculate squared differences
    squared_diff = (coords1 - coords2) ** 2
    
    # Calculate mean squared difference
    mean_squared_diff = squared_diff.mean()
    
    # Calculate square root of mean squared difference (RMSD)
    rmsd = torch.sqrt(mean_squared_diff)
    
    return rmsd


def calculate_full_protein_fsc_loss(predictions, targets, synthetic_density, shells):
    """
    Calculate FSC loss using synthetic density as a natural mask with GPU acceleration.
    This treats the synthetic density as a continuous mask where higher values
    indicate more important protein regions.
    
    Args:
        predictions: Model predictions (64³ tensor)
        targets: Ground truth targets (64³ tensor) 
        synthetic_density: Synthetic EM density (64³ tensor) - used as natural mask
        shells: Number of FSC shells
    
    Returns:
        fsc_loss: FSC loss value
    """
    # Ensure all tensors are on the same device
    device = predictions.device
    targets = targets.to(device)
    synthetic_density = synthetic_density.to(device)
    
    # Use synthetic density as a continuous mask
    # Higher density regions get more weight in the FSC calculation
    weighted_predictions = predictions * synthetic_density
    weighted_targets = targets * synthetic_density
    
    # Calculate FSC loss on the weighted volumes
    fsc_loss = fsc_loss_function(weighted_predictions, weighted_targets, shells)
    
    return fsc_loss


def calculate_subset_fsc_losses(homolog_ca_predictions, true_ca, voxel_mask, shells):
    

    chain_64_fsc_box_loss = fsc_loss_function(homolog_ca_predictions * voxel_mask, 
                                              true_ca * voxel_mask, shells)
    
    non_chain_64_fsc_box_loss = fsc_loss_function(homolog_ca_predictions * (1 - voxel_mask),
                                          true_ca * (1 - voxel_mask), shells)

    
    h_masked, t_masked = homolog_ca_predictions * voxel_mask, true_ca * voxel_mask
    mask_idx = torch.nonzero(voxel_mask, as_tuple=False)
    min_mask, max_mask = mask_idx.min(axis=0)[0], mask_idx.max(axis=0)[0]
    h_masked_resized = h_masked[min_mask[0]:max_mask[0]+1,
                                  min_mask[1]:max_mask[1]+1,
                                  min_mask[2]:max_mask[2]+1]
    t_masked_resized = t_masked[min_mask[0]:max_mask[0]+1,
                                  min_mask[1]:max_mask[1]+1,
                                  min_mask[2]:max_mask[2]+1]

    chain_fsc_subset_loss = fsc_loss_function(h_masked_resized, t_masked_resized, shells)
    del h_masked, h_masked_resized, t_masked, t_masked_resized, mask_idx
    # free up memory
    
    return chain_fsc_subset_loss.item(), chain_64_fsc_box_loss.item(), non_chain_64_fsc_box_loss.item()


def update_fsc_loss_dict(chain_fsc_subset_loss, chain_64_fsc_box_loss, non_chain_64_fsc_box_loss, pdb, fsc_loss_values):
    if pdb not in fsc_loss_values["subset_chain"]:
        fsc_loss_values["subset_chain"][pdb] = []
        fsc_loss_values["box_chain"][pdb] = []
        fsc_loss_values["box_non_chain"][pdb] = []
        
    fsc_loss_values["subset_chain"][pdb].append(chain_fsc_subset_loss)
    fsc_loss_values["box_chain"][pdb].append(chain_64_fsc_box_loss)
    fsc_loss_values["box_non_chain"][pdb].append(non_chain_64_fsc_box_loss)
    # should be in place?


# compute the loss given volumes
def fsc_loss_function(prediction, target, num_shells=20):
    """
    Calculate FSC loss with GPU acceleration.
    
    Args:
        prediction: Model predictions tensor
        target: Ground truth tensor
        num_shells: Number of FSC shells
    
    Returns:
        loss: FSC loss value
    """
    # Ensure tensors are on the same device
    device = prediction.device
    target = target.to(device)
    
    # Calculate FSC values
    fsc_values = calculate_fsc(prediction, target, num_shells)
    
    # Calculate loss (1 - mean FSC)
    loss = 1.0 - fsc_values.mean()
    
    return loss

def rmsd_loss_function(prediction, target):
    return calculate_rmsd(prediction, target)

def rmse_loss_function(prediction, target):
    return calculate_rmse(prediction, target)

def dice_loss_function(prediction, target):
    return 1 - calculate_dice(prediction, target)

def cosine_similarity_loss_function(output, target):
    output_flat = output.view(output.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    cosine_sim = F.cosine_similarity(output_flat, target_flat, dim=1)
    loss = 1.0 - cosine_sim.mean()
    return loss

def coord_rmsd_loss_function(output, target, scale_dict):
    target_coords = coords_from_scaled_vol(target, scale_dict)
    true_ca_count = target_coords.shape[0]
    assert target_coords.shape[1] == 3, "coordinate arr of target not right"
    output_coords = coords_from_scaled_vol(binarize_predictions(output, true_ca_count, min_distance = 1), scale_dict)
    # from here, order matters 
    assert output_coords.shape[1] == 3, "coordinate arr of output not right"
    target_coords_sorted, output_coords_sorted = greedy_selection(target_coords, output_coords)
    loss = calculate_coord_rmsd(target_coords_sorted, output_coords_sorted)
    return loss


def combined_loss_function(prediction, target, num_shells, a=1, b=1, g=1):
    fsc_loss = fsc_loss_function(prediction, target, num_shells)
    rmse_loss = rmse_loss_function(prediction, target)
    dice_loss = dice_loss_function(prediction, target)
    total_loss = a * fsc_loss + b * rmse_loss + g * dice_loss
    return total_loss, fsc_loss, rmse_loss, dice_loss


def check_distributions(trainLoader, testLoader, num_shells=20):
    train_fsc, train_rmsd, test_fsc, test_rmsd = [], [], [], []

    for batch in trainLoader:
        fsc_value = fsc_loss_function(batch['homolog_ca'].squeeze(), batch['true_ca'].squeeze(), num_shells=num_shells)
        rmsd_value = rmsd_loss_function(batch['homolog_ca'].squeeze(), batch['true_ca'].squeeze())
        train_fsc.append(fsc_value)
        train_rmsd.append(rmsd_value)
        
    for batch in testLoader:
        fsc_value = fsc_loss_function(batch['homolog_ca'].squeeze(), batch['true_ca'].squeeze(), num_shells=num_shells)
        rmsd_value = rmsd_loss_function(batch['homolog_ca'].squeeze(), batch['true_ca'].squeeze())
        test_fsc.append(fsc_value)
        test_rmsd.append(rmsd_value)

    return train_fsc, train_rmsd, test_fsc, test_rmsd
    
