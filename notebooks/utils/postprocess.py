import numpy as np
import scipy.ndimage
import mrcfile
import torch
from scipy.spatial import cKDTree


def apply_gaussian_smoothing(tensor_data, sigma=1.0):
    """Apply Gaussian smoothing to a 3D tensor where points are 1."""
    # Ensure the tensor is on CPU and convert it to numpy for processing
    numpy_data = tensor_data.cpu().numpy()

    # Apply Gaussian smoothing
    smoothed_data = scipy.ndimage.gaussian_filter(numpy_data, sigma=sigma)

    return torch.from_numpy(smoothed_data).float()

def save_mrc(tensor_data, filename):
    """Save a 3D tensor data to an MRC file."""
    # Add volume for easy visualization
    data = apply_gaussian_smoothing(tensor_data)
    
    # Convert tensor to numpy and ensure it's float32
    numpy_data = data.cpu().numpy().astype(np.float32)
    
    # Write data to MRC file
    with mrcfile.new(filename, overwrite=True) as mrc:
        mrc.set_data(numpy_data)
        mrc.voxel_size = 1.0  # Set the voxel size if necessary
    
def save_pdb(atoms, filename):
    with open(filename, 'w') as file:
        for i, atom in enumerate(atoms):
            file.write("ATOM  {:5d}  {:<4s}{:3s} {:1s}{:4d}    {:8.3f}{:8.3f}{:8.3f}\n".format(
                i + 1, atom['type'], 'MOL', 'A', i + 1, atom['x'], atom['y'], atom['z']))
        
    
    