'''
This file creates a data_list (list of dicts): A list where each element is a dictionary
with keys 'true_ca', 'homolog_ca', 'true_vol',
and save them into an H5 file
'''
import os
import numpy as np
from Bio.PDB import PDBParser
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom
import h5py


def list_sorted_pdbs(directory):
    """ List and sort PDB files by their PDB code. """
    pdb_files = [f for f in os.listdir(directory) if f.endswith('.pdb')]
    pdb_files.sort(key=lambda x: x.split('_')[0])  # Sorting by the PDB code prefix
    return pdb_files

def match_files(backbone_dir, homolog_dir):
    """ Match files from two directories based on the PDB codes. """
    backbone_files = list_sorted_pdbs(backbone_dir)
    homolog_files = list_sorted_pdbs(homolog_dir)

    backbone_dict = {f.split('_')[0]: f for f in backbone_files}
    homolog_dict = {f.split('_')[0]: f for f in homolog_files}

    matched_files = []
    missing_pairs = []

    for pdb_code, backbone_file in backbone_dict.items():
        if pdb_code in homolog_dict:
            matched_files.append((backbone_file, homolog_dict[pdb_code]))
        else:
            missing_pairs.append((backbone_file, None))

    for pdb_code, homolog_file in homolog_dict.items():
        if pdb_code not in backbone_dict:
            missing_pairs.append((None, homolog_file))

    return matched_files, missing_pairs

def parse_ca_atoms(pdb_filename):
    parser = PDBParser()
    structure = parser.get_structure('PDB', pdb_filename)
    ca_atoms = [residue['CA'].get_coord() for model in structure for chain in model for residue in chain if 'CA' in residue]
    return np.array(ca_atoms)

def rescale_3d_array(data, target_shape=(64, 64, 64)):
    """
    Rescale a 3D array to a new shape using spline interpolation.
    
    Parameters:
    - data (np.array): The original 3D numpy array to rescale.
    - target_shape (tuple): The target dimensions (z, y, x).
    
    Returns:
    - np.array: The rescaled 3D array.
    """
    # Calculate the zoom factors for each dimension
    zoom_factors = [n / o for n, o in zip(target_shape, data.shape)]
    
    # Use spline interpolation for rescaling
    # Order=3 uses cubic spline interpolation
    # Order=0 for nearest-neighbor interpolation to keep the array binary
    rescaled_data = zoom(data, zoom_factors, order=0)
    
    return rescaled_data

def create_gaussian_volume(ca_coords, sigma=3):
    volume = gaussian_filter(ca_coords, sigma=sigma)
    volume = rescale_3d_array(volume)
    min_value = np.min(volume)
    max_value = np.max(volume)
    
    # Perform min-max normalization
    normalized_volume = (volume - min_value) / (max_value - min_value)

    return normalized_volume

   
def coords_to_binary_grid(coords, grid_size=(64, 64, 64)):
    # Normalize coordinates
    min_coord = np.min(coords, axis=0)
    max_coord = np.max(coords, axis=0)
    norm_coords = (coords - min_coord) / (max_coord - min_coord) * (np.array(grid_size) - 1)

    # Initialize the grid
    grid = np.zeros(grid_size, dtype=np.float32)
    
    # Convert normalized coordinates to integer indices
    indices = np.round(norm_coords).astype(int)
    
    # Set the corresponding positions in the grid to 1
    for idx in indices:
        if all(0 <= idx[i] < grid_size[i] for i in range(3)):  # Ensure index is within grid bounds
            grid[tuple(idx)] = 1

    return grid

def preprocess_and_save(backbone_dir, homolog_dir, output_file):
    with h5py.File(output_file, 'w') as f:
        for backbone_file, homolog_file in zip(sorted(os.listdir(backbone_dir)), sorted(os.listdir(homolog_dir))):
            true_ca_coords = parse_ca_atoms(os.path.join(backbone_dir, backbone_file))
            homolog_ca_coords = parse_ca_atoms(os.path.join(homolog_dir, homolog_file))
            
            # Convert coordinates to binary grids
            true_ca = coords_to_binary_grid(true_ca_coords, (128,128,128))
            homolog_ca = coords_to_binary_grid(homolog_ca_coords)
            true_vol = create_gaussian_volume(true_ca)  
            
            grp = f.create_group(backbone_file[:-4])
            grp.create_dataset('true_ca', data=true_ca)
            grp.create_dataset('homolog_ca', data=homolog_ca)
            grp.create_dataset('true_vol', data=true_vol)




if __name__ == "__main__":
    
    # Example usage
    backbone_dir = '../data/backbones'
    homolog_dir = '../data/full_pdb_homologs'
    output_file = 'cryo_data.h5'
    
    matched_files, missing_pairs = match_files(backbone_dir, homolog_dir)
    print("No. of Matched Files:", len(matched_files))
    print("No. of Missing Pairs:", len(missing_pairs))
    preprocess_and_save(backbone_dir, homolog_dir, output_file)