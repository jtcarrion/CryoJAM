'''
This file creates a data_list (list of dicts): A list where each element is a dictionary
with keys 'true_ca', 'homolog_ca', 'true_vol',
and save them into an H5 file
'''
import os
import re
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom
import h5py
from tqdm import tqdm

def read_em_map(map_file_path):
    """Read EM density map file and return as numpy array."""
    try:
        # Try to read as CCP4 format (most common for EM maps)
        with open(map_file_path, 'rb') as f:
            # Read header (1024 bytes)
            header = f.read(1024)
            
            # Extract dimensions from header
            nx = int.from_bytes(header[0:4], byteorder='little')
            ny = int.from_bytes(header[4:8], byteorder='little')
            nz = int.from_bytes(header[8:12], byteorder='little')
            
            # Read data
            data = np.frombuffer(f.read(), dtype=np.float32)
            data = data.reshape((nz, ny, nx))  # Note: CCP4 uses z,y,x order
            
            return data
    except Exception as e:
        print(f"Error reading EM map {map_file_path}: {e}")
        return None

def downsample_em_map(em_map, target_size=(64, 64, 64)):
    """Downsample EM density map to target size."""
    if em_map is None:
        return None
    
    # Calculate zoom factors
    zoom_factors = [t / s for t, s in zip(target_size, em_map.shape)]
    
    # Apply zoom with cubic interpolation
    downsampled = zoom(em_map, zoom_factors, order=1)
    
    # Normalize to 0-1 range
    min_val = np.min(downsampled)
    max_val = np.max(downsampled)
    if max_val > min_val:
        downsampled = (downsampled - min_val) / (max_val - min_val)
    
    return downsampled

def list_sorted_pdbs(directory):
    """ List and sort PDB files by their PDB code. """
    pdb_files = [f for f in os.listdir(directory) if f.endswith('.pdb')]
    pdb_files.sort(key=lambda x: x.split('_')[0])  # Sorting by the PDB code prefix
    return pdb_files

def list_em_maps(directory):
    """ List and sort EM map files. """
    map_files = [f for f in os.listdir(directory) if f.endswith('.map')]
    map_files.sort()
    return map_files

def match_files_with_emdb(backbone_dir, homolog_dir, emdb_dir, mapping_file):
    """ Match files from three directories based on the mapping file. """
    # Read mapping file
    if mapping_file.endswith('.xlsx'):
        df = pd.read_excel(mapping_file)
    else:
        df = pd.read_csv(mapping_file)
    
    # Normalize column names
    df.columns = [c.strip().upper().replace(" ", "_") for c in df.columns]
    
    backbone_files = list_sorted_pdbs(backbone_dir)
    homolog_files = list_sorted_pdbs(homolog_dir)
    em_maps = list_em_maps(emdb_dir) if emdb_dir else []

    backbone_dict = {f.split('_')[0]: f for f in backbone_files}
    homolog_dict = {}
    
    # Only use hinge motion homologs (_perturbed2.pdb files)
    for f in homolog_files:
        if '_perturbed2.pdb' in f:
            base_name = f.split('_perturbed2.pdb')[0]
            homolog_dict[base_name] = f

    # Create EM map dictionary
    em_dict = {}
    for f in em_maps:
        # Extract EMDB ID from filename (e.g., emd_1234.map -> 1234)
        if f.startswith('emd_') and f.endswith('.map'):
            emdb_id = f[4:-4]  # Remove 'emd_' prefix and '.map' suffix
            em_dict[emdb_id] = f

    matched_files = []
    missing_pairs = []

    for _, row in df.iterrows():
        pdb_id = str(row.get("PDB_ID", "")).strip()
        emdb_id = str(row.get("EMDB_ID", "")).strip()
        
        if not pdb_id:
            continue
            
        # Find corresponding files
        backbone_file = backbone_dict.get(pdb_id)
        homolog_file = homolog_dict.get(pdb_id)
        em_map_file = em_dict.get(emdb_id) if emdb_id else None
        
        if backbone_file and homolog_file:
            matched_files.append({
                'pdb_id': pdb_id,
                'emdb_id': emdb_id,
                'backbone_file': backbone_file,
                'homolog_file': homolog_file,
                'em_map_file': em_map_file
            })
        else:
            missing_pairs.append({
                'pdb_id': pdb_id,
                'emdb_id': emdb_id,
                'backbone_file': backbone_file,
                'homolog_file': homolog_file,
                'em_map_file': em_map_file
            })

    return matched_files, missing_pairs

def parse_ca_atoms(pdb_filename, chain_selected=None):
    parser = PDBParser()
    structure = parser.get_structure('PDB', pdb_filename)
    ca_atoms = [residue['CA'].get_coord() for model in structure for chain in model for residue in chain if 'CA' in residue]
    if chain_selected:
        ca_atoms = [residue['CA'].get_coord() for model in structure for chain in model for residue in chain if 'CA' in residue and chain.id == chain_selected]
    return np.array(ca_atoms)

def parse_all_atoms(pdb_filename):
    parser = PDBParser()
    structure = parser.get_structure('PDB', pdb_filename)
    all_atoms = [atom.get_coord() for model in structure for chain in model for residue in chain for atom in residue]
    return np.array(all_atoms)

def rescale_3d_array(data, target_shape=(64, 64, 64)):
    """
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
    
    return zoom_factors, rescaled_data

def create_gaussian_volume(ca_coords, sigma=3):
    volume = gaussian_filter(ca_coords, sigma=sigma)
    scale_factors, volume = rescale_3d_array(volume)
    min_value = np.min(volume)
    max_value = np.max(volume)
    
    # Perform min-max normalization
    normalized_volume = (volume - min_value) / (max_value - min_value)

    return scale_factors, normalized_volume

   
def coords_to_binary_grid(coords, grid_size=(64, 64, 64)):
    # Normalize coordinates
    min_coord = np.min(coords, axis=0)
    max_coord = np.max(coords, axis=0)

    norm_coords = (coords - min_coord) / (max_coord - min_coord) * (np.array(grid_size) - 1)
    scale = {
        "min_coord" : min_coord,
        "norm" : 1 / (max_coord - min_coord) * (np.array(grid_size) - 1)
    }
    # Initialize the grid
    grid = np.zeros(grid_size, dtype=np.float32)
    
    # Convert normalized coordinates to integer indices
    indices = np.round(norm_coords).astype(int)
    # Set the corresponding positions in the grid to 1
    for idx in indices:
        if all(0 <= idx[i] < grid_size[i] for i in range(3)):  # Ensure index is within grid bounds
            grid[tuple(idx)] = 1

    return scale, grid

def get_voxel_mask(chain_true_ca_coords, scale, padding=4, grid_size=(64, 64, 64)):
    # apply the scale to the coordinates:
    grid = np.zeros(grid_size, dtype=np.float32)
    chain_scale_coords = (chain_true_ca_coords - scale["min_coord"]) * scale["norm"]
    chain_scale_coords = np.round(chain_scale_coords).astype(int)
    # pick out min and max:
    lower = np.min(chain_scale_coords, axis=0) # ok so these aren't scaled??
    upper = np.max(chain_scale_coords, axis=0)
    
    lower_adj = np.max(((0,0,0), lower - padding), axis=0)
    upper_adj =  np.min((np.array(grid_size) - 1, upper + padding), axis=0)
    
    grid[lower_adj[0]:upper_adj[0],
        lower_adj[1]:upper_adj[1],
        lower_adj[2]:upper_adj[2]] = 1
    print(np.sum(grid))
    return grid

def parse_file_name(homolog_file):
    pattern = r'chain_(.+)_deg_(\d+)_dir_(\d)'
    match = re.search(pattern, homolog_file)
    if match:
        chain = match.group(1)
        deg = int(match.group(2))
        direx = int(match.group(3))
        return chain, direx, deg
    else:
        print("issue", homolog_file)
        assert 0 == 1, "issue"

def preprocess_and_save(backbone_dir, homolog_dir, emdb_dir, mapping_file, output_file):
    with h5py.File(output_file, 'w') as f:
        matched_files, missing_pairs = match_files_with_emdb(backbone_dir, homolog_dir, emdb_dir, mapping_file)
        
        print(f"Processing {len(matched_files)} matched files...")
        print(f"Missing pairs: {len(missing_pairs)}")
        
        for item in tqdm(matched_files):
            pdb_id = item['pdb_id']
            emdb_id = item['emdb_id']
            backbone_file = item['backbone_file']
            homolog_file = item['homolog_file']
            em_map_file = item['em_map_file']
            
            true_ca_coords = parse_ca_atoms(os.path.join(backbone_dir, backbone_file))
            homolog_ca_coords = parse_ca_atoms(os.path.join(homolog_dir, homolog_file))
            
            # Parse homolog filename for chain info
            try:
                chain, direx, deg = parse_file_name(homolog_file)
                print(chain, backbone_file[:-4])
                chain_coords = parse_ca_atoms(os.path.join(backbone_dir, backbone_file), chain)
            except:
                # If parsing fails, use all chains
                chain_coords = true_ca_coords
            
            # Convert coordinates to binary grids
            true_scale, true_ca = coords_to_binary_grid(true_ca_coords, (64,64,64))
            chain_true_voxel_mask = get_voxel_mask(chain_coords, true_scale, grid_size = (64,64,64))
            homolog_scale, homolog_ca = coords_to_binary_grid(homolog_ca_coords)
            
            # Use EM density map if available, otherwise create synthetic volume
            if em_map_file and emdb_dir:
                em_map_path = os.path.join(emdb_dir, em_map_file)
                em_map = read_em_map(em_map_path)
                if em_map is not None:
                    true_vol = downsample_em_map(em_map, (64, 64, 64))
                else:
                    _, true_vol = create_gaussian_volume(true_ca)
            else:
                _, true_vol = create_gaussian_volume(true_ca)
            
            grp = f.create_group(backbone_file[:-4])
            grp.create_dataset('true_ca', data=true_ca)
            grp.create_dataset('homolog_ca', data=homolog_ca)
            grp.create_dataset('true_vol', data=true_vol)
            grp.create_dataset('true_chain_voxel_mask', data=chain_true_voxel_mask)
            grp.create_dataset('true_scale_norm', data=true_scale["norm"])
            grp.create_dataset('true_scale_min', data=true_scale["min_coord"])
            grp.create_dataset('homolog_scale_norm', data=homolog_scale["norm"])
            grp.create_dataset('homolog_scale_min', data=homolog_scale["min_coord"])
            
            # Add metadata
            grp.attrs['pdb_id'] = pdb_id
            if emdb_id:
                grp.attrs['emdb_id'] = emdb_id
            if em_map_file:
                grp.attrs['em_map_file'] = em_map_file


if __name__ == "__main__":
    
    # Example usage
    pdb_dir = './pdb_files'
    homolog_dir = './homologs'
    emdb_dir = './emdb_maps'  # Directory containing .map files
    mapping_file = './pdb_emdb_ids.xlsx'  # Excel file with EMDB-PDB mapping
    output_file = './20250713_cryo_data.h5'
    
    matched_files, missing_pairs = match_files_with_emdb(pdb_dir, homolog_dir, emdb_dir, mapping_file)
    print("No. of Matched Files:", len(matched_files))
    print("No. of Missing Pairs:", len(missing_pairs))
    preprocess_and_save(pdb_dir, homolog_dir, emdb_dir, mapping_file, output_file)