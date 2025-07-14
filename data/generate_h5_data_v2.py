'''
This file creates two H5 files: backbone and all-atom representations
organized by EMDB ID with multiple homolog types
'''
import os
import re
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom
import h5py
import random
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

def create_synthetic_em_density(all_atom_coords, sigma_range=(3, 8)):
    """
    Create synthetic EM density from all-atom coordinates with stochastic sigma.
    
    Args:
        all_atom_coords (np.array): All atom coordinates from PDB
        sigma_range (tuple): Range for uniform sampling of sigma (min, max)
    
    Returns:
        np.array: Synthetic EM density map (64, 64, 64)
    """
    # Sample sigma from uniform distribution
    sigma = np.random.uniform(sigma_range[0], sigma_range[1])
    
    # Create 3D volume from atom coordinates
    # Initialize a larger volume to accommodate the gaussian spread
    volume_size = 128  # Start with larger volume
    volume = np.zeros((volume_size, volume_size, volume_size), dtype=np.float32)
    
    # Normalize coordinates to fit in the volume
    min_coord = np.min(all_atom_coords, axis=0)
    max_coord = np.max(all_atom_coords, axis=0)
    
    # Add padding to ensure atoms don't touch edges
    padding = 10
    coord_range = max_coord - min_coord
    scale_factor = (volume_size - 2 * padding) / np.max(coord_range)
    
    # Scale and center coordinates
    scaled_coords = (all_atom_coords - min_coord) * scale_factor + padding
    
    # Place atoms in the volume
    for coord in scaled_coords:
        x, y, z = coord.astype(int)
        if 0 <= x < volume_size and 0 <= y < volume_size and 0 <= z < volume_size:
            volume[x, y, z] = 1.0
    
    # Apply gaussian filter to create density
    density = gaussian_filter(volume, sigma=sigma)
    
    # Downsample to target size (64, 64, 64)
    zoom_factors = [64 / volume_size] * 3
    downsampled = zoom(density, zoom_factors, order=1)
    
    # Normalize to 0-1 range
    min_val = np.min(downsampled)
    max_val = np.max(downsampled)
    if max_val > min_val:
        downsampled = (downsampled - min_val) / (max_val - min_val)
    
    return downsampled, sigma

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

def match_files_with_emdb(pdb_dir, homolog_dir, emdb_dir, mapping_file):
    """ Match files from directories based on the mapping file, organized by EMDB ID. """
    # Read mapping file
    if mapping_file.endswith('.xlsx'):
        df = pd.read_excel(mapping_file)
    else:
        df = pd.read_csv(mapping_file)
    
    # Normalize column names
    df.columns = [c.strip().upper().replace(" ", "_") for c in df.columns]
    
    print("[DEBUG] First 5 rows of mapping file:")
    print(df.head())
    print("[DEBUG] First 5 PDB IDs from mapping file:", df["PDB"].head().tolist() if "PDB" in df.columns else "No PDB column")
    print("[DEBUG] First 5 EMDB IDs from mapping file:", df["EMDB"].head().tolist() if "EMDB" in df.columns else "No EMDB column")
    
    pdb_files = list_sorted_pdbs(pdb_dir)
    homolog_files = list_sorted_pdbs(homolog_dir)
    em_maps = list_em_maps(emdb_dir) if emdb_dir else []

    pdb_dict = {f.split('_')[0].replace('.pdb', '').lower(): f for f in pdb_files}
    
    # Create homolog dictionary with all three types
    homolog_dict = {}
    for f in homolog_files:
        if '_perturbed1.pdb' in f:
            base_name = f.split('_perturbed1.pdb')[0].lower()
            if base_name not in homolog_dict:
                homolog_dict[base_name] = {}
            homolog_dict[base_name]['perturbed1'] = f
        elif '_perturbed2.pdb' in f:
            base_name = f.split('_perturbed2.pdb')[0].lower()
            if base_name not in homolog_dict:
                homolog_dict[base_name] = {}
            homolog_dict[base_name]['perturbed2'] = f
        elif '_complex.pdb' in f:
            base_name = f.split('_complex.pdb')[0].lower()
            if base_name not in homolog_dict:
                homolog_dict[base_name] = {}
            homolog_dict[base_name]['complex'] = f

    em_dict = {}
    for f in em_maps:
        if f.startswith('emd_') and f.endswith('.map'):
            emdb_id = f[4:-4]
            em_dict[emdb_id] = f
    
    print("[DEBUG] First 5 keys in pdb_dict:", list(pdb_dict.keys())[:5])
    print("[DEBUG] First 5 keys in homolog_dict:", list(homolog_dict.keys())[:5])
    print("[DEBUG] First 5 keys in em_dict:", list(em_dict.keys())[:5])

    matched_files = []
    missing_pairs = []

    for idx, row in df.iterrows():
        pdb_id = str(row.get("PDB", "")).strip().lower()
        emdb_id = str(row.get("EMDB", "")).strip()
        
        if not pdb_id or not emdb_id:
            continue
        
        pdb_file = pdb_dict.get(pdb_id)
        homolog_files_for_pdb = homolog_dict.get(pdb_id, {})
        em_map_file = em_dict.get(emdb_id)
        
        print(f"[DEBUG] Row {idx}: pdb_id={pdb_id}, emdb_id={emdb_id}, pdb_file={'FOUND' if pdb_file else 'MISSING'}, em_map_file={'FOUND' if em_map_file else 'MISSING'}, homolog_files={list(homolog_files_for_pdb.keys()) if homolog_files_for_pdb else 'MISSING'}")
        
        if pdb_file and em_map_file:
            matched_files.append({
                'emdb_id': emdb_id,
                'pdb_id': pdb_id,
                'pdb_file': pdb_file,
                'homolog_files': homolog_files_for_pdb,
                'em_map_file': em_map_file
            })
        else:
            missing_pairs.append({
                'emdb_id': emdb_id,
                'pdb_id': pdb_id,
                'pdb_file': pdb_file,
                'homolog_files': homolog_files_for_pdb,
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
    """
    Create 3D gaussian volume from Cα coordinates.
    
    Args:
        ca_coords (np.array): Cα atom coordinates
        sigma (float): Gaussian sigma for blurring
    
    Returns:
        tuple: (zoom_factors, normalized_volume)
    """
    # Create 3D volume from coordinates
    volume_size = 128  # Start with larger volume
    volume = np.zeros((volume_size, volume_size, volume_size), dtype=np.float32)
    
    # Normalize coordinates to fit in the volume
    min_coord = np.min(ca_coords, axis=0)
    max_coord = np.max(ca_coords, axis=0)
    
    # Add padding to ensure atoms don't touch edges
    padding = 10
    coord_range = max_coord - min_coord
    scale_factor = (volume_size - 2 * padding) / np.max(coord_range)
    
    # Scale and center coordinates
    scaled_coords = (ca_coords - min_coord) * scale_factor + padding
    
    # Place atoms in the volume
    for coord in scaled_coords:
        x, y, z = coord.astype(int)
        if 0 <= x < volume_size and 0 <= y < volume_size and 0 <= z < volume_size:
            volume[x, y, z] = 1.0
    
    # Apply gaussian filter to create density
    density = gaussian_filter(volume, sigma=sigma)
    
    # Downsample to target size (64, 64, 64)
    zoom_factors = [64 / volume_size] * 3
    downsampled = zoom(density, zoom_factors, order=1)
    
    # Normalize to 0-1 range
    min_val = np.min(downsampled)
    max_val = np.max(downsampled)
    if max_val > min_val:
        downsampled = (downsampled - min_val) / (max_val - min_val)
    
    return zoom_factors, downsampled

   
def coords_to_binary_grid(coords, grid_size=(64, 64, 64)):
    """
    Create binary grid from coordinates with consistent padding approach.
    
    Args:
        coords (np.array): Atom coordinates
        grid_size (tuple): Target grid size (default: 64³)
    
    Returns:
        tuple: (scale_dict, binary_grid)
    """
    # Use the same approach as synthetic EM density for consistency
    volume_size = 128  # Start with larger volume like synthetic EM
    volume = np.zeros((volume_size, volume_size, volume_size), dtype=np.float32)
    
    # Normalize coordinates to fit in the volume with padding
    min_coord = np.min(coords, axis=0)
    max_coord = np.max(coords, axis=0)
    
    # Add padding to ensure atoms don't touch edges (same as synthetic EM)
    padding = 10
    coord_range = max_coord - min_coord
    scale_factor = (volume_size - 2 * padding) / np.max(coord_range)
    
    # Scale and center coordinates
    scaled_coords = (coords - min_coord) * scale_factor + padding
    
    # Place atoms in the volume
    for coord in scaled_coords:
        x, y, z = coord.astype(int)
        if 0 <= x < volume_size and 0 <= y < volume_size and 0 <= z < volume_size:
            volume[x, y, z] = 1.0
    
    # Downsample to target size (64, 64, 64)
    zoom_factors = [64 / volume_size] * 3
    downsampled = zoom(volume, zoom_factors, order=0)  # Use order=0 to keep binary
    
    # Create scale dictionary for consistency
    scale = {
        "min_coord": min_coord,
        "norm": scale_factor,
        "padding": padding,
        "volume_size": volume_size
    }
    
    return scale, downsampled

def preprocess_and_save(pdb_dir, homolog_dir, emdb_dir, mapping_file, output_name):
    """Create two H5 files: backbone and all-atom representations."""
    
    matched_files, missing_pairs = match_files_with_emdb(pdb_dir, homolog_dir, emdb_dir, mapping_file)
    
    print(f"Processing {len(matched_files)} matched files...")
    print(f"Missing pairs: {len(missing_pairs)}")
    print("Saving both real and synthetic EM densities for each pair...")
    
    # Create both H5 files
    backbone_file = f"{output_name}_backbone.h5"
    allatom_file = f"{output_name}_allAtom.h5"
    
    with h5py.File(backbone_file, 'w') as f_backbone, h5py.File(allatom_file, 'w') as f_allatom:
        
        for item in tqdm(matched_files):
            emdb_id = item['emdb_id']
            pdb_id = item['pdb_id']
            pdb_file = item['pdb_file']
            homolog_files = item['homolog_files']
            em_map_file = item['em_map_file']
            
            # Parse coordinates
            backbone_coords = parse_ca_atoms(os.path.join(pdb_dir, pdb_file))
            all_atom_coords = parse_all_atoms(os.path.join(pdb_dir, pdb_file))
            
            # Process real EM map for backbone (if available)
            real_em_volume_backbone = None
            if em_map_file and emdb_dir:
                em_map_path = os.path.join(emdb_dir, em_map_file)
                em_map = read_em_map(em_map_path)
                if em_map is not None:
                    real_em_volume_backbone = downsample_em_map(em_map, (64, 64, 64))
            
            # Generate synthetic EM density for backbone
            synthetic_em_volume_backbone, backbone_sigma = create_synthetic_em_density(backbone_coords, sigma_range=(3, 8))
            
            # Generate synthetic EM density for all-atom
            synthetic_em_volume_allatom, allatom_sigma = create_synthetic_em_density(all_atom_coords, sigma_range=(3, 8))
            
            # Create groups for both files
            grp_backbone = f_backbone.create_group(f"EMDB_{emdb_id}")
            grp_allatom = f_allatom.create_group(f"EMDB_{emdb_id}")
            
            # Store ground truth data
            # Backbone file
            grp_backbone.create_dataset('ground_truth_coords', data=backbone_coords)  # Raw coordinates
            true_scale, true_backbone_grid = coords_to_binary_grid(backbone_coords, (64,64,64))
            grp_backbone.create_dataset('ground_truth_grid', data=true_backbone_grid)  # 64³ grid
            grp_backbone.create_dataset('scale_norm', data=true_scale["norm"])
            grp_backbone.create_dataset('scale_min', data=true_scale["min_coord"])
            
            # Store both real and synthetic EM densities for backbone
            grp_backbone.create_dataset('em_volume_synthetic', data=synthetic_em_volume_backbone)  # Synthetic EM density
            if real_em_volume_backbone is not None:
                grp_backbone.create_dataset('em_volume_real', data=real_em_volume_backbone)  # Real EM density
            
            # All-atom file
            grp_allatom.create_dataset('ground_truth_coords', data=all_atom_coords)  # Raw coordinates
            all_scale, all_atom_grid = coords_to_binary_grid(all_atom_coords, (64,64,64))
            grp_allatom.create_dataset('ground_truth_grid', data=all_atom_grid)  # 64³ grid
            grp_allatom.create_dataset('scale_norm', data=all_scale["norm"])
            grp_allatom.create_dataset('scale_min', data=all_scale["min_coord"])
            
            # Store both real and synthetic EM densities for all-atom
            grp_allatom.create_dataset('em_volume_synthetic', data=synthetic_em_volume_allatom)  # Synthetic EM density
            if real_em_volume_backbone is not None:
                # Use the same real EM map for all-atom (since it's the same structure)
                grp_allatom.create_dataset('em_volume_real', data=real_em_volume_backbone)  # Real EM density
            
            # Process homologs (only 64³ grids, no raw coordinates)
            for homolog_type in ['perturbed1', 'perturbed2', 'complex']:
                if homolog_type in homolog_files:
                    homolog_file = homolog_files[homolog_type]
                    
                    # Parse homolog coordinates
                    homolog_backbone_coords = parse_ca_atoms(os.path.join(homolog_dir, homolog_file))
                    homolog_all_atom_coords = parse_all_atoms(os.path.join(homolog_dir, homolog_file))
                    
                    # Create 64³ grids for homologs
                    _, homolog_backbone_grid = coords_to_binary_grid(homolog_backbone_coords, (64,64,64))
                    _, homolog_all_atom_grid = coords_to_binary_grid(homolog_all_atom_coords, (64,64,64))
                    
                    # Store in both files
                    grp_backbone.create_dataset(f'homolog_{homolog_type}', data=homolog_backbone_grid)
                    grp_allatom.create_dataset(f'homolog_{homolog_type}', data=homolog_all_atom_grid)
            
            # Add metadata
            for grp in [grp_backbone, grp_allatom]:
                grp.attrs['emdb_id'] = emdb_id
                grp.attrs['pdb_id'] = pdb_id
                grp.attrs['pdb_file'] = pdb_file
                if em_map_file:
                    grp.attrs['em_map_file'] = em_map_file
                grp.attrs['homolog_types'] = list(homolog_files.keys())
                grp.attrs['has_real_em'] = real_em_volume_backbone is not None
                if grp == grp_backbone:
                    grp.attrs['backbone_sigma'] = backbone_sigma
                else:
                    grp.attrs['allatom_sigma'] = allatom_sigma


if __name__ == "__main__":
    
    # Example usage
    pdb_dir = './data/pdb_files'
    homolog_dir = './data/homologs'
    emdb_dir = './data/emdb_maps'  # Directory containing .map files
    mapping_file = './data/pdb_emdb_ids.xlsx'  # Excel file with EMDB-PDB mapping
    output_name = './data/20250713_cryo_data'
    
    matched_files, missing_pairs = match_files_with_emdb(pdb_dir, homolog_dir, emdb_dir, mapping_file)
    print("No. of Matched Files:", len(matched_files))
    print("No. of Missing Pairs:", len(missing_pairs))
    preprocess_and_save(pdb_dir, homolog_dir, emdb_dir, mapping_file, output_name)