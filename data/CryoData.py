import torch
import numpy as np
from torch.utils.data import Dataset
import h5py


class CryoData(Dataset):
    def __init__(self, h5_file, representation='backbone'):
        """
        Initialize dataset for cryo-EM data.
        
        Args:
            h5_file (str): Path to H5 file
            representation (str): 'backbone' for Cα atoms or 'allatom' for all atoms
        """
        self.h5_file = h5_file
        self.representation = representation
        
        with h5py.File(self.h5_file, 'r') as file:
            self.keys = list(file.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as file:
            key_name = self.keys[idx]  # e.g., 'EMDB_1874'
            group = file[key_name]
            
            # Extract EMDB ID from key name
            emdb_id = key_name.replace('EMDB_', '')
            
            # Get ground truth data
            ground_truth_grid = torch.tensor(group['ground_truth_grid'][:])
            ground_truth_coords = torch.tensor(group['ground_truth_coords'][:])
            em_volume = torch.tensor(group['em_volume_real'][:])
            syn_volume = torch.tensor(group['em_volume_synthetic'][:])

            # Get scale information - handle scalar values properly
            def safe_read_dataset(dataset):
                """Safely read dataset whether it's scalar or array."""
                if dataset.shape == ():  # Scalar
                    return torch.tensor([dataset[()]])  # Convert scalar to 1D tensor
                else:  # Array
                    return torch.tensor(dataset[:])
            
            scale_norm = safe_read_dataset(group['scale_norm'])
            scale_min = safe_read_dataset(group['scale_min'])
            
            # Get metadata from attributes
            pdb_id = group.attrs.get('pdb_id', '')
            pdb_file = group.attrs.get('pdb_file', '')
            em_map_file = group.attrs.get('em_map_file', '')
            homolog_types = group.attrs.get('homolog_types', [])
            
            # Return ONLY tensor data for batching (no metadata)
            result = {
                'emdb_id': emdb_id,
                'pdb_id': pdb_id,
                'representation': self.representation,
                'gt_voxel': ground_truth_grid,  # 64³ binary grid
                'gt_coords': ground_truth_coords,  # Raw coordinates
                'em_density': em_volume,  # EM density map (64³)
                'syn_density': syn_volume,  # Synthetic EM density map (64³)
                'scale_norm': scale_norm,
                'scale_min': scale_min
            }
            
            # Add homologs with correct key names (matching actual H5 structure)
            if 'homolog_perturbed1' in group:
                result['homolog_1'] = torch.tensor(group['homolog_perturbed1'][:])
            if 'homolog_perturbed2' in group:
                result['homolog_2'] = torch.tensor(group['homolog_perturbed2'][:])
            if 'homolog_complex' in group:
                result['homolog_3'] = torch.tensor(group['homolog_complex'][:])
            
        return result

class CryoDataBackbone(CryoData):
    """Dataset for backbone (Cα) representation."""
    def __init__(self, h5_file):
        super().__init__(h5_file, representation='backbone')

class CryoDataAllAtom(CryoData):
    """Dataset for all-atom representation."""
    def __init__(self, h5_file):
        super().__init__(h5_file, representation='allatom')
