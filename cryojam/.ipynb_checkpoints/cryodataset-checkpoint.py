### remake a prediction
import torch
import numpy as np
from torch.utils.data import Dataset
import h5py

class CryoDataNew(Dataset):
    def __init__(self, h5_file):
        self.h5_file = h5_file
        with h5py.File(self.h5_file, 'r') as file:
            self.keys = list(file.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as file:
            key_name = self.keys[idx]
            group = file[key_name]
            true_ca = torch.tensor(group['true_ca'][:])
            homolog_ca = torch.tensor(group['homolog_ca'][:])
            true_vol = torch.tensor(group['true_vol'][:])
            true_chain_voxel_mask = torch.tensor(group['true_chain_voxel_mask'][:])
            
            true_scale = {"norm": torch.tensor(group['true_scale_norm'][:]),
                          "min_coord": torch.tensor(group['true_scale_min'][:])
                         }
            homolog_scale = {"norm": torch.tensor(group['homolog_scale_norm'][:]),
                          "min_coord": torch.tensor(group['homolog_scale_min'][:])
                         }

            # scale_factors = torch.tensor(group['scale_factors'][:])
        return {'name': key_name[:4],
                'true_ca': true_ca, 
                'homolog_ca': homolog_ca, 
                'true_vol': true_vol,
                'chain_voxel_mask': true_chain_voxel_mask,
                'true_scale': true_scale, 
                'homolog_scale': homolog_scale}
