## ideally, you just call this function with a sample config file and a checkpoint file, and it runs and saves all your output to a file
import torch
import numpy as np
import h5py
from utils.cryodataset import CryoDataNew # TODO: change
import torch
from torch.utils.data import DataLoader, random_split
from utils.loss_utils import check_distributions
import matplotlib.pyplot as plt
import numpy as np

def train(dataset_path=, 
         seed=42, 
         ):
    dataset = CryoDataNew(train_path)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # TODO: move this, we should do the data split elsewhere. If the H5 allows it we should repackage accordingly?

    dataset_size = len(dataset)
    train_size = int(0.9 * dataset_size)  # 90% for training
    test_size = dataset_size - train_size  # 10% for testing
    
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # Data loaders for both train and test sets
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    


    








if __name__ == "__main__":
    # TODO: take in config file, add arguments from file, and load accordingly 
    # arguments to add:
    # train path, default is below
    dataset_path = '/MIT/Project/GenAi_Project/data/20240512_cryo_data_with_scales_and_chains.h5' # TODO: might not be the right file path, need to fix
    # seed, default 42?
    # should the seed be used to determine both train/test split AND training, or should train/test split be done before?
    # Probably the latter, so TODO do that lol

   

    




    
    

