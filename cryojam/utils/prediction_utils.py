import numpy as np
from scipy.spatial import cKDTree
import numpy as np
# from scipy.spatial.distance import cdist #this needs to use tensors
from scipy.optimize import linear_sum_assignment
from Bio.PDB import PDBParser, PDBIO, Select
import torch 

######### 20240511 all the changes that were made have a comment that indicates what was the changed

def binarize_predictions(preds, true_atom_count, min_distance):
    preds = preds.ravel()
    indices = torch.argsort(-preds) #switched to torch here

    result = torch.zeros_like(preds) #switched to torch here
    selected_points = []
    tree = None

    count = 0
    for idx in indices[:true_atom_count]:
        point = torch.unravel_index(idx, preds.shape)
        point = tuple(tensor.cpu().numpy() for tensor in point)
        # point#switched here
        #print("Checking point:", point)

        if not selected_points:
            selected_points.append(point)
            result[idx] = 1
            #print("Adding first point:", point)
            count += 1
            if len(selected_points) == 1:
                tree = cKDTree(selected_points)
        else:
            dist, _ = tree.query(point)
            #print("Distance from nearest point:", dist)
            if dist >= min_distance:
                selected_points.append(point)
                tree = cKDTree(selected_points)  # Rebuild tree with new point
                result[idx] = 1
                count += 1
                #print("Adding point:", point)

    # print("Total points added:", count)
    return result.reshape((64,64,64))


def greedy_selection(true, estim):
    # oh dear does this (re)assignment have to occur EACH time? hopefully fast on GPU
    true = torch.from_numpy(true) # NEW
    # true, estim should be in coordinates shape: a 2d array w/ size (c_alphas, 3)
    dist_matrix = torch.cdist(true.to(true.dtype), estim.to(true.dtype)) #this was changed to torch
    
    # Apply the Hungarian algorithm to the distance matrix.
    
    # true_ind, estim_ind = linear_sum_assignment(dist_matrix) #commented out to try the code below
    dist_matrix_cpu = dist_matrix.cpu()  # Move the tensor to CPU
    dist_matrix_numpy = dist_matrix_cpu.numpy()  # Convert to NumPy array
    true_ind, estim_ind = linear_sum_assignment(dist_matrix_numpy)
    
    true_coords_sorted = true[true_ind]
    estim_coords_sorted = estim[estim_ind]
    return true_coords_sorted, estim_coords_sorted

    '''
def coords_from_scaled_vol(vol, scale_dict):
    vol = torch.from_numpy(vol)
    coords = torch.argwhere(vol == 1)
    scaled_coords = coords * 1 / scale_dict["norm"] + scale_dict["min_coord"]
    return scaled_coords
    '''
### NEW CHANGES 
def coords_from_scaled_vol(vol, scale_dict):
    coords = torch.argwhere(vol == 1)
    scaled_coords = coords * 1 / scale_dict["norm"] + scale_dict["min_coord"]
    return scaled_coords


def generate_pdb(pdb_key, prediction_vol, scale_dict, ca_assignment=greedy_selection):
    # should return a  prediction_vol based PDB
    # first collect top k:
    parser = PDBParser()
    structure = parser.get_structure("pdb", "../data/backbones/" + pdb_key + "_backbone.pdb")
    true_coords_pdb_scale = np.array([atom.get_coord() for atom in structure.get_atoms() if atom.get_name() == "CA"])

    true_ca_count = len(true_coords_pdb_scale)
    preds_binarized = binarize_predictions(prediction_vol, true_ca_count, min_distance = 1)
    # this is a volume, get out coordinates + scale 
    preds_binarized_pdb_scale_coords = coords_from_scaled_vol(preds_binarized, scale_dict)
    
    # assign Cas to atoms based on true_ca:
    true_coords_sorted, estim_coords_sorted = ca_assignment(true_coords_pdb_scale, 
                                                            preds_binarized_pdb_scale_coords)
    
    
    
    return preds_binarized, true_coords_sorted, estim_coords_sorted, structure


def revert_coordinates_using_dict(normalized_coords, true_scale):
    # Ensure norm and min_coord are 1D and have the same length as the dimension of coordinates
    norm = true_scale['norm'].flatten()
    min_coord = true_scale['min_coord'].flatten()
    
    # Inverse scaling factors are the reciprocal of the scaling factors
    inverse_scale_factors = 1 / norm
    
    # Ensure normalized_coords is compatible for broadcasting
    if normalized_coords.dim() == 1:
        normalized_coords = normalized_coords.unsqueeze(0)  # Make it [1, num_dimensions]
    
    # Revert the scaling
    scaled_up_coords = normalized_coords * inverse_scale_factors
    
    # Add the min_coord to shift back to the original location
    original_scale_coords = scaled_up_coords + min_coord
    
    return original_scale_coords
