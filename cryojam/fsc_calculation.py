#!/usr/bin/env python
# coding: utf-8

# In[105]:


import numpy as np
from scipy.fftpack import fftn, fftshift
from scipy.ndimage import distance_transform_edt as edt
import matplotlib.pyplot as plt


# In[283]:


'''
Here is the function to calculate FSC between two 3D volumes (3D np.array)
'''
def calculate_fsc(volume1, volume2, bins=100):
    # Ensure input volumes are numpy arrays
    volume1 = np.asarray(volume1)
    volume2 = np.asarray(volume2)

    # Check if volumes are of the same shape
    if volume1.shape != volume2.shape:
        raise ValueError("Volumes must have the same dimensions.")
    
    # Compute Fourier transforms
    F1 = fftshift(fftn(volume1))
    F2 = fftshift(fftn(volume2))
    
    # Calculate the magnitude of the frequency vector for each voxel
    freq_magnitude = np.sqrt(np.sum(np.array(np.indices(F1.shape))**2, axis=0))
    
    # Max frequency to define the shells
    max_freq = np.max(freq_magnitude)
    
    # Initialize lists to store the sums for FSC calculation
    cross_correlation = np.zeros(bins)
    auto_correlation1 = np.zeros(bins)
    auto_correlation2 = np.zeros(bins)
    
    # Define the bin edges for the histogram
    bin_edges = np.linspace(0, max_freq, bins + 1)
    
    # Calculate the cross-correlation and auto-correlations in each shell
    for i in range(bins):
        shell_mask = (freq_magnitude >= bin_edges[i]) & (freq_magnitude < bin_edges[i+1])
        cross_correlation[i] = np.sum(np.real(F1[shell_mask] * np.conj(F2[shell_mask])))
        auto_correlation1[i] = np.sum(np.abs(F1[shell_mask])**2)
        auto_correlation2[i] = np.sum(np.abs(F2[shell_mask])**2)
    
    # Calculate FSC
    fsc = cross_correlation / np.sqrt(auto_correlation1 * auto_correlation2)
    
    # Return the FSC and the middle of the bins for plotting or further analysis
    bin_middles = (bin_edges[:-1] + bin_edges[1:]) / 2
    return fsc, bin_middles

# Example usage:
# volume1 and volume2 should be your 3D numpy arrays
# fsc, bin_middles = calculate_fsc(volume1, volume2)
# You can then plot FSC vs. spatial frequency using bin_middles and fsc values


# In[289]:


def find_resolution_limit(fsc, bin_middles, threshold=0.143):
    """
    Find the spatial frequency (resolution limit) where FSC crosses the given threshold,
    then convert this frequency to a resolution in Ångströms assuming the bin middles are in 1/100 Ångström^-1.
    
    Parameters:
    - fsc: Array of FSC values.
    - bin_middles: Array of spatial frequencies corresponding to the FSC values, assumed to be in 1/100 Ångström^-1.
    - threshold: FSC threshold to determine the resolution limit.
    
    Returns:
    - The resolution at the resolution limit in Ångströms, or None if FSC does not drop below the threshold.
    """
  
    # Find indices where the FSC falls below the threshold
    below_threshold_indices = np.where(fsc < threshold)[0]
    
    if below_threshold_indices.size > 0:
        # Take the first index where FSC falls below threshold
        first_below_index = below_threshold_indices[0]
        
        # Ensure we have a previous value for interpolation
        if first_below_index > 0:
            # Linear interpolation to find the exact crossing point
            x1, y1 = bin_middles[first_below_index - 1], fsc[first_below_index - 1]
            x2, y2 = bin_middles[first_below_index], fsc[first_below_index]
            slope = (y2 - y1) / (x2 - x1)
            x_resolution_limit = x1 + (threshold - y1) / slope
            # Convert spatial frequency to resolution in Ångströms, adjusting for units if necessary
            resolution_in_angstroms = 1 / x_resolution_limit  # Adjust if bin_middles unit assumption changes
            return resolution_in_angstroms * 100  # Converting from 1/100 Ångström^-1 to Ångströms
        else:
            # If the first below-threshold value has no previous value, fall back to direct calculation
            resolution_in_angstroms = 1 / bin_middles[first_below_index]
            return resolution_in_angstroms 
            
    return None  # No crossing below the threshold was found


# In[290]:


def fsc_curve(fsc, bin_middles):
    """
    Adjust the FSC calculation to only return values starting from where FSC equals 1.
    
    Parameters:
    - fsc: Array of FSC values.
    - bin_middles: Array of spatial frequencies corresponding to the FSC values.
    
    Returns:
    - Sliced arrays of fsc and bin_middles starting from where FSC equals 1.
    """
    # Find the index of the maximum FSC value
    start_index = np.argmax(fsc)
    max_fsc_value = np.max(fsc)

    # Find the first index where the FSC falls below a small positive value (e.g., 0.007) after the max
    stop_indices = np.where(fsc[start_index:] < 0.001)[0]
    
    # If there are no values below the threshold, use the rest of the array; otherwise, use the first below-threshold index
    if stop_indices.size == 0:
        stop_index = len(fsc)
    else:
        stop_index = stop_indices[0] + start_index  # Adjust for the offset due to starting at max_fsc_index
    
    return fsc[start_index:stop_index], bin_middles[start_index:stop_index]
    


# In[291]:


# Creating simple 3D numpy arrays as examples
# Let's create two volumes with spherical objects in the center.
# Volume1 will have a perfect sphere, and Volume2 will have a slightly ellipsoidal shape.

def create_sphere(shape, radius, offset=(0, 0, 0)):
    """Create a 3D array with a sphere (1s inside sphere, 0s outside)."""
    semisizes = (radius,) * 3
    grid = np.mgrid[[slice(-x/2 + dx, x/2 + dx, 1) for x, dx in zip(shape, offset)]]
    phi = np.sum([(grid[i] / s)**2 for i, s in enumerate(semisizes)], axis=0)
    mask = phi <= 1
    return mask.astype(np.float32)

def create_ellipsoid(shape, radiuses, offset=(0, 0, 0)):
    """Create a 3D array with an ellipsoid (1s inside ellipsoid, 0s outside)."""
    semisizes = radiuses
    grid = np.mgrid[[slice(-x/2 + dx, x/2 + dx, 1) for x, dx in zip(shape, offset)]]
    phi = np.sum([(grid[i] / s)**2 for i, s in enumerate(semisizes)], axis=0)
    mask = phi <= 1
    return mask.astype(np.float32)


# In[292]:


if __name__ == '__main__':
    # Define shape of the volumes
    shape = (64, 64, 64)
    
    # Create volumes
    volume1 = create_sphere(shape, 20)
    volume2 = create_ellipsoid(shape, (20, 22, 20))
    
    volume1.shape, volume2.shape, np.sum(volume1), np.sum(volume2)

    fsc_data, bin_middles = calculate_fsc(volume1, volume2)
    
    plt.figure(figsize=(10, 6))
    plt.plot(bin_middles, fsc_data, label='FSC curve')
    plt.axhline(y=0.143, color='r', linestyle='--', label='Threshold (0.143)')
    plt.xlabel('Spatial Frequency (100/Å)')
    plt.ylabel('FSC')
    plt.legend()
    plt.title('Fourier Shell Correlation Curve')
    plt.show()

    # Use this function with your FSC and bin_middles data
    sliced_fsc, sliced_bin_middles = fsc_curve(fsc_data, bin_middles)

    plt.figure(figsize=(10, 6))
    plt.plot(sliced_bin_middles, sliced_fsc, label='FSC curve')
    plt.axhline(y=0.143, color='r', linestyle='--', label='Threshold (0.143)')
    plt.xlabel('Spatial Frequency (100/Å)')
    plt.ylabel('FSC')
    plt.legend()
    plt.title('Fourier Shell Correlation Curve')
    plt.show()

    resolution_limit = find_resolution_limit(sliced_fsc, sliced_bin_middles)
    if resolution_limit is not None:
        print(f"The resolution limit at 0.143 is: {resolution_limit:.2f}Å")


# In[ ]:




