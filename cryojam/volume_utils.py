import numpy as np
import mrcfile
from scipy.ndimage import gaussian_filter
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


def ft_process_volume_from_file(file, threshold_std_factor=10.0, smoothing_sigma=1):
    with mrcfile.open(file, permissive=True) as mrc:
        volume = mrc.data

    if smoothing_sigma:
        volume = gaussian_filter(volume, sigma=smoothing_sigma)
    
    threshold = np.mean(volume) + threshold_std_factor * np.std(volume)
    
    ft_volume = np.fft.fftn(volume)
    # get zero-frequency component to be center
    ft_volume_shift = np.fft.fftshift(ft_volume)
    # magnitudes, for thresholding.
    magnitude_spectrum = np.abs(ft_volume_shift)

    threshold = magnitude_spectrum.mean() + 15 * magnitude_spectrum.std()
    indices = np.where(magnitude_spectrum > threshold)

    # The actual magnitudes corresponding to the indices above the threshold
    magnitudes_above_threshold = magnitude_spectrum[indices]
    return magnitude_spectrum, indices, ft_volume_shift
