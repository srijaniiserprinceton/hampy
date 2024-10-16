from numpy import convolve as convolve1d
from scipy.signal import convolve2d

def convolve1d_w_vdf(vdf_1d, kernel_1d):
    return convolve1d(vdf_1d, kernel_1d, mode='same')

def convolve2d_w_vdf(vdf_2d, kernel_2d):
    return convolve2d(vdf_2d, kernel_2d, mode='same')