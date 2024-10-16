from numpy import convolve as convolve1D

def convolve1d_w_vdf(vdf_1d, kernel):
    return convolve1D(vdf_1d, kernel, mode='same')