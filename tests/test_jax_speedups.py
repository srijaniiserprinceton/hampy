import numpy as np
from hampy import jax_functions as jf
from hampy import nonjax_functions as f

if __name__=='__main__':
    a = np.random.rand(32)
    b = np.random.rand(8)
    jf.convolve1d_w_vdf(a, b)
    f.convolve1d_w_vdf(a, b)