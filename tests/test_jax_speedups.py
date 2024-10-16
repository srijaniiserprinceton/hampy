import numpy as np
import jax.numpy as jnp
import jax
from hampy import jax_functions as jf
from hampy import nonjax_functions as f

if __name__=='__main__':
    a = np.random.rand(32)
    b = np.random.rand(8)
    a_ = jnp.arange(32)
    b_ = jnp.arange(8)
    jfconv1d_jit = jax.jit(jf.convolve1d_w_vdf)
    # run to compile
    jfconv1d_jit(a_,b_)
    
    # 2D convolution
    a2d = np.random.rand(32,8)
    b2d = np.random.rand(5,2)

    a2d_ = jnp.ones((32,8))
    b2d_ = jnp.ones((5,2))

    jfconv2d_jit = jax.jit(jf.convolve2d_w_vdf)
    # run to compile
    jfconv2d_jit(a2d_,b2d_)
