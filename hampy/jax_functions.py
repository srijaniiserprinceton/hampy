from jax.numpy import convolve as convolve1d
from jax.scipy.signal import convolve2d
import jax.numpy as jnp
from jax.lax import fori_loop, cond, dynamic_update_slice

def convolve1d_w_vdf(logvdf_1d, kernel):
    return convolve1d(logvdf_1d, kernel, mode='same')

def convolve2d_w_vdf(logvdf_2d, kernel_2d):
    return convolve2d(logvdf_2d, kernel_2d, mode='same')

def detect_gap_1d(logvdf_2d, kernel, Ngap=2):
    # we choose 8*32 elements because of SPAN instrument grid constraints
    print('New kernel shape detected')
    gap_xvals_1d = jnp.zeros(8 * 32, dtype='int32')
    gap_yvals_1d = jnp.zeros(8 * 32, dtype='int32')
    Ngaps_1D = 0
    fill_idx = 0

    log_VDF = jnp.nan_to_num(logvdf_2d * 1.0)
    minval_log_VDF = jnp.nanmin(jnp.extract(log_VDF > 0, log_VDF, size=8*32, fill_value=jnp.nan))
    mask_vdf = log_VDF <= minval_log_VDF

    def gapfinder_1d(angle_idx, gap_vals_1d):
        gap_xvals_1d, gap_yvals_1d = gap_vals_1d

        convmat1d = convolve1d_w_vdf(mask_vdf[angle_idx], kernel)
        conv_maxval = jnp.max(convmat1d)

        vel_indices = jnp.arange(32)
        gap_xvals_angleidx = jnp.extract(convmat1d == Ngap, vel_indices, size=32, fill_value=0)
        gap_yvals_angleidx = jnp.extract(convmat1d == Ngap, jnp.ones(32, dtype='int32') * angle_idx, size=32, fill_value=0)

        gap_xvals_1d = dynamic_update_slice(gap_xvals_1d, gap_xvals_angleidx, (32*angle_idx,))
        gap_yvals_1d = dynamic_update_slice(gap_yvals_1d, gap_yvals_angleidx, (32*angle_idx,))
    
        return gap_xvals_1d, gap_yvals_1d
    
    gap_xvals_1d, gap_yvals_1d = fori_loop(0, 8, gapfinder_1d, (gap_xvals_1d, gap_yvals_1d))
    return gap_xvals_1d, gap_yvals_1d
