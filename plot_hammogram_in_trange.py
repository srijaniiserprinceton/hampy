import cdflib, xarray
from datetime import datetime
import numpy as np

from hampy import misc_functions as misc_fn
from hampy.plotter import misc_plots

# the desired time range to plot the hammogram
trange = ['2025-03-19T00:00:00', '2025-03-24T23:59:59']
# the directory containing the cdf files in the format of 'hamstring_YYYY-MM-DD_v02.cdf'
hamstring_dir = 'cdf'


# loading in the cdf files within the desired time range
filedates_to_read = misc_fn.get_file_indices_from_trange(trange, 'cdf')
xrs = []
for f in filedates_to_read:
    xr = cdflib.cdf_to_xarray(f'{hamstring_dir}/hamstring_{f}_v02.cdf')
    xrs.append(xr)
hamdata = xarray.concat(xrs, dim='record0')
hammertimes = np.asarray(cdflib.cdfepoch.to_datetime(hamdata['epoch'].data))

# slicing the hammertime only within the desired timerange
dt_start, dt_end = np.datetime64(trange[0]).astype(datetime), np.datetime64(trange[1]).astype(datetime)
hamtime_startidx, hamtime_endidx = np.argmin(np.abs(hammertimes-dt_start)), np.argmin(np.abs(hammertimes-dt_end))
hammertimes = hammertimes[hamtime_startidx:hamtime_endidx+1]

# loading in the pre-computed FOV dictionary (originally calculated in plot_misc_histogram.py)
FOV_dict = None #read_pickle('FOV_dict')

# plotting the within the time range
misc_plots.plot_hamogram_trange(trange, hammertimes, FOV_dict, tbin_in_sec=3000)