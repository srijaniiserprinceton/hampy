import numpy as np
from tqdm import tqdm
from scipy.interpolate import griddata

from hampy import load_data
from hampy import nonjax_functions as f

mass_p = 0.010438870      #proton mass in units eV/c^2 where c = 299792 km/s
charge_p = 1              #proton charge in units eV

if __name__=='__main__':
    # used defined start and end times in YYYY-MM-DD/hh:mm:ss format
    tstart = '2020-01-29/18:00:00'
    tend   = '2020-01-31/18:20:00'

    # setting up the data loading process [processing will happen one day at a time]
    span_data = load_data.span(tstart, tend)

    # velocity grid for span-grid interpolation for soft-hammerhead check
    v = np.linspace(200, 1000, 31)
    t = np.linspace(-55, 55, 30)
    vv, tt = np.meshgrid(v, t, indexing='ij')

    # creating the 1D and 3D convolution matrices once for the entire runtime
    convmat = f.convolve_hammergap()

    for day_idx in range(span_data.Ndays):
        print(f'Starting analysis of Day {span_data.day_arr[day_idx]}.')
        # loading in data for the current day
        span_data.start_new_day(day_idx)

        # looping over all timestamps in the day
        Ntimes = len(span_data.VDF_dict['epoch'])

        # dictionary for this day (to be stored as a .pkl file for each day's filtering)
        day_filter_dict = {}

        # DATA FORMAT OF THE VDF: phi is along dimension 0, while theta is along 2
        # choosing a cut through theta for interpolating
        phi_cut=0 

        phi_plane = span_data.VDF_dict['phi'][0,phi_cut,:,:]
        theta_plane = span_data.VDF_dict['theta'][0,phi_cut,:,:]
        energy_plane = span_data.VDF_dict['energy'][0,phi_cut,:,:]
        vel_plane = np.sqrt(2 * charge_p * energy_plane / mass_p)

        for time_idx in tqdm(range(Ntimes)):
            # VDF as a function of energy and theta (phi axis is summed over)
            df_theta = np.nansum(span_data.VDF_dict['vdf'][time_idx], axis=0)
            log_df_theta = f.gen_log_df(df_theta)

            vx_plane_theta = vel_plane * np.cos(np.radians(phi_plane)) * np.cos(np.radians(theta_plane))
            vy_plane_theta = vel_plane * np.sin(np.radians(phi_plane)) * np.cos(np.radians(theta_plane))
            vz_plane_theta = vel_plane *                                 np.sin(np.radians(theta_plane))

            # interpolating the log_df_theta
            log_df_theta_span = log_df_theta.T * 1.0   # creating a copy of the true SPAN data for convolution tests
            log_df_theta_interp = griddata((vel_plane.flatten(), theta_plane.flatten()), log_df_theta.flatten(), (vv, tt))
            log_df_theta = np.nan_to_num(log_df_theta_interp)
            
            # finding the 1D line for soft-hammerhead check
            pixsum = np.nansum(log_df_theta, axis=1)
            pixsum = pixsum / np.nanmax(pixsum)

            # detecting soft hammerhead
            hammerflag, peak_idx, isedgecase, intdip_idx = f.softham_finder(pixsum)

            # carrying out hard hammerhead tests if it detects a soft hammerhead
            if(bool(hammerflag + isedgecase)):
                # conducting 1D and 2D convolution tests to detect hammerhead-like necks in VDF
                convmat.conv1d_w_VDF(log_df_theta_span)
                convmat.conv2d_w_VDF(log_df_theta_span)

                hammer_epoch = span_data.VDF_dict['epoch'][time_idx]
                day_filter_dict[hammer_epoch] = {}

                day_filter_dict[hammer_epoch]['softham_dips'] = peak_idx
                
                if(isedgecase):
                    day_filter_dict[hammer_epoch]['softham_intdips'] = intdip_idx
                else:
                    day_filter_dict[hammer_epoch]['softham_intdips'] = None
                
                if(convmat.Ngaps_1D > 0):
                    day_filter_dict[hammer_epoch]['hardham_flag'] = True
                    day_filter_dict[hammer_epoch]['hardham_loc_1D'] =\
                              (vx_plane_theta[convmat.gap_xvals_1D, convmat.gap_yvals_1D],
                               vz_plane_theta[convmat.gap_xvals_1D, convmat.gap_yvals_1D])
                
                elif(convmat.Ngaps_2D > 0):
                    day_filter_dict[hammer_epoch]['hardham_flag'] = True
                    day_filter_dict[hammer_epoch]['hardham_loc_2D'] =\
                              (vx_plane_theta[convmat.gap_xvals_2D, convmat.gap_yvals_2D],
                               vz_plane_theta[convmat.gap_xvals_2D, convmat.gap_yvals_2D])

                else:
                    day_filter_dict[hammer_epoch]['hardham_flag'] = False


