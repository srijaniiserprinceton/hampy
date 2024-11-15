import numpy as np
from tqdm import tqdm
from scipy.interpolate import griddata
import matplotlib.pyplot as plt; plt.style.use('dark_background')
import cdflib, re, pickle

from hampy import load_data
from hampy import nonjax_functions as f
from hampy import comp_moments

mass_p = 0.010438870      #proton mass in units eV/c^2 where c = 299792 km/s
charge_p = 1              #proton charge in units eV

def write_pickle(x, fname):
    with open(f'{fname}.pkl', 'wb') as handle:
        pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)

def read_pickle(fname):
    with open(f'{fname}.pkl', 'rb') as handle:
        x = pickle.load(handle)
    return x

if __name__=='__main__':
    # used defined start and end times in YYYY-MM-DD/hh:mm:ss format
    tstart = '2020-02-2/18:00:00'
    tend   = '2020-02-3/18:20:00'

    # setting up the data loading process [processing will happen one day at a time]
    span_data = load_data.span(tstart, tend)

    # velocity grid for span-grid interpolation for soft-hammerhead check
    v = np.linspace(200, 1000, 31)
    t = np.linspace(-55, 55, 30)
    vv, tt = np.meshgrid(v, t, indexing='ij')

    for day_idx in range(span_data.Ndays):
        print(f'Starting analysis of Day {span_data.day_arr[day_idx]}.')
        # loading in data for the current day
        span_data.start_new_day(day_idx)

        epoch = cdflib.cdfepoch.to_datetime(span_data.VDF_dict['epoch'])

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

        vx_plane_theta = vel_plane * np.cos(np.radians(phi_plane)) * np.cos(np.radians(theta_plane))
        vy_plane_theta = vel_plane * np.sin(np.radians(phi_plane)) * np.cos(np.radians(theta_plane))
        vz_plane_theta = vel_plane *                                 np.sin(np.radians(theta_plane))

        # creating the 1D and 2D convolution matrices once for the entire day (assuming grids dont change)
        convmat = f.convolve_hammergap(vx_plane_theta, vz_plane_theta)  

        # initializing the moments calculation class
        calc_moments = comp_moments.hammer_moments(span_data.VDF_dict['phi'][0],
                                                   span_data.VDF_dict['energy'][0],
                                                   span_data.VDF_dict['theta'][0])

        # starting ham counter for the day
        hamcounter = 0
        theta_sw_vel = np.linspace(0, 2 * np.pi, 100)

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
            hammerline_sm, hammerflag, peak_idx, isedgecase, intdip_idx = f.softham_finder(pixsum)
            # hammerflag, peak_idx, isedgecase, intdip_idx = f.softham_finder(pixsum)

            # carrying out hard hammerhead tests if it detects a soft hammerhead
            if(bool(hammerflag + isedgecase)):
                # conducting 1D and 2D convolution tests to detect hammerhead-like necks in VDF
                convmat.conv1d_w_VDF(log_df_theta_span)
                convmat.conv2d_w_VDF(log_df_theta_span)
                convmat.merge_1D_2D(np.where(log_df_theta_span == np.nanmax(log_df_theta_span))[0][0])

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
                
                if(convmat.Ngaps_2D > 0):
                    day_filter_dict[hammer_epoch]['hardham_flag'] = True
                    day_filter_dict[hammer_epoch]['hardham_loc_2D'] =\
                              (vx_plane_theta.T[convmat.gap_xvals_2D, convmat.gap_yvals_2D],
                               vz_plane_theta.T[convmat.gap_xvals_2D, convmat.gap_yvals_2D])

                    # findging the core, neck and hammer
                    tSliceIndex  = np.argmin(np.abs(span_data.L3_data_fullday['EPOCH'] - span_data.VDF_dict['epoch'][time_idx]))
                    l3_data_dict = span_data.get_L3_monents_at_t(tSliceIndex) 
                    vel_sc = np.linalg.norm(l3_data_dict['VEL_INST'])    
                    Bmag = np.linalg.norm(l3_data_dict['MAGF_INST'])
                    dens = l3_data_dict['DENS']
                    valfven = 21.8 * Bmag / np.sqrt(dens)
                    vel_hamlet = vel_sc + 1.0 * valfven

                    try:
                        coremask, neckmask, hammermask, og_flag = f.find_masks(convmat, log_df_theta_span, vel_hamlet)
                        core, neck, hammer, og_flag = f.hamslicer(convmat, log_df_theta_span, vel_hamlet)
                    except:
                        coremask, neckmask, hammermask = None, None, None

                    if(coremask is None): pass
                    else:
                        # storing the moments of the core, neck and hammer
                        day_filter_dict[hammer_epoch]['core_moments'] = calc_moments.get_vdf_moments(coremask.T,
                                                                        span_data.VDF_dict['vdf'][time_idx]*1.0)
                        day_filter_dict[hammer_epoch]['neck_moments'] = calc_moments.get_vdf_moments(neckmask.T,
                                                                        span_data.VDF_dict['vdf'][time_idx]*1.0)
                        day_filter_dict[hammer_epoch]['hammer_moments'] = calc_moments.get_vdf_moments(hammermask.T,
                                                                          span_data.VDF_dict['vdf'][time_idx]*1.0)
                        day_filter_dict[hammer_epoch]['og_flag'] = og_flag

                        if(og_flag == True):
                            hamcounter += 1
                            print(f'# OG Hammerhead detected: {hamcounter}')

                        # plotting and saving
                        fig, ax = plt.subplots(1,1)
                        vmin, vmax = -1, 8
                        sw_x, sw_y = vel_hamlet * np.cos(theta_sw_vel), vel_hamlet * np.sin(theta_sw_vel)  

                        ax.pcolormesh(vx_plane_theta.T, vz_plane_theta.T, np.ma.masked_invalid(core),
                            cmap='Reds', rasterized='True', vmin=vmin, vmax=vmax)
                        ax.pcolormesh(vx_plane_theta.T, vz_plane_theta.T, np.ma.masked_invalid(neck),
                                    cmap='bone', rasterized='True', vmin=vmin, vmax=vmax)
                        ax.pcolormesh(vx_plane_theta.T, vz_plane_theta.T, np.ma.masked_invalid(hammer),
                                    cmap='hot', rasterized='True', vmin=vmin, vmax=vmax)
                        ax.scatter(vx_plane_theta[convmat.gap_xvals_1D, convmat.gap_yvals_1D],
                                    vz_plane_theta[convmat.gap_xvals_1D, convmat.gap_yvals_1D], marker='o', color='red')
                        ax.scatter(vx_plane_theta[convmat.gap_yvals_2D, convmat.gap_xvals_2D],
                                    vz_plane_theta[convmat.gap_yvals_2D, convmat.gap_xvals_2D], marker='x', color='yellow')
                        ax.plot(sw_x, sw_y, '--w')
                        ax.set_xlim(-1000,0)
                        ax.set_ylim(-500,500)
                        ax.set_aspect('equal')
                        ax.set_xlabel('$v_x$ km/s')
                        ax.set_ylabel('$v_z$ km/s')
                        ax.set_title(f'VDF SPAN-I $\\theta$-plane | OG Flag = {og_flag}')

                        plt.savefig(f'HammerFigs/day_{day_idx}_time_{time_idx}.png')
                        plt.close()
                        
                        

                if(convmat.Ngaps_1D == 0 and convmat.Ngaps_2D == 0):
                    day_filter_dict[hammer_epoch]['hardham_flag'] = False

        # writing the pkl file
        date_str = re.split('[ ]', str(epoch[0]))[0]
        write_pickle(day_filter_dict, f'hamstring_{date_str}')


