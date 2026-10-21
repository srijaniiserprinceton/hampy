import numpy as np
import os, cdflib, re, pickle, argparse
from tqdm import tqdm
from scipy.interpolate import griddata
import matplotlib.pyplot as plt; plt.style.use('dark_background')

from hampy import load_data
from hampy import nonjax_functions as f
from hampy import comp_moments
from hampy.plotter import plot_tools
from hampy import hamstring_cdf

mass_p = 0.010438870      #proton mass in units eV/c^2 where c = 299792 km/s
charge_p = 1              #proton charge in units eV

def write_pickle(x, fname):
    # make sure the target directory (e.g. pkl/) exists
    os.makedirs(os.path.dirname(f'{fname}.pkl') or '.', exist_ok=True)
    with open(f'{fname}.pkl', 'wb') as handle:
        pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)

def read_pickle(fname):
    with open(f'{fname}.pkl', 'rb') as handle:
        x = pickle.load(handle)
    return x

def main(tstart, tend, save_daywise_pkl=False, save_cdf=False, pkl_dir='pkl', cdf_dir='cdf', plot=False):
    # setting up the data loading process [processing will happen one day at a time]
    span_data = load_data.span(tstart, tend)

    # velocity grid for span-grid interpolation for soft-hammerhead check
    v = np.linspace(200, 1000, 31)
    t = np.linspace(-55, 55, 30)
    vv, tt = np.meshgrid(v, t, indexing='ij')

    # dictionary for this day (to be stored as a .pkl file for each day's filtering)
    global_day_filter_dict = {}

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
        phi_cut=1

        phi_plane = span_data.VDF_dict['phi'][0,phi_cut,:,:]
        theta_plane = span_data.VDF_dict['theta'][0,phi_cut,:,:]
        energy_plane = span_data.VDF_dict['energy'][0,phi_cut,:,:]
        vel_plane = np.sqrt(2 * charge_p * energy_plane / mass_p)

        phi_plane_nan = np.sum(np.isnan(phi_plane))
        theta_plane_nan = np.sum(np.isnan(theta_plane))
        energy_plane_nan = np.sum(np.isnan(energy_plane))
        vel_plane_nan = np.sum(np.isnan(vel_plane))

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

            if(np.nansum(~np.isnan(pixsum)) < 1): continue

            # detecting soft hammerhead
            hammerline_sm, hammerflag, peak_idx, isedgecase, intdip_idx = f.softham_finder(pixsum)

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
                        coremask, neckmask, hammermask, og_flag = None, None, None, False

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

                        # finding the number of cells in the hammerhead that are not zero or nan
                        day_filter_dict[hammer_epoch]['Ncells_hammer'] = np.sum(~np.isnan(hammer))


                        if(og_flag == True):
                            hamcounter += 1
                            print(f'# OG Hammerhead detected: {hamcounter}')


                        # plotting and saving (only when plot=True)
                        if plot:
                            os.makedirs('HammerFigs', exist_ok=True)
                            fig, ax = plt.subplots(1,1)
                            vmin, vmax = -1, 8
                            sw_x, sw_y = vel_hamlet * np.cos(theta_sw_vel), vel_hamlet * np.sin(theta_sw_vel)

                            vmin, vmax = np.nanmin(log_df_theta_span), np.nanmax(log_df_theta_span)

                            ax.pcolormesh(vx_plane_theta.T, vz_plane_theta.T, np.ma.masked_invalid(core),
                                        cmap='YlGn', rasterized='True', vmin=vmin, vmax=vmax)
                            ax.pcolormesh(vx_plane_theta.T, vz_plane_theta.T, np.ma.masked_invalid(neck),
                                        cmap='hot', rasterized='True', vmin=vmin, vmax=vmax)
                            ax.pcolormesh(vx_plane_theta.T, vz_plane_theta.T, np.ma.masked_invalid(hammer),
                                        cmap='pink', rasterized='True', vmin=vmin, vmax=vmax)
                            ax.scatter(vx_plane_theta[convmat.gap_xvals_1D, convmat.gap_yvals_1D],
                                        vz_plane_theta[convmat.gap_xvals_1D, convmat.gap_yvals_1D], marker='o', color='red')
                            ax.scatter(vx_plane_theta[convmat.gap_yvals_2D, convmat.gap_xvals_2D],
                                        vz_plane_theta[convmat.gap_yvals_2D, convmat.gap_xvals_2D], marker='x', color='yellow')
                            ax.plot(sw_x, sw_y, '--w')
                            ax.set_xlim(-2000,0)
                            ax.set_ylim(-1000,1000)
                            ax.set_aspect('equal')
                            ax.set_xlabel('$v_x$ km/s')
                            ax.set_ylabel('$v_z$ km/s')
                            time_str = re.split('[.]', str(epoch[int(time_idx)]))[0]
                            ax.set_title(f'{time_str} | OG Flag = {og_flag}')

                            plt.savefig(f'HammerFigs/{time_str}.png')
                            plt.close()


                if(convmat.Ngaps_1D == 0 and convmat.Ngaps_2D == 0):
                    day_filter_dict[hammer_epoch]['hardham_flag'] = False

        # writing the pkl file (day-only name, e.g. hamstring_2026-09-03.pkl)
        date_str = str(epoch[0])[:10]
        if save_daywise_pkl or save_cdf:
            write_pickle(day_filter_dict, f'{pkl_dir}/hamstring_{date_str}')
        if save_cdf:
            # standalone module that turns this day's pkl into a CDF (NaN-filled if no hams)
            hamstring_cdf.create_hamstring_cdf(date_str, pkl_dir=pkl_dir, cdf_dir=cdf_dir,
                                               credentials=span_data.credentials)

        global_day_filter_dict.update(day_filter_dict)

    return global_day_filter_dict

if __name__=='__main__':
    p = argparse.ArgumentParser(
        description='Run the hammerhead finder over one or more time windows. '
                    'Times are YYYY-MM-DD/hh:mm:ss. Writes pkl/ (always) and cdf/ (unless --no-cdf).')
    p.add_argument('--start', nargs='+', required=True,
                   help='start time(s), e.g. 2022-09-06/18:30:00')
    p.add_argument('--end', nargs='+', required=True,
                   help='end time(s), e.g. 2022-09-06/19:59:59 (one per --start)')
    p.add_argument('--no-cdf', action='store_true',
                   help='skip CDF creation and only write the pickles')
    p.add_argument('--plot', action='store_true',
                   help='save per-hammerhead VDF figures to HammerFigs/ (default: off)')
    p.add_argument('--pkl-dir', default='pkl', help='output dir for pickles (default: pkl)')
    p.add_argument('--cdf-dir', default='cdf', help='output dir for CDFs (default: cdf)')
    args = p.parse_args()

    tstart_arr, tend_arr = args.start, args.end
    if len(tstart_arr) != len(tend_arr):
        p.error('give the same number of --start and --end times')
    save_cdf = not args.no_cdf

    for window_idx in range(len(tstart_arr)):
        global_day_filter_dict = main(tstart_arr[window_idx], tend_arr[window_idx],
                                      save_daywise_pkl=True, save_cdf=save_cdf,
                                      pkl_dir=args.pkl_dir, cdf_dir=args.cdf_dir,
                                      plot=args.plot)

        # removing the epochs which have no hammerhead splitting
        purge_keys = []
        for key in global_day_filter_dict.keys():
            if('og_flag' in global_day_filter_dict[key]): continue
            purge_keys.append(key)

        for key in purge_keys:
            del global_day_filter_dict[key]

        # saving this as the grand .pkl file for a specific period
        composed_filename = f'{tstart_arr[window_idx].replace("/","T")}-{tend_arr[window_idx].replace("/","T")}'
        write_pickle(global_day_filter_dict, f'{args.pkl_dir}/hamstring_{composed_filename}')
