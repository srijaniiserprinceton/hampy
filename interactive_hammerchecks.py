import numpy as np
import cdflib, bisect, sys
from datetime import datetime
import matplotlib.pyplot as plt; plt.ion(); plt.style.use('dark_background')
# import matplotlib; matplotlib.use('tkagg')
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib import ticker
from scipy.signal import savgol_filter, convolve2d
from scipy.interpolate import griddata

# custom script imports
import get_span_data as get_data

# global variables
mass_p = 0.010438870      #proton mass in units eV/c^2 where c = 299792 km/s
charge_p = 1              #proton charge in units eV

def gen_log_df(df_theta):
    log_df_theta = np.nan_to_num(np.log10(df_theta), nan=np.nan, posinf=np.nan, neginf=np.nan)

    # filtering to throw out pixels which dont have a finite value on an adjacent (not diagonal) cell
    log_df_theta_padded = np.zeros((34, 10)) + np.nan
    log_df_theta_padded[1:-1, 1:-1] = log_df_theta

    filter_mask = np.zeros_like(log_df_theta, dtype='bool')

    filter_mask += ~np.isnan(log_df_theta_padded[0:-2,1:-1])
    filter_mask += ~np.isnan(log_df_theta_padded[2:,1:-1])
    filter_mask += ~np.isnan(log_df_theta_padded[1:-1,0:-2])
    filter_mask += ~np.isnan(log_df_theta_padded[1:-1,2:])

    log_df_theta_new = log_df_theta * 1.0
    log_df_theta_new[~filter_mask] = 0.0

    return log_df_theta_new 

class convolve_hammergap:
    def __init__(self, Ngap_max = 8):
        # initializing the different kinds of gap matrices (for convolution)
        self.Ngap_max = Ngap_max
        self.gap_mat_1D = None
        self.gap_mat_2D = None
        self.create_gap_matrices()
        self.gap_xvals_1D = None
        self.gap_yvals_1D = None
        self.gap_xvals_2D = None
        self.gap_yvals_2D = None
        self.Ngaps_1D = None
        self.Ngaps_2D = None
    
    def create_gap_matrices(self):
        gap_mat_1D = {}
        gap_mat_2D = {}
        
        for Ngap in range(2, self.Ngap_max):
            gap_mat_1D[f'{Ngap}'] = np.zeros(Ngap+3) - 1
            # gap_mat_1D[f'{Ngap}'][1:(1+Ngap)] = 1.0
            gap_mat_1D[f'{Ngap}'][-(1+Ngap):-1] = 1.0

            gap_mat_2D[f'{Ngap}'] = np.zeros((2, Ngap+3)) - 1
            gap_mat_2D[f'{Ngap}'][0,-(1+Ngap):-1] = 1.0

            # creating the reversed matrix
            gap_mat_2D[f'{Ngap}_r'] = np.flip(gap_mat_2D[f'{Ngap}'], axis=0)

        self.gap_mat_1D = gap_mat_1D
        self.gap_mat_2D = gap_mat_2D

    def conv2d_w_VDF(self, log_VDF):
        # reinitializing for a new VDF analysis
        self.gap_xvals_2D = np.array([0])
        self.gap_yvals_2D = np.array([0])
        self.Ngaps_2D = 0

        log_VDF = np.nan_to_num(log_VDF * 1.0)
        mask_vdf = log_VDF <= np.nanmin(log_VDF[log_VDF > 0])

        for Ngap in range(2, self.Ngap_max):
            hammermat = self.gap_mat_2D[f'{Ngap}']
            convmat = convolve2d(mask_vdf, hammermat, mode='same')
            conv_maxval = np.max(convmat)
            if(conv_maxval == Ngap):
                self.Ngaps_2D += 1
                xarr, yarr = np.where(convmat == Ngap)
                print(xarr[0], yarr[0], self.gap_xvals_2D)
                self.gap_xvals_2D = np.append(self.gap_xvals_2D, xarr[0])
                self.gap_yvals_2D = np.append(self.gap_yvals_2D, yarr[0])

            hammermat = self.gap_mat_2D[f'{Ngap}_r']
            convmat = convolve2d(mask_vdf, hammermat, mode='same')
            conv_maxval = np.max(convmat)
            if(conv_maxval == Ngap):
                self.Ngaps_2D += 1
                xarr, yarr = np.where(convmat == Ngap)
                self.gap_xvals_2D = np.append(self.gap_xvals_2D, xarr[0]-1)
                self.gap_yvals_2D = np.append(self.gap_yvals_2D, yarr[0])

        self.gap_xvals_2D = self.gap_xvals_2D[1:]
        self.gap_yvals_2D = self.gap_yvals_2D[1:]

    def conv1d_w_VDF(self, log_VDF):
        # reinitializing for a new VDF analysis
        self.gap_xvals_1D = np.array([0])
        self.gap_yvals_1D = np.array([0])
        self.Ngaps_1D = 0

        log_VDF = np.nan_to_num(log_VDF * 1.0)
        mask_vdf = log_VDF <= np.nanmin(log_VDF[log_VDF > 0])
            
        for Ngap in range(2, self.Ngap_max):
            hammermat = self.gap_mat_1D[f'{Ngap}']
            for angle_idx in range(mask_vdf.shape[0]):
                convmat = np.convolve(mask_vdf[angle_idx], hammermat, mode='same')
                conv_maxval = np.max(convmat)
                if(conv_maxval == Ngap):
                    self.Ngaps_1D += 1
                    xarr = np.where(convmat == Ngap)
                    self.gap_xvals_1D = np.append(self.gap_xvals_1D, xarr[0])
                    self.gap_yvals_1D = np.append(self.gap_yvals_1D, np.ones_like(xarr[0], dtype='int') * angle_idx)

        self.gap_xvals_1D = self.gap_xvals_1D[1:]
        self.gap_yvals_1D = self.gap_yvals_1D[1:]

def softham_finder(hammerline, intdip_threshold=0.5):
    hammerline_sm = savgol_filter(hammerline, 7, 5)
    maxval_idx = np.argmax(hammerline_sm)
    minval_idx = np.argmin(hammerline[maxval_idx:]) + maxval_idx
    peak_idx = np.where(np.diff(hammerline_sm[maxval_idx:minval_idx])>0)[0]

    # finding the minimum between maxval_idx and secondmaxvval_idx
    if(maxval_idx > 0): 
        # adding additional test to find gap behind tallest peak
        secondmaxval_idx = np.argmax(hammerline_sm[:maxval_idx])
        secondminval_idx = maxval_idx - np.where(np.diff(hammerline_sm[:maxval_idx][::-1])>0)[0] - 1

    else: 
        secondmaxval_idx = np.array([0])
        secondminval_idx = np.array([1])

    # throwing away the values below the very first peak which are typically below 0.1
    secondminval_idx = secondminval_idx[hammerline_sm[secondminval_idx] > 0.1]
    secondminval_idx = np.append(secondminval_idx, maxval_idx)  # this is to avoid the case where the array is empty

    # checking if it is less than a certain percent of the maxval peak
    edgecase_hammer = hammerline_sm[maxval_idx] / np.nanmin(hammerline_sm[secondminval_idx]) > (1/intdip_threshold)

    return hammerline_sm, np.sum(np.diff(hammerline_sm[maxval_idx:minval_idx])>0)>0, peak_idx + maxval_idx,\
           edgecase_hammer, secondminval_idx

if __name__=='__main__':
    # user defined date and time
    year, month, date = 2020, 2, 25
    hour, minute, second = 18, 10, 1

    # timestamp for Verniero et al 2022 hammerhead
    # year, month, date = 2020, 1, 29
    # hour, minute, second = 18, 10, 1

    # creating the 1D and 3D convolution matrices once for the entire runtime
    convmat = convolve_hammergap()

    # converting to datetime format to extract time index
    user_datetime = datetime(year, month, date)
    timeSlice  = np.datetime64(datetime(year, month, date, hour, minute, second))

    # loading the data (downloading the file if necessary)
    cdf_VDfile = get_data.download_VDF_file(user_datetime)

    # convert time
    epoch = cdflib.cdfepoch.to_datetime(cdf_VDfile['EPOCH'])
    # find index for desired timeslice
    tSliceIndex  = bisect.bisect_left(epoch, timeSlice)
    
    # getting the VDF dictionary at the desired timestamp
    vdf_dict = get_data.get_VDFdict_at_t(cdf_VDfile, tSliceIndex)

    # creating the first snapshot of the interactive figure
    fig, ax = plt.subplots(1, 3, figsize=(15,10))

    # leaving room at the bottom for interactive slider
    fig.subplots_adjust(bottom=0.25)

    fill_color = 'black'
    cmap = 'inferno'

    # DATA FORMAT OF THE VDF: phi is along dimension 0, while theta is along 2
    # choosing a cut through phi for plotting
    phi_cut=0 

    phi_plane = vdf_dict['phi'][phi_cut,:,:]
    theta_plane = vdf_dict['theta'][phi_cut,:,:]
    energy_plane = vdf_dict['energy'][phi_cut,:,:]
    vel_plane = np.sqrt(2 * charge_p * energy_plane / mass_p)

    # VDF as a function of energy and theta (phi axis is summed over)
    df_theta = np.nansum(vdf_dict['vdf'], axis=0)

    vx_plane_theta = vel_plane * np.cos(np.radians(phi_plane)) * np.cos(np.radians(theta_plane))
    vy_plane_theta = vel_plane * np.sin(np.radians(phi_plane)) * np.cos(np.radians(theta_plane))
    vz_plane_theta = vel_plane *                                 np.sin(np.radians(theta_plane))

    vmin, vmax = -1, 8
    im0 = ax[0].contourf(vx_plane_theta, vz_plane_theta, np.log10(df_theta),
                           cmap=cmap, rasterized='True', vmin=vmin, vmax=vmax)

    ax[0].set_xlim(-1000,0)
    ax[0].set_ylim(-500,500)
    ax[0].set_xlabel('$v_x$ km/s')
    ax[0].set_ylabel('$v_z$ km/s')
    ax[0].set_title('VDF SPAN-I $\\theta$-plane')
    ax[0].set_aspect('equal')

    # choosing a cut through theta for plotting 
    theta_cut = 1

    phi_plane = vdf_dict['phi'][:,:,theta_cut]
    theta_plane = vdf_dict['theta'][:,:,theta_cut]
    energy_plane = vdf_dict['energy'][:,:,theta_cut]
    vel_plane = np.sqrt(2 * charge_p * energy_plane / mass_p)

    # VDF as a function of energy and phi. theta dimension is summed over
    df_phi = np.nansum(vdf_dict['vdf'], axis=2)

    vx_plane_phi = vel_plane * np.cos(np.radians(phi_plane)) * np.cos(np.radians(theta_plane))
    vy_plane_phi = vel_plane * np.sin(np.radians(phi_plane)) * np.cos(np.radians(theta_plane))
    vz_plane_phi = vel_plane *                                 np.sin(np.radians(theta_plane))

    im1 = ax[1].contourf(np.transpose(vx_plane_phi), np.transpose(vy_plane_phi), np.log10(np.transpose(df_phi)),
                           cmap=cmap, rasterized='True', vmin=vmin, vmax=vmax)

    ax[1].set_xlim(-1000,0)
    ax[1].set_ylim(-200,500)
    ax[1].set_xlabel('$v_x$ km/s')
    ax[1].set_ylabel('$v_y$ km/s')
    ax[1].set_title('VDF SPAN-I $\\phi$-plane')
    ax[1].set_aspect('equal')


    # converting the VDF as a function of energy and theta to log space
    log_df_theta = gen_log_df(df_theta)

    # velocity grid for interpolation
    v = np.linspace(200, 1000, 31)
    t = np.linspace(90, 180, 30)
    vv, tt = np.meshgrid(v, t, indexing='ij')

    # interpolating the log df for better resolution of the expected dip of hammerhead
    log_df_theta_interp = griddata((vel_plane.flatten(), phi_plane.flatten()), log_df_theta.T.flatten(), (vv,tt))
    log_df_theta = np.nan_to_num(log_df_theta_interp)

    im2 =  ax[2].pcolormesh(vv, tt, log_df_theta, cmap='binary_r')
    ax[2].set_xlim([0, 1000])
    ax[2].set_ylim([85, 185])

    # making twin axis to plot the sum in theta for each velocity bin
    twax = ax[2].twinx()
    # summing the pixels in theta and scaling it from 0 to 1
    pixsum = np.nansum(log_df_theta, axis=1)
    pixsum = pixsum / np.nanmax(pixsum)
    # unsmoothened pixsum line
    [pixsum_line] = twax.plot(v, pixsum, 'r')
    # detecting the presence of soft hammerhead
    hammerline_sm, hammerflag, peak_idx, isedgecase, intdip_idx = softham_finder(pixsum)
    # smoothened pixsum line (on which dipfinder works)
    [pixsum_line_sm] = twax.plot(v, hammerline_sm, '--r')
    # plotting points of intermediate dips (where we have a neck before the maximum pixsum region) [for very large beams]
    [intdip_point] = twax.plot(v[intdip_idx], hammerline_sm[intdip_idx], 'o', color='orange')
    [dip_point] = twax.plot(v[peak_idx], hammerline_sm[peak_idx],'og')
    
    # adding flag in the title if it has found a soft hammerhead 
    ax[2].set_title(f'Hammerhead = {(bool(hammerflag + isedgecase))}')

    plt.subplots_adjust(hspace=0.5, wspace=0.3, left=0.1, right=0.95, top=0.95, bottom=0.1)

    cbar = fig.colorbar(im1, orientation='horizontal')
    cbar.set_label(f'f $(cm^2 \\ s \\ sr \\ eV)^{-1}$')

    #----------------- interactive plotting --------------------#
    # Define an axes area and draw a slider in it
    axis_color = 'white'
    time_slider_ax  = fig.add_axes([0.4, 0.12, 0.2, 0.03], facecolor=axis_color)
    time_slider = Slider(time_slider_ax, r'$\mu_{\phi}$', tSliceIndex-500, tSliceIndex+500, valinit=tSliceIndex)

    # Define an action for modifying the line when any slider's value changes
    def sliders_on_changed(val):
        # getting the new VDF dictionary for the new time
        vdf_dict = get_data.get_VDFdict_at_t(cdf_VDfile, int(time_slider.val))

        # clearing out the previous subplots
        ax[0].clear()
        ax[1].clear()

        df_theta = np.nansum(vdf_dict['vdf'], axis=0)
        ax[0].contourf(vx_plane_theta, vz_plane_theta, np.log10(df_theta),
                           cmap=cmap, rasterized='True', vmin=vmin, vmax=vmax)
        ax[0].set_xlim(-1000,0)
        ax[0].set_ylim(-500,500)
        ax[0].set_aspect('equal')
        ax[0].set_xlabel('$v_x$ km/s')
        ax[0].set_ylabel('$v_z$ km/s')
        ax[0].set_title('VDF SPAN-I $\\theta$-plane')

        df_phi = np.nansum(vdf_dict['vdf'], axis=2)
        ax[1].contourf(np.transpose(vx_plane_phi), np.transpose(vy_plane_phi), np.log10(np.transpose(df_phi)),
                           cmap=cmap, rasterized='True', vmin=vmin, vmax=vmax)
        ax[1].set_xlim(-1000,0)
        ax[1].set_ylim(-200,500)
        ax[1].set_aspect('equal')
        ax[1].set_xlabel('$v_x$ km/s')
        ax[1].set_ylabel('$v_y$ km/s')
        ax[1].set_title('VDF SPAN-I $\\phi$-plane')

        log_df_theta = gen_log_df(df_theta)
        # interpolating the log_df_theta
        log_df_theta_span = log_df_theta.T * 1.0   # creating a copy of the true SPAN data for convolution tests
        log_df_theta_interp = griddata((vel_plane.flatten(), phi_plane.flatten()), log_df_theta.T.flatten(), (vv,tt))
        log_df_theta = np.nan_to_num(log_df_theta_interp)
        im2.set_array(log_df_theta)

        pixsum = np.nansum(log_df_theta, axis=1)
        pixsum = pixsum / np.nanmax(pixsum)
        pixsum_line.set_ydata(pixsum)

        # detecting soft hammerhead
        hammerline_sm, hammerflag, peak_idx, isedgecase, intdip_idx = softham_finder(pixsum)
        pixsum_line_sm.set_ydata(hammerline_sm)
        intdip_point.set_data([v[intdip_idx]], [hammerline_sm[intdip_idx]])
        dip_point.set_data(v[peak_idx], hammerline_sm[peak_idx])
        
        # carrying out hard hammerhead tests if it detects a soft hammerhead
        if(bool(hammerflag + isedgecase)):
            # conducting 1D and 2D convolution tests to detect hammerhead-like necks in VDF
            convmat.conv1d_w_VDF(log_df_theta_span)
            convmat.conv2d_w_VDF(log_df_theta_span)

            # using different markers for 1D and 2D convolution test passes
            if(convmat.Ngaps_1D > 0):
                ax[0].scatter(vx_plane_theta[convmat.gap_xvals_1D, convmat.gap_yvals_1D],
                              vz_plane_theta[convmat.gap_xvals_1D, convmat.gap_yvals_1D], marker='o', color='red')
            if(convmat.Ngaps_2D > 0):
                ax[0].scatter(vx_plane_theta[convmat.gap_yvals_2D, convmat.gap_xvals_2D],
                              vz_plane_theta[convmat.gap_yvals_2D, convmat.gap_xvals_2D], marker='x', color='yellow')
                ax[2].set_title(f'Soft-Ham = True, Hard-Ham = True')
            if(convmat.Ngaps_1D == 0 and convmat.Ngaps_2D == 0):
                ax[2].set_title(f'Soft-Ham = True, Hard-Ham = False')
        
        else: 
            ax[2].set_title(f'Hammerhead = False')

        fig.canvas.draw_idle()

    time_slider.on_changed(sliders_on_changed)

    # Add a button for resetting the parameters
    reset_button_ax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    reset_button = Button(reset_button_ax, 'Reset', color='black', hovercolor='0.1')
    def reset_button_on_clicked(mouse_event):
        time_slider.reset()

    reset_button.on_clicked(reset_button_on_clicked)