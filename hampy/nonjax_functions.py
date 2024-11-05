from numpy import convolve as convolve1d
from scipy.signal import savgol_filter, convolve2d
import numpy as np

def convolve1d_w_vdf(vdf_1d, kernel_1d):
    return convolve1d(vdf_1d, kernel_1d, mode='same')

def convolve2d_w_vdf(vdf_2d, kernel_2d):
    return convolve2d(vdf_2d, kernel_2d, mode='same')

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

    return np.sum(np.diff(hammerline_sm[maxval_idx:minval_idx])>0)>0, peak_idx + maxval_idx,\
           edgecase_hammer, secondminval_idx

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

    # def find_1d_gap()
