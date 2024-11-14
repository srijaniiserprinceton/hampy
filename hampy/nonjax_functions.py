from numpy import convolve as convolve1d
from scipy.signal import savgol_filter, convolve2d
import numpy as np

NAX = np.newaxis

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


def get_st_line(p1, p2, x):
    x1, y1 = p1
    x2, y2 = p2

    m = (y1-y2)/(x1-x2)
    b = (x1*y2 - x2*y1)/(x1-x2)

    return m * x + b

def find_Eidx_band(convmat, log_vdf_2d, theta_idx, v_hamlets, ignorenans=None):
    vx, vz = convmat.vx_plane_theta.T, convmat.vz_plane_theta.T
    vel = np.sqrt(vx**2 + vz**2)
    mask_hamlets = vel > v_hamlets

    # extracting the specific theta index
    vdf_sliced = log_vdf_2d[theta_idx]
    mask_sliced = mask_hamlets[theta_idx]

    # removing the nan entries below the mask_sliced region
    mask_Eminval = np.where(mask_sliced)[0][0]
    vdf_sliced[:mask_Eminval] = 1.0

    # ignore nans if ignorenans is not None
    if(ignorenans == None): pass
    else: vdf_sliced[ignorenans] = 1.0

    # checking for the lowest nan gap
    idxmin, idxmax = np.where(np.diff(np.isnan(vdf_sliced).astype('int')))[0][:2]

    # these are the indices which have nan
    return idxmin+1, idxmax+1

def generate_masks(log_vdf_2d, gap1, gap2):
    # making masks for core, neck and hammerhead
    coremask = np.zeros_like(log_vdf_2d, dtype='bool')
    neckmask = np.zeros_like(log_vdf_2d, dtype='bool')
    hammermask = np.zeros_like(log_vdf_2d, dtype='bool')

    # making a 2D index grid
    mgrid = np.mgrid[0:log_vdf_2d.shape[0], 0:log_vdf_2d.shape[1]]

    # drawing the straight lines
    coreedge_line = get_st_line((gap1[0], gap1[1]),
                                (gap2[0], gap2[1]), 
                                 mgrid[0][:,0])
    hammeredge_line = get_st_line((gap1[0], gap1[2]),
                                  (gap2[0], gap2[2]), 
                                   mgrid[0][:,0])

    # finding the mask below the core edge line and above the hammeredgeline
    coremask[mgrid[1]+0.5 < coreedge_line[:,NAX]] = True
    hammermask[mgrid[1]+0.5 > hammeredge_line[:,NAX]] = True

    neckmask = ~(coremask + hammermask)

    return coremask, neckmask, hammermask

def find_masks(convmat, log_vdf_2d, v_hamlet):
    # first throwing away the ones which fall below solar wind velocity
    vx, vz = convmat.vx_plane_theta.T[convmat.gap_xvals, convmat.gap_yvals],\
             convmat.vz_plane_theta.T[convmat.gap_xvals, convmat.gap_yvals]
    
    vel_gaps = np.sqrt(vx**2 + vz**2)

    # finding if any of these are NOT hamlets
    mask_hamlets = vel_gaps > v_hamlet

    # only retaining the ones which are not hamlets (first filter)
    gap_xlocs = convmat.gap_xvals[mask_hamlets]
    gap_ylocs = convmat.gap_yvals[mask_hamlets]
    orientation = convmat.orientation[mask_hamlets]
    ngaps_arr = convmat.ngaps_arr[mask_hamlets]

    # finding if any or the ngaps are 2 or more (second filter)
    if(np.sum(ngaps_arr > 1) == 0): return None, None, None

    # finding the theta index of the lowest 2-grid gap beyond v_sw
    Emin_idx = np.argmin(gap_ylocs[ngaps_arr > 1])
    theta_idx = gap_xlocs[ngaps_arr > 1][Emin_idx]
    orientation_label = orientation[ngaps_arr > 1][Emin_idx]

    # finding the cells in the theta_idx which correspond to 1 cell gap
    ignorenan_Eidx = gap_ylocs[gap_xlocs == theta_idx][ngaps_arr[gap_xlocs == theta_idx] == 1]

    # scanning this theta_idx to find the nan boundaries of the gap
    gap1_Emin_idx, gap1_Emax_idx = find_Eidx_band(convmat, log_vdf_2d * 1.0, theta_idx, v_hamlet,
                                                  ignorenans=ignorenan_Eidx)

    Emin_idx_global = np.argmin(np.abs(gap_xlocs - gap_xlocs[ngaps_arr > 1][Emin_idx]))

    # finding the theta index of the closest opposite side gap
    oppositeside_mask = ~(orientation == orientation_label)
    gap_ylocs_oppositeside = gap_ylocs[oppositeside_mask]
    gap_xlocs_oppositeside = gap_xlocs[oppositeside_mask]

    # idx_r = np.argmin(np.abs(gap_xlocs_oppositeside - gap_xlocs[Emin_idx_global]))
    idx_r = np.argmin((gap_xlocs_oppositeside - gap_xlocs[Emin_idx_global])**2 +\
                      (gap_ylocs_oppositeside - gap_ylocs[Emin_idx_global])**2)
    global_idx_r = np.argmin(np.abs(gap_xlocs - gap_xlocs_oppositeside[idx_r]))
    theta_idx_r = gap_xlocs[global_idx_r]

    # scanning this theta_idx to find the nan boundaries of the gap
    gap2_Emin_idx, gap2_Emax_idx = find_Eidx_band(convmat, log_vdf_2d * 1.0, theta_idx_r, v_hamlet)

    # finding if the starting or the ending points of gap 2 are between the limits of the orginal gap
    cond1 = gap1_Emin_idx <= gap2_Emin_idx <= gap1_Emax_idx
    cond2 = gap1_Emin_idx <= gap2_Emax_idx <= gap1_Emax_idx

    if(~(cond1 + cond2)): return None, None, None

    coremask, neckmask, hammermask = generate_masks(log_vdf_2d,
                                                    (theta_idx, gap1_Emin_idx, gap1_Emax_idx),
                                                    (theta_idx_r, gap2_Emin_idx, gap2_Emax_idx))
    
    return coremask, neckmask, hammermask

def hamslicer(convmat, log_vdf_2d, v_hamlet):
    coremask, neckmask, hammermask = find_masks(convmat, log_vdf_2d, v_hamlet)

    if(coremask is None): return None, None, None

    # setting to badmasks nans for plotting
    core = np.zeros_like(log_vdf_2d)
    neck = np.zeros_like(log_vdf_2d)
    hammer = np.zeros_like(log_vdf_2d)

    core[coremask] = log_vdf_2d[coremask]
    neck[neckmask] = log_vdf_2d[neckmask]
    hammer[hammermask] = log_vdf_2d[hammermask]

    core[~coremask] = np.nan
    neck[~neckmask] = np.nan
    hammer[~hammermask] = np.nan

    return core, neck, hammer
