import numpy as np
import cdflib
import astropy.constants as c
from scipy.signal import savgol_filter

def Tani_beta_instability_relations(beta_arr):
    # making a beta array using extreme values of beta
    beta = np.logspace(np.log10(beta_arr.min()), np.log10(beta_arr.max()), 100)

    # using values from Hellinger et al 2006 (https://doi.org/10.1029/2006GL025925)
    a = [0.43, 0.77, -0.47, -1.4]
    b = [0.42, 0.76, 0.53, 1.0]
    beta0 = [-0.0004, -0.016, 0.59, -0.11]

    T_ani = np.zeros((4, len(beta)))

    for i in range(4):
        T_ani[i] = 1 + a[i] / np.power((beta - beta0[i]), b[i])

    return beta, T_ani

def convert_to_tensor(input_array):
    N_time = input_array.shape[0]
    tensors = np.zeros((N_time, 3, 3))

    for i in range(N_time):
        T_input = input_array[i]
        tensors[i] = np.array([[T_input[0], T_input[3], T_input[4]],
                               [T_input[3], T_input[1], T_input[5]],
                               [T_input[4], T_input[5], T_input[2]]])

    return tensors

def define_time_series_rot_vector(bvec):
    Ntime = bvec.shape[0]

    # Get the unit vector
    bunit = bvec / np.linalg.norm(bvec, axis=1)[:,None]

    # Get the 
    temp = np.stack([np.zeros(Ntime), np.ones(Ntime), np.zeros(Ntime)]).T

    # Define a vector normal to the magnetic field unit vector
    P = np.cross(bunit, temp)/np.linalg.norm(np.cross(bunit, temp), axis=1)[:,None]

    # Complete the matrix
    Q = np.cross(bunit, P)/np.linalg.norm(np.cross(bunit, P), axis=1)[:,None]

    # Define the rotation matrix
    R = np.stack([bunit, P, Q], axis=1)

    return(R)

def rotate_the_magnetic_field_vector(t_tensor, rot):
    
    Ntime = len(t_tensor)

    T_rot = []
    for i in range(Ntime):
        A = np.matmul(t_tensor[i], np.linalg.inv(rot[i]))
        T = np.matmul(rot[i], A)
        T_rot.append(T)

    T_rot = np.array(T_rot)

    return(T_rot)

def find_Tanisotropy(T_tensor, B, spi_epoch, hammerepoch):
    T_XX,T_YY,T_ZZ,T_XY,T_XZ,T_YZ = np.asarray(T_tensor).T

    #Access tensor elements -- The temperature is an array of 9 elements. We want to find out how much temp is aligned parallel or perp to the mag field.
    T_YX = T_XY
    T_ZX = T_XZ
    T_ZY = T_YZ

    #Access magnetic field in span-I coordinates
    # B_spi = get_data('psp_spi_MAGF_INST')
    B_X = B[:,0]
    B_Y = B[:,1]
    B_Z = B[:,2]
    B_mag_XYZ = np.sqrt(B_X**2 + B_Y**2 + B_Z**2)

    #Project Tensor onto B field, find perpendicular and parallel components
    T_parallel=[]
    T_perpendicular=[]
    Anisotropy=[]

    for hamepoch_idx in range(len(T_XX)):  #Calculates Tperp and Tpar from the projection of the magnetic field vector
        i = np.argmin(np.abs(spi_epoch - hammerepoch[hamepoch_idx]))
        Sum_1=B_X[i]*B_X[i]*T_XX[hamepoch_idx]
        Sum_2=B_X[i]*B_Y[i]*T_XY[hamepoch_idx]
        Sum_3=B_X[i]*B_Z[i]*T_XZ[hamepoch_idx]
        Sum_4=B_Y[i]*B_X[i]*T_YX[hamepoch_idx]
        Sum_5=B_Y[i]*B_Y[i]*T_YY[hamepoch_idx]
        Sum_6=B_Y[i]*B_Z[i]*T_YZ[hamepoch_idx]
        Sum_7=B_Z[i]*B_X[i]*T_ZX[hamepoch_idx]
        Sum_8=B_Z[i]*B_Y[i]*T_ZY[hamepoch_idx]
        Sum_9=B_Z[i]*B_Z[i]*T_ZZ[hamepoch_idx]    
        T_para=((Sum_1+Sum_2+Sum_3+Sum_4+Sum_5+Sum_6+Sum_7+Sum_8+Sum_9)/(B_mag_XYZ[i])**2)
        Trace_Temp=(T_XX[hamepoch_idx]+T_YY[hamepoch_idx]+T_ZZ[hamepoch_idx])
        T_perp=(Trace_Temp-T_para)/2.0
        T_parallel.append((Sum_1+Sum_2+Sum_3+Sum_4+Sum_5+Sum_6+Sum_7+Sum_8+Sum_9)/(B_mag_XYZ[i])**2)
        T_perpendicular.append(T_perp)
        Anisotropy.append(T_perp/T_para)

    return np.asarray(T_perpendicular), np.asarray(T_parallel), np.asarray(Anisotropy)


def extract_params(hammerdict, span_data, og_only=True, min_hammer_cells=10):
    # to store density moments
    core_density = []
    neck_density = []
    hammer_density = []
    total_hampy_density = []
    span_density = []

    # velocity moments
    U = {}

    # finding the times in UTC
    epoch_arr = np.asarray(list(hammerdict.keys()))
    dt_arr = []

    # distance from sun
    dist_rsun = []

    # to store drift velocity
    neck_vdrift = []
    hammer_vdrift = []

    # FIELDS measured magnetic field for computing Valfven later
    span_Bmag = []

    # to store the different temperature tensors for the different components of the VDF
    T_tensor = {}

    components = ['core', 'neck', 'hammer']

    # finding the magnetic fields for rotating the temperature tensor
    B_vec_inst = []

    for component in components:
        U[f'{component}'] = {}
        U[f'{component}']['Ux'] = []
        U[f'{component}']['Uy'] = []
        U[f'{component}']['Uz'] = []
        T_tensor[f'{component}'] = []

    for epoch_idx, epoch in enumerate(hammerdict.keys()):
        try:
            if(hammerdict[epoch]['Ncells_hammer'] < min_hammer_cells): continue
            if(og_only): 
                if(not hammerdict[epoch]['og_flag']): continue

            for component in components:
                # exctracting the velocities 
                U[f'{component}']['Ux'].append(hammerdict[epoch][f'{component}_moments']['Ux'])
                U[f'{component}']['Uy'].append(hammerdict[epoch][f'{component}_moments']['Uy'])
                U[f'{component}']['Uz'].append(hammerdict[epoch][f'{component}_moments']['Uz'])

                # extracting temperatures
                Txx = hammerdict[epoch][f'{component}_moments']['Txx']
                Txy = hammerdict[epoch][f'{component}_moments']['Txy']
                Txz = hammerdict[epoch][f'{component}_moments']['Txz']
                Tyy = hammerdict[epoch][f'{component}_moments']['Tyy']
                Tyz = hammerdict[epoch][f'{component}_moments']['Tyz']
                Tzz = hammerdict[epoch][f'{component}_moments']['Tzz']

                T_tensor[f'{component}'].append((np.asarray([Txx, Tyy, Tzz, Txy, Txz, Tyz])))                

            # extracting densities
            core_density.append(hammerdict[epoch]['core_moments']['n'])
            neck_density.append(hammerdict[epoch]['neck_moments']['n'])
            hammer_density.append(hammerdict[epoch]['hammer_moments']['n'])
            total_hampy_density.append(hammerdict[epoch]['core_moments']['n'] + 
                                       hammerdict[epoch]['neck_moments']['n'] + 
                                       hammerdict[epoch]['hammer_moments']['n'])

            dt_arr.append(epoch_arr[epoch_idx])
            # finding the closest epoch in the magnetic field data
            Bepoch = np.argmin(np.abs(span_data.L3_data_fullday['epoch'] - epoch))
            dist_rsun.append(span_data.L3_data_fullday['SUN_DIST'][Bepoch])
            B_vec_inst.append(span_data.L3_data_fullday['MAGF_INST'][Bepoch])
            span_density.append(span_data.L3_data_fullday['DENS'][Bepoch])
            neck_vdrift.append(hammerdict[epoch]['neck_moments']['Ux'] - hammerdict[epoch]['core_moments']['Ux'])
            hammer_vdrift.append(hammerdict[epoch]['hammer_moments']['Ux'] - hammerdict[epoch]['core_moments']['Ux'])

        except:
            pass

    # making array-like from list
    core_density = np.asarray(core_density)
    neck_density = np.asarray(neck_density)
    hammer_density = np.asarray(hammer_density)
    total_hampy_density = np.asarray(total_hampy_density)
    span_density = np.asarray(span_density)

    span_density_all = savgol_filter(np.asarray(span_data.L3_data_fullday['DENS']), 250, 3)
    span_time_all = cdflib.cdfepoch.to_datetime(np.asarray(span_data.L3_data_fullday['epoch']))

    for component in components:
        U[f'{component}']['Ux'] = np.asarray(U[f'{component}']['Ux'])
        U[f'{component}']['Uy'] = np.asarray(U[f'{component}']['Uy'])
        U[f'{component}']['Uz'] = np.asarray(U[f'{component}']['Uz'])
    T_tensor['core'] = np.asarray(T_tensor['core'])
    T_tensor['neck'] = np.asarray(T_tensor['neck'])
    T_tensor['hammer'] = np.asarray(T_tensor['hammer'])
    dt_arr = cdflib.cdfepoch.to_datetime(np.asarray(dt_arr))
    dist_rsun = np.asarray(dist_rsun) / c.R_sun.to('km').value
    B_vec_inst = np.asarray(B_vec_inst)
    neck_vdrift = np.asarray(neck_vdrift)
    hammer_vdrift = np.asarray(hammer_vdrift)
    span_Bmag = np.linalg.norm(B_vec_inst, axis=1)

    # computing the parallel and perpendicular temperatures
    T_perp = {}
    T_parallel = {}
    T_ani = {}

    T_perp['core'], T_parallel['core'], T_ani['core'] = find_Tanisotropy(T_tensor['core'], span_data.L3_data_fullday['MAGF_INST'],
                                                                         span_data.L3_data_fullday['epoch'], epoch_arr)
    T_perp['neck'], T_parallel['neck'], T_ani['neck'] = find_Tanisotropy(T_tensor['neck'], span_data.L3_data_fullday['MAGF_INST'],\
                                                                         span_data.L3_data_fullday['epoch'], epoch_arr)
    T_perp['hammer'], T_parallel['hammer'], T_ani['hammer'] = find_Tanisotropy(T_tensor['hammer'], span_data.L3_data_fullday['MAGF_INST'],\
                                                                               span_data.L3_data_fullday['epoch'], epoch_arr)

    return core_density, neck_density, hammer_density, total_hampy_density, span_density, T_perp, T_parallel, T_ani,\
           dt_arr, B_vec_inst, neck_vdrift, hammer_vdrift, span_Bmag, U, dist_rsun, span_density_all, span_time_all