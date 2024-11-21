import numpy as np

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

def find_Tanisotropy(T_tensor, B, Bepoch, hammerepoch):
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
        i = np.argmin(np.abs(Bepoch - hammerepoch[hamepoch_idx]))
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

    return np.asarray(T_perp), np.asarray(T_parallel), np.asarray(Anisotropy)