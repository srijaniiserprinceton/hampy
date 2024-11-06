import numpy as np
from datetime import datetime
import cdflib, bisect
import matplotlib.pyplot as plt; plt.ion()

# custom script imports
import get_span_data as get_data

# global variables
mass_p = 0.010438870      #proton mass in units eV/c^2 where c = 299792 km/s
charge_p = 1              #proton charge in units eV

def plot_2D_vdf(vdf_slice, xx, yy):
    plt.figure()
    plt.contourf(xx, yy, np.log10(vdf_slice), cmap='inferno', vmin=-1, vmax=8)
    plt.colorbar()

    # making scatter plots of the grids
    plt.scatter(xx.flatten(), yy.flatten(), color='grey', alpha=0.5, s=5)

    plt.xlabel('VX [km/s]')
    plt.ylabel('VZ [km/s]')
    plt.xlim(-1000,100)
    plt.ylim(-300,700)
    plt.gca().set_aspect('equal')
    plt.axhline(0.0, color='white')
    plt.tight_layout()

def rotate_coordinates(angle_rot, vr, vz):
    # converting the angle from degrees to radians
    angle_rot = np.radians(angle_rot)

    # building the rotation matrix
    rot_mat = np.array([[np.cos(angle_rot), np.sin(angle_rot)],
                        [-np.sin(angle_rot), np.cos(angle_rot)]])

    # making the velocity vector
    vel_vec = np.zeros((32, 2, 8))
    vel_vec[:,0] = vr
    vel_vec[:,1] = vz

    # rotating the 2D velocity vector
    v_new = rot_mat @ vel_vec

    vr_new = v_new[:,0]
    vz_new = v_new[:,1]

    return vr_new, vz_new

if __name__=='__main__':
    # user defined date and time
    year, month, date = 2020, 1, 29
    hour, minute, second = 18, 10, 1

    # converting to datetime format to extract time index
    user_datetime = datetime(year, month, date)
    timeSlice  = np.datetime64(datetime(year, month, date, hour, minute, second))

    # loading the data (downloading the file if necessary)
    cdf_VDfile = get_data.download_VDF_file(user_datetime)
    # getting the spi_vars 
    l3_data = get_data.download_L3_data(user_datetime)

    # convert time
    epoch = cdflib.cdfepoch.to_datetime(cdf_VDfile['EPOCH'])
    # find index for desired timeslice
    tSliceIndex  = bisect.bisect_left(epoch, timeSlice)
    # getting the VDF dictionary at the desired timestamp
    vdf_dict = get_data.get_VDFdict_at_t(cdf_VDfile, tSliceIndex)

    # convert time
    epoch = cdflib.cdfepoch.to_datetime(l3_data['EPOCH'])
    # find index for desired timeslice
    tSliceIndex  = bisect.bisect_left(epoch, timeSlice)
    # getting the required l3 data dictionary
    l3_data_dict = get_data.get_L3_monents_at_t(l3_data, tSliceIndex)

    # updating the vdf_dict with the required L3 data
    vdf_dict.update(l3_data_dict)

    # choosing a cut through phi for plotting
    phi_cut=0 

    phi_plane = vdf_dict['phi'][phi_cut,:,:]
    theta_plane = vdf_dict['theta'][phi_cut,:,:]
    energy_plane = vdf_dict['energy'][phi_cut,:,:]
    vel_plane = np.sqrt(2 * charge_p * energy_plane / mass_p)

    # VDF as a function of energy and theta (phi axis is summed over)
    df_theta = np.nansum(vdf_dict['vdf'], axis=0)

    # making the cartesian grid
    vx_plane_theta = vel_plane * np.cos(np.radians(phi_plane)) * np.cos(np.radians(theta_plane))
    vy_plane_theta = vel_plane * np.sin(np.radians(phi_plane)) * np.cos(np.radians(theta_plane))
    vz_plane_theta = vel_plane *                                 np.sin(np.radians(theta_plane))

    # plotting vdf on original grids
    plot_2D_vdf(df_theta, vx_plane_theta, vz_plane_theta)

    # finding the angle between b vector and instrument
    Bx, By, Bz = vdf_dict['MAGF_INST']
    Bmag = np.sqrt(Bx**2 + Bz**2)
    rot_angle =  -1. * np.arccos(Bx/Bmag) * 180/np.pi

    # # rotating the vx and vz
    # vx_, vz_ = rotate_coordinates(rot_angle, vx_plane_theta, vz_plane_theta)

    # # plotting vdf on rotated coordinates
    # plot_2D_vdf(df_theta, vx_, vz_)

    # rotating using a different method
    theta_plane = theta_plane + rot_angle
    vx_plane_theta = vel_plane * np.cos(np.radians(phi_plane)) * np.cos(np.radians(theta_plane))
    vy_plane_theta = vel_plane * np.sin(np.radians(phi_plane)) * np.cos(np.radians(theta_plane))
    vz_plane_theta = vel_plane *                                 np.sin(np.radians(theta_plane))

    # plotting vdf on new rotated grids
    plot_2D_vdf(df_theta, vx_plane_theta, vz_plane_theta)

    # plotting in the velocity-theta grid

    plt.figure()
    plt.pcolormesh(vel_plane, theta_plane, np.log10(df_theta))



