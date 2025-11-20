import wget, cdflib, bisect, sys, pyspedas
import numpy as np
from datetime import datetime
import os.path
import matplotlib.pyplot as plt; plt.ion()

from matplotlib import ticker, cm
import warnings 
warnings.filterwarnings("ignore")

from warnings import simplefilter 
simplefilter(action='ignore', category=DeprecationWarning)

# global variables
mass_p = 0.010438870      #proton mass in units eV/c^2 where c = 299792 km/s
charge_p = 1              #proton charge in units eV


package_dir = os.getcwd()  # os.path.dirname(current_dir)
with open(f"{package_dir}/.credentials", "r") as f:
    CREDENTIALS = f.read().splitlines()

def yyyymmdd(dt) : return f"{dt.year:04d}{dt.month:02d}{dt.day:02d}"

def VDfile_directoryRemote(user_datetime):
    VDF_RemoteDir = f'http://w3sweap.cfa.harvard.edu/pub/data/sci/sweap/spi/L2/spi_sf00/{user_datetime.year:04d}/{user_datetime.month:02d}/'
    VDF_filename = f'psp_swp_spi_sf00_L2_8Dx32Ex8A_{yyyymmdd(user_datetime)}_v04.cdf'

    return VDF_RemoteDir, VDF_filename

def download_VDF_file(user_datetime):
    # Import from file directory
    VDfile_RemoteDir, VDF_filename = VDfile_directoryRemote(user_datetime)

    #check if file is already downloaded. If so, skip download. If not, download in local directory.
    if os.path.isfile(f'./data/{VDF_filename}'):
        print(f"File already exists in local directory - [./data/{VDF_filename}]")
        VDfile = f'./data/{VDF_filename}'
    else:
        print("File doesn't exist. Downloading ...")
        VDfile = wget.download(VDfile_RemoteDir + VDF_filename, out='./data')

    #open CDF file
    dat_raw = cdflib.CDF(VDfile)
    dat = {}

    dat['EPOCH']  = dat_raw['EPOCH']
    dat['THETA']  = dat_raw['THETA'].reshape((-1,8,32,8))
    dat['PHI']    = dat_raw['PHI'].reshape((-1,8,32,8))
    dat['ENERGY'] = dat_raw['ENERGY'].reshape((-1,8,32,8))
    dat['EFLUX']  = dat_raw['EFLUX'].reshape((-1,8,32,8))

    return dat

def download_L3_data(user_datetime):
    yyyy, mm, dd = user_datetime.year, user_datetime.month, user_datetime.day

    trange = [f'{yyyy}-{mm}-{dd}/00:00:00', f'{yyyy}-{mm}-{dd}/23:59:59']
    try:
        try:
            spi_vars = pyspedas.psp.spi(trange=trange, datatype='spi_sf00_l3_mom', level='l3',
                                        time_clip=True, get_support_data= True, varnames=['*'],
                                        notplot=True, downloadonly=True, no_update=True)
            dat = cdflib.CDF(spi_vars[0])
        except:
            spi_vars = pyspedas.psp.spi(trange=trange, datatype='spi_sf00', level='L3',
                                        time_clip=True, get_support_data= True, varnames=['*'],
                                        notplot=True, downloadonly=True, username=CREDENTIALS[0],
                                        password=CREDENTIALS[1], no_update=True)
            dat = cdflib.CDF(spi_vars[0])
    # if local L3 file does not exist
    except:
        try:
            spi_vars = pyspedas.psp.spi(trange=trange, datatype='spi_sf00_l3_mom', level='l3',
                                        time_clip=True, get_support_data= True, varnames=['*'],
                                        notplot=True, downloadonly=True)
            dat = cdflib.CDF(spi_vars[0])
        except:
            spi_vars = pyspedas.psp.spi(trange=trange, datatype='spi_sf00', level='L3',
                                        time_clip=True, get_support_data= True, varnames=['*'],
                                        notplot=True, downloadonly=True, username=CREDENTIALS[0],
                                        password=CREDENTIALS[1])
            dat = cdflib.CDF(spi_vars[0])

    return dat

def get_VDFdict_at_t(cdf_VDfile, tSliceIndex):
    epochSlice  = cdf_VDfile['EPOCH'][tSliceIndex]
    thetaSlice  = cdf_VDfile['THETA'][tSliceIndex,:]
    phiSlice    = cdf_VDfile['PHI'][tSliceIndex,:]
    energySlice = cdf_VDfile['ENERGY'][tSliceIndex,:]
    efluxSlice  = cdf_VDfile['EFLUX'][tSliceIndex,:]

    # Define VDF
    numberFluxSlice = efluxSlice/energySlice
    vdfSlice = numberFluxSlice*(mass_p**2)/((2E-5)*energySlice)

    # making the dictionary
    vdf_bundle = {}

    vdf_bundle['t_index'] = tSliceIndex
    vdf_bundle['vdf'] = vdfSlice[:,::-1,:]
    vdf_bundle['theta'] = thetaSlice[:,::-1,:]
    vdf_bundle['phi'] = phiSlice[:,::-1,:]
    vdf_bundle['energy'] = energySlice[:,::-1,:]

    return vdf_bundle

def get_L3_monents_at_t(L3_data, tSliceIndex):
    # making the dictionary
    l3_data_bundle = {}

    # extracting out the required L3 data
    l3_data_bundle['MAGF_INST'] = L3_data['MAGF_INST'][tSliceIndex,:]
    l3_data_bundle['VEL_INST'] = L3_data['VEL_INST'][tSliceIndex,:]
    l3_data_bundle['DENS'] = L3_data['DENS'][tSliceIndex]
    l3_data_bundle['SUN_DIST'] = L3_data['SUN_DIST'][tSliceIndex]

    return l3_data_bundle

def theta_phi_cuts(vdf_dict, theta_cut=0, phi_cut=1, cmap='plasma'):
    plt.rcParams['font.size'] = 16
    plt.style.use('dark_background')
    # theta is along dimension 0, while phi is along 2
    # first cutting through theta
    theta_cut=0 

    phi_plane = vdf_dict['phi'][theta_cut,:,:]
    theta_plane = vdf_dict['theta'][theta_cut,:,:]
    energy_plane = vdf_dict['energy'][theta_cut,:,:]
    vel_plane = np.sqrt(2 * charge_p * energy_plane / mass_p)

    # VDF as a function of energy and phi (theta axis is summed over)
    df_theta = np.nansum(vdf_dict['vdf'], axis=0)

    vx_plane_theta = vel_plane * np.cos(np.radians(phi_plane)) * np.cos(np.radians(theta_plane))
    vy_plane_theta = vel_plane * np.sin(np.radians(phi_plane)) * np.cos(np.radians(theta_plane))
    vz_plane_theta = vel_plane *                                 np.sin(np.radians(theta_plane))

    fig, ax = plt.subplots(1, 2, figsize=(12,8))
    cs = ax[0].contourf(vx_plane_theta, vz_plane_theta, df_theta,
                        locator=ticker.LogLocator(numticks=10), cmap=cmap, rasterized='True')
    # cbar = fig.colorbar(cs)
    # cbar.set_label(f'f $(cm^2 \\ s \\ sr \\ eV)^{-1}$')

    ax[0].set_xlim(-700,0)
    ax[0].set_ylim(-500,500)
    ax[0].set_xlabel('$v_x$ km/s')
    ax[0].set_ylabel('$v_z$ km/s')
    ax[0].set_title('VDF SPAN-I $\\theta$-plane')
    ax[0].set_aspect('equal')

    # now cutting in phi dimension
    phi_cut = 1

    phi_plane = vdf_dict['phi'][:,:,phi_cut]
    theta_plane = vdf_dict['theta'][:,:,phi_cut]
    energy_plane = vdf_dict['energy'][:,:,phi_cut]
    vel_plane = np.sqrt(2 * charge_p * energy_plane / mass_p)

    # VDF as a function of energy and theta. phi dimension is summed over
    df_phi = np.nansum(vdf_dict['vdf'], axis=2)

    vx_plane_phi = vel_plane * np.cos(np.radians(phi_plane)) * np.cos(np.radians(theta_plane))
    vy_plane_phi = vel_plane * np.sin(np.radians(phi_plane)) * np.cos(np.radians(theta_plane))
    vz_plane_phi = vel_plane *                                 np.sin(np.radians(theta_plane))

    # fig, ax = plt.subplots()
    cs = ax[1].contourf(np.transpose(vx_plane_phi), np.transpose(vy_plane_phi), np.transpose(df_phi),
                        locator=ticker.LogLocator(numticks=10), cmap=cmap, rasterized='True')
    cbar = fig.colorbar(cs, orientation='horizontal')
    cbar.set_label(f'f $(cm^2 \\ s \\ sr \\ eV)^{-1}$')

    ax[1].set_xlim(-700,0)
    ax[1].set_ylim(-200,500)
    ax[1].set_xlabel('$v_x$ km/s')
    ax[1].set_ylabel('$v_y$ km/s')
    ax[1].set_title('VDF SPAN-I $\\phi$-plane')
    ax[1].set_aspect('equal')

    plt.suptitle(f'Time: {epoch[tSliceIndex]}', fontweight='bold')

    plt.subplots_adjust(hspace=0.5, wspace=0.5, left=0.1, right=0.97, top=0.97, bottom=0.05)


if __name__=='__main__':
    year, month, date = 2020, 1, 29
    hour, minute, second = 18, 10, 1

    user_datetime = datetime(year, month, date)

    # loading the data (downloading the file if necessary)
    cdf_VDfile = download_VDF_file(user_datetime)

    timeSlice  = np.datetime64(datetime(year, month, date, hour, minute, second))
    print('Desired timeslice:', timeSlice)

    #find index for desired timeslice
    epoch = cdflib.cdfepoch.to_datetime(cdf_VDfile['EPOCH'])
    tSliceIndex  = bisect.bisect_left(epoch, timeSlice)

    vdf_dict = get_VDFdict_at_t(cdf_VDfile, tSliceIndex)
    theta_phi_cuts(vdf_dict, theta_cut=0, phi_cut=1)

    

