import numpy as np
import pandas as pd
import astropy.units as u
import matplotlib.pyplot as plt; plt.ion()
from matplotlib.colors import Normalize
import cdflib, pickle, re, os, pyspedas, xarray
from datetime import datetime, timedelta
from scipy.interpolate import griddata
import astropy.constants as c
from scipy.stats.stats import pearsonr

from hampy import misc_functions as misc_fn
from hampy import load_data
from hampy import orbit_trajectory
from hampy import encounters

plt.rcParams.update({'font.size': 16})

def find_enc_from_trange(trange):
    perihelion_dates = []
    for enc in range(1, 27):
        if(enc < 10): enc_str = f'E0{enc}'
        else: enc_str = f'E{enc}'
        perihelion_dates.append(encounters.get_enc_dates(enc_str))
    perihelion_dates = np.sort(np.array(perihelion_dates))

    enc_idx = np.argmin(np.abs(perihelion_dates - np.array(trange).astype('datetime64[D]')[0]))
    return enc_idx + 1

def read_pickle(fname):
    with open(f'{fname}.pkl', 'rb') as handle:
        x = pickle.load(handle)
    return x

def write_pickle(x, fname):
    with open(f'{fname}.pkl', 'wb') as handle:
        pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_dates(start_date, end_date):
    start_yyyy, start_mm, start_dd = np.asarray(re.split('[-]', start_date)).astype('int')
    end_yyyy, end_mm, end_dd = np.asarray(re.split('[-]', end_date)).astype('int')

    return np.arange(datetime(start_yyyy, start_mm, start_dd),
                     datetime(end_yyyy, end_mm, end_dd) + timedelta(days=1), timedelta(days=1)).astype(datetime)

def plot_density():
    plt.figure(figsize=(15,4))

    for component in components:
        plt.semilogy(global_prop_dict['dt_arr'], global_prop_dict[f'{component}_density'], '.', alpha=0.7,
                     label=f'{component}')

    # plt.semilogy(global_prop_dict['span_time_all'], global_prop_dict['span_density_all'], 'w')
    plt.xlim([global_prop_dict['dt_arr'][0], global_prop_dict['dt_arr'][-1]])
    # plt.xlim([datetime(2024,4,1,0,0,0), datetime(2024,4,5,0,0,0)])
    # plt.ylim([0, 1e4])
    plt.ylabel(r'Density in $\rm{cm}^{-3}$')
    plt.xlabel(r'Time')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{dirname}/density_{start_date}_{end_date}.png')
    plt.close()

    plt.figure(figsize=(15,4))
    plt.plot(global_prop_dict['dt_arr'],\
             global_prop_dict['hammer_density']/global_prop_dict['neck_density'], '.w')
    plt.ylabel(r'$n_{\rm{hammer}} / n_{\rm{neck}}$')
    plt.ylim([0,2])
    plt.xlabel('Time')
    plt.tight_layout()
    plt.savefig(f'{dirname}/density_hammer_neck_ratio.png')
    plt.close()

def plot_velocity():
    fig, ax = plt.subplots(3, 1, figsize=(15,10), sharex=True)

    for component in components:
        ax[0].plot(global_prop_dict['dt_arr'], global_prop_dict['Ux'][f'{component}'], '.', alpha=0.7,
                       label=f'{component}')
        ax[1].plot(global_prop_dict['dt_arr'], global_prop_dict['Uy'][f'{component}'], '.', alpha=0.7,
                       label=f'{component}')
        ax[2].plot(global_prop_dict['dt_arr'], global_prop_dict['Uz'][f'{component}'], '.', alpha=0.7,
                       label=f'{component}')

    for axs in ax:
        axs.set_xlim([global_prop_dict['dt_arr'][0], global_prop_dict['dt_arr'][-1]])

    ax[0].set_ylabel('Ux [km/s]')
    ax[1].set_ylabel('Uy [km/s]')
    ax[2].set_ylabel('Uz [km/s]')
    ax[2].set_xlabel(r'Time')
    
    plt.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.98)

    plt.savefig(f'{dirname}/velocity_{start_date}_{end_date}.png')
    plt.close()

def plot_T_ani():
    plt.figure(figsize=(15,4))

    for component in components:
        plt.semilogy(global_prop_dict['dt_arr'], global_prop_dict['T_ani'][f'{component}'], '.', alpha=0.7,
                     label=f'{component}')

    plt.xlim([global_prop_dict['dt_arr'][0], global_prop_dict['dt_arr'][-1]])
    plt.ylabel(r'$T_{\perp} / T_{||}$')
    plt.xlabel(r'Time')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{dirname}/Tani_{start_date}_{end_date}.png')
    plt.close()

    # making histograms
    plt.figure(figsize=(10,10))

    for component in components:
        plt.hist(global_prop_dict['T_ani'][f'{component}'], range=(0, 10), bins=50,
                 label=f'{component}', histtype='step')

    plt.xlabel(r'$T_{\perp} / T_{||}$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{dirname}/Tani_histogram_{start_date}_{end_date}.png')
    plt.close()


    density_hampy = global_prop_dict['total_hampy_density']
    span_Bmag = global_prop_dict['span_Bmag']
    valfven = 21.8 * span_Bmag / np.sqrt(density_hampy)

    fig, ax = plt.subplots(2, 1, figsize=(15,10), sharex=True)
    Tani_effective = (global_prop_dict['T_ani']['hammer'] * global_prop_dict['hammer_density'] +\
                      global_prop_dict['T_ani']['neck'] * global_prop_dict['neck_density'])/\
                      (global_prop_dict['hammer_density'] + global_prop_dict['neck_density'])

    # ax[0].scatter(global_prop_dict['dt_arr'],\
    #               np.log10(Tani_effective/global_prop_dict['T_ani']['core']),
    #               c=global_prop_dict['hammer_vdrift']/valfven)
    ax[0].scatter(global_prop_dict['dt_arr'],\
                  global_prop_dict['T_ani']['hammer'],
                  c=global_prop_dict['hammer_vdrift']/valfven)
    # ax[0].set_ylim([0, 2])
    ax[0].set_ylabel(r'$Tani_{\rm{hammer+neck}} / Tani_{\rm{core}}$')
    ax[0].set_title(r'Color = $V_{\rm{drift,hammer}} / V_{\rm{Alfven}}$')
    
    # finding the correlation coefficient
    # xmin_idx = np.argmin(np.abs(global_prop_dict['dt_arr'] - datetime(2022,12,12,7,0,0)))
    # xmax_idx = np.argmin(np.abs(global_prop_dict['dt_arr'] - datetime(2022,12,12,10,30,0)))
    # x, y = global_prop_dict['dt_arr'], global_prop_dict['hammer_vdrift']/valfven
    # corr_coeff = pearsonr(x[xmin_idx:xmax_idx+1], y[xmin_idx:xmax_idx+1])
    # coeffs = numpy.polyfit(x, y, 1)
    # ax[1].plot(coeffs[0] * x + coeffs[1], '--w')
    ax[1].scatter(global_prop_dict['dt_arr'], global_prop_dict['hammer_vdrift']/valfven,
                  c=global_prop_dict['T_ani']['hammer'],  cmap='Reds')
    # ax[1].set_xlim([datetime(2022,9,6,15,0,0), datetime(2022,12,12,10,30,0)])
    ax[1].set_xlim([None, datetime(2022,9,6,17,30,0)])
    ax[1].set_ylim([-3, -1])
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel(r'$V_{\rm{drift,hammer}} / V_{\rm{Alfven}}$')
    ax[1].set_title(r'Color = $Tani_{\rm{hammer+neck}} / Tani_{\rm{core}}$')
    plt.savefig(f'{dirname}/Tani_hammer_core_ratio.png')
    plt.close()

def plot_as_function_of_distance(color=None):
    density_hampy = global_prop_dict['total_hampy_density']
    span_Bmag = global_prop_dict['span_Bmag']
    valfven = 21.8 * span_Bmag / np.sqrt(density_hampy)

    plt.figure()
    if(color is not None):
        plt.scatter(global_prop_dict['dist_rsun'], global_prop_dict['hammer_vdrift'], marker='.', c=color)
    else:
        plt.plot(global_prop_dict['dist_rsun'], global_prop_dict['hammer_vdrift'], '.')
    plt.xlabel('Distance in Rsun')
    plt.ylabel(r'$V_{\rm{drift, hammer}}$')
    plt.tight_layout()
    plt.savefig(f'{dirname}/Rsun_Vdrift.png')
    plt.close()

    plt.figure()

    if(color is not None):
        plt.scatter(global_prop_dict['dist_rsun'], global_prop_dict['hammer_density'],
                    marker='.', c=color)
    else:
        plt.plot(global_prop_dict['dist_rsun'], global_prop_dict['hammer_density'], '.')

    plt.xlabel('Distance in Rsun')
    plt.ylabel(r'$n_{\rm{hammer}}$')
    plt.tight_layout()
    plt.savefig(f'{dirname}/Rsun_nhammer.png')
    plt.close()

    plt.figure()
    if(color is not None):
        plt.scatter(global_prop_dict['dist_rsun'], -global_prop_dict['hammer_vdrift']/valfven, marker='.', c=color)
    else:
        plt.plot(global_prop_dict['dist_rsun'], -global_prop_dict['hammer_vdrift']/valfven, '.')
    
    plt.yscale('log')
    plt.xlabel('Distance in Rsun')
    plt.ylabel(r'$V_{\rm{drift, hammer}}/V_{\rm{Alfven}}$')
    plt.tight_layout()
    plt.savefig(f'{dirname}/Rsun_Vdrift_Valfven.png')
    plt.close()

    plt.figure()

    if(color is not None):
        plt.scatter(global_prop_dict['dist_rsun'], global_prop_dict['hammer_density']/global_prop_dict['total_hampy_density'],
                    marker='.', c=color)
    else:
        plt.plot(global_prop_dict['dist_rsun'], global_prop_dict['hammer_density']/global_prop_dict['total_hampy_density'], '.')

    plt.yscale('log')
    plt.xlabel('Distance in Rsun')
    plt.ylabel(r'$n_{\rm{hammer}}/n_{\rm{total}}$')
    plt.tight_layout()
    plt.savefig(f'{dirname}/Rsun_nhammer_ntotal.png')
    plt.close()

def plot_T_ani_2d_hist():
    density_hammer = global_prop_dict['hammer_density']
    density_hampy = global_prop_dict['total_hampy_density']
    Tani_hammer = global_prop_dict['T_ani']['hammer']

    plt.figure()
    h1, xe1, ye1, fig1 = plt.hist2d(density_hammer/density_hampy, Tani_hammer, bins=20, range=[[0, 0.08], [0, 8]], cmap='inferno')
    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    # interpolating to get a higgher resolution matrix
    x1, y1 = np.meshgrid(xe1[:-1], ye1[:-1], indexing='ij')
    xm1, ym1 = np.meshgrid(np.linspace(xe1.min(), xe1.max(), 100), np.linspace(ye1.min(), ye1.max(), 100), indexing='ij')
    h1_interp = griddata((x1.flatten(), y1.flatten()), h1.flatten(), (xm1, ym1), method='linear')
    h1_interp = h1_interp / np.nanmax(h1_interp)
    # ax[0].pcolormesh(xe1[:-1], ye1[:-1], h1.T, cmap='inferno')#, levels=20)
    ax[0].contourf(xm1, ym1, h1_interp, cmap='inferno', levels=50)
    levels = np.linspace(0.5, 1, 5)
    ax[0].contour(xm1, ym1, h1_interp, levels=levels, colors='black')
    ax[0].set_xlabel(r'$n_{\rm{hammer}} / n_{\rm{total}}$')
    ax[0].set_ylabel(r'$(T_{\perp}/T_{||})_{\rm{hammer}}$')
    ax[0].set_xlim([0, 0.08])
    ax[0].set_ylim([0,8])


    neck_vdrift = global_prop_dict['neck_vdrift']
    hammer_vdrift = global_prop_dict['hammer_vdrift']
    span_Bmag = global_prop_dict['span_Bmag']
    valfven = 21.8 * span_Bmag / np.sqrt(density_hampy)

    plt.figure()
    h2, xe2, ye2, fig2 = plt.hist2d(hammer_vdrift/valfven, Tani_hammer, bins=20, range=[[-3.5, 0.0], [0, 8]], cmap='inferno')
    plt.close()

    x2, y2 = np.meshgrid(xe2[:-1], ye2[:-1], indexing='ij')
    xm2, ym2 = np.meshgrid(np.linspace(xe2.min(), xe2.max(), 100), np.linspace(ye2.min(), ye2.max(), 100), indexing='ij')
    h2_interp = griddata((x2.flatten(), y2.flatten()), h2.flatten(), (xm2, ym2), method='linear')
    h2_interp = h2_interp / np.nanmax(h2_interp)
    # ax[0].pcolormesh(xe1[:-1], ye1[:-1], h1.T, cmap='inferno')#, levels=20)
    # ax[0].pcolormesh(xm1, ym1, h1_interp, cmap='inferno')#, levels=20)
    im = ax[1].contourf(xm2, ym2, h2_interp, cmap='inferno', levels=50)
    # ax[1].contour(xm2, ym2, h2_interp, levels=[0.4, 0.0.75, 0.8, 0.85, 0.9, 0.95, 0.99], colors='black')
    levels = np.linspace(0.5, 1, 5)
    ax[1].contour(xm2, ym2, h2_interp, levels=levels, colors='black')
    ax[1].set_xlabel(r'$V_{\rm{drift,hammer}} / V_{\rm{Alfven}}$')
    ax[1].set_ylabel(r'$(T_{\perp}/T_{||})_{\rm{hammer}}$')
    ax[1].set_xlim([-3.5, -1.0])
    ax[1].set_ylim([0, 8])

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.93, 0.2, 0.005, 0.75])
    fig.colorbar(im, cax=cbar_ax)

    plt.subplots_adjust(left=0.04, right=0.9, bottom=0.17, top=0.98)

    plt.savefig(f'{dirname}/Tani_density_{start_date}_{end_date}.png')
    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    ax[0].scatter(density_hammer/density_hampy, Tani_hammer, marker='.', c=density_hammer/density_hampy)
    ax[0].set_xlabel(r'$n_{\rm{hammer}} / n_{\rm{total}}$')
    ax[0].set_ylabel(r'$T_{\perp}/T_{||}$')
    ax[0].set_xlim([0, 0.08])
    ax[0].set_ylim([0,8])

    ax[1].scatter(hammer_vdrift/valfven, Tani_hammer, marker='.', c=density_hammer/density_hampy)
    ax[1].set_xlabel(r'$V_{\rm{drift,hammer}} / V_{\rm{Alfven}}$')
    ax[1].set_ylabel(r'$T_{\perp}/T_{||}$')
    ax[1].set_xlim([-3.5, -1.0])
    ax[1].set_ylim([0, 8])

    plt.subplots_adjust(left=0.04, right=0.9, bottom=0.17, top=0.98)
    plt.savefig(f'{dirname}/Tani_densitycolor_{start_date}_{end_date}.png')
    plt.close()

def make_brazil_plots():
    cmap = 'rainbow_r'
    valfven = 21.8 * global_prop_dict['span_Bmag']/ np.sqrt(global_prop_dict['total_hampy_density'])

    # making plasma beta
    T_parallel, T_perp = 0.0, 0.0
    for component in components:
        T_parallel += global_prop_dict['T_parallel'][f'{component}'] * global_prop_dict[f'{component}_density']
        T_perp += global_prop_dict['T_perp'][f'{component}'] * global_prop_dict[f'{component}_density']
    T_parallel = T_parallel / global_prop_dict['total_hampy_density']
    T_perp = T_perp / global_prop_dict['total_hampy_density']
                  
    # density (cc -> m^{-3}), temperature (MK -> K), magnetic field (nT -> T)
    beta = (global_prop_dict['total_hampy_density'] * 1e6) * c.k_B.value * (T_parallel * 1e6) * c.mu0.value \
           /(global_prop_dict['span_Bmag'] * 1e-9)**2

    beta_arr = np.logspace(-3, 2, 200)
    beta_brazil, Tani_brazil = misc_fn.Tani_beta_instability_relations(beta_arr)
    
    plt.figure()
    vdrift = -global_prop_dict['hammer_vdrift']
    # color = (vdrift - vdrift.mean())/(vdrift.max() - vdrift.min())
    color = -1. *  vdrift / valfven
    sc = plt.scatter(beta, T_perp/T_parallel, s=1, c=color, alpha=0.5, cmap=cmap)
    # sc = plt.scatter(beta, global_prop_dict['T_ani']['hammer'], s=1, c=color, alpha=0.5, cmap=cmap)
    plt.plot(beta_brazil, Tani_brazil[0], '--w', label='i-c')
    plt.plot(beta_brazil, Tani_brazil[1], 'w', label='m')
    plt.plot(beta_brazil, Tani_brazil[2], '--w', label='p-f')
    plt.plot(beta_brazil, Tani_brazil[3], 'w', label='o-f')
    plt.legend()
    plt.colorbar(sc)
    plt.ylim([1e-1, 1e2])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim([None, 1e2])
    plt.xlabel(r'$\beta_{||}$')
    plt.ylabel(r'$T_{\perp}/T_{||}$')
    plt.title(r'$V_{\rm{ham,drift}} / V_{\rm{Alfven}}$')
    plt.tight_layout()
    plt.savefig(f'{dirname}/Brazil_plot_Vdrift.png')
    plt.close()

    plt.figure()
    color = global_prop_dict['dist_rsun']
    sc = plt.scatter(beta, T_perp/T_parallel, s=1, c=color, alpha=0.5, cmap=cmap)
    plt.plot(beta_brazil, Tani_brazil[0], '--w', label='i-c')
    plt.plot(beta_brazil, Tani_brazil[1], 'w', label='m')
    plt.plot(beta_brazil, Tani_brazil[2], '--w', label='p-f')
    plt.plot(beta_brazil, Tani_brazil[3], 'w', label='o-f')
    plt.legend()
    plt.colorbar(sc)
    plt.ylim([1e-1, 1e2])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim([None, 1e2])
    plt.xlabel(r'$\beta_{||}$')
    plt.ylabel(r'$T_{\perp}/T_{||}$')
    plt.title(r'Distance in $R_{\rm{Sun}}$')
    plt.tight_layout()
    plt.savefig(f'{dirname}/Brazil_plot_distrsun.png')
    plt.close()

    plt.figure()
    color = global_prop_dict['hammer_density']/global_prop_dict['total_hampy_density']
    sc = plt.scatter(beta, T_perp/T_parallel, s=1, c=color, alpha=0.5, cmap=cmap)
    plt.plot(beta_brazil, Tani_brazil[0], '--w', label='i-c')
    plt.plot(beta_brazil, Tani_brazil[1], 'w', label='m')
    plt.plot(beta_brazil, Tani_brazil[2], '--w', label='p-f')
    plt.plot(beta_brazil, Tani_brazil[3], 'w', label='o-f')
    plt.legend()
    plt.colorbar(sc)
    plt.ylim([1e-1, 1e2])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim([None, 1e2])
    plt.xlabel(r'$\beta_{||}$')
    plt.ylabel(r'$T_{\perp}/T_{||}$')
    plt.title(r'$n_{\rm{hammer}}/n_{\rm{total}}$')
    plt.tight_layout()
    plt.savefig(f'{dirname}/Brazil_plot_reldensity.png')
    plt.close()

def make_fish_plots():
    # getting radius and longitude for the whole duration and not just hammerhead times
    radius_all = global_prop_dict['all_orbit'].radius.value / c.R_sun.to('km').value
    lon_all = np.radians(global_prop_dict['all_orbit'].lon.value)
    x_all, y_all = radius_all * np.cos(lon_all), radius_all * np.sin(lon_all)

    radius = global_prop_dict['orbit'].radius.value / c.R_sun.to('km').value
    lon = np.radians(global_prop_dict['orbit'].lon.value)
    x, y = radius * np.cos(lon), radius * np.sin(lon)

    plt.figure()
    # plotting the Sun
    plt.scatter(0, 0, c='yellow', s=40)
    color = (global_prop_dict['all_orbit'].obstime - global_prop_dict['all_orbit'].obstime[0]).value
    plt.scatter(x_all, y_all, c=color, s=3, alpha=0.8, cmap='seismic')
    plt.title('Orbit colored by time')
    plt.tight_layout()
    plt.savefig(f'{dirname}/Fishplot_orbit_timecolored.png')
    plt.close()

    plt.figure()
    # plotting the Sun
    plt.scatter(0, 0, c='yellow', s=40)
    plt.plot(x_all, y_all, c='cyan', lw=1, alpha=0.3)
    plt.scatter(x, y, c='white', s=1, alpha=0.5)
    plt.title('OG Hammerhead occurance rate')
    plt.tight_layout()
    plt.savefig(f'{dirname}/Fishplot_orbit_occurance.png')
    plt.close()

    plt.figure()
    # plotting the Sun
    plt.scatter(0, 0, c='yellow', s=40)
    plt.plot(x_all, y_all, c='cyan', lw=1, alpha=0.3)
    sc = plt.scatter(x, y, c=-global_prop_dict['hammer_vdrift'], s=3, alpha=0.7, cmap='seismic')
    plt.colorbar(sc)
    plt.title('Drift speed of hammerhead')
    plt.tight_layout()
    plt.savefig(f'{dirname}/Fishplot_orbit_vdrift.png')
    plt.close()


    plt.figure()
    # plotting the Sun
    plt.scatter(0, 0, c='yellow', s=40)
    color = np.log10(global_prop_dict['hammer_density']/global_prop_dict['total_hampy_density'])
    plt.plot(x_all, y_all, c='cyan', lw=1, alpha=0.3)
    sc = plt.scatter(x, y, c=color, s=3, alpha=0.7, cmap='seismic', norm=Normalize(vmin=-3, vmax=-1))
    plt.colorbar(sc)
    plt.title('Hammerhead density fraction [logscale]')
    plt.tight_layout()
    plt.savefig(f'{dirname}/Fishplot_orbit_reldens.png')
    plt.close()

    plt.figure()
    # plotting the Sun
    plt.scatter(0, 0, c='yellow', s=40)
    color = np.log10(global_prop_dict['T_ani']['hammer'])
    plt.plot(x_all, y_all, c='cyan', lw=1, alpha=0.3)
    sc = plt.scatter(x, y, c=color, s=3, alpha=0.7, cmap='seismic', norm=Normalize(vmin=-0.5, vmax=1))
    plt.colorbar(sc)
    plt.title(r'Hammehead $T_{\perp}/T_{||}$ [logscale]')
    plt.tight_layout()
    plt.savefig(f'{dirname}/Fishplot_orbit_Tani_hammer.png')
    plt.close()

def make_global_prop(components):
    # making the global properties dictionary
    global_prop_dict = {}
    global_prop_dict['core_density'] = np.array([])
    global_prop_dict['neck_density'] = np.array([])
    global_prop_dict['hammer_density'] = np.array([])
    global_prop_dict['total_hampy_density'] = np.array([])
    global_prop_dict['span_density'] = np.array([])
    global_prop_dict['span_density_all'] = np.array([])
    global_prop_dict['span_time_all'] = np.array([])
    global_prop_dict['Ux'] = {}
    global_prop_dict['Uy'] = {}
    global_prop_dict['Uz'] = {}
    global_prop_dict['T_perp'] = {}
    global_prop_dict['T_parallel'] = {}
    global_prop_dict['T_ani'] = {}
    for component in components:
        global_prop_dict['Ux'][f'{component}'] = np.array([])
        global_prop_dict['Uy'][f'{component}'] = np.array([])
        global_prop_dict['Uz'][f'{component}'] = np.array([])
        global_prop_dict['T_perp'][f'{component}'] = np.array([])
        global_prop_dict['T_parallel'][f'{component}'] = np.array([])
        global_prop_dict['T_ani'][f'{component}'] = np.array([])
    global_prop_dict['dt_arr'] = np.array([])
    global_prop_dict['dist_rsun'] = np.array([])
    global_prop_dict['B_vec_inst'] = np.array([])
    global_prop_dict['neck_vdrift'] = np.array([])
    global_prop_dict['hammer_vdrift'] = np.array([])
    global_prop_dict['span_Bmag'] = np.array([])
    global_prop_dict['orbit'] = None
    global_prop_dict['all_orbit'] = None

    return global_prop_dict

def mince_props(hammer_dayprops, global_prop_dict, components):
    core_density, neck_density, hammer_density, total_hampy_density, span_density, T_perp, T_parallel, T_ani, dt_arr,\
    B_vec_inst, neck_vdrift, hammer_vdrift, span_Bmag, U, dist_rsun, span_density_all, span_time_all = hammer_dayprops

    # inducting into the global dictionary
    global_prop_dict['core_density'] = np.append(global_prop_dict['core_density'], core_density)
    global_prop_dict['neck_density'] = np.append(global_prop_dict['neck_density'], neck_density)
    global_prop_dict['hammer_density'] = np.append(global_prop_dict['hammer_density'], hammer_density)
    global_prop_dict['total_hampy_density'] = np.append(global_prop_dict['total_hampy_density'], total_hampy_density)
    global_prop_dict['span_density'] = np.append(global_prop_dict['span_density'], span_density)

    for component in components:
        global_prop_dict['Ux'][f'{component}'] = np.append(global_prop_dict['Ux'][f'{component}'], U[f'{component}']['Ux'])
        global_prop_dict['Uy'][f'{component}'] = np.append(global_prop_dict['Uy'][f'{component}'], U[f'{component}']['Uy'])
        global_prop_dict['Uz'][f'{component}'] = np.append(global_prop_dict['Uz'][f'{component}'], U[f'{component}']['Uz'])
        global_prop_dict['T_perp'][f'{component}'] = np.append(global_prop_dict['T_perp'][f'{component}'], T_perp[f'{component}'])
        global_prop_dict['T_parallel'][f'{component}'] = np.append(global_prop_dict['T_parallel'][f'{component}'], T_parallel[f'{component}'])
        global_prop_dict['T_ani'][f'{component}'] = np.append(global_prop_dict['T_ani'][f'{component}'], T_ani[f'{component}'])

    global_prop_dict['dt_arr'] = np.append(global_prop_dict['dt_arr'], dt_arr)
    global_prop_dict['dist_rsun'] = np.append(global_prop_dict['dist_rsun'], dist_rsun)
    global_prop_dict['B_vec_inst'] = np.append(global_prop_dict['B_vec_inst'], B_vec_inst)
    global_prop_dict['neck_vdrift'] = np.append(global_prop_dict['neck_vdrift'], neck_vdrift)
    global_prop_dict['hammer_vdrift'] = np.append(global_prop_dict['hammer_vdrift'], hammer_vdrift)
    global_prop_dict['span_Bmag'] = np.append(global_prop_dict['span_Bmag'], span_Bmag)

    global_prop_dict['span_density_all'] = np.append(global_prop_dict['span_density_all'], span_density_all)
    global_prop_dict['span_time_all'] = np.append(global_prop_dict['span_time_all'], span_time_all)

    # getting the orbit trajectory
    global_prop_dict['orbit'] = orbit_trajectory.get_trajectory(dt_mission=global_prop_dict['dt_arr'])
    global_prop_dict['all_orbit'] = orbit_trajectory.get_trajectory(tstart=dates[0], tend=dates[-1])


def call_hamplotter(tstart, tend, og_only_flag, mincount=0, dir='.'):
    # get all dates
    start_date = re.split('[/]', tstart)[0]
    end_date = re.split('[/]', tend)[0]
    dates = get_dates(start_date, end_date)

    dirname = f'plots/{start_date}_{end_date}'

    if(os.path.exists(f'{dirname}')): pass
    else:
        os.mkdir(f'{dirname}')

    components = np.array(['core', 'neck', 'hammer'])
    
    # initializing the dictionary attributes
    global_prop_dict = make_global_prop(components)

    span_data = load_data.span(tstart, tend)

    for date_idx, date in enumerate(dates):
        # loading the pickle file for that particular date
        filename = f'{dir}/hamstring_{date.year:04d}-{date.month:02d}-{date.day:02d}'
        hammerdict = read_pickle(filename)

        # setting up the data loading process [processing will happen one day at a time]
        span_data.start_new_day(date_idx)

        try:
            # retrieving the necessary parameters for that day and appending to the global list
            hammer_props = misc_fn.extract_params(hammerdict, span_data, og_only=og_only_flag, min_hammer_cells=mincount)
            mince_props(hammer_props, global_prop_dict, components)
        except: continue

    return global_prop_dict

def plot_hamogram_trange(trange, hammertimes, FOV_dict=None, tbin_in_sec=100):
    # downloading the L3 data
    files = pyspedas.psp.spi(trange, datatype='spi_sf00_l3_mom', level='l3', notplot=True, time_clip=True, downloadonly=True, last_version=True)

    # Check if there are multiple datasets loaded for the interval.
    if len(files) > 1:
        L3_data = xarray.concat([cdflib.cdf_to_xarray(f).drop_vars(['ROTMAT_SC_INST']) for f in files], dim='Epoch')
    else:
        L3_data = cdflib.cdf_to_xarray(*files)
    xr_time_object = cdflib.epochs_astropy.CDFAstropy.convert_to_astropy(L3_data.Epoch.data)
    xr_time_array = xr_time_object.utc.datetime 
    L3_data['Epoch'] = xr_time_array
    L3_data = L3_data.sel(Epoch=slice(trange[0], trange[-1]))

    # binning the hammerheads according to tbin_in_sec
    hammertimes = [pd.Timestamp(ht).tz_localize("UTC") for ht in hammertimes]
    hammertimes_sec = np.array([ht.timestamp() for ht in hammertimes])
    # hammertimes_sec = np.array([hammertime.timestamp() for hammertime in hammertimes])
    hammertimes_bins = np.arange(hammertimes_sec[0], hammertimes_sec[-1] + tbin_in_sec, tbin_in_sec)

    # binning the SPAN measurements instead of hammertimes
    L3_times_dt = pd.to_datetime(L3_data.Epoch, unit='s', utc=True).to_pydatetime()
    L3_times = np.array([pd.Timestamp(dt) for dt in L3_times_dt])
    clip_mask = (np.array(L3_times) >= np.array(hammertimes[0])) & (np.array(L3_times) <= np.array(hammertimes[-1]))
    L3_times = L3_times[clip_mask]
    L3_times_sec = np.array([ht.timestamp() for ht in L3_times])
    L3_times_bins = np.arange(L3_times_sec[0], L3_times_sec[-1] + tbin_in_sec, tbin_in_sec)

    # making the hamogram with context parameter in the desired trange (Br, rho, ...)
    # bincounts_ham, final_bin_edges_ham = np.histogram(hammertimes_sec, bins=hammertimes_bins)
    bincounts_ham, final_bin_edges_ham = np.histogram(hammertimes_sec, bins=L3_times_bins)
    bincounts, final_bin_edges = np.histogram(L3_times_sec, bins=L3_times_bins)
    bin_cens = (final_bin_edges[1:] + final_bin_edges[:-1]) / 2

    # finding bin edges in datetime.datetime
    final_bin_edges_dt = pd.to_datetime(final_bin_edges, unit='s', utc=True) #np.array([datetime.fromtimestamp(t) for t in final_bin_edges])
    # finding the bin centers
    bin_cens_dt =  pd.to_datetime(bin_cens, unit='s', utc=True) #np.array([datetime.fromtimestamp(t) for t in bin_cens])

    # extracting the context parameters to be plotted
    B = L3_data.MAGF_SC.data
    rho = L3_data.DENS.data

    if(FOV_dict is None):
        print("FOV_dict is None. Skipping FOV subplot.")

        # making the plot
        fig, ax = plt.subplots(1, 1, figsize=(15, 8), sharex=True)

        distRsun = (L3_data['SUN_DIST'].data * u.km).to(u.R_sun)
        Rsq = distRsun ** 2
        ax.plot(L3_data['Epoch'].data, -B[:,1], color='green', lw=1, alpha=0.3, label='BN')
        ax.plot(L3_data['Epoch'].data, B[:,0], color='red', lw=1, alpha=0.3, label='BT')
        ax.plot(L3_data['Epoch'].data, -B[:,2], color='black', lw=1, alpha=0.8, label='BR')
        ax.axhline(0, color='grey')
        ax.legend()

        # hammogram on B
        axt = ax.twinx()
        axt.hist(bin_cens_dt, bins=final_bin_edges_dt, weights=bincounts_ham/bincounts, color='blue', alpha=0.6, lw=2)
        ax.set_ylabel('BRTN [nT]')
        ax.grid(True)
        ax.set_xlim([final_bin_edges_dt[0], final_bin_edges_dt[-1]])

    else:
        # in order to use the FOV dictonary, we need to know which encounter we are in
        enc = find_enc_from_trange(trange)

        total_counts = bincounts          # shape (nbins,)
        nbins = total_counts.size
        ncats = 8

        # now using the FOV information [using all times for FOV plot]
        span_dt = [pd.Timestamp(ht).tz_localize("UTC") for ht in FOV_dict[enc]['dt']]
        span_dt64 = np.array(span_dt, dtype='datetime64[ns]')
        bin_edges = np.array(final_bin_edges_dt, dtype='datetime64[ns]')
        anode_idx = np.array(FOV_dict[enc]['VDF_phimaxidx_t'])

        # 1) Clip to the time range covered by your bins
        mask_time = (span_dt64 >= bin_edges[0]) & (span_dt64 < bin_edges[-1]) & (span_dt64 >= np.datetime64(trange[0])) & (span_dt64 <= np.datetime64(trange[-1]))
        hammer_dt_clip = span_dt64[mask_time]
        anode_idx_clip = anode_idx[mask_time]

        # 2) Which time bin each event goes into (same edges as used for bincounts)
        bin_index = np.searchsorted(bin_edges, hammer_dt_clip, side='right') - 1

        # 3) Only keep “good” anode indices (0..7)
        mask_good = (bin_index >= 0) & (bin_index < nbins) & (anode_idx_clip >= 0) & (anode_idx_clip < ncats)
        bin_index = bin_index[mask_good]
        anode_idx_good = anode_idx_clip[mask_good]

        # 4) Build category_counts[bin, cat]
        category_counts = np.zeros((nbins, ncats), dtype=int)
        np.add.at(category_counts, (bin_index, anode_idx_good), 1)

        # Debug check
        print("Sum category counts:", category_counts.sum(axis=1)[:10])
        print("Total hammerheads:   ", total_counts[:10])


        # making the plot
        fig, ax = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

        distRsun = (L3_data['SUN_DIST'].data * u.km).to(u.R_sun)
        Rsq = distRsun ** 2
        ax[0].plot(L3_data['Epoch'].data, -B[:,1], color='green', lw=1, alpha=0.3, label='BN')
        ax[0].plot(L3_data['Epoch'].data, B[:,0], color='red', lw=1, alpha=0.3, label='BT')
        ax[0].plot(L3_data['Epoch'].data, -B[:,2], color='black', lw=1, alpha=0.8, label='BR')
        ax[0].axhline(0, color='grey')
        ax[0].legend()
        # ax[2].plot(L3_data['Epoch'].data, rho, color='blue')
        print(bincounts_ham.shape, bincounts.shape)
        ax[1].hist(bin_cens_dt, bins=final_bin_edges_dt, weights=bincounts_ham/bincounts, color='k', alpha=1.0, histtype='step', lw=3)
        ymin, ymax = ax[1].get_ylim()
        yrange = ymax - ymin
        plot_stacked_bars(ax[1], bin_edges, bincounts, category_counts, yrange)

        # hammogram on B
        axt = ax[0].twinx()
        # axt.hist(bin_cens_dt, bins=final_bin_edges_dt, weights=bincounts_ham/bincounts, color='blue', histtype='step', alpha=1.0, lw=2)
        axt.hist(bin_cens_dt, bins=final_bin_edges_dt, weights=bincounts_ham/bincounts, color='blue', alpha=0.6, lw=2)

        ax[0].set_ylabel('BRTN [nT]')
        # ax[2].set_ylabel('n-spi [cc]')
        ax[1].set_ylabel('Hammogram')
        [axs.grid(True) for axs in ax]

        ax[0].set_xlim([final_bin_edges_dt[0], final_bin_edges_dt[-1]])

        # Define custom tick locations (e.g., every 20th value) and set them
        current_ticks_float = ax[-1].get_xticks()
        custom_ticks_datetime = current_ticks_float[::10]
        ax[-1].set_xticks(custom_ticks_datetime)

        # # Format the labels (e.g., to show only Hour:Minute) and set them
        # labels = [i.strftime("%H:%M") for i in ticks_to_use]
        # ax.set_xticklabels(labels)

        # ax[0].set_xlim([np.datetime64(trange[0]), np.datetime64(trange[-1])])
        ax[1].text(0.01, 0.93, f'Time bin={tbin_in_sec} [s]', transform=ax[1].transAxes,
                        fontsize=12, color='black', ha='left', va='top', fontweight='bold')

        plt.subplots_adjust(left=0.08, right=0.95, bottom=0.15, top=0.98)

        # # saving the hammerhead count plotting information
        # hammogram_dict = {}
        # hammogram_dict['bin_cens_dt'] = bin_cens_dt
        # hammogram_dict['bin_edges_dt'] = final_bin_edges_dt
        # hammogram_dict['bincounts'] = bincounts
        # write_pickle(hammogram_dict, f'hammogram_{trange[0]}-{trange[1]}')

if __name__=='__main__':
    # tstart = '2024-07-01/16:30:00'
    # tend   = '2024-07-01/17:20:00'
    tstart = '2020-01-29/21:00:00'
    tend   = '2020-01-29/21:20:00'

    og_only_flag = False

    # get all dates
    start_date = re.split('[/]', tstart)[0]
    end_date = re.split('[/]', tend)[0]
    dates = get_dates(start_date, end_date)

    dirname = f'plots/{start_date}_{end_date}'

    if(os.path.exists(f'{dirname}')): pass
    else:
        os.mkdir(f'{dirname}')

    components = np.array(['core', 'neck', 'hammer'])

    # initializing the dictionary attributes
    global_prop_dict = make_global_prop(components)

    span_data = load_data.span(tstart, tend)

    for date_idx, date in enumerate(dates):
        # loading the pickle file for that particular date
        filename = f'hamstring_{date.year:04d}-{date.month:02d}-{date.day:02d}'
        hammerdict = read_pickle(filename)

        # setting up the data loading process [processing will happen one day at a time]
        span_data.start_new_day(date_idx)

        try:
            # retrieving the necessary parameters for that day and appending to the global list
            hammer_props = misc_fn.extract_params(hammerdict, span_data, og_only=og_only_flag, min_hammer_cells=0)
            mince_props(hammer_props, global_prop_dict, components)
        except: continue

    # plotting the density moments for different components
    plot_density()
    plot_velocity()
    plot_T_ani()
    plot_T_ani_2d_hist()
    plot_as_function_of_distance(color=global_prop_dict['T_ani']['hammer']/global_prop_dict['T_ani']['hammer'].max())
    make_brazil_plots()
    make_fish_plots()

    write_pickle(global_prop_dict, f'{dirname}/global_prop_dict')

