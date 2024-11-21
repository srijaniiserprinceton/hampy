import numpy as np
import matplotlib.pyplot as plt; plt.style.use('dark_background')
import cdflib, pickle, re
from datetime import datetime, timedelta

from hampy import misc_functions as misc_fn
from hampy import load_data

def read_pickle(fname):
    with open(f'{fname}.pkl', 'rb') as handle:
        x = pickle.load(handle)
    return x

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

    plt.xlim([global_prop_dict['dt_arr'][0], global_prop_dict['dt_arr'][-1]])
    plt.ylim([0, 1e4])
    plt.ylabel(r'Density in $\rm{cm}^{-3}$')
    plt.xlabel(r'Time')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/density_{start_date}_{end_date}.png')
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

    plt.savefig(f'plots/velocity_{start_date}_{end_date}.png')
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
    plt.savefig(f'plots/Tani_{start_date}_{end_date}.png')
    plt.close()

    # making histograms
    plt.figure(figsize=(10,10))

    for component in components:
        plt.hist(global_prop_dict['T_ani'][f'{component}'], range=(0, 10), bins=50,
                 label=f'{component}', histtype='step')

    plt.xlabel(r'$T_{\perp} / T_{||}$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/Tani_histogram_{start_date}_{end_date}.png')
    plt.close()

def plot_T_ani_2d_hist():
    density_hammer = global_prop_dict['hammer_density']
    density_hampy = global_prop_dict['total_hampy_density']
    Tani_hammer = global_prop_dict['T_ani']['hammer']

    plt.figure()
    h1, xe1, ye1, fig1 = plt.hist2d(density_hammer/density_hampy, Tani_hammer, bins=20, range=[[0, 0.08], [0, 8]], cmap='inferno')
    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(15, 8))
    ax[0].contourf(xe1[:-1], ye1[:-1], h1.T, cmap='inferno', levels=20)
    ax[0].set_xlabel(r'$n_{\rm{hammer}} / n_{\rm{total}}$')
    ax[0].set_ylabel(r'$T_{\perp}/T_{||}$')
    ax[0].set_xlim([0, 0.08])
    ax[0].set_ylim([0,8])


    neck_vdrift = global_prop_dict['neck_vdrift']
    hammer_vdrift = global_prop_dict['hammer_vdrift']
    span_Bmag = global_prop_dict['span_Bmag']
    valfven = 21.8 * span_Bmag / np.sqrt(density_hampy)

    plt.figure()
    h2, xe2, ye2, fig2 = plt.hist2d(hammer_vdrift/valfven, Tani_hammer, bins=20, range=[[-3.5, 0.0], [0, 8]], cmap='inferno')
    plt.close()

    ax[1].contourf(xe2[:-1], ye2[:-1], h2.T, cmap='inferno', levels=20)
    ax[1].set_xlabel(r'$V_{\rm{drift,hammer}} / V_{\rm{Alfven}}$')
    ax[1].set_ylabel(r'$T_{\perp}/T_{||}$')
    ax[1].set_xlim([-3.5, -1.0])
    ax[1].set_ylim([0, 8])

    plt.savefig(f'plots/Tani_density_{start_date}_{end_date}.png')

def make_global_prop():
    global_prop_dict['core_density'] = np.array([])
    global_prop_dict['neck_density'] = np.array([])
    global_prop_dict['hammer_density'] = np.array([])
    global_prop_dict['total_hampy_density'] = np.array([])
    global_prop_dict['span_density'] = np.array([])
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
    global_prop_dict['B_vec_inst'] = np.array([])
    global_prop_dict['neck_vdrift'] = np.array([])
    global_prop_dict['hammer_vdrift'] = np.array([])
    global_prop_dict['span_Bmag'] = np.array([])

def mince_props(hammer_dayprops):
    core_density, neck_density, hammer_density, total_hampy_density, span_density, T_perp, T_parallel, T_ani, dt_arr,\
    B_vec_inst, neck_vdrift, hammer_vdrift, span_Bmag, U = hammer_dayprops

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
    global_prop_dict['B_vec_inst'] = np.append(global_prop_dict['B_vec_inst'], B_vec_inst)
    global_prop_dict['neck_vdrift'] = np.append(global_prop_dict['neck_vdrift'], neck_vdrift)
    global_prop_dict['hammer_vdrift'] = np.append(global_prop_dict['hammer_vdrift'], hammer_vdrift)
    global_prop_dict['span_Bmag'] = np.append(global_prop_dict['span_Bmag'], span_Bmag)

if __name__=='__main__':
    tstart = '2020-02-04/00:00:00'
    tend   = '2020-02-04/23:59:59'

    og_only_flag = True

    # get all dates
    start_date = re.split('[/]', tstart)[0]
    end_date = re.split('[/]', tend)[0]
    dates = get_dates(start_date, end_date)

    components = np.array(['core', 'neck', 'hammer'])

    # making the global properties dictionary
    global_prop_dict = {}
    # initializing the dictionary attributes
    make_global_prop()

    span_data = load_data.span(tstart, tend)

    for date_idx, date in enumerate(dates):
        # loading the pickle file for that particular date
        filename = f'hamstring_{date.year:04d}-{date.month:02d}-{date.day:02d}'
        hammerdict = read_pickle(filename)

        # setting up the data loading process [processing will happen one day at a time]
        span_data.start_new_day(date_idx)

        # retrieving the necessary parameters for that day and appending to the global list
        hammer_props = misc_fn.extract_params(hammerdict, span_data, og_only=og_only_flag)
        mince_props(hammer_props)

    # plotting the density moments for different components
    plot_density()
    plot_velocity()
    plot_T_ani()
    plot_T_ani_2d_hist()

