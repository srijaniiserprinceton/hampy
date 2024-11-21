import numpy as np
import cdflib
import matplotlib.pyplot as plt; plt.ion(); plt.style.use('dark_background')

from hampy import misc_functions as misc_fn

def plot_temperature_anisotropy(hammerdict, span_data, daynum, og_only=True, return_Tani=False):
    T_tensor = {}

    components = ['core', 'neck', 'hammer']

    # finding the times in UTC
    epoch_arr = np.asarray(list(hammerdict.keys()))
    dt_arr = []

    # finding the magnetic fields for rotating the temperature tensor
    B_vec_inst = span_data.L3_data_fullday['MAGF_INST']

    for component in components:
        T_tensor[f'{component}'] = []

        for epoch_idx, epoch in enumerate(hammerdict.keys()):
            try:
                Txx = hammerdict[epoch][f'{component}_moments']['Txx']
                Txy = hammerdict[epoch][f'{component}_moments']['Txy']
                Txz = hammerdict[epoch][f'{component}_moments']['Txz']
                Tyy = hammerdict[epoch][f'{component}_moments']['Tyy']
                Tyz = hammerdict[epoch][f'{component}_moments']['Tyz']
                Tzz = hammerdict[epoch][f'{component}_moments']['Tzz']

                T_tensor[f'{component}'].append((np.asarray([Txx, Tyy, Tzz, Txy, Txz, Tyz])))
                
                if(component == 'core'):
                    dt_arr.append(epoch_arr[epoch_idx])

                    # finding the closest epoch in the magnetic field data
                    Bepoch = np.argmin(np.abs(span_data.L3_data_fullday['epoch'] - epoch))
                    B_vec_inst.append([span_data.L3_data_fullday['MAGF_INST'][Bepoch]])

            except:
                pass

        # T_tensor[f'{component}'] = misc_fn.convert_to_tensor(np.asarray(T_tensor[f'{component}']))

    # Smoothed B-field
    # window = 18

    # Bx_smooth = np.convolve(B_vec_inst[:,0], np.ones(window)/window, 'same')
    # By_smooth = np.convolve(B_vec_inst[:,1], np.ones(window)/window, 'same')
    # Bz_smooth = np.convolve(B_vec_inst[:,2], np.ones(window)/window, 'same')

    # smooth_bvec_inst = np.stack([Bx_smooth, By_smooth, Bz_smooth]).T

    # R = define_time_series_rot_vector(smooth_bvec_inst)

    # R = misc_fn.define_time_series_rot_vector(B_vec_inst)
    # Tcore_fa = misc_fn.rotate_the_magnetic_field_vector(T_tensor['core'], R)
    # Tneck_fa = misc_fn.rotate_the_magnetic_field_vector(T_tensor['neck'], R)
    # Thammer_fa = misc_fn.rotate_the_magnetic_field_vector(T_tensor['hammer'], R)

    # T_parallel_core = Tcore_fa[:,2,2]
    # T_perp_core = Tcore_fa[:,0,0]
    # T_parallel_neck = Tneck_fa[:,2,2]
    # T_perp_neck = Tneck_fa[:,0,0]
    # T_parallel_hammer = Thammer_fa[:,2,2]
    # T_perp_hammer = Thammer_fa[:,0,0]

    # Tani_core = T_perp_core / T_parallel_core
    # Tani_neck = T_perp_neck / T_parallel_neck
    # Tani_hammer = T_perp_hammer / T_parallel_hammer

    Tperp_core, Tparallel_core, Tani_core = misc_fn.find_Tanisotropy(T_tensor['core'], B_vec_inst, Bepoch, list(hammerdict.keys()))
    Tperp_neck, Tparallel_neck, Tani_neck = misc_fn.find_Tanisotropy(T_tensor['neck'], B_vec_inst, Bepoch, list(hammerdict.keys()))
    Tperp_hammer, Tparallel_hammer, Tani_hammer = misc_fn.find_Tanisotropy(T_tensor['hammer'], B_vec_inst, Bepoch, list(hammerdict.keys()))
    
    dt_arr = cdflib.cdfepoch.to_datetime(np.asarray(dt_arr))

    plt.figure(figsize=(10,4))
    plt.plot(dt_arr, Tani_core, '.', label=r'$T_{\rm{ani, core}}$')
    plt.plot(dt_arr, Tani_neck, '.', label=r'$T_{\rm{ani, neck}}$')
    plt.plot(dt_arr, Tani_hammer, '.', label=r'$T_{\rm{ani, hammer}}$')
    plt.legend()
    plt.ylabel(r'$T_{\perp}/T_{||}$')
    plt.xlabel('Time')
    plt.tight_layout()
    plt.savefig(f'plots/Tani_timeseries_{daynum}.pdf')

    plt.figure()
    plt.hist(Tani_neck, range=(0, 20), bins=200, alpha=0.4, label=r'$T_{\rm{ani, neck}}$', histtype='step')
    plt.hist(Tani_hammer, range=(0, 20), bins=200, label=r'$T_{\rm{ani, hammer}}$', histtype='step')
    plt.hist(Tani_core, range=(0, 20), bins=200, label=r'$T_{\rm{ani, core}}$', histtype='step')
    plt.xlim([0,7.5])
    plt.legend()
    plt.xlabel(r'$T_{\perp}/T_{||}$')
    plt.tight_layout()
    plt.savefig(f'plots/Tani_histogram_{daynum}.pdf')


    if(return_Tani):
        return Tani_core, Tani_neck, Tani_hammer

def compare_density(hammerdict, span_data, daynum, og_only=True, return_dens=False):
    core_density = []
    neck_density = []
    hammer_density = []
    total_hampy_density = []
    span_density = []

    # finding the times in UTC
    epoch_arr = np.asarray(list(hammerdict.keys()))
    dt_arr = []

    for epoch_idx, epoch in enumerate(hammerdict.keys()):
        try:
            core_density.append(hammerdict[epoch]['core_moments']['n'])
            neck_density.append(hammerdict[epoch]['neck_moments']['n'])
            hammer_density.append(hammerdict[epoch]['hammer_moments']['n'])

            total_hampy_density.append(hammerdict[epoch]['core_moments']['n'] + 
                                       hammerdict[epoch]['neck_moments']['n'] + 
                                       hammerdict[epoch]['hammer_moments']['n'])
            
            dt_arr.append(epoch_arr[epoch_idx])

            # finding the closest epoch in the magnetic field data
            Bepoch = np.argmin(np.abs(span_data.L3_data_fullday['epoch'] - epoch))
            span_density.append(span_data.L3_data_fullday['DENS'][Bepoch])

        except:
            pass

    span_density = np.asarray(span_density)
    total_hampy_density = np.asarray(total_hampy_density)
    core_density = np.asarray(core_density)
    neck_density = np.asarray(neck_density)
    hammer_density = np.asarray(hammer_density)

    dt_arr = cdflib.cdfepoch.to_datetime(np.asarray(dt_arr))

    # plotting the density comparison
    x = np.linspace(span_density.min(), span_density.max(), 100)
    plt.figure()
    plt.plot(dt_arr, total_hampy_density / span_density, '.')
    plt.ylabel(r'$n_{\rm{hampy}} / n_{\rm{span}}$')
    plt.xlabel('Time')
    plt.tight_layout()
    plt.savefig(f'plots/Density_comparison_{daynum}.pdf')

    if(return_dens):
        return core_density, neck_density, hammer_density, total_hampy_density

def get_vdrift(hammerdict, span_data, daynum, og_only=True):
    neck_vdrift = []
    hammer_vdrift = []
    span_Bmag = []

    # finding the times in UTC
    epoch_arr = np.asarray(list(hammerdict.keys()))
    dt_arr = []

    for epoch_idx, epoch in enumerate(hammerdict.keys()):
        try:
            neck_vdrift.append(hammerdict[epoch]['neck_moments']['Ux'] - hammerdict[epoch]['core_moments']['Ux'])
            hammer_vdrift.append(hammerdict[epoch]['hammer_moments']['Ux'] - hammerdict[epoch]['core_moments']['Ux'])

            dt_arr.append(epoch_arr[epoch_idx])

            # finding the closest epoch in the magnetic field data
            Bepoch = np.argmin(np.abs(span_data.L3_data_fullday['epoch'] - epoch))
            span_Bmag.append(np.linalg.norm(span_data.L3_data_fullday['MAGF_INST'][Bepoch]))

        except:
            pass

    neck_vdrift = np.asarray(neck_vdrift)
    hammer_vdrift = np.asarray(hammer_vdrift)

    # Bmag for computing the Alfven speed later
    span_Bmag = np.asarray(span_Bmag)

    return neck_vdrift, hammer_vdrift, span_Bmag

def plot_Tani_2d_hist(hammerdict, span_data, daynum, og_only=True):
    Tani_core, Tani_neck, Tani_hammer = plot_temperature_anisotropy(hammerdict, span_data, daynum, return_Tani=True)
    density_core, density_neck, density_hammer, density_hampy = compare_density(hammerdict, span_data, daynum, return_dens=True)

    # Figure 2(a) from Verniero et al 2022
    plt.figure()
    h1, xe1, ye1, fig1 = plt.hist2d(density_hammer/density_hampy, Tani_hammer, bins=20, range=[[0, 0.08], [0.5, 5]], cmap='inferno')

    plt.figure()
    plt.contourf(xe1[:-1], ye1[:-1], h1.T, cmap='inferno', levels=20)
    plt.xlabel(r'$n_{\rm{hammer}} / n_{\rm{total}}$')
    plt.ylabel(r'$T_{\perp}/T_{||}$')
    plt.xlim([0, 0.08])
    plt.ylim([0,8])
    plt.colorbar()
    plt.tight_layout()

    # Figure 2(b) from Verniero et al 2022
    neck_vdrift, hammer_vdrift, span_Bmag = get_vdrift(hammerdict, span_data, daynum)

    valfven = 21.8 * span_Bmag / np.sqrt(density_hampy)

    plt.figure()
    h2, xe2, ye2, fig2 = plt.hist2d(hammer_vdrift/valfven, Tani_hammer, bins=20, range=[[-3.25, 0.0], [0.5, 5]], cmap='inferno')

    plt.figure()
    plt.contourf(xe2[:-1], ye2[:-1], h2.T, cmap='inferno', levels=20)
    plt.xlabel(r'$V_{\rm{drift,hammer}} / V_{\rm{Alfven}}$')
    plt.ylabel(r'$T_{\perp}/T_{||}$')
    plt.xlim([-3.5, -1.5])
    plt.ylim([0, 8])
    plt.colorbar()
    plt.tight_layout()