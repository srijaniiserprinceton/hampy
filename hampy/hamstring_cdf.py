#!/usr/bin/env python3
"""
hamstring_cdf.py
================

Turn a hammerfinder day-wise pickle into a CDF file with metadata:

    pkl/hamstring_<YYYY-MM-DD>.pkl   ->   cdf/hamstring_<YYYY-MM-DD>_<version>.cdf

This consolidates the three earlier scripts
(create_hamstring_cdf_with_metadata.py, batch_create_hamstring_cdfs.py,
process_new_dates.py) into one reusable module:

  * `create_hamstring_cdf(date_str, ...)` builds one day's CDF and returns a
    result dict (it never raises; failures land in result['message']).
  * Run this file directly for one date, a date range, or every pickle in a
    directory (see the CLI at the bottom).

hammerfinder.py imports `create_hamstring_cdf` and calls it per day.

Credentials: SPAN-I downloads use the username/password passed in via
`credentials=[user, pass]`, or the repo-root `.credentials` file (same format
hampy already uses). Nothing is hard-coded.
"""

import glob, pickle, os, traceback, cdflib
from pathlib import Path
from datetime import datetime
from time import time
import numpy as np
import pandas as pd
from cdflib import CDF, cdfwrite

# --- physical constants -----------------------------------------------------
SPAN_RSUN_KM = 695700.0                 # km per solar radius
TEMP_CONV_FAC = 1.0 / (1e6 / 11600.0)   # divide 1e6-K temps by this to get eV

_MOMENT_SRC = {'core': 'core_moments', 'neck': 'neck_moments', 'ham': 'hammer_moments'}
_TENSOR_COMPS = ('Txx', 'Txy', 'Txz', 'Tyy', 'Tyz', 'Tzz')

# Full ordered list of data variables (everything except 'epoch').
_DATA_VARS = [
    'n_core', 'n_neck', 'n_ham',
    'vx_inst_core', 'vy_inst_core', 'vz_inst_core',
    'vx_inst_neck', 'vy_inst_neck', 'vz_inst_neck',
    'vx_inst_ham', 'vy_inst_ham', 'vz_inst_ham',
    'temp_core', 'temp_neck', 'temp_ham',
    'Tperp_core', 'Tpar_core', 'Tperp_neck', 'Tpar_neck', 'Tperp_ham', 'Tpar_ham',
    'Bx_inst', 'By_inst', 'Bz_inst', 'sun_dist_rsun',
]


def read_pickle(fname):
    with open(fname, 'rb') as handle:
        return pickle.load(handle)


def find_Tanisotropy(T_XX, T_YY, T_ZZ, T_XY, T_XZ, T_YZ, B_X, B_Y, B_Z, spi_epoch, hardham_list):
    """Project the temperature tensor onto B to get T_parallel and T_perp per hammer epoch."""
    T_YX, T_ZX, T_ZY = T_XY, T_XZ, T_YZ
    T_par, T_perp = [], []
    for k in range(len(T_XX)):
        i = np.argmin(np.abs(spi_epoch - hardham_list[k]))
        Sum = (B_X[i] * B_X[i] * T_XX[k] + B_X[i] * B_Y[i] * T_XY[k] + B_X[i] * B_Z[i] * T_XZ[k]
               + B_Y[i] * B_X[i] * T_YX[k] + B_Y[i] * B_Y[i] * T_YY[k] + B_Y[i] * B_Z[i] * T_YZ[k]
               + B_Z[i] * B_X[i] * T_ZX[k] + B_Z[i] * B_Y[i] * T_ZY[k] + B_Z[i] * B_Z[i] * T_ZZ[k])
        denom = B_X[i] ** 2 + B_Y[i] ** 2 + B_Z[i] ** 2
        Tpar = Sum / denom
        T_par.append(Tpar)
        T_perp.append(((T_XX[k] + T_YY[k] + T_ZZ[k]) - Tpar) / 2.0)
    return T_par, T_perp


def get_variable_metadata():
    """Units / descriptions / attributes for every CDF variable."""
    kms = lambda axis, pop, popname: {
        'UNITS': 'km/s', 'VAR_TYPE': 'data', 'SCALETYP': 'linear',
        'CATDESC': f'{axis}-component velocity of {pop} proton population in instrument coordinates',
        'FIELDNAM': f'{popname} Proton Velocity {axis}'}
    dens = lambda pop, popname: {
        'UNITS': 'cm^-3', 'VAR_TYPE': 'data', 'SCALETYP': 'linear',
        'CATDESC': f'Density of {pop} proton population', 'FIELDNAM': f'{popname} Proton Density'}
    temp = lambda pop, popname, kind, kindname: {
        'UNITS': 'eV', 'VAR_TYPE': 'data', 'SCALETYP': 'linear',
        'CATDESC': f'Temperature {kind} of {pop} proton population',
        'FIELDNAM': f'{popname} Proton Temperature{kindname}'}
    bfield = lambda axis: {
        'UNITS': 'nT', 'VAR_TYPE': 'data', 'SCALETYP': 'linear',
        'CATDESC': f'{axis}-component of magnetic field in instrument coordinates',
        'FIELDNAM': f'Magnetic Field {axis} Component'}

    meta = {
        'epoch': {
            'UNITS': 'ns', 'VAR_TYPE': 'support_data', 'SCALETYP': 'linear',
            'CATDESC': 'Time, center of measurement interval, in TT2000', 'FIELDNAM': 'Epoch',
            'VALIDMIN': 630720000000000000, 'VALIDMAX': 946728000000000000, 'MONOTON': 'INCREASE'},
        'n_core': dens('core', 'Core'), 'n_neck': dens('neck', 'Neck'), 'n_ham': dens('hammerhead', 'Hammerhead'),
        'temp_core': temp('core', 'Core', '', ''), 'temp_neck': temp('neck', 'Neck', '', ''),
        'temp_ham': temp('hammerhead', 'Hammerhead', '', ''),
        'Tperp_core': temp('core', 'Core', 'perpendicular to magnetic field', ' Perpendicular'),
        'Tpar_core':  temp('core', 'Core', 'parallel to magnetic field', ' Parallel'),
        'Tperp_neck': temp('neck', 'Neck', 'perpendicular to magnetic field', ' Perpendicular'),
        'Tpar_neck':  temp('neck', 'Neck', 'parallel to magnetic field', ' Parallel'),
        'Tperp_ham':  temp('hammerhead', 'Hammerhead', 'perpendicular to magnetic field', ' Perpendicular'),
        'Tpar_ham':   temp('hammerhead', 'Hammerhead', 'parallel to magnetic field', ' Parallel'),
        'Bx_inst': bfield('X'), 'By_inst': bfield('Y'), 'Bz_inst': bfield('Z'),
        'sun_dist_rsun': {
            'UNITS': 'R_sun', 'VAR_TYPE': 'data', 'SCALETYP': 'linear',
            'CATDESC': 'Distance from Sun in solar radii', 'FIELDNAM': 'Solar Distance'},
    }
    for pop, popname in (('core', 'Core'), ('neck', 'Neck'), ('ham', 'Hammerhead')):
        for axis in ('x', 'y', 'z'):
            meta[f'v{axis}_inst_{pop}'] = kms(axis.upper(), pop if pop != 'ham' else 'hammerhead', popname)
    return meta


def _find_spani_l3_file(date_str, credentials=None, download=True):
    """Locate the local SPAN-I L3 moment CDF for `date_str`; optionally download via pyspedas."""
    yyyy, mm = date_str[:4], date_str[5:7]
    no_dash = date_str.replace('-', '')
    psp_data_dir = os.environ.get('PSP_DATA_DIR')
    patterns = [
        psp_data_dir+f"/sweap/spi/l3/spi_sf00_l3_mom/{yyyy}/psp_swp_spi_sf00_*3_mom_{no_dash}_*.cdf",
        psp_data_dir+f"/sweap/spi/L3/spi_sf00/{yyyy}/{mm}/psp_swp_spi_sf00_L3_mom_{no_dash}_*.cdf",
    ]
    for pat in patterns:
        hits = glob.glob(pat)
        if hits:
            return sorted(hits)[-1]
    if not download:
        return None
    import pyspedas
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    trange = [date_obj.strftime('%Y-%m-%d'), (date_obj + pd.Timedelta(days=1)).strftime('%Y-%m-%d')]
    print(f"Downloading SPAN-I L3 moment CDF for {date_str} via pyspedas...")
    kwargs = dict(trange=trange, datatype='spi_sf00_l3_mom', level='l3',
                  downloadonly=True, time_clip=True, get_support_data=True, varnames=['*'])
    if credentials:
        kwargs.update(username=credentials[0], password=credentials[1])
    try:
        pyspedas.psp.spi(**kwargs)
    except Exception:
        return None
    for pat in patterns:
        hits = glob.glob(pat)
        if hits:
            return sorted(hits)[-1]
    return None


def _midnight_tt2000(date_str):
    """TT2000 epoch for 00:00:00 UTC of the given YYYY-MM-DD (for NaN-filled days)."""
    y, m, d = (int(x) for x in date_str.split('-'))
    return int(np.atleast_1d(cdflib.cdfepoch.compute_tt2000([y, m, d, 0, 0, 0, 0, 0, 0]))[0])


def _write_cdf(cdf_path, data_dict):
    """Write one CDF from an ordered {var_name: 1-D values} dict, applying metadata."""
    var_meta = get_variable_metadata()
    cdf_path.parent.mkdir(parents=True, exist_ok=True)
    with cdfwrite.CDF(str(cdf_path), delete=True) as cdf_out:
        for name, vals in data_dict.items():
            if name == 'epoch':
                arr, dtype = np.array(vals, dtype=np.int64), 33    # CDF_TIME_TT2000
            else:
                arr, dtype = np.array(vals, dtype=np.float64), 45  # CDF_DOUBLE
            spec = {'Variable': name, 'Data_Type': dtype, 'Num_Elements': 1,
                    'Rec_Vary': True, 'Dim_Sizes': []}
            cdf_out.write_var(spec, var_attrs=dict(var_meta.get(name, {})), var_data=arr)


def create_hamstring_cdf(date_str, pkl_dir='pkl', cdf_dir='cdf', version='v02',
                         credentials=None, skip_existing=True, download=True, verbose=True):
    """
    Build cdf/hamstring_<date>_<version>.cdf from pkl/hamstring_<date>.pkl.

    On a day with no detected hammerheads, still writes a CDF containing a single
    NaN-filled record (epoch = 00:00:00 of that day), so the daily archive has no
    gaps. Returns a result dict: date, success, message, data_points,
    file_size_kb, processing_time (and 'traceback' on an unexpected error).
    Never raises.
    """
    pkl_dir, cdf_dir = Path(pkl_dir), Path(cdf_dir)
    result = {'date': date_str, 'success': False, 'message': '',
              'data_points': 0, 'file_size_kb': 0.0, 'processing_time': 0.0}
    t0 = time()
    try:
        pkl_path = pkl_dir / f"hamstring_{date_str}.pkl"
        cdf_path = cdf_dir / f"hamstring_{date_str}_{version}.cdf"

        if skip_existing and cdf_path.exists():
            result.update(success=True, message='CDF already exists (skipped)',
                          file_size_kb=cdf_path.stat().st_size / 1024)
            return result
        if not pkl_path.exists():
            result['message'] = f"pickle not found: {pkl_path}"
            return result

        datham = read_pickle(pkl_path)

        # --- collect only entries that carry moment data (skip bare detections) ---
        cols = {c: [] for c in _DATA_VARS}   # only the scalar/velocity/density ones get filled below
        tens = {grp: {c: [] for c in _TENSOR_COMPS} for grp in ('core', 'neck', 'ham')}
        hardham_list = []
        for key in datham:
            try:
                for grp in ('core', 'neck', 'ham'):
                    m = datham[key][_MOMENT_SRC[grp]]
                    cols[f'n_{grp}'].append(m['n'])
                    cols[f'vx_inst_{grp}'].append(m['Ux'])
                    cols[f'vy_inst_{grp}'].append(m['Uy'])
                    cols[f'vz_inst_{grp}'].append(m['Uz'])
                    for c in _TENSOR_COMPS:
                        tens[grp][c].append(m[c])
                hardham_list.append(key)
            except (KeyError, TypeError):
                continue

        # --- no hammerheads this day: write a single NaN-filled record ---
        if not hardham_list:
            nan_dict = {'epoch': [_midnight_tt2000(date_str)]}
            nan_dict.update({v: [np.nan] for v in _DATA_VARS})
            _write_cdf(cdf_path, nan_dict)
            result.update(success=True, data_points=0,
                          file_size_kb=cdf_path.stat().st_size / 1024,
                          message='no hams this day — wrote NaN-filled CDF')
            return result

        # --- hammerheads present: need the SPAN-I L3 file for B, epoch, sun distance ---
        spani_file = _find_spani_l3_file(date_str, credentials=credentials, download=download)
        if spani_file is None:
            result['message'] = f"no SPAN-I L3 file found (and download failed/disabled) for {date_str}"
            return result

        cdf_in = CDF(spani_file)
        spi_epoch = cdf_in['Epoch'][...]
        sun_dist_rsun = cdf_in['SUN_DIST'][...] / SPAN_RSUN_KM
        magf = cdf_in['MAGF_INST'][...]
        B_X, B_Y, B_Z = magf[:, 0], magf[:, 1], magf[:, 2]

        # tensor components -> eV
        for grp in ('core', 'neck', 'ham'):
            for c in _TENSOR_COMPS:
                tens[grp][c] = np.asarray(tens[grp][c]) / TEMP_CONV_FAC

        temp = {grp: (tens[grp]['Txx'] + tens[grp]['Tyy'] + tens[grp]['Tzz']) / 3.0
                for grp in ('core', 'neck', 'ham')}

        Tpar, Tperp = {}, {}
        for grp in ('core', 'neck', 'ham'):
            Tpar[grp], Tperp[grp] = find_Tanisotropy(
                tens[grp]['Txx'], tens[grp]['Tyy'], tens[grp]['Tzz'],
                tens[grp]['Txy'], tens[grp]['Txz'], tens[grp]['Tyz'],
                B_X, B_Y, B_Z, spi_epoch, hardham_list)

        # map B and solar distance onto the hammer-epoch grid
        bx, by, bz, sdist = [], [], [], []
        for key in hardham_list:
            i = np.argmin(np.abs(spi_epoch - key))
            bx.append(B_X[i]); by.append(B_Y[i]); bz.append(B_Z[i]); sdist.append(sun_dist_rsun[i])

        data_dict = {
            'epoch': hardham_list,
            'n_core': cols['n_core'], 'n_neck': cols['n_neck'], 'n_ham': cols['n_ham'],
            'vx_inst_core': cols['vx_inst_core'], 'vy_inst_core': cols['vy_inst_core'], 'vz_inst_core': cols['vz_inst_core'],
            'vx_inst_neck': cols['vx_inst_neck'], 'vy_inst_neck': cols['vy_inst_neck'], 'vz_inst_neck': cols['vz_inst_neck'],
            'vx_inst_ham': cols['vx_inst_ham'], 'vy_inst_ham': cols['vy_inst_ham'], 'vz_inst_ham': cols['vz_inst_ham'],
            'temp_core': temp['core'], 'temp_neck': temp['neck'], 'temp_ham': temp['ham'],
            'Tperp_core': Tperp['core'], 'Tpar_core': Tpar['core'],
            'Tperp_neck': Tperp['neck'], 'Tpar_neck': Tpar['neck'],
            'Tperp_ham': Tperp['ham'], 'Tpar_ham': Tpar['ham'],
            'Bx_inst': bx, 'By_inst': by, 'Bz_inst': bz,
            'sun_dist_rsun': sdist,
        }

        _write_cdf(cdf_path, data_dict)

        result.update(success=True, data_points=len(hardham_list),
                      file_size_kb=cdf_path.stat().st_size / 1024, message='CDF created')
    except Exception as e:
        result['message'] = f"error: {e}"
        result['traceback'] = traceback.format_exc()
    finally:
        result['processing_time'] = time() - t0

    if verbose:
        tag = 'OK  ' if result['success'] else 'FAIL'
        extra = (f" ({result['data_points']} pts, {result['file_size_kb']:.1f} KB,"
                 f" {result['processing_time']:.1f}s)") if result['success'] else ''
        print(f"[{tag}] {date_str}: {result['message']}{extra}")
    return result


# --------------------------------------------------------------------------- #
# Standalone CLI  (replaces batch_create_hamstring_cdfs.py + process_new_dates.py)
# --------------------------------------------------------------------------- #
def _iter_dates(start, end):
    s, e = datetime.strptime(start, '%Y-%m-%d'), datetime.strptime(end, '%Y-%m-%d')
    out = []
    while s <= e:
        out.append(s.strftime('%Y-%m-%d'))
        s += pd.Timedelta(days=1)
    return out


def _load_credentials():
    cred = Path('.credentials')
    if cred.exists():
        lines = cred.read_text().splitlines()
        if len(lines) >= 2:
            return lines[:2]
    return None


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Create hamstring CDF(s) from day-wise pickles.')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--date', help='single date, YYYY-MM-DD')
    g.add_argument('--range', nargs=2, metavar=('START', 'END'), help='inclusive date range')
    g.add_argument('--all', action='store_true', help='every pickle in --pkl-dir')
    p.add_argument('--pkl-dir', default='pkl')
    p.add_argument('--cdf-dir', default='cdf')
    p.add_argument('--version', default='v02')
    p.add_argument('--overwrite', action='store_true', help='rebuild even if the CDF exists')
    p.add_argument('--no-download', action='store_true', help='use only local SPAN-I files')
    args = p.parse_args()

    if args.date:
        dates = [args.date]
    elif args.range:
        dates = _iter_dates(*args.range)
    else:
        dates = sorted(f.stem.replace('hamstring_', '')
                       for f in Path(args.pkl_dir).glob('hamstring_*.pkl'))

    creds = _load_credentials()
    ok = sum(create_hamstring_cdf(d, pkl_dir=args.pkl_dir, cdf_dir=args.cdf_dir,
                                  version=args.version, credentials=creds,
                                  skip_existing=not args.overwrite,
                                  download=not args.no_download)['success']
             for d in dates)
    print(f"\nDone: {ok}/{len(dates)} succeeded.")
