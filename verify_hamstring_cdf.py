#!/usr/bin/env python3
"""
verify_hamstring_cdf.py
=======================

Open hamstring CDF(s), print a readback of variables/metadata, and plot the
main populations as stacked time series. Works on a single daily file or a
multi-day range (daily files are concatenated along time).

Usage:
    # single day
    python verify_hamstring_cdf.py cdf/hamstring_2022-09-06_v02.cdf

    # multi-day range (inclusive); looks for cdf/hamstring_<date>_<version>.cdf
    python verify_hamstring_cdf.py --range 2026-08-31 2026-09-03
    python verify_hamstring_cdf.py --range 2026-08-31 2026-09-03 --cdf-dir cdf --version v02

    # options
    python verify_hamstring_cdf.py --range 2026-08-31 2026-09-03 --save check.png --no-show
"""

import argparse
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import cdflib
import matplotlib.pyplot as plt


def _iter_dates(start, end):
    s, e = datetime.strptime(start, '%Y-%m-%d'), datetime.strptime(end, '%Y-%m-%d')
    out = []
    while s <= e:
        out.append(s.strftime('%Y-%m-%d'))
        s += timedelta(days=1)
    return out


def load_cdf(path):
    """Return ({varname: ndarray}, {varname: units}) for one CDF file."""
    cdf = cdflib.CDF(str(path))
    info = cdf.cdf_info()
    if isinstance(info, dict):           # cdflib < 1.0 (e.g. 0.4.9)
        zvars = info.get('zVariables') or info.get('rVariables') or []
    else:                                # newer cdflib
        zvars = list(info.zVariables) or list(info.rVariables)
    data, units = {}, {}
    for v in zvars:
        data[v] = np.atleast_1d(np.asarray(cdf[v][...]))
        units[v] = str(cdf.varattsget(v).get('UNITS', ''))
    return data, units


def load_range(cdf_dir, start, end, version):
    """Load every daily CDF in [start, end] and concatenate along time (sorted by epoch)."""
    datasets, units, found, missing = [], {}, [], []
    for d in _iter_dates(start, end):
        path = Path(cdf_dir) / f"hamstring_{d}_{version}.cdf"
        if not path.exists():
            missing.append(d)
            continue
        dd, units = load_cdf(path)
        datasets.append(dd)
        found.append(d)

    if missing:
        print(f"[skip] {len(missing)} day(s) with no CDF: {', '.join(missing)}")
    if not datasets:
        raise SystemExit(f"No CDF files found in {cdf_dir} for {start}..{end} (version {version}).")

    keys = datasets[0].keys()
    combined = {k: np.concatenate([ds[k] for ds in datasets]) for k in keys}
    order = np.argsort(combined['epoch'])          # keep time monotonic across files
    combined = {k: v[order] for k, v in combined.items()}
    print(f"[ok] loaded {len(found)} day(s): {found[0]} .. {found[-1]}")
    return combined, units


def readback(data, units):
    print(f"\n{len(data)} variables, {len(data['epoch'])} records:\n")
    print(f"  {'variable':<16}{'units':<10}{'npts':>7}{'finite':>8}   range")
    print("  " + "-" * 62)
    for v, arr in data.items():
        if v == 'epoch':
            t0 = cdflib.cdfepoch.to_datetime(arr[0])[0]
            t1 = cdflib.cdfepoch.to_datetime(arr[-1])[0]
            print(f"  {v:<16}{units.get(v,''):<10}{len(arr):>7}{'-':>8}   {t0}  ..  {t1}")
        else:
            a = np.asarray(arr, dtype=float)
            nfin = int(np.isfinite(a).sum())
            rng = f"{np.nanmin(a):.3g} .. {np.nanmax(a):.3g}" if nfin else "all NaN"
            print(f"  {v:<16}{units.get(v,''):<10}{a.size:>7}{nfin:>8}   {rng}")


def plot(data, save=None, show=True):
    t = cdflib.cdfepoch.to_datetime(data['epoch'])
    get = lambda name: np.asarray(data[name], dtype=float)

    if np.all(~np.isfinite(get('n_ham'))):
        print("\n[note] No hammerheads in this range (all NaN). Nothing to plot.")
        return

    panels = [
        ('Density [cm$^{-3}$]', [('n_core', 'core'), ('n_neck', 'neck'), ('n_ham', 'ham')]),
        ('Temperature [eV]',    [('temp_core', 'core'), ('temp_neck', 'neck'), ('temp_ham', 'ham')]),
        ('Ham T [eV]',          [('Tpar_ham', r'$T_\parallel$'), ('Tperp_ham', r'$T_\perp$')]),
        ('B inst [nT]',         [('Bx_inst', 'Bx'), ('By_inst', 'By'), ('Bz_inst', 'Bz')]),
        ('Sun dist [R$_\\odot$]', [('sun_dist_rsun', 'r')]),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 11), sharex=True)
    for ax, (ylabel, series) in zip(axes, panels):
        for name, label in series:
            ax.plot(t, get(name), '.', ms=4, label=label)
        ax.set_ylabel(ylabel)
        ax.legend(loc='upper right', fontsize=8, ncol=len(series))
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel('Time (UTC)')
    d0 = np.datetime_as_string(np.datetime64(t[0]), unit='D')
    d1 = np.datetime_as_string(np.datetime64(t[-1]), unit='D')
    span = d0 if d0 == d1 else f"{d0} .. {d1}"
    fig.suptitle(f"hamstring CDF  —  {span}  ({len(t)} records)")
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=130, bbox_inches='tight')
        print(f"\nSaved figure -> {save}")
    if show:
        plt.show()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Verify and plot hamstring CDF(s).')
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('cdf_file', nargs='?', help='path to a single hamstring_*.cdf file')
    src.add_argument('--range', nargs=2, metavar=('START', 'END'),
                     help='inclusive date range YYYY-MM-DD YYYY-MM-DD (daily files concatenated)')
    p.add_argument('--cdf-dir', default='cdf', help='dir holding daily CDFs (default: cdf)')
    p.add_argument('--version', default='v02', help='file version tag (default: v02)')
    p.add_argument('--save', metavar='PNG', help='save the figure to this path')
    p.add_argument('--no-show', action='store_true', help='do not open an interactive window')
    args = p.parse_args()

    if args.range:
        data, units = load_range(args.cdf_dir, args.range[0], args.range[1], args.version)
    else:
        data, units = load_cdf(args.cdf_file)

    readback(data, units)
    plot(data, save=args.save, show=not args.no_show)
