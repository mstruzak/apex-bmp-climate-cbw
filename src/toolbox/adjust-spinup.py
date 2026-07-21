"""
Extend spinup in a continuous APEX climate file by duplicating the first N years of the file
Assumes the input .dly/.hly/.WP1 already have the intended spinup years at the front (e.g. 2011-2020 data), followed by the continuous simulation period (e.g. 2021-2090).
Takes the first N_SPINUP_YEARS years, repeats them SPINUP_CYCLES times, and relabels the repeated years so they land contiguously right before the simulation period
  
  1. Set paths and N_SPINUP_YEARS / SPINUP_CYCLES below
  2. Run script
  3. Copy output files into the target BMP folder
  4. Set APEXCONT.DAT line 1 NBYR to N_SPINUP_YEARS * SPINUP_CYCLES + n_future_years,
     IYR to the new starting year printed at the end of this script
"""

import os
import numpy as np
from collections import defaultdict

 
# CONFIG
 

DLY_IN = # path to existing dly file
HLY_IN = # path to existing hly file
WP1_IN = # path to existing wp1 file

OUT_DIR = # where to output new weather files

N_SPINUP_YEARS = 10   # how many years at the front of the file to treat as the block to duplicate
SPINUP_CYCLES  = 1    # how many times to repeat that block

 
# DLY
 

def rebuild_dly(path, out_path, n_spinup_years, spinup_cycles):
    with open(path, 'r') as f:
        lines = f.readlines()

    by_year = defaultdict(list)
    for line in lines:
        yr = int(line[2:6])
        by_year[yr].append(line)

    years_sorted = sorted(by_year.keys())
    spinup_years = years_sorted[:n_spinup_years]
    future_years = years_sorted[n_spinup_years:]
    future_start = future_years[0]

    total_spinup_years = n_spinup_years * spinup_cycles
    target_yr = future_start - total_spinup_years

    out = []
    for _ in range(spinup_cycles):
        for src_yr in spinup_years:
            for line in by_year[src_yr]:
                out.append(f"{target_yr:6d}" + line[6:])
            target_yr += 1

    for yr in future_years:
        out.extend(by_year[yr])

    with open(out_path, 'w') as f:
        f.writelines(out)

    new_start = future_start - total_spinup_years
    print(f"  DLY: {len(out)} lines -> {out_path}")
    print(f"       spinup years {spinup_years} x{spinup_cycles} -> {new_start}-{future_start - 1}, future unchanged from {future_start}")
    return out, new_start

 
# HLY
 

def rebuild_hly(path, out_path, n_spinup_years, spinup_cycles):
    with open(path, 'r') as f:
        lines = f.readlines()

    by_year = defaultdict(list)
    for line in lines:
        yr = int(line[0:4])
        by_year[yr].append(line)

    years_sorted = sorted(by_year.keys())
    spinup_years = years_sorted[:n_spinup_years]
    future_years = years_sorted[n_spinup_years:]
    future_start = future_years[0]

    total_spinup_years = n_spinup_years * spinup_cycles
    target_yr = future_start - total_spinup_years

    out = []
    for _ in range(spinup_cycles):
        for src_yr in spinup_years:
            for line in by_year[src_yr]:
                out.append(f"{target_yr:4d}" + line[4:])
            target_yr += 1

    for yr in future_years:
        out.extend(by_year[yr])

    with open(out_path, 'w') as f:
        f.writelines(out)
    print(f"  HLY: {len(out)} lines -> {out_path}")

 
# WP1: recompute monthly stats from the rebuilt .dly
 

def rebuild_wp1(wp1_path, dly_lines, out_path):
    monthly = defaultdict(lambda: {'tmax': [], 'tmin': [], 'prcp': 0.0, 'rain_days': 0})
    seen_years = set()

    for line in dly_lines:
        yr = int(line[2:6])
        mo = int(line[6:10])
        tmax = float(line[20:26])
        tmin = float(line[26:32])
        prcp = float(line[32:38])

        monthly[mo]['tmax'].append(tmax)
        monthly[mo]['tmin'].append(tmin)
        monthly[mo]['prcp'] += prcp
        if prcp > 0.5:
            monthly[mo]['rain_days'] += 1
        seen_years.add(yr)

    n_years = len(seen_years)

    with open(wp1_path, 'r') as f:
        wp1 = f.readlines()

    def fmt_line(values, label):
        return ''.join(f'{v:10.2f}' for v in values) + f' {label}\n'

    for mo in range(1, 13):
        m = monthly[mo]
        m['mean_tmax'] = np.mean(m['tmax'])
        m['mean_tmin'] = np.mean(m['tmin'])
        m['sd_tmax'] = np.std(m['tmax'], ddof=1)
        m['sd_tmin'] = np.std(m['tmin'], ddof=1)
        m['mean_prcp'] = m['prcp'] / n_years
        m['mean_dayp'] = m['rain_days'] / n_years

    wp1[2] = fmt_line([monthly[m]['mean_tmax'] for m in range(1, 13)], 'TMX')
    wp1[3] = fmt_line([monthly[m]['mean_tmin'] for m in range(1, 13)], 'TMN')
    wp1[4] = fmt_line([monthly[m]['sd_tmax'] for m in range(1, 13)], 'SDMX')
    wp1[5] = fmt_line([monthly[m]['sd_tmin'] for m in range(1, 13)], 'SDMN')
    wp1[6] = fmt_line([monthly[m]['mean_prcp'] for m in range(1, 13)], 'PRCP')
    wp1[11] = fmt_line([monthly[m]['mean_dayp'] for m in range(1, 13)], 'DAYP')

    with open(out_path, 'w') as f:
        f.writelines(wp1)
    print(f"  WP1: {len(wp1)} lines -> {out_path}")

 
# MAIN
 

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    dly_stem = os.path.splitext(os.path.basename(DLY_IN))[0]
    hly_stem = os.path.splitext(os.path.basename(HLY_IN))[0]
    wp1_stem = os.path.splitext(os.path.basename(WP1_IN))[0]

    dly_out_path = os.path.join(OUT_DIR, f"{dly_stem}.dly")
    hly_out_path = os.path.join(OUT_DIR, f"{hly_stem}.hly")
    wp1_out_path = os.path.join(OUT_DIR, f"{wp1_stem}.WP1")

    print("Building climate files with duplicated front-of-file spinup...")
    dly_lines, new_start = rebuild_dly(DLY_IN, dly_out_path, N_SPINUP_YEARS, SPINUP_CYCLES)
    rebuild_hly(HLY_IN, hly_out_path, N_SPINUP_YEARS, SPINUP_CYCLES)
    rebuild_wp1(WP1_IN, dly_lines, wp1_out_path)

    total_spinup_years = N_SPINUP_YEARS * SPINUP_CYCLES
    print(f"\nDone. Output in {OUT_DIR}")
    print(f"Set APEXCONT.DAT line 1 IYR to {new_start}, NBYR to {total_spinup_years} + (# future years)")

if __name__ == "__main__":
    main()
