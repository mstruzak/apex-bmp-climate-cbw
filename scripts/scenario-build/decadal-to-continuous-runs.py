"""
Compile discrete decadal climate data into one continuous .dly/.hly/.WP1 set.

Early years (2011-2050) are taken from the existing RCP4.5 files. 
Late years (2051-2090) reuse RCP4.5's SRAD/RH/WSPD structure but swap in TMX/TMN/PRCP (and hourly precip pattern) from the new decadal CSVs for the target RCP.

Writes into one test BMP folder; propagate to the remaining BMP folders for this region/scenario once the output has been checked.
"""

import os
import pandas as pd

 
# CONFIG
 
RCP45_BASE = # path to RCP45 files
TARGET_BASE = # path to RCP85 files
CSV_DIR = # path to output CSV

REGION_STEM = "appalachia_base"
DLY_NAME = f"{REGION_STEM}_2su.dly"
HLY_NAME = f"{REGION_STEM}_2su.hly"
WP1_NAME = f"{REGION_STEM}.WP1"

CSV_FILES = [
    "a_RCP85_2051-2060.csv", "a_RCP85_2061-2070.csv",
    "a_RCP85_2071-2080.csv", "a_RCP85_2081-2090.csv",
]

# Test BMP folder -- checked against APEX output before propagating.
OUT_DIR = os.path.join(TARGET_BASE, "corn", "a-corn", "a-corn-base")

REGION_PREFIX = "a"  # matches "a-corn", "a-corn-base", "a-soy-cc", etc.

# Set True only after confirming the test folder's APEX output looks right.
PROPAGATE = False

EARLY_YEARS = (2011, 2050)  # taken from RCP4.5
LATE_YEARS = (2051, 2090)   # TMX/TMN/PRCP replaced from CSV_FILES

 
# READ / WRITE HELPERS
 

def read_dly(path):
    return pd.read_fwf(
        path,
        colspecs=[(2, 6), (6, 10), (10, 14), (14, 20), (20, 26), (26, 32), (32, 38), (38, 44), (44, 50)],
        header=None,
        names=['year', 'month', 'day', 'srad', 'tmax', 'tmin', 'prcp', 'rh', 'wspd'],
        encoding='latin-1',
    ).apply(pd.to_numeric, errors='coerce')


def write_dly(df, path):
    with open(path, 'w') as f:
        for _, r in df.iterrows():
            f.write(f"  {int(r.year):4d}{int(r.month):4d}{int(r.day):4d}"
                    f"{r.srad:6.2f}{r.tmax:6.2f}{r.tmin:6.2f}"
                    f"{r.prcp:6.2f}{r.rh:6.2f}{r.wspd:6.2f}\n")


def read_hly(path):
    return pd.read_csv(path, sep=r'\s+', header=None, encoding='latin-1',
                        names=['year', 'month', 'day', 'hour', 'prcp'])


def write_hly(df, path):
    with open(path, 'w') as f:
        for _, r in df.iterrows():
            f.write(f"{int(r.year):4d}{int(r.month):4d}{int(r.day):4d}"
                    f"{r.hour:10.6f}{r.prcp:10.3f}\n")


def write_wp1(dly_df, template_path, out_path, from_year=2021):
    """Recompute TMX/TMN/SDMX/SDMN/PRCP from dly_df (>= from_year), keep template's other lines."""
    d = dly_df[dly_df['year'] >= from_year].copy()

    monthly = d.groupby('month').agg(
        tmx=('tmax', 'mean'), tmn=('tmin', 'mean'),
        sdmx=('tmax', 'std'), sdmn=('tmin', 'std'),
    ).reset_index()

    annual_prcp = d.groupby(['year', 'month'])['prcp'].sum().reset_index()
    monthly_prcp = annual_prcp.groupby('month')['prcp'].mean().reset_index()
    monthly = monthly.merge(monthly_prcp, on='month').sort_values('month').fillna(0)

    with open(template_path, 'r') as f:
        lines = f.readlines()

    def fmt(values, label):
        return "".join(f"{v:10.2f}" for v in values) + f" {label}\n"

    lines[2] = fmt(monthly['tmx'].tolist(), "TMX")
    lines[3] = fmt(monthly['tmn'].tolist(), "TMN")
    lines[4] = fmt(monthly['sdmx'].tolist(), "SDMX")
    lines[5] = fmt(monthly['sdmn'].tolist(), "SDMN")
    lines[6] = fmt(monthly['prcp'].tolist(), "PRCP")

    with open(out_path, 'w') as f:
        f.writelines(lines)

 
# BUILD
 

def load_daily_csv(csv_dir, csv_files):
    csv = pd.concat([pd.read_csv(os.path.join(csv_dir, f)) for f in csv_files], ignore_index=True)
    daily = csv.groupby(['year', 'month', 'day']).agg(
        tmax=('temp', 'max'), tmin=('temp', 'min'), prcp=('precip', 'sum')
    ).reset_index()
    return csv, daily


def build_dly(rcp45_dly, daily_csv, early_years, late_years):
    early = rcp45_dly[rcp45_dly['year'].between(*early_years)].copy()

    late_structure = rcp45_dly[rcp45_dly['year'].between(*late_years)][['year', 'month', 'day', 'srad', 'rh', 'wspd']]
    late = late_structure.merge(daily_csv, on=['year', 'month', 'day'], how='left')

    # Leap-day mismatches between the RCP4.5 structure and the new CSV show up as NaN - fill from the prior day.
    n_missing = late['tmax'].isna().sum()
    print(f"DLY rows needing leap-day fill: {n_missing}")
    late[['tmax', 'tmin', 'prcp']] = late[['tmax', 'tmin', 'prcp']].ffill()

    late = late[['year', 'month', 'day', 'srad', 'tmax', 'tmin', 'prcp', 'rh', 'wspd']]
    assert late.isna().sum().sum() == 0, "NaN values remain in late_dly"

    new_dly = pd.concat([early, late], ignore_index=True)
    print(f"New .dly: {len(new_dly)} rows")
    return new_dly


def build_hly(rcp45_hly, csv, early_years, late_years):
    early = rcp45_hly[rcp45_hly['year'].between(*early_years)].copy()

    late_structure = rcp45_hly[rcp45_hly['year'].between(*late_years)].copy()

    # Cumulative-within-day
    rcp45_inc = late_structure.copy()
    rcp45_inc['prcp'] = (rcp45_inc.groupby(['year', 'month', 'day'])['prcp']
                          .diff().fillna(rcp45_inc['prcp']))

    csv_inc = csv[['year', 'month', 'day', 'hour', 'precip']].rename(columns={'precip': 'prcp'})

    late = rcp45_inc.merge(csv_inc, on=['year', 'month', 'day', 'hour'],
                            how='left', suffixes=('_rcp45', '_csv'))
    late['prcp'] = late['prcp_csv'].fillna(late['prcp_rcp45'])
    late = late[['year', 'month', 'day', 'hour', 'prcp']]
    assert late['prcp'].isna().sum() == 0
    assert len(late) == len(late_structure)

    # Back to cumulative-within-day for writing.
    late['prcp'] = late.groupby(['year', 'month', 'day'])['prcp'].cumsum()

    new_hly = pd.concat([early, late], ignore_index=True)
    print(f"New .hly: {len(new_hly)} rows")
    return new_hly

 
# PROPAGATE
 

def find_bmp_folders(scenario_base, region_prefix):
    """All BMP folders for a region, across every crop, within one scenario base dir."""
    bmp_folders = []
    for crop in os.listdir(scenario_base):
        crop_path = os.path.join(scenario_base, crop)
        if not os.path.isdir(crop_path):
            continue
        for region_crop in os.listdir(crop_path):
            if not region_crop.startswith(region_prefix + "-"):
                continue
            rc_path = os.path.join(crop_path, region_crop)
            if not os.path.isdir(rc_path):
                continue
            for bmp in os.listdir(rc_path):
                bmp_path = os.path.join(rc_path, bmp)
                if os.path.isdir(bmp_path):
                    bmp_folders.append(bmp_path)
    return bmp_folders


def propagate_climate_files(bmp_folders, new_dly, new_hly, wp1_template_path):
    for bmp_path in bmp_folders:
        write_dly(new_dly, os.path.join(bmp_path, DLY_NAME))
        write_hly(new_hly, os.path.join(bmp_path, HLY_NAME))
        write_wp1(new_dly, wp1_template_path, os.path.join(bmp_path, WP1_NAME))
    print(f"Propagated climate files to {len(bmp_folders)} BMP folders")

 
# MAIN
 

def main():
    rcp45_dly_path = os.path.join(RCP45_BASE, "corn", "a-corn", "a-corn-base", DLY_NAME)
    rcp45_hly_path = os.path.join(RCP45_BASE, "corn", "a-corn", "a-corn-base", HLY_NAME)
    rcp45_wp1_path = os.path.join(RCP45_BASE, "corn", "a-corn", "a-corn-base", WP1_NAME)

    csv, daily_csv = load_daily_csv(CSV_DIR, CSV_FILES)
    rcp45_dly = read_dly(rcp45_dly_path)
    rcp45_hly = read_hly(rcp45_hly_path)

    new_dly = build_dly(rcp45_dly, daily_csv, EARLY_YEARS, LATE_YEARS)
    new_hly = build_hly(rcp45_hly, csv, EARLY_YEARS, LATE_YEARS)

    write_dly(new_dly, os.path.join(OUT_DIR, DLY_NAME))
    write_hly(new_hly, os.path.join(OUT_DIR, HLY_NAME))
    write_wp1(new_dly, rcp45_wp1_path, os.path.join(OUT_DIR, WP1_NAME))

    print(f"\nDone. Test files written to:\n  {OUT_DIR}")

    if PROPAGATE:
        bmp_folders = find_bmp_folders(TARGET_BASE, REGION_PREFIX)
        propagate_climate_files(bmp_folders, new_dly, new_hly, rcp45_wp1_path)
    else:
        print("Next: rerun APEX in this folder, check SAD output.")
        print("Once confirmed, set PROPAGATE = True and rerun to write to all region BMP folders.")


if __name__ == "__main__":
    main()