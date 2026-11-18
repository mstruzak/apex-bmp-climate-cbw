"""
Build all_annual_results.csv from APEX1501 SAD + ACY outputs.

For every scenario folder:
  - read the daily .SAD file, keep only cash-crop rows, drops the spinup years, and sum the daily fluxes to annual totals
  - read the annual .ACY file for YLDG/YLDF/BIOM
  - compute removal efficiencies (RE) for each BMP relative to its baseline

"""

import os
import glob
import pandas as pd


# CONFIG

ROOT = # path to scenario directory
OUTPUT_PATH = os.path.join(ROOT, "annual_results.csv")

# period folder -> (scenario label, first year to KEEP; earlier years = spinup)
PERIODS = {
    "1990-2000":       ("reference period", 1990),
    "RCP45_2021-2090": ("RCP 45", 2021),
    "RCP85_2021-2090": ("RCP 85", 2021),
}

CROPS   = ["alfalfa", "corn", "soy", "wheat"]
REGIONS = ["a", "cp", "vr", "p"]

# cash-crop name (CPNM) used to pick the correct row inside each file
CROP_CPNM = {"alfalfa": "ALFA", "corn": "CORN", "soy": "SOYB", "wheat": "WWHT"}

# 0-indexed column positions (verified against the sample files)
SAD_SKIPROWS = 10                 # 9 preamble lines + 1 header line
SAD_YEAR, SAD_MONTH, SAD_DAY, SAD_CPNM = 2, 3, 4, 5
SAD_SUM_COLS = {                  # daily fluxes -> summed to annual totals
    "PRCP": 23, "Q": 25, "WYLD": 29, "MUSL": 30, "ET": 33,
    "YP": 38, "QP": 39, "YN": 59, "QN": 60,
}

ACY_SKIPROWS = 9                  # 8 preamble lines + 1 header line
ACY_YEAR, ACY_CPNM = 2, 4
ACY_COLS = {"YLDG": 5, "YLDF": 6, "BIOM": 7}

RE_VARS = ["Q", "WYLD", "MUSL", "TN", "TP"]
# which baseline each BMP is compared against
REF_BMP = {"cc": "base", "gb": "base", "nm": "base", "nt": "base", "ma": "ma-base"}

FINAL_COLS = (
    ["scenario", "crop", "region", "bmp", "year",
     "PRCP", "Q", "WYLD", "MUSL", "ET", "YN", "QN", "YP", "QP", "TN", "TP",
     "YLDG", "YLDF", "BIOM"]
    + ["RE_" + v for v in RE_VARS]
)


# read one scenario -> annual table (year + all variables)

def read_scenario(sad_path, acy_path, cpnm, year_min, bmp):
    # daily SAD -> annual sums
    sad = pd.read_csv(sad_path, sep=r"\s+", skiprows=SAD_SKIPROWS, header=None)
    sad = sad[(sad[SAD_CPNM] == cpnm) & (sad[SAD_YEAR] >= year_min)]

    if bmp == "cc":
        sad = sad.drop_duplicates(subset=[SAD_YEAR, SAD_MONTH, SAD_DAY])

    annual = sad.groupby(SAD_YEAR)[list(SAD_SUM_COLS.values())].sum()
    annual.columns = list(SAD_SUM_COLS.keys())
    annual = annual.reset_index().rename(columns={SAD_YEAR: "year"})
    annual["TN"] = annual["YN"] + annual["QN"]
    annual["TP"] = annual["YP"] + annual["QP"]

    # annual ACY -> yields + biomass
    acy = pd.read_csv(acy_path, sep=r"\s+", skiprows=ACY_SKIPROWS, header=None)
    acy = acy[(acy[ACY_CPNM] == cpnm) & (acy[ACY_YEAR] >= year_min)]
    acy = acy[[ACY_YEAR] + list(ACY_COLS.values())]
    acy.columns = ["year"] + list(ACY_COLS.keys())

    return annual.merge(acy, on="year", how="left")


# walk every scenario folder and collect annual tables
 
rows = []
for period_folder, (scenario, year_min) in PERIODS.items():
    print(f"\n=== {scenario} ({period_folder}) ===")
    for crop in CROPS:
        cpnm = CROP_CPNM[crop]
        for region in REGIONS:
            rc_dir = os.path.join(ROOT, period_folder, crop, f"{region}-{crop}")
            if not os.path.isdir(rc_dir):
                continue
            for scen_folder in sorted(os.listdir(rc_dir)):
                full = os.path.join(rc_dir, scen_folder)
                if not os.path.isdir(full):
                    continue

                # bmp = folder name with the "{region}-{crop}-" prefix and any "-1990-2000" suffix removed
                name = scen_folder.replace("-1990-2000", "")
                prefix = f"{region}-{crop}-"
                if not name.startswith(prefix):
                    continue
                bmp = name[len(prefix):]

                sad_files = glob.glob(os.path.join(full, "*.SAD"))
                acy_files = glob.glob(os.path.join(full, "*.ACY"))
                if not sad_files or not acy_files:
                    print(f"  SKIP (missing SAD/ACY): {scen_folder}")
                    continue

                df = read_scenario(sad_files[0], acy_files[0], cpnm, year_min, bmp)
                df.insert(0, "bmp", bmp)
                df.insert(0, "region", region)
                df.insert(0, "crop", crop)
                df.insert(0, "scenario", scenario)
                rows.append(df)
                print(f"  OK: {scen_folder}  ({len(df)} years)")

df_all = pd.concat(rows, ignore_index=True)

 
# removal efficiencies:  RE = (baseline - bmp) / baseline * 100
# baselines (base, ma-base) get blank RE
 
df_all["ref_bmp"] = df_all["bmp"].map(REF_BMP)

baselines = (
    df_all[df_all["bmp"].isin(["base", "ma-base"])]
    [["scenario", "region", "crop", "year", "bmp"] + RE_VARS]
    .rename(columns={"bmp": "ref_bmp"})
    .rename(columns={v: v + "_ref" for v in RE_VARS})
)

df_all = df_all.merge(
    baselines, on=["scenario", "region", "crop", "year", "ref_bmp"], how="left"
)
for v in RE_VARS:
    # an 0 baseline load yields inf/NaN and is left as-is
    df_all["RE_" + v] = (df_all[v + "_ref"] - df_all[v]) / df_all[v + "_ref"] * 100

df_all = df_all.drop(columns=["ref_bmp"] + [v + "_ref" for v in RE_VARS])

 
# tidy + save
 
df_all = (
    df_all[FINAL_COLS]
    .sort_values(["scenario", "crop", "region", "bmp", "year"])
    .reset_index(drop=True)
)
df_all.to_csv(OUTPUT_PATH, index=False)
print(f"\nWrote {len(df_all)} rows to {OUTPUT_PATH}")