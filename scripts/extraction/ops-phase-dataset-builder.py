"""
Operational-phase RE dataset builder
Builds phase-year (preplant/midseason/postharvest) removal efficiency + covariates from SAD files
+ soil-state variables (ZNO3, ZNH3, ZPML) from MSA files.

PHASE_BOUNDS should be filled in with actual (region, crop) boundary dates before running. 
Operations are date-triggered (not growth/HU triggered) --> boundaries are fixed across every scenario-year

Postharvest windows that wrap across the calendar year (e.g. Oct -> following Mar) are assigned to the phase_year of their START date (the harvest year). 

Soil-state (ZNO3/ZNH3/ZPML): uses a single prior calendar month per phase (median)

outputs csv dataset
"""
import os, glob
import numpy as np
import pandas as pd

# -- USER SETTINGS --------------------------------------------------------------
CONT_DIR = # path to continuous future runs
HIST_DIR = # path to historic runs
OUT_CSV  = # output path + filename csv

FUT_START_YEAR, FUT_END_YEAR   = 2021, 2090
HIST_START_YEAR, HIST_END_YEAR = 1990, 1999

REGIONS = ["a", "cp", "p", "vr"]
CROPS   = ["corn", "soy", "wheat", "alfalfa"]
BMPS    = ["cc", "gb", "ma", "nm", "nt"]
RCPS    = ["RCP45", "RCP85"]
SCENARIOS = ["historical"] + RCPS
RCP_FOLDER = {"RCP45": "RCP45_2021-2090", "RCP85": "RCP85_2021-2090"}

CC_CROPS = ["corn", "soy"]
COVER_CPNM = "WWHT"

#  PHASE BOUNDARIES

PHASE_BOUNDS = {
    ("a", "corn"): {
        "preplant":    (1, 1,  4, 30),
        "midseason":   (5, 1, 10, 1),
        "postharvest": (10, 2, 12, 31),
    },
    ("a", "soy"):     {
        "preplant":    (1, 1,  4, 30),
        "midseason":   (5, 1, 10, 1),
        "postharvest": (10, 2, 12, 31),
        },

    ("cp", "corn"):   {
        "preplant":    (1, 1,  3, 31),
        "midseason":   (4, 1, 10, 1),
        "postharvest": (10, 2, 12, 31),
    },
    ("cp", "soy"):    {
        "preplant":    (1, 1,  4, 30),
        "midseason":   (5, 1, 10, 1),
        "postharvest": (10, 2, 12, 31),
    },

    ("p", "corn"):    {
        "preplant":    (1, 1,  3, 31),
        "midseason":   (4, 1, 10, 1),
        "postharvest": (10, 2, 12, 31),
    },
    ("p", "soy"):     {
        "preplant":    (1, 1,  4, 30),
        "midseason":   (5, 1, 10, 1),
        "postharvest": (10, 2, 12, 31),
    },

    ("vr", "corn"):   {
        "preplant":    (1, 1,  3, 31),
        "midseason":   (4, 1, 10, 1),
        "postharvest": (10, 2, 12, 31),
    },
    ("vr", "soy"):    {
        "preplant":    (1, 1,  4, 30),
        "midseason":   (5, 1, 10, 1),
        "postharvest": (10, 2, 12, 31),
    },

}


MSA_VARS = ["ZNO3", "ZNH3", "ZPML"]

# folder structure
def fut_run_dir(rcp, region, crop, run):
    return os.path.join(CONT_DIR, RCP_FOLDER[rcp], crop, f"{region}-{crop}", f"{region}-{crop}-{run}")
def hist_run_dir(region, crop, run):
    return os.path.join(HIST_DIR, crop, f"{region}-{crop}", f"{region}-{crop}-{run}-1990-2000")
def run_dir(scenario, region, crop, run):
    return hist_run_dir(region, crop, run) if scenario == "historical" else fut_run_dir(scenario, region, crop, run)
def baseline_run(bmp): return "ma-base" if bmp == "ma" else "base"
def year_window(scenario):
    return (HIST_START_YEAR, HIST_END_YEAR) if scenario == "historical" else (FUT_START_YEAR, FUT_END_YEAR)

# Phase asignment
def assign_phase(d, region, crop):
    """Return (phase, phase_year) for one date, using that region/crop's fixed bounds."""
    key = (region, crop)
    if key not in PHASE_BOUNDS:
        raise KeyError(f"No PHASE_BOUNDS entry for {key} -- fill this in before running.")
    md = (d.month, d.day)
    for ph, (sm, sd, em, ed) in PHASE_BOUNDS[key].items():
        start, end = (sm, sd), (em, ed)
        if start <= end:
            if start <= md <= end:
                return ph, d.year
        else:  # window wraps across Dec 31 -> Jan 1
            if md >= start:
                return ph, d.year
            if md <= end:
                return ph, d.year - 1
    return None, None  # outside all defined windows - dropped w warning

def phase_cols(dates, region, crop):
    result = [assign_phase(d, region, crop) for d in dates]
    phase = pd.Series([r[0] for r in result], index=dates)
    phase_year = pd.Series([r[1] for r in result], index=dates)
    return phase, phase_year

# Read SAD variables
LOAD_COLS    = ["PRCP","YN","QN","YP","QP","MUSL"]
WEATHER_COLS = ["TMX","TMN"]
_warned_no_temp = False

def load_sad(rd):
    f = glob.glob(rd + "/*.SAD") + glob.glob(rd + "/*.sad")
    s = pd.read_csv(f[0], skiprows=9, encoding="latin-1", sep=r"\s+")
    s.columns = s.columns.str.strip().str.lstrip("#")
    for c in LOAD_COLS + WEATHER_COLS + ["BIOM"]:
        if c in s.columns:
            s[c] = pd.to_numeric(s[c], errors="coerce")
    s["date"] = pd.to_datetime(dict(year=s.Y, month=s.M, day=s.D), errors="coerce")
    return s.dropna(subset=["date"])

def daily_loads_weather(sad, region, crop):
    """One row/day of loads (+weather if present). Drops the duplicate cover-crop row (CC runs)."""
    global _warned_no_temp
    d = sad.drop_duplicates(["Y","M","D"]).copy()
    d["TN"] = d.YN + d.QN
    d["TP"] = d.YP + d.QP
    d["sediment"] = d.MUSL
    for c in WEATHER_COLS:
        if c not in d.columns:
            d[c] = np.nan
            if not _warned_no_temp:
                print(f"NOTE: {c} not found in SAD header -- temperature columns will be blank.")
                _warned_no_temp = True
    d = d.set_index("date")
    d["phase"], d["phase_year"] = phase_cols(d.index, region, crop)
    n_unassigned = d["phase"].isna().sum()
    if n_unassigned:
        print(f"NOTE: {n_unassigned} days in {region}-{crop} fell outside all phase windows -- dropped.")
    d = d.dropna(subset=["phase"])
    return d[["PRCP","TN","YN","QN","TP","YP","QP","sediment","TMX","TMN","phase","phase_year"]]

def crop_biom(sad, cover, is_cc_crop):
    """Daily BIOM. For corn/soy CC runs, splits cash vs. WWHT cover by CPNM.
       For wheat/alfalfa, WWHT (or ALFA) IS the cash crop -- no cover to exclude."""
    if "CPNM" not in sad.columns or "BIOM" not in sad.columns:
        return pd.Series(dtype=float)
    if cover:
        mask = sad.CPNM == COVER_CPNM
    elif is_cc_crop:
        mask = sad.CPNM != COVER_CPNM
    else:
        mask = pd.Series(True, index=sad.index)
    return sad[mask].drop_duplicates(["Y","M","D"]).set_index("date")["BIOM"]

def phase_biom(biom_series, prefix, y0, y1, region, crop):
    biom_series = biom_series[(biom_series.index.year >= y0) & (biom_series.index.year <= y1)]
    cols = ["phase_year","phase", f"{prefix}_biom_mean", f"{prefix}_biom_median"]
    if biom_series.empty:
        return pd.DataFrame(columns=cols)
    phase, phase_year = phase_cols(biom_series.index, region, crop)
    g = pd.DataFrame({"BIOM": biom_series.values, "phase": phase.values, "phase_year": phase_year.values})
    g = g.dropna(subset=["phase"])
    return g.groupby(["phase_year","phase"])["BIOM"].agg(
        **{f"{prefix}_biom_mean": "mean", f"{prefix}_biom_median": "median"}).reset_index()

# Read ACY
ACY_COLS = ["SA#","ID","YR","YR#","CPNM","YLDG","YLDF","BIOM","WS","NS","PS","KS","TS","AS","SS",
            "ZNO3","ZQP","AP15","ZOC","OCPD","RSDP","ARSD","IRGA","FN","FP","FNMN","FNMA","FNO","FPL","FPO","YTHS","YWTH"]
def load_acy_failure(rd):
    f = glob.glob(rd + "/*.acy") + glob.glob(rd + "/*.ACY")
    if not f: return None
    a = pd.read_csv(f[0], skiprows=9, header=None, names=ACY_COLS, encoding="latin-1", sep=r"\s+")
    a["YLDG"] = pd.to_numeric(a.YLDG, errors="coerce").fillna(0)
    a["YLDF"] = pd.to_numeric(a.YLDF, errors="coerce").fillna(0)
    a["real_year"] = pd.to_numeric(a.YR, errors="coerce")
    a["crop_failure"] = ((a.YLDG + a.YLDF) == 0).astype(int)
    return a.set_index("real_year")["crop_failure"]

# Read MSA
def load_msa_soil_state(rd):
    """Read ZNO3/ZNH3/ZPML monthly values from the MSA file (long/transposed format:
    one row per variable per year, JAN..DEC as columns, no annual total for these three).
    Returns dict: {(year, var): [12 values, Jan..Dec]}"""
    f = glob.glob(rd + "/*.MSA") + glob.glob(rd + "/*.msa")
    if not f:
        return {}
    out = {}
    with open(f[0], encoding="latin-1") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 17:
                continue
            var = parts[4]
            if var not in MSA_VARS:
                continue
            try:
                year = int(parts[2])
                months = [float(x) for x in parts[5:17]]
            except ValueError:
                continue
            out[(year, var)] = months
    return out

def prior_month_for_phase(region, crop, phase):
    """Calendar month (and year offset) immediately preceding this phase's START
    date -- guarantees the soil-state value precedes ALL of that phase's own
    dynamics, not just one point inside it. Derived straight from PHASE_BOUNDS,
    so regional/crop differences (e.g. Appalachia's later calendar) come through
    automatically instead of needing a separate hardcoded table."""
    sm, sd, em, ed = PHASE_BOUNDS[(region, crop)][phase]
    if sm == 1:
        return 12, -1   # December of the previous year
    return sm - 1, 0

def add_msa_soil_state(merged, bdir, mdir, region, crop):
    b_msa = load_msa_soil_state(bdir)
    m_msa = load_msa_soil_state(mdir)
    for var in MSA_VARS:
        base_vals, bmp_vals = [], []
        for row in merged.itertuples():
            month, yoff = prior_month_for_phase(region, crop, row.phase)
            lookup_year = row.phase_year + yoff
            b_months = b_msa.get((lookup_year, var))
            m_months = m_msa.get((lookup_year, var))
            base_vals.append(b_months[month - 1] if b_months else np.nan)
            bmp_vals.append(m_months[month - 1] if m_months else np.nan)
        merged[f"baseline_{var}"] = base_vals
        merged[f"bmp_{var}"] = bmp_vals
    return merged

_sad_cache = {}
def csad(rd):
    if rd not in _sad_cache: _sad_cache[rd] = load_sad(rd)
    return _sad_cache[rd]

BL_AGG = dict(baseline_TN=("TN","sum"), baseline_YN=("YN","sum"), baseline_QN=("QN","sum"),
              baseline_TP=("TP","sum"), baseline_YP=("YP","sum"), baseline_QP=("QP","sum"),
              baseline_sediment=("sediment","sum"),
              precip_total_mm=("PRCP","sum"), precip_mean_daily_mm=("PRCP","mean"),
              precip_median_daily_mm=("PRCP","median"),
              tmax_mean_C=("TMX","mean"), tmin_mean_C=("TMN","mean"), n_days=("PRCP","size"))
BM_AGG = dict(bmp_TN=("TN","sum"), bmp_YN=("YN","sum"), bmp_QN=("QN","sum"),
              bmp_TP=("TP","sum"), bmp_YP=("YP","sum"), bmp_QP=("QP","sum"),
              bmp_sediment=("sediment","sum"))
POLL = ["TN","YN","QN","TP","YP","QP","sediment"]

# -- MAIN ---------------------------------------------------------------------------
if not PHASE_BOUNDS:
    raise ValueError("PHASE_BOUNDS is empty -- fill in the (region, crop) boundary dates before running.")

frames = []
for scenario in SCENARIOS:
    y0, y1 = year_window(scenario)
    for region in REGIONS:
        for crop in CROPS:
            for bmp in BMPS:
                if bmp == "nt" and crop == "alfalfa":
                    continue
                if bmp == "cc" and crop not in CC_CROPS:
                    continue
                bdir = run_dir(scenario, region, crop, baseline_run(bmp))
                mdir = run_dir(scenario, region, crop, bmp)
                if not (os.path.isdir(bdir) and os.path.isdir(mdir)):
                    print(f"Missing run: {scenario}/{region}/{crop}/{bmp}"); continue
                try:
                    b_sad, m_sad = csad(bdir), csad(mdir)
                    bl = daily_loads_weather(b_sad, region, crop)
                    bm = daily_loads_weather(m_sad, region, crop)
                except Exception as e:
                    print(f"SAD error {scenario}/{region}/{crop}/{bmp}: {e}"); continue
                bl = bl[(bl.index.year >= y0) & (bl.index.year <= y1)]
                bm = bm[(bm.index.year >= y0) & (bm.index.year <= y1)]
                if bl.empty or bm.empty:
                    continue

                bl_g = bl.groupby(["phase_year","phase"]).agg(**BL_AGG).reset_index()
                bm_g = bm.groupby(["phase_year","phase"]).agg(**BM_AGG).reset_index()
                merged = bl_g.merge(bm_g, on=["phase_year","phase"], how="inner")

                for p in POLL:
                    blv, bmv = merged[f"baseline_{p}"], merged[f"bmp_{p}"]
                    merged[f"re_{p}"] = np.where(blv > 0, (blv - bmv) / blv * 100, np.nan)

                cash_biom = phase_biom(crop_biom(m_sad, cover=False, is_cc_crop=(crop in CC_CROPS)),
                                        "cash", y0, y1, region, crop)
                merged = merged.merge(cash_biom, on=["phase_year","phase"], how="left")
                if bmp == "cc":
                    cover_biom = phase_biom(crop_biom(m_sad, cover=True, is_cc_crop=True),
                                             "cover", y0, y1, region, crop)
                    merged = merged.merge(cover_biom, on=["phase_year","phase"], how="left")
                else:
                    merged["cover_biom_mean"] = np.nan
                    merged["cover_biom_median"] = np.nan

                b_fail = load_acy_failure(bdir)
                m_fail = load_acy_failure(mdir)
                if b_fail is not None:
                    merged = merged.merge(b_fail.rename_axis("phase_year").reset_index(name="baseline_crop_failure"),
                                           on="phase_year", how="left")
                else:
                    merged["baseline_crop_failure"] = np.nan
                if m_fail is not None:
                    merged = merged.merge(m_fail.rename_axis("phase_year").reset_index(name="bmp_crop_failure"),
                                           on="phase_year", how="left")
                else:
                    merged["bmp_crop_failure"] = np.nan

                merged = add_msa_soil_state(merged, bdir, mdir, region, crop)

                merged.insert(0, "bmp", bmp)
                merged.insert(0, "crop", crop)
                merged.insert(0, "region", region)
                merged.insert(0, "scenario", scenario)
                frames.append(merged)

df = pd.concat(frames, ignore_index=True)
print("\n scenario x bmp combinations built")
print(df.groupby(["scenario","bmp"]).size().unstack(fill_value=0))
df.to_csv(OUT_CSV, index=False)
print(f"Done {len(df):,} rows -> {OUT_CSV}")
print("re_TN by phase:", df[df.baseline_TN > 0].groupby("phase").re_TN.median().round(1).to_dict())
print("n_days range by phase:", df.groupby("phase").n_days.agg(["min","max"]).to_dict())