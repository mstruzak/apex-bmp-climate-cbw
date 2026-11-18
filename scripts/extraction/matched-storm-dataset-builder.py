"""
Paired storm dataset builder

Storms are identified ONCE, on the historical base-run PRCP (1991-1999).
  - Each historical event is pairable by storm_id
  - Recurrences that fall below the wet threshold in the future are kept with ~0 load 

"""

import os
import glob
import numpy as np
import pandas as pd

# -- USER SETTINGS --------------------------------------------------------------
CONT_DIR = # path to continuous future runs
HIST_DIR = # path to reference scenarios
OUT_CSV  = # path to output + filename

IDENTIFY_FLOOR_MM = 25.0     # peak-day depth to include a historical event
WET_DAY_MM        = 1.0      # daily PRCP above this defines a wet day (spell boundary)
SEARCH_DAYS       = 5        # +/- window (beyond the lag shift) to snap a future recurrence

REF_FIRST_YEAR  = 1991       # future 2021 == historical 1991 (diagnostic: corr 1.00)
HIST_START_YEAR = 1991       # detect on the reference decade
HIST_END_YEAR   = 1999       # 2000 isn't in the hist run; cycle_pos 9 won't appear
REF_DECADE_LEN  = 10
FUT_DECADE_STARTS = [2021, 2031, 2041, 2051, 2061, 2071, 2081]   # tiles of the reference

# measured calendar-drift lag (days) by decade start, from leap-year accumulation
LAG_BY_DECADE = {
    2021: -2, 2031: -2,
    2041: -3, 2051: -3,
    2061: -4, 2071: -4,
    2081: -5,
}

REGIONS = ["a", "cp", "p", "vr"]
CROPS   = ["corn", "soy", "wheat", "alfalfa"]
BMPS    = ["cc", "gb", "ma", "nm", "nt"]
RCPS    = ["RCP45", "RCP85"]
SCENARIOS = ["historical"] + RCPS
RCP_FOLDER = {"RCP45": "RCP45_2021-2090", "RCP85": "RCP85_2021-2090"}

CC_CROPS = ["corn", "soy"]
COVER_CPNM = "WWHT"

ATLAS14 = {
    "a":  {2: 62,  5: 81,  10: 96,  25: 116, 50: 131, 100: 147},
    "cp": {2: 81,  5: 107, 10: 128, 25: 161, 50: 189, 100: 221},
    "p":  {2: 77,  5: 98,  10: 117, 25: 146, 50: 172, 100: 202},
    "vr": {2: 71,  5: 88,  10: 103, 25: 124, 50: 141, 100: 160},
}
RETURN_PERIODS = [2, 5, 10, 25, 50, 100]

# -- FOLDER HELPERS -------------------------------------------------------------
def fut_run_dir(rcp, region, crop, run):
    return os.path.join(CONT_DIR, RCP_FOLDER[rcp], crop,
                        f"{region}-{crop}", f"{region}-{crop}-{run}")

def hist_run_dir(region, crop, run):
    return os.path.join(HIST_DIR, crop, f"{region}-{crop}",
                        f"{region}-{crop}-{run}-1990-2000")    # VERIFY naming

def run_dir(scenario, region, crop, run):
    if scenario == "historical":
        return hist_run_dir(region, crop, run)
    return fut_run_dir(scenario, region, crop, run)

def baseline_run(bmp):
    return "ma-base" if bmp == "ma" else "base"

def opc_path(region, crop, run):
    rd = hist_run_dir(region, crop, run)          # OPC schedule identical across periods
    f = glob.glob(rd + "/*.opc") + glob.glob(rd + "/*.OPC")
    return f[0] if f else None

# -- SAD --------------------------------------------------------------------------
LOAD_COLS  = ["PRCP", "Q", "YN", "QN", "YP", "QP", "MUSL"]
STATE_COLS = ["HUI", "LAI", "BIOM", "STL", "STD"]

def load_sad(rd):
    f = glob.glob(rd + "/*.SAD") + glob.glob(rd + "/*.sad")
    s = pd.read_csv(f[0], skiprows=9, encoding="latin-1", sep=r"\s+")
    s.columns = s.columns.str.strip().str.lstrip("#")
    for c in LOAD_COLS + STATE_COLS:
        if c in s.columns:
            s[c] = pd.to_numeric(s[c], errors="coerce")
    s["date"] = pd.to_datetime(dict(year=s.Y, month=s.M, day=s.D), errors="coerce")
    return s.dropna(subset=["date"])

def daily_loads(sad):
    d = sad.drop_duplicates(["Y", "M", "D"]).copy()       # one row/day (CC runs carry 2)
    d["TN"] = d.YN + d.QN
    d["TP"] = d.YP + d.QP
    d["sediment"] = d.MUSL
    return d.set_index("date")[["PRCP", "Q", "YN", "QN", "TN", "YP", "QP", "TP", "sediment"]]

def cover_state_on(sad, day):
    if sad is None:
        return {f"cover_{c}": np.nan for c in STATE_COLS}
    r = sad[(sad.date == day) & (sad.CPNM == COVER_CPNM)]
    if r.empty:
        return {f"cover_{c}": np.nan for c in STATE_COLS}
    r = r.iloc[0]
    return {f"cover_{c}": r[c] for c in STATE_COLS}

# -- ACY --------------------------------------------------------------------------
ACY_COLS = ["SA#", "ID", "YR", "YR#", "CPNM", "YLDG", "YLDF", "BIOM", "WS", "NS", "PS", "KS",
            "TS", "AS", "SS", "ZNO3", "ZQP", "AP15", "ZOC", "OCPD", "RSDP", "ARSD", "IRGA",
            "FN", "FP", "FNMN", "FNMA", "FNO", "FPL", "FPO", "YTHS", "YWTH"]

def load_acy(rd):
    f = glob.glob(rd + "/*.acy") + glob.glob(rd + "/*.ACY")
    if not f:
        return None
    a = pd.read_csv(f[0], skiprows=9, header=None, names=ACY_COLS, encoding="latin-1", sep=r"\s+")
    a["YLDG"] = pd.to_numeric(a.YLDG, errors="coerce").fillna(0)
    a["YLDF"] = pd.to_numeric(a.YLDF, errors="coerce").fillna(0)
    a["real_year"] = pd.to_numeric(a.YR, errors="coerce")
    a["crop_failure"] = ((a.YLDG + a.YLDF) == 0).astype(int)
    return a[["real_year", "CPNM", "YLDG", "YLDF", "crop_failure"]]

def get_cash_yield(acy, yr):
    if acy is None:
        return 0.0, 0.0, np.nan
    row = acy[(acy.real_year == yr) & (acy.CPNM != COVER_CPNM)]
    if row.empty:
        return 0.0, 0.0, np.nan
    return row.YLDG.values[0], row.YLDF.values[0], row.crop_failure.values[0]

# -- OPS / phase / lags -----------------------------------------------------------
OP_KEYWORDS = {"plant": ["plant"], "fertilizer": ["fertilizer"],
               "kill": ["kill", "harvest"], "tillage": ["tillage"]}

def classify_crop(d):
    return "cover" if "(cc)" in d.lower() else "cash"

def classify_op(d):
    d = d.lower()
    for ot, kws in OP_KEYWORDS.items():
        if any(k in d for k in kws):
            return ot
    return None

def load_ops(path, real_years):
    if path is None:
        return pd.DataFrame(columns=["date", "op_type", "crop_role"])
    raw = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            if i < 2:
                continue
            p = line.split()
            if len(p) < 4:
                continue
            try:
                yr, mo, day = int(p[0]), int(p[1]), int(p[2])
            except ValueError:
                continue
            ot = classify_op(" ".join(p[3:]))
            if ot is None:
                continue
            raw.append({"int_yr": yr, "mo": mo, "day": day, "op_type": ot,
                        "crop_role": classify_crop(" ".join(p[3:]))})
    if not raw:
        return pd.DataFrame(columns=["date", "op_type", "crop_role"])
    raw = pd.DataFrame(raw)
    n = raw.int_yr.max()
    rows = []
    for off, yr in enumerate(real_years):
        m = (off % n) + 1
        for _, r in raw[raw.int_yr == m].iterrows():
            try:
                rows.append({"date": pd.Timestamp(yr, r.mo, r.day),
                            "op_type": r.op_type, "crop_role": r.crop_role})
            except ValueError:
                continue
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

def compute_lags(storms_df, ops_df):
    storms_df = storms_df.copy()

    def days_since(s, dates):
        if len(dates) == 0:
            return np.nan
        prior = dates[dates <= np.datetime64(s)]
        return np.nan if len(prior) == 0 else (s - pd.Timestamp(prior[-1])).days

    specs = {"cash_plant_lag": ("plant", "cash"), "cash_fert_lag": ("fertilizer", "cash"),
             "cash_kill_lag": ("kill", "cash"), "cover_plant_lag": ("plant", "cover"),
             "cover_kill_lag": ("kill", "cover")}
    for col, (ot, role) in specs.items():
        dts = ops_df[(ops_df.op_type == ot) & (ops_df.crop_role == role)]["date"].sort_values().values
        storms_df[col] = storms_df["start"].apply(lambda s: days_since(s, dts))

    cash = ops_df[ops_df.crop_role == "cash"]
    pm = {"plant": "post-cash-plant", "fertilizer": "post-cash-fert",
          "kill": "post-cash-harvest", "tillage": "post-cash-tillage"}

    def phase(s):
        prior = cash[cash.date <= s]
        return "pre-season" if prior.empty else pm.get(prior.iloc[-1].op_type, "pre-season")

    storms_df["crop_phase"] = storms_df["start"].apply(phase)
    return storms_df

# -- HISTORICAL STORM IDENTIFICATION ----------------------------------------------
def identify_storms(prcp, floor):
    d = prcp.to_frame("p")
    d["wet"] = d.p > WET_DAY_MM
    d["spell"] = (d.wet != d.wet.shift()).cumsum()
    out = []
    for _, g in d.groupby("spell"):
        if not g.wet.iloc[0]:
            continue
        if g.p.max() < floor:
            continue
        st, en = g.index[0], g.index[-1]
        out.append({"start": st, "end": en, "duration": len(g), "peak_24h": g.p.max(),
                    "event_total_depth": g.p.sum(),
                    "antecedent": prcp.loc[st - pd.Timedelta(days=7):st - pd.Timedelta(days=1)].sum()})
    return pd.DataFrame(out)

def assign_return_period(depth, region):
    ds = [ATLAS14[region][rp] for rp in RETURN_PERIODS]
    if depth < ds[0]:
        return np.nan
    if depth >= ds[-1]:
        return 100
    for i in range(len(RETURN_PERIODS) - 1):
        if ds[i] <= depth < ds[i + 1]:
            t = (depth - ds[i]) / (ds[i + 1] - ds[i])
            return round(RETURN_PERIODS[i] + t * (RETURN_PERIODS[i + 1] - RETURN_PERIODS[i]), 1)
    return np.nan

# -- FUTURE WINDOW: lag-center on the known drift, then snap to nearest wet spell -
def future_window(prcp, exp_start, exp_end, year):
    lag = LAG_BY_DECADE.get((year // 10) * 10, -2)
    a_start = exp_start + pd.Timedelta(days=lag)          # expected location after drift
    dur = (exp_end - exp_start).days
    seg = prcp.loc[a_start - pd.Timedelta(days=SEARCH_DAYS):
                   a_start + pd.Timedelta(days=SEARCH_DAYS + dur)]
    wet = seg > WET_DAY_MM
    if seg.empty or wet.sum() == 0:
        return exp_start, exp_end, 0.0, 0.0               # gone dry -> non-productive
    grp = (wet != wet.shift()).cumsum()
    best = None
    for _, g in seg[wet].groupby(grp[wet]):
        d = abs((g.index[0] - a_start).days)
        if best is None or d < best[0]:
            best = (d, g.index[0], g.index[-1], float(g.sum()), float(g.max()))
    return best[1], best[2], best[3], best[4]

_sad = {}
def csad(rd):
    if rd not in _sad:
        _sad[rd] = load_sad(rd)
    return _sad[rd]

# -- MAIN -----------------------------------------------------------------------
def main():
    records = []

    for region in REGIONS:
        for crop in CROPS:

            # 1) storms identified ONCE, on the historical base run
            hbase = hist_run_dir(region, crop, "base")
            if not os.path.isdir(hbase):
                print(f"Missing historic base: {hbase}")
                continue
            try:
                hist_sad = csad(hbase)
                hist_daily = daily_loads(hist_sad)
                hist_prcp = hist_daily["PRCP"]
                hist_prcp = hist_prcp[(hist_prcp.index.year >= HIST_START_YEAR) &
                                      (hist_prcp.index.year <= HIST_END_YEAR)]
                storms = identify_storms(hist_prcp, IDENTIFY_FLOOR_MM)
            except Exception as e:
                print(f"Historic identify error {region}/{crop}: {e}")
                continue
            if storms.empty:
                continue

            storms["return_period"] = storms.peak_24h.apply(lambda d: assign_return_period(d, region))
            storms["cycle_pos"] = storms.start.dt.year - REF_FIRST_YEAR
            storms["storm_id"] = storms.apply(
                lambda st: f"{region}_{crop}_{st.cycle_pos}_{st.start.strftime('%Y%m%d')}", axis=1)

            # 2) lags/phase from date-fixed ops: cash ops from base OPC, cover ops from CC OPC
            try:
                cash_ops = load_ops(opc_path(region, crop, "base"), range(HIST_START_YEAR - 1, HIST_END_YEAR + 1))
                cover_ops = load_ops(opc_path(region, crop, "cc"), range(HIST_START_YEAR - 1, HIST_END_YEAR + 1))
                ops_all = pd.concat([cash_ops[cash_ops.crop_role == "cash"],
                                     cover_ops[cover_ops.crop_role == "cover"]]).sort_values("date")
                storms = compute_lags(storms, ops_all)
            except Exception as e:
                print(f"OPC error {region}/{crop}: {e}")
                for c in ["crop_phase", "cash_plant_lag", "cash_fert_lag", "cash_kill_lag",
                          "cover_plant_lag", "cover_kill_lag"]:
                    storms[c] = np.nan

            # 3) loop BMPs and scenarios
            for bmp in BMPS:
                if bmp == "cc" and crop not in CC_CROPS:
                    continue
                base_run = baseline_run(bmp)

                for scenario in SCENARIOS:
                    if scenario == "historical":
                        bdir = hist_run_dir(region, crop, base_run)
                        mdir = hist_run_dir(region, crop, bmp)
                    else:
                        bdir = fut_run_dir(scenario, region, crop, base_run)
                        mdir = fut_run_dir(scenario, region, crop, bmp)
                    if not (os.path.isdir(bdir) and os.path.isdir(mdir)):
                        print(f"Missing run: {scenario}/{region}/{crop}/{bmp}")
                        continue

                    try:
                        bl_daily = daily_loads(csad(bdir))
                        bm_daily = daily_loads(csad(mdir))
                    except Exception as e:
                        print(f"SAD error {scenario}/{region}/{crop}/{bmp}: {e}")
                        continue
                    bacy, macy = load_acy(bdir), load_acy(mdir)

                    cc_sad = None
                    if crop in CC_CROPS:
                        cc_run = (hist_run_dir(region, crop, "cc") if scenario == "historical"
                                 else fut_run_dir(scenario, region, crop, "cc"))
                        if os.path.isdir(cc_run):
                            cc_sad = csad(cc_run)

                    for _, st in storms.iterrows():
                        c = st.cycle_pos

                        if scenario == "historical":
                            targets = [(st.start.year, st.start, st.end, st.event_total_depth, st.peak_24h, 0)]
                        else:
                            targets = []
                            for ds in FUT_DECADE_STARTS:
                                ty = ds + c
                                shift = ty - st.start.year
                                exp_s = st.start + pd.DateOffset(years=shift)
                                exp_e = st.end + pd.DateOffset(years=shift)
                                s, e, depth, peak = future_window(bl_daily["PRCP"], exp_s, exp_e, ty)
                                off = (s - exp_s).days
                                targets.append((ty, s, e, depth, peak, off))

                        for ty, s, e, depth, peak, off in targets:
                            loads = {}
                            for poll in ["TN", "YN", "QN", "TP", "YP", "QP", "sediment"]:
                                bl = bl_daily.loc[s:e, poll].sum()
                                bm = bm_daily.loc[s:e, poll].sum()
                                loads[f"baseline_{poll}"] = bl
                                loads[f"bmp_{poll}"] = bm
                                loads[f"re_{poll}"] = (bl - bm) / bl * 100 if bl > 0 else np.nan

                            byg, byf, bfail = get_cash_yield(bacy, ty)
                            myg, myf, mfail = get_cash_yield(macy, ty)

                            rec = {
                                "storm_id": st.storm_id,
                                "scenario": scenario, "region": region, "crop": crop, "bmp": bmp,
                                "sim_year": ty, "cycle_pos": c,
                                "start": s, "end": e, "duration": (e - s).days + 1,
                                "peak_24h": peak, "event_total_depth": depth,
                                "mapped_offset_days": off,
                                "return_period": st.return_period, "antecedent": st.antecedent,
                                "event_runoff_Q": bl_daily.loc[s:e, "Q"].sum(),
                                "crop_phase": st.crop_phase,
                                "cash_plant_lag": st.cash_plant_lag, "cash_fert_lag": st.cash_fert_lag,
                                "cash_kill_lag": st.cash_kill_lag, "cover_plant_lag": st.cover_plant_lag,
                                "cover_kill_lag": st.cover_kill_lag,
                                **loads,
                                "baseline_yldg": byg, "baseline_yldf": byf, "baseline_crop_failure": bfail,
                                "bmp_yldg": myg, "bmp_yldf": myf, "bmp_crop_failure": mfail,
                            }
                            rec.update(cover_state_on(cc_sad, s))
                            records.append(rec)

    df = pd.DataFrame(records)
    df.to_csv(OUT_CSV, index=False)
    print(f"Done -- {len(df):,} rows, {df['storm_id'].nunique():,} unique events "
          f"({df['scenario'].value_counts().to_dict()})")


if __name__ == "__main__":
    main()