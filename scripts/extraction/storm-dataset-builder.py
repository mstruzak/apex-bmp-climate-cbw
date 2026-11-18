"""
Independent Storm dataset builder 
 - days_since_last_storm: end of previous storm -> start of current, within each region x crop x scenario sequence. NaN for first storm.
 - rfv_storm:   DHY RFV summed over [start:end] (base run).
 - dhy_Q_storm: DHY Q  summed over [start:end] (base run) 
 Return-period based on PRCP depth
"""
import os, glob
import numpy as np
import pandas as pd

# -- USER SETTINGS --------------------------------------------------------------
CONT_DIR = # path to continuous future runs
HIST_DIR = # path to historic runs
OUT_CSV  = # output path + filename csv

IDENTIFY_FLOOR_MM = 25.0
WET_DAY_MM        = 1.0
FUT_START_YEAR, FUT_END_YEAR = 2021, 2090
HIST_START_YEAR, HIST_END_YEAR = 1990, 1999
REF_DECADE_LEN = 10

REGIONS = ["a", "cp", "p", "vr"]
CROPS   = ["corn", "soy", "wheat", "alfalfa"]
BMPS    = ["cc", "gb", "ma", "nm", "nt"]
RCPS    = ["RCP45", "RCP85"]
SCENARIOS = ["historical"] + RCPS
RCP_FOLDER = {"RCP45": "RCP45_2021-2090", "RCP85": "RCP85_2021-2090"}

CC_CROPS = ["corn", "soy"]
COVER_CPNM   = "WWHT"
CASH_IS_WHEAT = {"wheat"}

ATLAS14 = {
    "a":  {2: 62,  5: 81,  10: 96,  25: 116, 50: 131, 100: 147},
    "cp": {2: 81,  5: 107, 10: 128, 25: 161, 50: 189, 100: 221},
    "p":  {2: 77,  5: 98,  10: 117, 25: 146, 50: 172, 100: 202},
    "vr": {2: 71,  5: 88,  10: 103, 25: 124, 50: 141, 100: 160},
}
RETURN_PERIODS = [2, 5, 10, 25, 50, 100]

# -- FOLDERS --------------------------------------------------------------------
def fut_run_dir(rcp, region, crop, run):
    return os.path.join(CONT_DIR, RCP_FOLDER[rcp], crop, f"{region}-{crop}", f"{region}-{crop}-{run}")
def hist_run_dir(region, crop, run):
    return os.path.join(HIST_DIR, crop, f"{region}-{crop}", f"{region}-{crop}-{run}-1990-2000")
def run_dir(scenario, region, crop, run):
    return hist_run_dir(region, crop, run) if scenario == "historical" else fut_run_dir(scenario, region, crop, run)
def baseline_run(bmp): return "ma-base" if bmp == "ma" else "base"
def year_window(scenario):
    return (HIST_START_YEAR, HIST_END_YEAR) if scenario == "historical" else (FUT_START_YEAR, FUT_END_YEAR)
def opc_path(region, crop, run):
    rd = hist_run_dir(region, crop, run)           # OPC schedule identical across periods
    f = glob.glob(rd + "/*.opc") + glob.glob(rd + "/*.OPC")
    return f[0] if f else None

# -- SAD / DHY ------------------------------------------------------------------
LOAD_COLS  = ["PRCP","Q","YN","QN","YP","QP","MUSL"]
STATE_COLS = ["HUI","LAI","BIOM","STL","STD"]
DHY_COLS = ['ISA','NBSA','Y','M','D','CN','SCI','RFV','STMP2','SML','Q','SSF',
            'QRF','RSSF','WYLD','QRB','TC','DUR','ALTC','AL5','REP','RZSW','GWST']

def load_sad(rd):
    f = glob.glob(rd + "/*.SAD") + glob.glob(rd + "/*.sad")
    s = pd.read_csv(f[0], skiprows=9, encoding="latin-1", sep=r"\s+")
    s.columns = s.columns.str.strip().str.lstrip("#")
    for c in LOAD_COLS + STATE_COLS:
        if c in s.columns:
            s[c] = pd.to_numeric(s[c], errors="coerce")
    s["date"] = pd.to_datetime(dict(year=s.Y, month=s.M, day=s.D), errors="coerce")
    return s.dropna(subset=["date"])

def load_dhy(rd):
    f = glob.glob(rd + "/*.DHY") + glob.glob(rd + "/*.dhy")
    d = pd.read_csv(f[0], skiprows=10, sep=r"\s+", names=DHY_COLS, engine="python")
    for c in ["RFV","Q"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["date"] = pd.to_datetime(dict(year=d.Y, month=d.M, day=d.D), errors="coerce")
    return d.dropna(subset=["date"]).set_index("date")[["RFV","Q"]].sort_index()

def daily_loads(sad):
    d = sad.drop_duplicates(["Y","M","D"]).copy()        # one row/day
    d["TN"] = d.YN + d.QN
    d["TP"] = d.YP + d.QP
    d["sediment"] = d.MUSL
    return d.set_index("date")[["PRCP","Q","YN","QN","TN","YP","QP","TP","sediment"]]

def cover_state_on(sad, day):
    r = sad[(sad.date == day) & (sad.CPNM == COVER_CPNM)]
    if r.empty:
        return {f"cover_{c}": np.nan for c in STATE_COLS}
    r = r.iloc[0]
    return {f"cover_{c}": r[c] for c in STATE_COLS}

# -- ACY ------------------------------------------------------------------------
ACY_COLS = ["SA#","ID","YR","YR#","CPNM","YLDG","YLDF","BIOM","WS","NS","PS","KS","TS","AS","SS",
            "ZNO3","ZQP","AP15","ZOC","OCPD","RSDP","ARSD","IRGA","FN","FP","FNMN","FNMA","FNO","FPL","FPO","YTHS","YWTH"]
def load_acy(rd):
    f = glob.glob(rd + "/*.acy") + glob.glob(rd + "/*.ACY")
    if not f: return None
    a = pd.read_csv(f[0], skiprows=9, header=None, names=ACY_COLS, encoding="latin-1", sep=r"\s+")
    a["YLDG"] = pd.to_numeric(a.YLDG, errors="coerce").fillna(0)
    a["YLDF"] = pd.to_numeric(a.YLDF, errors="coerce").fillna(0)
    a["real_year"] = pd.to_numeric(a.YR, errors="coerce")
    a["crop_failure"] = ((a.YLDG + a.YLDF) == 0).astype(int)
    return a[["real_year","CPNM","YLDG","YLDF","crop_failure"]]
def get_cash_yield(acy, yr):
    if acy is None: return 0.0, 0.0, np.nan
    row = acy[(acy.real_year == yr) & (acy.CPNM != COVER_CPNM)]
    if row.empty: return 0.0, 0.0, np.nan
    return row.YLDG.values[0], row.YLDF.values[0], row.crop_failure.values[0]

# -- OPS / phase / lags ---------------------------------------------------------
OP_KEYWORDS = {"plant":["plant"],"fertilizer":["fertilizer"],"kill":["kill","harvest"],"tillage":["tillage"]}
def classify_crop(d):
    return "cover" if "(cc)" in d.lower() else "cash"
def classify_op(d):
    d = d.lower()
    for ot, kws in OP_KEYWORDS.items():
        if any(k in d for k in kws): return ot
    return None
def load_ops(path, real_years):
    if path is None: return pd.DataFrame(columns=["date","op_type","crop_role"])
    raw = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            if i < 2: continue
            p = line.split()
            if len(p) < 4: continue
            try: yr, mo, day = int(p[0]), int(p[1]), int(p[2])
            except ValueError: continue
            ot = classify_op(" ".join(p[3:]))
            if ot is None: continue
            raw.append({"int_yr": yr, "mo": mo, "day": day, "op_type": ot,
                        "crop_role": classify_crop(" ".join(p[3:]))})
    if not raw: return pd.DataFrame(columns=["date","op_type","crop_role"])
    raw = pd.DataFrame(raw); n = raw.int_yr.max(); rows = []
    for off, yr in enumerate(real_years):
        m = (off % n) + 1
        for _, r in raw[raw.int_yr == m].iterrows():
            try: rows.append({"date": pd.Timestamp(yr, r.mo, r.day),
                              "op_type": r.op_type, "crop_role": r.crop_role})
            except ValueError: continue
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
def compute_lags(storms_df, ops_df):
    storms_df = storms_df.copy()
    def days_since(s, dates):
        if len(dates) == 0: return np.nan
        prior = dates[dates <= np.datetime64(s)]
        return np.nan if len(prior) == 0 else (s - pd.Timestamp(prior[-1])).days
    specs = {"cash_plant_lag": ("plant","cash"), "cash_fert_lag": ("fertilizer","cash"),
             "cash_kill_lag": ("kill","cash"), "cover_plant_lag": ("plant","cover"),
             "cover_kill_lag": ("kill","cover")}
    for col, (ot, role) in specs.items():
        dts = ops_df[(ops_df.op_type == ot) & (ops_df.crop_role == role)]["date"].sort_values().values
        storms_df[col] = storms_df["start"].apply(lambda s: days_since(s, dts))
    cash = ops_df[ops_df.crop_role == "cash"]
    pm = {"plant":"post-cash-plant","fertilizer":"post-cash-fert","kill":"post-cash-harvest","tillage":"post-cash-tillage"}
    def phase(s):
        prior = cash[cash.date <= s]
        return "pre-season" if prior.empty else pm.get(prior.iloc[-1].op_type, "pre-season")
    storms_df["crop_phase"] = storms_df["start"].apply(phase)
    return storms_df

def identify_storms(prcp, floor):
    d = prcp.to_frame("p"); d["wet"] = d.p > WET_DAY_MM
    d["spell"] = (d.wet != d.wet.shift()).cumsum()
    out = []
    for _, g in d.groupby("spell"):
        if not g.wet.iloc[0]: continue
        if g.p.max() < floor: continue
        st, en = g.index[0], g.index[-1]
        out.append({"start": st, "end": en, "duration": len(g), "peak_24h": g.p.max(),
                    "event_total_depth": g.p.sum(),
                    "antecedent": prcp.loc[st - pd.Timedelta(days=7):st - pd.Timedelta(days=1)].sum()})
    return pd.DataFrame(out)

def assign_return_period(depth, region):
    ds = [ATLAS14[region][rp] for rp in RETURN_PERIODS]
    if depth < ds[0]: return np.nan
    if depth >= ds[-1]: return 100
    for i in range(len(RETURN_PERIODS) - 1):
        if ds[i] <= depth < ds[i+1]:
            t = (depth - ds[i]) / (ds[i+1] - ds[i])
            return round(RETURN_PERIODS[i] + t*(RETURN_PERIODS[i+1]-RETURN_PERIODS[i]), 1)
    return np.nan

_sad = {}
def csad(rd):
    if rd not in _sad: _sad[rd] = load_sad(rd)
    return _sad[rd]

_dhy = {}
def cdhy(rd):
    if rd not in _dhy: _dhy[rd] = load_dhy(rd)
    return _dhy[rd]

# -- MAIN -----------------------------------------------------------------------
records = []
for scenario in SCENARIOS:
    y0, y1 = year_window(scenario)
    cyc = HIST_START_YEAR if scenario == "historical" else FUT_START_YEAR
    for region in REGIONS:
        for crop in CROPS:
            base_dir = run_dir(scenario, region, crop, "base")
            if not os.path.isdir(base_dir):
                print(f"Missing base: {scenario}/{region}/{crop}"); continue
            try:
                base_sad_full = csad(base_dir)
                prcp = daily_loads(base_sad_full)["PRCP"]
                prcp = prcp[(prcp.index.year >= y0) & (prcp.index.year <= y1)]
                storms = identify_storms(prcp, IDENTIFY_FLOOR_MM)
            except Exception as e:
                print(f"Identify error {scenario}/{region}/{crop}: {e}"); continue
            if storms.empty: continue
            storms["return_period"] = storms.peak_24h.apply(lambda d: assign_return_period(d, region))
            storms["cycle_pos"] = ((storms.start.dt.year - cyc) % REF_DECADE_LEN)

            # (A) days since last storm: end(prev) -> start(cur), this region x crop x scenario
            storms = storms.sort_values("start").reset_index(drop=True)
            gap = pd.to_datetime(storms["start"]) - pd.to_datetime(storms["end"]).shift(1)
            storms["days_since_last_storm"] = gap.dt.days

            # (B,C) RFV and Q from base-run DHY, summed over [start:end]
            try:
                base_dhy = cdhy(base_dir)
                storms["rfv_storm"]   = storms.apply(lambda st: base_dhy.loc[st["start"]:st["end"], "RFV"].sum(), axis=1)
                storms["dhy_Q_storm"] = storms.apply(lambda st: base_dhy.loc[st["start"]:st["end"], "Q"].sum(), axis=1)
            except Exception as e:
                print(f"DHY error {scenario}/{region}/{crop}: {e}")
                storms["rfv_storm"] = np.nan; storms["dhy_Q_storm"] = np.nan

            # cash ops from base OPC, cover ops from CC OPC
            try:
                cash_ops  = load_ops(opc_path(region, crop, "base"), range(y0-1, y1+1))
                cover_ops = load_ops(opc_path(region, crop, "cc"),   range(y0-1, y1+1))
                ops_all = pd.concat([cash_ops[cash_ops.crop_role == "cash"],
                                     cover_ops[cover_ops.crop_role == "cover"]]).sort_values("date")
                storms = compute_lags(storms, ops_all)
            except Exception as e:
                print(f"OPC error {scenario}/{region}/{crop}: {e}")
                for c in ["crop_phase","cash_plant_lag","cash_fert_lag","cash_kill_lag",
                          "cover_plant_lag","cover_kill_lag"]:
                    storms[c] = np.nan

            cc_sad = None
            if crop in CC_CROPS:
                cc_run = run_dir(scenario, region, crop, "cc")
                if os.path.isdir(cc_run):
                    cc_sad = csad(cc_run)

            for bmp in BMPS:
                if bmp == "cc" and crop not in CC_CROPS:
                    continue
                else:
                    bdir = run_dir(scenario, region, crop, baseline_run(bmp))
                    mdir = run_dir(scenario, region, crop, bmp)
                if not (os.path.isdir(bdir) and os.path.isdir(mdir)):
                    print(f"Missing run: {scenario}/{region}/{crop}/{bmp}"); continue
                try:
                    bl_daily = daily_loads(csad(bdir)); bm_daily = daily_loads(csad(mdir))
                except Exception as e:
                    print(f"SAD error {scenario}/{region}/{crop}/{bmp}: {e}"); continue
                bacy, macy = load_acy(bdir), load_acy(mdir)
                for _, st in storms.iterrows():
                    s, e = st["start"], st["end"]; yr = s.year
                    rec = {"scenario": scenario, "region": region, "crop": crop, "bmp": bmp,
                           "sim_year": yr, "cycle_pos": st.cycle_pos, "start": s, "end": e,
                           "duration": st.duration, "peak_24h": st.peak_24h,
                           "event_total_depth": st.event_total_depth, "return_period": st.return_period,
                           "days_since_last_storm": st.days_since_last_storm,
                           "rfv_storm": st.rfv_storm, "dhy_Q_storm": st.dhy_Q_storm,
                           "antecedent": st.antecedent, "event_runoff_Q": bl_daily.loc[s:e, "Q"].sum(),
                           "crop_phase": st.crop_phase,
                           "cash_plant_lag": st.cash_plant_lag, "cash_fert_lag": st.cash_fert_lag,
                           "cash_kill_lag": st.cash_kill_lag, "cover_plant_lag": st.cover_plant_lag,
                           "cover_kill_lag": st.cover_kill_lag}
                    for poll in ["TN","YN","QN","TP","YP","QP","sediment"]:
                        bl = bl_daily.loc[s:e, poll].sum(); bm = bm_daily.loc[s:e, poll].sum()
                        rec[f"baseline_{poll}"] = bl
                        rec[f"bmp_{poll}"] = bm
                        rec[f"re_{poll}"] = (bl - bm)/bl*100 if bl > 0 else np.nan
                    byg, byf, bf = get_cash_yield(bacy, yr); myg, myf, mf = get_cash_yield(macy, yr)
                    rec.update({"baseline_yldg": byg, "baseline_yldf": byf, "baseline_crop_failure": bf,
                                "bmp_yldg": myg, "bmp_yldf": myf, "bmp_crop_failure": mf})
                    rec.update(cover_state_on(cc_sad, s) if cc_sad is not None
                               else {f"cover_{c}": np.nan for c in STATE_COLS})
                    records.append(rec)

df = pd.DataFrame(records)
print("\n=== crop x bmp combinations built ===")
print(df.groupby(["crop","bmp"]).size().unstack(fill_value=0))
df.to_csv(OUT_CSV, index=False)
print(f"Done -- {len(df):,} rows -> {OUT_CSV}")
print("re_TN by bmp:", df[df.baseline_TN>0].groupby("bmp").re_TN.median().round(1).to_dict())
print("days_since_last_storm non-null:", df.days_since_last_storm.notna().sum(), "/", len(df))
print("rfv_storm non-null:", df.rfv_storm.notna().sum(), "/", len(df))