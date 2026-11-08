"""
Healthcare Access & Resource Optimization Tracker - Step 1: Data preparation

Usage:
    python prepare_data.py --input train.csv --outdir output

Input : AV JanataHack Healthcare Hackathon II train file (Kaggle / OpenML)
Output: output/hospital_ops.db (SQLite) + one CSV per table (for Power BI)

Tables
  admissions      REAL patient/hospital fields + numeric LOS  (+ SYNTHETIC timestamps)
  wards           SYNTHETIC bed counts per hospital/ward
  staff_shifts    SYNTHETIC doctors/nurses per ward per shift
  daily_occupancy DERIVED from admissions + wards

IMPORTANT: timestamps, bed counts and staffing are simulated because the
public dataset does not contain them. State this in your README.
"""
import argparse
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_DAYS = 730                      # simulated calendar length
START = pd.Timestamp("2023-01-01")
ANALYSIS_FROM_DAY = 180           # skip burn-in when sizing wards
UTIL_LOW, UTIL_HIGH = 0.62, 0.88   # typical (median) ward utilisation range used to size beds

REQUIRED = ["case_id", "hospital_code", "ward_type", "type_of_admission", "age", "stay"]


def norm(col: str) -> str:
    return re.sub(r"\s+", "_", col.strip().lower())


def parse_stay(s: str):
    """'21-30' -> (21, 30); 'More than 100 Days' -> (101, None)."""
    s = str(s).strip().lower()
    if "more" in s:
        return 101, np.nan
    m = re.match(r"(\d+)\s*-\s*(\d+)", s)
    if not m:
        return np.nan, np.nan
    return int(m.group(1)), int(m.group(2))


def clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [norm(c) for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing expected columns: {missing}\nFound: {list(df.columns)}")

    df = df.drop_duplicates(subset="case_id").dropna(subset=["stay"]).copy()

    bounds = df["stay"].map(parse_stay)
    df["los_lo"] = [b[0] for b in bounds]
    df["los_hi"] = [b[1] for b in bounds]
    df = df.dropna(subset=["los_lo"]).copy()

    # Model target: bin midpoint (0-10 -> 5, 11-20 -> 15.5, ..., >100 -> 110)
    df["los_days_mid"] = np.where(
        df["los_hi"].isna(), 110.0, (df["los_lo"] + df["los_hi"]) / 2
    )

    # Known gaps in this dataset: fill with most frequent value
    for col in ("bed_grade", "city_code_patient"):
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    df["ward_id"] = df["hospital_code"].astype(int).map("H{:02d}".format) + "-" + df["ward_type"].astype(str)
    return df.reset_index(drop=True)


def simulate_timestamps(df: pd.DataFrame, rng) -> pd.DataFrame:
    n = len(df)
    days = pd.date_range(START, periods=N_DAYS)

    # Mondays busier, weekends quieter
    dow_w = np.array([1.15, 1.0, 1.0, 1.0, 1.05, 0.9, 0.9])
    p = dow_w[days.dayofweek]
    p = p / p.sum()
    day_idx = rng.choice(N_DAYS, size=n, p=p)

    # Hour-of-day profiles: emergencies peak late morning + evening, urgent = daytime
    emerg = np.array([2, 1.5, 1.2, 1, 1, 1.2, 2, 3, 4, 5, 5.5, 5.5, 5, 5, 5, 5, 5.5, 6, 6.5, 6, 5, 4, 3, 2.5])
    urgent = np.array([0.5, 0.3, 0.3, 0.3, 0.5, 1, 2, 4, 7, 8, 8, 7, 6, 6, 6, 5, 4, 3, 2, 1.5, 1, 0.8, 0.6, 0.5])

    hours = np.zeros(n, dtype=int)
    adm_type = df["type_of_admission"].astype(str).str.lower().values
    for is_urgent in (True, False):
        mask = (adm_type == "urgent") if is_urgent else (adm_type != "urgent")
        prof = urgent if is_urgent else emerg
        hours[mask] = rng.choice(24, size=mask.sum(), p=prof / prof.sum())

    minutes = rng.integers(0, 60, size=n)
    admit = days[day_idx] + pd.to_timedelta(hours, unit="h") + pd.to_timedelta(minutes, unit="m")

    # Simulated actual LOS: uniform inside the bin; >100 days gets an exponential tail
    lo, hi = df["los_lo"].values, df["los_hi"].values
    los_sim = np.where(
        np.isnan(hi),
        100 + rng.exponential(20, size=n),
        rng.uniform(lo, np.where(np.isnan(hi), lo, hi) + 1),
    )
    out = df.copy()
    out["admit_ts"] = admit
    out["los_days_sim"] = los_sim.round(2)
    out["discharge_ts"] = admit + pd.to_timedelta(los_sim, unit="D")
    out["admit_hour"] = hours
    out["admit_dow"] = admit.dayofweek
    return out


def build_daily_occupancy(adm: pd.DataFrame):
    """Occupied beds per ward per day, from admit/discharge dates."""
    wards = sorted(adm["ward_id"].unique())
    w_idx = {w: i for i, w in enumerate(wards)}
    width = N_DAYS + 600
    diff = np.zeros((len(wards), width), dtype=np.int32)

    a = ((adm["admit_ts"] - START).dt.floor("D").dt.days).clip(0, width - 1).values
    d = ((adm["discharge_ts"] - START).dt.floor("D").dt.days).clip(0, width - 1).values
    r = adm["ward_id"].map(w_idx).values
    np.add.at(diff, (r, a), 1)
    np.add.at(diff, (r, d), -1)
    occ = diff.cumsum(axis=1)[:, :N_DAYS]

    rows = pd.DataFrame(occ, index=wards, columns=pd.date_range(START, periods=N_DAYS))
    return rows


def build_wards(adm: pd.DataFrame, occ: pd.DataFrame, rng) -> pd.DataFrame:
    steady = occ.iloc[:, ANALYSIS_FROM_DAY:]
    median_occ = steady.median(axis=1)
    util = rng.uniform(UTIL_LOW, UTIL_HIGH, size=len(median_occ))   # target typical utilisation
    beds = np.maximum(5, np.ceil(median_occ.values / util)).astype(int)

    w = (adm.groupby("ward_id")[["hospital_code", "ward_type"]].first().reset_index())
    w["total_beds"] = w["ward_id"].map(dict(zip(median_occ.index, beds)))
    if "hospital_type_code" in adm.columns:
        w["hospital_type_code"] = w["ward_id"].map(adm.groupby("ward_id")["hospital_type_code"].first())
    return w


def build_staff(wards: pd.DataFrame, rng) -> pd.DataFrame:
    # beds per doctor / per nurse, by shift (night is leaner)
    shifts = {"Morning": (10, 4), "Evening": (14, 5), "Night": (22, 7)}
    recs = []
    for _, w in wards.iterrows():
        for shift, (bpd, bpn) in shifts.items():
            noise = rng.uniform(0.8, 1.2)
            recs.append({
                "ward_id": w["ward_id"],
                "shift": shift,
                "doctors": max(1, int(round(w["total_beds"] / (bpd * noise)))),
                "nurses": max(1, int(round(w["total_beds"] / (bpn * noise)))),
            })
    return pd.DataFrame(recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to train.csv")
    ap.add_argument("--outdir", default="output")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    print("Cleaning...")
    df = clean(args.input)
    print(f"  {len(df):,} admissions, {df['ward_id'].nunique()} wards")

    print("Simulating timestamps...")
    adm = simulate_timestamps(df, rng)

    print("Sizing wards and staffing...")
    occ = build_daily_occupancy(adm)
    wards = build_wards(adm, occ, rng)
    staff = build_staff(wards, rng)

    beds = wards.set_index("ward_id")["total_beds"]
    daily = (occ.stack().rename("occupied_beds").reset_index()
             .rename(columns={"level_0": "ward_id", "level_1": "date"}))
    daily["total_beds"] = daily["ward_id"].map(beds)
    daily["occupancy_pct"] = (daily["occupied_beds"] / daily["total_beds"] * 100).round(1)
    daily = daily[daily["date"] >= START + pd.Timedelta(days=ANALYSIS_FROM_DAY)]

    tables = {"admissions": adm, "wards": wards, "staff_shifts": staff, "daily_occupancy": daily}

    db_path = out / "hospital_ops.db"
    if db_path.exists():
        db_path.unlink()
    with sqlite3.connect(db_path) as con:
        for name, t in tables.items():
            t.to_sql(name, con, index=False)
            t.to_csv(out / f"{name}.csv", index=False)
        con.execute("CREATE INDEX idx_adm_ward ON admissions(ward_id)")
        con.execute("CREATE INDEX idx_adm_ts ON admissions(admit_ts)")
        con.execute("CREATE INDEX idx_occ ON daily_occupancy(ward_id, date)")

    print(f"\nDone. Files in {out.resolve()}")
    for name, t in tables.items():
        print(f"  {name:16s} {len(t):>9,} rows")
    print("\nSanity check - share of ward-days above 90% occupancy:",
          f"{(daily['occupancy_pct'] > 90).mean():.1%}")


if __name__ == "__main__":
    main()
