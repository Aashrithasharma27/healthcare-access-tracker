# Step 4: 24-hour bed capacity forecast + ward alerts
# Put this file in your healthcare-access-tracker folder (next to los_model.py) and run:  python forecast_capacity.py
# Needs the los_predictions table created by los_model.py.
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

SIGMA = 15.5          # typical LOS prediction error in days (model RMSE)
RED, YELLOW = 90, 75  # alert thresholds (% occupancy)
START = pd.Timestamp("2023-01-01")
SNAPSHOT_HOUR = 8     # forecasts are made at 08:00

root = Path(__file__).resolve().parent
db_path = next(p for p in root.rglob("hospital_ops.db") if p.is_file())
out_dir = root / "dashboard"
out_dir.mkdir(exist_ok=True)
con = sqlite3.connect(db_path)
print("Using database:", db_path)

adm = pd.read_sql(
    """SELECT a.case_id, a.ward_id, a.admit_ts, a.discharge_ts, a.admit_hour, p.predicted_los_days
       FROM admissions a JOIN los_predictions p USING (case_id)""",
    con, parse_dates=["admit_ts", "discharge_ts"])
beds = pd.read_sql("SELECT ward_id, total_beds FROM wards", con).set_index("ward_id")["total_beds"]
ward_ids = list(beds.index)
code = {w: i for i, w in enumerate(ward_ids)}
nW = len(ward_ids)
total_beds = beds.values.astype(float)

day = pd.Timedelta(days=1)
admit_d = ((adm["admit_ts"] - START) / day).values
disc_d = ((adm["discharge_ts"] - START) / day).values
pred = adm["predicted_los_days"].values
wcode = adm["ward_id"].map(code).values
hour_w = adm["admit_hour"].value_counts(normalize=True).sort_index().reindex(range(24), fill_value=0).values


def band(pct):
    return np.where(pct > RED, "RED", np.where(pct >= YELLOW, "YELLOW", "GREEN"))


def forecast(T):
    """Hourly expected occupied beds per ward for the 24 hours after time T."""
    t = (T - START) / day
    inh = (admit_d <= t) & (disc_d > t)                  # patients in the hospital at T
    e, p, w = t - admit_d[inh], pred[inh], wcode[inh]    # days stayed so far, predicted LOS, ward
    base = norm.cdf((e - p) / SIGMA)
    surv = np.clip(1 - base, 1e-6, None)
    occ0 = np.bincount(w, minlength=nW).astype(float)
    prior = (admit_d >= t - 28) & (admit_d < t)          # recent admission rate per ward
    rate = np.bincount(wcode[prior], minlength=nW) / 28.0
    out, cum = [], 0.0
    for h in range(1, 25):
        pdis = np.clip((norm.cdf((e + h / 24 - p) / SIGMA) - base) / surv, 0, 1)
        dis = np.bincount(w, weights=pdis, minlength=nW)
        cum += hour_w[(T.hour + h - 1) % 24]
        out.append(occ0 - dis + rate * cum)
    return occ0, np.array(out)                           # shape (24, nW)


def actual_occ(T):
    t = (T - START) / day
    return np.bincount(wcode[(admit_d <= t) & (disc_d > t)], minlength=nW).astype(float)


# ---------- Backtest: how good is the forecast 24h ahead? ----------
rows = []
for d in range(300, 728):
    T = START + pd.Timedelta(days=d, hours=SNAPSHOT_HOUR)
    occ0, fc = forecast(T)
    act = actual_occ(T + day)
    rows.append(pd.DataFrame({
        "ward_id": ward_ids, "total_beds": total_beds,
        "persist_pct": occ0 / total_beds * 100,
        "forecast_pct": fc[-1] / total_beds * 100,
        "actual_pct": act / total_beds * 100}))
bt = pd.concat(rows, ignore_index=True)
big = bt["total_beds"] >= 30
summary = []
for label, m in [("All wards", np.ones(len(bt), bool)), ("Wards with 30+ beds", big)]:
    b = bt[m]
    for name, col in [("Persistence (assume no change)", "persist_pct"), ("Model forecast", "forecast_pct")]:
        a_band, f_band = band(b["actual_pct"].values), band(b[col].values)
        red_a, red_f = a_band == "RED", f_band == "RED"
        summary.append({
            "wards": label, "method": name,
            "MAE (% points)": round((b[col] - b["actual_pct"]).abs().mean(), 2),
            "alert band correct": f"{(a_band == f_band).mean():.1%}",
            "red recall": f"{(red_a & red_f).sum() / max(red_a.sum(), 1):.1%}",
            "red precision": f"{(red_a & red_f).sum() / max(red_f.sum(), 1):.1%}"})
summary = pd.DataFrame(summary)
print("\nBacktest: forecast 24 hours ahead,", len(bt) // nW, "daily snapshots\n")
print(summary.to_string(index=False))

# ---------- Latest forecast for the dashboard ----------
T_now = START + pd.Timedelta(days=729, hours=SNAPSHOT_HOUR) - day   # last day with a full 24h of data
occ0, fc = forecast(T_now)
hourly = []
for h in range(24):
    hourly.append(pd.DataFrame({
        "ward_id": ward_ids, "snapshot_ts": T_now, "hours_ahead": h + 1,
        "forecast_ts": T_now + pd.Timedelta(hours=h + 1),
        "total_beds": total_beds, "forecast_occupied": fc[h].round(1),
        "forecast_pct": (fc[h] / total_beds * 100).round(1)}))
hourly = pd.concat(hourly, ignore_index=True)
hourly["alert"] = band(hourly["forecast_pct"].values)

peak = hourly.groupby("ward_id")["forecast_pct"].max()
alerts = pd.DataFrame({
    "ward_id": ward_ids, "snapshot_ts": T_now, "total_beds": total_beds,
    "occupied_now": occ0, "occupancy_now_pct": (occ0 / total_beds * 100).round(1),
    "forecast_24h_pct": hourly[hourly.hours_ahead == 24].set_index("ward_id").loc[ward_ids, "forecast_pct"].values,
    "peak_forecast_pct": peak.loc[ward_ids].values})
alerts["alert_now"] = band(alerts["occupancy_now_pct"].values)
alerts["alert_24h"] = band(alerts["forecast_24h_pct"].values)
alerts["beds_free_24h"] = (alerts["total_beds"] - hourly[hourly.hours_ahead == 24].set_index("ward_id").loc[ward_ids, "forecast_occupied"].values).round(0)
alerts = alerts.sort_values("forecast_24h_pct", ascending=False)
print("\nForecast made at", T_now, "- wards by alert in 24h:")
print(alerts["alert_24h"].value_counts().to_string())
print("\nTop 8 at-risk wards:")
print(alerts.head(8)[["ward_id", "total_beds", "occupancy_now_pct", "forecast_24h_pct", "alert_now", "alert_24h"]].to_string(index=False))

for name, t in [("forecast_24h", hourly), ("ward_alerts", alerts), ("forecast_backtest", summary)]:
    t.to_csv(out_dir / f"{name}.csv", index=False)
    con.execute(f"DROP TABLE IF EXISTS {name}")
    t.to_sql(name, con, index=False)
con.commit()
con.close()
print("\nSaved forecast_24h, ward_alerts, forecast_backtest to", out_dir, "and the database")