# Step 3: Predict patient length of stay (LOS)
# Put this file in your healthcare-access-tracker folder (next to prepare_data.py) and run:  python los_model.py
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

root = Path(__file__).resolve().parent
db_path = next(p for p in root.rglob("hospital_ops.db") if p.is_file())
out_dir = root / "dashboard"
out_dir.mkdir(exist_ok=True)
print("Using database:", db_path)

# 1. Load
con = sqlite3.connect(db_path)
df = pd.read_sql("SELECT * FROM admissions", con)
print("Rows:", len(df))

# 2. Features (only fields known at admission; no timestamps, no stay/los columns, no IDs)
df["age_num"] = df["age"].str.split("-").str[0].astype(int) + 5
cat_cols = ["hospital_code", "hospital_type_code", "hospital_region_code", "department", "ward_type",
            "ward_facility_code", "type_of_admission", "severity_of_illness", "city_code_hospital", "city_code_patient"]
num_cols = ["age_num", "bed_grade", "visitors_with_patient", "admission_deposit", "available_extra_rooms_in_hospital"]
X = df[cat_cols + num_cols].copy()
for c in cat_cols:
    X[c] = X[c].astype(str)
y = df["los_days_mid"]

# 3. Split by patient so nobody appears in both sets
train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(X, y, df["patientid"]))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")

# 4. Models
def evaluate(name, pred):
    return {"model": name,
            "MAE (days)": round(mean_absolute_error(y_test, pred), 2),
            "RMSE (days)": round(np.sqrt(mean_squared_error(y_test, pred)), 2),
            "R2": round(r2_score(y_test, pred), 3)}

results = [evaluate("Baseline (mean)", np.full(len(y_test), y_train.mean()))]

pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)], remainder="passthrough")
lin = make_pipeline(pre, LinearRegression()).fit(X_train, y_train)
results.append(evaluate("Linear Regression", lin.predict(X_test)))

Xg_train, Xg_test = X_train.copy(), X_test.copy()
for c in cat_cols:
    Xg_train[c] = Xg_train[c].astype("category")
    Xg_test[c] = pd.Categorical(Xg_test[c], categories=Xg_train[c].cat.categories)
gbr = HistGradientBoostingRegressor(categorical_features="from_dtype", max_iter=300,
                                    learning_rate=0.08, random_state=42).fit(Xg_train, y_train)
gbr_pred = gbr.predict(Xg_test)
results.append(evaluate("Gradient Boosting", gbr_pred))
print("\n", pd.DataFrame(results).to_string(index=False))

# 5. Accuracy in 10-day bins
def to_bin(days):
    return np.clip(np.floor((np.asarray(days) - 1) / 10), 0, 10).astype(int)
tb, pb = to_bin(y_test.values), to_bin(gbr_pred)
bb = to_bin(np.full(len(y_test), y_train.mean()))
print(f"\nGradient Boosting - exact bin: {(tb == pb).mean():.1%}  within one bin: {(abs(tb - pb) <= 1).mean():.1%}")
print(f"Baseline          - exact bin: {(tb == bb).mean():.1%}  within one bin: {(abs(tb - bb) <= 1).mean():.1%}")

# 6. Feature importance
sample = Xg_test.sample(20000, random_state=42)
imp = permutation_importance(gbr, sample, y_test.loc[sample.index], n_repeats=3,
                             random_state=42, scoring="neg_mean_absolute_error")
imp_df = pd.DataFrame({"feature": sample.columns, "importance": imp.importances_mean}).sort_values("importance")
plt.figure(figsize=(7, 5))
plt.barh(imp_df["feature"], imp_df["importance"], color="#2a7ab0")
plt.xlabel("Increase in error (days) when feature is shuffled")
plt.title("Feature importance for LOS prediction")
plt.tight_layout()
plt.savefig(out_dir / "feature_importance.png", dpi=150)
plt.close()
print("\nTop drivers:\n", imp_df.sort_values("importance", ascending=False).head(5).to_string(index=False))

# 7. Predicted vs actual
cmp = pd.DataFrame({"actual": y_test.values, "predicted": gbr_pred})
g = cmp.groupby(to_bin(cmp["actual"]))["predicted"].mean()
labels = ["0-10", "11-20", "21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "81-90", "91-100", ">100"]
plt.figure(figsize=(8, 4.5))
plt.plot(range(11), [5, 15.5, 25.5, 35.5, 45.5, 55.5, 65.5, 75.5, 85.5, 95.5, 110], "k--", label="Perfect prediction")
plt.plot(g.index, g.values, "o-", color="#2a7ab0", label="Model average prediction")
plt.xticks(range(11), labels, rotation=45)
plt.xlabel("Actual LOS bin (days)")
plt.ylabel("Average predicted LOS (days)")
plt.title("Predicted vs actual length of stay")
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "predicted_vs_actual.png", dpi=150)
plt.close()

# 8. Save predictions for Power BI and the database
Xg_all = X.copy()
for c in cat_cols:
    Xg_all[c] = pd.Categorical(Xg_all[c], categories=Xg_train[c].cat.categories)
df["predicted_los_days"] = gbr.predict(Xg_all).round(1)
df["split"] = "train"
df.loc[df.index[test_idx], "split"] = "test"
pred = df[["case_id", "ward_id", "admit_ts", "los_days_mid", "predicted_los_days", "split"]]
pred.to_csv(out_dir / "los_predictions.csv", index=False)
con.execute("DROP TABLE IF EXISTS los_predictions")
pred.to_sql("los_predictions", con, index=False)
con.commit()
con.close()
print(f"\nSaved {len(pred):,} predictions and 2 charts to {out_dir}")