"""
Non-GAN baselines for the 7-day temperature forecasting experiment.

Rebuilds the notebook's forecasting setup exactly (Section 7.2):
  - the same chronological split at 30-day-block boundaries (70% / 15% / 15%),
  - the same test windows: 30-day context, 7-day horizon, stride 7 (596 windows),
  - the same metric function (RMSE, MAE and pooled R^2 in degrees C).
Every baseline is fitted on the TRAINING period only; the validation period is used
only to choose hyper-parameters (ARIMA order, ridge alpha).

Baselines
  1. Persistence           tomorrow ... day 7 = the last observed day
  2. Climatology           smooth seasonal cycle (3 harmonics) fitted on training years
  3. Anomaly persistence   climatology + damped last-day anomaly (one factor per lead day)
  4. ARIMA                 ARIMA(p,0,q) on the seasonal anomaly, order chosen by validation RMSE
  5. Ridge regression      direct 7-output linear model on the full 30-day x 6-variable context

Also reports a skill score against climatology, 1 - (RMSE / RMSE_climatology)^2, because the seasonal
cycle alone already gives a high R^2 on this data.

Run from the repository root:
    python baselines/run_baselines.py
Writes outputs/baseline_forecast_metrics.csv, outputs/baseline_rmse_by_lead_day.csv,
outputs/forecast_metrics_all_models.csv and outputs/baselines_vs_gan.png.
"""
import itertools
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")  # statsmodels convergence chatter

FEATURES = ["temp", "surface_pressure", "total_precipitation_sum",
            "dewpoint_temperature_2m", "u_component_of_wind_10m", "v_component_of_wind_10m"]
T = 0
BLOCK_LEN, CONTEXT, HORIZON = 30, 30, 7
DATA_PATH = "data/era5_chennai_1950_2025.csv"
OUT_DIR = "outputs"

# ---------------------------------------------------------------- data and split (as in the notebook)
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)[["date"] + FEATURES]
n = len(df)
nb_total = n // BLOCK_LEN
n_tr, n_va = int(nb_total * 0.70), int(nb_total * 0.15)
tr_end, va_end = n_tr * BLOCK_LEN, (n_tr + n_va) * BLOCK_LEN

temp = df["temp"].to_numpy(float)
doy = df["date"].dt.dayofyear.to_numpy()


def window_starts(lo, hi, stride):
    """Start rows of windows that lie fully inside [lo, hi), as make_windows() in the notebook."""
    return np.arange(lo, hi - CONTEXT - HORIZON + 1, stride)


te_starts = window_starts(va_end, n, HORIZON)
va_starts = window_starts(tr_end, va_end, HORIZON)
tr_starts = window_starts(0, tr_end, 1)
assert len(te_starts) == 596, len(te_starts)          # same test windows as the GAN models


def targets(starts):
    return np.stack([temp[s + CONTEXT:s + CONTEXT + HORIZON] for s in starts])   # (N, 7) in deg C


Y_te, Y_va, Y_tr = targets(te_starts), targets(va_starts), targets(tr_starts)


def fc_metrics(p, y):
    """Identical to fc_metrics() in the notebook, on arrays already in deg C."""
    rmse, mae = np.sqrt(np.mean((p - y) ** 2)), np.mean(np.abs(p - y))
    r2 = 1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2)
    return {"RMSE": rmse, "MAE": mae, "R2": r2}, np.sqrt(np.mean((p - y) ** 2, 0))


# ---------------------------------------------------------------- seasonal cycle fitted on training years
def harmonics(d, k=3):
    w = 2 * np.pi * d / 365.25
    return np.column_stack([np.ones_like(w)] + [f(j * w) for j in range(1, k + 1) for f in (np.sin, np.cos)])


coef, *_ = np.linalg.lstsq(harmonics(doy[:tr_end]), temp[:tr_end], rcond=None)
clim = harmonics(doy) @ coef            # climatology for every day
anom = temp - clim                      # seasonal anomaly


def future_rows(starts):
    return np.stack([np.arange(s + CONTEXT, s + CONTEXT + HORIZON) for s in starts])


preds = {}

# 1. persistence
preds["Persistence"] = np.repeat(temp[te_starts + CONTEXT - 1][:, None], HORIZON, axis=1)

# 2. climatology
preds["Climatology"] = clim[future_rows(te_starts)]

# 3. damped anomaly persistence: anom(t+h) = phi_h * anom(t), phi_h by least squares on training windows
a_last_tr = anom[tr_starts + CONTEXT - 1]
a_fut_tr = anom[future_rows(tr_starts)]
phi = (a_last_tr[:, None] * a_fut_tr).sum(0) / (a_last_tr ** 2).sum()
preds["Anomaly persistence"] = clim[future_rows(te_starts)] + anom[te_starts + CONTEXT - 1][:, None] * phi

# 4. ARIMA on the anomaly: fit on training anomalies, choose (p, q) by validation RMSE,
#    then forecast each window from its own 30-day context with the fitted parameters.
def arima_forecast(res, starts):
    out = np.empty((len(starts), HORIZON))
    for i, s in enumerate(starts):
        r = res.apply(anom[s:s + CONTEXT], refit=False)
        out[i] = r.forecast(HORIZON)
    return out


best = None
for p, q in itertools.product(range(1, 4), range(0, 3)):
    res = ARIMA(anom[:tr_end], order=(p, 0, q), trend="n").fit()
    v = fc_metrics(clim[future_rows(va_starts)] + arima_forecast(res, va_starts), Y_va)[0]["RMSE"]
    if best is None or v < best[0]:
        best = (v, (p, q), res)
_, (p_best, q_best), arima_res = best
preds[f"ARIMA({p_best},0,{q_best}) on anomaly"] = clim[future_rows(te_starts)] + arima_forecast(arima_res, te_starts)

# 5. ridge regression on the full 30-day x 6-variable context (scaled on training rows only)
scaler = StandardScaler().fit(df.loc[:tr_end - 1, FEATURES])
sc = scaler.transform(df[FEATURES])


def ridge_X(starts):
    ctx = np.stack([sc[s:s + CONTEXT].ravel() for s in starts])
    first = doy[starts + CONTEXT]                     # calendar position of the first forecast day
    return np.column_stack([ctx, harmonics(first, 2)[:, 1:]])


X_tr, X_va, X_te = ridge_X(tr_starts), ridge_X(va_starts), ridge_X(te_starts)
alphas = [0.1, 1, 10, 100, 1000]
scores = {a: fc_metrics(Ridge(alpha=a).fit(X_tr, Y_tr).predict(X_va), Y_va)[0]["RMSE"] for a in alphas}
alpha = min(scores, key=scores.get)
preds["Ridge regression"] = Ridge(alpha=alpha).fit(X_tr, Y_tr).predict(X_te)

# ---------------------------------------------------------------- results
rows, lead = {}, {}
for name, p in preds.items():
    rows[name], lead[name] = fc_metrics(p, Y_te)

os.makedirs(OUT_DIR, exist_ok=True)
base = pd.DataFrame(rows).T
base.to_csv(f"{OUT_DIR}/baseline_forecast_metrics.csv")
pd.DataFrame(lead, index=[f"day_{d}" for d in range(1, HORIZON + 1)]).T.to_csv(f"{OUT_DIR}/baseline_rmse_by_lead_day.csv")

gan = pd.read_csv(f"{OUT_DIR}/forecast_comparison_metrics.csv", index_col=0)[["RMSE", "MAE", "R2"]]
allm = pd.concat([gan.assign(kind="GAN"), base.assign(kind="baseline")]).sort_values("RMSE")
# Skill score against climatology: the share of the seasonal-cycle forecast's squared error that a model removes.
# R^2 alone is flattering here because the seasonal cycle by itself already explains most of the variance.
allm.insert(3, "skill_vs_climatology", 1 - (allm["RMSE"] / rows["Climatology"]["RMSE"]) ** 2)
allm.to_csv(f"{OUT_DIR}/forecast_metrics_all_models.csv")

print(f"Test windows: {len(te_starts)} (30-day context, 7-day horizon)")
print(f"Chosen hyper-parameters on validation: ARIMA order=({p_best},0,{q_best}), ridge alpha={alpha}")
print(allm.round(4).to_string())
print("\nRMSE by lead day (baselines):")
for k, v in lead.items():
    print(f"  {k:28s} {np.round(v, 3)}")

# figure: RMSE and R^2 for every model, GANs and baselines together
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
colors = ["#0d9488" if k == "GAN" else "#94a3b8" for k in allm["kind"]]
ax[0].barh(allm.index, allm["RMSE"], color=colors); ax[0].invert_yaxis()
ax[0].set_xlabel("RMSE (deg C, lower is better)"); ax[0].set_title("7-day forecast error, 596 test windows")
ax[1].barh(allm.index, allm["R2"], color=colors); ax[1].invert_yaxis(); ax[1].set_yticklabels([])
ax[1].set_xlabel("R$^2$ (higher is better)"); ax[1].set_title("Explained variance")
for a, col in zip(ax, ["RMSE", "R2"]):
    a.set_xlim(0, allm[col].max() * 1.15)
    for i, v in enumerate(allm[col]):
        a.text(v, i, f" {v:.3f}", va="center", fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/baselines_vs_gan.png", dpi=140)
print(f"\nSaved {OUT_DIR}/baselines_vs_gan.png")
