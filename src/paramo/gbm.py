"""Predictors and training of the gradient-boosting structures (GBM and GR4H_GBM).

Used by stage 10 (training) and stage 11 (SHAP contributions of the error model).
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.signal import lfilter

from .core import in_period, path, read_simulation, simulation_path
from .metrics import variance_adjustment


def predictors(cfg, catchment: str, scenario: str, hybrid: bool) -> pd.DataFrame:
    gc = cfg["gbm"]
    forcing = pd.read_parquet(path(cfg, "processed") / f"forcing_{catchment}.parquet")
    gr = read_simulation(simulation_path(cfg, "GR4H", scenario, catchment))
    forcing = forcing.reindex(gr.index)
    p = forcing[f"P__{scenario}"].clip(lower=0).fillna(0)
    x = pd.DataFrame(index=gr.index)
    for w in gc["precipitation_windows_h"]:
        x[f"P_{w}h"] = p.rolling(w, min_periods=1).sum()
    for k in gc["api_k"]:
        x[f"API_{k}"] = lfilter([1.0], [1.0, -k], p.to_numpy())
    t = forcing["T_catchment"].interpolate(limit_direction="both")
    x["T"], x["T_24h"] = t, t.rolling(24, min_periods=1).mean()
    pet = forcing["PET"].fillna(0)
    x["PET_24h"] = pet.rolling(24, min_periods=1).sum()
    x["PET_720h"] = pet.rolling(720, min_periods=1).sum()
    h, d = x.index.hour.to_numpy(), x.index.dayofyear.to_numpy()
    x["hour_sin"], x["hour_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    x["doy_sin"], x["doy_cos"] = np.sin(2 * np.pi * d / 365.25), np.cos(2 * np.pi * d / 365.25)
    if hybrid:
        q = np.sqrt(gr["Qsim"].clip(lower=0))
        x["sqrtQ_gr4h"] = q
        for lag in (1, 6, 24):
            x[f"dQ_gr4h_{lag}h"] = q - q.shift(lag)
        x["Prod"], x["Rout"] = gr["Prod"], gr["Rout"]
    x["Qobs"], x["Q_gr4h"] = gr["Qobs"], gr["Qsim"]
    x["excluded"] = gr["excluded"] | gr["Qobs"].isna()
    return x


def fit_predict(xc, xt, cols, hybrid, cfg, seed, return_raw: bool = False):
    """Calibration and transfer series with the variance adjustment of paramo.metrics.

    With return_raw, the unadjusted series are also returned (same volume rule for the hybrid).
    """
    gc, per = cfg["gbm"], cfg["periods"]
    valid = ~xc["excluded"].to_numpy()
    calibration = in_period(xc.index, per["calibration"])
    stop = in_period(xc.index, cfg["lstm"]["stop_period"])
    train, early = calibration & ~stop & valid, stop & valid
    target = np.sqrt(xc["Qobs"].clip(lower=0))
    if hybrid:
        target = target - xc["sqrtQ_gr4h"]
    model = lgb.LGBMRegressor(random_state=seed, **gc["lgbm"])
    callbacks = [lgb.early_stopping(gc["early_stopping_rounds"], verbose=False)]
    try:                                                   # LightGBM >= 4.6
        model.fit(xc.loc[train, cols], target[train], eval_X=(xc.loc[early, cols],),
                  eval_y=(target[early],), callbacks=callbacks)
    except TypeError:
        model.fit(xc.loc[train, cols], target[train],
                  eval_set=[(xc.loc[early, cols], target[early])], callbacks=callbacks)

    def to_discharge(x):
        y = model.predict(x[cols])
        base = x["sqrtQ_gr4h"].to_numpy() if hybrid else 0.0
        return np.clip(base + y, 0, None) ** 2

    rc, rt = to_discharge(xc), to_discharge(xt)
    adjust, c = variance_adjustment(rc, xc["Qobs"].to_numpy(), train)
    qc, qt = adjust(rc), adjust(rt)
    if hybrid:                      # keep the GR4H volume over the calibration period
        m = calibration & valid
        volume = xc["Q_gr4h"].to_numpy()[m].mean()
        k = volume / max(qc[m].mean(), 1e-12)
        qc, qt = qc * k, qt * k
        k = volume / max(rc[m].mean(), 1e-12)
        rc, rt = rc * k, rt * k
    if return_raw:
        return qc, qt, model, c, (rc, rt)
    return qc, qt, model, c
