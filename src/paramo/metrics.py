"""Performance metrics shared by all stages (identical definitions in R stage 08).

KGE' follows Kling et al. (2012): gamma is the ratio of coefficients of variation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .core import in_period

METRIC_KEYS = ("kge", "r", "beta", "gamma", "nse", "nse_log", "kge_sqrt")


def _pairs(obs, sim):
    o, s = np.asarray(obs, float), np.asarray(sim, float)
    ok = np.isfinite(o) & np.isfinite(s)
    return o[ok], s[ok]


def kge_components(o: np.ndarray, s: np.ndarray) -> tuple[float, float, float, float]:
    r = np.corrcoef(o, s)[0, 1]
    beta = s.mean() / o.mean()
    gamma = (s.std(ddof=1) / s.mean()) / (o.std(ddof=1) / o.mean())
    return 1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2), r, beta, gamma


def kge(obs, sim) -> dict:
    o, s = _pairs(obs, sim)
    if len(o) < 10 or o.std() == 0 or s.std() == 0:
        return {"kge": np.nan, "r": np.nan, "beta": np.nan, "gamma": np.nan, "n": len(o)}
    k, r, b, g = kge_components(o, s)
    return {"kge": k, "r": r, "beta": b, "gamma": g, "n": len(o)}


def streamflow_metrics(obs, sim) -> dict:
    """KGE' and components, NSE, NSE of log flows and KGE' of square-root flows."""
    o, s = _pairs(obs, sim)
    if len(o) < 100:
        return {k: np.nan for k in METRIC_KEYS} | {"n": len(o)}
    k, r, b, g = kge_components(o, s)
    eps = o.mean() / 100
    lo, ls = np.log(o + eps), np.log(np.clip(s, 0, None) + eps)
    return {"kge": k, "r": r, "beta": b, "gamma": g,
            "nse": 1 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2),
            "nse_log": 1 - np.sum((ls - lo) ** 2) / np.sum((lo - lo.mean()) ** 2),
            "kge_sqrt": kge_components(np.sqrt(o), np.sqrt(np.clip(s, 0, None)))[0],
            "n": len(o)}


def evaluate(df: pd.DataFrame, period, min_hours_day: int = 20) -> list[dict]:
    """Hourly and daily metrics for a table with Qobs, Qsim and excluded."""
    sel = in_period(df.index, period) & ~df["excluded"].to_numpy(bool)
    z = df.loc[sel, ["Qobs", "Qsim"]]
    rows = [{"scale": "hourly", **streamflow_metrics(z["Qobs"], z["Qsim"])}]
    n = z["Qobs"].notna().groupby(z.index.normalize()).sum()
    d = z.groupby(z.index.normalize()).mean()[n >= min_hours_day]
    rows.append({"scale": "daily", **streamflow_metrics(d["Qobs"], d["Qsim"])})
    return rows


def contingency(obs, sim, threshold: float) -> dict:
    """POD, FAR, CSI and frequency ratio for events >= threshold."""
    o, s = _pairs(obs, sim)
    o, s = o >= threshold, s >= threshold
    hit, false, miss = np.sum(o & s), np.sum(~o & s), np.sum(o & ~s)
    return {"pod": hit / (hit + miss) if hit + miss else np.nan,
            "far": false / (hit + false) if hit + false else np.nan,
            "csi": hit / (hit + false + miss) if hit + false + miss else np.nan,
            "frequency_ratio": s.sum() / o.sum() if o.sum() else np.nan}


def variance_adjustment(q_sim: np.ndarray, q_obs: np.ndarray, train: np.ndarray):
    """Affine adjustment that restores the observed standard deviation of Q (c >= 1).

    Least-squares learners predict the conditional mean and shrink hydrograph variability,
    which KGE' penalises through gamma. The factor is estimated on training data only and
    leaves the correlation unchanged.
    """
    s, o = np.clip(q_sim, 0, None), np.clip(q_obs, 0, None)
    ok = train & np.isfinite(o) & np.isfinite(s)
    centre = s[ok].mean()
    c = float(np.clip(o[ok].std() / max(s[ok].std(), 1e-12), 1.0, 3.0))

    def apply(q):
        return np.clip(centre + (np.clip(q, 0, None) - centre) * c, 0, None)

    return apply, c
