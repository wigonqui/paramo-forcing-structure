"""Common-sample construction shared by the factorial analysis (11) and sensitivity (12)."""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from .core import in_period, read_simulation, simulation_path


def load_simulations(cfg, structures, scenarios, catchment) -> dict:
    sims = {}
    for s, sc in product(structures, scenarios):
        f = simulation_path(cfg, s, sc, catchment)
        if f.exists():
            sims[(s, sc)] = read_simulation(f)
    return sims


def common_sample(sims: dict, period) -> tuple[np.ndarray, pd.DataFrame]:
    """Hours shared by all simulations within the period, valid in every one of them."""
    idx = None
    for s in sims.values():
        idx = s.index if idx is None else idx.intersection(s.index)
    idx = idx[in_period(idx, period)]
    q_obs = next(iter(sims.values())).loc[idx, "Qobs"].to_numpy()
    valid = np.isfinite(q_obs)
    q = {}
    for k, s in sims.items():
        s = s.loc[idx]
        valid &= ~s["excluded"].to_numpy() & np.isfinite(s["Qsim"].to_numpy())
        q[k] = s["Qsim"].to_numpy()
    return valid, pd.DataFrame(q, index=idx).assign(Qobs=q_obs)


def daily_table(valid: np.ndarray, q: pd.DataFrame, min_hours: int = 20):
    z = q[valid]
    n = z.groupby(z.index.normalize()).size()
    d = z.groupby(z.index.normalize()).mean()[n >= min_hours]
    d = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="D"))
    return d["Qobs"].notna().to_numpy(), d
