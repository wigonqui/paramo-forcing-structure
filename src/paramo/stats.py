"""Statistical tools: homogeneity tests, moving-block bootstrap, Diebold–Mariano test,
Holm correction and two-way ANOVA variance fractions."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps


# --------------------------------------------------------------------------- homogeneity
def pettitt(x) -> tuple[int, float, float]:
    """Pettitt change-point test: (index of change, K statistic, approximate p-value)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return -1, np.nan, np.nan
    ranks = pd.Series(x).rank().to_numpy()
    u = 2 * np.cumsum(ranks) - np.arange(1, n + 1) * (n + 1)
    k = int(np.argmax(np.abs(u)))
    big_k = float(np.abs(u[k]))
    return k, big_k, min(2 * np.exp(-6 * big_k ** 2 / (n ** 3 + n ** 2)), 1.0)


def snht(x) -> tuple[int, float]:
    """Standard normal homogeneity test (Alexandersson): (index of change, T0)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return -1, np.nan
    z = (x - x.mean()) / x.std(ddof=1)
    t = np.array([k * z[:k].mean() ** 2 + (n - k) * z[k:].mean() ** 2 for k in range(1, n)])
    k = int(np.argmax(t))
    return k + 1, float(t[k])


def snht_critical(n: int, alpha: float = 0.05, n_sim: int = 2000, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    return float(np.quantile([snht(rng.standard_normal(n))[1] for _ in range(n_sim)],
                             1 - alpha))


# --------------------------------------------------------------------------- bootstrap
class BlockBootstrap:
    """Moving-block bootstrap of KGE' or NSE using block sums of (n, Σo, Σs, Σo², Σs², Σos).

    Every simulation is resampled with the same blocks, so paired differences between models
    keep their dependence. Each replicate costs O(number of blocks).
    """

    def __init__(self, length: int, block: int, replicates: int, seed: int):
        self.block = min(block, length)
        n_blocks = int(np.ceil(length / self.block))
        rng = np.random.default_rng(seed)
        self.starts = rng.integers(0, length - self.block + 1, size=(replicates, n_blocks))

    def _sum(self, x: np.ndarray) -> np.ndarray:
        c = np.concatenate([[0.0], np.cumsum(x)])
        return (c[self.starts + self.block] - c[self.starts]).sum(axis=1)

    def score(self, obs: np.ndarray, sim: np.ndarray, valid: np.ndarray,
              metric: str = "kge") -> np.ndarray:
        w = valid.astype(float)
        o, s = np.where(valid, obs, 0.0), np.where(valid, sim, 0.0)
        n, so, ss = self._sum(w), self._sum(o), self._sum(s)
        soo, sss, sos = self._sum(o * o), self._sum(s * s), self._sum(o * s)
        mo, ms = so / n, ss / n
        if metric == "nse":
            return 1 - (sss - 2 * sos + soo) / (soo - n * mo ** 2)
        vo, vs = (soo - n * mo ** 2) / (n - 1), (sss - n * ms ** 2) / (n - 1)
        r = ((sos - n * mo * ms) / (n - 1)) / np.sqrt(vo * vs)
        beta, gamma = ms / mo, (np.sqrt(vs) / ms) / (np.sqrt(vo) / mo)
        return 1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)


def interval(x, alpha: float = 0.05) -> tuple[float, float]:
    return float(np.nanquantile(x, alpha / 2)), float(np.nanquantile(x, 1 - alpha / 2))


# --------------------------------------------------------------------------- tests
def diebold_mariano(err_a: np.ndarray, err_b: np.ndarray, lag: int) -> tuple[float, float]:
    """H0: equal mean squared error. Negative statistic means model A is more accurate.
    Long-run variance with Newey–West weights."""
    d = err_a ** 2 - err_b ** 2
    d = d[np.isfinite(d)]
    t = len(d)
    if t < 30:
        return np.nan, np.nan
    dc = d - d.mean()
    lrv = dc @ dc / t
    for k in range(1, lag + 1):
        lrv += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / t
    stat = d.mean() / np.sqrt(lrv / t)
    return float(stat), float(2 * sps.t.sf(abs(stat), t - 1))


def holm(p: pd.Series) -> pd.Series:
    order = p.sort_values()
    m = len(order)
    adj = np.maximum.accumulate([(m - i) * v for i, v in enumerate(order.to_numpy())])
    return pd.Series(np.minimum(adj, 1.0), index=order.index).reindex(p.index)


def anova_fractions(m: np.ndarray) -> dict:
    """Two-way ANOVA without replication on a (forcing × structure) skill matrix."""
    g = m.mean()
    ss_f = m.shape[1] * ((m.mean(1) - g) ** 2).sum()
    ss_s = m.shape[0] * ((m.mean(0) - g) ** 2).sum()
    ss_t = ((m - g) ** 2).sum()
    return {"forcing": ss_f / ss_t, "structure": ss_s / ss_t,
            "interaction": (ss_t - ss_f - ss_s) / ss_t}


def anova_replicates(stack: np.ndarray) -> pd.DataFrame:
    """ANOVA fractions for each bootstrap replicate of a (B, forcing, structure) array."""
    rows = [anova_fractions(stack[b]) for b in range(stack.shape[0])
            if np.isfinite(stack[b]).all()]
    return pd.DataFrame(rows)
