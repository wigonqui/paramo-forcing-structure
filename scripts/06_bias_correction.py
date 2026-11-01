"""
Stage 06 - Bias correction, fusion and QM reference

Purpose
    Correct the hourly precipitation of IMERG, GSMaP and ERA5-Land against the rain gauges of
    the site, build a fused product from the three, and produce a quantile-mapping (QM)
    reference that isolates what machine learning adds to the correction.

Methods
    ML correction in two stages with LightGBM: a classifier for P >= wet threshold and a
    Tweedie regressor for the conditional depth; the expectation p·E[P|wet] is then mapped to
    the observed distribution at two scales (monthly empirical quantile mapping of hourly
    values for intensity, and of daily totals for volume, to which each day is rescaled).
    Without the daily scale the expectation, which is smooth in time, inflates daily
    variability. Above the last training quantile the mapping uses a bounded tail ratio.
    Predictors come only from the product itself (lags -3..+3 h, moving sums, hour and
    day-of-year harmonics), so every corrected scenario remains a satellite-only forcing; the
    fusion uses the predictors of the three products. The QM reference applies the same
    two-scale mapping directly to the raw product (fusion: mean of the three products).
    Validation: leave-one-gauge-out (spatial) and training restricted to
    bias_correction.training_periods, which never overlap the hydrological validation period
    (temporal). Gauges with too few hours in the validation period contribute to training but
    are not scored. The final models are trained with all gauges and applied to the
    catchment-mean series. Features are computed once per product and location.

Tools
    LightGBM (deterministic, multi-threaded), pandas, numpy, matplotlib; paramo.metrics.

Inputs
    data/processed/<site>/hourly_panel.parquet (stage 05)

Outputs
    data/processed/<site>/corrected_precipitation.parquet  <product>__<raw|corr|qm>__catchment-<id>
    data/processed/<site>/bias_correction_validation.parquet  leave-one-out validation series
    outputs/<site>/tables/06_bias_correction_evaluation.csv, 06_feature_importance.csv,
                          06_annual_bias.csv
    outputs/<site>/figures/06_exceedance.(png|pdf)

Usage
    python scripts/06_bias_correction.py
    python scripts/06_bias_correction.py --figures-only   redraw the figure only

Next stage
    07 - Catchment forcing scenarios.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import lightgbm as lgb
import numpy as np
import pandas as pd

from paramo import plotting as pl
from paramo.core import get_logger, in_period, in_periods, load_config, path
from paramo.metrics import contingency, kge

log = get_logger("06_bias_correction")


# --------------------------------------------------------------------------- features
def features(p: pd.Series, bc: dict, calendar: bool = True) -> pd.DataFrame:
    x = {f"p_t{lag:+d}": p.shift(-lag) for lag in bc["lags_h"]}
    for w in bc["windows_h"]:
        x[f"sum_{w}h"] = p.rolling(w, center=True, min_periods=1).sum()
    if calendar:
        h, d = p.index.hour.to_numpy(), p.index.dayofyear.to_numpy()
        x["hour_sin"], x["hour_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
        x["doy_sin"], x["doy_cos"] = (np.sin(2 * np.pi * d / 365.25),
                                      np.cos(2 * np.pi * d / 365.25))
    return pd.DataFrame(x, index=p.index)


class FeatureCache:
    """Raw series and predictors per (product, location); 'fusion' combines the products."""

    def __init__(self, panel, bc, base):
        self.panel, self.bc, self.base, self.cache = panel, bc, base, {}

    def get(self, product: str, ident: str):
        key = (product, ident)
        if key not in self.cache:
            if product != "fusion":
                p = self.panel[f"{product}__precipitation__{ident}"]
                self.cache[key] = (p, features(p, self.bc))
            else:
                series = {b: self.panel[f"{b}__precipitation__{ident}"] for b in self.base}
                p = pd.concat(series, axis=1).mean(axis=1, skipna=False)
                x = pd.concat([features(s, self.bc, calendar=False).add_prefix(f"{b}_")
                               for b, s in series.items()]
                              + [features(p, self.bc).filter(regex="^(hour|doy)_")], axis=1)
                self.cache[key] = (p, x)
        return self.cache[key]


# --------------------------------------------------------------------------- correction models
def _map(v, qp, qo, tail_max):
    out = np.interp(v, qp, qo)
    above = v > qp[-1]
    if above.any():
        ratio = np.clip(qo[-1] / qp[-1], 1 / tail_max, tail_max) if qp[-1] > 0 else 1.0
        out[above] = qo[-1] + (v[above] - qp[-1]) * ratio
    return out


class TwoScaleMapping:
    """Monthly empirical quantile mapping at hourly and daily scale; days rescaled to the
    daily mapping so that both intensity and volume follow the observations."""

    def __init__(self, bc):
        self.bc = bc

    @staticmethod
    def _days(idx, group):
        g = np.zeros(len(idx), int) if group is None else np.asarray(group)
        return pd.MultiIndex.from_arrays([g, idx.normalize()])

    def fit(self, e: np.ndarray, y: np.ndarray, idx: pd.DatetimeIndex, group=None):
        q = np.linspace(0, 0.999, self.bc["quantiles"])
        months = idx.month
        self.hourly = {m: (np.quantile(e[months == m], q), np.quantile(y[months == m], q))
                       for m in range(1, 13) if (months == m).sum() >= 500}
        d = pd.DataFrame({"e": e, "y": y, "n": 1}, index=self._days(idx, group))
        d = d.groupby(level=[0, 1]).sum()
        d = d[d.n >= 20]
        dm = d.index.get_level_values(1).month
        self.daily = {m: (np.quantile(d.e.to_numpy()[dm == m], q),
                          np.quantile(d.y.to_numpy()[dm == m], q))
                      for m in range(1, 13) if (dm == m).sum() >= 30}
        return self

    def apply(self, e: np.ndarray, idx: pd.DatetimeIndex, group=None) -> np.ndarray:
        tail = self.bc["tail_ratio_max"]
        months = idx.month
        h = np.full(len(e), np.nan)
        for m, (qp, qo) in self.hourly.items():
            sel = months == m
            h[sel] = _map(e[sel], qp, qo, tail)
        h = np.clip(h, 0, None)
        totals = pd.DataFrame({"e": e, "h": h}, index=self._days(idx, group)) \
            .groupby(level=[0, 1]).transform("sum")
        target = np.full(len(e), np.nan)
        for m, (qp, qo) in self.daily.items():
            sel = months == m
            target[sel] = np.clip(_map(totals.e.to_numpy()[sel], qp, qo, tail), 0, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            k_h = np.where(totals.h.to_numpy() > 0, target / totals.h.to_numpy(), np.nan)
            k_e = np.where(totals.e.to_numpy() > 0, target / totals.e.to_numpy(), 0.0)
        return np.where(np.isfinite(k_h), h * k_h, e * k_e)


class MLCorrector:
    def __init__(self, bc):
        self.bc = bc

    def _expectation(self, x):
        return self.clf.predict_proba(x)[:, 1] * np.clip(self.reg.predict(x), 0, None)

    def fit(self, x, y, group):
        bc = self.bc
        yv = y.to_numpy()
        wet = yv >= bc["wet_threshold_mm"]          # positional: the stacked index repeats
        self.clf = lgb.LGBMClassifier(**bc["lgbm"]).fit(x, wet.astype(int))
        self.reg = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=bc["tweedie_power"],
                                     **bc["lgbm"]).fit(x[wet], yv[wet])
        self.mapping = TwoScaleMapping(bc).fit(self._expectation(x), yv, x.index, group)
        return self

    def predict(self, x, group=None):
        return self.mapping.apply(self._expectation(x), x.index, group)


class QMReference:
    def __init__(self, bc):
        self.bc = bc

    def fit(self, p_raw, y, group):
        self.mapping = TwoScaleMapping(self.bc).fit(np.clip(p_raw.to_numpy(), 0, None),
                                                    y.to_numpy(), p_raw.index, group)
        return self

    def predict(self, p_raw, group=None):
        return self.mapping.apply(np.clip(p_raw.to_numpy(), 0, None), p_raw.index, group)


# --------------------------------------------------------------------------- helpers
def stack(cache, panel, product, gauges, periods):
    xs, ps, ys, gs = [], [], [], []
    for k, code in enumerate(gauges):
        p, x = cache.get(product, code)
        y = panel[f"obs__precipitation__{code}"]
        m = in_periods(x.index, periods) & y.notna().to_numpy() & p.notna().to_numpy()
        xs.append(x[m]); ps.append(p[m]); ys.append(y[m]); gs.append(np.full(int(m.sum()), k))
    return pd.concat(xs), pd.concat(ps), pd.concat(ys), np.concatenate(gs)


def scores(obs: pd.Series, sim: pd.Series, wet: float) -> list[dict]:
    z = pd.concat([obs, sim], axis=1, keys=["o", "s"]).dropna()
    rows = []
    for scale, zz, threshold in (("hourly", z, wet),
                                 ("daily", z.resample("D").sum(min_count=20).dropna(), 1.0)):
        k = kge(zz.o, zz.s)
        p99 = zz.o.quantile(0.99)
        rows.append({"scale": scale, **k, **contingency(zz.o, zz.s, threshold),
                     "p99_ratio": zz.s.quantile(0.99) / p99 if p99 > 0 else np.nan})
    return rows


def annual_bias(panel, products, gauges, d_tab):
    rows = []
    for product in products:
        for code in gauges:
            z = panel[[f"obs__precipitation__{code}", f"{product}__precipitation__{code}"]].dropna()
            z.columns = ["o", "s"]
            g = z.groupby(z.index.year)
            ratio = (g.s.sum() / g.o.sum())[g.size() >= 0.7 * 8760]
            rows += [{"product": product, "gauge": code, "year": y, "ratio": round(v, 3)}
                     for y, v in ratio.items()]
    df = pd.DataFrame(rows, columns=["product", "gauge", "year", "ratio"])
    df.to_csv(d_tab / "06_annual_bias.csv", index=False)
    if len(df):
        era = np.where(df.year <= 2014, "TRMM era (<=2014)", "GPM era (>=2015)")
        log.info("Median product/gauge ratio by era:\n%s",
                 df.groupby(["product", era]).ratio.median().unstack().round(3).to_string())


def exceedance_figure(curves: pd.DataFrame, stem: Path):
    """Exceedance curves of hourly precipitation in validation. Line style identifies the
    series and colour the product; the legend therefore uses neutral colours."""
    from matplotlib.lines import Line2D
    products = list(dict.fromkeys(curves["product"]))
    fig, axs = pl.new_figure("double", 62, ncols=len(products), sharey=True)
    for k, (ax, product) in enumerate(zip(np.atleast_1d(axs), products)):
        color = pl.PRODUCT_COLOR[product]
        styles = {"Gauges": dict(color="black", lw=1.4), "Raw": dict(color=color, ls="--"),
                  "QM": dict(color="#7F7F7F", ls=":", lw=1.2),
                  "ML-corrected": dict(color=color, lw=1.2)}
        for label, style in styles.items():
            v = np.sort(curves.loc[(curves["product"] == product) & (curves.series == label),
                                   "value"].to_numpy())[::-1]
            if len(v):
                ax.loglog(np.arange(1, len(v) + 1) / len(v), np.clip(v, 1e-3, None), **style)
        ax.set(title=pl.PRODUCT_LABEL[product], xlabel="Exceedance probability",
               xlim=(1e-5, 0.5), ylim=(1e-2, 60))
        pl.panel_label(ax, "abcd"[k])
    np.atleast_1d(axs)[0].set_ylabel("Hourly precipitation (mm)")
    legend = [Line2D([], [], color="black", lw=1.4), Line2D([], [], color="#4D4D4D", ls="--"),
              Line2D([], [], color="#7F7F7F", ls=":", lw=1.2),
              Line2D([], [], color="#A6A6A6", lw=1.6)]
    fig.legend(legend, ["Gauges", "Raw product", "QM reference", "ML-corrected"],
               loc="outside lower center", ncol=4)
    pl.save(fig, stem)


# --------------------------------------------------------------------------- main
def main(only_figures: bool = False):
    cfg = load_config()
    if only_figures:
        f = path(cfg, "processed") / "bias_correction_validation.parquet"
        if not f.exists():
            sys.exit(f"{f} not found: run the stage once without --figures-only.")
        exceedance_figure(pd.read_parquet(f), path(cfg, "figures") / "06_exceedance")
        return
    bc = cfg["bias_correction"]
    panel = pd.read_parquet(path(cfg, "processed") / "hourly_panel.parquet")
    gauges = [c for c in cfg["stations"]["rain_gauges"] if f"obs__precipitation__{c}" in panel]
    base = [p for p in bc["products"] if f"{p}__precipitation__{gauges[0]}" in panel]
    if not base:
        sys.exit("The hourly panel has no satellite products: run stages 04 and 05.")
    products = base + (["fusion"] if bc["fusion"] and len(base) > 1 else [])
    validation, training = cfg["periods"]["validation"], bc["training_periods"]
    in_val = in_period(panel.index, validation)
    scored = [g for g in gauges
              if panel.loc[in_val, f"obs__precipitation__{g}"].notna().sum() >= bc["min_validation_hours"]]
    skipped = sorted(set(gauges) - set(scored))
    log.info("Site %s | gauges %s | scored in validation %s%s", cfg["site"], gauges, scored,
             f" | trained only: {skipped}" if skipped else "")
    cache = FeatureCache(panel, bc, base)
    annual_bias(panel, base, gauges, path(cfg, "tables"))

    evaluation, importance, output, curves = [], [], {}, []
    for product in products:
        obs_val, raw_val, qm_val, ml_val = [], [], [], []
        for code in scored:                                  # leave one gauge out
            others = [c for c in gauges if c != code]
            x, p_tr, y, g = stack(cache, panel, product, others, training)
            ml = MLCorrector(bc).fit(x, y, g)
            qm = QMReference(bc).fit(p_tr, y, g)
            p, xv = cache.get(product, code)
            m = in_period(xv.index, validation) & p.notna().to_numpy()
            obs = panel.loc[m, f"obs__precipitation__{code}"]
            versions = {"raw": p[m],
                        "qm_logo": pd.Series(qm.predict(p[m]), index=p.index[m]),
                        "ml_logo": pd.Series(ml.predict(xv[m]), index=p.index[m])}
            for version, sim in versions.items():
                for row in scores(obs, sim, bc["wet_threshold_mm"]):
                    evaluation.append({"product": product, "gauge": code, "version": version,
                                       **row})
            ok = obs.notna().to_numpy()
            obs_val.append(obs[ok]); raw_val.append(versions["raw"][ok])
            qm_val.append(versions["qm_logo"][ok]); ml_val.append(versions["ml_logo"][ok])
            log.info("%s: leave-out %s done", product, code)
        for label, parts in (("Gauges", obs_val), ("Raw", raw_val), ("QM", qm_val),
                             ("ML-corrected", ml_val)):
            v = pd.concat(parts).dropna().to_numpy()
            curves.append(pd.DataFrame({"product": product, "series": label, "value": v}))

        x, p_tr, y, g = stack(cache, panel, product, gauges, training)
        ml = MLCorrector(bc).fit(x, y, g)
        qm = QMReference(bc).fit(p_tr, y, g)
        importance += [pd.Series(ml.reg.booster_.feature_importance("gain"), index=x.columns,
                                 name=f"{product}_regressor"),
                       pd.Series(ml.clf.booster_.feature_importance("gain"), index=x.columns,
                                 name=f"{product}_classifier")]
        for cu in cfg["forcing"]["catchments"]:
            needed = [f"{b}__precipitation__catchment-{cu}"
                      for b in (base if product == "fusion" else [product])]
            if any(c not in panel for c in needed):
                log.warning("Missing catchment series for %s in %s", product, cu)
                continue
            p, xc = cache.get(product, f"catchment-{cu}")
            m = p.notna().to_numpy()
            output[f"{product}__raw__catchment-{cu}"] = p
            output[f"{product}__corr__catchment-{cu}"] = pd.Series(
                ml.predict(xc[m]), index=p.index[m]).reindex(p.index)
            if bc["qm_reference"]:
                output[f"{product}__qm__catchment-{cu}"] = pd.Series(
                    qm.predict(p[m]), index=p.index[m]).reindex(p.index)

    ev = pd.DataFrame(evaluation)
    d_tab = path(cfg, "tables")
    ev.round(4).to_csv(d_tab / "06_bias_correction_evaluation.csv", index=False)
    pd.concat(importance, axis=1).round(1).to_csv(d_tab / "06_feature_importance.csv")
    pd.DataFrame(output).to_parquet(path(cfg, "processed") / "corrected_precipitation.parquet")
    curves = pd.concat(curves, ignore_index=True)
    curves.to_parquet(path(cfg, "processed") / "bias_correction_validation.parquet")
    exceedance_figure(curves, path(cfg, "figures") / "06_exceedance")
    summary = ev.groupby(["product", "version", "scale"])[
        ["kge", "r", "beta", "gamma", "csi", "frequency_ratio", "p99_ratio"]].mean().round(3)
    log.info("Validation %s to %s (mean of %d gauges):\n%s", validation[0], validation[1],
             len(scored), summary.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures-only", action="store_true",
                    help="redraw the exceedance figure from the saved validation series")
    main(ap.parse_args().figures_only)
