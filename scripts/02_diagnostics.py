"""
Stage 02 - Rating curve, homogeneity and gauge diagnostics

Purpose
    Describe the quality-controlled series before modelling: daily aggregation, stability of
    the stage-discharge relation (where stage is available), homogeneity of annual and monthly
    series, double-mass consistency between rain gauges and with discharge, the relation
    between the nested streamflow gauges, and which rain gauge best represents each catchment.

Methods
    Year-by-year power-law fits of discharge against stage (a single, never-updated rating
    curve yields identical discharge for a given stage in every year); Pettitt and SNHT tests
    (SNHT critical values by Monte Carlo for each sample size) on annual totals and monthly
    anomalies; double-mass curves; hourly cross-correlation; Spearman correlation between
    daily rainfall and the change in daily discharge.

Tools
    pandas, numpy, scipy, matplotlib; paramo.stats.

Inputs
    data/interim/<site>/<code>_<variable>.parquet (stage 01)

Outputs
    data/interim/<site>/daily.parquet
    outputs/<site>/tables/02_rating_curve.csv, 02_homogeneity.csv, 02_gauge_relation.csv,
                          02_rain_flow_correlation.csv
    outputs/<site>/figures/02_rating_curve, 02_double_mass, 02_nested_gauges (png|pdf)

Next stage
    03 - DEM, catchments and attributes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from paramo import plotting as pl
from paramo.core import get_logger, load_config, path, read_qc, thiessen_gauges
from paramo.stats import pettitt, snht, snht_critical

log = get_logger("02_diagnostics")
LINE_COLORS = [pl.OKABE_ITO[c] for c in ("blue", "orange", "green", "purple", "vermillion",
                                         "sky", "black")]


def hourly_series(cfg) -> dict:
    out = {}
    st = cfg["stations"]
    for code in st["streamflow"]:
        for var in ("discharge", "stage"):
            d = read_qc(cfg, code, var)
            if d is not None:
                out[f"{code}_{var}"] = d["value_qc"]
    for code in st["rain_gauges"]:
        d = read_qc(cfg, code, "precipitation")
        if d is not None:
            out[f"{code}_precipitation"] = d["value_qc"]
    for var in ("temperature", "radiation"):
        for code in st["weather"]:
            d = read_qc(cfg, code, var)
            if d is not None:
                out[f"{code}_{var}"] = d["value_qc"]
    return out


def daily(s: pd.Series, how: str, min_hours: int) -> pd.Series:
    g = s.resample("D")
    agg = g.sum() if how == "sum" else g.mean()
    return agg.where(g.count() >= min_hours)


def rating_curve(h: pd.Series, q: pd.Series, code: str, ax):
    z = pd.concat([h, q], axis=1, keys=["h", "q"]).dropna()
    z = z[(z.h > 0) & (z.q > 0)]
    levels = z.h.quantile([0.1, 0.5, 0.9]).to_numpy()
    rows = []
    for year, g in z.groupby(z.index.year):
        if len(g) < 500:
            continue
        b, a = np.polyfit(np.log(g.h), np.log(g.q), 1)
        row = {"code": code, "year": year, "n": len(g), "exponent_b": round(b, 4)}
        for name, lv in zip(("p10", "p50", "p90"), levels):
            row[f"Q_at_stage_{name}"] = round(float(np.exp(a + b * np.log(lv))), 4)
        rows.append(row)
    classes = pd.cut(z.h, 200)
    cv = z.groupby(classes, observed=True).q.std() / z.groupby(classes, observed=True).q.mean()
    log.info("%s: median CV of Q within stage classes = %.4f (about 0 means a single curve)",
             code, cv.median())
    sc = ax.scatter(z.h, z.q, c=z.index.year, s=0.6, cmap="viridis", rasterized=True)
    ax.set(xscale="log", yscale="log", xlabel="Stage (cm)", ylabel=f"Discharge ({pl.Q_UNIT})",
           title=code)
    return rows, sc


def homogeneity(name: str, s: pd.Series, scale: str, crit_cache: dict) -> dict:
    s = s.dropna()
    k, _, p = pettitt(s.to_numpy())
    ks, t0 = snht(s.to_numpy())
    n = len(s)
    if n not in crit_cache:
        crit_cache[n] = snht_critical(n)
    return {"series": name, "scale": scale, "n": n, "start": s.index.min().date(),
            "end": s.index.max().date(),
            "pettitt_date": s.index[k].date() if k >= 0 else None, "pettitt_p": round(p, 4),
            "snht_date": s.index[ks].date() if ks >= 0 else None, "snht_T0": round(t0, 2),
            "snht_crit95": round(crit_cache[n], 2), "snht_reject": bool(t0 > crit_cache[n])}


def main():
    cfg = load_config()
    ag = cfg["aggregation"]
    d_tab, d_fig = path(cfg, "tables"), path(cfg, "figures")
    hourly = hourly_series(cfg)
    day = pd.DataFrame({k: daily(s, "sum" if k.endswith("precipitation") else "mean",
                                 ag["min_hours_day"]) for k, s in hourly.items()})
    day.to_parquet(path(cfg, "interim") / "daily.parquet")
    gauges = cfg["stations"]["streamflow"]

    # --- rating curve (only gauges with a stage record)
    with_stage = [c for c in gauges if f"{c}_stage" in hourly and f"{c}_discharge" in hourly]
    rows = []
    if with_stage:
        fig, axs = pl.new_figure("double", 70, ncols=len(with_stage))
        for k, (ax, code) in enumerate(zip(np.atleast_1d(axs), with_stage)):
            r, sc = rating_curve(hourly[f"{code}_stage"], hourly[f"{code}_discharge"], code, ax)
            rows += r
            pl.panel_label(ax, "abcd"[k])
        fig.colorbar(sc, ax=axs, shrink=0.9, pad=0.01).set_label("Year")
        pl.save(fig, d_fig / "02_rating_curve")
    else:
        log.info("No stage records at this site: rating-curve diagnostic skipped")
    pd.DataFrame(rows, columns=["code", "year", "n", "exponent_b", "Q_at_stage_p10",
                                "Q_at_stage_p50", "Q_at_stage_p90"]).to_csv(
        d_tab / "02_rating_curve.csv", index=False)

    # --- homogeneity: annual series (complete years) and monthly anomalies
    crit, rows = {}, []
    for k in [c for c in day if c.endswith(("discharge", "precipitation"))]:
        s = day[k]
        how = "sum" if k.endswith("precipitation") else "mean"
        n_days = s.resample("YE").count()
        annual = s.resample("YE").sum() if how == "sum" else s.resample("YE").mean()
        rows.append(homogeneity(k, annual[n_days >= 365 * ag["min_annual_completeness"] / 100],
                                "annual", crit))
        n_month = s.resample("ME").count()
        monthly = s.resample("ME").sum() if how == "sum" else s.resample("ME").mean()
        monthly = monthly[n_month >= ag["min_days_month"]]
        anom = monthly - monthly.groupby(monthly.index.month).transform("mean")
        rows.append(homogeneity(k, anom, "monthly_anomaly", crit))
    hom = pd.DataFrame(rows)
    hom.to_csv(d_tab / "02_homogeneity.csv", index=False)
    log.info("Homogeneity:\n%s", hom.to_string(index=False))

    # --- double mass (gauges of the areal precipitation)
    pcols = [f"{g}_precipitation" for g in thiessen_gauges(cfg) if f"{g}_precipitation" in day]
    monthly_p = day[pcols].resample("ME").sum(min_count=ag["min_days_month"]).dropna()
    fig, axs = pl.new_figure("double", 70, ncols=2)
    ref = monthly_p.mean(axis=1).cumsum()
    for c, color in zip(pcols, LINE_COLORS):
        axs[0].plot(ref, monthly_p[c].cumsum(), color=color, label=c.split("_")[0])
    axs[0].set(xlabel="Cumulative network mean P (mm)", ylabel="Cumulative station P (mm)")
    axs[0].legend(loc="upper left")
    pl.panel_label(axs[0], "a")
    # common months (precipitation and both gauges valid), so the curves share the x axis
    p_m = day[pcols].resample("ME").sum(min_count=ag["min_days_month"]).mean(axis=1, skipna=False)
    q_m = {}
    for code in gauges:
        q = day[f"{code}_discharge"]
        q_m[code] = q.resample("ME").mean().where(q.resample("ME").count() >= ag["min_days_month"])
    z = pd.concat([p_m.rename("P")] + [q_m[c].rename(c) for c in gauges], axis=1).dropna()
    cum_p = z.P.cumsum()
    for code, color in zip(gauges, (pl.OKABE_ITO["vermillion"], "black")):
        vol = (z[code] * z.index.days_in_month).cumsum()
        axs[1].plot(cum_p, vol / vol.iloc[-1], color=color, marker="o", ms=1.2, lw=0.8,
                    label=code)
    vol = (z[gauges[-1]] * z.index.days_in_month).cumsum()
    for t in z.index[(z.index.month == 1) & (z.index.year % 2 == 0)]:
        axs[1].annotate(str(t.year), (cum_p[t], vol[t] / vol.iloc[-1]), xytext=(4, -9),
                        textcoords="offset points", fontsize=6, color="#4D4D4D")
    log.info("Double mass of discharge: %d common months (%s to %s)", len(z),
             f"{z.index[0]:%Y-%m}", f"{z.index[-1]:%Y-%m}")
    axs[1].set(xlabel="Cumulative mean P (mm)", ylabel="Cumulative discharge volume (fraction)")
    axs[1].legend(loc="upper left")
    pl.panel_label(axs[1], "b")
    pl.save(fig, d_fig / "02_double_mass")

    # --- nested gauges (first = upstream/transfer, second = downstream/calibration)
    up, down = gauges[0], gauges[1]
    a, b = hourly[f"{up}_discharge"], hourly[f"{down}_discharge"]
    x = pd.concat([a, b], axis=1, keys=["up", "down"], sort=True).dropna()
    lags = range(-6, 13)
    cc = {lag: x.up.corr(x.down.shift(-lag)) for lag in lags}
    dd = day[[f"{up}_discharge", f"{down}_discharge"]].dropna()
    dd.columns = ["up", "down"]
    rel = []
    for season, months in {"all": range(1, 13), "Jun-Aug": [6, 7, 8],
                           "Nov-Feb": [11, 12, 1, 2]}.items():
        sub = dd[dd.index.month.isin(months)]
        rel.append({"period": season, "days": len(sub),
                    "pearson_daily": round(sub.corr().iloc[0, 1], 3),
                    "spearman_daily": round(sub.corr("spearman").iloc[0, 1], 3),
                    "ratio_down_up": round(sub.down.mean() / sub.up.mean(), 2)})
    rel = pd.DataFrame(rel)
    rel["lag_h_max_corr"] = max(cc, key=cc.get)
    rel.to_csv(d_tab / "02_gauge_relation.csv", index=False)
    log.info("Nested gauges:\n%s", rel.to_string(index=False))

    fig, axs = pl.new_figure("double", 70, ncols=2)
    sc = axs[0].scatter(dd.up, dd.down, s=1.5, c=dd.index.month, cmap="twilight",
                        rasterized=True)
    axs[0].set(xscale="log", yscale="log", xlabel=f"{up} daily Q ({pl.Q_UNIT})",
               ylabel=f"{down} daily Q ({pl.Q_UNIT})")
    pl.plain_log_axis(axs[0].xaxis)
    pl.plain_log_axis(axs[0].yaxis)
    fig.colorbar(sc, ax=axs[0], pad=0.01).set_label("Month")
    pl.panel_label(axs[0], "a")
    axs[1].plot(list(cc), list(cc.values()), marker="o", ms=2.5, color=pl.OKABE_ITO["blue"])
    axs[1].set(xlabel=f"Lag of {down} relative to {up} (h)", ylabel="Hourly correlation")
    pl.integer_axis(axs[1].xaxis)
    pl.panel_label(axs[1], "b")
    pl.save(fig, d_fig / "02_nested_gauges")

    # --- which rain gauge represents each catchment
    rows = []
    for qc in [f"{g}_discharge" for g in gauges]:
        for pc in [c for c in day if c.endswith("precipitation")]:
            z = day[[pc, qc]].dropna()
            for lag in (0, 1, 2):
                rows.append({"gauge": qc.split("_")[0], "rain_gauge": pc.split("_")[0],
                             "lag_days": lag, "n": len(z),
                             "spearman_P_dQ": round(z[pc].corr(z[qc].diff().shift(-lag),
                                                               method="spearman"), 3)})
    pd.DataFrame(rows).to_csv(d_tab / "02_rain_flow_correlation.csv", index=False)


if __name__ == "__main__":
    main()
