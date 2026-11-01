"""
Stage 05 - Hourly panel, time-stamp anchor and time offsets

Purpose
    Combine station records and Earth Engine extractions into one hourly panel in local time
    (UTC-5) where the value labelled T always refers to the interval (T-1 h, T], and verify
    that convention independently for every source.

Methods
    UTC to local time, 30-min IMERG rates averaged to hourly depths, unit conversion and the
    time-label convention of each source (config: project.station_time_label and
    gee.products.<p>.time_label). Solar anchor: the mean diurnal cycle of measured and ERA5-Land
    radiation is matched against extraterrestrial radiation; the shift of the hour centre does
    not depend on clouds or on any other product and fixes the convention of stations and
    ERA5-Land. Residual offsets of the precipitation products are then estimated by lagged
    Spearman correlation with a parabolic sub-hourly peak; they reflect retrieval timing and
    are absorbed later by the bias correction, not by relabelling.

Tools
    pandas, numpy, matplotlib; paramo.solar.

Inputs
    data/interim/<site>/<code>_<variable>.parquet (stage 01); data/raw/<site>/gee/*.csv (04)

Outputs
    data/processed/<site>/hourly_panel.parquet     columns <source>__<variable>__<id>
    outputs/<site>/tables/05_solar_anchor.csv, 05_time_offsets.csv
    outputs/<site>/figures/05_diurnal_precipitation.(png|pdf)

Next stage
    06 - Bias correction, fusion and QM reference.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from paramo import plotting as pl
from paramo.core import ROOT, get_logger, load_config, path, read_qc
from paramo.solar import interval_offset

log = get_logger("05_harmonization")
ONE_HOUR = np.timedelta64(1, "h")
BAND_VARIABLE = {"precipitation": "precipitation", "hourlyPrecipRate": "precipitation",
                 "total_precipitation_hourly": "precipitation",
                 "temperature_2m": "temperature",
                 "surface_solar_radiation_downwards_hourly": "radiation"}


def read_product(cfg, product: str, d_gee: Path) -> pd.DataFrame | None:
    p = cfg["gee"]["products"][product]
    files = sorted(d_gee.glob(f"{product}_*_*.csv"))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f, engine="pyarrow").assign(target=f.stem.split("_")[1])
                    for f in files])
    df["t"] = (pd.to_datetime(df["t"], unit="ms", utc=True)
               .dt.tz_convert(cfg["project"]["local_timezone"]).dt.tz_localize(None))
    long = []
    for band in p["bands"]:
        if band not in df:
            continue
        factor, offset = p["conversion"][band]
        x = df[["t", "id", "target", band]].rename(columns={band: "v"})
        x["v"] = x["v"] * factor + offset
        x["hour"] = x["t"].dt.floor("h")          # 30-min rates -> hourly mean rate = depth
        x = x.groupby(["hour", "id", "target"], as_index=False)["v"].mean()
        variable = BAND_VARIABLE[band]
        if p["time_label"] == "start" and variable != "temperature":
            x["hour"] = x["hour"] + ONE_HOUR       # accumulations: label at interval end
        ident = x["id"].where(x["target"] == "points", "catchment-" + x["id"])
        x["col"] = product + "__" + variable + "__" + ident
        long.append(x)
    return pd.concat(long).pivot_table(index="hour", columns="col", values="v")


def observations(cfg) -> pd.DataFrame:
    st = cfg["stations"]
    codes = sorted({*st["streamflow"], *st["rain_gauges"], *st["weather"]})
    cols = {}
    for code in codes:
        for var in cfg["variables"]:
            d = read_qc(cfg, code, var)
            if d is not None:
                cols[f"obs__{var}__{code}"] = d["value_qc"]
    obs = pd.DataFrame(cols)
    if cfg["project"]["station_time_label"] == "start":
        obs.index = obs.index + ONE_HOUR
    obs.index.name = "time"
    return obs


def solar_anchor(panel, cfg) -> pd.DataFrame:
    st = pd.read_csv(ROOT / cfg["paths"]["tables"] / "00_stations.csv").set_index("code")
    rows = []
    for c in [c for c in panel if "__radiation__" in c and "catchment-" not in c]:
        source, _, code = c.split("__")
        if code not in st.index or panel[c].notna().sum() < 24 * 90:
            continue
        d, r = interval_offset(panel[c], st.at[code, "lat"], st.at[code, "lon"])
        shift_h = int(round((d + 30) / 60))
        if source == "obs":
            current, key = cfg["project"]["station_time_label"], "project.station_time_label"
        else:
            current, key = (cfg["gee"]["products"][source]["time_label"],
                            f"gee.products.{source}.time_label")
        if shift_h == 0:
            advice = "correct"
        elif shift_h == 1 and current == "end":
            advice = f"set {key} to 'start'"
        elif shift_h == -1 and current == "start":
            advice = f"set {key} to 'end'"
        else:
            advice = f"offset of {shift_h:+d} h not explained by the label: check the time zone"
        rows.append({"series": c, "centre_offset_min": d + 30, "shape_r": r,
                     "current_label": current, "advice": advice})
        (log.info if advice == "correct" else log.warning)(
            "Solar anchor %s: hour centre at %+d min from expected -> %s", c, d + 30, advice)
    return pd.DataFrame(rows, columns=["series", "centre_offset_min", "shape_r",
                                       "current_label", "advice"])


def parabolic_peak(r: dict) -> float:
    lags = sorted(r)
    k = max(r, key=r.get)
    if k in (lags[0], lags[-1]):
        return float(k)
    y0, y1, y2 = r[k - 1], r[k], r[k + 1]
    den = y0 - 2 * y1 + y2
    return float(k) if den == 0 else round(k + 0.5 * (y0 - y2) / den, 2)


def time_offsets(panel, cfg) -> pd.DataFrame:
    rows, lags = [], cfg["harmonization"]["test_lags_h"]
    for c_obs in [c for c in panel if c.startswith(("obs__precipitation__", "obs__temperature__"))]:
        _, var, code = c_obs.split("__")
        for c_sat in [c for c in panel if c.endswith(f"__{var}__{code}") and not c.startswith("obs")]:
            z = panel[[c_obs, c_sat]].dropna()
            if len(z) < 1000:
                continue
            method = "spearman" if var == "precipitation" else "pearson"
            r = {lag: z[c_obs].corr(z[c_sat].shift(lag), method=method) for lag in lags}
            peak = parabolic_peak(r)
            rows.append({"station": code, "product": c_sat.split("__")[0], "variable": var,
                         "hours": len(z), **{f"r_lag{lag:+d}": round(v, 3) for lag, v in r.items()},
                         "best_lag_h": max(r, key=r.get), "subhourly_lag_h": peak})
            if var == "precipitation" and abs(peak) > 0.75:
                log.warning("%s vs %s: peak at %+.2f h (negative = product late); check the "
                            "solar anchor first", c_obs, c_sat, peak)
    return pd.DataFrame(rows)


def diurnal_figure(panel, cfg, stem: Path):
    gauges = [c for c in cfg["stations"]["rain_gauges"] if f"obs__precipitation__{c}" in panel]
    if not gauges:
        return
    ncols = min(3, len(gauges))
    nrows = math.ceil(len(gauges) / ncols)
    fig, axs = pl.new_figure("double", 18 + 48 * nrows, nrows=nrows, ncols=ncols, sharey=True,
                             squeeze=False)
    for k, code in enumerate(gauges):
        ax = axs.flat[k]
        cols = sorted([c for c in panel if c.endswith(f"__precipitation__{code}")],
                      key=lambda c: (not c.startswith("obs"), c))
        z = panel[cols].dropna()
        for c in cols:
            src = c.split("__")[0]
            cycle = z[c].groupby(z.index.hour).mean()
            ax.plot(cycle.index, cycle / cycle.mean(),
                    color=pl.PRODUCT_COLOR.get("gauge" if src == "obs" else src, "grey"),
                    lw=1.6 if src == "obs" else 1.0,
                    label="Gauge" if src == "obs" else pl.PRODUCT_LABEL.get(src, src))
        ax.set(title=code, xticks=range(0, 24, 6), xlim=(0, 23))
        if k // ncols == nrows - 1 or k + ncols >= len(gauges):
            ax.set_xlabel("Local hour (end of interval)")
        pl.panel_label(ax, "abcdefghi"[k])
    for ax in axs.flat[len(gauges):]:
        ax.set_visible(False)
    for r in range(nrows):
        axs[r, 0].set_ylabel("Hourly mean P / daily mean")
    handles, labels = axs.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels))
    pl.save(fig, stem)


def main():
    cfg = load_config()
    d_gee = path(cfg, "raw_gee")
    panel = observations(cfg)
    n_products = 0
    for product in cfg["gee"]["products"]:
        w = read_product(cfg, product, d_gee)
        if w is None:
            log.warning("No Earth Engine files for %s in %s", product, d_gee)
            continue
        n_products += 1
        panel = panel.join(w, how="outer")
        log.info("%s: %d columns, %s to %s", product, w.shape[1], w.index.min(), w.index.max())
    if n_products == 0:
        log.error("No satellite data in %s; run stage 04 for site %s first.", d_gee, cfg["site"])
        sys.exit(1)
    idx = pd.date_range(panel.index.min(), panel.index.max(), freq="h", name="time")
    panel = panel.reindex(idx).sort_index(axis=1)
    panel.to_parquet(path(cfg, "processed") / "hourly_panel.parquet")
    log.info("Panel: %d hours x %d columns", *panel.shape)

    anchor = solar_anchor(panel, cfg)
    anchor.to_csv(path(cfg, "tables") / "05_solar_anchor.csv", index=False)
    offsets = time_offsets(panel, cfg)
    offsets.to_csv(path(cfg, "tables") / "05_time_offsets.csv", index=False)
    if len(offsets):
        log.info("Offsets:\n%s", offsets[["station", "product", "variable", "best_lag_h",
                                          "subhourly_lag_h"]].to_string(index=False))
    diurnal_figure(panel, cfg, path(cfg, "figures") / "05_diurnal_precipitation")


if __name__ == "__main__":
    main()
