"""
Stage 07 - Catchment forcing scenarios

Purpose
    Build the hourly model inputs of each catchment of the site: the precipitation scenarios
    of the factorial design, temperature, potential evapotranspiration (PET), observed
    discharge in mm h-1 and the hypsometric curve.

Methods
    Gauge scenario: monthly consistency control against the neighbouring gauges (normal-ratio
    expectation; months with a persistent 12-month ratio outside the configured band, or an
    extreme isolated ratio, are removed), gap filling by the monthly normal-ratio method and
    fixed Thiessen weights over forcing.thiessen_gauges. Fixed weights avoid the artificial
    jumps that a dynamic Thiessen introduces when a dry and a wet gauge alternate. Satellite
    scenarios come from stage 06 (raw, ML-corrected and QM reference). Temperature: ERA5-Land
    at the temperature station corrected by its month-by-hour mean difference with the station,
    then moved to the mean catchment elevation with a constant lapse rate. PET: Oudin et al.
    (2005) at daily scale, distributed within the day by hourly extraterrestrial radiation.
    Temperature and PET are identical in every scenario, so skill differences between
    scenarios come only from precipitation. A water-balance check flags scenarios whose P - Q
    is negative or exceeds PET.

Tools
    geopandas/shapely (Thiessen polygons), rasterio (hypsometry), pandas, numpy, matplotlib;
    paramo.solar.

Inputs
    data/processed/<site>/hourly_panel.parquet (05), corrected_precipitation.parquet (06)
    data/interim/<site>/stations.gpkg, catchments.gpkg (00, 03); data/raw/<site>/dem (03)

Outputs
    data/processed/<site>/forcing_<catchment>.parquet and .csv (the CSV is read by stage 08)
    outputs/<site>/tables/07_thiessen_weights.csv, 07_hypsometry.csv,
        07_temperature_correction.csv, 07_forcing_summary.csv, 07_flagged_gauge_months.csv,
        07_annual_coherence.csv
    outputs/<site>/figures/07_annual_balance.(png|pdf)

Next stage
    08 - GR4H calibration, validation and transfer (R, airGR).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as raster_mask
from shapely.geometry import MultiPoint
from shapely.ops import voronoi_diagram

from paramo import plotting as pl
from paramo.core import ROOT, get_logger, in_period, load_config, path, thiessen_gauges
from paramo.metrics import kge
from paramo.solar import extraterrestrial_hourly

log = get_logger("07_forcing")
HOURS_YEAR = 8766


# --------------------------------------------------------------------------- gauges
def consistency_control(panel, codes, q, d_tab) -> pd.DataFrame:
    """Remove months in which a gauge departs from its neighbour-based expectation. The
    expectation uses the neighbours available that month (at least two), so a gap at one
    neighbour does not hide a faulty gauge. A faulty gauge biases the expectation of the
    others by a fraction of its error; the 12-month window and the ±30–40 % band avoid
    flagging the sound gauges."""
    p = panel[[f"obs__precipitation__{c}" for c in codes]].copy()
    p.columns = codes
    monthly = p.resample("ME").sum(min_count=24 * 20)
    rows, drop = [], {}
    for g in codes:
        others = [c for c in codes if c != g]
        common = monthly[[g] + others].dropna()
        clim = common.groupby(common.index.month).mean()
        scaled = pd.concat([monthly[j] * (clim[g] / clim[j]).reindex(monthly.index.month).to_numpy()
                            for j in others], axis=1)
        # expectation from the neighbours available that month (at least two, or all if fewer)
        need = min(2, len(others))
        expected = scaled.mean(axis=1).where(scaled.notna().sum(axis=1) >= need)
        valid = monthly[g].notna() & expected.notna()
        ratio = monthly[g] / expected
        w = q["window_months"]
        moving = (monthly[g].where(valid).rolling(w, center=True, min_periods=w // 2).sum()
                  / expected.where(valid).rolling(w, center=True, min_periods=w // 2).sum())
        persistent = (moving < q["moving_ratio_min"]) | (moving > q["moving_ratio_max"])
        isolated = (((ratio < q["ratio_min"]) | (ratio > q["ratio_max"]))
                    & (expected >= q["min_expected_mm"]))
        bad = (persistent | isolated) & valid
        drop[g] = ratio.index[bad]
        for month in ratio.index[bad]:
            rows.append({"gauge": g, "month": month.strftime("%Y-%m"),
                         "P_mm": round(monthly.at[month, g], 1),
                         "expected_mm": round(expected[month], 1), "ratio": round(ratio[month], 2),
                         "moving_ratio": round(moving[month], 2),
                         "criterion": "persistent" if persistent[month] else "isolated"})
    for g, months in drop.items():
        for month in months:
            p.loc[(p.index.year == month.year) & (p.index.month == month.month), g] = np.nan
    flagged = pd.DataFrame(rows, columns=["gauge", "month", "P_mm", "expected_mm", "ratio",
                                          "moving_ratio", "criterion"])
    flagged.to_csv(d_tab / "07_flagged_gauge_months.csv", index=False)
    log.info("Months removed by the consistency control: %s",
             flagged.groupby("gauge").size().to_dict() if len(flagged) else "none")
    out = panel.copy()
    for g in codes:
        out[f"obs__precipitation__{g}"] = p[g]
    return out


def normal_ratio_fill(panel, codes, window) -> pd.DataFrame:
    """Fill each gauge from the others scaled by monthly mean ratios; the share of filled
    hours is reported for the simulated window (warm-up to the last evaluated day)."""
    p = panel[[f"obs__precipitation__{c}" for c in codes]].copy()
    p.columns = codes
    filled = p.copy()
    for g in codes:
        estimates = []
        for j in [c for c in codes if c != g]:
            z = p[[g, j]].dropna()
            ratio = z[g].groupby(z.index.month).mean() / z[j].groupby(z.index.month).mean()
            estimates.append(p[j] * ratio.reindex(p.index.month).to_numpy())
        filled[g] = p[g].fillna(pd.concat(estimates, axis=1).mean(axis=1))
    w = in_period(p.index, window)
    log.info("Fraction of hours filled by normal ratio (%s to %s):\n%s", window[0], window[1],
             (p[w].isna() & filled[w].notna()).mean().round(3).to_string())
    return filled


def thiessen_weights(geom, stations: gpd.GeoDataFrame, codes) -> dict:
    pts = stations.set_index("code").loc[list(codes)]
    cells = voronoi_diagram(MultiPoint(list(pts.geometry)), envelope=geom.buffer(20000))
    w = {}
    for cell in cells.geoms:
        code = pts.index[[cell.contains(g) for g in pts.geometry]][0]
        w[code] = cell.intersection(geom).area
    total = sum(w.values())
    return {c: w.get(c, 0.0) / total for c in codes}


def annual_coherence(panel, areas: dict, codes, d_tab):
    rows = []
    for year, g in panel.groupby(panel.index.year):
        row = {"year": year}
        for c in codes:
            s = g[f"obs__precipitation__{c}"]
            row[f"P_{c}"] = round(s.mean() * HOURS_YEAR) if s.notna().mean() > 0.8 else np.nan
        values = [row[f"P_{c}"] for c in codes if pd.notna(row[f"P_{c}"])]
        p_mean = np.mean(values) if values else np.nan
        for cu, area in areas.items():
            q = g.get(f"obs__discharge__{cu}")
            qa = q.mean() * 3.6 / area * HOURS_YEAR if q is not None and q.notna().mean() > 0.8 else np.nan
            row[f"Q_{cu}"] = round(qa) if pd.notna(qa) else np.nan
            row[f"QP_{cu}"] = round(qa / p_mean, 2) if pd.notna(qa) and pd.notna(p_mean) else np.nan
        rows.append(row)
    pd.DataFrame(rows).to_csv(d_tab / "07_annual_coherence.csv", index=False)


# --------------------------------------------------------------------------- temperature, PET
def corrected_temperature(panel, code) -> tuple[pd.Series, pd.DataFrame]:
    era, obs = panel[f"era5land__temperature__{code}"], panel[f"obs__temperature__{code}"]
    z = pd.concat([obs, era], axis=1, keys=["o", "e"]).dropna()
    delta = (z.o - z.e).groupby([z.index.month, z.index.hour]).mean()
    corrected = era + delta.reindex(list(zip(era.index.month, era.index.hour))).to_numpy()
    c = corrected.reindex(z.index)
    summary = pd.DataFrame([
        {"series": "ERA5-Land raw", "bias_c": (z.e - z.o).mean(), "r": kge(z.o, z.e)["r"],
         "rmse_c": np.sqrt(((z.e - z.o) ** 2).mean())},
        {"series": "ERA5-Land corrected (in sample)", "bias_c": (c - z.o).mean(),
         "r": kge(z.o, c)["r"], "rmse_c": np.sqrt(((c - z.o) ** 2).mean())}]).round(3)
    return corrected, summary


def oudin_hourly(t_hour: pd.Series, lat: float, lon: float) -> pd.Series:
    """Oudin: PE (mm d-1) = Ra/(λρ)·(T+5)/100 for T > -5 °C, with daily mean T and hourly Ra."""
    ra = extraterrestrial_hourly(t_hour.index, lat, lon)
    t_day = t_hour.groupby(t_hour.index.normalize()).transform("mean")
    return (ra / 2.45 * (t_day + 5) / 100).where(t_day > -5, 0.0).where(t_hour.notna())


# --------------------------------------------------------------------------- figure
def balance_figure(annual: dict, stem: Path):
    fig, axs = pl.new_figure("double", 75, ncols=len(annual), sharey=True)
    for k, (ax, (cu, a)) in enumerate(zip(np.atleast_1d(axs), annual.items())):
        years = a.index.year
        for col in [c for c in a if c.startswith("P__") and not c.endswith("_qm")]:
            name = col[3:]
            product = "gauge" if name == "gauge" else name.rsplit("_", 1)[0]
            raw = name.endswith("_raw")
            ax.plot(years, a[col], color=pl.PRODUCT_COLOR[product], ls="--" if raw else "-",
                    lw=0.8 if raw else 1.1, label=pl.scenario_label(name))
        ax.plot(years, a["Qobs_mm"], color="black", lw=2.0, label="Observed Q")
        ax.plot(years, a["PET"], color="#7F7F7F", ls=":", lw=1.2, label="PET")
        ax.set(title=cu, xlabel="Year")
        pl.integer_axis(ax.xaxis)
        pl.panel_label(ax, "ab"[k])
    np.atleast_1d(axs)[0].set_ylabel("mm yr$^{-1}$")
    handles, labels = np.atleast_1d(axs)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=6)
    pl.save(fig, stem)


# --------------------------------------------------------------------------- main
def main():
    cfg = load_config()
    fz = cfg["forcing"]
    d_int, d_tab, d_proc = path(cfg, "interim"), path(cfg, "tables"), path(cfg, "processed")
    panel = pd.read_parquet(d_proc / "hourly_panel.parquet")
    corrected = pd.read_parquet(d_proc / "corrected_precipitation.parquet")
    stations = gpd.read_file(d_int / "stations.gpkg", layer="stations")
    catchments = gpd.read_file(d_int / "catchments.gpkg", layer="catchments").set_index("code")
    attrs = pd.read_csv(d_tab / "03_catchments.csv").set_index("catchment")
    gauges = [c for c in thiessen_gauges(cfg) if f"obs__precipitation__{c}" in panel]
    log.info("Site %s | areal precipitation from %s", cfg["site"], gauges)

    annual_coherence(panel, {cu: float(attrs.at[cu, "area_km2"]) for cu in fz["catchments"]},
                     gauges, d_tab)
    panel = consistency_control(panel, gauges, fz["gauge_qc"], d_tab)
    per = cfg["periods"]
    window = (per["warmup"][0], max(per["calibration"][1], per["validation"][1],
                                    per["transfer"][1]))
    filled = normal_ratio_fill(panel, gauges, window)

    t_code = fz["temperature_station"]
    t_ref, t_summary = corrected_temperature(panel, t_code)
    t_summary.to_csv(d_tab / "07_temperature_correction.csv", index=False)
    log.info("Temperature correction at %s:\n%s", t_code, t_summary.to_string(index=False))
    z_ref = float(stations.set_index("code").at[t_code, "elevation_m"])

    weights_rows, hypsometry, summary, annual = [], {}, [], {}
    for cu in fz["catchments"]:
        geom = catchments.at[cu, "geometry"]
        area, z_mean = float(attrs.at[cu, "area_km2"]), float(attrs.at[cu, "z_mean"])
        w = thiessen_weights(geom, stations, gauges)
        weights_rows.append({"catchment": cu, **{c: round(w[c], 4) for c in gauges}})
        log.info("Thiessen weights %s: %s", cu, {c: round(v, 3) for c, v in w.items()})
        df = pd.DataFrame(index=panel.index)
        df.index.name = "time"
        df["P__gauge"] = sum(filled[c] * w[c] for c in gauges if w[c] > 0)
        for col in corrected.columns:
            product, version, target = col.split("__")
            if target == f"catchment-{cu}":
                df[f"P__{product}_{version}"] = corrected[col]
        centroid = gpd.GeoSeries([geom.centroid], crs=catchments.crs).to_crs("EPSG:4326").iloc[0]
        df["T_ref"] = t_ref
        df["T_catchment"] = t_ref + fz["lapse_rate_c_per_km"] / 1000 * (z_mean - z_ref)
        df["PET"] = oudin_hourly(df["T_catchment"], centroid.y, centroid.x)
        q = panel.get(f"obs__discharge__{cu}")
        df["Qobs_mm"] = q * 3.6 / area if q is not None else np.nan
        df.to_parquet(d_proc / f"forcing_{cu}.parquet")
        df.to_csv(d_proc / f"forcing_{cu}.csv", float_format="%.5g",
                  date_format="%Y-%m-%d %H:%M:%S")

        with rasterio.open(ROOT / cfg["paths"]["raw_dem"] / "dem_tmq.tif") as r:
            arr, _ = raster_mask(r, [geom], crop=True, nodata=-9999, filled=True)
        zc = arr[0][arr[0] > -9999]
        hypsometry[cu] = np.percentile(zc, np.linspace(0, 100, fz["hypsometry_bands"]))

        for name in ("calibration", "validation", "transfer"):
            sub = df[in_period(df.index, cfg["periods"][name])]
            if sub["Qobs_mm"].notna().sum() == 0:
                continue
            for col in [c for c in df if c.startswith("P__")]:
                ok = sub[col].notna() & sub["Qobs_mm"].notna()
                p_a = sub.loc[ok, col].mean() * HOURS_YEAR
                q_a = sub.loc[ok, "Qobs_mm"].mean() * HOURS_YEAR
                pet_a = sub.loc[ok, "PET"].mean() * HOURS_YEAR
                summary.append({"catchment": cu, "period": name, "scenario": col[3:],
                                "P_coverage_pct": round(100 * sub[col].notna().mean(), 1),
                                "hours_P_and_Q": int(ok.sum()), "P_mm_yr": round(p_a),
                                "Q_mm_yr": round(q_a), "PET_mm_yr": round(pet_a),
                                "runoff_ratio": round(q_a / p_a, 3) if p_a > 0 else np.nan,
                                "AET_over_PET": round((p_a - q_a) / pet_a, 2)})
        cover = df.resample("YE").count() / 8760          # full calendar years only
        annual[cu] = (df.resample("YE").mean() * HOURS_YEAR).where(cover > 0.8)

    pd.DataFrame(weights_rows).to_csv(d_tab / "07_thiessen_weights.csv", index=False)
    pd.DataFrame(hypsometry).round(1).to_csv(d_tab / "07_hypsometry.csv", index_label="band")
    res = pd.DataFrame(summary)
    res.to_csv(d_tab / "07_forcing_summary.csv", index=False)
    balance_figure(annual, path(cfg, "figures") / "07_annual_balance")
    bad = res[((res.AET_over_PET < 0) | (res.AET_over_PET > 1.1)) & (res.hours_P_and_Q > 24 * 365)]
    if len(bad):
        log.warning("Water balance not closable (P - Q < 0: underestimation or occult "
                    "precipitation; P - Q > PET: overestimation); the models will compensate "
                    "through exchange parameters:\n%s",
                    bad[["catchment", "period", "scenario", "runoff_ratio", "AET_over_PET"]]
                    .to_string(index=False))


if __name__ == "__main__":
    main()
