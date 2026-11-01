"""
Supplementary Fig. S2 - Consistency check between nested discharge gauges (Chalpi)

Purpose
    Show the evidence used to flag the four sensor-fault windows of the Chalpi gauges
    (quality_control.manual_exclusions): daily areal precipitation, daily discharge at the
    calibration gauge and at the nested gauge scaled by the area ratio, and the ratio of daily
    discharges, with the excluded windows shaded.

Methods
    Hourly discharge as recorded ('value' of stage 01, i.e. before the manual exclusions, so
    that the faulty windows remain visible) is averaged to daily values; days with fewer than
    20 valid hours are omitted. Areal precipitation is the rain-gauge (Thiessen) forcing of
    stage 07 summed by day. The area ratio comes from stage 03.

Tools
    pandas, numpy, matplotlib; paramo.core, paramo.plotting.

Inputs
    data/interim/<site>/<gauge>_discharge.parquet (stage 01)
    data/processed/<site>/forcing_<calibration catchment>.parquet (stage 07)
    outputs/<site>/tables/03_catchments.csv (stage 03)

Outputs
    outputs/<site>/figures/S2_nested_gauges (png|pdf)

Usage
    python scripts/S2_nested_gauges.py                       Chalpi, July 2011-December 2018
    python scripts/S2_nested_gauges.py --start 2012-01-01 --end 2018-12-31
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from paramo import plotting as pl
from paramo.core import ROOT, get_logger, load_config, path

log = get_logger("S2_nested_gauges")
MIN_HOURS = 20


def daily_mean(f: Path, start, end) -> pd.Series:
    v = pd.read_parquet(f)["value"].loc[start:end]
    n = v.notna().groupby(v.index.normalize()).sum()
    return v.groupby(v.index.normalize()).mean().where(n >= MIN_HOURS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=os.environ.get("PARAMO_SITE", "chalpi"))
    ap.add_argument("--start", default="2011-07-01")
    ap.add_argument("--end", default="2018-12-31 23:00")
    a = ap.parse_args()
    cfg = load_config(a.site)
    cc, ct = cfg["gr4h"]["calibration_catchment"], cfg["gr4h"]["transfer_catchment"]
    d_int, d_fig = ROOT / cfg["paths"]["interim"], path(cfg, "figures")

    qd = daily_mean(d_int / f"{cc}_discharge.parquet", a.start, a.end)
    qu = daily_mean(d_int / f"{ct}_discharge.parquet", a.start, a.end)
    cat = pd.read_csv(ROOT / cfg["paths"]["tables"] / "03_catchments.csv").set_index("catchment")
    ratio_area = cat.loc[cc, "area_km2"] / cat.loc[ct, "area_km2"]
    p = pd.read_parquet(ROOT / cfg["paths"]["processed"] / f"forcing_{cc}.parquet")["P__gauge"]
    p = p.loc[a.start:a.end]
    pdly = p.groupby(p.index.normalize()).sum(min_count=MIN_HOURS)
    days = pd.date_range(pd.Timestamp(a.start).normalize(), pd.Timestamp(a.end).normalize())
    qd, qu, pdly = qd.reindex(days), qu.reindex(days), pdly.reindex(days)
    ratio = qd / qu

    windows = cfg.get("quality_control", {}).get("manual_exclusions", []) or []
    faulty = pd.Series(False, index=days)
    for w in windows:
        faulty |= (days >= pd.Timestamp(w["start"]).normalize()) & \
                  (days <= pd.Timestamp(w["end"]).normalize())
    log.info("%s/%s: area ratio %.1f; daily correlation %.2f with all days, %.2f without the "
             "excluded windows", cc, ct, ratio_area, qd.corr(qu),
             qd[~faulty].corr(qu[~faulty]))
    for label, r in (("outside the excluded windows", ratio[~faulty].dropna()),
                     ("inside the excluded windows", ratio[faulty].dropna())):
        if len(r):
            q05, q25, q50, q75, q95 = r.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
            log.info("Daily ratio %s (%d days): median %.1f, IQR %.1f-%.1f, 5-95 %% %.1f-%.1f; "
                     "share within 16-18: %.0f %%, within 10-30: %.0f %%", label, len(r), q50,
                     q25, q75, q05, q95, 100 * r.between(16, 18).mean(),
                     100 * r.between(10, 30).mean())

    fig, axs = pl.new_figure("double", 150, nrows=3, sharex=True,
                             gridspec_kw={"height_ratios": [1, 2, 1.5]})
    axs[0].bar(pdly.index, pdly.to_numpy(), width=1.0, color=pl.OKABE_ITO["grey"], lw=0)
    axs[0].invert_yaxis()
    axs[0].set_ylabel("Areal P\n(mm d$^{-1}$)")
    axs[1].plot(qd.index, qd, lw=0.7, color=pl.OKABE_ITO["blue"],
                label=f"{cc} ({cat.loc[cc, 'area_km2']:.1f} km$^2$)")
    axs[1].plot(qu.index, qu * ratio_area, lw=0.7, color=pl.OKABE_ITO["vermillion"],
                label=f"{ct} × {ratio_area:.1f} (area ratio)")
    axs[1].set(yscale="log", ylabel=f"Daily Q ({pl.Q_UNIT})")
    axs[1].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
                  borderaxespad=0.1)                # above the panel, never over the data
    axs[2].plot(ratio.index, ratio, lw=0.6, color="0.2")
    axs[2].axhspan(16, 18, color=pl.OKABE_ITO["sky"], alpha=0.3, lw=0)
    axs[2].set(yscale="log", ylabel=f"Ratio {cc} / {ct}")
    axs[2].set_yticks([2, 5, 10, 16, 30, 100], ["2", "5", "10", "16", "30", "100"])
    # Window labels sit in head-room added above the data, centred on their window; when two
    # windows are close, the later label is moved one line down so that they never overlap
    top = float(np.nanmax(ratio.to_numpy())) if ratio.notna().any() else 100.0
    axs[2].set_ylim(top=top * 4)
    roman = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii"]
    centres = []
    for k, w in enumerate(windows):                  # numbered in configuration order
        t0, t1 = pd.Timestamp(w["start"]), pd.Timestamp(w["end"])
        for ax in axs:
            ax.axvspan(t0, t1, color="0.85", lw=0, zorder=0)
        tc = t0 + (t1 - t0) / 2
        row = sum(abs((tc - c).days) < 150 for c in centres)
        centres.append(tc)
        axs[2].annotate(f"({roman[k]}) {w['series'].split('_')[0]}", (tc, 1),
                        xycoords=("data", "axes fraction"), xytext=(0, -3 - 9 * row),
                        textcoords="offset points", fontsize=6.5, color="0.3",
                        ha="center", va="top")
    pl.plain_log_axis(axs[1].yaxis)
    for ax, letter in zip(axs, "abc"):
        pl.panel_label(ax, letter)
    pl.save(fig, d_fig / "S2_nested_gauges")
    log.info("Figure: %s", d_fig / "S2_nested_gauges")


if __name__ == "__main__":
    main()
