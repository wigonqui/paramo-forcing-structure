"""
Stage 01 - Hourly quality control

Purpose
    Assign a quality flag to every hourly value (0 ok, 1 suspect, 2 error, 3 missing) and a
    reason, without deleting data. 'value_qc' keeps ok and suspect values; errors and gaps are
    set to NaN so that every later decision remains traceable.

Methods
    Completeness of the sub-hourly record; physical range checks; flat-line detection;
    isolated spikes in discharge and stage (robust z-score of log values combined with a jump
    against both neighbours, so that genuine floods, which rise and recede over several
    hours, are not flagged). Sensor faults documented in the configuration
    (quality_control.manual_exclusions: series, start, end and reason) are flagged as errors,
    so every manual decision is explicit and reproducible.

Tools
    pandas, numpy, matplotlib.

Inputs
    data/<site>_input/<code>-*.csv, data/interim/<site>/stations.gpkg (stage 00)

Outputs
    data/interim/<site>/<code>_<variable>.parquet   value, value_qc, flag, reason, completeness
    outputs/<site>/tables/01_flag_summary.csv, 01_annual_completeness.csv,
                          01_top_suspect_values.csv
    outputs/<site>/figures/01_annual_completeness.(png|pdf)

Next stage
    02 - Rating curve, homogeneity and gauge diagnostics.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from paramo import plotting as pl
from paramo.core import (FLAG_ERROR, FLAG_MISSING, FLAG_NAMES, FLAG_OK, FLAG_SUSPECT,
                         find_series_file, get_logger, load_config, path, read_series,
                         read_stations)

log = get_logger("01_qc")
VARIABLE_LABEL = {"precipitation": "P", "discharge": "Q", "stage": "Stage",
                  "temperature": "T", "radiation": "Rs"}


def flat_run(s: pd.Series, n: int, positive_only: bool = False) -> pd.Series:
    """True where a value repeats unchanged for at least n consecutive hours."""
    groups = (s != s.shift()).cumsum()
    length = s.groupby(groups).transform("size")
    mask = (length >= n) & s.notna()
    return mask & (s > 0) if positive_only else mask


def robust_z(s: pd.Series, window: int) -> pd.Series:
    med = s.rolling(window, center=True, min_periods=window // 3).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=window // 3).median()
    return (s - med) / (1.4826 * mad.replace(0, np.nan))


class Flags:
    """Keeps the highest severity per record and concatenates the reasons."""

    def __init__(self, index):
        self.flag = pd.Series(FLAG_OK, index=index, dtype="int8")
        self.reason = pd.Series("", index=index, dtype="object")

    def mark(self, mask: pd.Series, level: int, text: str):
        m = mask.fillna(False).astype(bool)
        if m.any():
            self.flag[m] = np.maximum(self.flag[m], level).astype("int8")
            self.reason[m] = self.reason[m].where(self.reason[m] == "",
                                                  self.reason[m] + ";") + text


def check(df: pd.DataFrame, variable: str, cfg: dict, faults=()) -> pd.DataFrame:
    q = cfg["quality_control"]
    v = df["value"].astype(float)
    fl = Flags(v.index)
    fl.mark(v.isna(), FLAG_MISSING, "missing")
    if "completeness" in df:
        fl.mark(v.notna() & (df["completeness"] < q["completeness_min"]), FLAG_ERROR,
                "low_completeness")

    if variable == "precipitation":
        p = q["precipitation"]
        fl.mark(v < 0, FLAG_ERROR, "negative")
        fl.mark(v > p["max_hourly"], FLAG_SUSPECT, "extreme")
        fl.mark(flat_run(v, p["flat_hours"], True) & (v >= p["flat_min_mm"]), FLAG_SUSPECT,
                "flat")
    elif variable == "temperature":
        p = q["temperature"]
        fl.mark((v < p["range"][0]) | (v > p["range"][1]), FLAG_ERROR, "out_of_range")
        fl.mark(v.diff().abs() > p["max_step"], FLAG_SUSPECT, "step")
        fl.mark(flat_run(v, p["flat_hours"]), FLAG_SUSPECT, "flat")
    elif variable == "radiation":
        p = q["radiation"]
        fl.mark(v < p["range"][0] - 5, FLAG_ERROR, "negative")
        fl.mark(v > p["range"][1], FLAG_ERROR, "out_of_range")
        fl.mark(flat_run(v.round(1), p["flat_hours"], True), FLAG_SUSPECT, "flat")
    elif variable in ("discharge", "stage"):
        p = q[variable]
        fl.mark(v <= 0, FLAG_ERROR, "non_positive")
        lv = np.log(v.where(v > 0))
        z = robust_z(lv, p["window_hours"])
        jump = (lv - 0.5 * (lv.shift(1) + lv.shift(-1))).abs()
        fl.mark((z.abs() > p["z_max"]) & (jump > np.log(p["spike_factor"])), FLAG_SUSPECT,
                "isolated_spike")
        fl.mark(flat_run(v, p["flat_hours"]), FLAG_SUSPECT, "flat")

    for fault in faults:
        span = (v.index >= pd.Timestamp(fault["start"])) & (v.index <= pd.Timestamp(fault["end"]))
        fl.mark(pd.Series(span, index=v.index) & v.notna(), FLAG_ERROR, "documented_fault")

    out = pd.DataFrame({"value": v, "flag": fl.flag, "reason": fl.reason})
    out["value_qc"] = v.where(fl.flag <= FLAG_SUSPECT)
    if variable == "radiation":
        out["value_qc"] = out["value_qc"].clip(lower=0)
    if "completeness" in df:
        out["completeness"] = df["completeness"]
    return out


def completeness_figure(comp: pd.DataFrame, stem: Path):
    fig, ax = pl.new_figure("double", 22 + 6.5 * comp.shape[1])
    data = comp.T.to_numpy()
    im = ax.imshow(data, aspect="auto", cmap="cividis", vmin=0, vmax=100)
    ax.set_yticks(range(comp.shape[1]), comp.columns)
    years = comp.index.to_numpy()
    step = 2 if len(years) > 16 else 1
    ax.set_xticks(range(0, len(years), step), years[::step], rotation=90)
    for (i, j), val in np.ndenumerate(data):
        if np.isfinite(val):
            ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=5,
                    color="white" if val < 55 else "black")
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.01)
    cb.set_label("Valid hours after QC (%)")
    pl.save(fig, stem)


def main():
    cfg = load_config()
    st = read_stations(cfg)
    d_int, d_tab, d_fig = path(cfg, "interim"), path(cfg, "tables"), path(cfg, "figures")
    summary, comp, suspects = [], {}, []
    faults = cfg["quality_control"].get("manual_exclusions") or []
    for code in st.code:
        for variable in cfg["variables"]:
            f = find_series_file(cfg, code, variable)
            if f is None:
                continue
            own = [x for x in faults if x["series"] == f"{code}_{variable}"]
            for x in own:
                log.info("%s_%s: documented fault %s to %s (%s)", code, variable, x["start"],
                         x["end"], x["reason"])
            out = check(read_series(f), variable, cfg, own)
            out.to_parquet(d_int / f"{code}_{variable}.parquet")
            counts = out["flag"].map(FLAG_NAMES).value_counts()
            reasons = out.loc[out.reason != "", "reason"].str.split(";").explode().value_counts()
            summary.append({"code": code, "variable": variable, "hours": len(out),
                            **{k: int(counts.get(k, 0)) for k in FLAG_NAMES.values()},
                            "reasons": "; ".join(f"{k}={n}" for k, n in reasons.items()
                                                 if k != "missing")})
            label = f"{code} {VARIABLE_LABEL[variable]}"
            comp[label] = (out["value_qc"].notna().groupby(out.index.year).mean() * 100).round(1)
            s = out[out.flag == FLAG_SUSPECT]
            if len(s):
                top = s.reindex(s["value"].abs().sort_values(ascending=False).index).head(10)
                suspects.append(top.assign(code=code, variable=variable).reset_index())
            log.info("%s %-13s ok=%d suspect=%d error=%d missing=%d", code, variable,
                     counts.get("ok", 0), counts.get("suspect", 0), counts.get("error", 0),
                     counts.get("missing", 0))

    pd.DataFrame(summary).to_csv(d_tab / "01_flag_summary.csv", index=False)
    comp = pd.DataFrame(comp).sort_index()
    comp.index.name = "year"
    comp.to_csv(d_tab / "01_annual_completeness.csv")
    if suspects:
        pd.concat(suspects)[["code", "variable", "time", "value", "reason"]].to_csv(
            d_tab / "01_top_suspect_values.csv", index=False)
    completeness_figure(comp, d_fig / "01_annual_completeness")
    log.info("Review %s/01_top_suspect_values.csv before continuing.", cfg["paths"]["tables"])


if __name__ == "__main__":
    main()
