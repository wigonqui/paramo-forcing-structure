"""
Verify the environment, configuration, inputs, modelling periods and stage status of the
páramo streamflow workflow, for each site.

Checks (nothing is modified)
  1. Environment: Python version, Python packages, PyTorch when an LSTM structure is used,
     R with airGR and yaml.
  2. Configuration of every site, for the baseline and each experiment: ordered periods,
     calibration and validation without overlap, warm-up before the simulated period,
     transfer period inside the simulated window, bias-correction training outside the
     validation period, early-stopping period inside calibration, Thiessen gauges among the
     rain gauges, calibration and transfer catchments among the streamflow stations, and
     reversed experiments that swap (not change) the simulated window.
  3. Inputs: station file and the hourly series of every configured station.
  4. Data support of the periods (from the QC series of stage 01 when present, otherwise
     from the raw records): valid fraction of hourly discharge in calibration, validation
     and transfer, availability of the Thiessen gauges, balance of the effective
     calibration and validation lengths (config.yaml > period_checks), and the mean
     discharge and precipitation of each period (hydrological contrast of the
     split-sample test).
  5. Stage status per site and experiment from the run records of run_pipeline.py:
     complete / unrecorded / outdated / pending.
  6. Internal consistency (--consistency, after stage 11): GR4H metrics computed in R
     (stage 08) equal those recomputed in Python; the bootstrap reproduces the direct KGE′;
     ANOVA fractions add up to one.

Usage
  python verify.py                         all sites
  python verify.py --site chalpi
  python verify.py --no-environment --no-status
  python verify.py --consistency
Exit code 1 when errors are found.
"""
import argparse
import importlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from paramo.core import (find_series_file, in_period, load_config, read_qc,  # noqa: E402
                         read_series, site_names, thiessen_gauges)

PACKAGES = {"numpy": "numpy", "pandas": "pandas", "pyarrow": "pyarrow", "yaml": "pyyaml",
            "scipy": "scipy", "matplotlib": "matplotlib", "pyproj": "pyproj",
            "shapely": "shapely", "geopandas": "geopandas", "rasterio": "rasterio",
            "pysheds": "pysheds", "ee": "earthengine-api", "sklearn": "scikit-learn",
            "lightgbm": "lightgbm"}
KNOWN_STRUCTURES = {"GR4H", "LSTM", "GBM", "GR4H_GBM", "LSTM_raw", "LSTM_hybrid"}
HOURS_YEAR = 8766
results = []


def report(level: str, item: str, detail: str):
    results.append((level, item, detail))


def ts(day) -> pd.Timestamp:
    return pd.Timestamp(day)


# --------------------------------------------------------------------------- 1. environment
def check_environment(cfgs):
    v = sys.version_info
    report("ok" if v >= (3, 10) else "error", "python",
           f"{v.major}.{v.minor}.{v.micro} ({sys.executable})")
    missing = []
    for module, dist in PACKAGES.items():
        try:
            importlib.import_module(module)
        except Exception:  # noqa: BLE001
            missing.append(dist)
    report("error" if missing else "ok", "python packages",
           f"missing: {missing} (python install_environment.py)" if missing
           else "all importable")
    if any(s.startswith("LSTM") for c in cfgs for s in c["analysis"]["structures"]):
        try:
            import torch
            report("ok", "pytorch", f"{torch.__version__} | GPU: {torch.cuda.is_available()}")
        except Exception as err:  # noqa: BLE001
            report("error", "pytorch", f"{type(err).__name__}: {err}")
    from paramo.pipeline import rscript
    r = rscript()
    if r is None:
        report("error", "R", "Rscript not found (stage 08)")
        return
    code = ('for (p in c("airGR","yaml")) cat(p, ":", '
            'tryCatch(as.character(packageVersion(p)), error=function(e) "MISSING"), "| ")')
    out = subprocess.run([r, "-e", code], capture_output=True, text=True).stdout.strip()
    report("error" if "MISSING" in out else "ok", "R", out.rstrip(" |"))


# --------------------------------------------------------------------------- 2. configuration
def check_config(cfg, base: dict | None = None):
    tag = f"{cfg['site']}[{cfg['experiment'] or 'baseline'}]"
    per, st = cfg["periods"], cfg["stations"]
    n0 = len([r for r in results if r[0] == "error"])
    for key in ("warmup", "calibration", "validation", "transfer"):
        if ts(per[key][0]) >= ts(per[key][1]):
            report("error", tag, f"periods.{key} has start >= end")
    cal, val, tra = per["calibration"], per["validation"], per["transfer"]
    if not (ts(cal[1]) < ts(val[0]) or ts(val[1]) < ts(cal[0])):
        report("error", tag, "calibration and validation overlap")
    first = min(ts(cal[0]), ts(val[0]))
    last = max(ts(cal[1]), ts(val[1]))
    if ts(per["warmup"][1]) >= first:
        report("error", tag, "the warm-up does not precede the simulated period")
    if ts(tra[0]) < first or ts(tra[1]) > last:
        report("error", tag, f"transfer {tra} lies outside the simulated window "
               f"{first:%Y-%m-%d} – {last:%Y-%m-%d}")
    for tp in cfg["bias_correction"]["training_periods"]:
        if ts(tp[0]) <= ts(val[1]) and ts(val[0]) <= ts(tp[1]):
            report("error", tag, f"bias-correction training {tp} overlaps the validation period")
    stop = cfg["lstm"]["stop_period"]
    if ts(stop[0]) < ts(cal[0]) or ts(stop[1]) > ts(cal[1]):
        report("error", tag, f"lstm.stop_period {stop} is not inside the calibration period")
    gr = cfg["gr4h"]
    for key in ("calibration_catchment", "transfer_catchment"):
        if gr[key] not in st["streamflow"] or gr[key] not in cfg["forcing"]["catchments"]:
            report("error", tag, f"gr4h.{key} {gr[key]} is not a configured streamflow station")
    if gr["calibration_catchment"] == gr["transfer_catchment"]:
        report("error", tag, "calibration and transfer catchments are the same")
    extra = set(thiessen_gauges(cfg)) - set(st["rain_gauges"])
    if extra:
        report("error", tag, f"thiessen_gauges not among rain_gauges: {sorted(extra)}")
    if cfg["forcing"]["temperature_station"] not in st["weather"]:
        report("error", tag, "forcing.temperature_station is not a weather station")
    unknown = (set(cfg["analysis"]["structures"] + cfg["analysis"]["supplementary"])
               - KNOWN_STRUCTURES)
    if unknown:
        report("error", tag, f"unknown structures: {sorted(unknown)}")
    if base is not None:
        b = base["periods"]
        window = lambda p: (min(ts(p["calibration"][0]), ts(p["validation"][0])),
                            max(ts(p["calibration"][1]), ts(p["validation"][1])))
        if window(b) != window(per) or b["transfer"] != tra:
            report("warning", tag, "the experiment changes the simulated window or the "
                   "transfer period, not only the order of calibration and validation")
        if cfg["experiment"] == "reversed" and (b["calibration"] != val or b["validation"] != cal):
            report("error", tag, "the reversed periods are not the baseline periods swapped")
    if len([r for r in results if r[0] == "error"]) == n0:
        report("ok", tag, f"calibration {cal[0]}–{cal[1]} | validation {val[0]}–{val[1]} | "
               f"transfer {tra[0]}–{tra[1]} ({gr['transfer_catchment']})")


# --------------------------------------------------------------------------- 3. inputs
def check_inputs(cfg):
    tag = f"{cfg['site']} inputs"
    st_file = ROOT / cfg["paths"]["input"] / cfg["stations"]["file"]
    if not st_file.exists():
        report("error", tag, f"missing {st_file.relative_to(ROOT)}")
        return
    st = cfg["stations"]
    required = ([(c, "discharge") for c in st["streamflow"]]
                + [(c, "precipitation") for c in st["rain_gauges"]]
                + [(cfg["forcing"]["temperature_station"], "temperature")])
    missing = [f"{c}:{v}" for c, v in required if find_series_file(cfg, c, v) is None]
    no_weather = [c for c in st["weather"] if not any(
        find_series_file(cfg, c, v) for v in ("temperature", "radiation"))]
    stage_files = [c for c in st["streamflow"] if find_series_file(cfg, c, "stage")]
    if missing or no_weather:
        report("error", tag, f"missing series {missing + no_weather}")
    else:
        report("ok", tag, f"{len(required)} required series present; stage records: "
               f"{stage_files or 'none (rating-curve check skipped)'}")


# --------------------------------------------------------------------------- 4. periods
def valid_series(cfg, code: str, variable: str) -> pd.Series | None:
    """Hourly values that pass QC (stage 01) or, before stage 01, the completeness rule."""
    qc = read_qc(cfg, code, variable)
    if qc is not None:
        return qc["value_qc"]
    f = find_series_file(cfg, code, variable)
    if f is None:
        return None
    df = read_series(f)
    ok = df["completeness"].fillna(0) >= cfg["quality_control"]["completeness_min"]
    return df["value"].where(ok)


def coverage(s: pd.Series | None, period) -> tuple[float, float, int]:
    """(valid fraction, mean of valid values, valid hours) of a series within a period."""
    hours = pd.date_range(ts(period[0]), ts(period[1]) + pd.Timedelta(hours=23), freq="h")
    if s is None:
        return 0.0, np.nan, 0
    v = s.reindex(hours)
    n = int(v.notna().sum())
    return n / len(hours), float(v.mean()), n


def check_periods(cfg_base, cfgs_exp) -> pd.DataFrame:
    pc = cfg_base["period_checks"]
    gr, site = cfg_base["gr4h"], cfg_base["site"]
    q = {c: valid_series(cfg_base, c, "discharge")
         for c in (gr["calibration_catchment"], gr["transfer_catchment"])}
    gauges = {g: valid_series(cfg_base, g, "precipitation") for g in thiessen_gauges(cfg_base)}
    rows = []
    for cfg in [cfg_base] + cfgs_exp:
        tag = f"{site}[{cfg['experiment'] or 'baseline'}] periods"
        per = cfg["periods"]
        eff = {}
        for pname, catchment in (("calibration", gr["calibration_catchment"]),
                                 ("validation", gr["calibration_catchment"]),
                                 ("transfer", gr["transfer_catchment"])):
            frac, mean_q, n = coverage(q[catchment], per[pname])
            g = [coverage(gauges[k], per[pname]) for k in gauges]
            g_frac = float(np.mean([x[0] for x in g])) if g else 0.0
            p_mean = np.nanmean([x[1] for x in g]) * HOURS_YEAR if g else np.nan
            length = (ts(per[pname][1]) - ts(per[pname][0])).days / 365.25
            eff[pname] = n / HOURS_YEAR
            rows.append({"site": site, "run": cfg["experiment"] or "baseline", "period": pname,
                         "catchment": catchment, "dates": f"{per[pname][0]} – {per[pname][1]}",
                         "years": round(length, 1), "Q_valid": round(frac, 2),
                         "effective_years": round(n / HOURS_YEAR, 1),
                         "Q_mean_m3s": round(mean_q, 3), "gauges_valid": round(g_frac, 2),
                         "P_mean_mm_yr": round(p_mean)})
            if frac < pc["min_discharge_fraction"]:
                report("warning", tag, f"{pname} ({catchment}): only {frac:.0%} of hourly "
                       f"discharge is valid (minimum {pc['min_discharge_fraction']:.0%})")
            if g_frac < pc["min_gauge_fraction"]:
                report("warning", tag, f"{pname}: Thiessen gauges available {g_frac:.0%} of the "
                       f"time (minimum {pc['min_gauge_fraction']:.0%})")
        ratio = max(eff["calibration"], eff["validation"]) / max(
            min(eff["calibration"], eff["validation"]), 1e-9)
        level = "ok" if ratio <= pc["max_length_ratio"] else "warning"
        report(level, tag, f"effective years: calibration {eff['calibration']:.1f}, "
               f"validation {eff['validation']:.1f} (ratio {ratio:.2f}), transfer "
               f"{eff['transfer']:.1f}")
        # Warm-up: missing gauges are filled by normal ratio (stage 07), so what matters is
        # that at least one Thiessen gauge reports each hour
        p = per["warmup"]
        hours = pd.date_range(ts(p[0]), ts(p[1]) + pd.Timedelta(hours=23), freq="h")
        avail = pd.DataFrame({k: s.reindex(hours) for k, s in gauges.items()
                              if s is not None}).notna()
        any_gauge = float(avail.any(axis=1).mean()) if avail.size else 0.0
        level = "ok" if any_gauge >= pc["min_gauge_fraction"] else "warning"
        report(level, tag, f"warm-up {p[0]} – {p[1]}: hours with at least one Thiessen gauge "
               f"{any_gauge:.0%} (mean gauge availability {avail.to_numpy().mean():.0%})")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 5. status
def stage_status(site: str) -> pd.DataFrame:
    from paramo.pipeline import STAGES, stage_status as status
    rows = []
    base = load_config(site, "", prepare=False)
    base_times = {}
    for name in [None] + list(base.get("experiments", {})):
        cfg = load_config(site, name or "", prepare=False)
        upstream = 0.0
        for s in STAGES:
            if s.cross_site or (name and not s.experiment):
                if name and s.id < "06":
                    upstream = max(upstream, base_times.get(s.id, 0.0))
                continue
            st, detail, t = status(s, cfg, site, name, upstream)
            if not name:
                base_times[s.id] = t
            if st != "pending" and not s.external:
                upstream = max(upstream, t)
            rows.append({"site": site, "run": name or "baseline", "stage": s.id, "status": st,
                         "description": s.description, "detail": detail})
    return pd.DataFrame(rows)


def synthesis_status(sites: list[str]) -> pd.DataFrame:
    from paramo.pipeline import read_record, stage, stage_status as status
    s14 = stage("14")
    upstream = max([(read_record(stage(i), s, None) or {}).get("time", 0.0)
                    for s in sites for i in ("11", "13")] or [0.0])
    st, detail, _ = status(s14, load_config(sites[0], "", prepare=False), None, None, upstream)
    return pd.DataFrame([{"site": "all", "run": "baseline", "stage": "14", "status": st,
                          "description": s14.description, "detail": detail}])


# --------------------------------------------------------------------------- 6. consistency
def consistency(cfg):
    from paramo.core import path, read_simulation, simulation_path
    from paramo.metrics import evaluate
    from paramo.stats import BlockBootstrap
    tab, site = path(cfg, "tables"), cfg["site"]
    f08 = tab / "08_metrics.csv"
    if f08.exists():
        m08 = pd.read_csv(f08)
        cc = cfg["gr4h"]["calibration_catchment"]
        ref = cfg["analysis"]["reference_forcing"]
        sim = simulation_path(cfg, "GR4H", ref, cc)
        if sim.exists():
            df = read_simulation(sim)
            py = evaluate(df, cfg["periods"]["validation"])[0]["kge"]
            r = m08[(m08.scenario == ref) & (m08.catchment == cc) &
                    (m08.period == "validation") & (m08.scale == "hourly")].kge.iat[0]
            report("ok" if abs(py - r) < 1e-3 else "error", f"{site} R vs Python KGE′",
                   f"GR4H validation: R {r:.4f} | Python {py:.4f}")
            o, qs = df["Qobs"].to_numpy(), df["Qsim"].to_numpy()
            sel = (~df["excluded"].to_numpy()) & np.isfinite(o) & \
                in_period(df.index, cfg["periods"]["validation"])
            n = int(sel.sum())
            single = BlockBootstrap(n, n, 1, 0).score(o[sel], qs[sel], np.ones(n, bool))[0]
            report("ok" if abs(single - py) < 1e-6 else "error", f"{site} bootstrap identity",
                   f"single full block {single:.6f} vs direct {py:.6f}")
    fa = tab / "11_skill_anova.csv"
    if fa.exists():
        a = pd.read_csv(fa)
        worst = a.groupby(["catchment", "period"]).fraction.sum().sub(1).abs().max()
        report("ok" if worst < 1e-3 else "error", f"{site} ANOVA fractions",
               f"max |sum - 1| = {worst:.2e}")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="all", choices=site_names() + ["all"])
    ap.add_argument("--no-environment", action="store_true")
    ap.add_argument("--no-status", action="store_true")
    ap.add_argument("--no-periods", action="store_true", help="skip the data-support check")
    ap.add_argument("--consistency", action="store_true")
    a = ap.parse_args()
    sites = site_names() if a.site == "all" else [a.site]
    bases = {s: load_config(s, "", prepare=False) for s in sites}
    if not a.no_environment:
        check_environment(list(bases.values()))
    tables = []
    for site, base in bases.items():
        check_config(base)
        exps = [load_config(site, e, prepare=False) for e in base.get("experiments", {})]
        for cfg in exps:
            check_config(cfg, base)
        check_inputs(base)
        if not a.no_periods and not any(r[0] == "error" and r[1].startswith(f"{site} inputs")
                                        for r in results):
            tables.append(check_periods(base, exps))
        if a.consistency:
            consistency(base)

    print("\n=== Verification ===")
    for level, item, detail in results:
        print(f"  [{level:<7}] {item:<32} {detail}")
    if tables:
        print("\n=== Data support of the modelling periods ===")
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(pd.concat(tables).to_string(index=False))
    if not a.no_status:
        st = pd.concat([stage_status(s) for s in sites] +
                       ([synthesis_status(sites)] if a.site == "all" else []))
        print("\n=== Stage status ===")
        print(st.drop(columns="detail").to_string(index=False))
        todo = st[st.status.isin(["pending", "outdated"])]
        if len(todo):
            first = todo.groupby(["site", "run"]).stage.min().reset_index()
            print("\nResume points:")
            for r in first.itertuples():
                exp = "" if r.run == "baseline" else f" --experiment {r.run}"
                site = "" if r.site == "all" else f"--site {r.site} "
                print(f"  python run_pipeline.py {site}--from {r.stage}{exp}")
        if (st.status == "unrecorded").any():
            print("\n'unrecorded': outputs exist but were produced outside run_pipeline.py; "
                  "their validity is tracked from the next orchestrated run.")
    n_err = sum(1 for r in results if r[0] == "error")
    n_warn = sum(1 for r in results if r[0] == "warning")
    print(f"\n{n_err} error(s), {n_warn} warning(s).")
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
