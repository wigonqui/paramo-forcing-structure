"""
Stage 10 - GBM and GR4H-GBM

Purpose
    Train the two machine-learning structures designed for short records: a purely
    data-driven gradient-boosting model (GBM) and a hybrid error model of GR4H (GR4H_GBM).

Methods
    GBM: the hydrological memory that an LSTM must learn is supplied explicitly as backward
    moving sums of precipitation (1 h to 30 days), antecedent precipitation indices with hourly
    decay 0.9/0.99/0.999 (half-lives of about 7 h, 3 days and 29 days), temperature, PET and
    calendar terms. Target: square root of discharge.
    GR4H_GBM: learns the residual sqrt(Qobs) - sqrt(Q_GR4H) with the same predictors plus the
    GR4H discharge, its recent changes and the production and routing stores. When there is
    nothing to learn it returns GR4H. The result is rescaled to the GR4H mean volume over the
    calibration period, so the error model corrects dynamics and leaves the water balance to
    the conceptual model.
    Both receive the variance adjustment of paramo.metrics (training data only). Only forcing
    and simulated variables are used, never past observed discharge, so this is simulation,
    not data-assimilating forecasting. Same protocol as stage 09: training on calibration
    minus lstm.stop_period (early stopping), gbm.seeds ensemble, identical excluded hours and
    metrics, transfer without retraining.
    Post-processing sensitivity (stage 12): the unadjusted series are kept as GBM_raw and
    GR4H_GBM_raw, and GR4H receives the same adjustment, estimated on its calibration period,
    as GR4H_adj.

Tools
    LightGBM (deterministic, multi-threaded), SciPy (recursive filters), pandas, numpy;
    paramo.gbm (predictors and training, shared with stage 11).

Inputs
    data/processed/<site>/forcing_<catchment>.parquet (07); simulations/GR4H__* (08)

Outputs
    data/processed/<site>/simulations/GBM__*, GR4H_GBM__*, GBM_raw__*, GR4H_GBM_raw__*,
        GR4H_adj__*
    outputs/<site>/tables/10_metrics.csv, 10_feature_importance.csv

Usage
    python scripts/10_gbm.py [--scenarios gauge fusion_corr] [--models GR4H_GBM]

Next stage
    11 - Factorial analysis of skill.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from paramo.core import (get_logger, in_period, load_config, path, simulation_path,
                         write_simulation)
from paramo.gbm import fit_predict, predictors
from paramo.metrics import evaluate, variance_adjustment

log = get_logger("10_gbm")


def save_and_evaluate(cfg, rows: list, name: str, scenario: str, catchment: str, x, q_sim):
    """Write one simulation and append its metrics for the periods of that catchment."""
    per, cc = cfg["periods"], cfg["gr4h"]["calibration_catchment"]
    out = pd.DataFrame({"Qobs": x["Qobs"].to_numpy(), "Qsim": q_sim,
                        "excluded": x["excluded"].to_numpy()}, index=x.index)
    write_simulation(out, simulation_path(cfg, name, scenario, catchment))
    periods = ([("calibration", per["calibration"]), ("validation", per["validation"])]
               if catchment == cc else [("transfer", per["transfer"])])
    for pname, rng_ in periods:
        for row in evaluate(out, rng_):
            rows.append({"model": name, "scenario": scenario, "catchment": catchment,
                         "period": pname, **row})


def gr4h_adjusted(cfg, rows: list, scenarios: list[str]):
    """GR4H with the variance adjustment of the learning models (GR4H_adj)."""
    cc, ct = cfg["gr4h"]["calibration_catchment"], cfg["gr4h"]["transfer_catchment"]
    for scenario in scenarios:
        xc, xt = predictors(cfg, cc, scenario, False), predictors(cfg, ct, scenario, False)
        train = in_period(xc.index, cfg["periods"]["calibration"]) & ~xc["excluded"].to_numpy()
        adjust, c = variance_adjustment(xc["Q_gr4h"].to_numpy(), xc["Qobs"].to_numpy(), train)
        for catchment, x in ((cc, xc), (ct, xt)):
            save_and_evaluate(cfg, rows, "GR4H_adj", scenario, catchment, x,
                              adjust(x["Q_gr4h"].to_numpy()))
        log.info("%-9s %-15s variance factor %.2f", "GR4H_adj", scenario, c)


def main(a):
    cfg = load_config()
    gc, gr = cfg["gbm"], cfg["gr4h"]
    cc, ct = gr["calibration_catchment"], gr["transfer_catchment"]
    forcing = pd.read_parquet(path(cfg, "processed") / f"forcing_{cc}.parquet")
    scenarios = a.scenarios or [c[3:] for c in forcing if c.startswith("P__")]
    models = a.models or gc["models"]
    log.info("Site %s | models %s | %d scenarios", cfg["site"], models, len(scenarios))
    rows, importance = [], []
    gr4h_adjusted(cfg, rows, scenarios)
    for name in models:
        hybrid = name == "GR4H_GBM"
        for scenario in scenarios:
            t0 = time.time()
            xc, xt = predictors(cfg, cc, scenario, hybrid), predictors(cfg, ct, scenario, hybrid)
            cols = [c for c in xc if c not in ("Qobs", "Q_gr4h", "excluded")]
            pc, pt, rc, rt, info = [], [], [], [], []
            for seed in range(gc["seeds"]):
                qc, qt, model, c, (q0c, q0t) = fit_predict(xc, xt, cols, hybrid, cfg, 42 + seed,
                                                           return_raw=True)
                pc.append(qc)
                pt.append(qt)
                rc.append(q0c)
                rt.append(q0t)
                info.append(f"{model.best_iteration_}/{c:.2f}")
                importance.append(pd.Series(model.booster_.feature_importance("gain"),
                                            index=cols, name=f"{name}__{scenario}__{seed}"))
            for label, sim_c, sim_t in ((name, pc, pt), (f"{name}_raw", rc, rt)):
                save_and_evaluate(cfg, rows, label, scenario, cc, xc, np.vstack(sim_c).mean(0))
                save_and_evaluate(cfg, rows, label, scenario, ct, xt, np.vstack(sim_t).mean(0))
            k = {r["period"]: r["kge"] for r in rows if r["model"] == name
                 and r["scenario"] == scenario and r["scale"] == "hourly"}
            log.info("%-9s %-15s KGE' hourly: calibration=%.3f | validation=%.3f | "
                     "%s=%.3f | trees/variance factor %s (%.0f s)", name, scenario,
                     k["calibration"], k["validation"], ct, k["transfer"], info,
                     time.time() - t0)
    pd.DataFrame(rows).round(4).to_csv(path(cfg, "tables") / "10_metrics.csv", index=False)
    pd.concat(importance, axis=1).round(1).to_csv(path(cfg, "tables") /
                                                  "10_feature_importance.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="*")
    ap.add_argument("--models", nargs="*", choices=["GBM", "GR4H_GBM"])
    main(ap.parse_args())
