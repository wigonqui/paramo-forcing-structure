"""
Stage 12 - Sensitivity of the conclusions

Purpose
    Test whether the main conclusion of the site (forcing explains more of the skill variance
    than model structure) and the ranking of structures depend on analysis choices. Nothing
    is recalibrated or retrained; only the evaluation changes.

Methods
    A. Metric: forcing × structure ANOVA with hourly KGE', daily KGE', NSE, NSE of log flows
       (low flows) and KGE' of square-root flows, each with bootstrap intervals; agreement of
       the structure ranking across metrics (Kendall's W).
    B. Bootstrap block length (15, 30 and 60 days): interval width and ANOVA stability.
    C. Observation error of discharge (single rating curve, no gaugings):
       Q_true = Q_obs · (1 + δ) · exp(η_t − σ_t²/2), with δ a systematic error per realization
       and η_t a slow AR(1) error whose standard deviation depends on the flow regime.
       Distribution of KGE', of the ANOVA fractions and the probability that each structure
       is the best one.
    D. Post-processing: the forcing × structure ANOVA of hourly KGE' with the structures as
       reported (variance adjustment on the learning models only), without the adjustment on
       any structure and with it on all four, GR4H included (sensitivity.
       postprocessing_variants), on the same hours and with bootstrap intervals.
    E. Forcing set: the forcing × structure ANOVA of hourly KGE' on subsets of the forcings
       (sensitivity.forcing_subsets), e.g. without the raw satellite products, whose large
       volume biases could by themselves inflate the forcing fraction; same hours and
       bootstrap replicates as the full design.

Tools
    pandas, numpy, SciPy, matplotlib; paramo.stats, paramo.evaluation.

Inputs
    data/processed/<site>/simulations/* (stages 08–10, including LSTM_raw, GBM_raw,
        GR4H_GBM_raw and GR4H_adj); forcing_<catchment>.parquet (07)

Outputs
    outputs/<site>/tables/12_anova_by_metric.csv, 12_structure_ranking.csv,
        12_block_length.csv, 12_obs_error_kge.csv, 12_obs_error_anova.csv,
        12_obs_error_best.csv, 12_postprocessing_anova.csv, 12_postprocessing_skill.csv,
        12_forcing_subset_anova.csv
    outputs/<site>/figures/12_anova_by_metric, 12_obs_error_best (png|pdf)

Next stage
    13 - Reversed split-sample comparison.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy.signal import lfilter
from scipy.stats import rankdata

from paramo import plotting as pl
from paramo.core import get_logger, load_config, path, split_scenarios
from paramo.evaluation import common_sample, load_simulations
from paramo.stats import BlockBootstrap, anova_fractions, anova_replicates, interval

log = get_logger("12_sensitivity")


def metric_series(name, q, valid):
    """(obs, {key: sim}, valid, score type, block unit) after the metric's transformation."""
    keys = [c for c in q.columns if c != "Qobs"]
    if name == "kge_daily":
        z = q[valid]
        n = z.groupby(z.index.normalize()).size()
        d = z.groupby(z.index.normalize()).mean()[n >= 20]
        d = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="D"))
        return d["Qobs"].to_numpy(), {k: d[k].to_numpy() for k in keys}, \
            d["Qobs"].notna().to_numpy(), "kge", "days"
    o = q["Qobs"].to_numpy()
    if name in ("kge", "nse"):
        return o, {k: q[k].to_numpy() for k in keys}, valid, name, "hours"
    if name == "kge_sqrt":
        f = lambda x: np.sqrt(np.clip(x, 0, None))
        return f(o), {k: f(q[k].to_numpy()) for k in keys}, valid, "kge", "hours"
    eps = np.nanmean(o[valid]) / 100
    f = lambda x: np.log(np.clip(x, 0, None) + eps)
    return f(o), {k: f(q[k].to_numpy()) for k in keys}, valid, "nse", "hours"


def stack(values: dict, structures, scenarios) -> np.ndarray:
    return np.stack([np.stack([values[(s, sc)] for s in structures], -1)
                     for sc in scenarios], -2)


def observation_realizations(o, valid, sc, rng) -> np.ndarray:
    n = len(o)
    q10, q90 = np.nanquantile(o[valid], [0.1, 0.9])
    sd = np.where(o < q10, sc["obs_sd_low"], np.where(o > q90, sc["obs_sd_high"],
                                                      sc["obs_sd_mid"]))
    rho = sc["obs_rho"]
    z = rng.standard_normal((sc["obs_realizations"], n))
    z[:, 1:] *= np.sqrt(1 - rho ** 2)                      # stationary AR(1), unit variance
    eta = lfilter([1.0], [1.0, -rho], z, axis=1)
    delta = rng.uniform(-sc["obs_systematic"], sc["obs_systematic"],
                        size=(sc["obs_realizations"], 1))
    return o[None, :] * (1 + delta) * np.exp(eta * sd[None, :] - 0.5 * sd[None, :] ** 2)


def kge_matrix(obs: np.ndarray, sims: np.ndarray) -> np.ndarray:
    """obs (R, n) perturbed observations; sims (n, K) -> KGE' (R, K)."""
    n = obs.shape[1]
    mo, ms = obs.mean(1, keepdims=True), sims.mean(0, keepdims=True)
    so, ss = obs.std(1, ddof=1, keepdims=True), sims.std(0, ddof=1, keepdims=True)
    r = ((obs - mo) @ (sims - ms) / (n - 1)) / (so * ss)
    return 1 - np.sqrt((r - 1) ** 2 + (ms / mo - 1) ** 2 + ((ss / ms) / (so / mo) - 1) ** 2)


def postprocessing_anova(cfg, catchment, pname, period, scenarios, rows_a, rows_s):
    """ANOVA of hourly KGE' for each post-processing variant, on hours shared by all."""
    an, sc = cfg["analysis"], cfg["sensitivity"]
    variants = sc.get("postprocessing_variants") or {}
    sims = {v: load_simulations(cfg, m, scenarios, catchment) for v, m in variants.items()}
    complete = {v: m for v, m in variants.items()
                if all((s, x) in sims[v] for s in m for x in scenarios)}
    for v in sorted(set(variants) - set(complete)):
        log.warning("%s: post-processing variant '%s' incomplete (run stage 10); skipped",
                    catchment, v)
    if not complete:
        return
    pool = {k: s for v in complete for k, s in sims[v].items()}
    valid, q = common_sample(pool, period)
    o = q["Qobs"].to_numpy()
    bs = BlockBootstrap(len(o), an["block_h"], sc["bootstrap_replicates"], an["seed"])
    point = BlockBootstrap(len(o), len(o), 1, 0)
    boot = {k: bs.score(o, q[k].to_numpy(), valid) for k in pool}
    kge0 = {k: point.score(o, q[k].to_numpy(), valid)[0] for k in pool}
    summary = {}
    for v, members in complete.items():
        m0 = np.array([[kge0[(s, x)] for s in members] for x in scenarios])
        reps = anova_replicates(stack(boot, members, scenarios))
        summary[v] = anova_fractions(m0)
        for comp, f in summary[v].items():
            lo, hi = interval(reps[comp])
            rows_a.append({"catchment": catchment, "period": pname, "variant": v,
                           "structures": "+".join(members), "component": comp,
                           "fraction": f, "ci_low": lo, "ci_high": hi})
        for j, s in enumerate(members):
            rows_s.append({"catchment": catchment, "period": pname, "variant": v,
                           "structure": s, "kge_mean": m0[:, j].mean(),
                           "kge_gauge": kge0.get((s, "gauge"), np.nan)})
    log.info("%s %s, post-processing variants:\n%s", catchment, pname,
             pd.DataFrame(summary).T.round(2).to_string())


def forcing_subset_anova(cfg, catchment, pname, q, valid, structures, scenarios, rows):
    """ANOVA of hourly KGE' on each subset of forcings, on the hours of the full design."""
    an, sc = cfg["analysis"], cfg["sensitivity"]
    subsets = sc.get("forcing_subsets") or {}
    if not subsets:
        return
    o = q["Qobs"].to_numpy()
    bs = BlockBootstrap(len(o), an["block_h"], sc["bootstrap_replicates"], an["seed"])
    point = BlockBootstrap(len(o), len(o), 1, 0)
    keys = [(s, x) for x in scenarios for s in structures]
    boot = {k: bs.score(o, q[k].to_numpy(), valid) for k in keys}
    kge0 = {k: point.score(o, q[k].to_numpy(), valid)[0] for k in keys}
    summary = {}
    for name, members in subsets.items():
        missing = [x for x in members if x not in scenarios]
        if missing or len(members) < 2:
            log.warning("%s: forcing subset '%s' skipped (missing %s)", catchment, name, missing)
            continue
        m0 = np.array([[kge0[(s, x)] for s in structures] for x in members])
        reps = anova_replicates(stack(boot, structures, members))
        summary[name] = anova_fractions(m0)
        for comp, f in summary[name].items():
            lo, hi = interval(reps[comp])
            rows.append({"catchment": catchment, "period": pname, "subset": name,
                         "scenarios": "+".join(members), "component": comp,
                         "fraction": f, "ci_low": lo, "ci_high": hi})
    log.info("%s %s, forcing subsets:\n%s", catchment, pname,
             pd.DataFrame(summary).T.round(2).to_string())


def anova_metric_figure(table, metrics, stem):
    groups = list(dict.fromkeys(zip(table.catchment, table.period)))
    fig, axs = pl.new_figure("double", 72, ncols=len(groups), sharey=True)
    comps = ["forcing", "structure", "interaction"]
    for k, (ax, (cu, per)) in enumerate(zip(np.atleast_1d(axs), groups)):
        g = table[(table.catchment == cu) & (table.period == per)]
        t = g.pivot_table(index="metric", columns="component", values="fraction").reindex(metrics)
        bottom = np.zeros(len(t))
        for comp in comps:
            ax.bar(range(len(t)), t[comp], bottom=bottom, color=pl.COMPONENT_COLOR[comp],
                   edgecolor="black", lw=0.4, width=0.65, label=pl.COMPONENT_LABEL[comp])
            bottom += t[comp].to_numpy()
        ax.set_xticks(range(len(t)), [pl.METRIC_LABEL[m] for m in t.index], rotation=30,
                      ha="right")
        ax.set(title=f"{cu}, {per}", ylim=(0, 1))
        pl.panel_label(ax, "ab"[k])
    np.atleast_1d(axs)[0].set_ylabel("Fraction of skill variance")
    handles, labels = np.atleast_1d(axs)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3)
    pl.save(fig, stem)


def best_structure_figure(best, stem):
    v = best[best.period == "validation"].pivot_table(index="scenario", columns="structure",
                                                      values="p_best")
    rows = pl.scenario_sort(v.index)
    cols = pl.structure_sort(v.columns)
    v = v.reindex(index=rows, columns=cols)
    fig, ax = pl.new_figure("onehalf", 95)
    im = ax.imshow(v.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for (i, j), x in np.ndenumerate(v.to_numpy()):
        ax.text(j, i, f"{x:.2f}", ha="center", va="center", fontsize=6.5,
                color="white" if x > 0.6 else "black")
    ax.set_xticks(range(len(cols)), [pl.STRUCTURE_LABEL[c] for c in cols])
    ax.set_yticks(range(len(rows)), [pl.scenario_label(r) for r in rows])
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    ax.spines[["left", "bottom"]].set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02).set_label(
        "Probability of being the best structure")
    pl.save(fig, stem)


def main():
    cfg = load_config()
    an, sc, per, gr = cfg["analysis"], cfg["sensitivity"], cfg["periods"], cfg["gr4h"]
    d_tab, d_fig = path(cfg, "tables"), path(cfg, "figures")
    cc, ct = gr["calibration_catchment"], gr["transfer_catchment"]
    forcing = pd.read_parquet(path(cfg, "processed") / f"forcing_{cc}.parquet")
    scenarios = split_scenarios(forcing.columns)[0]
    rng = np.random.default_rng(sc["seed"])
    rows_a, rows_r, rows_b, rows_ok, rows_oa, rows_best = [], [], [], [], [], []
    rows_pa, rows_ps, rows_fs = [], [], []

    for catchment, pname, rng_ in ((cc, "validation", per["validation"]),
                                   (ct, "transfer", per["transfer"])):
        sims = load_simulations(cfg, an["structures"], scenarios, catchment)
        structures = [s for s in an["structures"] if all((s, x) in sims for x in scenarios)]
        if len(structures) < 2:
            log.warning("%s: fewer than two complete structures; skipped", catchment)
            continue
        sims = {k: v for k, v in sims.items() if k[0] in structures}
        valid, q = common_sample(sims, rng_)
        log.info("%s %s %s: structures %s", cfg["site"], catchment, pname, structures)

        ranking = {}
        for name in sc["metrics"]:
            o, sim, ok, kind, unit = metric_series(name, q, valid)
            block = an["block_h"] if unit == "hours" else max(1, an["block_h"] // 24)
            bs = BlockBootstrap(len(o), block, sc["bootstrap_replicates"], an["seed"])
            boot = {k: bs.score(o, sim[k], ok, kind) for k in sims}
            point = BlockBootstrap(len(o), len(o), 1, 0)
            m0 = stack({k: point.score(o, sim[k], ok, kind) for k in sims},
                       structures, scenarios)[0]
            reps = anova_replicates(stack(boot, structures, scenarios))
            for comp, v in anova_fractions(m0).items():
                lo, hi = interval(reps[comp])
                rows_a.append({"catchment": catchment, "period": pname, "metric": name,
                               "component": comp, "fraction": v, "ci_low": lo, "ci_high": hi})
            ranking[name] = m0.mean(0)
            for j, s in enumerate(structures):
                rows_r.append({"catchment": catchment, "period": pname, "metric": name,
                               "structure": s, "mean_over_scenarios": m0[:, j].mean()})
        r = np.array([rankdata(-v) for v in ranking.values()])
        m_, k_ = r.shape
        w = 12 * ((r.sum(0) - m_ * (k_ + 1) / 2) ** 2).sum() / (m_ ** 2 * (k_ ** 3 - k_))
        rows_r.append({"catchment": catchment, "period": pname, "metric": "kendall_W",
                       "structure": "all", "mean_over_scenarios": w})
        log.info("%s %s: Kendall's W across metrics = %.2f", catchment, pname, w)

        for block in sc["block_lengths_h"]:
            bs = BlockBootstrap(len(q), block, sc["bootstrap_replicates"], an["seed"])
            boot = {k: bs.score(q["Qobs"].to_numpy(), q[k].to_numpy(), valid) for k in sims}
            reps = anova_replicates(stack(boot, structures, scenarios))
            fusion = boot.get(("GR4H", "fusion_corr"), np.nan) - boot.get(("GR4H", "gauge"), np.nan)
            hybrid = boot.get(("GR4H_GBM", "gauge"), np.nan) - boot.get(("GR4H", "gauge"), np.nan)
            rows_b.append({"catchment": catchment, "period": pname, "block_h": block,
                           "median_ci_width": np.median([np.subtract(*interval(v)[::-1])
                                                         for v in boot.values()]),
                           "forcing_ci": "[%.2f, %.2f]" % interval(reps["forcing"]),
                           "structure_ci": "[%.2f, %.2f]" % interval(reps["structure"]),
                           "p_fusion_corr_better_than_gauge_GR4H": np.nanmean(fusion > 0),
                           "p_GR4H_GBM_better_than_GR4H_gauge": np.nanmean(hybrid > 0)})

        postprocessing_anova(cfg, catchment, pname, rng_, scenarios, rows_pa, rows_ps)
        forcing_subset_anova(cfg, catchment, pname, q, valid, structures, scenarios, rows_fs)

        keys = [(s, x) for x in scenarios for s in structures]
        obs = q["Qobs"].to_numpy()
        obs_r = observation_realizations(np.where(valid, obs, np.nanmean(obs[valid])), valid,
                                         sc, rng)[:, valid]
        kr = kge_matrix(obs_r, np.column_stack([q[k].to_numpy()[valid] for k in keys]))
        for j, (s, x) in enumerate(keys):
            rows_ok.append({"catchment": catchment, "period": pname, "model": s, "scenario": x,
                            "kge_median": np.median(kr[:, j]), "p05": np.quantile(kr[:, j], 0.05),
                            "p95": np.quantile(kr[:, j], 0.95)})
        mr = kr.reshape(len(kr), len(scenarios), len(structures))
        reps = anova_replicates(mr)
        for comp in ("forcing", "structure", "interaction"):
            rows_oa.append({"catchment": catchment, "period": pname, "component": comp,
                            "median": reps[comp].median(), "p05": reps[comp].quantile(0.05),
                            "p95": reps[comp].quantile(0.95)})
        best = mr.argmax(-1)
        for i, x in enumerate(scenarios):
            for j, s in enumerate(structures):
                rows_best.append({"catchment": catchment, "period": pname, "scenario": x,
                                  "structure": s, "p_best": np.mean(best[:, i] == j)})

    anova = pd.DataFrame(rows_a)
    anova.round(4).to_csv(d_tab / "12_anova_by_metric.csv", index=False)
    pd.DataFrame(rows_r).round(4).to_csv(d_tab / "12_structure_ranking.csv", index=False)
    pd.DataFrame(rows_b).round(4).to_csv(d_tab / "12_block_length.csv", index=False)
    pd.DataFrame(rows_ok).round(4).to_csv(d_tab / "12_obs_error_kge.csv", index=False)
    obs_anova = pd.DataFrame(rows_oa)
    obs_anova.round(4).to_csv(d_tab / "12_obs_error_anova.csv", index=False)
    pd.DataFrame(rows_pa).round(4).to_csv(d_tab / "12_postprocessing_anova.csv", index=False)
    pd.DataFrame(rows_ps).round(4).to_csv(d_tab / "12_postprocessing_skill.csv", index=False)
    pd.DataFrame(rows_fs).round(4).to_csv(d_tab / "12_forcing_subset_anova.csv", index=False)
    best = pd.DataFrame(rows_best)
    best.round(3).to_csv(d_tab / "12_obs_error_best.csv", index=False)
    if len(anova):
        anova_metric_figure(anova, sc["metrics"], d_fig / "12_anova_by_metric")
        log.info("ANOVA by metric:\n%s", anova.pivot_table(index=["catchment", "metric"],
                                                          columns="component",
                                                          values="fraction").round(3).to_string())
    if len(best):
        best_structure_figure(best, d_fig / "12_obs_error_best")
    if len(obs_anova):
        log.info("ANOVA under observation error (median [p05, p95]):\n%s",
                 obs_anova.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
