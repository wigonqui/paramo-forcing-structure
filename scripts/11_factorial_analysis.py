"""
Stage 11 - Factorial analysis of skill

Purpose
    Answer the central question of the study at the selected site, whether forcing or model
    structure controls hourly simulation skill out of sample, with uncertainty and
    significance.

Methods
    1. Common sample: every simulation is evaluated on the same hours (intersection of valid
       hours across structures and scenarios) with the same metrics.
    2. Moving-block bootstrap (analysis.block_h) with identical blocks for all simulations,
       giving confidence intervals of KGE' and of paired differences: against GR4H with the
       same forcing, against the gauges with the same structure, and ML-corrected against the
       QM reference of the same product.
    3. Diebold–Mariano test on squared errors of daily mean discharge with Newey–West
       long-run variance and Holm correction within each family of comparisons.
    4. Two-way ANOVA without replication of KGE' over the forcing × structure matrix:
       fractions explained by forcing, structure and interaction, with bootstrap intervals.
    5. SHAP contributions (LightGBM TreeSHAP) of the GR4H_GBM error model.
    6. Study-design schematic of the site.

Tools
    pandas, numpy, SciPy, LightGBM, matplotlib; paramo.stats, paramo.metrics, paramo.gbm,
    paramo.evaluation.

Inputs
    data/processed/<site>/simulations/* (stages 08, 09, 10); forcing_<catchment>.parquet (07)

Outputs
    outputs/<site>/tables/11_metrics.csv, 11_bootstrap_ci.csv, 11_differences.csv,
                          11_diebold_mariano.csv, 11_skill_anova.csv, 11_shap.csv
    outputs/<site>/figures/11_skill_matrix, 11_differences_gr4h, 11_ml_vs_qm,
                           11_skill_anova, 11_shap, 11_study_design (png|pdf)

Next stage
    12 - Sensitivity of the conclusions.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from paramo import plotting as pl
from paramo.core import get_logger, in_period, load_config, path, simulation_path, split_scenarios
from paramo.evaluation import common_sample, daily_table, load_simulations
from paramo.gbm import fit_predict, predictors
from paramo.metrics import streamflow_metrics
from paramo.stats import (BlockBootstrap, anova_fractions, anova_replicates, diebold_mariano,
                          holm, interval)

log = get_logger("11_factorial")


# --------------------------------------------------------------------------- comparisons
def references(structure, scenario, ref_s, ref_f):
    if scenario.endswith("_qm"):
        return [("vs_GR4H", (ref_s, scenario))]
    refs = [("vs_GR4H", (ref_s, scenario)), ("vs_gauge", (structure, ref_f))]
    if scenario.endswith("_corr"):
        refs.append(("corr_vs_qm", (structure, scenario[:-5] + "_qm")))
    return refs


# --------------------------------------------------------------------------- figures
def skill_matrix_figure(boot, structures, scenarios, stem):
    v = boot[(boot.period == "validation")]
    med = v.pivot_table(index="scenario", columns="model", values="kge_median")
    lo = v.pivot_table(index="scenario", columns="model", values="ci_low")
    hi = v.pivot_table(index="scenario", columns="model", values="ci_high")
    rows, cols = pl.scenario_sort([s for s in scenarios if s in med.index]), \
        [s for s in structures if s in med.columns]
    med, lo, hi = (t.reindex(index=rows, columns=cols) for t in (med, lo, hi))
    fig, ax = pl.new_figure("onehalf", 105)
    im = ax.imshow(med.to_numpy(), cmap="viridis", vmin=0, vmax=0.8, aspect="auto")
    for (i, j), val in np.ndenumerate(med.to_numpy()):
        if np.isfinite(val):
            ax.text(j, i, f"{val:.2f}\n[{lo.iat[i, j]:.2f}, {hi.iat[i, j]:.2f}]", ha="center",
                    va="center", fontsize=6, color="black" if val > 0.45 else "white")
    ax.set_xticks(range(len(cols)), [pl.STRUCTURE_LABEL[c] for c in cols])
    ax.set_yticks(range(len(rows)), [pl.scenario_label(r) for r in rows])
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    ax.grid(which="minor", color="white", lw=1.2)
    ax.xaxis.tick_top()
    ax.tick_params(which="both", length=0)
    ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02).set_label("KGE′ hourly (bootstrap median)")
    pl.save(fig, stem)


def forest_figure(dif, kind, models, scenarios, stem, xlabel, title_fmt):
    from matplotlib.lines import Line2D
    d = dif[(dif.period == "validation") & (dif.kind == kind) & dif.model.isin(models)]
    models = [m for m in models if m in set(d.model)]
    if not models:
        return
    rows = pl.scenario_sort([s for s in scenarios if s in set(d.scenario)])
    fig, axs = pl.new_figure("double", 30 + 7 * len(rows), ncols=len(models), sharey=True,
                             sharex=True)
    for k, (ax, m) in enumerate(zip(np.atleast_1d(axs), models)):
        g = d[d.model == m].set_index("scenario").reindex(rows)
        y = np.arange(len(rows))[::-1]
        ax.axvline(0, color="black", lw=0.6)
        sig = (g.ci_low > 0) | (g.ci_high < 0)
        ax.errorbar(g.delta_median, y, xerr=[g.delta_median - g.ci_low, g.ci_high - g.delta_median],
                    fmt="none", ecolor=pl.STRUCTURE_COLOR[m], elinewidth=0.9, capsize=1.5)
        ax.scatter(g.delta_median[sig], y[sig.to_numpy()], s=14, color=pl.STRUCTURE_COLOR[m],
                   zorder=3)
        ax.scatter(g.delta_median[~sig], y[~sig.to_numpy()], s=14, facecolor="white",
                   edgecolor=pl.STRUCTURE_COLOR[m], zorder=3)
        ax.set_yticks(y, [pl.scenario_label(s) for s in rows])
        ax.set(title=title_fmt.format(pl.STRUCTURE_LABEL[m]), xlabel=xlabel)
        ax.grid(axis="x", color="#E5E5E5", lw=0.5)
        pl.panel_label(ax, "abcdef"[k])
    proxies = [Line2D([], [], marker="o", ls="none", color="black", markersize=4),
               Line2D([], [], marker="o", ls="none", markerfacecolor="white",
                      markeredgecolor="black", markersize=4)]
    fig.legend(proxies, ["95 % CI excludes 0", "95 % CI includes 0"],
               loc="outside lower center", ncol=2)
    pl.save(fig, stem)


def anova_figure(anova, stem):
    order = [p for p in ("calibration", "validation", "transfer") if p in set(anova.period)]
    comps = ["forcing", "structure", "interaction"]
    fig, ax = pl.new_figure("onehalf", 62)
    width = 0.26
    x = np.arange(len(order))
    for k, comp in enumerate(comps):
        g = anova[anova.component == comp].set_index("period").reindex(order)
        ax.bar(x + (k - 1) * width, g.fraction, width, color=pl.COMPONENT_COLOR[comp],
               edgecolor="black", lw=0.4, label=pl.COMPONENT_LABEL[comp],
               yerr=[g.fraction - g.ci_low, g.ci_high - g.fraction], capsize=2,
               error_kw=dict(elinewidth=0.7))
    labels = {p: f"{p.capitalize()}\n({anova[anova.period == p].catchment.iat[0]})" for p in order}
    ax.set_xticks(x, [labels[p] for p in order])
    ax.set(ylabel="Fraction of KGE′ variance", ylim=(0, 1))
    ax.legend(loc="upper left", ncol=3)
    pl.save(fig, stem)


FEATURE_LABEL = {"T": "Temperature", "T_24h": "Temperature (24 h mean)",
                 "PET_24h": "PET (24 h)", "PET_720h": "PET (30 d)",
                 "hour_sin": "Hour of day (sin)", "hour_cos": "Hour of day (cos)",
                 "doy_sin": "Day of year (sin)", "doy_cos": "Day of year (cos)",
                 "sqrtQ_gr4h": "GR4H √Q", "Prod": "GR4H production store",
                 "Rout": "GR4H routing store"}


def feature_label(name: str) -> str:
    if name in FEATURE_LABEL:
        return FEATURE_LABEL[name]
    if m := re.fullmatch(r"P_(\d+)h", name):
        h = int(m.group(1))
        return f"P sum ({h // 24} d)" if h >= 24 and h % 24 == 0 else f"P sum ({h} h)"
    if m := re.fullmatch(r"API_([\d.]+)", name):
        return f"API (k = {m.group(1)})"
    if m := re.fullmatch(r"dQ_gr4h_(\d+)h", name):
        return f"GR4H Δ√Q ({m.group(1)} h)"
    return name


def shap_analysis(cfg, scenarios, d_tab, d_fig):
    cc, ct = cfg["gr4h"]["calibration_catchment"], cfg["gr4h"]["transfer_catchment"]
    rows = []
    for sc in scenarios:
        if not simulation_path(cfg, "GR4H", sc, cc).exists():
            continue
        xc, xt = predictors(cfg, cc, sc, True), predictors(cfg, ct, sc, True)
        cols = [c for c in xc if c not in ("Qobs", "Q_gr4h", "excluded")]
        _, _, model, _ = fit_predict(xc, xt, cols, True, cfg, 42)
        val = in_period(xc.index, cfg["periods"]["validation"]) & ~xc["excluded"].to_numpy()
        contrib = model.booster_.predict(xc.loc[val, cols], pred_contrib=True)[:, :-1]
        imp = pd.Series(np.abs(contrib).mean(0), index=cols)
        rows += [{"scenario": sc, "feature": k, "mean_abs_shap": v, "fraction": v / imp.sum()}
                 for k, v in imp.sort_values(ascending=False).items()]
    if not rows:
        return
    sh = pd.DataFrame(rows)
    sh.round(5).to_csv(d_tab / "11_shap.csv", index=False)
    groups = list(dict.fromkeys(sh.scenario))
    fig, axs = pl.new_figure("double", 75, ncols=len(groups))
    for k, (ax, sc) in enumerate(zip(np.atleast_1d(axs), groups)):
        g = sh[sh.scenario == sc].nlargest(10, "fraction")[::-1]
        ax.barh([feature_label(f) for f in g.feature], 100 * g.fraction,
                color=pl.STRUCTURE_COLOR["GR4H_GBM"], edgecolor="black", lw=0.3)
        ax.set(title=f"GR4H–GBM, {pl.scenario_label(sc)}", xlabel="Share of mean |SHAP| (%)")
        pl.panel_label(ax, "ab"[k])
    pl.save(fig, d_fig / "11_shap")
    log.info("SHAP (top 5 per scenario):\n%s",
             sh.groupby("scenario").head(5).round(3).to_string(index=False))


# --------------------------------------------------------------------------- main
def main():
    cfg = load_config()
    an, per, gr = cfg["analysis"], cfg["periods"], cfg["gr4h"]
    d_tab, d_fig = path(cfg, "tables"), path(cfg, "figures")
    cc, ct = gr["calibration_catchment"], gr["transfer_catchment"]
    forcing = pd.read_parquet(path(cfg, "processed") / f"forcing_{cc}.parquet")
    design, qm_ref = split_scenarios(forcing.columns)
    scenarios = design + qm_ref
    structures_all = an["structures"] + an["supplementary"]
    ref_s, ref_f = an["reference_structure"], an["reference_forcing"]
    log.info("Site %s%s", cfg["site"], f" | experiment {cfg['experiment']}" if cfg["experiment"] else "")

    unified, boot_rows, dif_rows, dm_rows, anova_rows = [], [], [], [], []
    for catchment, pname, rng_ in ((cc, "calibration", per["calibration"]),
                                   (cc, "validation", per["validation"]),
                                   (ct, "transfer", per["transfer"])):
        sims = load_simulations(cfg, structures_all, scenarios, catchment)
        if not sims:
            continue
        valid, q = common_sample(sims, rng_)
        valid_d, d = daily_table(valid, q)
        log.info("%s %s: common sample %d h (%.0f %%), %d days", catchment, pname, valid.sum(),
                 100 * valid.mean(), valid_d.sum())
        for (s, sc) in sims:
            for scale, o, sim in (("hourly", q["Qobs"].to_numpy()[valid], q[(s, sc)].to_numpy()[valid]),
                                  ("daily", d["Qobs"].to_numpy()[valid_d], d[(s, sc)].to_numpy()[valid_d])):
                unified.append({"model": s, "scenario": sc, "catchment": catchment,
                                "period": pname, "scale": scale, **streamflow_metrics(o, sim)})
        bs = BlockBootstrap(len(q), an["block_h"], an["bootstrap_replicates"], an["seed"])
        boot = {k: bs.score(q["Qobs"].to_numpy(), q[k].to_numpy(), valid) for k in sims}
        for (s, sc), v in boot.items():
            lo, hi = interval(v)
            boot_rows.append({"model": s, "scenario": sc, "catchment": catchment,
                              "period": pname, "kge_median": np.nanmedian(v), "ci_low": lo,
                              "ci_high": hi})
        obs_d = d["Qobs"].to_numpy()
        for (s, sc) in sims:
            for kind, ref in references(s, sc, ref_s, ref_f):
                if ref == (s, sc) or ref not in sims:
                    continue
                delta = boot[(s, sc)] - boot[ref]
                lo, hi = interval(delta)
                dif_rows.append({"kind": kind, "model": s, "scenario": sc, "catchment": catchment,
                                 "period": pname, "delta_median": np.nanmedian(delta),
                                 "ci_low": lo, "ci_high": hi,
                                 "p_improvement": np.nanmean(delta > 0)})
                stat, p = diebold_mariano(d[(s, sc)].to_numpy() - obs_d,
                                          d[ref].to_numpy() - obs_d, an["dm_lag_days"])
                dm_rows.append({"kind": kind, "model": s, "scenario": sc, "catchment": catchment,
                                "period": pname, "dm": stat, "p": p,
                                "more_accurate": "model" if stat < 0 else "reference"})
        est = [s for s in an["structures"] if all((s, sc) in boot for sc in design)]
        sce = [sc for sc in design if all((s, sc) in boot for s in est)]
        if len(est) >= 2 and len(sce) >= 2:
            m0 = np.array([[streamflow_metrics(q["Qobs"].to_numpy()[valid],
                                               q[(s, sc)].to_numpy()[valid])["kge"]
                            for s in est] for sc in sce])
            point = anova_fractions(m0)
            reps = anova_replicates(np.stack([np.stack([boot[(s, sc)] for s in est], -1)
                                              for sc in sce], -2))
            for comp, v in point.items():
                lo, hi = interval(reps[comp])
                anova_rows.append({"catchment": catchment, "period": pname, "component": comp,
                                   "fraction": v, "ci_low": lo, "ci_high": hi,
                                   "structures": "+".join(est), "n_scenarios": len(sce)})

    met = pd.DataFrame(unified)
    met.round(4).to_csv(d_tab / "11_metrics.csv", index=False)
    point = met[met.scale == "hourly"][["model", "scenario", "catchment", "period", "kge"]]
    boot = pd.DataFrame(boot_rows).merge(point.rename(columns={"kge": "kge_point"}),
                                         on=["model", "scenario", "catchment", "period"])
    boot.round(4).to_csv(d_tab / "11_bootstrap_ci.csv", index=False)
    dif = pd.DataFrame(dif_rows)
    dif.round(4).to_csv(d_tab / "11_differences.csv", index=False)
    dm = pd.DataFrame(dm_rows)
    if len(dm):
        dm["p_holm"] = dm.groupby(["kind", "catchment", "period"])["p"].transform(holm)
        dm["significant_5pct"] = dm["p_holm"] < 0.05
    dm.round(4).to_csv(d_tab / "11_diebold_mariano.csv", index=False)
    anova = pd.DataFrame(anova_rows)
    anova.round(4).to_csv(d_tab / "11_skill_anova.csv", index=False)

    main_structures = [s for s in an["structures"] if s in set(boot.model)]
    skill_matrix_figure(boot[boot.model.isin(main_structures)], main_structures, design,
                        d_fig / "11_skill_matrix")
    forest_figure(dif, "vs_GR4H", [s for s in main_structures if s != ref_s], design,
                  d_fig / "11_differences_gr4h", "ΔKGE′ (95 % CI)", "{} − GR4H")
    forest_figure(dif, "corr_vs_qm", main_structures, [s for s in design if s.endswith("_corr")],
                  d_fig / "11_ml_vs_qm", "ΔKGE′ (95 % CI)", "{}: ML − QM")
    if len(anova):
        anova_figure(anova, d_fig / "11_skill_anova")
        log.info("Variance fractions of KGE':\n%s",
                 anova[["catchment", "period", "component", "fraction", "ci_low", "ci_high"]]
                 .round(3).to_string(index=False))
    shap_analysis(cfg, an["shap_scenarios"], d_tab, d_fig)
    pl.design_schematic(d_fig / "11_study_design", [cfg])
    q = dif[(dif.kind == "corr_vs_qm") & (dif.period == "validation")] if len(dif) else dif
    if len(q):
        log.info("ML correction minus QM reference (validation):\n%s",
                 q[["model", "scenario", "delta_median", "ci_low", "ci_high"]].round(3)
                 .to_string(index=False))


if __name__ == "__main__":
    main()
