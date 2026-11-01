"""
Stage 14 - Cross-site synthesis

Purpose
    Bring the two catchments together and check whether the main conclusions of the
    factorial design (forcing versus structure, ranking of precipitation products, value of
    the ML correction) are reproduced in an independent páramo catchment.

Methods
    For every site and experiment (direct and reversed) the stage collects the two-way ANOVA
    fractions of KGE′ with their bootstrap intervals (stage 11), the hourly KGE′ of every
    forcing × structure combination and the findings table of the reversed split-sample test
    (stage 13). Agreement between sites is measured with the Spearman correlation of the
    forcing ranking (mean KGE′ over the structures common to both sites) and of the structure
    ranking. Each site keeps its own periods, gauges and catchments; only the protocol is
    shared, so the comparison concerns conclusions, not absolute scores.

Tools
    pandas, numpy, SciPy, matplotlib; paramo.plotting (study-design schematic).

Inputs
    outputs/<site>/tables[_reversed]/11_skill_anova.csv, 11_metrics.csv (stage 11)
    outputs/<site>/tables/13_findings.csv (stage 13)

Outputs
    outputs/synthesis/14_anova_sites.csv, 14_skill_sites.csv, 14_rankings_sites.csv,
                      14_findings_sites.csv
    outputs/synthesis/14_sites_comparison.(png|pdf), 14_study_design.(png|pdf)

Usage
    python scripts/14_synthesis.py                 all sites in config.yaml
    python scripts/14_synthesis.py --sites tambo chalpi

Next stage
    End of the workflow: tables and figures for the manuscript.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from paramo import plotting as pl
from paramo.core import ROOT, get_logger, load_config, path, site_names

log = get_logger("14_synthesis")
EXPERIMENTS = (None, "reversed")


def tables_dir(cfg: dict, experiment: str | None) -> Path:
    base = ROOT / load_config(cfg["site"], experiment="", prepare=False)["paths"]["tables"]
    return base if experiment is None else Path(f"{base}_{experiment}")


def collect(cfgs: list[dict]):
    anova, skill, findings = [], [], []
    for cfg in cfgs:
        site = cfg["site"]
        cc, ct = cfg["gr4h"]["calibration_catchment"], cfg["gr4h"]["transfer_catchment"]
        for experiment in EXPERIMENTS:
            d = tables_dir(cfg, experiment)
            tag = experiment or "direct"
            if (d / "11_skill_anova.csv").exists():
                a = pd.read_csv(d / "11_skill_anova.csv")
                anova.append(a.assign(site=site, experiment=tag))
            if (d / "11_metrics.csv").exists():
                m = pd.read_csv(d / "11_metrics.csv")
                m = m[(m.scale == "hourly") & (((m.catchment == cc) & (m.period != "transfer"))
                                               | ((m.catchment == ct) & (m.period == "transfer")))]
                skill.append(m[["model", "scenario", "catchment", "period", "kge"]]
                             .assign(site=site, experiment=tag))
        f = tables_dir(cfg, None) / "13_findings.csv"
        if f.exists():
            findings.append(pd.read_csv(f).assign(site=site))
    cat = lambda x: pd.concat(x, ignore_index=True) if x else pd.DataFrame()
    return cat(anova), cat(skill), cat(findings)


def rankings(skill: pd.DataFrame, structures: list[str], sites: list[str]) -> pd.DataFrame:
    """Spearman agreement of forcing and structure rankings between pairs of sites."""
    rows = []
    for (experiment, period), g in skill.groupby(["experiment", "period"]):
        if period == "calibration":
            continue
        piv = g[~g.scenario.str.endswith("_qm") & g.model.isin(structures)]
        for i, a in enumerate(sites):
            for b in sites[i + 1:]:
                ga, gb = piv[piv.site == a], piv[piv.site == b]
                common_s = sorted(set(ga.model) & set(gb.model))
                common_f = sorted(set(ga.scenario) & set(gb.scenario))
                if len(common_s) < 1 or len(common_f) < 3:
                    continue
                sel = lambda x: x[x.model.isin(common_s) & x.scenario.isin(common_f)]
                fa = sel(ga).groupby("scenario").kge.mean()
                fb = sel(gb).groupby("scenario").kge.mean().reindex(fa.index)
                rho, p = spearmanr(fa, fb)
                rows.append({"experiment": experiment, "period": period, "ranking": "forcings",
                             "sites": f"{a}–{b}", "n": len(fa), "spearman": rho, "p": p,
                             "best_" + a: fa.idxmax(), "best_" + b: fb.idxmax()})
                if len(common_s) >= 3:
                    sa = sel(ga).groupby("model").kge.mean()
                    sb = sel(gb).groupby("model").kge.mean().reindex(sa.index)
                    rho, p = spearmanr(sa, sb)
                    rows.append({"experiment": experiment, "period": period,
                                 "ranking": "structures", "sites": f"{a}–{b}", "n": len(sa),
                                 "spearman": rho, "p": p, "best_" + a: sa.idxmax(),
                                 "best_" + b: sb.idxmax()})
    return pd.DataFrame(rows)


def findings_table(findings: pd.DataFrame, sites: list[str]) -> pd.DataFrame:
    if findings.empty:
        return findings
    wide = None
    for site in sites:
        f = findings[findings.site == site][["finding", "direct", "reversed"]]
        f = f.rename(columns={"direct": f"{site}_direct", "reversed": f"{site}_reversed"})
        wide = f if wide is None else wide.merge(f, on="finding", how="outer")
    cols = [c for c in wide if c != "finding"]
    wide["holds_everywhere"] = wide[cols].astype(str).nunique(axis=1).eq(1) & \
        wide[cols].notna().all(axis=1)
    return wide


def figure(anova: pd.DataFrame, skill: pd.DataFrame, cfgs: list[dict], stem: Path):
    labels = {c["site"]: pl.site_label(c) for c in cfgs}
    sites = [c["site"] for c in cfgs]
    fig, (ax1, ax2) = pl.new_figure("double", 95, ncols=2,
                                    gridspec_kw={"width_ratios": [1.0, 1.15]})
    # (a) variance fractions: site × period, direct and reversed side by side
    groups = [(s, p) for s in sites for p in ("validation", "transfer")]
    width, x_ticks, x_labels = 0.36, [], []
    for i, (site, period) in enumerate(groups):
        for j, experiment in enumerate(("direct", "reversed")):
            g = anova[(anova.site == site) & (anova.period == period) &
                      (anova.experiment == experiment)].set_index("component")
            if g.empty:
                continue
            x = i + (j - 0.5) * (width + 0.04)
            bottom = 0.0
            for comp in ("forcing", "structure", "interaction"):
                v = float(g.fraction.get(comp, 0.0))
                ax1.bar(x, v, width, bottom=bottom, color=pl.COMPONENT_COLOR[comp],
                        edgecolor="black", lw=0.3, hatch="////" if j else None,
                        label=pl.COMPONENT_LABEL[comp] if (i == 0 and j == 0) else None)
                bottom += v
        x_ticks.append(i)
        x_labels.append(f"{labels[site]}\n{period}")
    ax1.bar(np.nan, np.nan, color="white", edgecolor="black", lw=0.3, hatch="////",
            label="Reversed periods")
    ax1.set_xticks(x_ticks, x_labels)
    ax1.set(ylim=(0, 1), ylabel="Fraction of KGE′ variance")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    pl.panel_label(ax1, "a")

    # (b) hourly KGE′ in validation, mean over structures and range, direct experiment
    structures = cfgs[0]["analysis"]["structures"]          # only the 4 structures of the design
    v = skill[(skill.period == "validation") & (skill.experiment == "direct") &
              ~skill.scenario.str.endswith("_qm") & skill.model.isin(structures)]
    scen = [s for s in pl.scenario_sort(v.scenario.unique())]
    y = np.arange(len(scen))[::-1]
    markers, offsets = ["o", "s", "D", "^"], np.linspace(-0.18, 0.18, len(sites))
    site_colors = [pl.OKABE_ITO["blue"], pl.OKABE_ITO["vermillion"], pl.OKABE_ITO["green"],
                   pl.OKABE_ITO["purple"]]
    for k, site in enumerate(sites):
        g = v[v.site == site].groupby("scenario").kge
        mean, lo, hi = g.mean().reindex(scen), g.min().reindex(scen), g.max().reindex(scen)
        ax2.hlines(y + offsets[k], lo, hi, color=site_colors[k], lw=1.0)
        ax2.scatter(mean, y + offsets[k], marker=markers[k], s=18, color=site_colors[k],
                    edgecolor="black", lw=0.3, zorder=3, label=labels[site])
    ax2.set_yticks(y, [pl.scenario_label(s) for s in scen])
    ax2.axvline(0, color="#7F7F7F", lw=0.5, ls=":")
    ax2.set_xlabel("KGE′ validation (mean and range over structures)")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=len(sites))
    pl.panel_label(ax2, "b")
    pl.save(fig, stem)


def main(a):
    sites = a.sites or site_names()
    cfgs = [load_config(s, experiment="", prepare=False) for s in sites]
    out = path(cfgs[0], "synthesis")
    structures = cfgs[0]["analysis"]["structures"]
    anova, skill, findings = collect(cfgs)
    if anova.empty or skill.empty:
        sys.exit("No stage 11 results were found; run the site stages first.")
    done = sorted(set(skill.site))
    missing = [s for s in sites if s not in done]
    if missing:
        log.warning("Sites without stage 11 results: %s", missing)
    cols = ["site", "experiment", "catchment", "period", "component", "fraction", "ci_low",
            "ci_high", "structures", "n_scenarios"]
    anova[cols].round(4).to_csv(out / "14_anova_sites.csv", index=False)
    skill.round(4).to_csv(out / "14_skill_sites.csv", index=False)
    r = rankings(skill, structures, done)
    r.round(4).to_csv(out / "14_rankings_sites.csv", index=False)
    f = findings_table(findings, done)
    f.to_csv(out / "14_findings_sites.csv", index=False)
    figure(anova, skill, [c for c in cfgs if c["site"] in done], out / "14_sites_comparison")
    pl.design_schematic(out / "14_study_design", cfgs)
    piv = anova.pivot_table(index=["site", "experiment", "period"], columns="component",
                            values="fraction").round(3)
    log.info("Variance fractions of KGE' by site:\n%s", piv.to_string())
    if len(r):
        log.info("Ranking agreement between sites:\n%s", r.round(3).to_string(index=False))
    if len(f):
        log.info("Findings held in every site and experiment: %d of %d",
                 int(f.holds_everywhere.sum()), len(f))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="*")
    main(ap.parse_args())
