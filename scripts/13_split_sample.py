"""
Stage 13 - Reversed split-sample comparison

Purpose
    Repeat the modelling chain with calibration and validation periods swapped
    (differential split-sample test, Klemeš, 1986) and check which findings of the study hold
    in both directions.

Methods
    Stages 06–12 are re-run for the current site with PARAMO_EXPERIMENT=reversed
    (config.yaml > sites > <site> > experiments): each site swaps its own periods, the
    bias correction is retrained outside the new validation period and the early-stopping
    period of the learning models moves to the last year of the new calibration. Results go to
    the '*_reversed' folders, so the baseline is never overwritten. The comparison reports
    KGE' of every combination in both experiments, Spearman agreement of scenario and
    structure rankings, the ANOVA with the same structures in both experiments and a table of
    findings with a flag indicating whether each one holds.

Tools
    paramo.pipeline (stage execution), pandas, numpy, SciPy, matplotlib.

Inputs
    outputs/<site>/tables/ and outputs/<site>/tables_reversed/ (stages 06–12)

Outputs
    outputs/<site>/tables/13_kge_comparison.csv, 13_rankings.csv, 13_anova.csv,
    13_findings.csv
    outputs/<site>/figures/13_direct_vs_reversed.(png|pdf)

Usage
    python scripts/13_split_sample.py                  run the reversed chain and compare
    python scripts/13_split_sample.py --compare-only
    python scripts/13_split_sample.py --from 10        resume the reversed chain
    python scripts/13_split_sample.py --skip-lstm      omit stage 09 (hours on CPU)

Next stage
    Next site, or 14 - Cross-site synthesis once both sites are complete.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from paramo import plotting as pl
from paramo.core import ROOT, get_logger, load_config, path
from paramo.stats import anova_fractions

log = get_logger("13_split_sample")
EXPERIMENT = "reversed"


def run_chain(site: str, skip_lstm: bool, start: str | None):
    from paramo.pipeline import run_stages
    ids = ["06", "07", "08"] + ([] if skip_lstm else ["09"]) + ["10", "11", "12"]
    if start:
        if start not in ids:
            sys.exit(f"--from must be one of {ids}")
        ids = ids[ids.index(start):]
    d_log = ROOT / "outputs" / "logs"
    d_log.mkdir(parents=True, exist_ok=True)
    log_file = d_log / f"{site}_{EXPERIMENT}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    if not run_stages(ids, site, EXPERIMENT, log_file=log_file):
        sys.exit(1)


def findings(k, anova, dif, column) -> dict:
    out = {}
    val, tra = anova[anova.period == "validation"], anova[anova.period == "transfer"]
    a_v, a_t = val.set_index("component")[column], tra.set_index("component")[column]
    out["Forcing explains more variance than structure (validation)"] = \
        bool(a_v["forcing"] > a_v["structure"])
    out["Forcing explains more variance than structure (transfer)"] = \
        bool(a_t["forcing"] > a_t["structure"])
    v = k[k.period == "validation"]
    g = v[v.model == "GR4H"].set_index("scenario")[column]
    out["Gauges are the best forcing (GR4H)"] = bool(g.idxmax() == "gauge")
    out["Best satellite forcing (GR4H)"] = g.drop("gauge").idxmax()
    out["Corrected beats raw for all products (GR4H)"] = bool(all(
        g.get(f"{p}_corr", np.nan) > g.get(f"{p}_raw", np.nan)
        for p in ("imerg", "gsmap", "era5land", "fusion")))
    if dif is not None:
        x = dif[(dif.period == "validation") & (dif.kind == "vs_GR4H") & dif.model.isin(set(k.model))]
        out["No structure significantly better than GR4H with gauges"] = \
            bool((x[x.scenario == "gauge"].ci_low <= 0).all())
        raw = x[x.scenario.str.endswith("_raw")]
        out["Learning models better than GR4H with raw products (share, CI > 0)"] = \
            round(float((raw.ci_low > 0).mean()), 2)
        qm = dif[(dif.period == "validation") & (dif.kind == "corr_vs_qm") & (dif.model == "GR4H")]
        if len(qm):
            out["ML correction better than QM with GR4H (share of products)"] = \
                round(float((qm.delta_median > 0).mean()), 2)
            fus = qm[qm.scenario == "fusion_corr"]
            if len(fus):
                out["ML better than QM for the fusion (GR4H, CI > 0)"] = bool(fus.ci_low.iat[0] > 0)
    cal = k[k.period == "calibration"].groupby("model")[column].mean()
    drop = cal - v.groupby("model")[column].mean()
    if "GR4H" in drop:
        out["Calibration-to-validation loss larger for learning models"] = \
            bool((drop.drop("GR4H") > drop["GR4H"]).all())
    return out


def compare(cfg: dict):
    d_tab, d_fig = path(cfg, "tables"), path(cfg, "figures")
    d_rev = ROOT / f"{cfg['paths']['tables']}_{EXPERIMENT}"
    read = lambda d, f: pd.read_csv(d / f) if (d / f).exists() else None
    md, mr = read(d_tab, "11_metrics.csv"), read(d_rev, "11_metrics.csv")
    if md is None or mr is None:
        sys.exit(f"11_metrics.csv is missing for {cfg['site']} in the baseline or in the "
                 "reversed experiment.")
    md, mr = md[md.scale == "hourly"], mr[mr.scale == "hourly"]
    cc, ct = cfg["gr4h"]["calibration_catchment"], cfg["gr4h"]["transfer_catchment"]
    rows_k, rows_r, rows_a = [], [], []
    for catchment, pname in ((cc, "calibration"), (cc, "validation"), (ct, "transfer")):
        a = md[(md.catchment == catchment) & (md.period == pname)].set_index(["model", "scenario"]).kge
        b = mr[(mr.catchment == catchment) & (mr.period == pname)].set_index(["model", "scenario"]).kge
        common = a.index.intersection(b.index)
        scen = sorted(s for s in set(common.get_level_values(1)) if not s.endswith("_qm"))
        struct = [s for s in cfg["analysis"]["structures"]
                  if all((s, x) in common for x in scen)]
        for s in struct:
            for x in scen:
                rows_k.append({"catchment": catchment, "period": pname, "model": s,
                               "scenario": x, "direct": a[(s, x)], "reversed": b[(s, x)]})
        if len(struct) < 2:
            continue
        ma = np.array([[a[(s, x)] for s in struct] for x in scen])
        mb = np.array([[b[(s, x)] for s in struct] for x in scen])
        for comp in ("forcing", "structure", "interaction"):
            rows_a.append({"catchment": catchment, "period": pname, "component": comp,
                           "direct": anova_fractions(ma)[comp],
                           "reversed": anova_fractions(mb)[comp], "structures": "+".join(struct)})
        for j, s in enumerate(struct):
            rho, p = spearmanr(ma[:, j], mb[:, j])
            rows_r.append({"catchment": catchment, "period": pname, "ranking": "scenarios",
                           "within": s, "spearman": rho, "p": p})
        rho, p = spearmanr(ma.mean(0), mb.mean(0))
        rows_r.append({"catchment": catchment, "period": pname, "ranking": "structures",
                       "within": "mean over scenarios", "spearman": rho, "p": p})

    k = pd.DataFrame(rows_k)
    k["delta"] = k.reversed - k.direct
    k.round(4).to_csv(d_tab / "13_kge_comparison.csv", index=False)
    r = pd.DataFrame(rows_r)
    r.round(4).to_csv(d_tab / "13_rankings.csv", index=False)
    anova = pd.DataFrame(rows_a)
    anova.round(4).to_csv(d_tab / "13_anova.csv", index=False)

    fd = findings(k, anova, read(d_tab, "11_differences.csv"), "direct")
    fr = findings(k, anova, read(d_rev, "11_differences.csv"), "reversed")
    table = pd.DataFrame([{"finding": key, "direct": fd[key], "reversed": fr.get(key)}
                          for key in fd])
    table["holds"] = table.direct.astype(str) == table.reversed.astype(str)
    table.to_csv(d_tab / "13_findings.csv", index=False)

    fig, axs = pl.new_figure("double", 85, ncols=2)
    for n, (ax, pname) in enumerate(zip(axs, ("validation", "transfer"))):
        g = k[k.period == pname]
        lim = (min(g.direct.min(), g.reversed.min()) - 0.05,
               max(g.direct.max(), g.reversed.max()) + 0.05)
        ax.plot(lim, lim, color="#7F7F7F", lw=0.7, ls="--")
        for s in pl.structure_sort(g.model.unique()):
            gg = g[g.model == s]
            ax.scatter(gg.direct, gg.reversed, s=18, color=pl.STRUCTURE_COLOR[s],
                       marker=pl.STRUCTURE_MARKER.get(s, "o"),
                       edgecolor="black", lw=0.3, label=pl.STRUCTURE_LABEL[s])
        ax.set(xlim=lim, ylim=lim, xlabel="KGE′ direct experiment",
               ylabel="KGE′ reversed experiment")
        ax.set_title(f"{pname.capitalize()} ({cc if pname == 'validation' else ct})",
                     loc="right")
        ax.set_aspect("equal")
        pl.panel_label(ax, "ab"[n])
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels))
    pl.save(fig, d_fig / "13_direct_vs_reversed")
    log.info("Site %s", cfg["site"])
    log.info("ANOVA with the same structures:\n%s", anova.round(3).to_string(index=False))
    log.info("Ranking agreement:\n%s", r.round(3).to_string(index=False))
    log.info("Findings:\n%s", table.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare-only", action="store_true")
    ap.add_argument("--skip-lstm", action="store_true")
    ap.add_argument("--from", dest="start", help="resume the reversed chain at 06, 07, ... 12")
    a = ap.parse_args()
    config = load_config(experiment="")
    if not a.compare_only:
        run_chain(config["site"], a.skip_lstm, a.start)
    compare(config)
