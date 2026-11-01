"""
Run the páramo streamflow workflow end to end for one or both sites, with a log of every
stage.

Order
  For each site (config.yaml > sites), stages 00 to 13 run in sequence; stage 14 (cross-site
  synthesis) runs once at the end when every site in the plan has finished.
  - 03 uses --no-download when the site DEM (data/raw/<site>/dem/dem_tmq.tif) already
    exists; Copernicus tiles are shared between sites in data/raw/dem_tiles.
  - 04 (Earth Engine) runs automatically when any product, target or year is missing for
    the site (data/raw/<site>/gee) and downloads only what is missing; the first time it
    opens the browser for Earth Engine authentication. Files of the former version
    (_puntos_/_cuencas_) are renamed. --with-gee forces stage 04 to run anyway.
  - 09 (LSTM, hours on CPU) runs when LSTM is part of analysis.structures, unless --skip-lstm.
  - 13 re-runs stages 06-12 of the site with its reversed periods and compares both.
The output of each run goes to outputs/logs/pipeline_<date>.log, and each successful stage
leaves a run record (outputs/.runs) used by verify.py.

Usage (from the project folder)
  python run_pipeline.py                         both sites, stages 00-14
  python run_pipeline.py --site chalpi           one site, stages 00-13
  python run_pipeline.py --site chalpi --from 06 resume at a stage
  python run_pipeline.py --only 14               synthesis only
  python run_pipeline.py --site tambo --experiment reversed --from 06 --to 12
  python run_pipeline.py --dry-run               show the plan only
"""
import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from paramo.core import load_config, site_names  # noqa: E402
from paramo.pipeline import (SITE_STAGE_IDS, STAGE_IDS, gee_missing,  # noqa: E402
                             gee_years, rename_legacy_gee, run_stages, stage)


def site_plan(a, site: str) -> tuple[list[str], dict, str | None]:
    """Stages of one site, extra arguments and a note on the Earth Engine files."""
    cfg = load_config(site, a.experiment or "", prepare=False)
    if a.only:
        ids = [i for i in a.only if i in SITE_STAGE_IDS]
    else:
        lo = STAGE_IDS.index(a.start or "00")
        hi = STAGE_IDS.index(a.end or "14")
        ids = [i for i in SITE_STAGE_IDS if lo <= STAGE_IDS.index(i) <= hi]
    if a.experiment:
        ids = [i for i in ids if stage(i).experiment]
    extra, note = {}, None
    if "03" in ids and (ROOT / cfg["paths"]["raw_dem"] / "dem_tmq.tif").exists():
        extra["03"] = ["--no-download"]
    renamed = 0 if a.dry_run else rename_legacy_gee(cfg)
    missing = gee_missing(cfg)
    if missing and "05" in ids and "04" not in ids:       # the panel needs every year
        ids.insert(ids.index("05"), "04")
    if "04" in ids:
        if not missing and not a.with_gee and not (a.only and "04" in a.only):
            ids.remove("04")
        elif missing:
            note = (f"{len(missing)} of {len(gee_years(cfg)) * 2 * len(cfg['gee']['products'])}"
                    f" Earth Engine files missing in {cfg['paths']['raw_gee']}: they will be "
                    "downloaded (stage 04)")
        if renamed:
            note = f"{renamed} files renamed from _puntos_/_cuencas_; " + (note or "")
    uses_lstm = any(s.startswith("LSTM") for s in cfg["analysis"]["structures"])
    skip_lstm = a.skip_lstm or not uses_lstm
    if "09" in ids and skip_lstm and not (a.only and "09" in a.only):
        ids.remove("09")
    if "13" in ids:
        extra["13"] = ["--skip-lstm"] if skip_lstm else []
    return ids, extra, note


def main():
    sites_all = site_names()
    ap = argparse.ArgumentParser(description="Páramo streamflow workflow")
    ap.add_argument("--site", default="all", choices=sites_all + ["all"])
    ap.add_argument("--from", dest="start", choices=STAGE_IDS)
    ap.add_argument("--to", dest="end", choices=STAGE_IDS)
    ap.add_argument("--only", nargs="+", choices=STAGE_IDS)
    ap.add_argument("--experiment", help="experiment defined for the site (e.g. reversed)")
    ap.add_argument("--with-gee", action="store_true",
                    help="run stage 04 even when every Earth Engine file is present")
    ap.add_argument("--skip-lstm", action="store_true", help="omit stage 09")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    sites = sites_all if a.site == "all" else [a.site]

    if not a.no_verify:
        r = subprocess.run([sys.executable, str(ROOT / "verify.py"), "--site", a.site,
                            "--no-environment", "--no-status"], cwd=ROOT)
        if r.returncode != 0:
            sys.exit("verify.py found configuration or input errors; fix them or use --no-verify.")

    cfg0 = load_config(sites[0], "", prepare=False)
    d_log = ROOT / cfg0["paths"]["logs"]
    d_log.mkdir(parents=True, exist_ok=True)
    log_file = None if a.dry_run else d_log / f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"
    plans = {s: site_plan(a, s) for s in sites}
    synthesis = (not a.experiment and
                 ("14" in a.only if a.only else
                  a.site == "all" and STAGE_IDS.index(a.end or "14") >= STAGE_IDS.index("14")))
    print(f"\nExperiment: {a.experiment or 'baseline'}")
    for s, (ids, _, note) in plans.items():
        print(f"  {s:<8} {' -> '.join(ids) or '(nothing)'}")
        if note:
            print(f"           {note}")
    if synthesis:
        print("  all      14 (cross-site synthesis)")
    if log_file:
        print(f"Log: {log_file.relative_to(ROOT)}")

    for s, (ids, extra, _) in plans.items():
        if ids and not run_stages(ids, s, a.experiment, extra, log_file, a.dry_run):
            sys.exit(1)
    if synthesis and not run_stages(["14"], None, None, None, log_file, a.dry_run):
        sys.exit(1)
    if not a.dry_run:
        print(f"\nWorkflow finished ({a.experiment or 'baseline'}). Status: python verify.py")


if __name__ == "__main__":
    main()
