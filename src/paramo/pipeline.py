"""Stage registry, execution and run records shared by run_pipeline.py and verify.py."""
from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .core import EXPERIMENT_ENV, ROOT, SITE_ENV


@dataclass(frozen=True)
class Stage:
    id: str
    script: str
    description: str
    outputs: tuple            # ((path key, glob pattern), ...)
    experiment: bool = False  # re-run inside experiments
    external: bool = False    # needs network services (Earth Engine)
    long: bool = False        # hours on CPU
    cross_site: bool = False  # runs once for all sites


STAGES = [
    Stage("00", "scripts/00_stations.py", "Station coordinates and series inventory",
          (("tables", "00_stations.csv"), ("interim", "stations.gpkg"))),
    Stage("01", "scripts/01_quality_control.py", "Hourly quality control",
          (("tables", "01_flag_summary.csv"),)),
    Stage("02", "scripts/02_diagnostics.py", "Rating curve, homogeneity and gauge diagnostics",
          (("tables", "02_homogeneity.csv"),)),
    Stage("03", "scripts/03_catchments.py", "DEM, catchments and attributes",
          (("tables", "03_catchments.csv"), ("interim", "catchments_wgs84.geojson"))),
    Stage("04", "scripts/04_satellite_extraction.py", "Earth Engine extraction",
          (("raw_gee", "imerg_points_*.csv"), ("raw_gee", "gsmap_points_*.csv"),
           ("raw_gee", "era5land_points_*.csv"), ("raw_gee", "imerg_catchments_*.csv")),
          external=True),
    Stage("05", "scripts/05_harmonization.py", "Hourly panel and time-stamp anchor",
          (("processed", "hourly_panel.parquet"), ("tables", "05_solar_anchor.csv"))),
    Stage("06", "scripts/06_bias_correction.py", "Bias correction, fusion and QM reference",
          (("processed", "corrected_precipitation.parquet"),
           ("tables", "06_bias_correction_evaluation.csv")), experiment=True),
    Stage("07", "scripts/07_catchment_forcing.py", "Catchment forcing scenarios",
          (("processed", "forcing_*.parquet"), ("tables", "07_forcing_summary.csv")),
          experiment=True),
    Stage("08", "scripts/08_gr4h.R", "GR4H calibration, validation and transfer",
          (("tables", "08_metrics.csv"),), experiment=True),
    Stage("09", "scripts/09_lstm.py", "LSTM benchmark", (("tables", "09_metrics.csv"),),
          experiment=True, long=True),
    Stage("10", "scripts/10_gbm.py", "GBM and GR4H–GBM", (("tables", "10_metrics.csv"),),
          experiment=True),
    Stage("11", "scripts/11_factorial_analysis.py", "Factorial analysis of skill",
          (("tables", "11_skill_anova.csv"), ("tables", "11_differences.csv")),
          experiment=True),
    Stage("12", "scripts/12_sensitivity.py", "Sensitivity of the conclusions",
          (("tables", "12_anova_by_metric.csv"),), experiment=True),
    Stage("13", "scripts/13_split_sample.py", "Reversed split-sample comparison",
          (("tables", "13_findings.csv"),)),
    Stage("14", "scripts/14_synthesis.py", "Cross-site synthesis",
          (("synthesis", "14_anova_sites.csv"),), cross_site=True),
]
STAGE_IDS = [s.id for s in STAGES]
SITE_STAGE_IDS = [s.id for s in STAGES if not s.cross_site]


def stage(stage_id: str) -> Stage:
    return next(s for s in STAGES if s.id == stage_id)


def rscript() -> str | None:
    found = shutil.which("Rscript")
    if found:
        return found
    if os.name == "nt":
        candidates = sorted(glob.glob(r"C:\Program Files\R\R-*\bin\Rscript.exe"))
        return candidates[-1] if candidates else None
    return None


def command(s: Stage, extra: list[str] | None = None) -> list[str]:
    if s.script.endswith(".R"):
        r = rscript()
        if r is None:
            raise FileNotFoundError("Rscript not found; install R >= 4.2 (stage 08).")
        return [r, s.script] + (extra or [])
    return [sys.executable, s.script] + (extra or [])


GEE_TARGETS = ("points", "catchments")
LEGACY_GEE_NAMES = {"_puntos_": "_points_", "_cuencas_": "_catchments_"}


def gee_years(cfg: dict) -> list[int]:
    start, end = cfg["gee"]["period"]
    return list(range(int(str(start)[:4]), int(str(end)[:4])))


def gee_missing(cfg: dict) -> list[str]:
    """Earth Engine files (product × target × year) still absent for the site."""
    folder = ROOT / cfg["paths"]["raw_gee"]
    return [f"{p}_{t}_{y}.csv" for t in GEE_TARGETS for p in cfg["gee"]["products"]
            for y in gee_years(cfg) if not (folder / f"{p}_{t}_{y}.csv").exists()]


def rename_legacy_gee(cfg: dict) -> int:
    """Rename extractions of the former version (_puntos_/_cuencas_) to the current names."""
    folder = ROOT / cfg["paths"]["raw_gee"]
    n = 0
    for f in (folder.glob("*.csv") if folder.exists() else []):
        new = f.name
        for old, cur in LEGACY_GEE_NAMES.items():
            new = new.replace(old, cur)
        if new != f.name and not (folder / new).exists():
            f.rename(folder / new)
            n += 1
    return n


def outputs_present(s: Stage, cfg: dict) -> tuple[bool, list[str]]:
    if s.id == "04":                       # complete only when every year is present
        missing = gee_missing(cfg)
        return not missing, [f"{cfg['paths']['raw_gee']}/{m}" for m in missing]
    missing = [f"{cfg['paths'][k]}/{pattern}" for k, pattern in s.outputs
               if not glob.glob(str(ROOT / cfg["paths"][k] / pattern))]
    return not missing, missing


# --------------------------------------------------------------------------- run records
# File times are unreliable across machines (archives, time zones), so validity is tracked
# with a record written after each successful run: script fingerprint and run time.
def _record_file(s: Stage, site: str | None, experiment: str | None) -> Path:
    d = ROOT / "outputs" / ".runs"
    d.mkdir(parents=True, exist_ok=True)
    scope = "all" if s.cross_site else site
    return d / f"{scope}__{experiment or 'baseline'}__{s.id}.json"


def fingerprint(script: str) -> str:
    return hashlib.sha1((ROOT / script).read_bytes()).hexdigest()[:12]


def write_record(s: Stage, site: str | None, experiment: str | None, fp: str):
    """fp is the fingerprint of the script taken when the run started."""
    _record_file(s, site, experiment).write_text(json.dumps(
        {"stage": s.id, "script": s.script, "fingerprint": fp,
         "time": time.time(), "date": time.strftime("%Y-%m-%d %H:%M:%S")}), encoding="utf-8")


def read_record(s: Stage, site: str | None, experiment: str | None) -> dict | None:
    f = _record_file(s, site, experiment)
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def stage_status(s: Stage, cfg: dict, site: str | None, experiment: str | None,
                 upstream_time: float) -> tuple[str, str, float]:
    """('complete' | 'unrecorded' | 'outdated' | 'pending', detail, run time)."""
    ok, missing = outputs_present(s, cfg)
    if not ok:
        return "pending", missing[0], 0.0
    rec = read_record(s, site, experiment)
    if rec is None:
        return "unrecorded", "outputs exist but were not produced by run_pipeline.py", 0.0
    if rec["fingerprint"] != fingerprint(s.script):
        return "outdated", f"{s.script} changed after the run of {rec['date']}", rec["time"]
    if upstream_time > rec["time"]:
        return "outdated", "an upstream stage ran later", rec["time"]
    return "complete", f"run on {rec['date']}", rec["time"]


def run_stages(ids: list[str], site: str | None, experiment: str | None = None,
               extra: dict | None = None, log_file: Path | None = None,
               dry_run: bool = False) -> bool:
    """Run stages in order with the current interpreter; stop at the first failure."""
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    if site:
        env[SITE_ENV] = site
    if experiment:
        env[EXPERIMENT_ENV] = experiment
    else:
        env.pop(EXPERIMENT_ENV, None)
    log = open(log_file, "a", encoding="utf-8") if log_file else None
    tag = f"{site or 'all sites'} / {experiment or 'baseline'}"

    def emit(text: str):
        print(text, flush=True)
        if log:
            log.write(text + "\n")
            log.flush()

    try:
        for sid in ids:
            s = stage(sid)
            cmd = command(s, (extra or {}).get(sid))
            emit(f"\n{'=' * 78}\n[{tag}] stage {s.id}: {s.description}"
                 f"\n$ {' '.join(cmd)}\n{'=' * 78}")
            if dry_run:
                continue
            t0, fp = time.time(), fingerprint(s.script)
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                    errors="replace")
            for line in proc.stdout:
                emit(line.rstrip())
            proc.wait()
            if proc.returncode != 0:
                emit(f"*** Stage {s.id} failed for {tag} (exit code {proc.returncode}). "
                     f"Resume with: --site {site} --from {s.id}")
                return False
            emit(f"--- stage {s.id} finished in {(time.time() - t0) / 60:.1f} min")
            write_record(s, site, experiment, fp)
        return True
    finally:
        if log:
            log.close()
