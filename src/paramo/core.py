"""Configuration, sites, experiments, paths, logging and common I/O of the workflow."""
from __future__ import annotations

import copy
import logging
import os
import re
import shutil
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
SITE_ENV = "PARAMO_SITE"
EXPERIMENT_ENV = "PARAMO_EXPERIMENT"

# Quality flags (the highest severity is kept per record)
FLAG_OK, FLAG_SUSPECT, FLAG_ERROR, FLAG_MISSING = 0, 1, 2, 3
FLAG_NAMES = {0: "ok", 1: "suspect", 2: "error", 3: "missing"}

# Folders that change with the experiment; the others are shared by all experiments of a site
EXPERIMENT_PATHS = ("processed", "tables", "figures")


# --------------------------------------------------------------------------- configuration
def _merge(base: dict, extra: dict) -> dict:
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def read_raw_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def site_names() -> list[str]:
    return list(read_raw_config()["sites"])


def load_config(site: str | None = None, experiment: str | None = None,
                prepare: bool = True) -> dict:
    """Configuration of one site and, optionally, one experiment.

    The site is taken from the argument, from PARAMO_SITE or from project.default_site; the
    experiment from the argument or from PARAMO_EXPERIMENT. Site settings override the common
    ones, and the experiment settings of that site override both. Path templates receive the
    site name; an experiment redirects processed data, tables and figures to
    '<folder>_<experiment>', so the baseline is never overwritten. With prepare=True the
    experiment folders are created and the inputs that experiments read but do not regenerate
    (hourly panel, station and catchment tables) are copied from the baseline.
    """
    raw = read_raw_config()
    site = site or os.environ.get(SITE_ENV, "").strip() or raw["project"]["default_site"]
    if site not in raw["sites"]:
        raise KeyError(f"Site '{site}' is not defined in config.yaml (sites: {list(raw['sites'])})")
    if experiment is None:
        experiment = os.environ.get(EXPERIMENT_ENV, "").strip() or None
    cfg = {k: copy.deepcopy(v) for k, v in raw.items() if k != "sites"}
    site_cfg = copy.deepcopy(raw["sites"][site])
    experiments = site_cfg.pop("experiments", {})
    _merge(cfg, site_cfg)
    cfg["site"] = site
    cfg["experiments"] = experiments
    cfg["paths"] = {k: v.format(site=site) for k, v in cfg["paths"].items()}
    cfg["experiment"] = experiment or None
    if not experiment:
        return cfg
    if experiment not in experiments:
        raise KeyError(f"Experiment '{experiment}' is not defined for site '{site}'")
    base = dict(cfg["paths"])
    _merge(cfg, experiments[experiment])
    for key in EXPERIMENT_PATHS:
        cfg["paths"][key] = f"{base[key]}_{experiment}"
    if prepare:
        for key in EXPERIMENT_PATHS:
            (ROOT / cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
        for key, fname in (("processed", "hourly_panel.parquet"),
                           ("tables", "00_stations.csv"), ("tables", "03_catchments.csv")):
            src, dst = ROOT / base[key] / fname, ROOT / cfg["paths"][key] / fname
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
    return cfg


def path(cfg: dict, key: str) -> Path:
    p = ROOT / cfg["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, datefmt="%H:%M:%S",
                        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    for noisy in ("fontTools", "matplotlib", "PIL", "pyogrio", "rasterio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger(name)


# --------------------------------------------------------------------------- text and stations
def normalize(text: str) -> str:
    """Lower case without accents; also decodes names such as 'Precipitaci#U00f3n'."""
    text = re.sub(r"#U([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), text)
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c)).lower()


def read_stations(cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(ROOT / cfg["paths"]["input"] / cfg["stations"]["file"], sep="\t",
                     encoding="utf-8")
    cols = {normalize(c).strip(): c for c in df.columns}
    return pd.DataFrame({"code": df[cols["codigo"]].astype(str).str.strip(),
                         "name": df[cols["estacion"]].astype(str).str.strip(),
                         "x_tmq": df[cols["x"]].astype(float),
                         "y_tmq": df[cols["y"]].astype(float),
                         "elevation_m": df[cols["altura"]].astype(float)})


def find_series_file(cfg: dict, code: str, variable: str) -> Path | None:
    key = cfg["variables"][variable]["file_key"]
    for f in sorted((ROOT / cfg["paths"]["input"]).glob(f"{code}-*.csv")):
        if key in normalize(f.name):
            return f
    return None


def read_series(path_: Path) -> pd.DataFrame:
    """Read an hourly FONAG/EPMAPS file and place it on an exact hourly index (local time)."""
    df = pd.read_csv(path_)
    df.columns = [normalize(c) for c in df.columns]
    df["fecha"] = pd.to_datetime(df["fecha"], format="%Y/%m/%d %H:%M:%S")
    n_dup = int(df["fecha"].duplicated().sum())
    df = df.drop_duplicates("fecha").set_index("fecha").sort_index()
    df = df.rename(columns={"valor": "value", "completo_mediciones": "completeness"})
    idx = pd.date_range(df.index.min().floor("h"), df.index.max().ceil("h"), freq="h",
                        name="time")
    df = df.reindex(idx)
    df.attrs["duplicates"] = n_dup
    return df


def read_qc(cfg: dict, code: str, variable: str) -> pd.DataFrame | None:
    f = ROOT / cfg["paths"]["interim"] / f"{code}_{variable}.parquet"
    return pd.read_parquet(f) if f.exists() else None


# --------------------------------------------------------------------------- periods and scenarios
def period_end(day) -> pd.Timestamp:
    """Last hour of a calendar day (built directly; avoids Timedelta arithmetic warnings)."""
    return pd.Timestamp(f"{pd.Timestamp(day):%Y-%m-%d} 23:00:00")


def in_period(idx: pd.DatetimeIndex, period) -> np.ndarray:
    return (idx >= pd.Timestamp(period[0])) & (idx <= period_end(period[1]))


def in_periods(idx: pd.DatetimeIndex, periods) -> np.ndarray:
    mask = np.zeros(len(idx), bool)
    for p in periods:
        mask |= in_period(idx, p)
    return mask


def split_scenarios(columns) -> tuple[list[str], list[str]]:
    """Precipitation scenarios from 'P__*' columns: (factorial design, QM reference)."""
    names = [str(c)[3:] for c in columns if str(c).startswith("P__")]
    return [n for n in names if not n.endswith("_qm")], [n for n in names if n.endswith("_qm")]


def thiessen_gauges(cfg: dict) -> list[str]:
    """Gauges that build the areal precipitation (defaults to all rain gauges)."""
    return list(cfg["forcing"].get("thiessen_gauges") or cfg["stations"]["rain_gauges"])


# --------------------------------------------------------------------------- simulations
def simulation_path(cfg: dict, structure: str, scenario: str, catchment: str) -> Path:
    d = path(cfg, "processed") / "simulations"
    d.mkdir(exist_ok=True)
    return d / f"{structure}__{scenario}__{catchment}.csv"


def read_simulation(file: Path) -> pd.DataFrame:
    """Simulation table written by R or Python: time, Qobs, Qsim, excluded (+ states)."""
    df = pd.read_csv(file, engine="pyarrow")
    df["time"] = pd.to_datetime(df["time"])
    df["excluded"] = df["excluded"].astype(str).str.upper().isin(["TRUE", "1", "1.0"])
    return df.set_index("time")


def write_simulation(df: pd.DataFrame, file: Path):
    out = df.copy()
    out.index.name = "time"
    out.to_csv(file, float_format="%.6g", date_format="%Y-%m-%d %H:%M:%S")
