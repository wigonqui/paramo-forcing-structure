"""
Stage 04 - Earth Engine extraction of satellite and reanalysis forcing

Purpose
    Extract hourly IMERG V07 (30 min), GSMaP v8 and ERA5-Land series for the selected site at
    the pixel of each rain gauge and weather station ('points', for the bias correction, the
    time offsets and the solar anchor) and as
    area-weighted means over each catchment ('catchments', the model forcing). Each site needs
    its own extraction: its stations and catchments are different pixels and polygons.

Methods
    One reduceRegions call per image, flattened into a table. In 'direct' mode the table is
    downloaded month by month in parallel through the high-volume endpoint; months are kept
    as partial files so an interrupted run resumes where it stopped, and a month rejected for
    size or time-out is split in halves down to one day. 'drive' mode launches one export
    task per product, target and year instead, into the Drive folder
    '<gee.drive_folder>_<site>' so the files of the two sites never mix.

Tools
    earthengine-api, pandas, concurrent.futures.

Inputs
    outputs/<site>/tables/00_stations.csv (stage 00)
    data/interim/<site>/catchments_wgs84.geojson (stage 03)
    Earth Engine authentication: python install_environment.py --gee-auth

Outputs
    data/raw/<site>/gee/<product>_<points|catchments>_<year>.csv   columns: id, t (ms UTC), bands

Usage
    python scripts/04_satellite_extraction.py --site chalpi --info      date range per product
    python scripts/04_satellite_extraction.py --site chalpi --dry-run   files still missing
    python scripts/04_satellite_extraction.py --site chalpi             all products and years
    python scripts/04_satellite_extraction.py --site chalpi --mode drive

Next stage
    05 - Hourly panel, time-stamp anchor and offsets.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from paramo.core import ROOT, get_logger, load_config, path

log = get_logger("04_extraction")
HIGH_VOLUME_URL = "https://earthengine-highvolume.googleapis.com"
RETRIES, MIN_DAYS = 3, 1


def point_stations(cfg) -> pd.DataFrame:
    """Stations whose pixels are used later: rain gauges (bias correction, time offsets) and
    weather stations (temperature and the solar anchor)."""
    st = pd.read_csv(ROOT / cfg["paths"]["tables"] / "00_stations.csv")
    used = set(cfg["stations"]["rain_gauges"]) | set(cfg["stations"]["weather"])
    return st[st.code.isin(used)]


def features(cfg, target: str):
    import ee
    if target == "points":
        st = point_stations(cfg)
        return ee.FeatureCollection([ee.Feature(ee.Geometry.Point([r.lon, r.lat]), {"id": r.code})
                                     for r in st.itertuples()])
    f = ROOT / cfg["paths"]["interim"] / "catchments_wgs84.geojson"
    if not f.exists():
        raise FileNotFoundError(f"{f}: run stage 03 first")
    gj = json.loads(f.read_text(encoding="utf-8"))
    return ee.FeatureCollection([ee.Feature(ee.Geometry(ft["geometry"]),
                                            {"id": ft["properties"]["code"]})
                                 for ft in gj["features"]])


def table(cfg, product: str, fc, start: pd.Timestamp, end: pd.Timestamp, target: str):
    import ee
    p = cfg["gee"]["products"][product]
    col = (ee.ImageCollection(p["collection"])
           .filterDate(start.strftime("%Y-%m-%dT%H:%M"), end.strftime("%Y-%m-%dT%H:%M"))
           .select(p["bands"]))
    scale = p["scale_m"] if target == "points" else cfg["gee"]["polygon_scale_m"]
    reducer = ee.Reducer.mean()
    if len(p["bands"]) == 1:              # a single band would otherwise be named 'mean'
        reducer = reducer.setOutputs(p["bands"])

    def per_image(img):
        t = img.get("system:time_start")
        return (img.reduceRegions(collection=fc, reducer=reducer, scale=scale)
                .map(lambda f: f.set("t", t).setGeometry(None)))

    return col, col.map(per_image).flatten()


def year_window(cfg, year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = max(pd.Timestamp(cfg["gee"]["period"][0]), pd.Timestamp(f"{year}-01-01"))
    end = min(pd.Timestamp(cfg["gee"]["period"][1]), pd.Timestamp(f"{year + 1}-01-01"))
    return start, end


def fetch(cfg, product, fc, start, end, target) -> pd.DataFrame:
    """Download [start, end); on size or time-out errors split the interval in halves."""
    import ee
    columns = ["id", "t"] + cfg["gee"]["products"][product]["bands"]
    last_error = None
    for attempt in range(1, RETRIES + 1):
        try:
            _, t = table(cfg, product, fc, start, end, target)
            df = ee.data.computeFeatures({"expression": t, "fileFormat": "PANDAS_DATAFRAME"})
            return df.reindex(columns=columns)
        except Exception as err:  # noqa: BLE001  (Earth Engine and network errors)
            last_error = f"{type(err).__name__}: {err}"
            if any(k in last_error.lower() for k in ("memory", "too many", "timed out",
                                                     "limit", "capacity")):
                break
            time.sleep(5 * attempt)
    if (end - start) <= pd.Timedelta(days=MIN_DAYS):
        raise RuntimeError(f"{product} {target} {start}-{end}: {last_error}")
    mid = start + (end - start) / 2
    log.info("%s %s %s to %s split in halves", product, target, start.date(), end.date())
    return pd.concat([fetch(cfg, product, fc, start, mid, target),
                      fetch(cfg, product, fc, mid, end, target)])


def download_year(cfg, product, target, fc, year, d_gee: Path) -> str:
    final = d_gee / f"{product}_{target}_{year}.csv"
    if final.exists():
        return f"{final.name}: already present"
    d_part = d_gee / "_partial"
    d_part.mkdir(exist_ok=True)
    start, end = year_window(cfg, year)
    months = pd.date_range(start.normalize().replace(day=1), end, freq="MS")
    spans = [(max(m, start), min(m + pd.offsets.MonthBegin(1), end)) for m in months]
    spans = [(a, b) for a, b in spans if a < b]

    def job(a, b):
        f = d_part / f"{product}_{target}_{a:%Y%m%d%H}.csv"
        if not f.exists():
            fetch(cfg, product, fc, a, b, target).to_csv(f, index=False)
        return f

    with ThreadPoolExecutor(max_workers=cfg["gee"]["workers"]) as ex:
        parts = [fu.result() for fu in as_completed([ex.submit(job, a, b) for a, b in spans])]
    df = pd.concat([pd.read_csv(f) for f in parts]).dropna(subset=["t"])
    df = df.drop_duplicates(["id", "t"]).sort_values(["t", "id"])
    df.to_csv(final, index=False)
    for f in parts:
        f.unlink()
    return f"{final.name}: {len(df)} rows"


def initialize(cfg, mode: str):
    """Start Earth Engine; on the first use, authenticate in the browser and retry."""
    import ee
    url = HIGH_VOLUME_URL if mode == "direct" else None
    try:
        ee.Initialize(project=cfg["gee"]["project"], url=url)
    except Exception as err:  # noqa: BLE001
        log.info("Earth Engine credentials not available (%s); opening the browser to sign in "
                 "with the Google account of project '%s'", type(err).__name__,
                 cfg["gee"]["project"])
        ee.Authenticate(auth_mode="localhost")
        ee.Initialize(project=cfg["gee"]["project"], url=url)
    log.info("Earth Engine ready (project %s)", cfg["gee"]["project"])
    return ee


def main(a):
    cfg = load_config(a.site)
    y0 = pd.Timestamp(cfg["gee"]["period"][0]).year
    y1 = pd.Timestamp(cfg["gee"]["period"][1]).year - 1
    years = a.years or list(range(y0, y1 + 1))
    products = a.products or list(cfg["gee"]["products"])
    d_gee = path(cfg, "raw_gee")
    log.info("Site %s -> %s | points: %s | years %d-%d", cfg["site"], cfg["paths"]["raw_gee"],
             point_stations(cfg).code.tolist(), years[0], years[-1])
    if a.dry_run:
        todo = [f"{p}_{t}_{y}" for t in a.targets for p in products for y in years
                if not (d_gee / f"{p}_{t}_{y}.csv").exists()]
        log.info("[dry-run] %d files to extract (%s mode), %d already present: %s", len(todo),
                 a.mode, len(a.targets) * len(products) * len(years) - len(todo),
                 ", ".join(todo[:6]) + (" ..." if len(todo) > 6 else ""))
        return
    ee = initialize(cfg, a.mode)
    if a.info:
        for prod in products:
            col = ee.ImageCollection(cfg["gee"]["products"][prod]["collection"])
            r = col.reduceColumns(ee.Reducer.minMax(), ["system:time_start"]).getInfo()
            log.info("%-9s %s -> %s", prod, pd.to_datetime(r["min"], unit="ms"),
                     pd.to_datetime(r["max"], unit="ms"))
        return
    drive_folder = f"{cfg['gee']['drive_folder']}_{cfg['site']}"   # one Drive folder per site
    failed = []
    for target in a.targets:
        todo = [(p, y) for p in products for y in years
                if a.mode == "drive" or not (d_gee / f"{p}_{target}_{y}.csv").exists()]
        log.info("%s: %d product-years to extract", target, len(todo))
        if not todo:
            continue
        fc = features(cfg, target)
        for prod, year in todo:
            name = f"{prod}_{target}_{year}"
            if a.mode == "direct":
                try:
                    log.info(download_year(cfg, prod, target, fc, year, d_gee))
                except Exception as err:  # noqa: BLE001
                    failed.append(name)
                    log.error("%s: %s", name, err)
                continue
            col, t = table(cfg, prod, fc, *year_window(cfg, year), target)
            if col.size().getInfo() == 0:
                log.warning("%s: empty collection, skipped", name)
                continue
            ee.batch.Export.table.toDrive(
                collection=t, description=f"{cfg['site']}_{name}",
                folder=drive_folder, fileNamePrefix=name, fileFormat="CSV",
                selectors=["id", "t"] + cfg["gee"]["products"][prod]["bands"]).start()
            log.info("Export task started: %s", name)
    if a.mode == "drive":
        log.info("When the tasks finish, copy the CSV files from Drive/%s to %s and run "
                 "'python run_pipeline.py --site %s --from 05'", drive_folder, d_gee, cfg["site"])
        return
    if failed:
        log.error("Years with errors (run again to resume from the partial files): %s", failed)
        sys.exit(1)
    missing = [f"{p}_{t}_{y}" for t in a.targets for p in products for y in years
               if not (d_gee / f"{p}_{t}_{y}.csv").exists()]
    if missing:
        log.error("Files still missing: %s", missing)
        sys.exit(1)
    log.info("Earth Engine extraction complete: %d files in %s",
             len(a.targets) * len(products) * len(years), cfg["paths"]["raw_gee"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", help="site in config.yaml (default: PARAMO_SITE or default_site)")
    ap.add_argument("--products", nargs="*", choices=["imerg", "gsmap", "era5land"])
    ap.add_argument("--targets", nargs="*", default=["points", "catchments"],
                    choices=["points", "catchments"])
    ap.add_argument("--years", nargs="*", type=int)
    ap.add_argument("--mode", choices=["direct", "drive"], default="direct")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--info", action="store_true")
    main(ap.parse_args())
