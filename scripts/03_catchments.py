"""
Stage 03 - DEM, catchments and attributes

Purpose
    Obtain the Copernicus GLO-30 DEM for the site, delineate the catchments of its streamflow
    gauges, confirm that the transfer catchment is nested in the calibration catchment, assign
    stations to catchments and compute the physiographic attributes used later.

Methods
    D8 flow routing after pit filling, depression filling and flat resolution; outlets
    snapped to the stream network; perimeter and Gravelius index on a contour simplified by
    one cell (raster contours inflate the perimeter); HAND; area above the glacier elevation
    threshold. A catchment touching the DEM border triggers a warning. DEM tiles are cached in
    data/raw/dem_tiles and shared by all sites, so a tile already downloaded for one site is
    not downloaded again.

Tools
    rasterio (download, mosaic, reprojection), pysheds (hydrological conditioning and
    delineation), geopandas/shapely, matplotlib; Natural Earth country boundaries for the
    locator inset.

Inputs
    data/interim/<site>/stations.gpkg (stage 00); Copernicus GLO-30 tiles (public AWS bucket)

Outputs
    data/raw/<site>/dem/dem_tmq.tif and derived rasters
    data/interim/<site>/catchments.gpkg, catchments_wgs84.geojson
    outputs/<site>/tables/03_catchments.csv, 03_station_membership.csv
    outputs/<site>/figures/03_study_area.(png|pdf)

Usage
    python scripts/03_catchments.py                build the DEM (cached tiles) and delineate
    python scripts/03_catchments.py --no-download  use data/raw/<site>/dem/dem_tmq.tif
    python scripts/03_catchments.py --figures-only redraw the map only

Next stage
    04 - Earth Engine extraction of satellite and reanalysis forcing.
"""
import argparse
import math
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib import patheffects
from matplotlib.colors import LightSource, ListedColormap
from rasterio.features import shapes
from rasterio.merge import merge
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely.geometry import LineString, Point, shape
from shapely.ops import unary_union

from paramo import plotting as pl
from paramo.core import get_logger, load_config, path

log = get_logger("03_catchments")
DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)


def tile_name(lat: int, lon: int) -> str:
    ns = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}"
    ew = f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}"
    return f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"


def required_tiles(cfg) -> list[str]:
    lon0, lat0, lon1, lat1 = cfg["dem"]["bbox_geographic"]
    return [tile_name(lat, lon) for lat in range(math.floor(lat0), math.floor(lat1) + 1)
            for lon in range(math.floor(lon0), math.floor(lon1) + 1)]


def download_dem(cfg, d_dem: Path) -> Path:
    lon0, lat0, lon1, lat1 = cfg["dem"]["bbox_geographic"]
    d_tiles = path(cfg, "dem_tiles")
    files = []
    for name in required_tiles(cfg):
        f = d_tiles / f"{name}.tif"
        if f.exists():
            log.info("Tile %s found in %s", name, cfg["paths"]["dem_tiles"])
        else:
            log.info("Downloading %s", name)
            urllib.request.urlretrieve(f"{cfg['dem']['base_url']}/{name}/{name}.tif", f)
        files.append(f)
    sources = [rasterio.open(f) for f in files]
    mosaic, transform = merge(sources, bounds=(lon0, lat0, lon1, lat1))
    profile = sources[0].profile | {"height": mosaic.shape[1], "width": mosaic.shape[2],
                                    "transform": transform, "driver": "GTiff",
                                    "compress": "deflate", "tiled": False}
    for key in ("blockxsize", "blockysize"):
        profile.pop(key, None)
    for s in sources:
        s.close()
    out = d_dem / "glo30_mosaic.tif"
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(mosaic)
    return out


def reproject_tmq(cfg, src_path: Path, d_dem: Path) -> Path:
    res = cfg["dem"]["resolution_m"]
    out = d_dem / "dem_tmq.tif"
    with rasterio.open(src_path) as src:
        transform, w, h = calculate_default_transform(src.crs, cfg["crs"]["tmq"], src.width,
                                                      src.height, *src.bounds, resolution=res)
        profile = src.profile | {"crs": cfg["crs"]["tmq"], "transform": transform, "width": w,
                                 "height": h, "dtype": "float32", "nodata": -9999.0,
                                 "compress": "deflate", "tiled": False}
        for key in ("blockxsize", "blockysize"):
            profile.pop(key, None)
        with rasterio.open(out, "w", **profile) as dst:
            reproject(rasterio.band(src, 1), rasterio.band(dst, 1), dst_nodata=-9999.0,
                      resampling=Resampling.bilinear)
    return out


def write_raster(arr, ref: Path, out: Path, dtype="float32", nodata=-9999):
    with rasterio.open(ref) as r:
        profile = r.profile | {"dtype": dtype, "nodata": nodata, "compress": "deflate",
                               "tiled": False}
    for k in ("blockxsize", "blockysize"):
        profile.pop(k, None)
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(np.asarray(arr).astype(dtype), 1)


def condition(dem_path: Path):
    if not hasattr(np, "in1d"):      # pysheds 0.5 still calls np.in1d (removed in NumPy 2.4)
        np.in1d = np.isin
    from pysheds.grid import Grid
    grid = Grid.from_raster(str(dem_path))
    dem = grid.read_raster(str(dem_path))
    dem_c = grid.resolve_flats(grid.fill_depressions(grid.fill_pits(dem)))
    fdir = grid.flowdir(dem_c, dirmap=DIRMAP)
    acc = grid.accumulation(fdir, dirmap=DIRMAP)
    return grid, dem, dem_c, fdir, acc


def attributes(name, geom, mask, dem, hand, res, stream_mask, outlet_z, glacier_z) -> dict:
    z = dem[mask]
    gy, gx = np.gradient(np.where(mask, dem, np.nan), res)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))[mask]
    aspect = (np.degrees(np.arctan2(-gx, gy)) % 360)[mask]
    area = geom.area / 1e6
    perimeter = geom.simplify(res, preserve_topology=True).length / 1000
    return {"catchment": name, "area_km2": round(area, 2), "perimeter_km": round(perimeter, 2),
            "gravelius": round(0.28 * perimeter / math.sqrt(area), 3),
            "outlet_z_dem": round(float(outlet_z), 1), "z_min": round(float(np.nanmin(z)), 1),
            "z_mean": round(float(np.nanmean(z)), 1), "z_max": round(float(np.nanmax(z)), 1),
            "z_p10": round(float(np.nanpercentile(z, 10)), 1),
            "z_p90": round(float(np.nanpercentile(z, 90)), 1),
            "area_above_glacier_z_km2": round(float(np.sum(z > glacier_z)) * res ** 2 / 1e6, 2),
            "fraction_above_glacier_z": round(float(np.mean(z > glacier_z)), 4),
            "mean_slope_deg": round(float(np.nanmean(slope)), 2),
            "fraction_east_facing": round(float(np.mean((aspect > 45) & (aspect <= 135))), 3),
            "mean_hand_m": round(float(np.nanmean(hand[mask])), 2),
            "drainage_density_km_km2": round(stream_mask[mask].sum() * res / 1000 / area, 3)}


def location_inset(ax, lon: float, lat: float, cfg):
    """Small locator map of Ecuador (Natural Earth 1:50m, cached in data/raw/naturalearth).
    Skipped with a warning when the file cannot be obtained."""
    f = path(cfg, "natural_earth") / "ne_50m_admin_0_countries.zip"
    try:
        if not f.exists():
            urllib.request.urlretrieve(cfg["dem"]["countries_url"], f)
        countries = gpd.read_file(f)
    except Exception as err:  # noqa: BLE001
        log.warning("Location inset skipped (country boundaries unavailable: %s)", err)
        return
    name_col = next(c for c in ("ADMIN", "NAME", "admin", "name") if c in countries)
    inset = ax.inset_axes([0.0, 0.72, 0.30, 0.28])
    countries.plot(ax=inset, color="white", edgecolor="#7F7F7F", linewidth=0.3)
    countries[countries[name_col] == "Ecuador"].plot(ax=inset, color="#D9D9D9",
                                                     edgecolor="black", linewidth=0.5)
    inset.plot(lon, lat, marker="s", markersize=4, color=pl.OKABE_ITO["vermillion"],
               markeredgecolor="black", markeredgewidth=0.4)
    inset.text(-78.5, -3.2, "Ecuador", ha="center", fontsize=6)
    inset.set(xlim=(-81.5, -74.5), ylim=(-5.5, 2.0), xticks=[], yticks=[], xlabel="",
              ylabel="", aspect="equal")
    inset.set_facecolor("#EAF2FA")
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)


def study_area_map(dem_path, catchments, streams, stations, cfg, stem):
    with rasterio.open(dem_path) as r:
        z = r.read(1, masked=True).astype(float)
        left, bottom, right, top = r.bounds
        res = r.res[0]
    # window: catchments and every station used by the site, with a margin
    st = cfg["stations"]
    used = stations[stations.code.isin(st["streamflow"] + st["rain_gauges"] + st["weather"])]
    b = catchments.total_bounds
    pad = 2000
    x0, x1 = min(b[0], used.x_tmq.min()) - pad, max(b[2], used.x_tmq.max()) + pad
    y0, y1 = min(b[1], used.y_tmq.min()) - pad, max(b[3], used.y_tmq.max()) + pad
    # colour range from the mapped window only, not from the whole DEM
    rows = slice(max(0, int((top - y1) / res)), max(0, int((top - y0) / res)))
    cols = slice(max(0, int((x0 - left) / res)), max(0, int((x1 - left) / res)))
    window = z[rows, cols].compressed()
    if window.size == 0:
        window = z.compressed()
    aspect = (y1 - y0) / (x1 - x0)
    fig, ax = pl.new_figure("onehalf", float(np.clip(95 * aspect + 30, 80, 180)))
    shade = LightSource(azdeg=315, altdeg=45).hillshade(z.filled(np.nanmin(z)), vert_exag=2,
                                                        dx=res, dy=res)
    extent = (left / 1000, right / 1000, bottom / 1000, top / 1000)
    ax.imshow(shade, extent=extent, cmap="gray", vmin=0, vmax=1)
    terrain = ListedColormap(plt.get_cmap("terrain")(np.linspace(0.25, 1.0, 256)))
    im = ax.imshow(z, extent=extent, cmap=terrain, alpha=0.6,
                   vmin=float(np.percentile(window, 1)), vmax=float(np.percentile(window, 99.9)))
    for geom in streams.geometry:
        xs, ys = geom.xy
        ax.plot(np.asarray(xs) / 1000, np.asarray(ys) / 1000, color="#2C7FB8", lw=0.5)
    styles = {cfg["gr4h"]["calibration_catchment"]: "black",
              cfg["gr4h"]["transfer_catchment"]: pl.OKABE_ITO["vermillion"]}
    for _, c in catchments.sort_values("area_km2", ascending=False).iterrows():
        x, y = c.geometry.exterior.xy if c.geometry.geom_type == "Polygon" else \
            max(c.geometry.geoms, key=lambda g: g.area).exterior.xy
        ax.plot(np.asarray(x) / 1000, np.asarray(y) / 1000, color=styles.get(c.code, "black"),
                lw=1.4, label=f"{c.code} ({c.area_km2:.1f} km²)")
    kind = {code: "Other station" for code in stations.code}
    kind |= {code: "Weather station" for code in st["weather"]}
    kind |= {code: "Rain gauge" for code in st["rain_gauges"]}
    kind |= {code: "Streamflow gauge" for code in st["streamflow"]}
    marker = {"Streamflow gauge": ("^", pl.OKABE_ITO["blue"], 7),   # drawn above co-located
              "Rain gauge": ("o", pl.OKABE_ITO["orange"], 6),       # rain gauges
              "Weather station": ("D", pl.OKABE_ITO["green"], 6),
              "Other station": ("s", "#BFBFBF", 5)}
    offsets = cfg.get("map", {}).get("label_offsets", {})
    halo = [patheffects.withStroke(linewidth=2, foreground="white")]
    inside = (stations.x_tmq.between(x0, x1)) & (stations.y_tmq.between(y0, y1))
    for k, (m, color, z) in marker.items():
        sub = stations[(stations.code.map(kind) == k) & inside]
        if len(sub):
            ax.scatter(sub.x_tmq / 1000, sub.y_tmq / 1000, marker=m, s=18 if m in "sD" else 28,
                       c=color, edgecolors="black", linewidths=0.5, zorder=z, label=k)
    for _, s in stations[inside].iterrows():
        dx, dy = offsets.get(s.code, (6, 4))
        ax.annotate(s.code, (s.x_tmq / 1000, s.y_tmq / 1000), xytext=(dx, dy),
                    textcoords="offset points", fontsize=7, fontweight="bold",
                    ha="right" if dx < 0 else "left", path_effects=halo, zorder=8)
    ax.set_xlim(x0 / 1000, x1 / 1000)
    ax.set_ylim(y0 / 1000, y1 / 1000)
    ax.set_aspect("equal")
    ax.set(xlabel="Easting (km, TMQ)", ylabel="Northing (km, TMQ)")
    ax.ticklabel_format(useOffset=False, style="plain")
    bar = 5 if (x1 - x0) > 15000 else 2
    sx, sy = x0 / 1000 + 1.5, y0 / 1000 + 1.5               # scale bar
    ax.plot([sx, sx + bar], [sy, sy], color="black", lw=2)
    ax.text(sx + bar / 2, sy + 0.4, f"{bar} km", ha="center", va="bottom", fontsize=7,
            path_effects=halo)
    nx, ny = x1 / 1000 - 1.5, y1 / 1000 - 3.5               # north arrow
    ax.annotate("N", xy=(nx, ny + 2.0), xytext=(nx, ny), ha="center", va="center",
                fontsize=8, fontweight="bold", path_effects=halo,
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1))
    centre = catchments.to_crs(cfg["crs"]["geographic"]).geometry.union_all().centroid
    location_inset(ax, centre.x, centre.y, cfg)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2 if len(labels) <= 4 else 3)
    fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02).set_label("Elevation (m a.s.l.)")
    pl.save(fig, stem)


def figures_only(cfg):
    """Redraw the study-area map from the outputs of a previous full run."""
    d_int = path(cfg, "interim")
    stations = gpd.read_file(d_int / "stations.gpkg", layer="stations")
    catchments = gpd.read_file(d_int / "catchments.gpkg", layer="catchments")
    streams = gpd.read_file(d_int / "catchments.gpkg", layer="streams")
    table = pd.read_csv(path(cfg, "tables") / "03_catchments.csv")
    catchments = catchments.merge(table[["catchment", "area_km2"]], left_on="code",
                                  right_on="catchment")
    study_area_map(path(cfg, "raw_dem") / "dem_tmq.tif", catchments, streams, stations, cfg,
                   path(cfg, "figures") / "03_study_area")


def main(no_download: bool, only_figures: bool):
    cfg = load_config()
    if only_figures:
        figures_only(cfg)
        return
    d_dem, d_int = path(cfg, "raw_dem"), path(cfg, "interim")
    d_tab, d_fig = path(cfg, "tables"), path(cfg, "figures")
    dem_tmq = d_dem / "dem_tmq.tif"
    if not no_download:
        dem_tmq = reproject_tmq(cfg, download_dem(cfg, d_dem), d_dem)
    if not dem_tmq.exists():
        raise FileNotFoundError(dem_tmq)

    stations = gpd.read_file(d_int / "stations.gpkg", layer="stations")
    grid, dem, dem_c, fdir, acc = condition(dem_tmq)
    res = abs(grid.affine.a)
    thr = cfg["dem"]["stream_threshold_cells"]
    hand = grid.compute_hand(fdir, dem_c, acc > cfg["dem"]["hand_threshold_cells"], dirmap=DIRMAP)
    for arr, name, dtype in ((fdir, "fdir_tmq", "int16"), (acc, "acc_tmq", "float32"),
                             (hand, "hand_tmq", "float32")):
        write_raster(arr, dem_tmq, d_dem / f"{name}.tif", dtype)
    with rasterio.open(dem_tmq) as r:
        transform = r.transform
    dem_np = np.asarray(dem, float)
    dem_np[dem_np <= -9999] = np.nan

    rows, polys, outlets, masks = [], [], [], {}
    for code in cfg["stations"]["streamflow"]:
        s = stations[stations.code == code].iloc[0]
        xs, ys = grid.snap_to_mask(acc > thr, (s.x_tmq, s.y_tmq))
        dist = math.hypot(xs - s.x_tmq, ys - s.y_tmq)
        if dist > cfg["dem"]["snap_radius_m"]:
            log.warning("%s snapped %.0f m from its coordinates; check the outlet", code, dist)
        catch = np.asarray(grid.catchment(x=xs, y=ys, fdir=fdir, dirmap=DIRMAP,
                                          xytype="coordinate")).astype(bool)
        if catch[0].any() or catch[-1].any() or catch[:, 0].any() or catch[:, -1].any():
            log.warning("Catchment %s touches the DEM border; enlarge dem.bbox_geographic", code)
        masks[code] = catch
        geom = unary_union([shape(g) for g, v in shapes(catch.astype("uint8"), mask=catch,
                                                         transform=transform) if v == 1])
        row, col = rasterio.transform.rowcol(transform, xs, ys)
        rows.append(attributes(code, geom, catch, dem_np, np.asarray(hand, float), res,
                               np.asarray(acc) > thr, dem_np[row, col],
                               cfg["dem"]["glacier_elevation_m"])
                    | {"station_elevation_m": s.elevation_m, "snap_distance_m": round(dist, 1)})
        polys.append({"code": code, "geometry": geom})
        outlets.append({"code": code, "snap_distance_m": dist, "geometry": Point(xs, ys)})

    catchments = gpd.GeoDataFrame(polys, crs=cfg["crs"]["tmq"])
    table = pd.DataFrame(rows)
    up, down = cfg["stations"]["streamflow"]
    nested = (masks[up] & masks[down]).sum() / masks[up].sum()
    table["fraction_nested"] = round(float(nested), 3)
    area = table.set_index("catchment").area_km2
    log.info("Fraction of %s inside %s: %.3f | area ratio %.2f", up, down, nested,
             area[down] / area[up])
    if nested < 0.95:
        log.warning("%s is not nested in %s (fraction %.2f): check outlets and DEM", up, down,
                    nested)
    network = grid.extract_river_network(fdir, acc > thr * 5, dirmap=DIRMAP)
    streams = gpd.GeoDataFrame(geometry=[LineString(f["geometry"]["coordinates"])
                                         for f in network["features"]], crs=cfg["crs"]["tmq"])
    catchments.to_file(d_int / "catchments.gpkg", layer="catchments", driver="GPKG")
    streams.to_file(d_int / "catchments.gpkg", layer="streams", driver="GPKG")
    gpd.GeoDataFrame(outlets, crs=cfg["crs"]["tmq"]).to_file(
        d_int / "catchments.gpkg", layer="outlets", driver="GPKG")
    catchments.to_crs(cfg["crs"]["geographic"]).to_file(d_int / "catchments_wgs84.geojson",
                                                        driver="GeoJSON")
    table.to_csv(d_tab / "03_catchments.csv", index=False)

    buffered = catchments[["code", "geometry"]].rename(columns={"code": "catchment"})
    buffered["geometry"] = buffered.buffer(2 * res)      # outlets lie on the boundary
    member = gpd.sjoin(stations, buffered, how="left", predicate="within")
    member = member.groupby("code")["catchment"].apply(lambda s: ",".join(s.dropna()))
    member.reset_index().to_csv(d_tab / "03_station_membership.csv", index=False)
    log.info("Attributes:\n%s", table.T.to_string())
    log.info("Station membership:\n%s", member.to_string())

    catchments = catchments.merge(table[["catchment", "area_km2"]],
                                  left_on="code", right_on="catchment")
    study_area_map(dem_tmq, catchments, streams, stations, cfg, d_fig / "03_study_area")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--figures-only", action="store_true",
                    help="redraw the map from existing outputs (no download or delineation)")
    a = ap.parse_args()
    main(a.no_download, a.figures_only)
