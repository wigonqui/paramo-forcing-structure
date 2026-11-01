"""
Stage 00 - Station coordinates and series inventory

Purpose
    Convert station coordinates from the local TMQ grid to geographic and UTM 17S
    coordinates, check for duplicated or implausible positions, and list the hourly
    series available for each station of the selected site.

Tools
    pyproj (coordinate transformation), geopandas (GeoPackage output), pandas.

Inputs
    data/<site>_input/estaciones.txt          station codes, TMQ coordinates, elevation
    data/<site>_input/<code>-*.csv            hourly FONAG/EPMAPS series

Outputs
    data/interim/<site>/stations.gpkg         station layer (TMQ)
    outputs/<site>/tables/00_stations.csv     coordinates in TMQ, WGS84 and UTM 17S
    outputs/<site>/tables/00_series_inventory.csv

Next stage
    01 - Hourly quality control.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer

from paramo.core import find_series_file, get_logger, load_config, path, read_series, read_stations

log = get_logger("00_stations")


def main():
    cfg = load_config()
    log.info("Site: %s", cfg["site"])
    st = read_stations(cfg)
    to_geo = Transformer.from_crs(cfg["crs"]["tmq"], cfg["crs"]["geographic"], always_xy=True)
    to_utm = Transformer.from_crs(cfg["crs"]["tmq"], cfg["crs"]["utm"], always_xy=True)
    st["lon"], st["lat"] = to_geo.transform(st.x_tmq.to_numpy(), st.y_tmq.to_numpy())
    st["x_utm17s"], st["y_utm17s"] = to_utm.transform(st.x_tmq.to_numpy(), st.y_tmq.to_numpy())

    xy = st[["x_tmq", "y_tmq"]].to_numpy()
    dist_km = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)) / 1000
    for i, j in zip(*np.triu_indices(len(st), 1)):
        if dist_km[i, j] < 0.05:
            log.info("Stations %s and %s are co-located (%.0f m apart)", st.code[i],
                     st.code[j], dist_km[i, j] * 1000)
    for col in ("x_tmq", "y_tmq"):
        dup = st[st[col].duplicated(keep=False)]
        if len(dup):
            log.warning("Repeated %s values: %s", col, dup.code.tolist())

    gdf = gpd.GeoDataFrame(st, geometry=gpd.points_from_xy(st.x_tmq, st.y_tmq),
                           crs=cfg["crs"]["tmq"])
    gdf.to_file(path(cfg, "interim") / "stations.gpkg", layer="stations", driver="GPKG")
    st.round({"lon": 6, "lat": 6, "x_utm17s": 1, "y_utm17s": 1}).to_csv(
        path(cfg, "tables") / "00_stations.csv", index=False)

    rows = []
    for code in st.code:
        for variable in cfg["variables"]:
            f = find_series_file(cfg, code, variable)
            if f is None:
                continue
            df = read_series(f)
            v = df["value"]
            rows.append({"code": code, "variable": variable, "file": f.name,
                         "start": v.index.min(), "end": v.index.max(), "hours": len(v),
                         "valid_pct": round(100 * v.notna().mean(), 1),
                         "duplicates": df.attrs.get("duplicates", 0)})
    inv = pd.DataFrame(rows)
    inv.to_csv(path(cfg, "tables") / "00_series_inventory.csv", index=False)
    log.info("Inter-station distances (km):\n%s",
             pd.DataFrame(dist_km.round(1), index=st.code, columns=st.code))
    log.info("Inventory:\n%s", inv.drop(columns="file").to_string(index=False))


if __name__ == "__main__":
    main()
