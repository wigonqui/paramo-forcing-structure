# Data access

The repository contains the code, the configuration and the result tables and figures of the
article. The hydrometeorological station records are **not** redistributed: they belong to the
Empresa Pública Metropolitana de Agua Potable y Saneamiento de Quito (EPMAPS) and the Fondo para
la Protección del Agua (FONAG). All other inputs are public and are downloaded by the pipeline.

| Data | Source | How it enters the pipeline |
|---|---|---|
| Hourly discharge, stage, precipitation, temperature and solar radiation | EPMAPS hydrometeorological data portal, <https://paramh2o.aguaquito.gob.ec/reportes/consultas/> (accessed 12 March 2026) | Downloaded by the user and placed in `data/<site>_input/` (see below) |
| IMERG V07, GSMaP v8, ERA5-Land | Google Earth Engine (`NASA/GPM_L3/IMERG_V07`, `JAXA/GPM_L3/GSMaP/v8/operational`, `ECMWF/ERA5_LAND/HOURLY`; see `config.yaml > gee`) | Stage 04, with the user's own Earth Engine project |
| Copernicus GLO-30 DEM | ESA, <https://doi.org/10.5270/ESA-c5d3d65> | Stage 03 (missing tiles are downloaded automatically) |

## Station records (EPMAPS)

Download the hourly series of the stations below from the portal, for the full available
period, and save them unchanged (CSV as exported) in `data/tambo_input/` and
`data/chalpi_input/`. The exact files used in the article, with their periods and completeness,
are listed in `outputs/<site>/tables/00_series_inventory.csv`.

| Site | Discharge (and stage) | Precipitation | Temperature | Solar radiation |
|---|---|---|---|---|
| Tambo | H20 Tambo 12, H18 Tambo 2A | C01 Maucatambo, P39 Yangahuagra, P40 Tambo 2A | C01 | C01 |
| Chalpi | H13 Chalpi Grande, H42 Chalpi Norte | C16 Guaytaloma, P46 Chalpi Grande, P64 Laguna Encantada, M5022 Control Baños, P74 Mogotes, P75 Laguna Los Patos | M5022 | C16, M5025 La Virgen Papallacta |

File conventions read by `paramo.core`:

- One CSV per station and variable, named `<code>-<name>_<variable>-Horario.csv`
  (e.g. `H13-Chalpi_Grande_Caudal-Horario.csv`). The variable is recognized from the file name
  (`caudal`, `nivel`, `precipit`, `temperatura`, `radiaci`; accents and `#U00f3`-type codes are
  accepted, see `config.yaml > variables`).
- Columns as exported by the portal: date-time `Fecha` (`YYYY/MM/DD HH:MM:SS`, local time UTC−5,
  labelled at the start of the hour), `Valor` and the completeness of the hourly measurements.
- `estaciones.txt` (tab-separated) with columns `Código`, `Estación`, `X`, `Y` (local TMQ grid,
  WGS84) and `Altura` (m a.s.l.). Coordinates of all stations used are given in
  `outputs/<site>/tables/00_stations.csv`.

Documented exclusions (sensor faults) are part of `config.yaml > sites > chalpi >
quality_control > manual_exclusions`, so the same series give the same results.

## What is included in this repository

- `scripts/`, `src/paramo/`, `run_pipeline.py`, `verify.py`, `install_environment.py`, `config.yaml`
- `outputs/<site>/tables[_reversed]/` and `outputs/synthesis/`: all result tables (CSV) used in the
  article, including the variance fractions, bootstrap intervals and sensitivity analyses
- `outputs/<site>/figures[_reversed]/`: the figures of the article and the supplementary material
- `outputs/environment_versions.csv`: exact package versions of the run

Station series, intermediate files (`data/interim`, `data/processed`) and run logs are not
included. Processed series derived from EPMAPS/FONAG records can be shared only with the
permission of these institutions.
