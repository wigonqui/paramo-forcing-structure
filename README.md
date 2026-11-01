# Forcing or structure? Hourly streamflow modeling in two data-scarce Andean páramo catchments

This repository contains the complete workflow of the article *Forcing or structure? Hourly
streamflow modeling in two data-scarce Andean páramo catchments* (González Quilachamin and
Ochoa García, submitted to Hydrological Sciences Journal) on two páramo catchments of the Ecuadorian Andes
monitored by EPMAPS and FONAG: the Tambo River on the eastern flank of
Cotopaxi volcano and the Chalpi Grande River near Papallacta. The question is which decision
controls hourly simulation skill in data-scarce páramo catchments: the precipitation forcing
or the model structure. The same factorial protocol is applied to both catchments, each with
its own stations, periods and nested transfer catchment, and the conclusions are compared in
a final cross-site synthesis.

## Study design

| Element | Tambo | Chalpi |
|---|---|---|
| Calibration catchment | H20 (Tambo 12) | H13 (Chalpi Grande) |
| Nested transfer catchment | H18 (Tambo 2A) | H42 (Chalpi Norte) |
| Thiessen gauges | C01, P39, P40 | C16, P46, P64, M5022 |
| Additional gauges (bias correction only) | – | P74, P75 (from December 2018) |
| Temperature | C01 (lapse rate) | M5022 (lapse rate) |
| Radiation (solar anchor) | C01 | C16, M5025 |
| Warm-up | 2007 | 2011 |
| Calibration / validation (direct) | 2008–2011 / 2012 – Jul 2015 | 2012–2015 / 2016–2018 |
| Calibration / validation (reversed) | 2012 – Jul 2015 / 2008–2011 | 2016–2018 / 2012–2015 |
| Transfer period | 2010–2014 | 2012–2018 |
| Data limit | August 2015 (Cotopaxi unrest) | December 2018 (H13 gap and level shift in 2019) |

Common to both sites:

| Element | Definition |
|---|---|
| Forcings (9) | `gauge` (fixed Thiessen after a consistency control) and IMERG V07, GSMaP v8, ERA5-Land and their fusion, each `raw` and ML-`corr`ected |
| QM reference | `{product}_qm`: the same two-scale quantile mapping without ML; isolates the ML contribution and is not part of the factorial design |
| Structures (4) | `GR4H` (conceptual, airGR), `LSTM` (deep-learning reference), `GBM` (LightGBM with explicit memory), `GR4H_GBM` (GR4H with a gradient-boosting error model) |
| Supplementary | `LSTM_raw` (without variance adjustment) and, optionally, `LSTM_hybrid` |
| Post-processing variants | `GBM_raw`, `GR4H_GBM_raw` (without variance adjustment) and `GR4H_adj` (GR4H with the same adjustment); used only by the sensitivity of stage 12 |

What is compared. The outputs of `LSTM`, `GBM` and `GR4H_GBM` receive an affine variance
adjustment (`paramo.metrics.variance_adjustment`, Eq. 4 of the manuscript): the factor
c = σ_obs/σ_sim is estimated on training data only and bounded to [1, 3], and the series is
scaled around its simulated training mean, so r and the training volume do not change and the
adjustment acts mainly on γ. GR4H does not receive it because it is calibrated on KGE′, which
already contains γ, whereas the learning models minimise squared errors, which shrink variance.
Each structure is therefore evaluated as model + objective function + post-processing. Stage
12 repeats the ANOVA without the adjustment on any structure and with it on all four
(`sensitivity.postprocessing_variants`): forcing dominates in the same 7 of 8 cases in every
variant (supplementary Table S2).

Retrospective forcing. The bias correction uses the product at hours t − 3 to t + 3 and moving
sums centred on t (`bias_correction.lags_h`, `windows_h`; the 24 h sum reaches t + 11), and
IMERG V07 (final run) and ERA5-Land are released with latencies of months and days. The
corrected satellite scenarios are a retrospective (offline) reconstruction, not a real-time
product; an operational version would need causal predictors (lags ≤ 0, backward sums) and
near-real-time products. GR4H, the LSTM and the GBM structures themselves only use forcing up
to the simulated hour.

Why the periods differ between sites. Each site uses the longest window in which discharge
at both gauges and the Thiessen network are reliable, split into two blocks of similar
effective length so that the reversed experiment is a true swap. `python verify.py` reports
the data support of every period (valid discharge, gauge availability, effective years, mean
discharge and precipitation):

| Site | Period | Catchment | Valid Q | Effective years | Gauges valid |
|---|---|---|---|---|---|
| Tambo | calibration 2008–2011 | H20 | 0.76 | 3.0 | 0.93 |
| Tambo | validation 2012 – Jul 2015 | H20 | 0.97 | 3.5 | 0.90 |
| Tambo | transfer 2010–2014 | H18 | 0.95 | 4.7 | 0.94 |
| Chalpi | calibration 2012–2015 | H13 | 0.86 | 3.4 | 0.92 |
| Chalpi | validation 2016–2018 | H13 | 0.83 | 2.5 | 0.89 |
| Chalpi | transfer 2012–2018 | H42 | 0.77 | 5.4 | 0.91 |

Valid Q and effective years are counted after the documented sensor faults are excluded
(see Quality control of nested gauges below). The bias correction of each experiment is
trained only outside its own validation period, and the early-stopping year of the learning
models lies inside its own calibration period.

### Quality control of nested gauges (Chalpi)

Outside four windows, the daily discharge ratio H13/H42 fluctuates around the area ratio
(16.1): median 18.5, between 10 and 30 on 93 % of days. In those windows one gauge stops
responding to rain and the ratio stays at 2–8 or 40–118 for weeks; they are flagged as errors
in stage 01 (`quality_control.manual_exclusions`) and shown in supplementary Fig. S2
(`python scripts/S2_nested_gauges.py`, which prints these statistics):

| Series | Window | Evidence |
|---|---|---|
| H42 | 2013-03-11 – 2013-10-24 | level drift without rain response; ratio 2–8 |
| H13 | 2016-03-04 – 2016-06-07 | drops to a flat ~2.2 m³/s within hours and does not respond to 280 mm of rain in April while H42 does; ratio 2–8, 30-day correlation down to −0.45 |
| H13 | 2018-09-17 – 2018-11-08 | saw-tooth drift with abrupt resets; ratio 40–118 |
| H42 | 2018-10-22 – 2018-11-15 | level stuck for 23 days after an abrupt step |

After the exclusions the daily correlation H13–H42 rises from 0.72 to 0.85 (stage 02,
`02_gauge_relation.csv`).

## Quick start (Windows, Linux or macOS, from the project folder)

Before the first run: (1) download the station records from the EPMAPS portal and place them in
`data/tambo_input/` and `data/chalpi_input/` as described in [DATA.md](DATA.md); (2) set your own
Google Cloud project with Earth Engine access in `config.yaml > gee > project`.

```bash
python install_environment.py --venv --gee-auth   # once: .venv, Python and R packages, Earth Engine
# activate: .venv\Scripts\activate (Windows) | source .venv/bin/activate (Linux/macOS)
python verify.py                                  # environment, configuration, inputs, periods, status
python run_pipeline.py                            # both sites (00-13) and the synthesis (14);
                                                  # missing DEM tiles and Earth Engine years
                                                  # are downloaded automatically
python verify.py --consistency                    # R/Python and bootstrap consistency checks
```

**After changing quality control or the hourly panel of a site** (stages 01–05), the reversed
experiment does not rebuild its own panel: `data/processed/<site>_reversed/hourly_panel.parquet`
is only created once. Copy the updated panel before re-running stage 13, otherwise the reversed
experiment keeps the old data:

```bash
python run_pipeline.py --site chalpi --from 01 --to 12
cp data/processed/chalpi/hourly_panel.parquet data/processed/chalpi_reversed/   # Windows: Copy-Item ... -Force
python run_pipeline.py --site chalpi --only 13
python run_pipeline.py --only 14
```

Check: in `outputs/<site>/tables_reversed/07_forcing_summary.csv` the calibration hours of the
reversed run must equal the validation hours of the direct run (`hours_P_and_Q`).

**After changing stage 10 or the post-processing variants** (stages 06–09 untouched), stages
10–12 are enough; the simulations of the main structures are deterministic and unchanged:

```bash
python run_pipeline.py --site tambo --only 10 11 12
python run_pipeline.py --site tambo --experiment reversed --only 10 11 12
python run_pipeline.py --site chalpi --only 10 11 12
python run_pipeline.py --site chalpi --experiment reversed --only 10 11 12
```

`verify.py` then reports stages 13 and 14 as outdated (an upstream stage ran later) although
their results do not change; `python run_pipeline.py --site <site> --only 13` (reruns the
reversed chain without LSTM, about 17 min per site) and `python run_pipeline.py --only 14`
restore a clean status.

Useful options of `run_pipeline.py`: `--site tambo|chalpi|all`, `--from 06`, `--to 12`,
`--only 11 12`, `--skip-lstm`, `--with-gee`, `--experiment reversed --from 06 --to 12`,
`--dry-run`. Single scripts read the site from `PARAMO_SITE` (default `tambo`), e.g.
`PARAMO_SITE=chalpi python scripts/11_factorial_analysis.py`; in R,
`Rscript scripts/08_gr4h.R --site chalpi` or `SITE <- "chalpi"` before sourcing in RStudio.

Figures of stages 03 and 06 can be redrawn without recomputing:
`python scripts/03_catchments.py --figures-only` and
`python scripts/06_bias_correction.py --figures-only`.

Requirements: Python 3.11–3.14, R ≥ 4.2 with airGR, and an Earth Engine project for stage 04.
Stages 06, 08, 09 and 10 use the cores configured in `config.yaml` (4 by default).

## Downloads

| Data | Tambo | Chalpi |
|---|---|---|
| Copernicus GLO-30 | tile S01 W079 | tiles S01 W079 (shared with Tambo) and S01 W078 |
| Earth Engine (IMERG, GSMaP, ERA5-Land) | extracted by stage 04 at the station pixels and catchment polygons of the site | idem |

Stage 03 downloads only the tiles missing from `data/raw/dem_tiles/`. Stage 04 runs
automatically whenever a product, target or year is missing in `data/raw/<site>/gee/`,
downloads only the missing files (an interrupted run resumes from the monthly partial files)
and, the first time, opens the browser to sign in to Earth Engine. Chalpi is extracted from
2005 at the pixels of its rain gauges and weather stations (126 files). Alternative for slow
connections: `python scripts/04_satellite_extraction.py --site chalpi --mode drive`, then copy
the files of the Drive folder `paramo_gee_chalpi` to `data/raw/chalpi/gee/`.

## Stages

| Stage | Script | Content |
|---|---|---|
| 00 | `00_stations.py` | Station coordinates (TMQ, WGS84, UTM 17S) and series inventory |
| 01 | `01_quality_control.py` | Hourly quality flags; nothing is deleted |
| 02 | `02_diagnostics.py` | Rating curve (when stage records exist), homogeneity, double mass, nested gauges |
| 03 | `03_catchments.py` | Copernicus GLO-30, delineation, attributes, study-area map |
| 04 | `04_satellite_extraction.py` | IMERG, GSMaP and ERA5-Land from Earth Engine (resumable) |
| 05 | `05_harmonization.py` | Hourly panel, solar time-stamp anchor, time offsets |
| 06 | `06_bias_correction.py` | ML correction, fusion and QM reference with leave-one-gauge-out validation |
| 07 | `07_catchment_forcing.py` | Gauge consistency control, forcing scenarios, temperature, PET, hypsometry |
| 08 | `08_gr4h.R` | GR4H calibration, validation and transfer (parallel scenarios) |
| 09 | `09_lstm.py` | LSTM benchmark with variance adjustment (parallel scenarios) |
| 10 | `10_gbm.py` | GBM and GR4H_GBM; unadjusted series (`*_raw`) and `GR4H_adj` for stage 12 |
| 11 | `11_factorial_analysis.py` | Common sample, block bootstrap, Diebold–Mariano, ANOVA of skill, SHAP |
| 12 | `12_sensitivity.py` | Sensitivity to metric, bootstrap block, discharge observation error, post-processing (variance adjustment) and forcing set (without raw products) |
| 13 | `13_split_sample.py` | Reversed split-sample run of stages 06–12 and comparison of findings |
| 14 | `14_synthesis.py` | Cross-site comparison of variance fractions, rankings and findings |
| S2 | `S2_nested_gauges.py` | Supplementary Fig. S2: nested-gauge consistency check of Chalpi (after stages 01, 03 and 07) |

Shared library (`src/paramo/`): `core` (configuration, sites, experiments, I/O), `metrics`,
`stats`, `solar`, `gbm`, `evaluation`, `plotting` and `pipeline`.

## Folders

```
data/<site>_input/            station records and estaciones.txt (FONAG/EPMAPS)
data/raw/dem_tiles/           Copernicus tiles shared by both sites
data/raw/<site>/{gee,dem}/    Earth Engine extractions and site DEM products
data/interim/<site>/          QC series, stations and catchments
data/processed/<site>[_reversed]/   hourly panel, forcing and simulations
outputs/<site>/tables[_reversed]/, outputs/<site>/figures[_reversed]/
outputs/synthesis/            stage 14
outputs/logs/, outputs/.runs/ run logs and run records
```

## Conventions

- Working CRS: TMQ (WGS84, central meridian −78.5°); Earth Engine geometry in EPSG:4326.
- Time: local UTC−5. In the hourly panel the value at T refers to the interval (T−1 h, T].
  FONAG/EPMAPS records are labelled at the start of the hour and ERA5-Land (Earth Engine) at
  the end; both conventions are verified against extraterrestrial radiation in stage 05
  for each site.
- Quality flags: 0 ok, 1 suspect, 2 error, 3 missing; `value_qc` excludes 2 and 3.
- Lags: in stage 05 a positive best lag (`best_lag_h`) means that the product records rain
  before the gauge; in stage 06, predictor `p_t+k` is the product k hours after the hour being
  corrected (positive k uses later hours).
- Evaluation: hours with missing precipitation in any scenario, plus the next 72 h, are
  excluded for every simulation; metric definitions are identical in R and Python
  (KGE′ of Kling et al., 2012).
- Configuration: common settings first, then `sites.<site>` overrides what differs, then
  `sites.<site>.experiments.<name>` overrides the site. `PARAMO_SITE` and
  `PARAMO_EXPERIMENT` select them; an experiment writes to the `*_<name>` folders.
- Figures: Elsevier column widths (90, 140, 190 mm), sans-serif type of 7–9 pt, Okabe–Ito
  colour-blind-safe palette with fixed colours per structure and product, 600 dpi PNG and
  vector PDF.

## Main outputs for the manuscript

| Figure | File |
|---|---|
| Study area (each site) | `outputs/<site>/figures/03_study_area` |
| Study design (both sites) | `outputs/synthesis/14_study_design` |
| Precipitation exceedance (raw, QM, ML-corrected) | `outputs/<site>/figures/06_exceedance` |
| Skill matrix with 95 % intervals | `outputs/<site>/figures/11_skill_matrix` |
| Differences against GR4H | `outputs/<site>/figures/11_differences_gr4h` |
| ML correction against QM | `outputs/<site>/figures/11_ml_vs_qm` |
| Variance fractions of skill | `outputs/<site>/figures/11_skill_anova` |
| Sensitivity by metric | `outputs/<site>/figures/12_anova_by_metric` |
| Direct against reversed experiment | `outputs/<site>/figures/13_direct_vs_reversed` |
| Cross-site comparison | `outputs/synthesis/14_sites_comparison` |
| SHAP of the error model (direct experiment) | `outputs/<site>/figures/11_shap` |
| SHAP of the error model, reversed experiment (supplementary Fig. S1) | `outputs/<site>/figures_reversed/11_shap` |

| Table | File |
|---|---|
| Variance fractions (Table 5) | `outputs/<site>/tables[_reversed]/11_skill_anova.csv`; observation error: `12_obs_error_anova.csv` |
| LSTM with and without variance adjustment (supplementary Table S1) | `11_metrics.csv`, models `LSTM` and `LSTM_raw` |
| Post-processing sensitivity (supplementary Table S2) | `outputs/<site>/tables[_reversed]/12_postprocessing_anova.csv` and `12_postprocessing_skill.csv` (variants `reported`, `none`, `all`) |
| Forcing-set sensitivity (section 4.6, supplementary Table S3) | `outputs/<site>/tables[_reversed]/12_forcing_subset_anova.csv` (subsets `all`, 9 × 4, and `without_raw`, gauges + 4 corrected products, 5 × 4; `sensitivity.forcing_subsets`) |

Stage 12 uses 500 bootstrap replicates and stage 11 uses 1000, so the intervals can differ by
about 0.01 between Table S2 and Table 5; the point estimates are identical.
Supplementary Fig. S2 (daily H13 and H42 discharge, their ratio and the four excluded windows)
is drawn by `python scripts/S2_nested_gauges.py` (`outputs/chalpi/figures/S2_nested_gauges`).

Supplementary material: `01_annual_completeness`, `02_rating_curve` (Tambo),
`02_double_mass`, `02_nested_gauges`, `05_diurnal_precipitation`, `07_annual_balance`,
`12_obs_error_best` and the `*_reversed` figures of each site.

## Known limitations

One calibration catchment and one nested transfer catchment per site, with no gaugings;
about 7 years of usable discharge in each calibration catchment; the Tambo transfer gauge
(H18) drifts under a single rating curve; Chalpi has no stage records to check its rating
curve, and its 2019 discontinuity limits the record to 2018; four sensor faults of the Chalpi
gauges are excluded by documented rules (`quality_control.manual_exclusions`, see above), and
H42 shows a strongly damped response in 2012–2014 that should be checked with EPMAPS; the
diurnal signal learned by the GR4H error model in Tambo appears only when it is trained on
2008–2011 (direct experiment) and is not interpreted as a physical process; with some
satellite forcings (13 of 48 calibrations, mostly raw or QM) GR4H converges to extreme
parameters (X1 < 25 mm, X3 > 500 mm or < 5 mm, X4 > 20 h); bias correction learned at gauge
pixels and applied to catchment means; extreme hourly peaks underestimated after correction;
in Chalpi, hourly discharge at H13 is one third less variable in 2016–2018 than in 2012–2015
(CV 0.63 vs 0.93) while gauge rainfall is not less variable and H42 changes much less (CV 0.93
vs 1.01); without gaugings this change cannot be attributed to hydrology, the hydraulic
control or the sensor, and it is the case in which structure weighs as much as forcing; the
corrected satellite scenarios are a retrospective reconstruction (see Study design); the
learning structures are compared together with their variance adjustment (Table S2 shows the
conclusions without it); single-catchment LSTM without hyperparameter search (deliberately, to reproduce the data
situation of a local water utility); constant lapse rate (−6 °C km⁻¹) and Oudin PET,
identical in every scenario.

## Data

Station records belong to EPMAPS and FONAG and are **not** included; they can be downloaded from
the EPMAPS hydrometeorological data portal (<https://paramh2o.aguaquito.gob.ec/reportes/consultas/>).
IMERG, GSMaP, ERA5-Land and Copernicus GLO-30 are public and are downloaded by stages 03 and 04.
[DATA.md](DATA.md) lists the stations, the file conventions and what this repository contains.

## Reproducibility

Software versions of the published run are in `outputs/environment_versions.csv` (Python 3.14,
LightGBM 4.7.0, PyTorch 2.14.1, R airGR 1.7.8) and `requirements-lock.txt`. All stages use fixed
seeds; the LSTM (stage 09) is an ensemble of three seeds and may differ slightly with other
PyTorch versions or hardware. Re-running stages 10–14 from the same simulations reproduces the
published tables exactly.

## How to cite

Please cite the article and the archived version of this code (see `CITATION.cff`):

> González Quilachamin, W., Ochoa García, S., 2026. Forcing or structure? Hourly streamflow modeling in
> two data-scarce Andean páramo catchments. Submitted to Hydrological Sciences Journal.

> González Quilachamin, W., Ochoa García, S., 2026. paramo-forcing-structure: processing and modeling
> pipeline (v1.0.0). Zenodo. https://doi.org/[Zenodo DOI]

## License

Code: MIT (see `LICENSE`). Station data and third-party software keep their own terms (see
`DATA.md`).
