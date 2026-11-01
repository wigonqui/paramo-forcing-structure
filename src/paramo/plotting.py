"""Figure style for publication (Elsevier column widths, sans-serif type, colour-blind safe).

Widths follow Elsevier artwork guidance: single column 90 mm, 1.5 column 140 mm and double
column 190 mm. Figures are exported as 600 dpi PNG and vector PDF. Colours use the
Okabe–Ito palette, which remains distinguishable under common colour-vision deficiencies.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MM = 1 / 25.4
WIDTH_MM = {"single": 90, "onehalf": 140, "double": 190}

OKABE_ITO = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
             "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00",
             "purple": "#CC79A7", "grey": "#7F7F7F"}

STRUCTURE_COLOR = {"GR4H": OKABE_ITO["blue"], "LSTM": OKABE_ITO["vermillion"],
                   "GBM": OKABE_ITO["green"], "GR4H_GBM": OKABE_ITO["purple"],
                   "LSTM_raw": "#B0B0B0", "LSTM_hybrid": "#707070"}
STRUCTURE_LABEL = {"GR4H": "GR4H", "LSTM": "LSTM", "GBM": "GBM", "GR4H_GBM": "GR4H–GBM",
                   "LSTM_raw": "LSTM (no var. adj.)", "LSTM_hybrid": "LSTM hybrid"}
STRUCTURE_MARKER = {"GR4H": "o", "LSTM": "s", "GBM": "^", "GR4H_GBM": "D", "LSTM_raw": "v", "LSTM_hybrid": "P"}
STRUCTURE_ORDER = ["GR4H", "LSTM", "GBM", "GR4H_GBM", "LSTM_raw", "LSTM_hybrid"]

PRODUCT_COLOR = {"gauge": OKABE_ITO["black"], "imerg": OKABE_ITO["blue"],
                 "gsmap": OKABE_ITO["orange"], "era5land": OKABE_ITO["green"],
                 "fusion": OKABE_ITO["purple"]}
PRODUCT_LABEL = {"gauge": "Gauges", "imerg": "IMERG", "gsmap": "GSMaP",
                 "era5land": "ERA5-Land", "fusion": "Fusion"}
VERSION_LABEL = {"raw": "raw", "corr": "corrected", "qm": "QM"}
SCENARIO_ORDER = ["gauge", "fusion_corr", "era5land_corr", "gsmap_corr", "imerg_corr",
                  "fusion_raw", "era5land_raw", "gsmap_raw", "imerg_raw",
                  "fusion_qm", "era5land_qm", "gsmap_qm", "imerg_qm"]

COMPONENT_COLOR = {"forcing": OKABE_ITO["blue"], "structure": OKABE_ITO["orange"],
                   "interaction": "#BDBDBD"}
COMPONENT_LABEL = {"forcing": "Forcing", "structure": "Structure",
                   "interaction": "Interaction"}

Q_UNIT = "m$^3$ s$^{-1}$"           # mathtext: renders with the text font (no missing glyphs)

METRIC_LABEL = {"kge": "KGE′ (hourly)", "kge_daily": "KGE′ (daily)", "nse": "NSE",
                "nse_log": "NSE (log Q)", "kge_sqrt": "KGE′ (√Q)"}


def scenario_label(name: str) -> str:
    if name == "gauge":
        return "Gauges"
    product, version = name.rsplit("_", 1)
    return f"{PRODUCT_LABEL.get(product, product)} {VERSION_LABEL.get(version, version)}"


def scenario_sort(names) -> list[str]:
    rank = {s: i for i, s in enumerate(SCENARIO_ORDER)}
    return sorted(names, key=lambda s: rank.get(s, len(rank)))


def structure_sort(names) -> list[str]:
    rank = {s: i for i, s in enumerate(STRUCTURE_ORDER)}
    return sorted(names, key=lambda s: rank.get(s, len(rank)))


def site_label(cfg: dict) -> str:
    return cfg.get("label", cfg["site"].capitalize())


def setup():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "legend.title_fontsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 600,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.titleweight": "normal",
        "figure.constrained_layout.use": True, "mathtext.default": "regular",
        "hatch.linewidth": 0.5,
    })


def new_figure(width: str = "double", height_mm: float = 80, **kwargs):
    setup()
    return plt.subplots(figsize=(WIDTH_MM[width] * MM, height_mm * MM), **kwargs)


def plain_log_axis(axis):
    """Log axis labelled with plain numbers at 1, 2 and 5 times powers of ten."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    axis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def integer_axis(axis):
    """Ticks on whole numbers only (years, lags in hours)."""
    from matplotlib.ticker import MaxNLocator
    axis.set_major_locator(MaxNLocator(integer=True))


def panel_label(ax, letter: str):
    """Bold panel letter placed as a left-aligned title (never overlaps tick labels)."""
    ax.set_title(f"({letter})", loc="left", fontsize=9, fontweight="bold")


def save(fig, stem: Path):
    """Write <stem>.png (600 dpi) and <stem>.pdf (vector) and close the figure."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def design_schematic(stem: Path, cfgs: list[dict]):
    """Study-design schematic built from the configuration of one or more sites."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fmt = lambda d: pd.Timestamp(d).strftime("%b %Y")
    data_lines = ["FONAG/EPMAPS stations"]
    for c in cfgs:
        st = c["stations"]
        data_lines += [f"{site_label(c)}: Q {', '.join(st['streamflow'])}",
                       f"  P {', '.join(c['forcing'].get('thiessen_gauges') or st['rain_gauges'])}"]
    data_lines += ["IMERG, GSMaP, ERA5-Land", "Copernicus GLO-30 DEM"]
    columns = [
        ("Data", data_lines),
        ("Preparation", ["Hourly QC flags", "Solar time-stamp anchor",
                         "Gauge consistency QC", "Catchment delineation",
                         "Corrected T, Oudin PET", "Common hourly panel"]),
        ("Forcings (9)", ["Rain gauges (Thiessen)", "IMERG V07", "GSMaP v8", "ERA5-Land",
                          "Fusion of the three", "Each raw and ML-corrected",
                          "QM reference (no ML)"]),
        ("Structures (4)", ["GR4H (conceptual)", "LSTM (deep learning)",
                            "GBM (gradient boosting)", "GR4H–GBM (hybrid)"]),
        ("Evaluation", ["Common hourly sample", "Block bootstrap CIs",
                        "Diebold–Mariano + Holm", "Two-way ANOVA of skill",
                        "Sensitivity analyses", "Reversed split sample",
                        "Nested transfer catchment"]),
    ]
    if len(cfgs) > 1:
        columns[-1][1].append("Cross-site synthesis")
    fill = ["#EAF2FA", "#EAF6EF", "#FDF4E3", "#F7ECF4", "#F0F0F0"]
    n_lines = max(len(lines) for _, lines in columns)
    bottom = 5.0
    top = bottom + 5.0 * n_lines + 9.0                  # box height follows the longest column
    y_max = top + 2.0 + 4.0 * len(cfgs)
    fig, ax = new_figure("double", 0.85 * (y_max - 3.0) + 4.0)
    ax.set_xlim(0, 100)
    ax.set_ylim(3.0, y_max)
    ax.axis("off")
    w, gap = 17.0, 3.1
    for i, (title, lines) in enumerate(columns):
        x = 0.8 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, bottom), w, top - bottom,
                                    boxstyle="round,pad=0.3,rounding_size=1.2",
                                    fc=fill[i], ec="#4D4D4D", lw=0.6))
        ax.text(x + w / 2, top - 3.5, title, ha="center", va="center", fontsize=8,
                fontweight="bold")
        for j, line in enumerate(lines):
            ax.text(x + 1.0, top - 9.5 - j * 5.0, line, ha="left", va="center", fontsize=6.5)
        if i < len(columns) - 1:
            mid = (top + bottom) / 2
            ax.add_patch(FancyArrowPatch((x + w + 0.3, mid), (x + w + gap - 0.3, mid),
                                         arrowstyle="-|>", mutation_scale=8, lw=0.8,
                                         color="#4D4D4D"))
    for k, c in enumerate(cfgs):
        per, gr = c["periods"], c["gr4h"]
        ax.text(50, top + 2.5 + 4 * (len(cfgs) - 1 - k),
                (f"{site_label(c)}: calibration {fmt(per['calibration'][0])} – "
                 f"{fmt(per['calibration'][1])}, validation {fmt(per['validation'][0])} – "
                 f"{fmt(per['validation'][1])} ({gr['calibration_catchment']}); transfer to "
                 f"{gr['transfer_catchment']}; repeated with the periods reversed"),
                ha="center", va="center", fontsize=7, color="#333333")
    save(fig, stem)
