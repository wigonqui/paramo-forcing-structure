"""Solar geometry (Spencer, 1971) for the time-stamp anchor and hourly PET distribution."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _declination_eot(doy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = 2 * np.pi * (doy - 1) / 365
    dec = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
           + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    eot = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                    - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    return dec, eot


def _cos_zenith(utc: pd.DatetimeIndex, lat: float, lon: float) -> np.ndarray:
    dec, eot = _declination_eot(utc.dayofyear.to_numpy())
    solar_minutes = (utc.hour * 60 + utc.minute).to_numpy() + eot + 4 * lon
    omega = np.radians(solar_minutes / 4 - 180)
    phi = np.radians(lat)
    return np.clip(np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(omega), 0, None)


def diurnal_cos_zenith(start, end, lat: float, lon: float, utc_offset_h: int = -5) -> np.ndarray:
    """Mean cos(zenith) for each minute of the local day (1440 values) over [start, end]."""
    t = pd.date_range(pd.Timestamp(start).floor("D"), pd.Timestamp(end).ceil("D"), freq="1min")
    utc = t + np.timedelta64(int(-utc_offset_h), "h")
    cz = _cos_zenith(utc, lat, lon)
    return pd.Series(cz, index=t.hour * 60 + t.minute).groupby(level=0).mean().to_numpy()


def interval_offset(series: pd.Series, lat: float, lon: float) -> tuple[int, float]:
    """Minutes d such that the value labelled at hour h represents the hour centred at h + d.

    Compares the mean diurnal cycle of a radiation series with extraterrestrial radiation.
    Under the 'end of interval' convention the expected value is d = -30 min.
    """
    s = series.dropna()
    cycle = s.groupby(s.index.hour).mean().reindex(range(24)).to_numpy()
    cz = diurnal_cos_zenith(s.index.min(), s.index.max(), lat, lon)
    best = (-np.inf, 0)
    for d in range(-150, 151, 5):
        ref = [cz[np.arange(h * 60 + d - 30, h * 60 + d + 30) % 1440].mean() for h in range(24)]
        best = max(best, (np.corrcoef(cycle, ref)[0, 1], d))
    return best[1], round(float(best[0]), 4)


def extraterrestrial_hourly(idx: pd.DatetimeIndex, lat: float, lon: float,
                            utc_offset_h: int = -5) -> pd.Series:
    """Mean extraterrestrial radiation (MJ m-2 h-1) of the interval (T-1 h, T], local time."""
    sub = pd.to_timedelta(np.arange(-55, 1, 10), unit="min")
    t = np.repeat(idx.to_numpy(), len(sub)) + np.tile(sub.to_numpy(), len(idx))
    utc = pd.DatetimeIndex(t) + np.timedelta64(int(-utc_offset_h), "h")
    dr = 1 + 0.033 * np.cos(2 * np.pi * utc.dayofyear.to_numpy() / 365)
    ra = 0.0820 * 60 * dr * _cos_zenith(utc, lat, lon)
    return pd.Series(ra.reshape(len(idx), len(sub)).mean(axis=1), index=idx)
