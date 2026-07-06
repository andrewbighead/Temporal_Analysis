import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import MSTL


def deseasonalize_MSTL(series: pd.Series, period_days: list[float]) -> tuple[pd.Series, pd.Series]:
    nan_mask = series.isna()

    # interpolazione per MSTL (non gestisce NaN)
    y_interp = series.interpolate(method="linear")

    # MSTL lavora su scala settimanale: resample + interpolate
    weekly = y_interp.resample("W").mean().interpolate()

    # periodi in campioni SETTIMANALI: giorni / 7
    periods = [int(round(p / 7)) for p in period_days]

    # trend rigido in campioni settimanali: ~2 anni = ~104 settimane -> 105 (dispari)
    res = MSTL(weekly, periods=periods, stl_kwargs={"robust": True, "trend": 105}).fit()

    # res.seasonal: DataFrame (una colonna per periodo) o Series (un solo periodo)
    seasonal_weekly = res.seasonal
    if isinstance(seasonal_weekly, pd.DataFrame):
        seasonal_weekly = seasonal_weekly.sum(axis=1)   # somma di tutte le stagionali

    # riporta la stagionalità settimanale all'indice orario originale
    seasonality = (
        seasonal_weekly
        .reindex(series.index.union(seasonal_weekly.index))
        .interpolate(method="time")
        .reindex(series.index)
    )

    # sottrazione sulla serie ORIGINALE: i NaN restano NaN
    deseasonalized = series - seasonality
    deseasonalized[nan_mask] = np.nan

    return deseasonalized, seasonality


def detrend(series: pd.Series) -> tuple[pd.Series, pd.Series]:

    nan_mask = series.isna()

    # interpolazione per STL (non gestisce NaN)
    y_interp = series.interpolate(method="linear")

    # STL lavora su scala giornaliera: resample + period annuale
    daily = y_interp.resample("D").mean().interpolate()
    stl_trend_daily = STL(daily, period=365, trend=731, robust=True).fit().trend

    # riporta il trend giornaliero sull'indice orario originale
    full_index = series.index.union(stl_trend_daily.index)
    trend = (
        stl_trend_daily
        .reindex(full_index)
        .interpolate(method="linear")
        .reindex(series.index)
    )

    detrended = y_interp - trend
    detrended[nan_mask] = np.nan

    return detrended, trend


def remove_median(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Centra la serie sottraendo la mediana (livello costante).

    Da usare sulle serie senza trend significativo (identificate dalle
    PDF grezze come unimodali): non c'e' un andamento da rimuovere, solo
    un offset di livello. La mediana è robusta a outlier residui.

    Parameters
    ----------
    series : pd.Series
        Serie con DatetimeIndex (puo' contenere NaN).

    Returns
    -------
    centered : pd.Series
        Serie centrata (NaN ripristinati).
    level : pd.Series
        Livello costante rimosso, sull'indice originale (per coerenza
        di firma con detrend, che restituisce trend).
    """
    nan_mask = series.isna()
    med = series.median()

    level = pd.Series(med, index=series.index)
    centered = series - med
    centered[nan_mask] = np.nan

    return centered, level