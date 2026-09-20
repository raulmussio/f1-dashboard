"""Analisis de ritmo: filtrado de vueltas, aire limpio, correccion por combustible.

Modulo puro: solo pandas/numpy. No importa Streamlit ni dibuja nada.
Todas las funciones reciben y devuelven DataFrames.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Consumo tipico: el coche se aligera y gana ~0.035 s/vuelta. Es un supuesto
# del modelo, no un dato medido; la UI debe mostrarlo como tal.
DEFAULT_FUEL_EFFECT_S_PER_LAP = 0.035

# Umbral por defecto de aire limpio (segundos al coche de adelante).
DEFAULT_CLEAN_AIR_GAP = 1.5


def filter_laps(
    laps: pd.DataFrame,
    *,
    exclude_pit: bool = True,
    exclude_dirty_status: bool = True,
    exclude_inaccurate: bool = True,
    max_percent_off_best: float | None = 107.0,
) -> pd.DataFrame:
    """Devuelve solo las vueltas utilizables para medir ritmo.

    - `exclude_pit`: descarta vueltas de entrada y salida de boxes.
    - `exclude_dirty_status`: descarta vueltas con bandera amarilla, SC, VSC o roja.
    - `exclude_inaccurate`: usa la marca `IsAccurate` de FastF1.
    - `max_percent_off_best`: descarta vueltas por encima de ese % del mejor
      tiempo de la sesion (107 % por defecto). `None` lo desactiva.
    """
    if laps.empty:
        return laps

    mask = laps["LapTimeSeconds"].notna()
    if exclude_pit:
        mask &= ~laps["IsPitIn"].fillna(False) & ~laps["IsPitOut"].fillna(False)
    if exclude_dirty_status and "HasDirtyTrackStatus" in laps.columns:
        mask &= ~laps["HasDirtyTrackStatus"].fillna(False)
    if exclude_inaccurate and "IsAccurate" in laps.columns:
        mask &= laps["IsAccurate"].fillna(False).astype(bool)

    filtered = laps.loc[mask].copy()
    if max_percent_off_best is not None and not filtered.empty:
        best = filtered["LapTimeSeconds"].min()
        filtered = filtered.loc[filtered["LapTimeSeconds"] <= best * max_percent_off_best / 100.0]
    return filtered.reset_index(drop=True)


def gap_to_car_ahead(laps: pd.DataFrame) -> pd.DataFrame:
    """Agrega `GapAheadSeconds`: distancia temporal al coche de adelante en esa vuelta.

    Se calcula con el instante en que cada piloto cruza la linea (`SessionTimeSeconds`)
    dentro de cada numero de vuelta. El lider recibe `inf` porque va en aire limpio
    por definicion.
    """
    if laps.empty or "SessionTimeSeconds" not in laps.columns:
        return laps.assign(GapAheadSeconds=np.nan)

    frame = laps.copy()
    frame["GapAheadSeconds"] = np.nan
    for lap_number, group in frame.groupby("LapNumber", sort=False):
        ordered = group.sort_values("SessionTimeSeconds")
        gaps = ordered["SessionTimeSeconds"].diff()
        # El primero de la vuelta no tiene coche delante: aire limpio absoluto.
        gaps.iloc[0] = np.inf if len(gaps) else np.nan
        frame.loc[ordered.index, "GapAheadSeconds"] = gaps.to_numpy()
    return frame


def clean_air_laps(
    laps: pd.DataFrame,
    *,
    threshold: float = DEFAULT_CLEAN_AIR_GAP,
) -> pd.DataFrame:
    """Vueltas cuyo hueco al coche de adelante supera `threshold` segundos."""
    frame = laps if "GapAheadSeconds" in laps.columns else gap_to_car_ahead(laps)
    if frame.empty:
        return frame
    return frame.loc[frame["GapAheadSeconds"] >= threshold].reset_index(drop=True)


def apply_fuel_correction(
    laps: pd.DataFrame,
    *,
    coefficient: float = DEFAULT_FUEL_EFFECT_S_PER_LAP,
    total_laps: int | None = None,
) -> pd.DataFrame:
    """Agrega `FuelCorrectedSeconds`: tiempo normalizado a coche en tanque vacio.

    ESTIMACION. Supone un efecto lineal constante de `coefficient` s/vuelta de
    combustible restante. El tiempo corregido resta el lastre teorico que
    quedaba en ese momento:

        corregido = bruto - coefficient * (total_laps - lap_number)

    Asi la vuelta 1 (tanque lleno) se acelera y la ultima queda igual.
    """
    if laps.empty:
        return laps.assign(FuelCorrectedSeconds=np.nan)

    frame = laps.copy()
    reference = total_laps if total_laps is not None else int(frame["LapNumber"].max())
    remaining = (reference - frame["LapNumber"]).clip(lower=0)
    frame["FuelLoadPenaltySeconds"] = remaining * coefficient
    frame["FuelCorrectedSeconds"] = frame["LapTimeSeconds"] - frame["FuelLoadPenaltySeconds"]
    return frame


def track_evolution(laps: pd.DataFrame, *, window: int = 5, quantile: float = 0.5) -> pd.DataFrame:
    """Evolucion de pista: referencia del peloton por vuelta, suavizada.

    Toma el cuantil indicado de los tiempos de todo el peloton en cada vuelta y
    le aplica una media movil centrada de `window` vueltas.
    """
    if laps.empty:
        return pd.DataFrame(columns=["LapNumber", "FieldLapTime", "Smoothed", "Drivers"])

    grouped = (
        laps.groupby("LapNumber")["LapTimeSeconds"]
        .agg(FieldLapTime=lambda values: values.quantile(quantile), Drivers="count")
        .reset_index()
        .sort_values("LapNumber")
    )
    grouped["Smoothed"] = (
        grouped["FieldLapTime"].rolling(window=window, center=True, min_periods=1).mean()
    )
    return grouped.reset_index(drop=True)


def pace_summary(laps: pd.DataFrame, *, column: str = "LapTimeSeconds") -> pd.DataFrame:
    """Resumen por piloto: vueltas contadas, mediana, media, mejor y desvio."""
    if laps.empty or column not in laps.columns:
        return pd.DataFrame(
            columns=["Driver", "Team", "Laps", "Median", "Mean", "Best", "StdDev", "Range"]
        )

    rows = []
    for driver, group in laps.groupby("Driver"):
        values = group[column].dropna()
        if values.empty:
            continue
        rows.append(
            {
                "Driver": driver,
                "Team": group["Team"].iloc[0] if "Team" in group.columns else "",
                "Laps": int(values.count()),
                "Median": float(values.median()),
                "Mean": float(values.mean()),
                "Best": float(values.min()),
                "StdDev": float(values.std(ddof=1)) if values.count() > 1 else np.nan,
                "Range": float(values.max() - values.min()),
            }
        )
    summary = pd.DataFrame(rows)
    return summary.sort_values("Median").reset_index(drop=True) if not summary.empty else summary


def field_average_pace(laps: pd.DataFrame, drivers: list[str] | None = None) -> float:
    """Ritmo medio (mediana de medianas) del grupo de pilotos indicado."""
    if laps.empty:
        return float("nan")
    subset = laps if drivers is None else laps.loc[laps["Driver"].isin(drivers)]
    if subset.empty:
        return float("nan")
    return float(subset.groupby("Driver")["LapTimeSeconds"].median().median())


# ---------------------------------------------------------------- sectores


SECTOR_COLUMNS = ["Sector1Seconds", "Sector2Seconds", "Sector3Seconds"]
SPEED_COLUMNS = {"SpeedI1": "Trampa S1", "SpeedI2": "Trampa S2", "SpeedFL": "Meta", "SpeedST": "Recta principal"}


def sector_summary(laps: pd.DataFrame) -> pd.DataFrame:
    """Mejor tiempo por sector y mejores velocidades de trampa, por piloto."""
    if laps.empty:
        return pd.DataFrame()

    rows = []
    for driver, group in laps.groupby("Driver"):
        row: dict[str, object] = {
            "Driver": driver,
            "Team": group["Team"].iloc[0] if "Team" in group.columns else "",
        }
        for index, column in enumerate(SECTOR_COLUMNS, start=1):
            values = group[column].dropna() if column in group.columns else pd.Series(dtype=float)
            row[f"S{index}"] = float(values.min()) if not values.empty else np.nan
        for column, label in SPEED_COLUMNS.items():
            values = pd.to_numeric(group.get(column), errors="coerce").dropna()
            row[label] = float(values.max()) if not values.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def theoretical_best(laps: pd.DataFrame) -> pd.DataFrame:
    """Vuelta ideal (suma de los tres mejores sectores) contra la vuelta real mas rapida.

    `Delta` es cuanto tiempo dejo el piloto sobre la mesa por no encadenar sus
    mejores sectores en la misma vuelta.
    """
    if laps.empty:
        return pd.DataFrame(columns=["Driver", "Team", "S1", "S2", "S3", "Ideal", "Actual", "Delta"])

    rows = []
    for driver, group in laps.groupby("Driver"):
        sectors = {}
        for index, column in enumerate(SECTOR_COLUMNS, start=1):
            values = group[column].dropna() if column in group.columns else pd.Series(dtype=float)
            sectors[f"S{index}"] = float(values.min()) if not values.empty else np.nan
        actual_values = group["LapTimeSeconds"].dropna()
        actual = float(actual_values.min()) if not actual_values.empty else np.nan
        ideal = sum(sectors.values()) if not any(np.isnan(v) for v in sectors.values()) else np.nan
        rows.append(
            {
                "Driver": driver,
                "Team": group["Team"].iloc[0] if "Team" in group.columns else "",
                **sectors,
                "Ideal": ideal,
                "Actual": actual,
                "Delta": actual - ideal if not (np.isnan(ideal) or np.isnan(actual)) else np.nan,
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values("Ideal").reset_index(drop=True)


def consistency_index(laps: pd.DataFrame, *, min_laps: int = 3) -> pd.DataFrame:
    """Desvio estandar de las vueltas limpias de cada piloto (menor = mas constante)."""
    if laps.empty:
        return pd.DataFrame(columns=["Driver", "Team", "Laps", "StdDev", "Median", "CoefVar"])

    rows = []
    for driver, group in laps.groupby("Driver"):
        values = group["LapTimeSeconds"].dropna()
        if values.count() < min_laps:
            continue
        std = float(values.std(ddof=1))
        median = float(values.median())
        rows.append(
            {
                "Driver": driver,
                "Team": group["Team"].iloc[0] if "Team" in group.columns else "",
                "Laps": int(values.count()),
                "StdDev": std,
                "Median": median,
                "CoefVar": std / median * 100 if median else np.nan,
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values("StdDev").reset_index(drop=True) if not frame.empty else frame
