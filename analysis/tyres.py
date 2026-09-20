"""Analisis de neumaticos: stints y degradacion por regresion lineal.

Modulo puro: pandas/numpy/scipy. No importa Streamlit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# Minimo de vueltas utiles para que una regresion de degradacion signifique algo.
MIN_LAPS_FOR_REGRESSION = 4


def stints(laps: pd.DataFrame) -> pd.DataFrame:
    """Un renglon por piloto y stint: compuesto, rango de vueltas y ritmo.

    Usa todas las vueltas del stint para delimitarlo (incluidas entrada/salida
    de boxes) pero calcula el ritmo solo con las vueltas cronometradas.
    """
    if laps.empty:
        return pd.DataFrame(
            columns=[
                "Driver", "Team", "Stint", "Compound", "LapStart", "LapEnd",
                "Laps", "TyreLifeStart", "FreshTyre", "MedianLapTime", "BestLapTime",
            ]
        )

    rows = []
    for (driver, stint), group in laps.groupby(["Driver", "Stint"], dropna=True):
        group = group.sort_values("LapNumber")
        times = group["LapTimeSeconds"].dropna()
        compounds = group["Compound"].dropna()
        rows.append(
            {
                "Driver": driver,
                "Team": group["Team"].iloc[0] if "Team" in group.columns else "",
                "Stint": int(stint),
                "Compound": compounds.iloc[0] if not compounds.empty else "UNKNOWN",
                "LapStart": int(group["LapNumber"].min()),
                "LapEnd": int(group["LapNumber"].max()),
                "Laps": int(len(group)),
                "TyreLifeStart": float(group["TyreLife"].min()) if "TyreLife" in group.columns else np.nan,
                "FreshTyre": bool(group["FreshTyre"].iloc[0]) if "FreshTyre" in group.columns else False,
                "MedianLapTime": float(times.median()) if not times.empty else np.nan,
                "BestLapTime": float(times.min()) if not times.empty else np.nan,
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values(["Driver", "Stint"]).reset_index(drop=True)


def stint_degradation(
    laps: pd.DataFrame,
    *,
    min_laps: int = MIN_LAPS_FOR_REGRESSION,
    confidence: float = 0.95,
    column: str = "LapTimeSeconds",
) -> pd.DataFrame:
    """Regresion lineal del tiempo por vuelta contra la vida del neumatico.

    ESTIMACION. Ajusta `tiempo = intercepto + pendiente * vida_neumatico` por
    stint. La pendiente es la degradacion en s/vuelta. Devuelve tambien el
    intervalo de confianza de la pendiente y el R2 para poder juzgar el ajuste.

    `column` elige la variable dependiente: con `LapTimeSeconds` la pendiente
    mezcla desgaste con aligeramiento por combustible y evolucion de pista;
    con `FuelCorrectedSeconds` (ver `analysis.pace.apply_fuel_correction`) se
    aisla mejor el desgaste, al precio de depender del supuesto de combustible.

    Las vueltas deben venir ya filtradas (sin boxes, sin SC/VSC): esta funcion
    no decide que es una vuelta limpia.
    """
    columns = [
        "Driver", "Team", "Stint", "Compound", "Laps", "LapStart", "LapEnd",
        "SlopeSecPerLap", "Intercept", "StdErr", "CILow", "CIHigh", "R2", "PValue", "MedianLapTime",
    ]
    if laps.empty or column not in laps.columns:
        return pd.DataFrame(columns=columns)

    rows = []
    for (driver, stint), group in laps.groupby(["Driver", "Stint"], dropna=True):
        subset = group.dropna(subset=[column])
        if "TyreLife" in subset.columns and subset["TyreLife"].notna().all():
            x = subset["TyreLife"].astype(float)
        else:
            x = subset["LapNumber"].astype(float)
        y = subset[column].astype(float)

        if len(subset) < min_laps or x.nunique() < 2:
            continue

        result = stats.linregress(x, y)
        degrees_of_freedom = len(subset) - 2
        t_critical = stats.t.ppf(0.5 + confidence / 2, degrees_of_freedom) if degrees_of_freedom > 0 else np.nan
        margin = t_critical * result.stderr if degrees_of_freedom > 0 else np.nan
        compounds = subset["Compound"].dropna()

        rows.append(
            {
                "Driver": driver,
                "Team": subset["Team"].iloc[0] if "Team" in subset.columns else "",
                "Stint": int(stint),
                "Compound": compounds.iloc[0] if not compounds.empty else "UNKNOWN",
                "Laps": int(len(subset)),
                "LapStart": int(subset["LapNumber"].min()),
                "LapEnd": int(subset["LapNumber"].max()),
                "SlopeSecPerLap": float(result.slope),
                "Intercept": float(result.intercept),
                "StdErr": float(result.stderr),
                "CILow": float(result.slope - margin) if margin == margin else np.nan,
                "CIHigh": float(result.slope + margin) if margin == margin else np.nan,
                "R2": float(result.rvalue**2),
                "PValue": float(result.pvalue),
                "MedianLapTime": float(y.median()),
            }
        )
    frame = pd.DataFrame(rows, columns=columns)
    return frame.sort_values(["Driver", "Stint"]).reset_index(drop=True)


def degradation_fit_line(
    degradation_row: pd.Series,
    *,
    tyre_life: np.ndarray | None = None,
) -> pd.DataFrame:
    """Puntos de la recta ajustada de un stint, para superponer al scatter."""
    if tyre_life is None:
        span = int(degradation_row["Laps"])
        tyre_life = np.arange(1, span + 1, dtype=float)
    fitted = degradation_row["Intercept"] + degradation_row["SlopeSecPerLap"] * tyre_life
    return pd.DataFrame({"TyreLife": tyre_life, "Fitted": fitted})


def compound_degradation(degradation: pd.DataFrame) -> pd.DataFrame:
    """Promedio de degradacion por compuesto, ponderado por vueltas del stint."""
    if degradation.empty:
        return pd.DataFrame(columns=["Compound", "Stints", "Laps", "MeanSlope", "MedianSlope"])

    rows = []
    for compound, group in degradation.groupby("Compound"):
        weights = group["Laps"].astype(float)
        rows.append(
            {
                "Compound": compound,
                "Stints": int(len(group)),
                "Laps": int(weights.sum()),
                "MeanSlope": float(np.average(group["SlopeSecPerLap"], weights=weights)),
                "MedianSlope": float(group["SlopeSecPerLap"].median()),
            }
        )
    return pd.DataFrame(rows).sort_values("MeanSlope").reset_index(drop=True)


def estimate_pit_loss(laps: pd.DataFrame, *, reference_quantile: float = 0.5) -> dict[str, float]:
    """Estima la perdida de tiempo en pit lane a partir de las vueltas reales.

    ESTIMACION derivada de los datos de la propia sesion: compara el tiempo de
    las vueltas de entrada y de salida de boxes contra el ritmo de referencia
    de ese mismo piloto. Devuelve la mediana entre todas las paradas.

    Devuelve un dict con `in_lap_loss`, `out_lap_loss`, `total` y `stops`.
    """
    empty = {"in_lap_loss": float("nan"), "out_lap_loss": float("nan"), "total": float("nan"), "stops": 0}
    if laps.empty or "IsPitIn" not in laps.columns:
        return empty

    in_losses: list[float] = []
    out_losses: list[float] = []
    for _, group in laps.groupby("Driver"):
        clean = group.loc[
            ~group["IsPitIn"].fillna(False)
            & ~group["IsPitOut"].fillna(False)
            & group["LapTimeSeconds"].notna()
        ]
        if len(clean) < 3:
            continue
        reference = clean["LapTimeSeconds"].quantile(reference_quantile)
        in_laps = group.loc[group["IsPitIn"].fillna(False), "LapTimeSeconds"].dropna()
        out_laps = group.loc[group["IsPitOut"].fillna(False), "LapTimeSeconds"].dropna()
        in_losses.extend((in_laps - reference).tolist())
        out_losses.extend((out_laps - reference).tolist())

    if not in_losses and not out_losses:
        return empty

    in_median = float(np.median(in_losses)) if in_losses else 0.0
    out_median = float(np.median(out_losses)) if out_losses else 0.0
    return {
        "in_lap_loss": in_median,
        "out_lap_loss": out_median,
        "total": in_median + out_median,
        "stops": int(max(len(in_losses), len(out_losses))),
    }
