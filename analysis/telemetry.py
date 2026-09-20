"""Comparacion de telemetria entre vueltas: alineacion, delta y mini-sectores.

Modulo puro: pandas/numpy. No importa Streamlit.

Toda la comparacion se hace contra distancia recorrida, no contra tiempo,
que es como se comparan dos vueltas en ingenieria de pista.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CHANNELS = ["Speed", "Throttle", "Brake", "nGear", "RPM", "DRS"]

CHANNEL_LABELS = {
    "Speed": "Velocidad (km/h)",
    "Throttle": "Acelerador (%)",
    "Brake": "Freno",
    "nGear": "Marcha",
    "RPM": "RPM",
    "DRS": "DRS",
}

DEFAULT_MINISECTORS = 25


def normalise(telemetry: pd.DataFrame) -> pd.DataFrame:
    """Deja la telemetria lista para comparar: distancia y tiempo desde cero.

    Quita puntos con distancia no creciente, que rompen cualquier interpolacion.
    """
    if telemetry.empty:
        return telemetry

    frame = telemetry.dropna(subset=["Distance", "TimeSeconds"]).sort_values("Distance").copy()
    frame["Distance"] = frame["Distance"] - frame["Distance"].iloc[0]
    frame["TimeSeconds"] = frame["TimeSeconds"] - frame["TimeSeconds"].iloc[0]
    increasing = frame["Distance"].diff().fillna(1) > 0
    increasing.iloc[0] = True
    return frame.loc[increasing].reset_index(drop=True)


def common_distance_grid(
    telemetry_a: pd.DataFrame,
    telemetry_b: pd.DataFrame,
    *,
    points: int = 1500,
) -> np.ndarray:
    """Rejilla de distancia comun al tramo que ambas vueltas cubren."""
    start = max(telemetry_a["Distance"].min(), telemetry_b["Distance"].min())
    end = min(telemetry_a["Distance"].max(), telemetry_b["Distance"].max())
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        return np.array([])
    return np.linspace(start, end, points)


def resample_on_distance(telemetry: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    """Interpola todos los canales sobre una rejilla de distancia."""
    if telemetry.empty or grid.size == 0:
        return pd.DataFrame({"Distance": grid})

    resampled = {"Distance": grid}
    source_distance = telemetry["Distance"].to_numpy(dtype=float)
    for column in telemetry.columns:
        if column == "Distance":
            continue
        values = pd.to_numeric(telemetry[column], errors="coerce").to_numpy(dtype=float)
        if np.isnan(values).all():
            continue
        resampled[column] = np.interp(grid, source_distance, values)
    return pd.DataFrame(resampled)


def delta_time(reference: pd.DataFrame, comparison: pd.DataFrame, *, points: int = 1500) -> pd.DataFrame:
    """Delta de tiempo acumulado entre dos vueltas contra distancia.

    `DeltaSeconds` positivo significa que `comparison` va perdiendo tiempo
    respecto de `reference` en ese punto del trazado.
    """
    ref = normalise(reference)
    cmp_ = normalise(comparison)
    grid = common_distance_grid(ref, cmp_, points=points)
    if grid.size == 0:
        return pd.DataFrame(columns=["Distance", "DeltaSeconds"])

    reference_time = np.interp(grid, ref["Distance"], ref["TimeSeconds"])
    comparison_time = np.interp(grid, cmp_["Distance"], cmp_["TimeSeconds"])
    return pd.DataFrame({"Distance": grid, "DeltaSeconds": comparison_time - reference_time})


def aligned_channels(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    *,
    points: int = 1500,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ambas vueltas interpoladas sobre la misma rejilla de distancia."""
    ref = normalise(reference)
    cmp_ = normalise(comparison)
    grid = common_distance_grid(ref, cmp_, points=points)
    return resample_on_distance(ref, grid), resample_on_distance(cmp_, grid)


def minisector_dominance(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    *,
    label_reference: str,
    label_comparison: str,
    minisectors: int = DEFAULT_MINISECTORS,
    points: int = 1500,
) -> pd.DataFrame:
    """Quien es mas rapido en cada mini-sector del trazado.

    Parte la vuelta en `minisectors` tramos de igual distancia y compara la
    velocidad media de cada piloto en cada tramo. Devuelve una fila por
    mini-sector con el ganador y la diferencia de velocidad media.
    """
    ref_grid, cmp_grid = aligned_channels(reference, comparison, points=points)
    if ref_grid.empty or "Speed" not in ref_grid.columns or "Speed" not in cmp_grid.columns:
        return pd.DataFrame(
            columns=["Minisector", "DistanceStart", "DistanceEnd", "SpeedReference",
                     "SpeedComparison", "Winner", "SpeedDelta"]
        )

    distance = ref_grid["Distance"].to_numpy()
    edges = np.linspace(distance.min(), distance.max(), minisectors + 1)
    index = np.clip(np.digitize(distance, edges[1:-1], right=False), 0, minisectors - 1)

    rows = []
    for sector in range(minisectors):
        mask = index == sector
        if not mask.any():
            continue
        speed_reference = float(ref_grid.loc[mask, "Speed"].mean())
        speed_comparison = float(cmp_grid.loc[mask, "Speed"].mean())
        rows.append(
            {
                "Minisector": sector + 1,
                "DistanceStart": float(edges[sector]),
                "DistanceEnd": float(edges[sector + 1]),
                "SpeedReference": speed_reference,
                "SpeedComparison": speed_comparison,
                "Winner": label_reference if speed_reference >= speed_comparison else label_comparison,
                "SpeedDelta": speed_reference - speed_comparison,
            }
        )
    return pd.DataFrame(rows)


def track_map_points(
    telemetry: pd.DataFrame,
    *,
    minisectors: int | None = None,
) -> pd.DataFrame:
    """Coordenadas X/Y del trazado con velocidad y, opcionalmente, mini-sector.

    Se usa para dibujar el mapa coloreado por velocidad o por dominancia.
    """
    if telemetry.empty or "X" not in telemetry.columns:
        return pd.DataFrame(columns=["X", "Y", "Speed", "Distance", "Minisector"])

    frame = telemetry.dropna(subset=["X", "Y"]).copy()
    frame = frame[["X", "Y", "Speed", "Distance"]].reset_index(drop=True)
    if minisectors:
        distance = frame["Distance"].to_numpy(dtype=float)
        edges = np.linspace(distance.min(), distance.max(), minisectors + 1)
        frame["Minisector"] = np.clip(
            np.digitize(distance, edges[1:-1], right=False), 0, minisectors - 1
        ) + 1
    return frame


def channel_summary(telemetry: pd.DataFrame) -> dict[str, float]:
    """Cifras resumen de una vuelta: punta, media, % a fondo y % de freno."""
    if telemetry.empty:
        return {}

    summary: dict[str, float] = {}
    if "Speed" in telemetry.columns:
        summary["SpeedMax"] = float(telemetry["Speed"].max())
        summary["SpeedMean"] = float(telemetry["Speed"].mean())
    if "Throttle" in telemetry.columns:
        summary["FullThrottlePct"] = float((telemetry["Throttle"] >= 99).mean() * 100)
    if "Brake" in telemetry.columns:
        summary["BrakingPct"] = float((telemetry["Brake"] > 0).mean() * 100)
    if "TimeSeconds" in telemetry.columns:
        summary["LapDuration"] = float(telemetry["TimeSeconds"].max() - telemetry["TimeSeconds"].min())
    if "Distance" in telemetry.columns:
        summary["Distance"] = float(telemetry["Distance"].max() - telemetry["Distance"].min())
    return summary
