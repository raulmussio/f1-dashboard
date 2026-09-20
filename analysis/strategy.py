"""Forma de carrera y estrategia: posiciones, huecos, banderas y undercut.

Modulo puro: pandas/numpy. No importa Streamlit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.pace import gap_to_car_ahead

# Codigos de estado de pista que merecen marcarse en los graficos de carrera.
NOTABLE_TRACK_STATUS = {
    "2": "Bandera amarilla",
    "4": "Safety Car",
    "5": "Bandera roja",
    "6": "VSC",
    "7": "VSC terminando",
}


def position_by_lap(laps: pd.DataFrame) -> pd.DataFrame:
    """Matriz vuelta x piloto con la posicion en pista."""
    if laps.empty or "Position" not in laps.columns:
        return pd.DataFrame()
    return (
        laps.pivot_table(index="LapNumber", columns="Driver", values="Position", aggfunc="min")
        .sort_index()
    )


def gap_to_leader(laps: pd.DataFrame) -> pd.DataFrame:
    """Agrega `GapToLeaderSeconds`: diferencia contra el lider en cada vuelta.

    Se compara el instante de cruce de meta del mismo numero de vuelta. Para un
    piloto doblado el valor refleja su propio ritmo acumulado, no la distancia
    en pista; se interpreta como tiempo perdido contra el lider.
    """
    if laps.empty or "SessionTimeSeconds" not in laps.columns:
        return laps.assign(GapToLeaderSeconds=np.nan)

    frame = laps.copy()
    leader = frame.groupby("LapNumber")["SessionTimeSeconds"].transform("min")
    frame["GapToLeaderSeconds"] = frame["SessionTimeSeconds"] - leader
    return frame


def race_shape(laps: pd.DataFrame) -> pd.DataFrame:
    """Vueltas con posicion, hueco al lider y hueco al coche de adelante."""
    return gap_to_car_ahead(gap_to_leader(laps))


def cumulative_race_time(laps: pd.DataFrame, drivers: list[str]) -> pd.DataFrame:
    """Tiempo acumulado por vuelta de cada piloto, y delta contra el primero de la lista.

    Usa los tiempos por vuelta reales incluyendo paradas, que es lo que importa
    para comparar estrategias completas.
    """
    if laps.empty or not drivers:
        return pd.DataFrame(columns=["LapNumber", "Driver", "CumulativeSeconds", "DeltaSeconds"])

    subset = laps.loc[laps["Driver"].isin(drivers), ["Driver", "LapNumber", "LapTimeSeconds"]].copy()
    subset = subset.dropna(subset=["LapTimeSeconds"]).sort_values(["Driver", "LapNumber"])
    subset["CumulativeSeconds"] = subset.groupby("Driver")["LapTimeSeconds"].cumsum()

    reference = subset.loc[subset["Driver"] == drivers[0], ["LapNumber", "CumulativeSeconds"]]
    reference = reference.rename(columns={"CumulativeSeconds": "ReferenceSeconds"})
    merged = subset.merge(reference, on="LapNumber", how="left")
    merged["DeltaSeconds"] = merged["CumulativeSeconds"] - merged["ReferenceSeconds"]
    return merged.reset_index(drop=True)


def track_status_segments(track_status: pd.DataFrame, laps: pd.DataFrame) -> pd.DataFrame:
    """Tramos de bandera amarilla, SC, VSC y roja, ubicados en numeros de vuelta.

    Convierte los cambios de estado (que vienen en tiempo de sesion) en
    intervalos y los traduce a vueltas usando el paso del lider por meta.
    """
    columns = ["Status", "Label", "StartSeconds", "EndSeconds", "StartLap", "EndLap"]
    if track_status.empty or laps.empty:
        return pd.DataFrame(columns=columns)

    status = track_status.sort_values("SessionTimeSeconds").reset_index(drop=True)
    end_of_session = laps["SessionTimeSeconds"].max()
    status["EndSeconds"] = status["SessionTimeSeconds"].shift(-1).fillna(end_of_session)

    leader = (
        laps.groupby("LapNumber")["SessionTimeSeconds"].min().sort_index()
    )
    lap_numbers = leader.index.to_numpy(dtype=float)
    lap_times = leader.to_numpy(dtype=float)

    def to_lap(seconds: float) -> float:
        if not np.isfinite(seconds) or lap_times.size == 0:
            return np.nan
        return float(np.interp(seconds, lap_times, lap_numbers))

    rows = []
    for _, row in status.iterrows():
        code = str(row["Status"])
        flags = [char for char in code if char in NOTABLE_TRACK_STATUS]
        if not flags:
            continue
        # Cuando se solapan varios codigos, se reporta el mas severo.
        severity = {"2": 1, "7": 2, "6": 3, "4": 4, "5": 5}
        worst = max(flags, key=lambda char: severity.get(char, 0))
        rows.append(
            {
                "Status": worst,
                "Label": NOTABLE_TRACK_STATUS[worst],
                "StartSeconds": float(row["SessionTimeSeconds"]),
                "EndSeconds": float(row["EndSeconds"]),
                "StartLap": to_lap(float(row["SessionTimeSeconds"])),
                "EndLap": to_lap(float(row["EndSeconds"])),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def pit_stop_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Vueltas en las que cada piloto entro a boxes, con el compuesto que monto."""
    if laps.empty:
        return pd.DataFrame(columns=["Driver", "LapNumber", "NextCompound"])

    stops = laps.loc[laps["IsPitIn"].fillna(False), ["Driver", "LapNumber", "Stint"]].copy()
    if stops.empty:
        return pd.DataFrame(columns=["Driver", "LapNumber", "NextCompound"])

    next_compound = (
        laps.sort_values(["Driver", "LapNumber"])
        .groupby(["Driver", "Stint"])["Compound"]
        .first()
        .reset_index()
    )
    stops["Stint"] = stops["Stint"] + 1
    merged = stops.merge(next_compound, on=["Driver", "Stint"], how="left")
    return merged.rename(columns={"Compound": "NextCompound"}).drop(columns=["Stint"])


# ------------------------------------------------------- simulador de undercut


def undercut_simulation(
    *,
    attacker: str,
    defender: str,
    gap_seconds: float,
    laps_earlier: int,
    pit_loss_seconds: float,
    fresh_intercept: float,
    fresh_slope: float,
    old_intercept: float,
    old_slope: float,
    defender_tyre_age: float,
    defender_pit_loss_seconds: float | None = None,
) -> dict:
    """Proyecta el resultado de un undercut/overcut con el modelo lineal ajustado.

    PROYECCION, NO PREDICCION. Supone que:
      - el ritmo de cada neumatico sigue la recta ajustada en la vista de
        degradacion (`tiempo = intercepto + pendiente * vida_neumatico`);
      - la perdida en pit lane es la misma en cada parada y se paga entera en
        la vuelta de la parada;
      - no hay trafico, banderas, errores ni cambios de condiciones.

    Convencion de signos: `gap_seconds` positivo significa que el atacante va
    POR DETRAS del defensor. Un `ProjectedGap` negativo significa que el
    atacante quedaria POR DELANTE.

    Devuelve `timeline` (una fila por vuelta de la ventana) y un resumen.
    """
    window = max(int(laps_earlier), 1)
    defender_pit_loss = (
        pit_loss_seconds if defender_pit_loss_seconds is None else defender_pit_loss_seconds
    )

    rows = []
    attacker_cumulative = 0.0
    defender_cumulative = 0.0
    for lap in range(1, window + 1):
        attacker_lap = fresh_intercept + fresh_slope * lap
        if lap == 1:
            # El atacante paga el pit lane en la primera vuelta de la ventana.
            attacker_lap += pit_loss_seconds
        defender_lap = old_intercept + old_slope * (defender_tyre_age + lap)
        if lap == window:
            # El defensor reacciona y para al final de la ventana.
            defender_lap += defender_pit_loss

        attacker_cumulative += attacker_lap
        defender_cumulative += defender_lap
        rows.append(
            {
                "LapInWindow": lap,
                "AttackerLapTime": attacker_lap,
                "DefenderLapTime": defender_lap,
                "AttackerCumulative": attacker_cumulative,
                "DefenderCumulative": defender_cumulative,
                "ProjectedGap": gap_seconds + attacker_cumulative - defender_cumulative,
            }
        )

    timeline = pd.DataFrame(rows)
    final_gap = float(timeline["ProjectedGap"].iloc[-1]) if not timeline.empty else np.nan
    return {
        "timeline": timeline,
        "attacker": attacker,
        "defender": defender,
        "initial_gap": gap_seconds,
        "final_gap": final_gap,
        "net_change": final_gap - gap_seconds,
        "undercut_works": bool(final_gap < 0),
        "assumptions": {
            "Vueltas de anticipacion": window,
            "Perdida en pit lane (atacante)": round(pit_loss_seconds, 2),
            "Perdida en pit lane (defensor)": round(defender_pit_loss, 2),
            "Ritmo neumatico nuevo": f"{fresh_intercept:.3f} + {fresh_slope:.3f} x vida",
            "Ritmo neumatico usado": f"{old_intercept:.3f} + {old_slope:.3f} x vida",
            "Vida del neumatico del defensor": round(float(defender_tyre_age), 1),
            "Hueco inicial (atacante detras)": round(gap_seconds, 2),
        },
    }


def head_to_head(results: pd.DataFrame, left: str, right: str, *, key: str = "position") -> dict:
    """Comparacion directa entre dos pilotos sobre un DataFrame de resultados.

    Cuenta cuantas veces cada uno termino por delante en las carreras que ambos
    completaron con posicion registrada.
    """
    if results.empty:
        return {"left": left, "right": right, "left_wins": 0, "right_wins": 0, "compared": 0}

    subset = results.loc[results["driverId"].isin([left, right])]
    pivot = subset.pivot_table(index="round", columns="driverId", values=key, aggfunc="min")
    if left not in pivot.columns or right not in pivot.columns:
        return {"left": left, "right": right, "left_wins": 0, "right_wins": 0, "compared": 0}

    both = pivot.dropna(subset=[left, right])
    return {
        "left": left,
        "right": right,
        "left_wins": int((both[left] < both[right]).sum()),
        "right_wins": int((both[right] < both[left]).sum()),
        "compared": int(len(both)),
    }


def cumulative_points(results: pd.DataFrame, *, group_key: str, name_key: str) -> pd.DataFrame:
    """Puntos acumulados por ronda a lo largo de la temporada.

    Suma los puntos de cada resultado ronda a ronda. Acepta resultados de
    carrera y de sprint concatenados, que es como se arma el campeonato real.
    """
    required = {"round", "points", group_key}
    if results.empty or not required.issubset(results.columns):
        return pd.DataFrame(columns=["round", group_key, name_key, "RoundPoints", "CumulativePoints"])

    per_round = (
        results.groupby(["round", group_key, name_key], dropna=False)["points"]
        .sum()
        .reset_index()
        .rename(columns={"points": "RoundPoints"})
    )
    rounds = sorted(per_round["round"].unique())
    entities = per_round[[group_key, name_key]].drop_duplicates()

    grid = entities.merge(pd.DataFrame({"round": rounds}), how="cross")
    complete = grid.merge(per_round, on=["round", group_key, name_key], how="left")
    complete["RoundPoints"] = complete["RoundPoints"].fillna(0.0)
    complete = complete.sort_values([group_key, "round"])
    complete["CumulativePoints"] = complete.groupby(group_key)["RoundPoints"].cumsum()
    return complete.reset_index(drop=True)
