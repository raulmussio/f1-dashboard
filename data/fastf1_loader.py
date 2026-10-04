"""Carga de sesiones, vueltas y telemetria desde FastF1.

Responsabilidad unica: obtener datos crudos de FastF1 y normalizarlos a
DataFrames de pandas. No calcula metricas (eso vive en `analysis/`) ni dibuja
nada (eso vive en `ui/`).

FastF1 solo publica datos de cronometraje y telemetria desde 2018. Para
temporadas anteriores este modulo levanta `TelemetryUnavailable` con un mensaje
explicito en vez de devolver datos vacios o inventados.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import fastf1
import fastf1.plotting
import pandas as pd
import streamlit as st

LOGGER = logging.getLogger(__name__)

# Primera temporada con datos de cronometraje/telemetria en la API de F1.
TELEMETRY_MIN_YEAR = 2018

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"

# Codigos de estado de pista que usa la API de F1 (columna `TrackStatus`).
TRACK_STATUS_LABELS = {
    "1": "Pista libre",
    "2": "Bandera amarilla",
    "3": "Bandera amarilla",
    "4": "Safety Car",
    "5": "Bandera roja",
    "6": "VSC desplegado",
    "7": "VSC terminando",
}

# Estados que ensucian un tiempo por vuelta para analisis de ritmo.
DIRTY_TRACK_STATUS = {"2", "3", "4", "5", "6", "7"}

TELEMETRY_CHANNELS = ["Speed", "Throttle", "Brake", "nGear", "RPM", "DRS"]


class TelemetryUnavailable(RuntimeError):
    """No hay datos de FastF1 para lo pedido. El mensaje es apto para la UI."""


# ------------------------------------------------------------------ cache disco


@st.cache_resource(show_spinner=False)
def enable_cache(cache_dir: str | os.PathLike[str] = CACHE_DIR) -> str:
    """Habilita el cache en disco de FastF1, creando el directorio si falta.

    Cacheado como recurso para que se ejecute una sola vez por proceso.
    """
    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(path))
    fastf1.set_log_level(logging.WARNING)
    LOGGER.info("Cache de FastF1 habilitado en %s", path)
    return str(path)


# --------------------------------------------------------------- calendario


@st.cache_data(show_spinner=False)
def get_event_schedule(year: int) -> pd.DataFrame:
    """Calendario de la temporada segun FastF1 (solo >= 2018)."""
    if year < TELEMETRY_MIN_YEAR:
        raise TelemetryUnavailable(
            f"FastF1 no publica calendario ni telemetria para {year}. "
            f"Los datos de sesion empiezan en {TELEMETRY_MIN_YEAR}."
        )
    enable_cache()
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
    except Exception as exc:  # noqa: BLE001 - la causa real va en el mensaje
        raise TelemetryUnavailable(
            f"FastF1 no pudo cargar el calendario de {year}: {exc}"
        ) from exc
    return pd.DataFrame(schedule)


def list_sessions(schedule: pd.DataFrame, round_number: int) -> list[str]:
    """Nombres de sesion realmente disponibles para un evento del calendario."""
    rows = schedule.loc[schedule["RoundNumber"] == round_number]
    if rows.empty:
        return []
    row = rows.iloc[0]
    names = []
    for index in range(1, 6):
        name = row.get(f"Session{index}")
        if isinstance(name, str) and name.strip() and name.lower() != "none":
            names.append(name)
    return names


# ------------------------------------------------------------------- sesion


# Cada sesion cargada con telemetria ocupa unos 150 MB. El limite se mantiene
# bajo a proposito: en un servidor de 1 GB (Streamlit Community Cloud) seis
# sesiones simultaneas desbordarian la memoria y el proceso se reinicia.
MAX_CACHED_SESSIONS = 3


@st.cache_resource(show_spinner=False, max_entries=MAX_CACHED_SESSIONS)
def load_session(year: int, round_number: int, session_name: str, *, laps: bool = True):
    """Carga una sesion de FastF1 con telemetria, clima y estado de pista.

    Cacheada como recurso: el objeto `Session` es pesado y no serializable,
    asi que se comparte entre reruns en vez de reconstruirse.
    """
    if year < TELEMETRY_MIN_YEAR:
        raise TelemetryUnavailable(
            f"FastF1 no tiene datos de sesion para {year}. "
            f"La telemetria y los tiempos por vuelta empiezan en {TELEMETRY_MIN_YEAR}."
        )
    enable_cache()
    # La comprobacion de vueltas va DENTRO del try a proposito: cuando `load()`
    # falla a medias -por ejemplo si el servidor no alcanza la API de F1- no
    # levanta excepcion, pero el primer acceso a `session.laps` lanza
    # `DataNotLoadedError`. Dejarlo fuera hacia que esa excepcion escapara y
    # tirara abajo la aplicacion entera con un traceback, en vez de mostrar el
    # mensaje explicativo de esta funcion.
    try:
        session = fastf1.get_session(year, round_number, session_name)
        session.load(laps=laps, telemetry=True, weather=True, messages=True)
        session_laps = session.laps if laps else None
    except Exception as exc:  # noqa: BLE001
        raise TelemetryUnavailable(
            f"No se pudo cargar {session_name} de la ronda {round_number} de {year} "
            f"desde FastF1: {type(exc).__name__}: {exc}"
        ) from exc

    if laps and (session_laps is None or len(session_laps) == 0):
        raise TelemetryUnavailable(
            f"FastF1 cargo la sesion {session_name} ({year}, ronda {round_number}) "
            "pero no contiene vueltas. Puede ser una sesion cancelada o sin cronometraje."
        )
    return session


# ------------------------------------------------------------ frames derivados


def _timedelta_seconds(series: pd.Series) -> pd.Series:
    return pd.to_timedelta(series, errors="coerce").dt.total_seconds()


def laps_frame(session) -> pd.DataFrame:
    """Vueltas de la sesion como DataFrame plano con columnas en segundos.

    Agrega columnas derivadas que el resto de la app espera, sin filtrar nada:
    el filtrado de vueltas sucias es responsabilidad de `analysis.pace`.
    """
    laps = pd.DataFrame(session.laps).copy()
    if laps.empty:
        return laps

    laps["LapTimeSeconds"] = _timedelta_seconds(laps["LapTime"])
    for sector in (1, 2, 3):
        laps[f"Sector{sector}Seconds"] = _timedelta_seconds(laps[f"Sector{sector}Time"])
    laps["LapStartSeconds"] = _timedelta_seconds(laps["LapStartTime"])
    laps["SessionTimeSeconds"] = _timedelta_seconds(laps["Time"])
    laps["PitInSeconds"] = _timedelta_seconds(laps["PitInTime"])
    laps["PitOutSeconds"] = _timedelta_seconds(laps["PitOutTime"])
    laps["IsPitIn"] = laps["PitInTime"].notna()
    laps["IsPitOut"] = laps["PitOutTime"].notna()
    laps["TrackStatus"] = laps["TrackStatus"].fillna("").astype(str)
    laps["HasDirtyTrackStatus"] = laps["TrackStatus"].apply(
        lambda code: any(char in DIRTY_TRACK_STATUS for char in code)
    )
    laps["LapNumber"] = pd.to_numeric(laps["LapNumber"], errors="coerce")
    laps["Stint"] = pd.to_numeric(laps["Stint"], errors="coerce")
    laps["Compound"] = laps["Compound"].fillna("UNKNOWN")
    return laps


def results_frame(session) -> pd.DataFrame:
    """Clasificacion final de la sesion tal como la entrega FastF1."""
    results = pd.DataFrame(session.results).copy()
    if results.empty:
        return results
    results["TimeSeconds"] = _timedelta_seconds(results["Time"])
    for segment in ("Q1", "Q2", "Q3"):
        if segment in results.columns:
            results[f"{segment}Seconds"] = _timedelta_seconds(results[segment])
    return results


def weather_frame(session) -> pd.DataFrame:
    weather = getattr(session, "weather_data", None)
    if weather is None or len(weather) == 0:
        return pd.DataFrame()
    frame = pd.DataFrame(weather).copy()
    frame["SessionTimeSeconds"] = _timedelta_seconds(frame["Time"])
    return frame


def track_status_frame(session) -> pd.DataFrame:
    """Cambios de estado de pista con su etiqueta legible."""
    status = getattr(session, "track_status", None)
    if status is None or len(status) == 0:
        return pd.DataFrame()
    frame = pd.DataFrame(status).copy()
    frame["SessionTimeSeconds"] = _timedelta_seconds(frame["Time"])
    frame["Status"] = frame["Status"].astype(str)
    frame["Label"] = frame["Status"].map(TRACK_STATUS_LABELS).fillna("Desconocido")
    return frame


def race_control_frame(session) -> pd.DataFrame:
    messages = getattr(session, "race_control_messages", None)
    if messages is None or len(messages) == 0:
        return pd.DataFrame()
    return pd.DataFrame(messages).copy()


@st.cache_data(show_spinner=False, max_entries=12)
def get_laps(year: int, round_number: int, session_name: str) -> pd.DataFrame:
    return laps_frame(load_session(year, round_number, session_name))


@st.cache_data(show_spinner=False, max_entries=12)
def get_results(year: int, round_number: int, session_name: str) -> pd.DataFrame:
    return results_frame(load_session(year, round_number, session_name))


@st.cache_data(show_spinner=False, max_entries=12)
def get_weather(year: int, round_number: int, session_name: str) -> pd.DataFrame:
    return weather_frame(load_session(year, round_number, session_name))


@st.cache_data(show_spinner=False, max_entries=12)
def get_track_status(year: int, round_number: int, session_name: str) -> pd.DataFrame:
    return track_status_frame(load_session(year, round_number, session_name))


@st.cache_data(show_spinner=False, max_entries=12)
def get_race_control(year: int, round_number: int, session_name: str) -> pd.DataFrame:
    return race_control_frame(load_session(year, round_number, session_name))


# --------------------------------------------------------------- telemetria


def _telemetry_frame(lap) -> pd.DataFrame:
    telemetry = lap.get_telemetry()
    if telemetry is None or len(telemetry) == 0:
        raise TelemetryUnavailable("La vuelta seleccionada no tiene canales de telemetria.")
    frame = pd.DataFrame(telemetry).copy()
    frame["TimeSeconds"] = _timedelta_seconds(frame["Time"])
    frame["Brake"] = frame["Brake"].astype(float)
    keep = ["Distance", "TimeSeconds", "X", "Y", "Z", *TELEMETRY_CHANNELS]
    available = [column for column in keep if column in frame.columns]
    frame = frame[available].dropna(subset=["Distance"]).reset_index(drop=True)
    if frame.empty:
        raise TelemetryUnavailable("La telemetria de la vuelta llego sin datos de distancia.")
    return frame


@st.cache_data(show_spinner=False, max_entries=40)
def get_lap_telemetry(
    year: int,
    round_number: int,
    session_name: str,
    driver: str,
    lap_number: int | None = None,
) -> pd.DataFrame:
    """Telemetria de una vuelta concreta, o de la mas rapida si `lap_number` es None.

    Devuelve columnas Distance, TimeSeconds, X, Y, Z y los seis canales
    (Speed, Throttle, Brake, nGear, RPM, DRS).
    """
    session = load_session(year, round_number, session_name)
    driver_laps = session.laps.pick_drivers(driver)
    if len(driver_laps) == 0:
        raise TelemetryUnavailable(f"{driver} no tiene vueltas registradas en esta sesion.")

    if lap_number is None:
        lap = driver_laps.pick_fastest()
        if lap is None or (hasattr(lap, "empty") and lap.empty):
            raise TelemetryUnavailable(f"{driver} no tiene una vuelta rapida valida en esta sesion.")
    else:
        selected = driver_laps.loc[driver_laps["LapNumber"] == lap_number]
        if selected.empty:
            raise TelemetryUnavailable(f"{driver} no completo la vuelta {lap_number}.")
        lap = selected.iloc[0]

    try:
        return _telemetry_frame(lap)
    except TelemetryUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001
        raise TelemetryUnavailable(
            f"FastF1 no pudo entregar telemetria de {driver} "
            f"(vuelta {'mas rapida' if lap_number is None else lap_number}): {exc}"
        ) from exc


@st.cache_data(show_spinner=False, max_entries=8)
def get_circuit_info(year: int, round_number: int, session_name: str) -> pd.DataFrame:
    """Curvas del circuito (numero y posicion) si FastF1 las publica."""
    session = load_session(year, round_number, session_name)
    try:
        info = session.get_circuit_info()
    except Exception:  # noqa: BLE001 - dato opcional, la app funciona sin el
        return pd.DataFrame()
    if info is None or info.corners is None or len(info.corners) == 0:
        return pd.DataFrame()
    return pd.DataFrame(info.corners).copy()


# ------------------------------------------------------------- estilos FastF1


def driver_styles(session) -> dict[str, dict[str, str]]:
    """Color de equipo y estilo de linea por piloto, segun `fastf1.plotting`.

    Compañeros de equipo comparten color y se distinguen por linea solida vs
    punteada, que es el criterio oficial de FastF1.
    """
    styles: dict[str, dict[str, str]] = {}
    for abbreviation in fastf1.plotting.list_driver_abbreviations(session):
        try:
            style = fastf1.plotting.get_driver_style(
                abbreviation, style=["color", "linestyle"], session=session
            )
        except Exception:  # noqa: BLE001 - piloto sin estilo conocido
            style = {"color": "#9aa4b2", "linestyle": "solid"}
        styles[abbreviation] = {
            "color": style.get("color", "#9aa4b2"),
            "linestyle": style.get("linestyle", "solid"),
        }
    return styles


def compound_colors(session) -> dict[str, str]:
    try:
        return dict(fastf1.plotting.get_compound_mapping(session=session))
    except Exception:  # noqa: BLE001
        return {}


def team_colors(session) -> dict[str, str]:
    colors: dict[str, str] = {}
    try:
        names = fastf1.plotting.list_team_names(session)
    except Exception:  # noqa: BLE001
        return colors
    for name in names:
        try:
            colors[name] = fastf1.plotting.get_team_color(name, session=session)
        except Exception:  # noqa: BLE001
            continue
    return colors
