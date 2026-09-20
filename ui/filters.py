"""Filtros en cascada de la barra lateral.

Responsabilidad unica: recoger la seleccion del usuario y garantizar que sea
coherente. El año limita los circuitos, el circuito limita las sesiones y la
sesion limita los pilotos: una opcion que no existe nunca se ofrece.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd
import streamlit as st

from data import jolpica_client as jolpica
from data.fastf1_loader import TELEMETRY_MIN_YEAR, TelemetryUnavailable, get_event_schedule, list_sessions

FIRST_SEASON = 1950

# Sesiones que Jolpica puede describir para temporadas sin telemetria.
HISTORICAL_SESSIONS = ["Race", "Qualifying"]


@dataclass
class SessionSelection:
    """Lo que el usuario eligio, ya validado contra el calendario real."""

    year: int
    round_number: int
    event_name: str
    location: str
    country: str
    session_name: str
    telemetry_available: bool
    schedule: pd.DataFrame = field(default_factory=pd.DataFrame)
    circuit_id: str | None = None
    schedule_error: str | None = None

    @property
    def label(self) -> str:
        return f"{self.year} · {self.event_name} · {self.session_name}"


@dataclass
class AnalysisOptions:
    """Parametros de los modelos y de los filtros de vueltas."""

    exclude_pit_laps: bool = True
    exclude_dirty_status: bool = True
    exclude_inaccurate: bool = True
    percent_threshold: float = 107.0
    clean_air_gap: float = 1.5
    fuel_coefficient: float = 0.035
    apply_fuel_correction: bool = True
    min_stint_laps: int = 4


def _available_years() -> list[int]:
    """Temporadas ofrecidas, de la mas reciente a 1950."""
    return list(range(date.today().year, FIRST_SEASON - 1, -1))


@st.cache_data(show_spinner=False)
def _historical_schedule(year: int) -> pd.DataFrame:
    """Calendario desde Jolpica para temporadas sin datos de FastF1."""
    return jolpica.load_schedule(year)


def _normalise_fastf1_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    frame = schedule.copy()
    frame["round"] = frame["RoundNumber"].astype(int)
    frame["name"] = frame["EventName"]
    frame["location"] = frame["Location"]
    frame["country"] = frame["Country"]
    return frame[["round", "name", "location", "country"]]


def _normalise_jolpica_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    frame = schedule.copy()
    frame["name"] = frame["raceName"]
    frame["location"] = frame["locality"]
    return frame[["round", "name", "location", "country", "circuitId"]]


def session_selector() -> SessionSelection:
    """Cascada año -> circuito -> sesion. Devuelve siempre algo consistente."""
    sidebar = st.sidebar
    sidebar.markdown("### Sesion")

    years = _available_years()
    default_year = years.index(2024) if 2024 in years else 0
    year = sidebar.selectbox("Temporada", years, index=default_year)

    telemetry_available = year >= TELEMETRY_MIN_YEAR
    schedule = pd.DataFrame()
    normalised = pd.DataFrame()
    schedule_error: str | None = None

    if telemetry_available:
        try:
            schedule = get_event_schedule(year)
            normalised = _normalise_fastf1_schedule(schedule)
        except TelemetryUnavailable as exc:
            schedule_error = str(exc)
    else:
        try:
            normalised = _normalise_jolpica_schedule(_historical_schedule(year))
        except jolpica.JolpicaError as exc:
            schedule_error = exc.message

    if normalised.empty:
        sidebar.error(schedule_error or f"No hay calendario disponible para {year}.")
        return SessionSelection(
            year=year,
            round_number=0,
            event_name="",
            location="",
            country="",
            session_name="",
            telemetry_available=telemetry_available,
            schedule_error=schedule_error or f"No hay calendario disponible para {year}.",
        )

    labels = {
        int(row["round"]): f"R{int(row['round']):02d} · {row['name']} ({row['location']})"
        for _, row in normalised.iterrows()
    }
    rounds = sorted(labels)
    default_round = rounds.index(16) if 16 in rounds and year == 2024 else 0
    round_number = sidebar.selectbox(
        "Gran Premio", rounds, index=default_round, format_func=lambda value: labels[value]
    )
    event = normalised.loc[normalised["round"] == round_number].iloc[0]

    if telemetry_available:
        sessions = list_sessions(schedule, round_number) or ["Race"]
        default_session = sessions.index("Race") if "Race" in sessions else len(sessions) - 1
        session_name = sidebar.selectbox("Sesion", sessions, index=default_session)
    else:
        session_name = sidebar.selectbox(
            "Sesion",
            HISTORICAL_SESSIONS,
            index=0,
            help=(
                f"FastF1 solo publica cronometraje y telemetria desde {TELEMETRY_MIN_YEAR}. "
                "Para esta temporada solo estan disponibles las vistas de campeonato e historico."
            ),
        )
        sidebar.warning(
            f"Telemetria no disponible para {year}: los datos de sesion empiezan en {TELEMETRY_MIN_YEAR}.",
            icon="⚠️",
        )

    return SessionSelection(
        year=year,
        round_number=int(round_number),
        event_name=str(event["name"]),
        location=str(event["location"]),
        country=str(event.get("country", "")),
        session_name=session_name,
        telemetry_available=telemetry_available,
        schedule=schedule,
        circuit_id=event.get("circuitId"),
        schedule_error=schedule_error,
    )


def analysis_form(
    laps: pd.DataFrame,
    selection: SessionSelection,
    *,
    default_count: int = 10,
) -> tuple[AnalysisOptions, list[str]]:
    """Filtros de vueltas, parametros del modelo y pilotos, dentro de un formulario.

    Al estar en un `st.form`, mover un control no recalcula nada: los cambios se
    aplican todos juntos al pulsar el boton. Evita recargar seis veces mientras
    se ajustan seis filtros.

    La seleccion de temporada/GP/sesion queda fuera del formulario a proposito,
    porque tiene que ir en cascada: el año limita los circuitos y el circuito
    limita las sesiones.
    """
    sidebar = st.sidebar
    teams = (
        laps.groupby("Driver")["Team"].first().to_dict()
        if not laps.empty and "Team" in laps.columns
        else {}
    )
    ranking = (
        laps.groupby("Driver")["LapTimeSeconds"].median().sort_values().index.tolist()
        if not laps.empty
        else []
    )
    available_teams = sorted({team for team in teams.values() if isinstance(team, str) and team})

    # Las claves llevan la sesion: al cambiar de carrera, la seleccion de pilotos
    # se reinicia en vez de arrastrar pilotos que no corrieron esta.
    scope = f"{selection.year}_{selection.round_number}_{selection.session_name}"

    with sidebar.form("filtros_analisis", border=False):
        st.markdown("### Filtro de vueltas")
        exclude_pit = st.checkbox("Excluir vueltas de entrada y salida de boxes", value=True)
        exclude_status = st.checkbox("Excluir SC, VSC y banderas", value=True)
        exclude_inaccurate = st.checkbox(
            "Excluir vueltas marcadas como imprecisas",
            value=True,
            help="Usa la marca `IsAccurate` de FastF1, que descarta vueltas con cronometraje dudoso.",
        )
        percent = st.slider(
            "Descartar vueltas por encima del % del mejor tiempo",
            min_value=101.0,
            max_value=130.0,
            value=107.0,
            step=0.5,
        )

        st.markdown("### Parametros del modelo")
        clean_air = st.slider(
            "Umbral de aire limpio (s al coche de adelante)",
            min_value=0.5,
            max_value=5.0,
            value=1.5,
            step=0.1,
        )
        apply_fuel = st.checkbox("Aplicar correccion por combustible", value=True)
        fuel_coefficient = st.slider(
            "Efecto del combustible (s por vuelta)",
            min_value=0.0,
            max_value=0.12,
            value=0.035,
            step=0.005,
            format="%.3f",
            help="Supuesto lineal. Todo valor derivado de el se marca como estimacion en la UI.",
        )
        min_stint = st.slider(
            "Vueltas minimas por stint para ajustar degradacion",
            min_value=3,
            max_value=12,
            value=4,
        )

        st.markdown("### Pilotos")
        chosen_teams = st.multiselect(
            "Filtrar por equipo",
            available_teams,
            default=[],
            key=f"teams_{scope}",
            placeholder="Todos los equipos",
            help="Vacio = todos los equipos. Se aplica al pulsar el boton.",
        )
        selected = st.multiselect(
            "Pilotos en pantalla",
            ranking,
            default=ranking[:default_count],
            key=f"drivers_{scope}",
            format_func=lambda driver: f"{driver} · {teams.get(driver, '')}".strip(" ·"),
            placeholder="Todos los pilotos",
            help="Vacio = todos los pilotos de la sesion.",
        )

        st.form_submit_button("Aplicar filtros", width="stretch", type="primary")

    options = AnalysisOptions(
        exclude_pit_laps=exclude_pit,
        exclude_dirty_status=exclude_status,
        exclude_inaccurate=exclude_inaccurate,
        percent_threshold=percent,
        clean_air_gap=clean_air,
        fuel_coefficient=fuel_coefficient,
        apply_fuel_correction=apply_fuel,
        min_stint_laps=min_stint,
    )

    if laps.empty:
        sidebar.info("La sesion cargada no tiene vueltas de ningun piloto.")
        return options, []

    drivers = selected or ranking
    if chosen_teams:
        drivers = [driver for driver in drivers if teams.get(driver) in chosen_teams]
        if not drivers:
            sidebar.warning(
                "Ningun piloto seleccionado pertenece a los equipos filtrados.", icon="⚠️"
            )
    return options, drivers


def sidebar_footer(cache_path: str, jolpica_requests: int | None = None) -> None:
    """Pie de la barra lateral con el estado del cache y del limite de Jolpica."""
    sidebar = st.sidebar
    sidebar.markdown("---")
    lines = [f"Cache FastF1: `{cache_path}`"]
    if jolpica_requests is not None:
        lines.append(f"Jolpica: {jolpica_requests}/500 req en la ultima hora")
    sidebar.caption("  \n".join(lines))
