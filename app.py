"""F1 Race Engineering Dashboard.

Entrada de la aplicacion y layout de las siete vistas. Este modulo solo
orquesta: pide datos a `data/`, metricas a `analysis/` y figuras a `ui/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from analysis import pace, prediction, strategy, tyres
from analysis import telemetry as telem
from data import fastf1_loader as ff1
from data import jolpica_client as jolpica
from data import weather_client
from ui import charts, filters, theme

VIEWS = [
    "1 · Resumen de sesion",
    "2 · Ritmo de carrera",
    "3 · Neumaticos y estrategia",
    "4 · Comparacion de telemetria",
    "5 · Vueltas y sectores",
    "6 · Carrera vuelta a vuelta",
    "7 · Campeonato e historico",
    "8 · Predictor de carrera",
]

# Vistas que no pueden dibujarse sin datos de FastF1.
TELEMETRY_VIEWS = set(VIEWS[:6])


# --------------------------------------------------------------------- utiles


def format_lap_time(seconds: float | None) -> str:
    """Segundos a `m:ss.mmm`. Devuelve guion si no hay dato."""
    if seconds is None or not np.isfinite(seconds):
        return "—"
    minutes, remainder = divmod(float(seconds), 60)
    if minutes >= 60:
        hours, minutes = divmod(minutes, 60)
        return f"{int(hours)}:{int(minutes):02d}:{remainder:06.3f}"
    return f"{int(minutes)}:{remainder:06.3f}"


def format_gap(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "—"
    return f"+{seconds:.3f}"


def pace_order(laps: pd.DataFrame, restrict_to: list[str] | None = None) -> list[str]:
    """Pilotos ordenados por su mejor vuelta, para que los desplegables abran en cabeza."""
    if laps.empty:
        return []
    order = laps.groupby("Driver")["LapTimeSeconds"].min().sort_values().index.tolist()
    if restrict_to is not None:
        allowed = set(restrict_to)
        order = [driver for driver in order if driver in allowed]
        order += [driver for driver in restrict_to if driver not in order]
    return order


@dataclass
class SessionContext:
    """Todo lo que las vistas necesitan de una sesion ya cargada."""

    selection: filters.SessionSelection
    options: filters.AnalysisOptions
    session: object
    laps: pd.DataFrame
    clean_laps: pd.DataFrame
    results: pd.DataFrame
    weather: pd.DataFrame
    track_status: pd.DataFrame
    flags: pd.DataFrame
    styles: dict[str, dict[str, str]]
    compound_colors: dict[str, str]
    drivers: list[str]

    @property
    def is_race(self) -> bool:
        return "Race" in self.selection.session_name or "Sprint" in self.selection.session_name

    def subset(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Filtra un frame de vueltas a los pilotos elegidos en la barra lateral."""
        if frame.empty or not self.drivers or "Driver" not in frame.columns:
            return frame
        return frame.loc[frame["Driver"].isin(self.drivers)]


@dataclass
class RawSession:
    """Datos crudos de FastF1, antes de aplicar los filtros del usuario."""

    session: object
    laps: pd.DataFrame
    results: pd.DataFrame
    weather: pd.DataFrame
    track_status: pd.DataFrame


def load_raw_session(
    selection: filters.SessionSelection, *, show_errors: bool = True
) -> RawSession | None:
    """Carga la sesion de FastF1. Devuelve None y explica el motivo si no hay datos."""
    try:
        return RawSession(
            session=ff1.load_session(selection.year, selection.round_number, selection.session_name),
            laps=ff1.get_laps(selection.year, selection.round_number, selection.session_name),
            results=ff1.get_results(selection.year, selection.round_number, selection.session_name),
            weather=ff1.get_weather(selection.year, selection.round_number, selection.session_name),
            track_status=ff1.get_track_status(
                selection.year, selection.round_number, selection.session_name
            ),
        )
    except ff1.TelemetryUnavailable as exc:
        if show_errors:
            st.error(f"**No se pudieron cargar los datos de la sesion.** {exc}", icon="🚫")
            st.info(
                "Las vistas de campeonato e historico (7) siguen disponibles porque usan Jolpica, "
                "que cubre desde 1950.",
                icon="ℹ️",
            )
        return None


def build_context(
    selection: filters.SessionSelection,
    options: filters.AnalysisOptions,
    drivers: list[str],
    raw: RawSession,
) -> SessionContext:
    """Aplica los filtros del usuario sobre los datos crudos ya cargados."""
    laps = strategy.race_shape(raw.laps)
    if options.apply_fuel_correction:
        laps = pace.apply_fuel_correction(laps, coefficient=options.fuel_coefficient)

    clean_laps = pace.filter_laps(
        laps,
        exclude_pit=options.exclude_pit_laps,
        exclude_dirty_status=options.exclude_dirty_status,
        exclude_inaccurate=options.exclude_inaccurate,
        max_percent_off_best=options.percent_threshold,
    )

    return SessionContext(
        selection=selection,
        options=options,
        session=raw.session,
        laps=laps,
        clean_laps=clean_laps,
        results=raw.results,
        weather=raw.weather,
        track_status=raw.track_status,
        flags=strategy.track_status_segments(raw.track_status, laps),
        styles=ff1.driver_styles(raw.session),
        compound_colors=ff1.compound_colors(raw.session),
        drivers=drivers,
    )


# ------------------------------------------------------- 1. resumen de sesion


def render_summary(context: SessionContext) -> None:
    selection = context.selection
    theme.header(
        f"{selection.event_name}",
        f"{selection.year} · {selection.location} · {selection.session_name}",
    )

    results = context.results
    laps = context.laps
    clean = context.clean_laps

    fastest_lap = clean["LapTimeSeconds"].min() if not clean.empty else np.nan
    fastest_driver = (
        clean.loc[clean["LapTimeSeconds"].idxmin(), "Driver"] if not clean.empty else "—"
    )
    total_laps = int(laps["LapNumber"].max()) if not laps.empty else 0

    if not results.empty:
        leader = results.iloc[0]
        leader_label = "Ganador" if context.is_race else "Pole"
        leader_value = str(leader.get("Abbreviation", "—"))
        leader_delta = str(leader.get("TeamName", ""))
    else:
        leader_label, leader_value, leader_delta = "Lider", "—", "Sin clasificacion publicada"

    top_drivers = results.head(10)["Abbreviation"].tolist() if not results.empty else []
    top_pace = pace.field_average_pace(clean, top_drivers)

    weather = context.weather
    air = weather["AirTemp"].mean() if not weather.empty else np.nan
    track = weather["TrackTemp"].mean() if not weather.empty else np.nan
    rain_share = float(weather["Rainfall"].mean() * 100) if not weather.empty else np.nan
    rain = bool(rain_share > 0) if np.isfinite(rain_share) else False

    columns = st.columns(6)
    columns[0].metric(leader_label, leader_value, leader_delta, delta_color="off", border=True)
    columns[1].metric("Vuelta rapida", format_lap_time(fastest_lap), str(fastest_driver),
                      delta_color="off", border=True)
    columns[2].metric("Vueltas", f"{total_laps}", f"{len(laps)} registros", delta_color="off", border=True)
    columns[3].metric(
        "Temperaturas",
        f"{air:.1f} °C" if np.isfinite(air) else "—",
        f"Pista {track:.1f} °C" if np.isfinite(track) else "Sin dato de pista",
        delta_color="off",
        border=True,
    )
    columns[4].metric(
        "Lluvia",
        "Si" if rain else "No",
        f"{rain_share:.0f} % de las muestras" if rain else "Sin lluvia registrada",
        delta_color="off",
        border=True,
    )
    columns[5].metric("Ritmo medio top 10", format_lap_time(top_pace),
                      "Mediana de medianas", delta_color="off", border=True)

    if weather.empty:
        st.warning("FastF1 no entrego datos de clima para esta sesion.", icon="⚠️")

    st.markdown("#### Clasificacion")
    if results.empty:
        st.warning("Esta sesion no tiene una tabla de resultados publicada por FastF1.", icon="⚠️")
        return

    stints = tyres.stints(laps)
    compounds = (
        stints.groupby("Driver")["Compound"].apply(lambda values: " → ".join(values)).to_dict()
        if not stints.empty
        else {}
    )
    stops = laps.groupby("Driver")["IsPitIn"].sum().to_dict() if not laps.empty else {}
    driver_laps = laps.groupby("Driver")["LapNumber"].max().to_dict()

    rows = []
    for _, row in results.iterrows():
        abbreviation = row.get("Abbreviation", "")
        seconds = row.get("TimeSeconds", np.nan)
        position = row.get("Position", np.nan)
        if context.is_race:
            time_text = format_lap_time(seconds) if position == 1 else format_gap(seconds)
        else:
            best = row.get("Q3Seconds", np.nan)
            for segment in ("Q3Seconds", "Q2Seconds", "Q1Seconds"):
                value = row.get(segment, np.nan)
                if pd.notna(value):
                    best = value
                    break
            time_text = format_lap_time(best)
        rows.append(
            {
                "Pos": int(position) if pd.notna(position) else None,
                "Piloto": abbreviation,
                "Equipo": row.get("TeamName", ""),
                "Tiempo / gap": time_text,
                "Vueltas": int(driver_laps.get(abbreviation, row.get("Laps", 0) or 0)),
                "Compuestos": compounds.get(abbreviation, "—"),
                "Paradas": int(stops.get(abbreviation, 0)),
                "Estado": row.get("Status", ""),
                "Puntos": row.get("Points", np.nan),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, height=560)
    theme.caption(
        "Tiempo/gap: para carrera, tiempo total del ganador y diferencia del resto; "
        "para clasificacion, el mejor tiempo alcanzado. Compuestos y paradas salen de las "
        "vueltas reales de FastF1."
    )


# ------------------------------------------------------- 2. ritmo de carrera


def render_pace(context: SessionContext) -> None:
    theme.header("Ritmo de carrera", context.selection.label)
    options = context.options
    clean = context.subset(context.clean_laps)

    if clean.empty:
        st.warning(
            "Ninguna vuelta pasa los filtros actuales. Relaja el umbral de % o desactiva "
            "algun filtro en la barra lateral.",
            icon="⚠️",
        )
        return

    controls = st.columns([1, 1, 2])
    kind = controls[0].segmented_control("Grafico", ["Box", "Violin"], default="Box") or "Box"
    only_clean_air = controls[1].toggle(
        "Solo aire limpio", value=False,
        help=f"Vueltas con mas de {options.clean_air_gap:.1f} s al coche de adelante.",
    )

    frame = pace.clean_air_laps(clean, threshold=options.clean_air_gap) if only_clean_air else clean
    if frame.empty:
        st.warning(
            f"Ningun piloto seleccionado tiene vueltas con mas de {options.clean_air_gap:.1f} s "
            "de hueco al coche de adelante.",
            icon="⚠️",
        )
        return

    controls[2].caption(
        f"{len(frame)} vueltas en analisis de {len(context.laps)} registradas · "
        f"{frame['Driver'].nunique()} pilotos"
    )

    left, right = st.columns([3, 2])
    with left.container(border=True):
        theme.plotly(
            charts.lap_time_distribution(
                frame,
                context.styles,
                kind=kind.lower(),
                title=("Distribucion de tiempos" + (" · aire limpio" if only_clean_air else "")),
            )
        )
    with right.container(border=True):
        summary = pace.pace_summary(frame)
        comparison = None
        if options.apply_fuel_correction and "FuelCorrectedSeconds" in frame.columns:
            corrected = pace.pace_summary(frame, column="FuelCorrectedSeconds")
            summary = summary.merge(
                corrected[["Driver", "Median"]].rename(columns={"Median": "MedianCorrected"}),
                on="Driver",
                how="left",
            )
            comparison = "MedianCorrected"
        theme.plotly(
            charts.pace_ranking(summary, context.styles, comparison_column=comparison,
                                title="Ritmo mediano por piloto")
        )
        if comparison:
            theme.estimate_note(
                "El ritmo corregido por combustible sale de un modelo lineal, no de una medicion.",
                {"Efecto asumido": f"{options.fuel_coefficient:.3f} s/vuelta",
                 "Referencia": "coche en tanque vacio"},
            )

    with st.container(border=True):
        theme.plotly(charts.track_evolution(pace.track_evolution(context.clean_laps)))
        theme.caption(
            "Mediana de todo el peloton por vuelta (no solo los pilotos filtrados) con media "
            "movil de 5 vueltas. Mezcla agarre ganado por la pista con el aligeramiento del coche."
        )

    with st.container(border=True):
        st.markdown("##### Resumen numerico")
        display = pace.pace_summary(frame).copy()
        for column in ("Median", "Mean", "Best", "StdDev", "Range"):
            display[column] = display[column].round(3)
        st.dataframe(display, hide_index=True, height=340)


# -------------------------------------------------- 3. neumaticos y estrategia


def render_tyres(context: SessionContext) -> None:
    theme.header("Neumaticos y estrategia", context.selection.label)
    options = context.options
    laps = context.subset(context.laps)
    clean = context.subset(context.clean_laps)

    if laps.empty:
        st.warning("Sin vueltas para los pilotos seleccionados.", icon="⚠️")
        return

    stint_table = tyres.stints(laps)
    with st.container(border=True):
        theme.plotly(charts.stint_gantt(stint_table, context.compound_colors, order=context.drivers))

    degradation_column = "LapTimeSeconds"
    use_corrected = False
    if options.apply_fuel_correction and "FuelCorrectedSeconds" in clean.columns:
        use_corrected = st.toggle(
            "Ajustar degradacion sobre tiempos corregidos por combustible",
            value=True,
            help=(
                "Sin corregir, la pendiente mezcla desgaste con el aligeramiento del coche y "
                "suele salir negativa."
            ),
        )
        if use_corrected:
            degradation_column = "FuelCorrectedSeconds"

    degradation = tyres.stint_degradation(
        clean, min_laps=options.min_stint_laps, column=degradation_column
    )

    if degradation.empty:
        st.warning(
            f"Ningun stint tiene {options.min_stint_laps} vueltas limpias o mas. "
            "Baja el minimo en la barra lateral para ajustar rectas.",
            icon="⚠️",
        )
    else:
        left, right = st.columns([3, 2])
        with left.container(border=True):
            available = pace_order(clean, degradation["Driver"].unique().tolist())
            chosen = st.multiselect(
                "Pilotos en la curva de degradacion", available, default=available[:4]
            )
            selected = degradation.loc[degradation["Driver"].isin(chosen)]
            theme.plotly(
                charts.degradation_scatter(
                    clean,
                    selected,
                    compound_colors=context.compound_colors,
                    column=degradation_column,
                    title=(
                        "Degradacion por stint"
                        + (" · tiempos corregidos" if use_corrected else " · tiempos brutos")
                    ),
                )
            )
            theme.estimate_note(
                "Pendiente e intervalo salen de una regresion lineal sobre las vueltas limpias "
                "del stint. No es una medicion de desgaste del neumatico.",
                {
                    "Modelo": "tiempo = intercepto + pendiente × vida del neumatico",
                    "Variable": ("tiempo corregido por combustible" if use_corrected else "tiempo bruto"),
                    "Minimo de vueltas": options.min_stint_laps,
                    "Confianza": "95 %",
                },
            )
        with right.container(border=True):
            theme.plotly(charts.degradation_ranking(degradation, context.compound_colors))

    st.markdown("#### Comparador de estrategias")
    with st.container(border=True):
        available = pace_order(laps)
        chosen = st.multiselect(
            "Pilotos a comparar (2 o mas)", available, default=available[:2], key="strategy_drivers"
        )
        if len(chosen) < 2:
            st.info("Elige al menos dos pilotos para comparar estrategias.", icon="ℹ️")
        else:
            cumulative = strategy.cumulative_race_time(context.laps, chosen)
            columns = st.columns(2)
            with columns[0]:
                theme.plotly(charts.strategy_delta(cumulative, context.styles, chosen[0]))
            with columns[1]:
                positions = strategy.position_by_lap(context.laps)
                theme.plotly(
                    charts.position_chart(positions, context.styles, flags=context.flags, drivers=chosen)
                )
            totals = (
                cumulative.sort_values("LapNumber").groupby("Driver").last().reset_index()
            )[["Driver", "LapNumber", "CumulativeSeconds", "DeltaSeconds"]]
            totals.columns = ["Piloto", "Ultima vuelta", "Tiempo acumulado (s)", f"Delta vs {chosen[0]} (s)"]
            st.dataframe(totals.round(3), hide_index=True)

    render_undercut(context, laps)


def render_undercut(context: SessionContext, laps: pd.DataFrame) -> None:
    st.markdown("#### Simulador de undercut / overcut")
    degradation = tyres.stint_degradation(
        context.clean_laps, min_laps=context.options.min_stint_laps
    )
    if degradation.empty:
        st.info(
            "El simulador necesita al menos un stint con regresion ajustada. "
            "Baja el minimo de vueltas por stint en la barra lateral.",
            icon="ℹ️",
        )
        return

    pit_loss = tyres.estimate_pit_loss(context.laps)
    with st.container(border=True):
        drivers = pace_order(context.clean_laps, degradation["Driver"].unique().tolist())
        columns = st.columns(4)
        attacker = columns[0].selectbox("Atacante (para antes)", drivers, index=0)
        defender_options = [driver for driver in drivers if driver != attacker] or drivers
        defender = columns[1].selectbox("Defensor", defender_options, index=0)
        laps_earlier = columns[2].number_input(
            "Vueltas de anticipacion", min_value=1, max_value=10, value=2, step=1
        )
        default_loss = pit_loss["total"] if np.isfinite(pit_loss["total"]) else 22.0
        pit_loss_value = columns[3].number_input(
            "Perdida en pit lane (s)",
            min_value=10.0,
            max_value=60.0,
            value=float(round(default_loss, 1)),
            step=0.5,
            help=(
                "Valor inicial estimado con las vueltas de entrada y salida de boxes de esta "
                "misma sesion. Editalo si conoces el dato real del circuito."
            ),
        )

        attacker_stints = degradation.loc[degradation["Driver"] == attacker]
        defender_stints = degradation.loc[degradation["Driver"] == defender]
        if attacker_stints.empty or defender_stints.empty:
            st.warning("Alguno de los dos pilotos no tiene stints ajustables.", icon="⚠️")
            return

        model_columns = st.columns(3)
        attacker_label = model_columns[0].selectbox(
            "Stint del atacante (neumatico nuevo)",
            attacker_stints.apply(lambda r: f"S{int(r['Stint'])} · {r['Compound']}", axis=1).tolist(),
        )
        defender_label = model_columns[1].selectbox(
            "Stint del defensor (neumatico en uso)",
            defender_stints.apply(lambda r: f"S{int(r['Stint'])} · {r['Compound']}", axis=1).tolist(),
        )
        attacker_row = attacker_stints.iloc[
            attacker_stints.apply(lambda r: f"S{int(r['Stint'])} · {r['Compound']}", axis=1)
            .tolist()
            .index(attacker_label)
        ]
        defender_row = defender_stints.iloc[
            defender_stints.apply(lambda r: f"S{int(r['Stint'])} · {r['Compound']}", axis=1)
            .tolist()
            .index(defender_label)
        ]

        reference_lap = int(defender_row["LapEnd"])
        measured_gap = _gap_between(context.laps, attacker, defender, reference_lap)
        gap = model_columns[2].number_input(
            "Hueco inicial (s, positivo = atacante detras)",
            min_value=-30.0,
            max_value=60.0,
            value=float(round(measured_gap, 2)) if np.isfinite(measured_gap) else 2.0,
            step=0.1,
            help=(
                f"Medido en la vuelta {reference_lap} de la sesion real. Podes cambiarlo para "
                "probar otros escenarios."
            ),
        )

        defender_age = float(defender_row["LapEnd"] - defender_row["LapStart"] + 1)
        simulation = strategy.undercut_simulation(
            attacker=attacker,
            defender=defender,
            gap_seconds=gap,
            laps_earlier=int(laps_earlier),
            pit_loss_seconds=pit_loss_value,
            fresh_intercept=float(attacker_row["Intercept"]),
            fresh_slope=float(attacker_row["SlopeSecPerLap"]),
            old_intercept=float(defender_row["Intercept"]),
            old_slope=float(defender_row["SlopeSecPerLap"]),
            defender_tyre_age=defender_age,
        )

        columns = st.columns([2, 1])
        with columns[0]:
            theme.plotly(charts.undercut_projection(simulation))
        with columns[1]:
            st.metric(
                "Hueco proyectado al final",
                f"{simulation['final_gap']:+.2f} s",
                f"{simulation['net_change']:+.2f} s vs inicio",
                delta_color="inverse",
                border=True,
            )
            st.metric(
                "Resultado del modelo",
                f"{attacker} delante" if simulation["undercut_works"] else f"{defender} delante",
                border=True,
            )
            if np.isfinite(pit_loss["total"]):
                st.metric(
                    "Pit loss estimado de la sesion",
                    f"{pit_loss['total']:.1f} s",
                    f"{pit_loss['stops']} paradas medidas",
                    delta_color="off",
                    border=True,
                )

        theme.estimate_note(
            "PROYECCION, no prediccion. Extrapola las rectas de degradacion ajustadas y supone "
            "pista libre: no modela trafico, banderas, errores ni cambios de condiciones.",
            simulation["assumptions"],
        )


def _gap_between(laps: pd.DataFrame, attacker: str, defender: str, lap_number: int) -> float:
    """Hueco real entre dos pilotos en una vuelta, medido con el cruce de meta."""
    subset = laps.loc[laps["LapNumber"] == lap_number]
    attacker_time = subset.loc[subset["Driver"] == attacker, "SessionTimeSeconds"]
    defender_time = subset.loc[subset["Driver"] == defender, "SessionTimeSeconds"]
    if attacker_time.empty or defender_time.empty:
        return float("nan")
    return float(attacker_time.iloc[0] - defender_time.iloc[0])


# ------------------------------------------------- 4. comparacion de telemetria


def _lap_options(laps: pd.DataFrame, driver: str) -> dict[str, int | None]:
    """Etiquetas de vuelta seleccionables para un piloto, mas la opcion 'mas rapida'."""
    subset = laps.loc[(laps["Driver"] == driver) & laps["LapTimeSeconds"].notna()]
    subset = subset.sort_values("LapNumber")
    options: dict[str, int | None] = {"Vuelta mas rapida": None}
    for _, row in subset.iterrows():
        label = (
            f"V{int(row['LapNumber']):02d} · {format_lap_time(row['LapTimeSeconds'])} "
            f"· {row['Compound']}"
        )
        options[label] = int(row["LapNumber"])
    return options


def render_telemetry(context: SessionContext) -> None:
    theme.header("Comparacion de telemetria", context.selection.label)
    laps = context.laps
    # Ordenados por ritmo: el desplegable abre con los de cabeza, no con la A.
    available = pace_order(laps)
    if not available:
        st.warning("Sin pilotos con vueltas en esta sesion.", icon="⚠️")
        return

    columns = st.columns(4)
    driver_a = columns[0].selectbox("Piloto A (referencia)", available, index=0)
    options_a = _lap_options(laps, driver_a)
    lap_label_a = columns[1].selectbox("Vuelta de A", list(options_a), index=0)

    default_b = 1 if len(available) > 1 else 0
    driver_b = columns[2].selectbox("Piloto B (comparacion)", available, index=default_b)
    options_b = _lap_options(laps, driver_b)
    lap_label_b = columns[3].selectbox("Vuelta de B", list(options_b), index=0)

    selection = context.selection
    try:
        telemetry_a = ff1.get_lap_telemetry(
            selection.year, selection.round_number, selection.session_name,
            driver_a, options_a[lap_label_a],
        )
        telemetry_b = ff1.get_lap_telemetry(
            selection.year, selection.round_number, selection.session_name,
            driver_b, options_b[lap_label_b],
        )
    except ff1.TelemetryUnavailable as exc:
        st.error(f"**Telemetria no disponible.** {exc}", icon="🚫")
        return

    normalised_a = telem.normalise(telemetry_a)
    normalised_b = telem.normalise(telemetry_b)
    summary_a = telem.channel_summary(normalised_a)
    summary_b = telem.channel_summary(normalised_b)

    label_a = f"{driver_a} · {lap_label_a.split(' · ')[0]}"
    label_b = f"{driver_b} · {lap_label_b.split(' · ')[0]}"
    color_a = context.styles.get(driver_a, {}).get("color", theme.COLORS["info"])
    color_b = context.styles.get(driver_b, {}).get("color", theme.COLORS["warning"])
    if color_a == color_b:
        # Compañeros de equipo: mismo color, se separan por tipo de linea.
        dash_a, dash_b = "solid", "dash"
    else:
        dash_a, dash_b = "solid", "solid"

    metrics = st.columns(6)
    metrics[0].metric(f"{driver_a} · vuelta", format_lap_time(summary_a.get("LapDuration")),
                      border=True, delta_color="off")
    metrics[1].metric(f"{driver_b} · vuelta", format_lap_time(summary_b.get("LapDuration")),
                      border=True, delta_color="off")
    metrics[2].metric(f"{driver_a} · punta", f"{summary_a.get('SpeedMax', float('nan')):.0f} km/h",
                      border=True, delta_color="off")
    metrics[3].metric(f"{driver_b} · punta", f"{summary_b.get('SpeedMax', float('nan')):.0f} km/h",
                      border=True, delta_color="off")
    metrics[4].metric(f"{driver_a} · a fondo", f"{summary_a.get('FullThrottlePct', float('nan')):.1f} %",
                      border=True, delta_color="off")
    metrics[5].metric(f"{driver_b} · a fondo", f"{summary_b.get('FullThrottlePct', float('nan')):.1f} %",
                      border=True, delta_color="off")

    with st.container(border=True):
        theme.plotly(
            charts.telemetry_channels(
                normalised_a,
                normalised_b,
                label_reference=label_a,
                label_comparison=label_b,
                color_reference=color_a,
                color_comparison=color_b,
                dash_reference=dash_a,
                dash_comparison=dash_b,
            )
        )

    with st.container(border=True):
        delta = telem.delta_time(normalised_a, normalised_b)
        theme.plotly(charts.delta_trace(delta, label_reference=label_a, label_comparison=label_b))
        theme.caption(
            "El delta se calcula interpolando el tiempo de cada vuelta sobre una rejilla comun "
            "de distancia. Es una medicion directa de la telemetria, no un modelo."
        )

    columns = st.columns(2)
    with columns[0].container(border=True):
        theme.plotly(
            charts.track_map_speed(
                telem.track_map_points(normalised_a),
                title=f"Trazado por velocidad · {label_a}",
            )
        )
    with columns[1].container(border=True):
        dominance = telem.minisector_dominance(
            normalised_a, normalised_b, label_reference=driver_a, label_comparison=driver_b
        )
        points = telem.track_map_points(normalised_a, minisectors=telem.DEFAULT_MINISECTORS)
        colors = {driver_a: color_a, driver_b: color_b}
        if color_a == color_b:
            colors[driver_b] = theme.COLORS["warning"]
        theme.plotly(
            charts.track_map_dominance(
                points, dominance, colors=colors,
                title=f"Dominancia por mini-sector ({telem.DEFAULT_MINISECTORS})",
            )
        )
        if not dominance.empty:
            counts = dominance["Winner"].value_counts()
            st.caption(
                " · ".join(
                    f"**{driver}**: {count} de {len(dominance)} mini-sectores"
                    for driver, count in counts.items()
                )
            )


# ------------------------------------------------------ 5. vueltas y sectores


def render_sectors(context: SessionContext) -> None:
    theme.header("Vueltas y sectores", context.selection.label)
    clean = context.subset(context.clean_laps)
    if clean.empty:
        st.warning("Sin vueltas limpias para los pilotos seleccionados.", icon="⚠️")
        return

    sectors = pace.sector_summary(clean)
    theoretical = pace.theoretical_best(clean)
    consistency = pace.consistency_index(clean)

    with st.container(border=True):
        theme.plotly(charts.sector_comparison(sectors, context.styles))

    columns = st.columns(2)
    with columns[0].container(border=True):
        theme.plotly(charts.ideal_vs_actual(theoretical, context.styles))
        theme.caption(
            "La vuelta ideal suma los tres mejores sectores del piloto aunque pertenezcan a "
            "vueltas distintas. Es un calculo directo, no una estimacion."
        )
    with columns[1].container(border=True):
        theme.plotly(charts.consistency_chart(consistency, context.styles))

    with st.container(border=True):
        st.markdown("##### Velocidades de trampa")
        speed_columns = [column for column in pace.SPEED_COLUMNS.values() if column in sectors.columns]
        if not speed_columns:
            st.warning("FastF1 no registro velocidades de trampa en esta sesion.", icon="⚠️")
        else:
            chosen = st.segmented_control(
                "Punto de medicion", speed_columns, default=speed_columns[-1]
            ) or speed_columns[-1]
            theme.plotly(charts.speed_traps(sectors, context.styles, chosen))

    with st.container(border=True):
        st.markdown("##### Tabla de sectores y vuelta ideal")
        table = theoretical.merge(
            consistency[["Driver", "StdDev", "Laps"]], on="Driver", how="left"
        )
        table = table.rename(
            columns={
                "Driver": "Piloto", "Team": "Equipo", "Ideal": "Vuelta ideal",
                "Actual": "Mejor real", "Delta": "Deja (s)", "StdDev": "Desvio (s)",
                "Laps": "Vueltas limpias",
            }
        )
        st.dataframe(table.round(3), hide_index=True, height=420)


# -------------------------------------------------- 6. carrera vuelta a vuelta


def render_race_trace(context: SessionContext) -> None:
    theme.header("Carrera vuelta a vuelta", context.selection.label)
    laps = context.laps
    if laps.empty:
        st.warning("Sin vueltas registradas en esta sesion.", icon="⚠️")
        return

    flags = context.flags
    if flags.empty:
        st.info(
            "FastF1 no reporto banderas amarillas, safety car ni VSC en esta sesion: "
            "los graficos no llevan sombreado.",
            icon="ℹ️",
        )
    else:
        counts = flags["Label"].value_counts().to_dict()
        st.caption("Tramos detectados por estado de pista: " +
                   " · ".join(f"**{label}** ×{count}" for label, count in counts.items()))

    with st.container(border=True):
        theme.plotly(
            charts.position_chart(
                strategy.position_by_lap(laps), context.styles, flags=flags, drivers=context.drivers
            )
        )

    columns = st.columns(2)
    with columns[0].container(border=True):
        clip = st.slider("Recortar eje en (s)", 5, 180, 90, step=5, key="gap_leader_clip")
        theme.plotly(
            charts.gap_chart(
                laps, context.styles, column="GapToLeaderSeconds",
                title="Hueco al lider", flags=flags, drivers=context.drivers, clip=float(clip),
            )
        )
        theme.caption(
            "Diferencia de tiempo de cruce de meta en el mismo numero de vuelta. Para un piloto "
            "doblado refleja tiempo perdido acumulado, no distancia en pista."
        )
    with columns[1].container(border=True):
        clip_ahead = st.slider("Recortar eje en (s)", 2, 60, 20, step=1, key="gap_ahead_clip")
        theme.plotly(
            charts.gap_chart(
                laps, context.styles, column="GapAheadSeconds",
                title="Hueco al coche de adelante", flags=flags, drivers=context.drivers,
                clip=float(clip_ahead),
            )
        )

    with st.container(border=True):
        st.markdown("##### Paradas en boxes")
        stops = strategy.pit_stop_laps(laps)
        if stops.empty:
            st.info("No hay paradas en boxes registradas en esta sesion.", icon="ℹ️")
        else:
            stops = stops.rename(
                columns={"Driver": "Piloto", "LapNumber": "Vuelta", "NextCompound": "Compuesto montado"}
            )
            st.dataframe(stops.sort_values("Vuelta"), hide_index=True, height=300)


# ------------------------------------------------ 7. campeonato e historico


def _jolpica_guard(function, *args, **kwargs):
    """Ejecuta una consulta a Jolpica mostrando el error en pantalla si falla."""
    try:
        return function(*args, **kwargs)
    except jolpica.JolpicaRateLimited as exc:
        st.error(
            f"**Jolpica limito las consultas.** {exc.message} "
            "El cliente ya reintenta con backoff exponencial; volve a intentar en unos segundos.",
            icon="⏳",
        )
    except jolpica.JolpicaError as exc:
        st.error(f"**Jolpica no devolvio datos.** {exc.message}", icon="🚫")
    return None


def render_championship(selection: filters.SessionSelection, context: SessionContext | None) -> None:
    theme.header("Campeonato e historico", f"Temporada {selection.year} · fuente Jolpica (Ergast)")
    st.caption(
        "Esta vista no usa FastF1: funciona para cualquier temporada desde 1950. "
        "Jolpica limita a 4 req/s y 500 req/h, asi que las consultas se cachean una hora."
    )

    sub_view = st.segmented_control(
        "Bloque",
        ["Evolucion de puntos", "Head-to-head de compañeros", "Historico"],
        default="Evolucion de puntos",
    ) or "Evolucion de puntos"

    if sub_view == "Evolucion de puntos":
        _render_points_evolution(selection, context)
    elif sub_view == "Head-to-head de compañeros":
        _render_head_to_head(selection)
    else:
        _render_history(selection)


def _season_results(year: int) -> pd.DataFrame | None:
    """Resultados de carrera y sprint de la temporada, concatenados."""
    with st.spinner(f"Consultando resultados de {year} en Jolpica..."):
        races = _jolpica_guard(jolpica.load_race_results, year)
        if races is None:
            return None
        sprints = _jolpica_guard(jolpica.load_sprint_results, year)
    frames = [frame for frame in (races, sprints) if frame is not None and not frame.empty]
    if not frames:
        return races
    return pd.concat(frames, ignore_index=True)


def _render_points_evolution(selection: filters.SessionSelection, context: SessionContext | None) -> None:
    results = _season_results(selection.year)
    if results is None or results.empty:
        st.warning(f"Jolpica no tiene resultados para {selection.year}.", icon="⚠️")
        return

    team_colors = ff1.team_colors(context.session) if context is not None else {}

    columns = st.columns(2)
    with columns[0].container(border=True):
        drivers = strategy.cumulative_points(results, group_key="driverId", name_key="driverName")
        theme.plotly(
            charts.points_evolution(
                drivers, group_key="driverId", name_key="driverName",
                title=f"Campeonato de pilotos {selection.year}",
            )
        )
    with columns[1].container(border=True):
        constructors = strategy.cumulative_points(
            results, group_key="constructorId", name_key="constructorName"
        )
        theme.plotly(
            charts.points_evolution(
                constructors, group_key="constructorId", name_key="constructorName",
                colors=team_colors,
                title=f"Campeonato de constructores {selection.year}",
            )
        )

    theme.caption(
        "Los puntos se acumulan sumando cada resultado de carrera y de sprint publicado por "
        "Jolpica, incluido el punto por vuelta rapida cuando corresponde."
    )

    with st.container(border=True):
        st.markdown("##### Clasificacion final publicada")
        standings = _jolpica_guard(jolpica.load_driver_standings, selection.year)
        if standings is not None and not standings.empty:
            table = standings[["position", "driverName", "constructorName", "points", "wins"]]
            table.columns = ["Pos", "Piloto", "Equipo", "Puntos", "Victorias"]
            st.dataframe(table, hide_index=True, height=420)


def _render_head_to_head(selection: filters.SessionSelection) -> None:
    results = _season_results(selection.year)
    if results is None or results.empty:
        st.warning(f"Jolpica no tiene resultados para {selection.year}.", icon="⚠️")
        return

    teams = sorted(results["constructorName"].dropna().unique())
    if not teams:
        st.warning("No se pudo determinar la lista de equipos de esta temporada.", icon="⚠️")
        return

    team = st.selectbox("Equipo", teams)
    team_results = results.loc[results["constructorName"] == team]
    line_up = (
        team_results.groupby(["driverId", "driverName"])["round"].count().sort_values(ascending=False)
    )
    if len(line_up) < 2:
        st.info(f"{team} tuvo un solo piloto con resultados en {selection.year}.", icon="ℹ️")
        return

    pairs = [f"{name}" for _, name in line_up.index[:6]]
    columns = st.columns(2)
    left_name = columns[0].selectbox("Piloto A", pairs, index=0)
    right_name = columns[1].selectbox("Piloto B", pairs, index=1)
    if left_name == right_name:
        st.info("Elegi dos pilotos distintos.", icon="ℹ️")
        return

    name_to_id = {name: driver_id for driver_id, name in line_up.index}
    left_id, right_id = name_to_id[left_name], name_to_id[right_name]

    race_h2h = strategy.head_to_head(results, left_id, right_id, key="position")
    with st.spinner("Consultando clasificaciones en Jolpica..."):
        qualifying = _jolpica_guard(jolpica.load_qualifying_results, selection.year)
    quali_h2h = (
        strategy.head_to_head(qualifying, left_id, right_id, key="position")
        if qualifying is not None and not qualifying.empty
        else {"left_wins": 0, "right_wins": 0, "compared": 0}
    )

    points = results.groupby("driverId")["points"].sum()
    left_points = float(points.get(left_id, 0.0))
    right_points = float(points.get(right_id, 0.0))

    metrics = st.columns(3)
    metrics[0].metric(
        "Carrera (llegadas por delante)",
        f"{race_h2h['left_wins']} – {race_h2h['right_wins']}",
        f"{race_h2h['compared']} carreras comparables",
        delta_color="off",
        border=True,
    )
    metrics[1].metric(
        "Clasificacion",
        f"{quali_h2h['left_wins']} – {quali_h2h['right_wins']}",
        f"{quali_h2h['compared']} sesiones comparables",
        delta_color="off",
        border=True,
    )
    metrics[2].metric(
        "Puntos de la temporada",
        f"{left_points:.0f} – {right_points:.0f}",
        f"Diferencia {left_points - right_points:+.0f}",
        delta_color="off",
        border=True,
    )

    comparison = pd.DataFrame(
        {
            "Metrica": ["Carrera", "Clasificacion", "Puntos"],
            "Izquierda": [race_h2h["left_wins"], quali_h2h["left_wins"], left_points],
            "Derecha": [race_h2h["right_wins"], quali_h2h["right_wins"], right_points],
        }
    )
    comparison.attrs.update({"left": left_name, "right": right_name})
    with st.container(border=True):
        theme.plotly(charts.head_to_head_bars(comparison, title=f"{left_name} vs {right_name} · {team}"))

    if qualifying is None or qualifying.empty:
        st.warning(
            "No se pudieron traer las clasificaciones de Jolpica: el duelo de clasificacion "
            "queda en blanco en vez de rellenarse con otro dato.",
            icon="⚠️",
        )

    with st.container(border=True):
        st.markdown("##### Carrera a carrera")
        subset = results.loc[results["driverId"].isin([left_id, right_id])]
        pivot = subset.pivot_table(index=["round", "raceName"], columns="driverName",
                                   values="position", aggfunc="min").reset_index()
        st.dataframe(pivot, hide_index=True, height=420)


def _render_history(selection: filters.SessionSelection) -> None:
    mode = st.radio("Historico de", ["Piloto", "Circuito"], horizontal=True)

    if mode == "Piloto":
        season_results = _season_results(selection.year)
        if season_results is None or season_results.empty:
            st.warning(f"Jolpica no tiene resultados para {selection.year}.", icon="⚠️")
            return
        names = season_results[["driverId", "driverName"]].drop_duplicates().sort_values("driverName")
        chosen = st.selectbox("Piloto", names["driverName"].tolist())
        driver_id = names.loc[names["driverName"] == chosen, "driverId"].iloc[0]

        with st.spinner(f"Consultando la carrera completa de {chosen} en Jolpica..."):
            career = _jolpica_guard(jolpica.load_driver_career, driver_id)
        if career is None or career.empty:
            return

        per_season = (
            career.groupby("season")
            .agg(
                Carreras=("round", "count"),
                Puntos=("points", "sum"),
                Victorias=("position", lambda values: int((values == 1).sum())),
                Podios=("position", lambda values: int((values <= 3).sum())),
                PosicionMedia=("position", "mean"),
            )
            .reset_index()
        )
        metrics = st.columns(4)
        metrics[0].metric("Temporadas", f"{per_season['season'].nunique()}", border=True)
        metrics[1].metric("Grandes Premios", f"{int(per_season['Carreras'].sum())}", border=True)
        metrics[2].metric("Victorias", f"{int(per_season['Victorias'].sum())}", border=True)
        metrics[3].metric("Podios", f"{int(per_season['Podios'].sum())}", border=True)

        columns = st.columns(2)
        with columns[0].container(border=True):
            theme.plotly(
                charts.history_chart(
                    per_season, x="season", y="Puntos", title=f"Puntos por temporada · {chosen}",
                    y_title="Puntos",
                )
            )
        with columns[1].container(border=True):
            theme.plotly(
                charts.history_chart(
                    per_season, x="season", y="PosicionMedia",
                    title=f"Posicion media de llegada · {chosen}",
                    y_title="Posicion media", reverse_y=True,
                    color=theme.COLORS["info"],
                )
            )
        with st.container(border=True):
            st.dataframe(per_season.round(2), hide_index=True, height=360)

    else:
        circuit_id = selection.circuit_id
        if not circuit_id:
            schedule = _jolpica_guard(jolpica.load_schedule, selection.year)
            if schedule is None or schedule.empty:
                st.warning("No se pudo obtener el calendario para identificar el circuito.", icon="⚠️")
                return
            match = schedule.loc[schedule["round"] == selection.round_number]
            circuit_id = match["circuitId"].iloc[0] if not match.empty else None
        if not circuit_id:
            st.warning(
                "No se pudo identificar el circuito seleccionado en la nomenclatura de Jolpica.",
                icon="⚠️",
            )
            return

        with st.spinner(f"Consultando el historico de {circuit_id} en Jolpica..."):
            winners = _jolpica_guard(jolpica.load_circuit_winners, circuit_id)
        if winners is None or winners.empty:
            return

        wins_by_driver = winners["driverName"].value_counts()
        top_count = int(wins_by_driver.iloc[0])
        tied = wins_by_driver[wins_by_driver == top_count].index.tolist()
        metrics = st.columns(3)
        metrics[0].metric("Ediciones", f"{len(winners)}", border=True)
        metrics[1].metric("Primera", f"{int(winners['season'].min())}", border=True)
        metrics[2].metric(
            "Mas victorias",
            tied[0] if len(tied) == 1 else f"{len(tied)} empatados",
            f"{top_count} triunfos" + ("" if len(tied) == 1 else f": {', '.join(tied)}"),
            delta_color="off",
            border=True,
        )

        columns = st.columns([2, 1])
        with columns[0].container(border=True):
            winners_sorted = winners.sort_values("season")
            theme.plotly(
                charts.history_chart(
                    winners_sorted, x="season", y="grid",
                    title=f"Posicion de salida del ganador · {winners.iloc[0]['circuitName']}",
                    y_title="Puesto de partida", reverse_y=True, text="driverName",
                )
            )
        with columns[1].container(border=True):
            top = winners["driverName"].value_counts().head(12).reset_index()
            top.columns = ["Piloto", "Victorias"]
            st.dataframe(top, hide_index=True, height=440)

        with st.container(border=True):
            table = winners.sort_values("season", ascending=False)[
                ["season", "driverName", "constructorName", "grid", "laps", "timeText"]
            ]
            table.columns = ["Temporada", "Ganador", "Equipo", "Salida", "Vueltas", "Tiempo"]
            st.dataframe(table, hide_index=True, height=420)


# ----------------------------------------------------------------------- main


def main() -> None:
    theme.apply()
    cache_path = ff1.enable_cache()

    st.sidebar.markdown("## 🏁 Race Engineering")
    view = st.sidebar.radio("Vista", VIEWS, label_visibility="collapsed")
    st.sidebar.markdown("---")

    selection = filters.session_selector()

    raw: RawSession | None = None
    if selection.round_number and selection.telemetry_available:
        raw = load_raw_session(selection, show_errors=view in TELEMETRY_VIEWS)

    # El formulario necesita las vueltas para ofrecer solo los pilotos que corrieron.
    options, drivers = filters.analysis_form(
        raw.laps if raw is not None else pd.DataFrame(), selection
    )

    context: SessionContext | None = None
    if raw is not None:
        context = build_context(selection, options, drivers, raw)
    elif not selection.telemetry_available and view in TELEMETRY_VIEWS:
        st.error(
            f"**Telemetria no disponible para {selection.year}.** FastF1 publica cronometraje y "
            f"telemetria desde {ff1.TELEMETRY_MIN_YEAR}; esta vista necesita esos datos y no se "
            "rellena con estimaciones ni con datos de otra sesion.",
            icon="🚫",
        )
        st.info("Abri la vista **7 · Campeonato e historico**, que cubre desde 1950.", icon="ℹ️")

    try:
        requests_used = jolpica.get_repository().client.hourly_requests_used
    except Exception:  # noqa: BLE001 - el pie de pagina nunca debe romper la app
        requests_used = None
    filters.sidebar_footer(cache_path, requests_used)

    if view == VIEWS[6]:
        render_championship(selection, context)
        return

    if view == VIEWS[7]:
        render_predictor(selection, context)
        return

    if context is None:
        return

    renderers = {
        VIEWS[0]: render_summary,
        VIEWS[1]: render_pace,
        VIEWS[2]: render_tyres,
        VIEWS[3]: render_telemetry,
        VIEWS[4]: render_sectors,
        VIEWS[5]: render_race_trace,
    }
    renderers[view](context)



# ------------------------------------------------------ 8. predictor de carrera


@st.cache_data(ttl=60 * 60, show_spinner=False)
def prediction_dataset(seasons: tuple[int, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resultados y clasificaciones de las temporadas de entrenamiento."""
    results = pd.concat([jolpica.load_race_results(season) for season in seasons], ignore_index=True)
    qualifying = pd.concat(
        [jolpica.load_qualifying_results(season) for season in seasons], ignore_index=True
    )
    return results, qualifying


@st.cache_data(ttl=60 * 60, show_spinner=False)
def prediction_frames(
    seasons: tuple[int, ...],
    target_season: int,
    target_round: int,
    circuit_id: str,
    race_name: str,
    race_date: str | None,
    half_life: float = prediction.DEFAULT_HALF_LIFE,
    season_half_life: float = prediction.DEFAULT_SEASON_HALF_LIFE,
) -> dict:
    """Arma el historico con la carrera objetivo incluida y calcula las variables.

    La carrera objetivo se agrega como una fila sin resultado: asi el calculo
    causal le asigna exactamente lo que se sabia antes de correrla.
    """
    results, qualifying = prediction_dataset(seasons)
    already_run = not results.loc[
        (results["season"] == target_season) & (results["round"] == target_round)
    ].empty

    grid_known = True
    if already_run:
        combined = results
    else:
        upcoming = prediction.build_upcoming_rows(
            results,
            season=target_season,
            round_number=target_round,
            circuit_id=circuit_id,
            race_name=race_name,
            race_date=race_date,
        )
        upcoming, grid_known = prediction.attach_known_grid(
            upcoming, qualifying, season=target_season, round_number=target_round
        )
        combined = pd.concat([results, upcoming], ignore_index=True)

    history = prediction.prepare_history(combined, qualifying)
    # Similitud entre circuitos: informativa para la UI, no alimenta el modelo
    # (se midio y empeoraba la prediccion; ver `add_circuit_similarity_features`).
    featured = prediction.add_causal_features(
        history, half_life=half_life, season_half_life=season_half_life
    )
    # El perfil de pista se mide solo con carreras ya disputadas.
    profiles = prediction.track_profiles(history.loc[history["position"].notna()])
    featured = prediction.attach_track_profile(featured, profiles)

    traits = prediction.circuit_traits(history, profiles)
    similarity = prediction.circuit_similarity(traits)
    return {
        "featured": featured,
        "profiles": profiles,
        "similarity": similarity,
        "target_index": target_season * 100 + target_round,
        "already_run": already_run,
        "grid_known": bool(grid_known or already_run),
    }


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def weather_calibration(seasons: tuple[int, ...]) -> prediction.WeatherCalibration:
    """Mide con clima observado cuanto cambia una carrera mojada respecto de una seca.

    Requiere una consulta de archivo por carrera de entrenamiento, asi que solo
    se invoca cuando hay probabilidad de lluvia: si el pronostico da seco, el
    factor resultante seria 1.0 y no cambiaria nada.
    """
    results, qualifying = prediction_dataset(seasons)
    history = prediction.prepare_history(results, qualifying)
    circuits = jolpica.load_circuits()
    coordinates = {
        row.circuitId: (row.lat, row.lon)
        for row in circuits.itertuples()
        if pd.notna(row.lat) and pd.notna(row.lon)
    }

    races = history[["race_index", "circuitId", "date"]].drop_duplicates("race_index").dropna()
    wet_flags: dict[int, bool] = {}
    for race in races.itertuples():
        location = coordinates.get(race.circuitId)
        if not location:
            continue
        try:
            observed = weather_client.load_observed_weather(
                location[0], location[1], str(pd.to_datetime(race.date).date())
            )
        except weather_client.WeatherError:
            continue
        wet_flags[race.race_index] = (
            observed["precipitation_mm"] >= prediction.WET_THRESHOLD_MM
        )
    return prediction.calibrate_weather(history, wet_flags)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def run_backtest(
    seasons: tuple[int, ...],
    grid_known: bool,
    simulations: int,
    only_season: int | None = None,
) -> dict:
    """Backtest walk-forward cacheado: es lo mas caro de toda la vista."""
    results, qualifying = prediction_dataset(seasons)
    history = prediction.prepare_history(results, qualifying)
    featured = prediction.add_causal_features(history)
    profiles = prediction.track_profiles(history)
    featured = prediction.attach_track_profile(featured, profiles)
    return prediction.backtest(
        featured,
        min_training_races=25,
        simulations=simulations,
        grid_known=grid_known,
        only_season=only_season,
    )


def _next_unraced_round(schedule: pd.DataFrame, results: pd.DataFrame, season: int) -> int:
    """Primera ronda del calendario que todavia no tiene resultados."""
    if schedule.empty:
        return 1
    done = set(results.loc[results["season"] == season, "round"].astype(int))
    pending = [int(value) for value in schedule["round"] if int(value) not in done]
    return pending[0] if pending else int(schedule["round"].max())


def _render_weather_panel(circuits: pd.DataFrame, circuit_id: str, race_date: str | None) -> dict:
    """Panel de clima real consultado online. Devuelve el resumen para el modelo."""
    row = circuits.loc[circuits["circuitId"] == circuit_id]
    if row.empty or not race_date:
        st.warning(
            "No se pudo ubicar el circuito en el catalogo de Jolpica: la prediccion corre "
            "sin componente meteorologica.",
            icon="⚠️",
        )
        return {}

    latitude, longitude = float(row["lat"].iloc[0]), float(row["lon"].iloc[0])
    try:
        with st.spinner("Consultando el pronostico en Open-Meteo..."):
            summary = weather_client.load_race_weather(latitude, longitude, race_date)
    except weather_client.WeatherError as exc:
        st.warning(
            f"**Clima no disponible.** {exc} La prediccion sigue, pero sin ajuste por lluvia.",
            icon="⚠️",
        )
        return {}

    source_labels = {
        "pronostico": "Pronostico Open-Meteo",
        "observado": "Observado (ya disputada)",
        "climatologia": "Climatologia de años previos",
    }
    columns = st.columns(5)
    columns[0].metric("Temperatura", f"{summary['air_temp']:.1f} °C",
                      f"max {summary['air_temp_max']:.1f} °C", delta_color="off", border=True)
    columns[1].metric("Prob. de lluvia", f"{summary['rain_probability']:.0f} %",
                      f"{summary['precipitation_mm']:.1f} mm", delta_color="off", border=True)
    columns[2].metric("Humedad", f"{summary['humidity']:.0f} %", delta_color="off", border=True)
    columns[3].metric("Viento", f"{summary['wind_speed']:.0f} km/h",
                      f"nubes {summary['cloud_cover']:.0f} %", delta_color="off", border=True)
    columns[4].metric("Fuente", source_labels.get(summary.get("source"), "—"),
                      (f"a {summary['lead_days']} dias" if "lead_days" in summary else "medido"),
                      delta_color="off", border=True)

    if summary.get("source") == "climatologia":
        st.info(
            f"La carrera esta a mas de {weather_client.FORECAST_HORIZON_DAYS} dias: todavia no "
            f"existe pronostico. Se usa el promedio realmente observado en esa fecha en "
            f"{int(summary.get('samples', 0))} años anteriores, no una suposicion.",
            icon="ℹ️",
        )
    return summary


def render_predictor(selection: filters.SessionSelection, context: SessionContext | None) -> None:
    theme.header("Predictor de carrera", "Modelo probabilistico · Jolpica y Open-Meteo")

    # El predictor arranca en la temporada en curso y en la proxima carrera, no
    # en lo que este elegido en la barra lateral: es lo que se quiere predecir.
    current_year = date.today().year
    controls = st.columns([1, 1, 2])
    season = controls[0].number_input(
        "Temporada a predecir", min_value=2015, max_value=current_year, value=current_year, step=1
    )
    training_depth = controls[1].slider(
        "Temporadas de entrenamiento", min_value=2, max_value=6, value=5,
        help="Mas temporadas dan un ajuste mas estable; menos reflejan mejor el reglamento actual.",
    )
    seasons = tuple(range(int(season) - training_depth + 1, int(season) + 1))

    try:
        with st.spinner(f"Consultando {len(seasons)} temporadas en Jolpica (puede tardar)..."):
            schedule = jolpica.load_schedule(int(season))
            circuits = jolpica.load_circuits(int(season))
            results, _ = prediction_dataset(seasons)
    except jolpica.JolpicaError as exc:
        st.error(f"**Jolpica no devolvio datos.** {exc.message}", icon="🚫")
        return

    if schedule.empty:
        st.warning(f"Jolpica no tiene calendario para {season}.", icon="⚠️")
        return

    default_round = _next_unraced_round(schedule, results, int(season))
    labels = {
        int(row["round"]): f"R{int(row['round']):02d} · {row['raceName']}"
        for _, row in schedule.iterrows()
    }
    rounds = sorted(labels)
    target_round = controls[2].selectbox(
        "Gran Premio a predecir",
        rounds,
        index=rounds.index(default_round) if default_round in rounds else 0,
        format_func=lambda value: labels[value],
    )

    race = schedule.loc[schedule["round"] == target_round].iloc[0]
    circuit_id = str(race["circuitId"])
    race_date = str(race["date"]) if pd.notna(race.get("date")) else None

    with st.expander("Peso de la forma reciente", expanded=False):
        st.caption(
            "Cuanto pesa lo ultimo frente al resto del año. Bajar estos valores hace al modelo "
            "mas reactivo a un cambio de nivel (por ejemplo un paquete de mejoras a mitad de "
            "temporada); subirlos lo hace mas estable. En el backtest sobre 81 carreras, mover "
            "estas ventanas NO cambio los resultados por encima del ruido: estan aca para "
            "experimentar, no porque exista un ajuste optimo conocido."
        )
        window_columns = st.columns(2)
        half_life = window_columns[0].slider(
            "Vida media de la forma reciente (carreras)",
            min_value=1.0, max_value=8.0, value=float(prediction.DEFAULT_HALF_LIFE), step=0.5,
        )
        season_half_life = window_columns[1].slider(
            "Vida media de la forma de temporada (carreras)",
            min_value=2.0, max_value=20.0,
            value=float(prediction.DEFAULT_SEASON_HALF_LIFE), step=1.0,
        )

    st.markdown("#### Clima en el circuito")
    weather_summary = _render_weather_panel(circuits, circuit_id, race_date)

    with st.spinner("Calculando variables y ajustando el modelo..."):
        frames = prediction_frames(
            seasons, int(season), int(target_round), circuit_id,
            str(race["raceName"]), race_date, half_life, season_half_life,
        )
        featured = frames["featured"]
        entries = featured.loc[featured["race_index"] == frames["target_index"]].copy()
        training = featured.loc[
            (featured["race_index"] < frames["target_index"]) & featured["position"].notna()
        ]

    if entries.empty or training.empty:
        st.warning(
            "No hay suficientes carreras previas para ajustar el modelo con esta configuracion.",
            icon="⚠️",
        )
        return

    ranking = prediction.fit_ranking_model(training, prediction.RACE_FEATURES)
    dnf_model = prediction.fit_dnf_model(training)
    ranking.temperature = prediction.fit_temperature(training, ranking, dnf_model)

    grid_known = bool(frames["grid_known"] and entries["grid"].notna().all())
    quali_model = None
    if not grid_known:
        quali_model = prediction.fit_ranking_model(
            training, prediction.QUALI_FEATURES, order_column="quali_position"
        )
        quali_model.temperature = prediction.fit_temperature(
            training, quali_model, order_column="quali_position"
        )

    rain_probability = float(weather_summary.get("rain_probability", 0.0))
    calibration = prediction.WeatherCalibration()
    if rain_probability > 5:
        # Solo se paga el costo de traer el clima historico si la lluvia puede
        # cambiar algo: con 0 % de probabilidad el factor seria 1.0.
        with st.spinner("Midiendo el efecto de la lluvia en carreras historicas..."):
            calibration = weather_calibration(seasons)

    result = prediction.simulate_race(
        entries,
        ranking,
        dnf_model,
        simulations=20000,
        rain_probability=rain_probability,
        weather=calibration,
        grid_known=grid_known,
        quali_model=quali_model,
    )

    if rain_probability > 5:
        if calibration.measured:
            st.info(
                f"**Efecto de la lluvia, medido sobre {calibration.wet_races} carreras mojadas "
                f"y {calibration.dry_races} secas** (umbral {prediction.WET_THRESHOLD_MM:.0f} mm): "
                f"los abandonos se multiplican por {calibration.dnf_ratio:.2f} y el desorden "
                f"respecto de la parrilla por {calibration.noise_ratio:.2f}. Ese efecto se aplica "
                f"en proporcion al {rain_probability:.0f} % de probabilidad pronosticada. "
                "Con tan pocas carreras mojadas la estimacion es ruidosa: tomala como una "
                "correccion suave, no como una ley.",
                icon="🌧️",
            )
        else:
            st.warning(
                f"Hay {rain_probability:.0f} % de probabilidad de lluvia, pero en la ventana de "
                f"entrenamiento no hay suficientes carreras mojadas para medir su efecto "
                f"({calibration.wet_races} mojadas, {calibration.dry_races} secas). La simulacion "
                "corre sin ajuste por lluvia en vez de aplicar un factor inventado.",
                icon="⚠️",
            )

    mode = (
        "parrilla real publicada" if grid_known
        else "antes de la clasificacion: la parrilla tambien se simula"
    )
    st.caption(
        f"**{race['raceName']}** · {race.get('circuitName', circuit_id)} · {race_date or 's/f'} — "
        f"modo: {mode} · entrenado con {ranking.races_used} carreras de {seasons[0]}-{seasons[-1]}"
    )

    table = result.table
    leader = table.iloc[0]
    metrics = st.columns(5)
    metrics[0].metric("Favorito", str(leader["Piloto"]),
                      f"{leader['P(victoria)']:.1f} % de ganar", delta_color="off", border=True)
    podium = table.nlargest(3, "P(podio)")["Piloto"].tolist()
    metrics[1].metric("Podio mas probable", " · ".join(podium), delta_color="off", border=True)
    if "P(pole)" in table.columns:
        pole = table.nlargest(1, "P(pole)").iloc[0]
        metrics[2].metric("Pole prevista", str(pole["Piloto"]),
                          f"{pole['P(pole)']:.1f} %", delta_color="off", border=True)
    else:
        metrics[2].metric("Pole", str(entries.nsmallest(1, "grid")["code"].iloc[0]),
                          "parrilla real", delta_color="off", border=True)
    metrics[3].metric("Prob. de lluvia", f"{rain_probability:.0f} %",
                      weather_summary.get("source", "sin dato"), delta_color="off", border=True)
    metrics[4].metric("Simulaciones", f"{result.simulations:,}".replace(",", "."),
                      "Monte Carlo", delta_color="off", border=True)

    st.markdown("### Orden de llegada previsto")
    order = prediction.predicted_order(table)
    podium = order.head(3)
    podium_columns = st.columns(3)
    medals = ["🥇", "🥈", "🥉"]
    for index, (_, row) in enumerate(podium.iterrows()):
        podium_columns[index].metric(
            f"{medals[index]}  P{int(row['Pos'])}",
            str(row["Piloto"]),
            f"{row['Equipo']} · {row['P(victoria)']:.1f} % de ganar",
            delta_color="off",
            border=True,
        )

    with st.container(border=True):
        display_order = order[
            ["Pos", "Piloto", "Equipo", "P(victoria)", "P(podio)", "P(puntos)", "P(abandono)"]
        ].copy()
        for column in ("P(victoria)", "P(podio)", "P(puntos)", "P(abandono)"):
            display_order[column] = display_order[column].round(1)
        st.dataframe(
            display_order,
            hide_index=True,
            height=min(760, 36 * len(display_order) + 40),
            column_config={
                "Pos": st.column_config.NumberColumn("Pos", width="small"),
                "P(victoria)": st.column_config.ProgressColumn(
                    "Gana", format="%.1f %%", min_value=0.0, max_value=100.0
                ),
                "P(podio)": st.column_config.ProgressColumn(
                    "Podio", format="%.1f %%", min_value=0.0, max_value=100.0
                ),
                "P(puntos)": st.column_config.ProgressColumn(
                    "Puntos", format="%.1f %%", min_value=0.0, max_value=100.0
                ),
                "P(abandono)": st.column_config.NumberColumn("Abandona", format="%.1f %%"),
            },
        )
        theme.caption(
            "Ordenado por posicion esperada en las simulaciones. Las barras son la probabilidad "
            "de cada resultado para ese piloto, no su posicion."
        )

    team_colors = charts.match_team_colors(
        sorted(table["Equipo"].dropna().astype(str).unique().tolist()),
        ff1.team_colors(context.session) if context is not None else None,
    )

    columns = st.columns(2)
    with columns[0].container(border=True):
        theme.plotly(
            charts.probability_bars(table, column="P(victoria)", colors=team_colors,
                                    title="Probabilidad de ganar")
        )
    with columns[1].container(border=True):
        theme.plotly(
            charts.probability_bars(table, column="P(podio)", colors=team_colors,
                                    title="Probabilidad de podio")
        )

    if "P(pole)" in table.columns:
        with st.container(border=True):
            theme.plotly(
                charts.probability_bars(table, column="P(pole)", colors=team_colors,
                                        title="Probabilidad de pole")
            )

    with st.container(border=True):
        theme.plotly(charts.position_heatmap(result.position_matrix))
        theme.caption(
            "Cada fila suma 100 %: es la distribucion completa del resultado de ese piloto, "
            "no solo su posicion mas probable."
        )

    with st.expander("Tabla completa con todas las columnas", expanded=False):
        display = table.drop(columns=["driverId"]).copy()
        for column in display.columns:
            if display[column].dtype.kind == "f":
                display[column] = display[column].round(2)
        st.dataframe(display, hide_index=True, height=520)

    theme.estimate_note(
        "Todo lo de arriba sale de un modelo estadistico ajustado sobre carreras pasadas. "
        "Las probabilidades son del modelo, no de la realidad.",
        result.assumptions,
    )

    if frames["already_run"]:
        actual = entries.sort_values("position")[["code", "position"]].head(3)
        st.success(
            "Esta carrera ya se disputo. Podio real: "
            + " · ".join(f"{row['code']} (P{int(row['position'])})" for _, row in actual.iterrows())
            + ". El modelo se entreno solo con carreras anteriores a ella.",
            icon="✅",
        )

    _render_track_panel(frames["profiles"], circuit_id, frames.get("similarity"))
    _render_model_panel(ranking, quali_model, dnf_model, seasons, grid_known, int(season))


def _render_track_panel(
    profiles: pd.DataFrame, circuit_id: str, similarity: pd.DataFrame | None = None
) -> None:
    st.markdown("#### Tipo de pista")
    current = profiles.loc[profiles["circuitId"] == circuit_id]
    columns = st.columns([2, 3])
    with columns[0]:
        if current.empty:
            st.warning(
                "Este circuito no tiene ediciones previas en la ventana de entrenamiento: "
                "se usa el comportamiento mediano del calendario.",
                icon="⚠️",
            )
        else:
            row = current.iloc[0]
            st.metric("Cambio medio de posiciones", f"{row['overtaking_index']:.2f}",
                      f"{int(row['races'])} ediciones medidas", delta_color="off", border=True)
            st.metric("Abandonos historicos", f"{row['dnf_rate'] * 100:.0f} %",
                      delta_color="off", border=True)
            st.metric("Peso de la parrilla", f"{row['grid_retention']:.2f}",
                      "correlacion parrilla-llegada", delta_color="off", border=True)
            st.caption(
                "Cuanto mas alta la correlacion, mas manda la parrilla. El modelo aprende esa "
                "diferencia y por eso no trata igual a Monaco que a Interlagos."
            )
    with columns[1].container(border=True):
        theme.plotly(charts.track_profile_scatter(profiles, highlight=circuit_id))

    if similarity is not None and not similarity.empty and circuit_id in similarity.index:
        with st.container(border=True):
            st.markdown("##### Circuitos mas parecidos a este")
            names = profiles.set_index("circuitId")["circuitName"].to_dict()
            ranked = similarity.loc[circuit_id].drop(index=circuit_id, errors="ignore")
            ranked = ranked.sort_values(ascending=False)
            table = pd.DataFrame(
                {
                    "Circuito": [names.get(cid, cid) for cid in ranked.index],
                    "Parecido": (ranked.to_numpy() * 100).round(0),
                }
            )
            top, bottom = st.columns(2)
            with top:
                st.caption("**Mas parecidos**")
                st.dataframe(table.head(6), hide_index=True, height=240)
            with bottom:
                st.caption("**Menos parecidos**")
                st.dataframe(table.tail(6).iloc[::-1], hide_index=True, height=240)
            st.caption(
                "Parecido medido con la duracion de la vuelta, cuanto se adelanta, cuanto manda "
                "la parrilla y los abandonos historicos. **Es informativo: no entra en la "
                "prediccion.** Se probo ponderar la forma de cada equipo por este parecido "
                "-la idea de que lo hecho en Imola no deberia pesar igual que lo hecho en "
                "Monaco para un callejero- y empeoraba el acierto de forma consistente."
            )


def _render_model_panel(
    ranking, quali_model, dnf_model, seasons: tuple[int, ...], grid_known: bool,
    target_season: int | None = None,
) -> None:
    st.markdown("#### Como decide el modelo")
    columns = st.columns([2, 3])
    with columns[0].container(border=True):
        theme.plotly(charts.coefficient_chart(ranking.summary))
    with columns[1]:
        st.markdown(
            "El ajuste reparte el peso solo; nada esta elegido a mano. Lo que suele quedar arriba:\n\n"
            "- **grid_score**: donde sale el piloto. De lejos, lo mas informativo.\n"
            "- **grid_x_overtaking**: cuanto pesa esa parrilla *en este circuito concreto*. El "
            "coeficiente sale negativo, es decir: donde se adelanta, la parrilla importa menos.\n"
            "- **quali_margin**: el margen en segundos, no el orden. Una pole por 0.6 s no dice lo "
            "mismo que una por 0.02 s.\n"
            "- **form_race_gain / team_race_gain**: posiciones que piloto y equipo suelen ganar el "
            "domingo respecto de donde salen (salidas, ritmo de carrera, estrategia).\n"
            "- **team_trend**: si el equipo viene mejorando. **Es el unico rastro posible de las "
            "actualizaciones tecnicas**: no existe ninguna API publica de mejoras, asi que el "
            "modelo no las lee, las infiere del rendimiento.\n\n"
            "\n\nCon la parrilla ya conocida, las variables de forma aportan poco: **la "
            "parrilla ya contiene esa informacion**. Donde la forma reciente si manda es en el "
            "modelo de clasificacion, en el que el margen de qualy de las ultimas carreras es "
            "la variable numero uno.\n\n"
            f"Temperatura de calibracion ajustada: **{ranking.temperature:.2f}** (mayor que 1 "
            "significa que el ajuste base repartia demasiada probabilidad y hubo que separar mas "
            "a los de cabeza)."
        )
        st.caption(
            f"Modelo de abandono ajustado aparte sobre la misma ventana "
            f"(tasa base {dnf_model.base_rate * 100:.1f} %). "
            + ("Modelo de clasificacion activo para simular la parrilla." if quali_model else "")
        )

    st.markdown("#### Backtest: cuanto acierta de verdad")
    st.markdown(
        "Esto **pone a prueba el modelo con carreras que ya se corrieron**, para saber si sus "
        "predicciones sirven de algo.\n\n"
        "Recorre esas carreras una por una y, en cada una, vuelve a entrenar el modelo desde "
        "cero **usando solo las carreras anteriores**: para predecir Monza, el modelo no sabe "
        "nada de Monza ni de lo que vino despues. Luego compara lo que predijo contra lo que "
        "realmente paso.\n\n"
        "Se mide contra dos reglas tontas que cualquiera puede aplicar sin modelo: **«gana el "
        "poleman»** y **«gana el lider del campeonato»**. Si el modelo no le gana a esas dos, "
        "no aporta nada."
    )
    st.caption(
        "**Alcance**: solo la temporada elegida, o las 81 carreras disponibles. "
        "**Modo a evaluar**: «con parrilla real» usa la parrilla de clasificacion (mas facil, "
        "porque la parrilla ya dice mucho); «antes de la clasificacion» no la usa y ademas "
        "predice la pole, que es la prediccion de verdad. «El modo de arriba» repite el que "
        "esta usando la prediccion mostrada."
    )
    options = ["Todas las temporadas"]
    if target_season is not None:
        options.insert(0, f"Solo {target_season}")
    scope = st.segmented_control("Alcance", options, default=options[0]) or options[0]
    only_season = target_season if scope.startswith("Solo") else None

    modes = ["El modo de arriba", "Con parrilla real", "Antes de la clasificacion"]
    mode_choice = st.segmented_control("Modo a evaluar", modes, default=modes[0]) or modes[0]
    evaluate_with_grid = {
        "El modo de arriba": grid_known,
        "Con parrilla real": True,
        "Antes de la clasificacion": False,
    }[mode_choice]

    if not st.button("Ejecutar backtest", type="primary"):
        st.caption("Tarda entre 10 y 40 segundos segun el alcance. El resultado queda cacheado.")
        return

    with st.spinner("Reentrenando carrera por carrera..."):
        outcome = run_backtest(seasons, evaluate_with_grid, 3000, only_season)
    grid_known = evaluate_with_grid

    metrics = outcome.get("metrics", {})
    if not metrics:
        st.warning("No hay suficientes carreras para un backtest con esta ventana.", icon="⚠️")
        return

    if not grid_known:
        st.info(
            "El backtest corre en el mismo modo que la prediccion de arriba: **antes de la "
            "clasificacion**. La referencia justa aqui es «gana el lider del campeonato», porque "
            "la referencia de parrilla usa un dato que todavia no existe cuando se predice. Se "
            "muestra igual, pero comparar contra ella no es limpio.",
            icon="ℹ️",
        )

    columns = st.columns([3, 2])
    with columns[0].container(border=True):
        theme.plotly(charts.backtest_comparison(metrics))
    with columns[1]:
        model_hit = metrics.get("Acierto del ganador (modelo)", 0)
        grid_hit = metrics.get("Acierto del ganador (parrilla)", 0)
        standings_hit = metrics.get("Acierto del ganador (campeonato)", 0)
        # Antes de la clasificacion la parrilla no existe: la referencia honesta
        # es el campeonato, que solo usa informacion disponible en ese momento.
        reference, reference_label = (
            (grid_hit, "vs la parrilla") if grid_known else (standings_hit, "vs el campeonato")
        )
        st.metric("Carreras evaluadas", f"{metrics.get('Carreras evaluadas', 0)}",
                  delta_color="off", border=True)
        st.metric("Acierto del ganador", f"{model_hit:.1f} %",
                  f"{model_hit - reference:+.1f} pp {reference_label}", border=True)
        st.metric("Prob. media dada al ganador real",
                  f"{metrics.get('Prob. media asignada al ganador real', 0):.1f} %",
                  "calibracion", delta_color="off", border=True)
        st.metric("Brier de victoria",
                  f"{metrics.get('Brier de victoria (menor es mejor)', 0):.4f}",
                  "menor es mejor", delta_color="off", border=True)

    frame = pd.DataFrame(
        {
            "Metrica": list(metrics.keys()),
            "Valor": [round(value, 3) if isinstance(value, float) else value
                      for value in metrics.values()],
        }
    )
    with st.container(border=True):
        st.dataframe(frame, hide_index=True, height=360)

    if "Acierto de la pole" in metrics:
        st.caption(
            f"En este modo el modelo tambien predice la clasificacion: acierta la pole el "
            f"**{metrics['Acierto de la pole']:.1f} %** de las veces."
        )

    podium_model = metrics.get("Podio: aciertos de 3 (modelo)", 0)
    podium_grid = metrics.get("Podio: aciertos de 3 (parrilla)", 0)
    if grid_known and podium_model < podium_grid:
        st.warning(
            f"**El modelo NO le gana a la parrilla para acertar el podio** ({podium_model:.2f} "
            f"contra {podium_grid:.2f} aciertos de 3). Donde si aporta es en dar probabilidades "
            "calibradas de todo el peloton y en funcionar antes de que exista una parrilla que "
            "copiar.",
            icon="⚠️",
        )

    detail = outcome.get("detail")
    if detail is not None and not detail.empty:
        with st.container(border=True):
            st.markdown("##### Carrera por carrera")
            wanted = ["season", "round", "raceName", "winner", "model_pick", "model_hit",
                      "grid_pick", "grid_hit", "winner_probability"]
            view = detail[[column for column in wanted if column in detail.columns]].copy()
            view["winner_probability"] = (view["winner_probability"] * 100).round(1)
            view.columns = ["Temporada", "Ronda", "Gran Premio", "Ganador real",
                            "Favorito del modelo", "Acierto", "Poleman", "Acierto parrilla",
                            "Prob. al ganador (%)"]
            st.dataframe(view.iloc[::-1], hide_index=True, height=420)


if __name__ == "__main__":
    main()
