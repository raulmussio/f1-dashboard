"""Constructores de figuras Plotly.

Responsabilidad unica: convertir DataFrames ya calculados en figuras. No
carga datos ni calcula metricas; todo lo que dibuja llega como argumento.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ui.theme import COLORS, SPEED_SCALE, TRACK_STATUS_COLORS

# FastF1 distingue compañeros de equipo con linea solida vs punteada.
DASH_BY_LINESTYLE = {"solid": "solid", "dashed": "dash", "dotted": "dot", "dashdot": "dashdot"}

Styles = dict[str, dict[str, str]]


def _color(styles: Styles, driver: str) -> str:
    return styles.get(driver, {}).get("color", COLORS["info"])


def _dash(styles: Styles, driver: str) -> str:
    return DASH_BY_LINESTYLE.get(styles.get(driver, {}).get("linestyle", "solid"), "solid")


def _empty(message: str) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        showarrow=False,
        font=dict(size=14, color=COLORS["text_soft"]),
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
    )
    figure.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), height=260)
    return figure


def _add_flag_shading(figure: go.Figure, flags: pd.DataFrame, *, x_column: str = "StartLap") -> None:
    """Sombrea tramos de bandera amarilla, SC, VSC y roja sobre el eje X de vueltas."""
    if flags is None or flags.empty:
        return
    end_column = "EndLap" if x_column == "StartLap" else "EndSeconds"
    for _, row in flags.iterrows():
        start, end = row[x_column], row[end_column]
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            continue
        figure.add_vrect(
            x0=start,
            x1=end,
            fillcolor=TRACK_STATUS_COLORS.get(row["Label"], "rgba(139,151,168,0.12)"),
            line_width=0,
            layer="below",
            annotation_text=row["Label"],
            annotation_position="top left",
            annotation=dict(font=dict(size=11, color=COLORS["text_soft"])),
        )


# ------------------------------------------------------------------- ritmo


def lap_time_distribution(
    laps: pd.DataFrame,
    styles: Styles,
    *,
    kind: str = "box",
    column: str = "LapTimeSeconds",
    order: list[str] | None = None,
    title: str = "Distribucion de tiempos por vuelta",
) -> go.Figure:
    """Box plot o violin de los tiempos por vuelta de cada piloto."""
    if laps.empty:
        return _empty("Sin vueltas que cumplan los filtros seleccionados.")

    drivers = order or laps.groupby("Driver")[column].median().sort_values().index.tolist()
    figure = go.Figure()
    for driver in drivers:
        values = laps.loc[laps["Driver"] == driver, column].dropna()
        if values.empty:
            continue
        color = _color(styles, driver)
        common = dict(
            name=driver,
            y=values,
            x=[driver] * len(values),
            marker=dict(color=color, size=3, opacity=0.55),
            line=dict(color=color, width=1.4),
            hovertemplate=f"<b>{driver}</b><br>%{{y:.3f}} s<extra></extra>",
        )
        if kind == "violin":
            figure.add_trace(
                go.Violin(
                    **common,
                    fillcolor=color,
                    opacity=0.35,
                    points="outliers",
                    box_visible=True,
                    meanline_visible=True,
                    spanmode="hard",
                )
            )
        else:
            figure.add_trace(
                go.Box(**common, fillcolor=color, opacity=0.35, boxpoints="outliers", boxmean=True)
            )

    figure.update_layout(
        title=title,
        showlegend=False,
        yaxis_title="Tiempo por vuelta (s)",
        xaxis_title=None,
        height=420,
    )
    return figure


def pace_ranking(
    summary: pd.DataFrame,
    styles: Styles,
    *,
    value_column: str = "Median",
    comparison_column: str | None = None,
    title: str = "Ritmo medio",
) -> go.Figure:
    """Barras horizontales de ritmo por piloto, con una serie de comparacion opcional."""
    if summary.empty:
        return _empty("Sin datos de ritmo para los filtros seleccionados.")

    frame = summary.sort_values(value_column, ascending=False)
    reference = frame[value_column].min()
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            y=frame["Driver"],
            x=frame[value_column] - reference,
            orientation="h",
            marker=dict(color=[_color(styles, d) for d in frame["Driver"]]),
            customdata=frame[value_column],
            hovertemplate="<b>%{y}</b><br>%{customdata:.3f} s<br>+%{x:.3f} s vs mejor<extra></extra>",
            name="Bruto",
        )
    )
    if comparison_column and comparison_column in frame.columns:
        comparison_reference = frame[comparison_column].min()
        figure.add_trace(
            go.Scatter(
                y=frame["Driver"],
                x=frame[comparison_column] - comparison_reference,
                mode="markers",
                marker=dict(symbol="diamond", size=9, color=COLORS["warning"],
                            line=dict(width=1, color=COLORS["background"])),
                customdata=frame[comparison_column],
                hovertemplate="<b>%{y}</b><br>Corregido: %{customdata:.3f} s<extra></extra>",
                name="Corregido por combustible",
            )
        )
        figure.update_layout(showlegend=True)

    figure.update_layout(
        title=title,
        xaxis_title="Delta al mejor ritmo (s)",
        yaxis_title=None,
        height=max(320, 22 * len(frame) + 120),
        bargap=0.3,
    )
    return figure


def track_evolution(evolution: pd.DataFrame, *, title: str = "Evolucion de pista") -> go.Figure:
    """Referencia del peloton por vuelta y su media movil."""
    if evolution.empty:
        return _empty("Sin vueltas suficientes para medir evolucion de pista.")

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=evolution["LapNumber"],
            y=evolution["FieldLapTime"],
            mode="markers",
            marker=dict(size=4, color=COLORS["text_muted"], opacity=0.5),
            name="Mediana del peloton",
            hovertemplate="Vuelta %{x}<br>%{y:.3f} s<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=evolution["LapNumber"],
            y=evolution["Smoothed"],
            mode="lines",
            line=dict(color=COLORS["accent"], width=2.4),
            name="Media movil",
            hovertemplate="Vuelta %{x}<br>%{y:.3f} s<extra></extra>",
        )
    )
    figure.update_layout(
        title=title, xaxis_title="Vuelta", yaxis_title="Tiempo por vuelta (s)", height=340
    )
    return figure


# -------------------------------------------------------------- neumaticos


def stint_gantt(
    stints: pd.DataFrame,
    compound_colors: dict[str, str],
    *,
    order: list[str] | None = None,
    title: str = "Stints por piloto",
) -> go.Figure:
    """Diagrama tipo Gantt de stints, coloreado por compuesto."""
    if stints.empty:
        return _empty("Sin informacion de stints en esta sesion.")

    drivers = order or sorted(stints["Driver"].unique())
    figure = go.Figure()
    seen: set[str] = set()
    for _, row in stints.iterrows():
        compound = row["Compound"]
        figure.add_trace(
            go.Bar(
                y=[row["Driver"]],
                x=[row["Laps"]],
                base=[row["LapStart"] - 1],
                orientation="h",
                marker=dict(
                    color=compound_colors.get(compound, COLORS["text_muted"]),
                    line=dict(color=COLORS["background"], width=1),
                ),
                name=compound,
                legendgroup=compound,
                showlegend=compound not in seen,
                hovertemplate=(
                    f"<b>{row['Driver']}</b> · stint {row['Stint']}<br>{compound}<br>"
                    f"Vueltas {row['LapStart']}-{row['LapEnd']} ({row['Laps']})<br>"
                    f"Mediana: {row['MedianLapTime']:.3f} s<extra></extra>"
                ),
            )
        )
        seen.add(compound)

    figure.update_layout(
        title=title,
        barmode="stack",
        xaxis_title="Vuelta",
        yaxis=dict(categoryorder="array", categoryarray=drivers[::-1]),
        height=max(360, 24 * len(drivers) + 120),
        bargap=0.25,
    )
    return figure


def degradation_scatter(
    laps: pd.DataFrame,
    degradation: pd.DataFrame,
    *,
    compound_colors: dict[str, str],
    column: str = "LapTimeSeconds",
    title: str = "Degradacion por stint",
) -> go.Figure:
    """Nube de puntos por vida del neumatico con la recta ajustada y su banda de confianza."""
    if laps.empty or degradation.empty:
        return _empty("No hay stints con vueltas limpias suficientes para ajustar una recta.")

    figure = go.Figure()
    for _, row in degradation.iterrows():
        subset = laps.loc[(laps["Driver"] == row["Driver"]) & (laps["Stint"] == row["Stint"])]
        subset = subset.dropna(subset=[column])
        if subset.empty:
            continue
        x = subset["TyreLife"] if subset["TyreLife"].notna().all() else subset["LapNumber"]
        color = compound_colors.get(row["Compound"], COLORS["info"])
        label = f"{row['Driver']} S{row['Stint']} · {row['Compound']}"

        figure.add_trace(
            go.Scatter(
                x=x,
                y=subset[column],
                mode="markers",
                marker=dict(size=6, color=color, opacity=0.75,
                            line=dict(width=0.6, color=COLORS["background"])),
                name=label,
                legendgroup=label,
                hovertemplate=f"<b>{label}</b><br>Vida: %{{x:.0f}}<br>%{{y:.3f}} s<extra></extra>",
            )
        )

        grid = np.linspace(float(x.min()), float(x.max()), 40)
        fitted = row["Intercept"] + row["SlopeSecPerLap"] * grid
        low = row["Intercept"] + row["CILow"] * grid
        high = row["Intercept"] + row["CIHigh"] * grid
        if np.isfinite(low).all() and np.isfinite(high).all():
            figure.add_trace(
                go.Scatter(
                    x=np.concatenate([grid, grid[::-1]]),
                    y=np.concatenate([high, low[::-1]]),
                    fill="toself",
                    fillcolor="rgba(174,185,201,0.10)",
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=label,
                )
            )
        figure.add_trace(
            go.Scatter(
                x=grid,
                y=fitted,
                mode="lines",
                line=dict(color=color, width=2.2),
                showlegend=False,
                legendgroup=label,
                hovertemplate=(
                    f"<b>{label}</b><br>{row['SlopeSecPerLap']:+.3f} s/vuelta"
                    f" (IC95 {row['CILow']:+.3f} / {row['CIHigh']:+.3f})<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        title=title,
        xaxis_title="Vida del neumatico (vueltas)",
        yaxis_title="Tiempo (s)",
        height=440,
        # Un stint por entrada: en horizontal la leyenda se monta sobre el titulo.
        legend=dict(orientation="v", x=1.01, xanchor="left", y=1, yanchor="top"),
        margin=dict(l=56, r=190, t=56, b=48),
    )
    return figure


def degradation_ranking(degradation: pd.DataFrame, compound_colors: dict[str, str]) -> go.Figure:
    """Pendiente de degradacion por stint con barras de error del IC 95 %."""
    if degradation.empty:
        return _empty("Sin stints ajustables.")

    frame = degradation.sort_values("SlopeSecPerLap")
    labels = frame["Driver"] + " S" + frame["Stint"].astype(str)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=frame["SlopeSecPerLap"],
            y=labels,
            mode="markers",
            marker=dict(
                size=9,
                color=[compound_colors.get(c, COLORS["info"]) for c in frame["Compound"]],
                line=dict(width=1, color=COLORS["background"]),
            ),
            error_x=dict(
                type="data",
                symmetric=False,
                array=frame["CIHigh"] - frame["SlopeSecPerLap"],
                arrayminus=frame["SlopeSecPerLap"] - frame["CILow"],
                color=COLORS["text_muted"],
                thickness=1.1,
                width=3,
            ),
            customdata=np.stack([frame["Compound"], frame["R2"], frame["Laps"]], axis=-1),
            hovertemplate=(
                "<b>%{y}</b><br>%{customdata[0]}<br>%{x:+.3f} s/vuelta"
                "<br>R2 %{customdata[1]:.2f} · %{customdata[2]} vueltas<extra></extra>"
            ),
        )
    )
    figure.add_vline(x=0, line=dict(color=COLORS["border"], width=1, dash="dot"))
    figure.update_layout(
        title="Pendiente de degradacion por stint (IC 95 %)",
        xaxis_title="s / vuelta",
        yaxis_title=None,
        height=max(340, 18 * len(frame) + 120),
        showlegend=False,
    )
    return figure


def strategy_delta(cumulative: pd.DataFrame, styles: Styles, reference: str) -> go.Figure:
    """Tiempo acumulado de cada piloto relativo al de referencia, vuelta a vuelta."""
    if cumulative.empty:
        return _empty("Selecciona al menos dos pilotos para comparar estrategias.")

    figure = go.Figure()
    for driver, group in cumulative.groupby("Driver"):
        group = group.sort_values("LapNumber")
        figure.add_trace(
            go.Scatter(
                x=group["LapNumber"],
                y=-group["DeltaSeconds"],
                mode="lines",
                name=driver,
                line=dict(color=_color(styles, driver), width=2, dash=_dash(styles, driver)),
                hovertemplate=f"<b>{driver}</b><br>Vuelta %{{x}}<br>%{{y:+.2f}} s vs {reference}<extra></extra>",
            )
        )
    figure.add_hline(y=0, line=dict(color=COLORS["border"], width=1, dash="dot"))
    figure.update_layout(
        title=f"Ventaja acumulada contra {reference}",
        xaxis_title="Vuelta",
        yaxis_title=f"Segundos por delante de {reference}",
        height=380,
    )
    return figure


def undercut_projection(simulation: dict) -> go.Figure:
    """Hueco proyectado vuelta a vuelta durante la ventana del undercut."""
    timeline = simulation.get("timeline")
    if timeline is None or timeline.empty:
        return _empty("Sin datos para simular.")

    attacker = simulation["attacker"]
    defender = simulation["defender"]
    laps = [0, *timeline["LapInWindow"].tolist()]
    gaps = [simulation["initial_gap"], *timeline["ProjectedGap"].tolist()]
    colors = [COLORS["positive"] if value < 0 else COLORS["negative"] for value in gaps]

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=laps,
            y=gaps,
            mode="lines+markers",
            line=dict(color=COLORS["accent"], width=2.4),
            marker=dict(size=9, color=colors, line=dict(width=1, color=COLORS["background"])),
            hovertemplate="Vuelta +%{x}<br>%{y:+.2f} s<extra></extra>",
            name="Hueco proyectado",
        )
    )
    figure.add_hline(
        y=0,
        line=dict(color=COLORS["text_muted"], width=1.4, dash="dash"),
        annotation_text="Cruce de posiciones",
        annotation_position="top left",
        annotation=dict(font=dict(size=11, color=COLORS["text_soft"])),
    )
    figure.update_layout(
        title=f"Proyeccion: {attacker} para {len(timeline)} vuelta(s) antes que {defender}",
        xaxis_title="Vueltas desde la parada del atacante",
        yaxis_title=f"Hueco (s) — negativo = {attacker} delante",
        height=340,
        showlegend=False,
    )
    return figure


# -------------------------------------------------------------- telemetria


def telemetry_channels(
    reference: pd.DataFrame,
    comparison: pd.DataFrame | None,
    *,
    label_reference: str,
    label_comparison: str | None = None,
    color_reference: str,
    color_comparison: str | None = None,
    dash_reference: str = "solid",
    dash_comparison: str = "solid",
    channels: list[str] | None = None,
) -> go.Figure:
    """Los seis canales superpuestos contra distancia, en subplots compartidos."""
    channels = channels or ["Speed", "Throttle", "Brake", "nGear", "RPM", "DRS"]
    titles = {
        "Speed": "Velocidad (km/h)",
        "Throttle": "Acelerador (%)",
        "Brake": "Freno",
        "nGear": "Marcha",
        "RPM": "RPM",
        "DRS": "DRS",
    }
    available = [channel for channel in channels if channel in reference.columns]
    if reference.empty or not available:
        return _empty("Sin canales de telemetria disponibles para esta vuelta.")

    figure = make_subplots(
        rows=len(available),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.022,
        subplot_titles=[titles.get(channel, channel) for channel in available],
    )

    for index, channel in enumerate(available, start=1):
        figure.add_trace(
            go.Scattergl(
                x=reference["Distance"],
                y=reference[channel],
                mode="lines",
                name=label_reference,
                legendgroup=label_reference,
                showlegend=index == 1,
                line=dict(color=color_reference, width=1.6, dash=dash_reference),
                hovertemplate=f"<b>{label_reference}</b><br>%{{x:.0f}} m<br>%{{y}}<extra></extra>",
            ),
            row=index,
            col=1,
        )
        if comparison is not None and not comparison.empty and channel in comparison.columns:
            figure.add_trace(
                go.Scattergl(
                    x=comparison["Distance"],
                    y=comparison[channel],
                    mode="lines",
                    name=label_comparison or "Comparacion",
                    legendgroup=label_comparison or "Comparacion",
                    showlegend=index == 1,
                    line=dict(color=color_comparison or COLORS["warning"], width=1.6, dash=dash_comparison),
                    hovertemplate=(
                        f"<b>{label_comparison}</b><br>%{{x:.0f}} m<br>%{{y}}<extra></extra>"
                    ),
                ),
                row=index,
                col=1,
            )

    figure.update_xaxes(title_text="Distancia (m)", row=len(available), col=1)
    figure.update_layout(height=150 * len(available) + 90, margin=dict(t=60, b=44, l=52, r=20))
    for annotation in figure.layout.annotations:
        annotation.font.size = 12
        annotation.font.color = COLORS["text_soft"]
    return figure


def delta_trace(
    delta: pd.DataFrame,
    *,
    label_reference: str,
    label_comparison: str,
    color: str = COLORS["accent"],
) -> go.Figure:
    """Delta de tiempo acumulado entre dos vueltas contra distancia."""
    if delta.empty:
        return _empty("No hay tramo comun de distancia entre las dos vueltas.")

    figure = go.Figure()
    figure.add_trace(
        go.Scattergl(
            x=delta["Distance"],
            y=delta["DeltaSeconds"],
            mode="lines",
            line=dict(color=color, width=2),
            fill="tozeroy",
            fillcolor="rgba(225,6,0,0.12)",
            hovertemplate="%{x:.0f} m<br>%{y:+.3f} s<extra></extra>",
            name="Delta",
        )
    )
    figure.add_hline(y=0, line=dict(color=COLORS["border"], width=1, dash="dot"))
    figure.update_layout(
        title=f"Delta acumulado · positivo = {label_comparison} pierde contra {label_reference}",
        xaxis_title="Distancia (m)",
        yaxis_title="Delta (s)",
        height=300,
        showlegend=False,
    )
    return figure


def track_map_speed(points: pd.DataFrame, *, title: str = "Trazado coloreado por velocidad") -> go.Figure:
    """Mapa del circuito dibujado con las coordenadas X/Y de la telemetria."""
    if points.empty:
        return _empty("Sin coordenadas de posicion para esta vuelta.")

    figure = go.Figure()
    figure.add_trace(
        go.Scattergl(
            x=points["X"],
            y=points["Y"],
            mode="markers",
            marker=dict(
                size=5,
                color=points["Speed"],
                colorscale=SPEED_SCALE,
                showscale=True,
                colorbar=dict(
                    title=dict(text="km/h", font=dict(size=12, color=COLORS["text_soft"])),
                    thickness=11,
                    len=0.7,
                    tickfont=dict(size=11, color=COLORS["text_muted"]),
                ),
            ),
            hovertemplate="%{customdata:.0f} m<br>%{marker.color:.0f} km/h<extra></extra>",
            customdata=points["Distance"],
        )
    )
    figure.update_layout(
        title=title,
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False),
        height=440,
        showlegend=False,
    )
    return figure


def track_map_dominance(
    points: pd.DataFrame,
    dominance: pd.DataFrame,
    *,
    colors: dict[str, str],
    title: str = "Dominancia por mini-sector",
) -> go.Figure:
    """Trazado coloreado con el piloto mas rapido en cada mini-sector."""
    if points.empty or dominance.empty or "Minisector" not in points.columns:
        return _empty("Sin datos suficientes para calcular mini-sectores.")

    winners = dominance.set_index("Minisector")["Winner"].to_dict()
    deltas = dominance.set_index("Minisector")["SpeedDelta"].to_dict()

    figure = go.Figure()
    shown: set[str] = set()
    for sector, group in points.groupby("Minisector"):
        winner = winners.get(sector)
        if winner is None:
            continue
        delta = deltas.get(sector, 0.0)
        figure.add_trace(
            go.Scattergl(
                x=group["X"],
                y=group["Y"],
                mode="markers",
                marker=dict(size=5, color=colors.get(winner, COLORS["info"])),
                name=winner,
                legendgroup=winner,
                showlegend=winner not in shown,
                hovertemplate=(
                    f"Mini-sector {sector}<br><b>{winner}</b> mas rapido"
                    f"<br>Delta medio: {abs(delta):.1f} km/h<extra></extra>"
                ),
            )
        )
        shown.add(winner)

    figure.update_layout(
        title=title,
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False),
        height=440,
    )
    return figure


# ----------------------------------------------------------------- sectores


def sector_comparison(sectors: pd.DataFrame, styles: Styles) -> go.Figure:
    """Mejor tiempo de cada piloto en cada sector, como delta al mejor absoluto."""
    if sectors.empty:
        return _empty("Sin tiempos por sector en esta sesion.")

    figure = make_subplots(rows=1, cols=3, shared_yaxes=True,
                           subplot_titles=["Sector 1", "Sector 2", "Sector 3"])
    for index, sector in enumerate(["S1", "S2", "S3"], start=1):
        frame = sectors.dropna(subset=[sector]).sort_values(sector, ascending=False)
        if frame.empty:
            continue
        best = frame[sector].min()
        figure.add_trace(
            go.Bar(
                y=frame["Driver"],
                x=frame[sector] - best,
                orientation="h",
                marker=dict(color=[_color(styles, d) for d in frame["Driver"]]),
                customdata=frame[sector],
                hovertemplate="<b>%{y}</b><br>%{customdata:.3f} s (+%{x:.3f})<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=index,
        )
    figure.update_layout(
        title="Mejor tiempo por sector (delta al mejor)",
        height=max(360, 20 * sectors["Driver"].nunique() + 140),
        bargap=0.3,
    )
    figure.update_xaxes(title_text="Delta (s)")
    for annotation in figure.layout.annotations:
        annotation.font.size = 12
        annotation.font.color = COLORS["text_soft"]
    return figure


def ideal_vs_actual(theoretical: pd.DataFrame, styles: Styles) -> go.Figure:
    """Vuelta teorica ideal contra la vuelta real mas rapida de cada piloto."""
    if theoretical.empty:
        return _empty("Sin datos de sectores para construir la vuelta ideal.")

    frame = theoretical.dropna(subset=["Ideal", "Actual"]).sort_values("Ideal")
    figure = go.Figure()
    for _, row in frame.iterrows():
        figure.add_trace(
            go.Scatter(
                x=[row["Ideal"], row["Actual"]],
                y=[row["Driver"], row["Driver"]],
                mode="lines",
                line=dict(color=COLORS["border"], width=2),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    figure.add_trace(
        go.Scatter(
            x=frame["Ideal"],
            y=frame["Driver"],
            mode="markers",
            marker=dict(size=10, color=[_color(styles, d) for d in frame["Driver"]],
                        symbol="diamond", line=dict(width=1, color=COLORS["background"])),
            name="Vuelta ideal",
            hovertemplate="<b>%{y}</b><br>Ideal: %{x:.3f} s<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame["Actual"],
            y=frame["Driver"],
            mode="markers",
            marker=dict(size=9, color=COLORS["panel"], symbol="circle",
                        line=dict(width=2, color=COLORS["text_muted"])),
            name="Vuelta real mas rapida",
            customdata=frame["Delta"],
            hovertemplate="<b>%{y}</b><br>Real: %{x:.3f} s<br>Deja %{customdata:.3f} s<extra></extra>",
        )
    )
    figure.update_layout(
        title="Vuelta ideal (suma de mejores sectores) vs vuelta real",
        xaxis_title="Tiempo (s)",
        height=max(340, 20 * len(frame) + 140),
        yaxis=dict(autorange="reversed"),
    )
    return figure


def consistency_chart(consistency: pd.DataFrame, styles: Styles) -> go.Figure:
    """Indice de consistencia: desvio estandar de las vueltas limpias."""
    if consistency.empty:
        return _empty("Sin vueltas limpias suficientes para medir consistencia.")

    frame = consistency.sort_values("StdDev", ascending=False)
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            y=frame["Driver"],
            x=frame["StdDev"],
            orientation="h",
            marker=dict(color=[_color(styles, d) for d in frame["Driver"]]),
            customdata=np.stack([frame["Laps"], frame["CoefVar"]], axis=-1),
            hovertemplate=(
                "<b>%{y}</b><br>Desvio: %{x:.3f} s<br>"
                "%{customdata[0]} vueltas · CV %{customdata[1]:.2f} %<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title="Consistencia (desvio estandar de vueltas limpias, menor es mejor)",
        xaxis_title="Desvio estandar (s)",
        height=max(340, 20 * len(frame) + 120),
        showlegend=False,
        bargap=0.3,
    )
    return figure


def speed_traps(sectors: pd.DataFrame, styles: Styles, column: str) -> go.Figure:
    """Velocidad maxima registrada en un punto de medicion."""
    if sectors.empty or column not in sectors.columns:
        return _empty("Sin velocidades registradas en este punto.")

    frame = sectors.dropna(subset=[column]).sort_values(column)
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            y=frame["Driver"],
            x=frame[column],
            orientation="h",
            marker=dict(color=[_color(styles, d) for d in frame["Driver"]]),
            hovertemplate="<b>%{y}</b><br>%{x:.0f} km/h<extra></extra>",
        )
    )
    figure.update_layout(
        title=f"Velocidad maxima · {column}",
        xaxis_title="km/h",
        height=max(340, 20 * len(frame) + 120),
        showlegend=False,
        bargap=0.3,
    )
    figure.update_xaxes(range=[frame[column].min() * 0.97, frame[column].max() * 1.01])
    return figure


# ------------------------------------------------------------ vuelta a vuelta


def position_chart(
    positions: pd.DataFrame,
    styles: Styles,
    *,
    flags: pd.DataFrame | None = None,
    drivers: list[str] | None = None,
) -> go.Figure:
    """Posicion en pista vuelta a vuelta."""
    if positions.empty:
        return _empty("Sin datos de posicion en esta sesion.")

    figure = go.Figure()
    _add_flag_shading(figure, flags)
    columns = drivers or list(positions.columns)
    for driver in columns:
        if driver not in positions.columns:
            continue
        series = positions[driver].dropna()
        figure.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=driver,
                line=dict(color=_color(styles, driver), width=2, dash=_dash(styles, driver)),
                hovertemplate=f"<b>{driver}</b><br>Vuelta %{{x}}<br>P%{{y:.0f}}<extra></extra>",
            )
        )
    figure.update_layout(
        title="Posicion vuelta a vuelta",
        xaxis_title="Vuelta",
        yaxis_title="Posicion",
        yaxis=dict(autorange="reversed", dtick=1),
        height=460,
    )
    return figure


def gap_chart(
    laps: pd.DataFrame,
    styles: Styles,
    *,
    column: str,
    title: str,
    flags: pd.DataFrame | None = None,
    drivers: list[str] | None = None,
    clip: float | None = None,
) -> go.Figure:
    """Hueco al lider o al coche de adelante, vuelta a vuelta."""
    if laps.empty or column not in laps.columns:
        return _empty("Sin datos de huecos en esta sesion.")

    figure = go.Figure()
    _add_flag_shading(figure, flags)
    selected = drivers or sorted(laps["Driver"].unique())
    for driver in selected:
        subset = laps.loc[laps["Driver"] == driver, ["LapNumber", column]].dropna()
        subset = subset.replace([np.inf, -np.inf], np.nan).dropna().sort_values("LapNumber")
        if subset.empty:
            continue
        values = subset[column].clip(upper=clip) if clip else subset[column]
        figure.add_trace(
            go.Scatter(
                x=subset["LapNumber"],
                y=values,
                mode="lines",
                name=driver,
                line=dict(color=_color(styles, driver), width=1.8, dash=_dash(styles, driver)),
                hovertemplate=f"<b>{driver}</b><br>Vuelta %{{x}}<br>%{{y:.2f}} s<extra></extra>",
            )
        )
    figure.update_layout(title=title, xaxis_title="Vuelta", yaxis_title="Segundos", height=420)
    return figure


# --------------------------------------------------------------- campeonato


def points_evolution(
    cumulative: pd.DataFrame,
    *,
    group_key: str,
    name_key: str,
    colors: dict[str, str] | None = None,
    title: str = "Evolucion de puntos",
) -> go.Figure:
    """Puntos acumulados ronda a ronda."""
    if cumulative.empty:
        return _empty("Sin resultados para esta temporada.")

    order = (
        cumulative.sort_values("round").groupby(name_key)["CumulativePoints"].last().sort_values(ascending=False)
    )
    figure = go.Figure()
    for rank, name in enumerate(order.index):
        group = cumulative.loc[cumulative[name_key] == name].sort_values("round")
        color = (colors or {}).get(name)
        figure.add_trace(
            go.Scatter(
                x=group["round"],
                y=group["CumulativePoints"],
                mode="lines",
                name=str(name),
                line=dict(width=2.2 if rank < 5 else 1.2, color=color),
                opacity=1.0 if rank < 5 else 0.55,
                hovertemplate=f"<b>{name}</b><br>Ronda %{{x}}<br>%{{y:.0f}} pts<extra></extra>",
            )
        )
    figure.update_layout(
        title=title, xaxis_title="Ronda", yaxis_title="Puntos acumulados", height=460
    )
    return figure


def head_to_head_bars(comparison: pd.DataFrame, *, title: str) -> go.Figure:
    """Barras enfrentadas del duelo entre compañeros de equipo."""
    if comparison.empty:
        return _empty("Sin datos para el duelo seleccionado.")

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            y=comparison["Metrica"],
            x=-comparison["Izquierda"],
            orientation="h",
            name=comparison.attrs.get("left", "Piloto A"),
            marker=dict(color=comparison.attrs.get("left_color", COLORS["info"])),
            customdata=comparison["Izquierda"],
            hovertemplate="%{y}: %{customdata}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Bar(
            y=comparison["Metrica"],
            x=comparison["Derecha"],
            orientation="h",
            name=comparison.attrs.get("right", "Piloto B"),
            marker=dict(color=comparison.attrs.get("right_color", COLORS["warning"])),
            hovertemplate="%{y}: %{x}<extra></extra>",
        )
    )
    figure.update_layout(
        title=title,
        barmode="relative",
        height=260,
        xaxis_title=None,
        xaxis=dict(showticklabels=False),
        bargap=0.35,
    )
    return figure


def history_chart(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    title: str,
    y_title: str,
    color: str = COLORS["accent"],
    reverse_y: bool = False,
    text: str | None = None,
) -> go.Figure:
    """Serie historica generica (piloto o circuito a lo largo de los años)."""
    if frame.empty:
        return _empty("Sin historico disponible para la seleccion.")

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=frame[x],
            y=frame[y],
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=7, color=color, line=dict(width=1, color=COLORS["background"])),
            text=frame[text] if text and text in frame.columns else None,
            hovertemplate=(
                "%{x}<br>%{y}" + ("<br>%{text}" if text and text in frame.columns else "") + "<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title=title,
        xaxis_title=None,
        yaxis_title=y_title,
        height=380,
        yaxis=dict(autorange="reversed") if reverse_y else None,
        showlegend=False,
    )
    return figure


# --------------------------------------------------------------- prediccion


def match_team_colors(
    names: list[str], reference: dict[str, str] | None = None
) -> dict[str, str]:
    """Empareja nombres de equipo de Jolpica con los colores oficiales de FastF1.

    Jolpica dice "Red Bull" y FastF1 "Red Bull Racing": se empareja por
    coincidencia de texto. Lo que no encaja cae en la paleta general, nunca en un
    color inventado que sugiera un equipo equivocado.
    """
    palette = [
        "#58A6FF", "#F24E1E", "#45C4B0", "#F2C14E", "#BC8CFF",
        "#3FB950", "#FF7B72", "#79C0FF", "#D29922", "#AEB9C9",
    ]
    reference = reference or {}
    lowered = {key.lower(): value for key, value in reference.items()}
    colors: dict[str, str] = {}
    for index, name in enumerate(names):
        needle = str(name).lower()
        match = next(
            (value for key, value in lowered.items() if needle in key or key in needle), None
        )
        colors[name] = match or palette[index % len(palette)]
    return colors


def probability_bars(
    table: pd.DataFrame,
    *,
    column: str,
    colors: dict[str, str],
    title: str,
    top: int = 12,
) -> go.Figure:
    """Barras horizontales de una probabilidad (victoria, podio, puntos o pole)."""
    if table.empty or column not in table.columns:
        return _empty("Sin prediccion disponible.")

    frame = table.nlargest(top, column).sort_values(column)
    bar_colors = [colors.get(team, COLORS["info"]) for team in frame.get("Equipo", frame["Piloto"])]
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            y=frame["Piloto"],
            x=frame[column],
            orientation="h",
            marker=dict(color=bar_colors),
            text=[f"{value:.1f} %" for value in frame[column]],
            textposition="outside",
            textfont=dict(size=12, color=COLORS["text"]),
            customdata=frame.get("Equipo", frame["Piloto"]),
            hovertemplate="<b>%{y}</b><br>%{customdata}<br>%{x:.1f} %<extra></extra>",
        )
    )
    figure.update_layout(
        title=title,
        xaxis_title="Probabilidad (%)",
        height=max(320, 26 * len(frame) + 120),
        showlegend=False,
        bargap=0.25,
        xaxis=dict(range=[0, min(100, float(frame[column].max()) * 1.25 + 4)]),
    )
    return figure


def position_heatmap(matrix: pd.DataFrame, *, top: int = 12, title: str = "") -> go.Figure:
    """Mapa de calor: probabilidad de que cada piloto acabe en cada posicion."""
    if matrix.empty:
        return _empty("Sin distribucion de posiciones.")

    ordered = matrix.iloc[:, : min(len(matrix.columns), 20)]
    ordered = ordered.loc[ordered.iloc[:, :3].sum(axis=1).sort_values(ascending=False).index[:top]]
    figure = go.Figure(
        go.Heatmap(
            z=ordered.to_numpy() * 100,
            x=list(ordered.columns),
            y=list(ordered.index),
            colorscale=[[0, COLORS["panel"]], [0.35, "#2E8BC0"], [0.7, "#F2C14E"], [1, "#F24E1E"]],
            hovertemplate="<b>%{y}</b><br>%{x}: %{z:.1f} %<extra></extra>",
            colorbar=dict(
                title=dict(text="%", font=dict(size=12, color=COLORS["text_soft"])),
                thickness=11,
                len=0.8,
                tickfont=dict(size=11, color=COLORS["text_muted"]),
            ),
        )
    )
    figure.update_layout(
        title=title or "Probabilidad de cada posicion final",
        height=max(360, 26 * len(ordered) + 140),
        yaxis=dict(autorange="reversed"),
        xaxis=dict(side="top"),
    )
    return figure


def backtest_comparison(metrics: dict) -> go.Figure:
    """Modelo contra las referencias que tiene que superar para justificarse."""
    if not metrics:
        return _empty("Sin backtest ejecutado.")

    labels = ["Modelo", "Parrilla (gana el poleman)", "Campeonato (gana el lider)"]
    values = [
        metrics.get("Acierto del ganador (modelo)", 0),
        metrics.get("Acierto del ganador (parrilla)", 0),
        metrics.get("Acierto del ganador (campeonato)", 0),
    ]
    palette = [COLORS["accent"], COLORS["info"], COLORS["text_muted"]]
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=labels,
            y=values,
            marker=dict(color=palette),
            text=[f"{value:.1f} %" for value in values],
            textposition="outside",
            textfont=dict(size=13, color=COLORS["text"]),
            hovertemplate="<b>%{x}</b><br>%{y:.1f} % de aciertos<extra></extra>",
        )
    )
    figure.update_layout(
        title="Acierto del ganador: modelo contra referencias simples",
        yaxis_title="% de carreras acertadas",
        yaxis=dict(range=[0, max(values) * 1.3 if values else 100]),
        height=340,
        showlegend=False,
        bargap=0.45,
    )
    return figure


def track_profile_scatter(profiles: pd.DataFrame, highlight: str | None = None) -> go.Figure:
    """Circuitos ordenados por cuanto se adelanta, con el actual destacado."""
    if profiles.empty:
        return _empty("Sin perfil de circuitos.")

    frame = profiles.sort_values("overtaking_index")
    is_current = frame["circuitId"] == highlight
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            y=frame["circuitName"],
            x=frame["overtaking_index"],
            orientation="h",
            marker=dict(
                color=[COLORS["accent"] if flag else COLORS["info"] for flag in is_current],
                opacity=[1.0 if flag else 0.55 for flag in is_current],
            ),
            customdata=np.stack([frame["dnf_rate"] * 100, frame["races"]], axis=-1),
            hovertemplate=(
                "<b>%{y}</b><br>Cambio medio de posiciones: %{x:.2f}"
                "<br>Abandonos: %{customdata[0]:.0f} %<br>%{customdata[1]} carreras<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title="Cuanto se adelanta en cada circuito (medido, no estimado)",
        xaxis_title="Cambio medio |parrilla - llegada|",
        height=max(380, 17 * len(frame) + 130),
        showlegend=False,
        bargap=0.25,
    )
    return figure


def coefficient_chart(summary: pd.DataFrame) -> go.Figure:
    """Peso que el ajuste le dio a cada variable."""
    if summary.empty:
        return _empty("Modelo sin ajustar.")

    frame = summary.sort_values("Coeficiente")
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            y=frame["Variable"],
            x=frame["Coeficiente"],
            orientation="h",
            marker=dict(
                color=[
                    COLORS["positive"] if value >= 0 else COLORS["negative"]
                    for value in frame["Coeficiente"]
                ]
            ),
            hovertemplate="<b>%{y}</b><br>%{x:+.3f}<extra></extra>",
        )
    )
    figure.add_vline(x=0, line=dict(color=COLORS["border"], width=1))
    figure.update_layout(
        title="Peso ajustado de cada variable (datos estandarizados)",
        xaxis_title="Coeficiente",
        height=max(300, 26 * len(frame) + 120),
        showlegend=False,
        bargap=0.3,
    )
    return figure
