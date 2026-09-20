"""Tema visual: paleta oscura, template de Plotly y CSS de la app.

Responsabilidad unica: como se ve la app. No carga datos ni calcula nada.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

TEMPLATE_NAME = "f1_dark"

# Paleta base. Oscura, de bajo contraste entre paneles, con un solo acento.
COLORS = {
    "background": "#0B0E14",
    "panel": "#131823",
    "panel_alt": "#171D2A",
    "border": "#2B3446",
    "grid": "#232B3C",
    "text": "#E6EDF3",
    # Gris legible sobre fondo oscuro: el anterior (#8B97A8) no pasaba contraste.
    "text_muted": "#AEB9C9",
    "text_soft": "#C9D3E0",
    "accent": "#E10600",
    "positive": "#3FB950",
    "negative": "#F85149",
    "warning": "#D29922",
    "info": "#58A6FF",
}

# Colores para marcar estados de pista en los graficos de carrera.
TRACK_STATUS_COLORS = {
    "Bandera amarilla": "rgba(210, 153, 34, 0.18)",
    "Safety Car": "rgba(248, 81, 73, 0.20)",
    "VSC": "rgba(88, 166, 255, 0.18)",
    "VSC terminando": "rgba(88, 166, 255, 0.10)",
    "Bandera roja": "rgba(225, 6, 0, 0.30)",
}

# Escala de velocidad para el mapa del trazado.
SPEED_SCALE = [
    [0.0, "#2C3E7B"],
    [0.35, "#2E8BC0"],
    [0.6, "#45C4B0"],
    [0.8, "#F2C14E"],
    [1.0, "#F24E1E"],
]

# Barra de herramientas de Plotly desactivada: la app no la necesita.
PLOTLY_CONFIG = {
    "displayModeBar": False,
    "displaylogo": False,
    "scrollZoom": False,
    "responsive": True,
}

FONT_FAMILY = "'Barlow Semi Condensed', 'Segoe UI', 'Roboto Condensed', sans-serif"


def _build_template() -> go.layout.Template:
    """Template de Plotly derivado de `plotly_dark` y ajustado a la paleta."""
    template = pio.templates["plotly_dark"]
    template = template.to_plotly_json()
    template = go.layout.Template(template)

    axis = dict(
        gridcolor=COLORS["grid"],
        zerolinecolor=COLORS["grid"],
        linecolor=COLORS["border"],
        tickfont=dict(family=FONT_FAMILY, size=12, color=COLORS["text_muted"]),
        title=dict(font=dict(family=FONT_FAMILY, size=13, color=COLORS["text_soft"])),
        showline=True,
        ticks="outside",
        ticklen=4,
        tickcolor=COLORS["border"],
    )

    template.layout.update(
        paper_bgcolor=COLORS["panel"],
        plot_bgcolor=COLORS["panel"],
        font=dict(family=FONT_FAMILY, size=13, color=COLORS["text"]),
        title=dict(
            font=dict(family=FONT_FAMILY, size=17, color=COLORS["text"], weight="bold"),
            x=0.01,
            xanchor="left",
            y=0.99,
            yanchor="top",
        ),
        # La franja superior hospeda titulo y leyenda en filas distintas: con
        # menos margen, una leyenda ancha se monta encima del titulo.
        margin=dict(l=56, r=24, t=78, b=48),
        xaxis=axis,
        yaxis=axis,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(family=FONT_FAMILY, size=12, color=COLORS["text_soft"]),
            orientation="h",
            yanchor="bottom",
            y=1.005,
            xanchor="right",
            x=1,
            itemsizing="constant",
        ),
        hoverlabel=dict(
            bgcolor=COLORS["panel_alt"],
            bordercolor=COLORS["border"],
            font=dict(family=FONT_FAMILY, size=13, color=COLORS["text"]),
        ),
        colorway=[
            "#58A6FF", "#F24E1E", "#45C4B0", "#F2C14E", "#BC8CFF",
            "#3FB950", "#FF7B72", "#79C0FF", "#D29922", "#8B97A8",
        ],
    )
    return template


def register_template() -> str:
    """Registra el template y lo deja como predeterminado de Plotly."""
    if TEMPLATE_NAME not in pio.templates:
        pio.templates[TEMPLATE_NAME] = _build_template()
    pio.templates.default = TEMPLATE_NAME
    return TEMPLATE_NAME


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Semi+Condensed:wght@400;500;600;700&family=Barlow+Condensed:wght@500;600;700&display=swap');

html, body, [class*="st-"], .stMarkdown, button, input, select, textarea {{
    font-family: {FONT_FAMILY};
}}
.stApp {{
    background: {COLORS['background']};
    color: {COLORS['text']};
}}
/* Densidad alta: menos aire vertical del que trae Streamlit por defecto. */
.block-container {{
    padding-top: 2.2rem;
    padding-bottom: 2rem;
    max-width: 1800px;
}}
h1, h2, h3, h4 {{
    font-family: 'Barlow Condensed', {FONT_FAMILY};
    letter-spacing: 0.01em;
    color: {COLORS['text']};
}}
h1 {{ font-size: 1.85rem; font-weight: 700; margin-bottom: 0.1rem; }}
h2 {{ font-size: 1.25rem; font-weight: 600; margin: 0.4rem 0 0.2rem 0; }}
h3 {{ font-size: 1.05rem; font-weight: 600; }}

/* Sidebar fija con todos los filtros. */
section[data-testid="stSidebar"] {{
    background: {COLORS['panel']};
    border-right: 1px solid {COLORS['border']};
    width: 320px !important;
}}
section[data-testid="stSidebar"] .block-container {{ padding-top: 1.2rem; }}

/* Tarjetas KPI y paneles de graficos con bordes suaves. */
div[data-testid="stMetric"] {{
    background: linear-gradient(180deg, {COLORS['panel_alt']} 0%, {COLORS['panel']} 100%);
    border: 1px solid {COLORS['border']};
    border-radius: 10px;
    padding: 0.7rem 0.9rem;
}}
div[data-testid="stMetricLabel"] p {{
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: {COLORS['text_soft']} !important;
    font-weight: 600;
}}
div[data-testid="stMetricValue"] {{
    font-family: 'Barlow Condensed', {FONT_FAMILY};
    font-size: 1.6rem !important;
    font-weight: 600;
    color: {COLORS['text']};
}}
/* El delta se usa como subtitulo descriptivo, no como variacion: se fuerza un
   gris legible y se oculta la flecha, que aqui no significa nada. */
div[data-testid="stMetricDelta"] {{
    font-size: 0.8rem !important;
    color: {COLORS['text_muted']} !important;
    opacity: 1 !important;
}}
div[data-testid="stMetricDelta"] > div {{ color: {COLORS['text_muted']} !important; }}
div[data-testid="stMetricDelta"] svg {{ display: none !important; }}

/* Etiquetas de los widgets: Streamlit las apaga demasiado sobre fondo oscuro. */
div[data-testid="stWidgetLabel"] p,
div[data-testid="stWidgetLabel"] label,
section[data-testid="stSidebar"] label p {{
    color: {COLORS['text_soft']} !important;
    font-size: 0.86rem !important;
}}
div[data-testid="stCaptionContainer"] p,
div[data-testid="stCaptionContainer"] {{
    color: {COLORS['text_muted']} !important;
    font-size: 0.8rem !important;
}}
/* Valores y topes de los sliders. */
div[data-testid="stSliderThumbValue"] {{
    color: {COLORS['accent']} !important;
    font-weight: 700;
    font-size: 0.86rem !important;
}}
div[data-testid="stSliderTickBarMin"],
div[data-testid="stSliderTickBarMax"] {{
    color: {COLORS['text_muted']} !important;
    font-size: 0.75rem !important;
}}

/* La barra superior de Streamlit no debe cortar el fondo oscuro. */
header[data-testid="stHeader"] {{
    background: transparent !important;
}}

/* Boton de aplicar filtros. */
section[data-testid="stSidebar"] button[kind="primaryFormSubmit"],
section[data-testid="stSidebar"] div[data-testid="stFormSubmitButton"] button {{
    background: {COLORS['accent']} !important;
    border: 1px solid {COLORS['accent']} !important;
    color: #ffffff !important;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-top: 0.4rem;
}}
section[data-testid="stSidebar"] div[data-testid="stFormSubmitButton"] button:hover {{
    filter: brightness(1.15);
}}
/* El formulario no debe agregar su propio marco dentro de la sidebar. */
section[data-testid="stSidebar"] div[data-testid="stForm"] {{
    border: none;
    padding: 0;
}}

/* Contenedores con borde = paneles de la grilla de graficos. */
div[data-testid="stVerticalBlockBorderWrapper"] > div > div[data-testid="stVerticalBlock"] {{
    gap: 0.6rem;
}}
div[data-testid="stElementContainer"] .js-plotly-plot {{
    border-radius: 10px;
}}
div[data-testid="stVerticalBlockBorderWrapper"][style*="border"] {{
    background: {COLORS['panel']};
}}

/* Tablas con la misma densidad que el resto. */
div[data-testid="stDataFrame"] {{
    border: 1px solid {COLORS['border']};
    border-radius: 10px;
}}

/* Pestanas y controles. */
button[data-testid="stBaseButton-segmented_controlActive"] {{
    background: {COLORS['accent']} !important;
    border-color: {COLORS['accent']} !important;
    color: #fff !important;
}}
.stTabs [data-baseweb="tab-list"] {{ gap: 0.2rem; border-bottom: 1px solid {COLORS['border']}; }}
.stTabs [data-baseweb="tab"] {{ font-size: 0.9rem; padding: 0.35rem 0.8rem; }}

/* Etiqueta de estimacion: todo lo que sale de un modelo la lleva. */
.f1-estimate {{
    display: inline-block;
    background: rgba(210, 153, 34, 0.14);
    border: 1px solid rgba(210, 153, 34, 0.45);
    color: {COLORS['warning']};
    border-radius: 5px;
    padding: 0.12rem 0.45rem;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-right: 0.4rem;
}}
.f1-assumption {{
    color: {COLORS['text_muted']};
    font-size: 0.82rem;
    line-height: 1.45;
}}
.f1-header {{
    display: flex;
    align-items: baseline;
    gap: 0.75rem;
    border-bottom: 2px solid {COLORS['accent']};
    padding-bottom: 0.4rem;
    margin-bottom: 0.8rem;
}}
.f1-header-sub {{
    color: {COLORS['text_soft']};
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}
.f1-caption {{
    color: {COLORS['text_muted']};
    font-size: 0.82rem;
    margin: -0.3rem 0 0.5rem 0;
}}
</style>
"""


def apply(page_title: str = "F1 Race Engineering") -> None:
    """Configura la pagina, inyecta el CSS y registra el template de Plotly."""
    st.set_page_config(
        page_title=page_title,
        page_icon="🏁",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    register_template()
    st.markdown(CSS, unsafe_allow_html=True)


def header(title: str, subtitle: str = "") -> None:
    st.markdown(
        f'<div class="f1-header"><h1>{title}</h1>'
        f'<span class="f1-header-sub">{subtitle}</span></div>',
        unsafe_allow_html=True,
    )


def caption(text: str) -> None:
    st.markdown(f'<div class="f1-caption">{text}</div>', unsafe_allow_html=True)


def estimate_note(text: str, assumptions: dict | None = None) -> None:
    """Marca visible de que un valor sale de un modelo, con sus supuestos a la vista."""
    detail = ""
    if assumptions:
        items = " &nbsp;·&nbsp; ".join(f"<b>{key}:</b> {value}" for key, value in assumptions.items())
        detail = f"<div class='f1-assumption'>{items}</div>"
    st.markdown(
        f"<div><span class='f1-estimate'>Estimacion</span>"
        f"<span class='f1-assumption'>{text}</span></div>{detail}",
        unsafe_allow_html=True,
    )


def plotly(figure: go.Figure, *, height: int | None = None, key: str | None = None) -> None:
    """Render estandar de Plotly: sin barra de herramientas y con el tema propio."""
    if height is not None:
        figure.update_layout(height=height)
    st.plotly_chart(figure, theme=None, config=PLOTLY_CONFIG, key=key)
