"""Cliente de clima (Open-Meteo) para las coordenadas de cada circuito.

Responsabilidad unica: traer el tiempo previsto u observado en un circuito y
una fecha. No decide como se usa: eso es del modelo de prediccion.

Open-Meteo es gratuito y no pide API key. Se usan dos endpoints:
  - `forecast`  : pronostico, horizonte de 16 dias.
  - `archive`   : reanalisis historico, desde 1940 y hasta hace ~5 dias.

Si una carrera cae mas alla del horizonte de pronostico NO se inventa un dato:
se devuelve la climatologia observada de años anteriores en esa misma fecha y
lugar, etiquetada como tal en `source`.
"""

from __future__ import annotations

import logging
import random
import time
from datetime import date, datetime, timedelta
from typing import Any, Callable

import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Horizonte maximo que publica Open-Meteo.
FORECAST_HORIZON_DAYS = 16

# Ventana horaria local en la que suele correrse un Gran Premio.
RACE_HOURS = (13, 18)

# Años de historico que se promedian cuando no hay pronostico disponible.
CLIMATOLOGY_YEARS = 5

HOURLY_VARIABLES = (
    "temperature_2m,relative_humidity_2m,precipitation,precipitation_probability,"
    "wind_speed_10m,cloud_cover"
)
ARCHIVE_VARIABLES = "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,cloud_cover"

DEFAULT_TTL = 60 * 60


class WeatherError(RuntimeError):
    """No se pudo obtener el clima. El mensaje es apto para mostrar en UI."""


class WeatherClient:
    """Cliente con reintentos y backoff, del mismo estilo que el de Jolpica."""

    def __init__(
        self,
        *,
        timeout: float = 30.0,
        max_attempts: int = 4,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.timeout = timeout
        self.max_attempts = max_attempts
        self._sleep = sleep
        self._session = session or requests.Session()
        self._session.headers.update({"User-Agent": "f1-race-engineering-dashboard/1.0"})

    def get(self, url: str, params: dict[str, Any]) -> dict:
        last_error = "sin detalle"
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self._session.get(url, params=params, timeout=self.timeout)
                if response.status_code == 200:
                    return response.json()
                last_error = f"codigo {response.status_code}"
                if response.status_code < 500 and response.status_code != 429:
                    raise WeatherError(
                        f"Open-Meteo rechazo la consulta ({response.status_code}). "
                        "Revisa las coordenadas del circuito."
                    )
            except requests.RequestException as exc:
                last_error = type(exc).__name__
            if attempt < self.max_attempts:
                self._sleep(random.uniform(0.5, 2.0 * attempt))
        raise WeatherError(
            f"No se pudo consultar Open-Meteo despues de {self.max_attempts} intentos "
            f"(ultimo fallo: {last_error})."
        )


def _summarise(frame: pd.DataFrame) -> dict[str, float]:
    """Resume las horas de carrera en las cifras que usa el modelo."""
    if frame.empty:
        return {}
    summary = {
        "air_temp": float(frame["temperature_2m"].mean()),
        "air_temp_max": float(frame["temperature_2m"].max()),
        "humidity": float(frame["relative_humidity_2m"].mean()),
        "wind_speed": float(frame["wind_speed_10m"].mean()),
        "cloud_cover": float(frame["cloud_cover"].mean()),
        "precipitation_mm": float(frame["precipitation"].sum()),
    }
    if "precipitation_probability" in frame.columns and frame["precipitation_probability"].notna().any():
        summary["rain_probability"] = float(frame["precipitation_probability"].max())
    else:
        # En el archivo historico no hay probabilidad: se deriva de si llovio.
        summary["rain_probability"] = 100.0 if summary["precipitation_mm"] >= 0.2 else 0.0
    return summary


def _race_window(payload: dict, target: date) -> pd.DataFrame:
    hourly = payload.get("hourly", {})
    if not hourly.get("time"):
        return pd.DataFrame()
    frame = pd.DataFrame(hourly)
    frame["time"] = pd.to_datetime(frame["time"])
    same_day = frame.loc[frame["time"].dt.date == target]
    if same_day.empty:
        return pd.DataFrame()
    start, end = RACE_HOURS
    window = same_day.loc[same_day["time"].dt.hour.between(start, end)]
    return window if not window.empty else same_day


class WeatherRepository:
    """Consultas de alto nivel: pronostico de carrera o climatologia."""

    def __init__(self, client: WeatherClient | None = None):
        self.client = client or WeatherClient()

    def forecast(self, latitude: float, longitude: float, target: date) -> dict:
        payload = self.client.get(
            FORECAST_URL,
            {
                "latitude": latitude,
                "longitude": longitude,
                "hourly": HOURLY_VARIABLES,
                "timezone": "auto",
                "forecast_days": FORECAST_HORIZON_DAYS,
            },
        )
        window = _race_window(payload, target)
        if window.empty:
            raise WeatherError(
                f"Open-Meteo no devolvio horas para {target.isoformat()} en esas coordenadas."
            )
        return _summarise(window)

    def observed(self, latitude: float, longitude: float, target: date) -> dict:
        payload = self.client.get(
            ARCHIVE_URL,
            {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": target.isoformat(),
                "end_date": target.isoformat(),
                "hourly": ARCHIVE_VARIABLES,
                "timezone": "auto",
            },
        )
        window = _race_window(payload, target)
        if window.empty:
            raise WeatherError(f"Open-Meteo no tiene archivo para {target.isoformat()}.")
        return _summarise(window)

    def climatology(
        self, latitude: float, longitude: float, target: date, *, years: int = CLIMATOLOGY_YEARS
    ) -> dict:
        """Promedio de lo observado en esa misma fecha en años anteriores.

        Es un dato real medido, no una invencion: se usa cuando la carrera cae
        fuera del horizonte de pronostico.
        """
        samples: list[dict] = []
        for offset in range(1, years + 1):
            try:
                day = target.replace(year=target.year - offset)
            except ValueError:  # 29 de febrero
                day = target.replace(year=target.year - offset, day=28)
            try:
                samples.append(self.observed(latitude, longitude, day))
            except WeatherError:
                continue
        if not samples:
            raise WeatherError(
                "Open-Meteo no devolvio historico para esa fecha y ubicacion, y la carrera "
                f"esta fuera del horizonte de pronostico ({FORECAST_HORIZON_DAYS} dias)."
            )
        frame = pd.DataFrame(samples)
        summary = {column: float(frame[column].mean()) for column in frame.columns}
        summary["rain_probability"] = float((frame["precipitation_mm"] >= 0.2).mean() * 100)
        summary["samples"] = len(samples)
        return summary

    def race_weather(self, latitude: float, longitude: float, target: date) -> dict:
        """Clima para la fecha de carrera, con la fuente declarada.

        `source` vale `pronostico`, `observado` o `climatologia`. La UI y el
        modelo deben mostrarlo: no es lo mismo un pronostico a 3 dias que el
        promedio de 5 años.
        """
        today = date.today()
        if target < today:
            summary = self.observed(latitude, longitude, target)
            summary["source"] = "observado"
            return summary

        if (target - today).days <= FORECAST_HORIZON_DAYS:
            summary = self.forecast(latitude, longitude, target)
            summary["source"] = "pronostico"
            summary["lead_days"] = (target - today).days
            return summary

        summary = self.climatology(latitude, longitude, target)
        summary["source"] = "climatologia"
        summary["lead_days"] = (target - today).days
        return summary


# ------------------------------------------------- capa cacheada para Streamlit

import streamlit as st  # noqa: E402  (import tardio: el cliente puro no depende de Streamlit)


@st.cache_resource(show_spinner=False)
def get_repository() -> WeatherRepository:
    return WeatherRepository()


@st.cache_data(ttl=DEFAULT_TTL, show_spinner=False)
def load_race_weather(latitude: float, longitude: float, race_date: str) -> dict:
    """Clima de carrera cacheado una hora. `race_date` en formato ISO."""
    target = datetime.fromisoformat(race_date).date()
    return get_repository().race_weather(latitude, longitude, target)


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def load_observed_weather(latitude: float, longitude: float, race_date: str) -> dict:
    """Clima observado (para backtesting). Se cachea un dia: el pasado no cambia."""
    target = datetime.fromisoformat(race_date).date()
    return get_repository().observed(latitude, longitude, target)
