"""Cliente HTTP para la API Jolpica-F1 (sucesor de Ergast).

Responsabilidad unica: hablar con `https://api.jolpi.ca/ergast/f1/` respetando
el limite de tasa publicado (4 req/s, 500 req/h) y devolver DataFrames limpios.

No contiene logica de analisis ni de presentacion. Las funciones del final
(prefijo `load_`) son la capa cacheada que consume la UI.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from collections import deque
from typing import Any, Callable, Iterable

import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)

BASE_URL = "https://api.jolpi.ca/ergast/f1/"

# Limites publicados por Jolpica para usuarios anonimos.
BURST_LIMIT_PER_SECOND = 4
SUSTAINED_LIMIT_PER_HOUR = 500

# Jolpica trunca `limit` a 100 por pagina.
MAX_PAGE_SIZE = 100

# TTL por defecto (segundos) para las funciones `load_*`.
DEFAULT_TTL = 60 * 60


class JolpicaError(RuntimeError):
    """Fallo al obtener datos de Jolpica. El mensaje es apto para mostrar en UI."""

    def __init__(self, message: str, *, status: int | None = None, path: str | None = None):
        super().__init__(message)
        self.message = message
        self.status = status
        self.path = path


class JolpicaRateLimited(JolpicaError):
    """Se agotaron los reintentos frente a respuestas 429."""


class _RateLimiter:
    """Token bucket doble: rafaga por segundo y presupuesto por hora.

    `acquire()` bloquea lo necesario para no exceder ninguno de los dos limites.
    Es thread-safe porque Streamlit ejecuta cada sesion en su propio hilo.
    """

    def __init__(
        self,
        per_second: int = BURST_LIMIT_PER_SECOND,
        per_hour: int = SUSTAINED_LIMIT_PER_HOUR,
    ):
        self._per_second = per_second
        self._per_hour = per_hour
        self._lock = threading.Lock()
        self._recent: deque[float] = deque()
        self._hourly: deque[float] = deque()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._evict(now)
                wait = self._wait_needed(now)
                if wait <= 0:
                    self._recent.append(now)
                    self._hourly.append(now)
                    return
            time.sleep(min(wait, 5.0))

    def _evict(self, now: float) -> None:
        while self._recent and now - self._recent[0] >= 1.0:
            self._recent.popleft()
        while self._hourly and now - self._hourly[0] >= 3600.0:
            self._hourly.popleft()

    def _wait_needed(self, now: float) -> float:
        wait = 0.0
        if len(self._recent) >= self._per_second:
            wait = max(wait, 1.0 - (now - self._recent[0]))
        if len(self._hourly) >= self._per_hour:
            wait = max(wait, 3600.0 - (now - self._hourly[0]))
        return wait

    @property
    def hourly_used(self) -> int:
        with self._lock:
            self._evict(time.monotonic())
            return len(self._hourly)


class JolpicaClient:
    """Cliente con reintentos y backoff exponencial.

    Reintenta ante 429, 5xx y errores de red/timeout. Respeta `Retry-After`
    cuando el servidor lo envia. Nunca inventa datos: si agota los reintentos
    levanta `JolpicaError` para que la UI lo muestre explicitamente.
    """

    def __init__(
        self,
        base_url: str = BASE_URL,
        *,
        timeout: float = 45.0,
        max_attempts: int = 5,
        backoff_base: float = 1.5,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.backoff_base = backoff_base
        self._sleep = sleep
        self._limiter = _RateLimiter()
        self._session = session or requests.Session()
        self._session.headers.update({"User-Agent": "f1-race-engineering-dashboard/1.0"})

    # ------------------------------------------------------------------ HTTP

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict:
        """GET con rate limiting y backoff. Devuelve el cuerpo JSON completo."""
        url = self.base_url + path.lstrip("/")
        last_error = "sin detalle"
        last_status: int | None = None

        for attempt in range(1, self.max_attempts + 1):
            self._limiter.acquire()
            try:
                response = self._session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = type(exc).__name__
                LOGGER.warning("Jolpica %s: error de red (%s), intento %d", path, last_error, attempt)
            else:
                last_status = response.status_code
                if response.status_code == 200:
                    try:
                        return response.json()
                    except ValueError as exc:
                        raise JolpicaError(
                            f"Jolpica devolvio una respuesta no-JSON para `{path}`.",
                            status=response.status_code,
                            path=path,
                        ) from exc

                if response.status_code == 429:
                    delay = self._retry_after(response, attempt)
                    LOGGER.warning("Jolpica %s: 429, esperando %.1fs (intento %d)", path, delay, attempt)
                    last_error = "429 Too Many Requests"
                    if attempt < self.max_attempts:
                        self._sleep(delay)
                    continue

                if response.status_code >= 500:
                    last_error = f"{response.status_code} del servidor"
                    LOGGER.warning("Jolpica %s: %s, intento %d", path, last_error, attempt)
                else:
                    # 4xx distinto de 429: reintentar no cambia el resultado.
                    raise JolpicaError(
                        f"Jolpica rechazo la consulta `{path}` con codigo {response.status_code}. "
                        "Probablemente esa combinacion de temporada/ronda no existe en la API.",
                        status=response.status_code,
                        path=path,
                    )

            if attempt < self.max_attempts:
                self._sleep(self._backoff_delay(attempt))

        if last_status == 429:
            raise JolpicaRateLimited(
                f"Jolpica sigue respondiendo 429 (limite de tasa) para `{path}` despues de "
                f"{self.max_attempts} intentos con backoff. Espera un momento y reintenta.",
                status=429,
                path=path,
            )
        raise JolpicaError(
            f"No se pudo obtener `{path}` de Jolpica despues de {self.max_attempts} intentos "
            f"(ultimo fallo: {last_error}).",
            status=last_status,
            path=path,
        )

    def _backoff_delay(self, attempt: int) -> float:
        """Backoff exponencial con jitter completo, tope de 30 s."""
        ceiling = min(self.backoff_base**attempt, 30.0)
        return random.uniform(ceiling / 2, ceiling)

    def _retry_after(self, response: requests.Response, attempt: int) -> float:
        """Espera tras un 429: nunca menor que el backoff propio.

        Jolpica a veces responde `Retry-After: 0` cuando el limite ya se agoto.
        Obedecerlo al pie de la letra quemaba los cinco reintentos en el mismo
        instante, sin esperar nada. Se toma el mayor entre lo que pide el
        servidor y el backoff exponencial.
        """
        backoff = self._backoff_delay(attempt)
        header = response.headers.get("Retry-After")
        if header:
            try:
                return min(max(float(header), backoff), 60.0)
            except ValueError:
                pass
        return backoff

    def get_paginated(self, path: str, *, max_records: int = 3000) -> list[dict]:
        """Recorre todas las paginas de un endpoint y devuelve los `MRData` crudos."""
        pages: list[dict] = []
        offset = 0
        while True:
            body = self.get(path, params={"limit": MAX_PAGE_SIZE, "offset": offset})
            mrdata = body.get("MRData", {})
            pages.append(mrdata)
            total = int(mrdata.get("total", 0))
            offset += MAX_PAGE_SIZE
            if offset >= min(total, max_records):
                break
        return pages

    @property
    def hourly_requests_used(self) -> int:
        return self._limiter.hourly_used


# --------------------------------------------------------------------- parseo


def _races_from(pages: Iterable[dict]) -> list[dict]:
    races: list[dict] = []
    for mrdata in pages:
        races.extend(mrdata.get("RaceTable", {}).get("Races", []))
    return races


def _race_meta(race: dict) -> dict:
    circuit = race.get("Circuit", {})
    location = circuit.get("Location", {})
    return {
        "season": int(race["season"]),
        "round": int(race["round"]),
        "raceName": race.get("raceName"),
        "date": race.get("date"),
        "circuitId": circuit.get("circuitId"),
        "circuitName": circuit.get("circuitName"),
        "locality": location.get("locality"),
        "country": location.get("country"),
    }


def _driver_fields(driver: dict) -> dict:
    return {
        "driverId": driver.get("driverId"),
        "code": driver.get("code"),
        "driverName": f"{driver.get('givenName', '')} {driver.get('familyName', '')}".strip(),
        "driverNumber": driver.get("permanentNumber"),
        "nationality": driver.get("nationality"),
    }


def _constructor_fields(constructor: dict) -> dict:
    return {
        "constructorId": constructor.get("constructorId"),
        "constructorName": constructor.get("name"),
    }


def _result_rows(races: list[dict], key: str) -> pd.DataFrame:
    rows: list[dict] = []
    for race in races:
        meta = _race_meta(race)
        for result in race.get(key, []):
            row = dict(meta)
            row.update(_driver_fields(result.get("Driver", {})))
            row.update(_constructor_fields(result.get("Constructor", {})))
            row["position"] = pd.to_numeric(result.get("position"), errors="coerce")
            row["positionText"] = result.get("positionText")
            row["points"] = pd.to_numeric(result.get("points"), errors="coerce")
            row["grid"] = pd.to_numeric(result.get("grid"), errors="coerce")
            row["laps"] = pd.to_numeric(result.get("laps"), errors="coerce")
            row["status"] = result.get("status")
            row["timeText"] = (result.get("Time") or {}).get("time")
            row["timeMillis"] = pd.to_numeric((result.get("Time") or {}).get("millis"), errors="coerce")
            fastest = result.get("FastestLap") or {}
            row["fastestLapRank"] = pd.to_numeric(fastest.get("rank"), errors="coerce")
            row["fastestLapTime"] = (fastest.get("Time") or {}).get("time")
            rows.append(row)
    return pd.DataFrame(rows)


def _qualifying_rows(races: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for race in races:
        meta = _race_meta(race)
        for result in race.get("QualifyingResults", []):
            row = dict(meta)
            row.update(_driver_fields(result.get("Driver", {})))
            row.update(_constructor_fields(result.get("Constructor", {})))
            row["position"] = pd.to_numeric(result.get("position"), errors="coerce")
            for segment in ("Q1", "Q2", "Q3"):
                row[segment] = result.get(segment)
                row[f"{segment}Seconds"] = _duration_to_seconds(result.get(segment))
            rows.append(row)
    return pd.DataFrame(rows)


def _standings_rows(pages: Iterable[dict], key: str) -> pd.DataFrame:
    rows: list[dict] = []
    for mrdata in pages:
        for standing_list in mrdata.get("StandingsTable", {}).get("StandingsLists", []):
            season = int(standing_list["season"])
            rnd = int(standing_list["round"])
            for entry in standing_list.get(key, []):
                row = {
                    "season": season,
                    "round": rnd,
                    "position": pd.to_numeric(entry.get("position"), errors="coerce"),
                    "points": pd.to_numeric(entry.get("points"), errors="coerce"),
                    "wins": pd.to_numeric(entry.get("wins"), errors="coerce"),
                }
                if "Driver" in entry:
                    row.update(_driver_fields(entry["Driver"]))
                    constructors = entry.get("Constructors", [])
                    if constructors:
                        row.update(_constructor_fields(constructors[0]))
                if "Constructor" in entry:
                    row.update(_constructor_fields(entry["Constructor"]))
                rows.append(row)
    return pd.DataFrame(rows)


def _duration_to_seconds(text: str | None) -> float:
    """Convierte '1:23.456' o '23.456' a segundos. Devuelve NaN si no parsea."""
    if not text:
        return float("nan")
    try:
        parts = str(text).split(":")
        seconds = float(parts[-1])
        if len(parts) > 1:
            seconds += float(parts[-2]) * 60
        if len(parts) > 2:
            seconds += float(parts[-3]) * 3600
        return seconds
    except (ValueError, TypeError):
        return float("nan")


# ------------------------------------------------------------------ consultas


class JolpicaRepository:
    """Consultas de alto nivel. Cada metodo devuelve un DataFrame tidy."""

    def __init__(self, client: JolpicaClient | None = None):
        self.client = client or JolpicaClient()

    def seasons(self) -> pd.DataFrame:
        pages = self.client.get_paginated("seasons/")
        rows = []
        for mrdata in pages:
            for season in mrdata.get("SeasonTable", {}).get("Seasons", []):
                rows.append({"season": int(season["season"]), "url": season.get("url")})
        return pd.DataFrame(rows).sort_values("season", ascending=False).reset_index(drop=True)

    def schedule(self, season: int) -> pd.DataFrame:
        races = _races_from(self.client.get_paginated(f"{season}/races/"))
        return pd.DataFrame([_race_meta(race) for race in races])

    def race_results(self, season: int, round_no: int | None = None) -> pd.DataFrame:
        path = f"{season}/results/" if round_no is None else f"{season}/{round_no}/results/"
        return _result_rows(_races_from(self.client.get_paginated(path)), "Results")

    def sprint_results(self, season: int) -> pd.DataFrame:
        return _result_rows(_races_from(self.client.get_paginated(f"{season}/sprint/")), "SprintResults")

    def qualifying_results(self, season: int, round_no: int | None = None) -> pd.DataFrame:
        path = f"{season}/qualifying/" if round_no is None else f"{season}/{round_no}/qualifying/"
        return _qualifying_rows(_races_from(self.client.get_paginated(path)))

    def pit_stops(self, season: int, round_no: int) -> pd.DataFrame:
        pages = self.client.get_paginated(f"{season}/{round_no}/pitstops/")
        rows: list[dict] = []
        for race in _races_from(pages):
            meta = _race_meta(race)
            for stop in race.get("PitStops", []):
                row = dict(meta)
                row["driverId"] = stop.get("driverId")
                row["lap"] = pd.to_numeric(stop.get("lap"), errors="coerce")
                row["stop"] = pd.to_numeric(stop.get("stop"), errors="coerce")
                row["timeOfDay"] = stop.get("time")
                row["durationText"] = stop.get("duration")
                row["durationSeconds"] = _duration_to_seconds(stop.get("duration"))
                rows.append(row)
        return pd.DataFrame(rows)

    def driver_standings(self, season: int, round_no: int | None = None) -> pd.DataFrame:
        path = f"{season}/driverstandings/" if round_no is None else f"{season}/{round_no}/driverstandings/"
        return _standings_rows(self.client.get_paginated(path), "DriverStandings")

    def constructor_standings(self, season: int, round_no: int | None = None) -> pd.DataFrame:
        path = (
            f"{season}/constructorstandings/"
            if round_no is None
            else f"{season}/{round_no}/constructorstandings/"
        )
        return _standings_rows(self.client.get_paginated(path), "ConstructorStandings")

    def drivers(self, season: int) -> pd.DataFrame:
        pages = self.client.get_paginated(f"{season}/drivers/")
        rows = []
        for mrdata in pages:
            for driver in mrdata.get("DriverTable", {}).get("Drivers", []):
                rows.append(_driver_fields(driver))
        return pd.DataFrame(rows)

    def circuits(self, season: int | None = None) -> pd.DataFrame:
        path = "circuits/" if season is None else f"{season}/circuits/"
        rows = []
        for mrdata in self.client.get_paginated(path):
            for circuit in mrdata.get("CircuitTable", {}).get("Circuits", []):
                location = circuit.get("Location", {})
                rows.append(
                    {
                        "circuitId": circuit.get("circuitId"),
                        "circuitName": circuit.get("circuitName"),
                        "locality": location.get("locality"),
                        "country": location.get("country"),
                        # Coordenadas: las usa el modulo de clima para pedir el pronostico.
                        "lat": pd.to_numeric(location.get("lat"), errors="coerce"),
                        "lon": pd.to_numeric(location.get("long"), errors="coerce"),
                    }
                )
        return pd.DataFrame(rows)

    def driver_career(self, driver_id: str) -> pd.DataFrame:
        """Todos los resultados de carrera de un piloto (una fila por Gran Premio)."""
        pages = self.client.get_paginated(f"drivers/{driver_id}/results/")
        return _result_rows(_races_from(pages), "Results")

    def circuit_winners(self, circuit_id: str) -> pd.DataFrame:
        """Ganadores historicos de un circuito (una fila por edicion)."""
        pages = self.client.get_paginated(f"circuits/{circuit_id}/results/1/")
        return _result_rows(_races_from(pages), "Results")


# ------------------------------------------------- capa cacheada para Streamlit

import streamlit as st  # noqa: E402  (import tardio: el cliente puro no depende de Streamlit)


@st.cache_resource(show_spinner=False)
def get_repository() -> JolpicaRepository:
    """Un unico repositorio por proceso, para compartir el rate limiter."""
    return JolpicaRepository()


def _cached(fn: Callable[..., pd.DataFrame]) -> Callable[..., pd.DataFrame]:
    return st.cache_data(ttl=DEFAULT_TTL, show_spinner=False)(fn)


@_cached
def load_seasons() -> pd.DataFrame:
    return get_repository().seasons()


@_cached
def load_schedule(season: int) -> pd.DataFrame:
    return get_repository().schedule(season)


@_cached
def load_race_results(season: int, round_no: int | None = None) -> pd.DataFrame:
    return get_repository().race_results(season, round_no)


@_cached
def load_sprint_results(season: int) -> pd.DataFrame:
    return get_repository().sprint_results(season)


@_cached
def load_qualifying_results(season: int, round_no: int | None = None) -> pd.DataFrame:
    return get_repository().qualifying_results(season, round_no)


@_cached
def load_pit_stops(season: int, round_no: int) -> pd.DataFrame:
    return get_repository().pit_stops(season, round_no)


@_cached
def load_driver_standings(season: int, round_no: int | None = None) -> pd.DataFrame:
    return get_repository().driver_standings(season, round_no)


@_cached
def load_constructor_standings(season: int, round_no: int | None = None) -> pd.DataFrame:
    return get_repository().constructor_standings(season, round_no)


@_cached
def load_drivers(season: int) -> pd.DataFrame:
    return get_repository().drivers(season)


@_cached
def load_driver_career(driver_id: str) -> pd.DataFrame:
    return get_repository().driver_career(driver_id)


@_cached
def load_circuit_winners(circuit_id: str) -> pd.DataFrame:
    return get_repository().circuit_winners(circuit_id)


@_cached
def load_circuits(season: int | None = None) -> pd.DataFrame:
    return get_repository().circuits(season)
