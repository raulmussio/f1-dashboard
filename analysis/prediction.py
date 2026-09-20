"""Modelo probabilistico de resultado de carrera.

Modulo puro: numpy/pandas/scipy. No importa Streamlit ni dibuja nada.

QUE ES Y QUE NO ES
------------------
Esto estima PROBABILIDADES, no resultados. Devuelve "Verstappen 34 % de ganar",
nunca "gana Verstappen". La Formula 1 tiene mucha varianza irreducible
(abandonos, safety cars, incidentes de primera vuelta) y el modelo la representa
simulando la carrera miles de veces en vez de dar un unico orden.

COMO FUNCIONA
-------------
1. A cada piloto se le asigna una "fuerza" para esa carrera: una combinacion
   lineal de variables causales (solo informacion anterior a la carrera).
2. Esa fuerza alimenta un modelo Plackett-Luce, el modelo estandar para ordenes
   de llegada: la probabilidad de un orden es el producto de elegir al ganador
   entre todos, luego al segundo entre los que quedan, etc.
3. Los coeficientes se ajustan por maxima verosimilitud sobre carreras pasadas
   (`scipy.optimize`), no se eligen a mano.
4. La carrera se simula N veces: primero se sortea quien abandona (modelo
   logistico aparte), despues el orden de los supervivientes con ruido Gumbel,
   que es el muestreo exacto de Plackett-Luce.

DONDE ENTRA EL CLIMA
--------------------
En un modelo de ranking, una variable identica para todos los pilotos de la
carrera SE CANCELA: no puede cambiar el orden esperado. Por eso la temperatura y
la lluvia no entran como termino principal, sino por los dos canales donde si
tienen efecto medible, ambos calibrados con carreras historicas:
  - la lluvia aumenta la varianza del resultado (mas sorpresas);
  - la lluvia aumenta la probabilidad de abandono.

EL TIPO DE PISTA
----------------
Entra como interaccion entre la posicion de salida y el indice de adelantamiento
del circuito, medido de los datos: en Monaco la parrilla manda casi por completo,
en Baku o Interlagos pesa mucho menos.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize
from scipy import stats

# Vida media (en carreras) del promedio exponencial de forma reciente.
DEFAULT_HALF_LIFE = 3.0

# Vida media de la ventana larga dentro de la temporada. `None` seria una media
# acumulada que nunca olvida; con un valor finito, un cambio de nivel real (por
# ejemplo un paquete de mejoras a mitad de año) termina imponiendose.
# Elegida por barrido sobre el backtest: globalmente las ventanas casi no cambian
# nada (Brier entre 0.0282 y 0.0285), pero acortar esta ventana mejora las
# temporadas con un salto de nivel a mitad de año, como 2026 con las mejoras de
# McLaren (64.3% -> 71.4% de acierto del ganador).
DEFAULT_SEASON_HALF_LIFE = 4.0

# Cuantas carreras atras se mira para estimar la tendencia de un equipo.
DEFAULT_TREND_WINDOW = 4

# Vida media del duelo contra el compañero de equipo. Es larga a proposito: mide
# nivel de piloto, que cambia despacio, y necesita muchas carreras para no ser
# ruido. Es la unica variable capaz de separar a dos pilotos del mismo coche,
# porque compara exactamente eso: el mismo coche.
DEFAULT_TEAMMATE_HALF_LIFE = 10.0

# Minimo de carreras previas para que un piloto tenga forma estimable.
MIN_PRIOR_RACES = 2

# Regularizacion L2 de los coeficientes: evita que una variable domine con pocos datos.
DEFAULT_L2 = 1.0

# Variables del modelo de orden de llegada.
#
# Las de "forma" clasicas (posicion media reciente) resultaron redundantes con la
# parrilla: correlacionan 0.70-0.92 con ella, porque la parrilla ya resume el
# nivel del coche. Las que aportan son las que la parrilla NO contiene:
#   - `quali_margin`: el margen en segundos, no el orden. No es lo mismo una pole
#     por 0.01 s que por medio segundo.
#   - `form_race_gain`: cuantas posiciones suele ganar el piloto el domingo
#     respecto de donde sale (salidas, ritmo de carrera, estrategia).
#   - `team_race_gain`: lo mismo a nivel equipo.
RACE_FEATURES = [
    "grid_score",
    "grid_x_overtaking",
    "quali_margin",
    "form_race_gain",
    "team_race_gain",
    "form_finish",
    "season_form",
    "season_points_rate",
    "teammate_race_edge",
    "teammate_quali_edge",
    "team_form",
    "team_trend",
    "track_affinity",
]

# Variables del modelo de clasificacion (sin parrilla, que es justamente lo que predice).
QUALI_FEATURES = [
    "form_quali",
    "form_quali_margin",
    "form_finish",
    "season_form",
    "season_points_rate",
    "teammate_quali_edge",
    "teammate_race_edge",
    "team_form",
    "team_trend",
    "track_affinity",
]

# Variables del modelo de abandono.
DNF_FEATURES = ["driver_dnf_rate", "team_dnf_rate", "circuit_dnf_rate"]


def _lap_time_seconds(text) -> float:
    """Convierte un tiempo de vuelta 'm:ss.mmm' a segundos."""
    if not isinstance(text, str) or not text:
        return float("nan")
    try:
        parts = text.split(":")
        seconds = float(parts[-1])
        if len(parts) > 1:
            seconds += float(parts[-2]) * 60
        return seconds
    except (ValueError, TypeError):
        return float("nan")


def _log_position_score(position: pd.Series) -> pd.Series:
    """Puntaje decreciente y no lineal: ganar vale mucho mas que ser segundo."""
    return -np.log(pd.to_numeric(position, errors="coerce").clip(lower=1))


def is_classified(status: pd.Series) -> pd.Series:
    """True si el piloto termino la carrera (incluye doblados)."""
    text = status.fillna("").astype(str)
    return text.str.startswith("Finished") | text.str.contains("Lap", case=False)


def race_index(frame: pd.DataFrame) -> pd.Series:
    """Indice temporal unico y ordenable: temporada y ronda en un solo entero."""
    return frame["season"].astype(int) * 100 + frame["round"].astype(int)


# ------------------------------------------------------------------ historia


def prepare_history(results: pd.DataFrame, qualifying: pd.DataFrame | None = None) -> pd.DataFrame:
    """Une resultados y clasificacion en una tabla por carrera y piloto.

    Agrega el hueco al ganador en segundos, la marca de abandono y el indice
    temporal. No calcula nada predictivo todavia.
    """
    if results.empty:
        return pd.DataFrame()

    history = results.copy()
    history["race_index"] = race_index(history)
    history["classified"] = is_classified(history["status"])
    history["dnf"] = ~history["classified"]
    history["entrants"] = history.groupby("race_index")["driverId"].transform("size")

    # La parrilla 0 en Ergast significa salida desde boxes: cuenta como ultimo.
    grid = pd.to_numeric(history["grid"], errors="coerce")
    history["grid"] = grid.where(grid > 0, history["entrants"]).fillna(history["entrants"])

    # `timeMillis` es el tiempo total acumulado: el hueco es la diferencia con el ganador.
    leader = history.groupby("race_index")["timeMillis"].transform("min")
    history["gap_seconds"] = (history["timeMillis"] - leader) / 1000.0
    # log1p comprime los huecos enormes de las carreras con safety car.
    history["gap_score"] = -np.log1p(history["gap_seconds"].clip(lower=0))

    history["finish_score"] = _log_position_score(history["position"])
    history["grid_score"] = _log_position_score(history["grid"])

    if qualifying is not None and not qualifying.empty:
        quali = qualifying.copy()
        quali["race_index"] = race_index(quali)
        segments = [c for c in ("Q1Seconds", "Q2Seconds", "Q3Seconds") if c in quali.columns]
        if segments:
            # Mejor vuelta lograda en cualquier segmento: mide ritmo, no orden.
            quali["quali_best"] = quali[segments].min(axis=1, skipna=True)
            pole_time = quali.groupby("race_index")["quali_best"].transform("min")
            # Margen porcentual respecto de la pole: comparable entre circuitos
            # de 70 s (Red Bull Ring) y de 105 s (Spa).
            quali["quali_margin_pct"] = (quali["quali_best"] / pole_time - 1.0) * 100
        else:
            quali["quali_margin_pct"] = np.nan
        quali = quali[["race_index", "driverId", "position", "quali_margin_pct"]].rename(
            columns={"position": "quali_position"}
        )
        history = history.merge(quali, on=["race_index", "driverId"], how="left")
    else:
        history["quali_position"] = np.nan
        history["quali_margin_pct"] = np.nan
    history["quali_score"] = _log_position_score(history["quali_position"])
    # Se invierte el signo para que, como en el resto, mas alto sea mejor. Los
    # huecos enormes de un Q1 abortado se recortan para que no dominen el ajuste.
    history["quali_margin"] = -history["quali_margin_pct"].clip(lower=0, upper=5).fillna(
        history["quali_margin_pct"].median()
    )

    # Posiciones ganadas el domingo respecto de la parrilla: lo que la parrilla
    # no puede saber de si misma.
    history["race_gain"] = history["grid"] - history["position"]

    return history.sort_values(["race_index", "position"]).reset_index(drop=True)


def _causal_ewm(frame: pd.DataFrame, key: str, column: str, half_life: float) -> pd.Series:
    """Media exponencial de `column` por `key`, usando SOLO carreras anteriores.

    El `shift(1)` es lo que hace honesto el backtest: en la carrera r solo se ve
    hasta la r-1.
    """
    ordered = frame.sort_values("race_index")
    shifted = ordered.groupby(key, observed=True)[column].shift(1)
    return (
        shifted.groupby(ordered[key], observed=True)
        .transform(lambda values: values.ewm(halflife=half_life, ignore_na=True).mean())
        .reindex(frame.index)
    )


def add_causal_features(
    history: pd.DataFrame,
    *,
    half_life: float = DEFAULT_HALF_LIFE,
    trend_window: int = DEFAULT_TREND_WINDOW,
    season_half_life: float | None = DEFAULT_SEASON_HALF_LIFE,
    teammate_half_life: float = DEFAULT_TEAMMATE_HALF_LIFE,
) -> pd.DataFrame:
    """Agrega las variables predictoras, todas calculadas con datos anteriores.

    Ninguna variable mira el resultado de la carrera que se quiere predecir.
    """
    if history.empty:
        return history

    frame = history.sort_values(["race_index", "driverId"]).copy()

    # --- forma del piloto
    frame["form_finish"] = _causal_ewm(frame, "driverId", "finish_score", half_life)
    frame["form_gap"] = _causal_ewm(frame, "driverId", "gap_score", half_life)
    frame["form_quali"] = _causal_ewm(frame, "driverId", "quali_score", half_life)
    frame["form_quali_margin"] = _causal_ewm(frame, "driverId", "quali_margin", half_life)
    # Ganancia tipica del domingo: salidas, ritmo de carrera y acierto de estrategia.
    frame["form_race_gain"] = _causal_ewm(frame, "driverId", "race_gain", half_life)
    frame["team_race_gain"] = _causal_ewm(frame, "constructorId", "race_gain", half_life)

    # Fiabilidad reciente del piloto y del equipo.
    frame["driver_dnf_rate"] = _causal_ewm(frame, "driverId", "dnf", half_life * 2)
    frame["team_dnf_rate"] = _causal_ewm(frame, "constructorId", "dnf", half_life * 2)

    # --- forma del equipo: promedio de sus coches en cada carrera
    team_race = (
        frame.groupby(["constructorId", "race_index"], observed=True)["finish_score"]
        .mean()
        .reset_index()
        .sort_values("race_index")
    )
    team_race["team_form"] = (
        team_race.groupby("constructorId", observed=True)["finish_score"]
        .shift(1)
        .groupby(team_race["constructorId"], observed=True)
        .transform(lambda values: values.ewm(halflife=half_life, ignore_na=True).mean())
    )
    # Tendencia: cuanto mejoro el equipo respecto de su nivel de hace N carreras.
    # Es el mejor proxy disponible de las actualizaciones tecnicas: no existe
    # ninguna API publica de mejoras, pero si el paquete funciona, aparece aqui.
    team_race["team_trend"] = team_race["team_form"] - team_race.groupby(
        "constructorId", observed=True
    )["team_form"].shift(trend_window)
    frame = frame.merge(
        team_race[["constructorId", "race_index", "team_form", "team_trend"]],
        on=["constructorId", "race_index"],
        how="left",
    )

    # --- rendimiento de ventana larga dentro de la temporada
    # La media exponencial de arriba tiene vida media corta: reacciona rapido
    # pero olvida la evidencia larga, y sin ella el modelo no distingue a dos
    # compañeros que alternan buenos fines de semana.
    #
    # `season_half_life=None` usa una media acumulada, que NUNCA olvida: si un
    # equipo trae mejoras a mitad de año, arrastra para siempre las carreras
    # malas de antes. Con una vida media larga la evidencia vieja se descuenta
    # sola y el modelo puede reconocer un cambio de nivel real.
    frame = frame.sort_values(["driverId", "season", "race_index"])
    grouped = frame.groupby(["driverId", "season"], observed=True)["finish_score"]
    if season_half_life is None:
        season_form = grouped.apply(lambda values: values.shift(1).expanding().mean())
    else:
        season_form = grouped.apply(
            lambda values: values.shift(1).ewm(halflife=season_half_life, ignore_na=True).mean()
        )
    frame["season_form"] = season_form.reset_index(level=[0, 1], drop=True)

    # --- puntos por carrera en la temporada en curso
    # `season_form` usa la posicion de llegada, donde ganar y salir segundo se
    # parecen bastante. El campeonato no: 25 contra 18. Esa diferencia es
    # justamente la que separa a un piloto que gana carreras de su compañero que
    # las acompaña, y explicaba por que la regla trivial "gana el lider del
    # campeonato" le ganaba al modelo en una temporada dominada.
    frame = frame.sort_values(["driverId", "season", "race_index"])
    frame["season_points_rate"] = (
        frame.groupby(["driverId", "season"], observed=True)["points"]
        .apply(lambda values: values.shift(1).expanding().mean())
        .reset_index(level=[0, 1], drop=True)
    )

    # --- duelo contra el compañero de equipo
    # Casi todas las variables de equipo (team_form, team_trend, team_race_gain)
    # valen LO MISMO para los dos pilotos del mismo coche, asi que no pueden
    # separarlos. Estas dos si: miden cuanto le saca cada piloto a su propio
    # compañero, que es la comparacion mas limpia que existe porque el coche es
    # el mismo. Se restan la media del equipo EN ESA CARRERA, con lo que el
    # coche se cancela y queda el piloto.
    for source, name in (("quali_score", "teammate_quali_edge"),
                         ("finish_score", "teammate_race_edge")):
        team_average = frame.groupby(["race_index", "constructorId"], observed=True)[
            source
        ].transform("mean")
        frame[f"_{name}_raw"] = frame[source] - team_average
        frame[name] = _causal_ewm(frame, "driverId", f"_{name}_raw", teammate_half_life)
        frame = frame.drop(columns=[f"_{name}_raw"])

    # --- afinidad con el circuito: rendimiento historico del piloto ahi
    # contra su propio nivel medio, siempre con ediciones anteriores.
    frame = frame.sort_values(["driverId", "circuitId", "race_index"])
    circuit_mean = (
        frame.groupby(["driverId", "circuitId"], observed=True)["finish_score"]
        .apply(lambda values: values.shift(1).expanding().mean())
        .reset_index(level=[0, 1], drop=True)
    )
    frame["circuit_mean"] = circuit_mean
    frame = frame.sort_values(["driverId", "race_index"])
    overall_mean = (
        frame.groupby("driverId", observed=True)["finish_score"]
        .apply(lambda values: values.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    )
    frame["overall_mean"] = overall_mean
    frame["track_affinity"] = (frame["circuit_mean"] - frame["overall_mean"]).fillna(0.0)
    # En la primera carrera del año todavia no hay temporada acumulada: se usa la
    # forma reciente, que arrastra el final del año anterior.
    frame["season_form"] = frame["season_form"].fillna(frame["form_finish"])
    # Un piloto sin historial junto a un compañero arranca en cero: sin ventaja
    # ni desventaja asumida.
    frame["season_points_rate"] = frame["season_points_rate"].fillna(0.0)
    frame["teammate_quali_edge"] = frame["teammate_quali_edge"].fillna(0.0)
    frame["teammate_race_edge"] = frame["teammate_race_edge"].fillna(0.0)

    # Nota de un intento fallido, para que no se repita: se probo una "señal de
    # mejora" = max(0, forma reciente del equipo - su linea base), pensada para
    # captar un paquete de actualizaciones a mitad de año. Medida de tres formas
    # -verosimilitud dejandola fuera en el modelo de carrera, lo mismo en el de
    # clasificacion, y ablacion sobre el backtest con varias semillas- aportaba
    # 0.0 en todas. Se quito: una variable que no mide nada no deberia figurar
    # como si modelara las actualizaciones tecnicas.

    return frame.sort_values(["race_index", "position"]).reset_index(drop=True)


def track_profiles(history: pd.DataFrame) -> pd.DataFrame:
    """Perfil medido de cada circuito, no opiniones sobre el trazado.

    - `overtaking_index`: cambio medio de posiciones entre parrilla y llegada.
      Alto = se adelanta (Interlagos, Baku); bajo = manda la parrilla (Monaco).
    - `dnf_rate`: proporcion historica de abandonos.
    - `grid_retention`: correlacion de Spearman entre parrilla y resultado.
    """
    columns = ["circuitId", "circuitName", "races", "overtaking_index", "dnf_rate", "grid_retention"]
    if history.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for circuit, group in history.groupby("circuitId", observed=True):
        classified = group.loc[group["classified"]]
        if classified.empty:
            continue
        movement = (classified["grid"] - classified["position"]).abs().mean()
        if len(classified) > 3 and classified["grid"].nunique() > 1:
            correlation = stats.spearmanr(classified["grid"], classified["position"]).statistic
        else:
            correlation = np.nan
        rows.append(
            {
                "circuitId": circuit,
                "circuitName": group["circuitName"].iloc[0] if "circuitName" in group else circuit,
                "races": int(group["race_index"].nunique()),
                "overtaking_index": float(movement),
                "dnf_rate": float(group["dnf"].mean()),
                "grid_retention": float(correlation) if correlation == correlation else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values("overtaking_index").reset_index(drop=True)


def attach_track_profile(frame: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """Agrega el indice de adelantamiento y su interaccion con la parrilla.

    La interaccion es la que hace que Monaco e Imola no se traten igual: donde
    no se adelanta, la parrilla pesa mas.
    """
    if frame.empty:
        return frame
    merged = frame.merge(
        profiles[["circuitId", "overtaking_index", "dnf_rate"]].rename(
            columns={"dnf_rate": "circuit_dnf_rate"}
        ),
        on="circuitId",
        how="left",
    )
    median_overtaking = profiles["overtaking_index"].median() if not profiles.empty else 0.0
    merged["overtaking_index"] = merged["overtaking_index"].fillna(median_overtaking)
    merged["circuit_dnf_rate"] = merged["circuit_dnf_rate"].fillna(
        profiles["dnf_rate"].median() if not profiles.empty else 0.15
    )
    # Centrada para que el termino principal de parrilla siga siendo interpretable.
    # La referencia es la mediana de TODO el calendario y se guarda como columna:
    # recalcularla dentro de una sola carrera daria cero siempre, porque todos
    # sus pilotos comparten circuito, y anularia la interaccion.
    merged["overtaking_reference"] = median_overtaking
    merged["grid_x_overtaking"] = merged["grid_score"] * (
        merged["overtaking_index"] - median_overtaking
    )
    return merged


# ------------------------------------------------- Plackett-Luce (orden de llegada)


def _reverse_logcumsumexp(scores: np.ndarray) -> np.ndarray:
    """logsumexp acumulado de derecha a izquierda, por filas."""
    rows, columns = scores.shape
    output = np.empty_like(scores)
    running = np.full(rows, -np.inf)
    for position in range(columns - 1, -1, -1):
        running = np.logaddexp(running, scores[:, position])
        output[:, position] = running
    return output


def _plackett_luce_nll(
    beta: np.ndarray, design: np.ndarray, mask: np.ndarray, l2: float
) -> float:
    """Log-verosimilitud negativa de los ordenes observados, con regularizacion."""
    scores = design @ beta
    scores = np.where(mask, scores, -np.inf)
    with np.errstate(invalid="ignore"):
        denominators = _reverse_logcumsumexp(scores)
        terms = np.where(mask, scores - denominators, 0.0)
    log_likelihood = float(np.nansum(terms))
    return -log_likelihood + l2 * float(beta @ beta)


@dataclass
class RankingModel:
    """Coeficientes ajustados de un modelo Plackett-Luce.

    `temperature` corrige un sesgo conocido del ajuste: la verosimilitud de
    Plackett-Luce optimiza el orden COMPLETO, y los muchos pares de media tabla
    -genuinamente impredecibles- comprimen la separacion entre los de cabeza. Sin
    corregir, el modelo reparte demasiada probabilidad y su favorito gana mucho
    mas seguido de lo que el mismo se asigna. La temperatura se estima de los
    datos (ver `fit_temperature`), no se elige a mano.
    """

    features: list[str]
    coefficients: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    races_used: int = 0
    converged: bool = True
    temperature: float = 1.0

    def strength(self, frame: pd.DataFrame) -> np.ndarray:
        """Fuerza latente de cada fila, ya calibrada (mayor = mejor)."""
        return self.raw_strength(frame) * self.temperature

    def raw_strength(self, frame: pd.DataFrame) -> np.ndarray:
        design = frame.reindex(columns=self.features).astype(float).fillna(0.0).to_numpy()
        standardised = (design - self.means) / self.scales
        return standardised @ self.coefficients

    @property
    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"Variable": self.features, "Coeficiente": self.coefficients}
        ).sort_values("Coeficiente", key=abs, ascending=False).reset_index(drop=True)


def _design_matrices(
    frame: pd.DataFrame, features: list[str], order_column: str
) -> tuple[np.ndarray, np.ndarray]:
    """Arma tensores (carreras x plazas x variables) ordenados por llegada."""
    usable = frame.dropna(subset=[order_column]).copy()
    races = list(usable.groupby("race_index", observed=True))
    if not races:
        return np.zeros((0, 0, len(features))), np.zeros((0, 0), dtype=bool)

    width = max(len(group) for _, group in races)
    design = np.zeros((len(races), width, len(features)))
    mask = np.zeros((len(races), width), dtype=bool)
    for index, (_, group) in enumerate(races):
        ordered = group.sort_values(order_column)
        values = ordered.reindex(columns=features).astype(float).fillna(0.0).to_numpy()
        design[index, : len(ordered)] = values
        mask[index, : len(ordered)] = True
    return design, mask


def fit_ranking_model(
    frame: pd.DataFrame,
    features: list[str],
    *,
    order_column: str = "position",
    l2: float = DEFAULT_L2,
) -> RankingModel:
    """Ajusta el modelo de orden por maxima verosimilitud.

    Los coeficientes salen de los datos; no hay ningun peso elegido a mano.
    """
    design, mask = _design_matrices(frame, features, order_column)
    if design.shape[0] < 3:
        return RankingModel(
            features=features,
            coefficients=np.zeros(len(features)),
            means=np.zeros(len(features)),
            scales=np.ones(len(features)),
            races_used=int(design.shape[0]),
            converged=False,
        )

    flat = design[mask]
    means = flat.mean(axis=0)
    scales = flat.std(axis=0)
    scales[scales < 1e-8] = 1.0
    standardised = (design - means) / scales
    standardised[~mask] = 0.0

    result = optimize.minimize(
        _plackett_luce_nll,
        x0=np.zeros(len(features)),
        args=(standardised, mask, l2),
        method="L-BFGS-B",
        options={"maxiter": 400},
    )
    return RankingModel(
        features=features,
        coefficients=result.x,
        means=means,
        scales=scales,
        races_used=int(design.shape[0]),
        converged=bool(result.success),
    )


def fit_temperature(
    frame: pd.DataFrame,
    ranking: RankingModel,
    dnf: "DnfModel | None" = None,
    *,
    bounds: tuple[float, float] = (0.2, 10.0),
    order_column: str = "position",
) -> float:
    """Escala las fuerzas para que las probabilidades de victoria sean honestas.

    Busca el factor que maximiza la verosimilitud del GANADOR REAL de cada
    carrera de entrenamiento, teniendo en cuenta que un piloto no puede ganar si
    abandona. Un valor mayor que 1 significa que el ajuste base era demasiado
    timido y hay que separar mas a los de cabeza.
    """
    races = [group for _, group in frame.dropna(subset=[order_column]).groupby("race_index")]
    prepared = []
    for group in races:
        ordered = group.sort_values(order_column)
        strengths = ranking.raw_strength(ordered)
        survival = np.zeros(len(ordered))
        if dnf is not None:
            survival = np.log(np.clip(1.0 - dnf.probability(ordered), 1e-4, 1.0))
        # La fila 0 es el ganador real de esa carrera (o el poleman, si se
        # calibra el modelo de clasificacion con `order_column="quali_position"`).
        prepared.append((strengths, survival))
    if len(prepared) < 5:
        return 1.0

    def negative_log_likelihood(temperature: float) -> float:
        total = 0.0
        for strengths, survival in prepared:
            logits = strengths * temperature + survival
            total -= logits[0] - np.logaddexp.reduce(logits)
        return total

    result = optimize.minimize_scalar(
        negative_log_likelihood, bounds=bounds, method="bounded", options={"xatol": 1e-3}
    )
    return float(result.x) if result.success else 1.0


# ------------------------------------------------------------ modelo de abandono


@dataclass
class DnfModel:
    features: list[str]
    coefficients: np.ndarray
    intercept: float
    base_rate: float

    def probability(self, frame: pd.DataFrame) -> np.ndarray:
        design = frame.reindex(columns=self.features).astype(float).fillna(self.base_rate).to_numpy()
        logits = self.intercept + design @ self.coefficients
        return 1.0 / (1.0 + np.exp(-logits))


def fit_dnf_model(frame: pd.DataFrame, features: list[str] = tuple(DNF_FEATURES)) -> DnfModel:
    """Regresion logistica de abandono ajustada con scipy."""
    features = list(features)
    usable = frame.dropna(subset=["dnf"])
    base_rate = float(usable["dnf"].mean()) if not usable.empty else 0.12
    design = usable.reindex(columns=features).astype(float).fillna(base_rate).to_numpy()
    target = usable["dnf"].astype(float).to_numpy()

    if len(target) < 30 or target.sum() == 0:
        logit = np.log(max(base_rate, 1e-3) / max(1 - base_rate, 1e-3))
        return DnfModel(features, np.zeros(len(features)), logit, base_rate)

    means, scales = design.mean(axis=0), design.std(axis=0)
    scales[scales < 1e-8] = 1.0
    standardised = (design - means) / scales

    def negative_log_likelihood(parameters: np.ndarray) -> float:
        intercept, weights = parameters[0], parameters[1:]
        logits = intercept + standardised @ weights
        # Forma estable de -log-verosimilitud logistica.
        return float(np.sum(np.logaddexp(0, logits) - target * logits) + 0.5 * weights @ weights)

    start = np.zeros(len(features) + 1)
    start[0] = np.log(max(base_rate, 1e-3) / max(1 - base_rate, 1e-3))
    result = optimize.minimize(negative_log_likelihood, start, method="L-BFGS-B")

    weights = result.x[1:] / scales
    intercept = float(result.x[0] - means @ weights)
    return DnfModel(features, weights, intercept, base_rate)


# ---------------------------------------------------------- calibracion de clima


# Lluvia acumulada en la ventana de carrera a partir de la cual se considera
# mojada. Con 0.2 mm se colaban carreras con llovizna irrelevante y el efecto
# medido salia invertido; con 2 mm el filtro identifica carreras realmente
# mojadas (Japon 2022, Brasil 2024, Gran Bretaña 2025).
WET_THRESHOLD_MM = 2.0

# Minimo de carreras de cada tipo para que la comparacion mojado/seco signifique
# algo. Por debajo, la calibracion se deja neutra en vez de inventar un efecto.
MIN_RACES_PER_CONDITION = 5


@dataclass
class WeatherCalibration:
    """Cuanto cambia una carrera mojada respecto de una seca, medido de los datos."""

    dnf_ratio: float = 1.0
    noise_ratio: float = 1.0
    wet_races: int = 0
    dry_races: int = 0

    @property
    def measured(self) -> bool:
        return (
            self.wet_races >= MIN_RACES_PER_CONDITION
            and self.dry_races >= MIN_RACES_PER_CONDITION
        )


def calibrate_weather(history: pd.DataFrame, wet_flags: dict[int, bool]) -> WeatherCalibration:
    """Compara abandonos y desorden entre carreras mojadas y secas.

    `wet_flags` mapea `race_index` -> hubo lluvia, segun el clima observado.
    Si no hay suficientes carreras de cada tipo devuelve una calibracion neutra
    (razon 1.0), es decir: la lluvia no altera nada, que es la posicion honesta
    cuando no hay evidencia.
    """
    if history.empty or not wet_flags:
        return WeatherCalibration()

    frame = history.copy()
    frame["wet"] = frame["race_index"].map(wet_flags)
    frame = frame.dropna(subset=["wet"])
    if frame.empty:
        return WeatherCalibration()

    wet, dry = frame.loc[frame["wet"].astype(bool)], frame.loc[~frame["wet"].astype(bool)]
    wet_races = int(wet["race_index"].nunique())
    dry_races = int(dry["race_index"].nunique())
    if wet_races < MIN_RACES_PER_CONDITION or dry_races < MIN_RACES_PER_CONDITION:
        # Sin evidencia suficiente no se aplica ningun efecto: razones en 1.0.
        return WeatherCalibration(wet_races=wet_races, dry_races=dry_races)

    dnf_wet, dnf_dry = float(wet["dnf"].mean()), float(dry["dnf"].mean())
    spread_wet = float((wet["grid"] - wet["position"]).abs().mean())
    spread_dry = float((dry["grid"] - dry["position"]).abs().mean())

    return WeatherCalibration(
        dnf_ratio=dnf_wet / dnf_dry if dnf_dry > 0 else 1.0,
        noise_ratio=spread_wet / spread_dry if spread_dry > 0 else 1.0,
        wet_races=wet_races,
        dry_races=dry_races,
    )


# ------------------------------------------------------------------ simulacion


@dataclass
class RacePrediction:
    """Resultado de la simulacion: siempre probabilidades, nunca un unico orden."""

    table: pd.DataFrame
    position_matrix: pd.DataFrame
    simulations: int
    grid_known: bool
    assumptions: dict = field(default_factory=dict)


def simulate_race(
    entries: pd.DataFrame,
    ranking: RankingModel,
    dnf: DnfModel,
    *,
    simulations: int = 20000,
    noise_scale: float = 1.0,
    rain_probability: float = 0.0,
    weather: WeatherCalibration | None = None,
    grid_known: bool = True,
    quali_model: RankingModel | None = None,
    seed: int | None = 7,
) -> RacePrediction:
    """Simula la carrera N veces y devuelve las probabilidades de cada resultado.

    Cuando la parrilla todavia no existe (`grid_known=False`) se simula primero
    la clasificacion con `quali_model` y cada simulacion usa SU propia parrilla,
    de modo que la incertidumbre de la qualy se propaga al resultado en vez de
    darse por conocida.
    """
    if entries.empty:
        return RacePrediction(pd.DataFrame(), pd.DataFrame(), 0, grid_known)

    generator = np.random.default_rng(seed)
    drivers = entries["driverId"].to_numpy()
    labels = entries.get("code", pd.Series(drivers)).fillna(pd.Series(drivers)).to_numpy()
    count = len(entries)

    rain = float(np.clip(rain_probability, 0, 100)) / 100.0
    calibration = weather or WeatherCalibration()
    # La lluvia interpola entre el comportamiento seco y el mojado medido.
    noise_multiplier = 1.0 + rain * (calibration.noise_ratio - 1.0)
    dnf_multiplier = 1.0 + rain * (calibration.dnf_ratio - 1.0)

    dnf_probability = np.clip(dnf.probability(entries) * dnf_multiplier, 0.005, 0.85)

    if grid_known:
        strengths = ranking.strength(entries)
        strength_matrix = np.tile(strengths, (simulations, 1))
        pole_counts = np.zeros(count)
    else:
        if quali_model is None:
            raise ValueError("Sin parrilla conocida hace falta un modelo de clasificacion.")
        quali_strength = quali_model.strength(entries)
        quali_noise = generator.gumbel(size=(simulations, count))
        simulated_grid_order = np.argsort(-(quali_strength + quali_noise), axis=1)
        # Posicion de parrilla simulada de cada piloto en cada simulacion.
        simulated_grid = np.empty((simulations, count), dtype=float)
        rows = np.arange(simulations)[:, None]
        simulated_grid[rows, simulated_grid_order] = np.arange(1, count + 1)
        pole_counts = (simulated_grid == 1).sum(axis=0)

        # La fuerza de carrera se recalcula con la parrilla de cada simulacion.
        strength_matrix = np.empty((simulations, count))
        base = entries.copy()
        grid_feature_index = [
            ranking.features.index(name)
            for name in ("grid_score", "grid_x_overtaking")
            if name in ranking.features
        ]
        if not grid_feature_index:
            strength_matrix[:] = ranking.strength(base)
        else:
            overtaking = base.get("overtaking_index", pd.Series(np.zeros(count))).to_numpy()
            # Referencia del calendario completo, calculada al armar las variables.
            reference = base.get("overtaking_reference", pd.Series(np.zeros(count))).to_numpy()
            design = base.reindex(columns=ranking.features).astype(float).fillna(0.0).to_numpy()
            for simulation in range(simulations):
                grid_score = -np.log(simulated_grid[simulation])
                current = design.copy()
                if "grid_score" in ranking.features:
                    current[:, ranking.features.index("grid_score")] = grid_score
                if "grid_x_overtaking" in ranking.features:
                    current[:, ranking.features.index("grid_x_overtaking")] = grid_score * (
                        overtaking - reference
                    )
                standardised = (current - ranking.means) / ranking.scales
                # La temperatura calibrada tambien se aplica aqui: sin ella las
                # probabilidades salen mucho mas planas que con parrilla conocida.
                strength_matrix[simulation] = (
                    standardised @ ranking.coefficients
                ) * ranking.temperature

    # Sorteo de abandonos y del orden de los que terminan.
    retired = generator.random((simulations, count)) < dnf_probability
    gumbel = generator.gumbel(size=(simulations, count)) * noise_scale * noise_multiplier
    scores = strength_matrix + gumbel
    # Los abandonos quedan detras de cualquier clasificado.
    scores = np.where(retired, scores - 1e6, scores)

    order = np.argsort(-scores, axis=1)
    positions = np.empty((simulations, count), dtype=np.int16)
    rows = np.arange(simulations)[:, None]
    positions[rows, order] = np.arange(1, count + 1)

    position_counts = np.zeros((count, count))
    for place in range(1, count + 1):
        position_counts[:, place - 1] = (positions == place).sum(axis=0)
    position_matrix = pd.DataFrame(
        position_counts / simulations,
        index=labels,
        columns=[f"P{place}" for place in range(1, count + 1)],
    )

    table = pd.DataFrame(
        {
            "driverId": drivers,
            "Piloto": labels,
            "Equipo": entries.get("constructorName", pd.Series([""] * count)).to_numpy(),
            "Parrilla": entries["grid"].to_numpy() if grid_known else np.nan,
            "P(victoria)": (positions == 1).mean(axis=0) * 100,
            "P(podio)": (positions <= 3).mean(axis=0) * 100,
            "P(puntos)": (positions <= 10).mean(axis=0) * 100,
            "P(abandono)": retired.mean(axis=0) * 100,
            "Posicion esperada": positions.mean(axis=0),
            "Posicion mediana": np.median(positions, axis=0),
        }
    )
    if not grid_known:
        table["P(pole)"] = pole_counts / simulations * 100
    table = table.sort_values("P(victoria)", ascending=False).reset_index(drop=True)

    assumptions = {
        "Simulaciones": f"{simulations:,}".replace(",", "."),
        "Parrilla": "real" if grid_known else "simulada desde el modelo de clasificacion",
        "Prob. de lluvia usada": f"{rain * 100:.0f} %",
    }
    if calibration.measured:
        assumptions["Efecto medido de la lluvia"] = (
            f"abandonos x{calibration.dnf_ratio:.2f}, desorden x{calibration.noise_ratio:.2f} "
            f"({calibration.wet_races} carreras mojadas vs {calibration.dry_races} secas)"
        )
    else:
        assumptions["Efecto de la lluvia"] = (
            "sin calibrar: no hay suficientes carreras mojadas en la ventana, se deja neutro"
        )
    return RacePrediction(table, position_matrix, simulations, grid_known, assumptions)


def predicted_order(table: pd.DataFrame) -> pd.DataFrame:
    """Orden de llegada mas probable, una fila por puesto de P1 a PN.

    Se ordena por posicion esperada (el promedio de las simulaciones), que es la
    respuesta a "donde termina cada uno". La probabilidad de victoria responde
    otra pregunta distinta y puede dar un orden ligeramente diferente.
    """
    if table.empty:
        return table
    ordered = table.sort_values(
        ["Posicion esperada", "P(victoria)"], ascending=[True, False]
    ).reset_index(drop=True)
    ordered.insert(0, "Pos", range(1, len(ordered) + 1))
    return ordered


# ------------------------------------------------------------------- backtest


def _podium_hits(predicted: list, actual: list) -> int:
    return len(set(predicted[:3]) & set(actual[:3]))


def backtest(
    featured: pd.DataFrame,
    *,
    features: list[str] = tuple(RACE_FEATURES),
    min_training_races: int = 20,
    max_test_races: int | None = None,
    simulations: int = 4000,
    seed: int = 7,
    grid_known: bool = True,
    only_season: int | None = None,
) -> dict:
    """Evaluacion honesta: para cada carrera se reentrena solo con las anteriores.

    Compara contra dos referencias que hay que superar para que el modelo sirva:
      - la parrilla (gana el poleman),
      - el campeonato (gana el lider del mundial en ese momento).

    Devuelve metricas y el detalle carrera por carrera.
    """
    features = list(features)
    frame = featured.dropna(subset=["position"]).copy()
    race_list = sorted(frame["race_index"].unique())
    test_races = race_list[min_training_races:]
    if only_season is not None:
        # Evalua solo esa temporada, pero el entrenamiento sigue usando todo lo
        # anterior: es la prueba de "como si cada carrera fuese nueva".
        test_races = [race for race in test_races if race // 100 == only_season]
    if max_test_races:
        test_races = test_races[-max_test_races:]

    rows = []
    for target in test_races:
        training = frame.loc[frame["race_index"] < target]
        current = frame.loc[frame["race_index"] == target].copy()
        if training.empty or current.empty:
            continue

        ranking = fit_ranking_model(training, features)
        dnf_model = fit_dnf_model(training)
        ranking.temperature = fit_temperature(training, ranking, dnf_model)

        quali_model = None
        if not grid_known:
            # Sin parrilla hay que predecir primero la clasificacion.
            quali_model = fit_ranking_model(training, QUALI_FEATURES, order_column="quali_position")
            quali_model.temperature = fit_temperature(
                training, quali_model, order_column="quali_position"
            )

        prediction = simulate_race(
            current,
            ranking,
            dnf_model,
            simulations=simulations,
            grid_known=grid_known,
            quali_model=quali_model,
            seed=seed,
        )
        if prediction.table.empty:
            continue

        actual_order = current.sort_values("position")["driverId"].tolist()
        predicted_order = prediction.table["driverId"].tolist()
        # Para acertar el podio hay que ordenar por probabilidad de PODIO, no de
        # victoria: son preguntas distintas y dan conjuntos distintos.
        podium_order = (
            prediction.table.sort_values("P(podio)", ascending=False)["driverId"].tolist()
        )
        grid_order = current.sort_values("grid")["driverId"].tolist()

        # Referencia de campeonato: puntos de LA TEMPORADA EN CURSO antes de esta
        # carrera. Acumular varias temporadas daba como "lider" a quien mas puntos
        # sumo en cinco años, que no es el campeonato de nadie.
        season_to_date = training.loc[training["season"] == current["season"].iloc[0]]
        if season_to_date.empty:
            season_to_date = training
        standings = (
            season_to_date.groupby("driverId", observed=True)["points"]
            .sum()
            .sort_values(ascending=False)
        )
        standings_order = [d for d in standings.index if d in set(actual_order)]

        winner = actual_order[0]
        win_probabilities = dict(
            zip(prediction.table["driverId"], prediction.table["P(victoria)"] / 100)
        )
        probability_of_winner = win_probabilities.get(winner, 0.0)

        merged = current.merge(
            prediction.table[["driverId", "Posicion esperada"]], on="driverId", how="left"
        ).dropna(subset=["Posicion esperada"])
        correlation = (
            stats.spearmanr(merged["position"], merged["Posicion esperada"]).statistic
            if len(merged) > 3
            else np.nan
        )

        rows.append(
            {
                "race_index": target,
                "season": int(current["season"].iloc[0]),
                "round": int(current["round"].iloc[0]),
                "raceName": current["raceName"].iloc[0] if "raceName" in current else "",
                "winner": winner,
                "model_pick": predicted_order[0],
                "model_hit": predicted_order[0] == winner,
                "grid_pick": grid_order[0] if grid_order else None,
                "grid_hit": bool(grid_order) and grid_order[0] == winner,
                "standings_pick": standings_order[0] if standings_order else None,
                "standings_hit": bool(standings_order) and standings_order[0] == winner,
                "model_podium_hits": _podium_hits(podium_order, actual_order),
                # Con dos pilotos practicamente empatados, "acerto el favorito" es
                # casi una moneda. Que el ganador real aparezca entre los primeros
                # del modelo mide lo mismo sin castigar un empate legitimo.
                "winner_in_top3": winner in predicted_order[:3],
                "winner_in_top5": winner in predicted_order[:5],
                "grid_winner_in_top3": winner in grid_order[:3],
                "grid_podium_hits": _podium_hits(grid_order, actual_order),
                "winner_probability": probability_of_winner,
                "brier": float(
                    np.mean(
                        [
                            (probability - (driver == winner)) ** 2
                            for driver, probability in win_probabilities.items()
                        ]
                    )
                ),
                "spearman": float(correlation) if correlation == correlation else np.nan,
                "pole_hit": (
                    bool(
                        prediction.table.sort_values("P(pole)", ascending=False)["driverId"].iloc[0]
                        == current.sort_values("grid")["driverId"].iloc[0]
                    )
                    if "P(pole)" in prediction.table.columns and not current.empty
                    else np.nan
                ),
            }
        )

    detail = pd.DataFrame(rows)
    if detail.empty:
        return {"detail": detail, "metrics": {}}

    metrics = {
        "Carreras evaluadas": int(len(detail)),
        "Acierto del ganador (modelo)": float(detail["model_hit"].mean() * 100),
        "Acierto del ganador (parrilla)": float(detail["grid_hit"].mean() * 100),
        "Acierto del ganador (campeonato)": float(detail["standings_hit"].mean() * 100),
        "Podio: aciertos de 3 (modelo)": float(detail["model_podium_hits"].mean()),
        "Podio: aciertos de 3 (parrilla)": float(detail["grid_podium_hits"].mean()),
        "Ganador entre los 3 primeros del modelo": float(detail["winner_in_top3"].mean() * 100),
        "Ganador entre los 3 primeros de la parrilla": float(detail["grid_winner_in_top3"].mean() * 100),
        "Ganador entre los 5 primeros del modelo": float(detail["winner_in_top5"].mean() * 100),
        "Correlacion de orden (Spearman)": float(detail["spearman"].mean()),
        "Brier de victoria (menor es mejor)": float(detail["brier"].mean()),
        "Prob. media asignada al ganador real": float(detail["winner_probability"].mean() * 100),
    }
    if "pole_hit" in detail.columns and detail["pole_hit"].notna().any():
        metrics["Acierto de la pole"] = float(detail["pole_hit"].dropna().mean() * 100)
    return {"detail": detail, "metrics": metrics}


# --------------------------------------------------------- carrera por disputar


def build_upcoming_rows(
    results: pd.DataFrame,
    *,
    season: int,
    round_number: int,
    circuit_id: str,
    race_name: str = "",
    circuit_name: str = "",
    race_date: str | None = None,
) -> pd.DataFrame:
    """Crea las filas de una carrera que todavia no se corrio.

    Toma la lista de inscriptos de la ultima carrera disputada de esa temporada:
    es un SUPUESTO (asume la misma alineacion) y la UI debe declararlo. Las
    columnas de resultado quedan vacias; se completan por simulacion, nunca con
    datos inventados.

    Concatenar estas filas al historico permite reutilizar el mismo calculo de
    variables causales: la carrera nueva ve exactamente lo anterior a ella.
    """
    season_results = results.loc[results["season"] == season]
    if season_results.empty:
        return pd.DataFrame()

    last_round = int(season_results["round"].max())
    entries = season_results.loc[season_results["round"] == last_round].copy()

    template = entries[
        [column for column in ("driverId", "code", "driverName", "constructorId", "constructorName")
         if column in entries.columns]
    ].drop_duplicates("driverId")

    upcoming = template.assign(
        season=season,
        round=round_number,
        circuitId=circuit_id,
        raceName=race_name or f"Ronda {round_number}",
        circuitName=circuit_name or circuit_id,
        date=race_date,
        position=np.nan,
        positionText=None,
        points=np.nan,
        grid=np.nan,
        laps=np.nan,
        status=None,
        timeText=None,
        timeMillis=np.nan,
        fastestLapRank=np.nan,
        fastestLapTime=None,
    )
    return upcoming.reset_index(drop=True)


def attach_known_grid(
    upcoming: pd.DataFrame, qualifying: pd.DataFrame, *, season: int, round_number: int
) -> tuple[pd.DataFrame, bool]:
    """Pega la parrilla real si la clasificacion de esa ronda ya se publico.

    Devuelve las filas y si la parrilla es real. No inventa una parrilla: si no
    hay clasificacion, la simulacion la estima y lo dice.
    """
    if qualifying is None or qualifying.empty:
        return upcoming, False
    session = qualifying.loc[
        (qualifying["season"] == season) & (qualifying["round"] == round_number)
    ]
    if session.empty:
        return upcoming, False

    grid = session[["driverId", "position"]].rename(columns={"position": "_grid"})
    merged = upcoming.merge(grid, on="driverId", how="left")
    if merged["_grid"].isna().all():
        return upcoming, False
    merged["grid"] = merged["_grid"]
    return merged.drop(columns=["_grid"]), True


# ------------------------------------------------- similitud entre circuitos


# Caracteristicas con las que se compara un circuito con otro. Todas medidas de
# los resultados, ninguna opinada: duracion de la vuelta (tamaño y velocidad),
# cuanto se adelanta, cuanto manda la parrilla y cuantos abandonos hay.
CIRCUIT_TRAITS = ["lap_seconds", "overtaking_index", "grid_retention", "dnf_rate"]

# Ancho del nucleo gaussiano de similitud, en desviaciones estandar. Mas chico =
# solo cuentan los circuitos muy parecidos; mas grande = todos pesan casi igual.
DEFAULT_SIMILARITY_WIDTH = 1.0


def circuit_traits(history: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """Caracteriza cada circuito con cifras medidas de las carreras."""
    if history.empty or profiles.empty:
        return pd.DataFrame(columns=["circuitId", *CIRCUIT_TRAITS])

    frame = history.copy()
    if "flap_seconds" not in frame.columns:
        frame["flap_seconds"] = frame["fastestLapTime"].apply(_lap_time_seconds)
    laps = (
        frame.groupby("circuitId", observed=True)["flap_seconds"]
        .median()
        .rename("lap_seconds")
        .reset_index()
    )
    traits = profiles.merge(laps, on="circuitId", how="left")
    traits["lap_seconds"] = traits["lap_seconds"].fillna(traits["lap_seconds"].median())
    return traits[["circuitId", *CIRCUIT_TRAITS]]


def circuit_similarity(
    traits: pd.DataFrame, *, width: float = DEFAULT_SIMILARITY_WIDTH
) -> pd.DataFrame:
    """Matriz circuito x circuito con cuanto se parecen, entre 0 y 1.

    Se estandariza cada caracteristica y se aplica un nucleo gaussiano sobre la
    distancia. Un circuito consigo mismo vale 1; dos muy distintos tienden a 0.
    """
    if traits.empty:
        return pd.DataFrame()

    values = traits[CIRCUIT_TRAITS].astype(float)
    standardised = (values - values.mean()) / values.std(ddof=0).replace(0, 1.0)
    matrix = standardised.to_numpy()
    squared = ((matrix[:, None, :] - matrix[None, :, :]) ** 2).sum(axis=2)
    similarity = np.exp(-squared / (2 * width**2 * len(CIRCUIT_TRAITS)))
    return pd.DataFrame(similarity, index=traits["circuitId"], columns=traits["circuitId"])


def add_circuit_similarity_features(
    frame: pd.DataFrame, similarity: pd.DataFrame, *, key: str = "constructorId"
) -> pd.DataFrame:
    """Cuanto rinde cada equipo en circuitos PARECIDOS a este, y no en general.

    Responde a que un coche no se comporta igual en todos lados: lo que hizo en
    Imola no deberia pesar lo mismo que lo que hizo en Monaco a la hora de
    predecir un callejero. En vez de agrupar circuitos en categorias arbitrarias
    se pondera cada carrera pasada por lo parecido que es su circuito al de
    ahora, con el nucleo de `circuit_similarity`.

    El valor es la DIFERENCIA contra el nivel general del equipo: positivo
    significa "aqui rinde mejor de lo que acostumbra". Solo mira carreras
    anteriores.

    MEDIDO Y DESCARTADO COMO VARIABLE DEL MODELO. La idea es correcta y la
    matriz de similitud es buena (Baku sale junto a Jeddah y Singapur, Monaco
    junto a Mexico), pero agregarla al ajuste empeoraba la prediccion de forma
    consistente: -3.5 pp antes de la clasificacion y -5 pp con parrilla, con el
    coeficiente saliendo NEGATIVO, es decir al reves de lo que tendria sentido.
    Con ~100 carreras y ~10 equipos, la diferencia entre el promedio ponderado y
    el general es una cantidad chica y ruidosa, y el ajuste termina modelando esa
    reversion a la media.

    La funcion se conserva porque la similitud entre circuitos si es informativa
    y la app la muestra, pero NO alimenta la prediccion.
    """
    column = "team_circuit_affinity"
    if frame.empty or similarity.empty:
        return frame.assign(**{column: 0.0})

    ordered = frame.sort_values("race_index").copy()
    ordered[column] = 0.0
    for _, group in ordered.groupby(key, observed=True):
        indices = group.index.to_numpy()
        circuits = group["circuitId"].to_numpy()
        scores = group["finish_score"].to_numpy(dtype=float)
        races = group["race_index"].to_numpy()
        for position in range(len(group)):
            past = races < races[position]
            if past.sum() < 5:
                continue
            past_scores = scores[past]
            valid = ~np.isnan(past_scores)
            if valid.sum() < 5:
                continue
            target = circuits[position]
            if target not in similarity.index:
                continue
            weights = similarity.loc[target, circuits[past][valid]].to_numpy(dtype=float)
            if weights.sum() <= 0:
                continue
            weighted = float(np.average(past_scores[valid], weights=weights))
            overall = float(past_scores[valid].mean())
            ordered.at[indices[position], column] = weighted - overall
    return ordered.sort_index()
