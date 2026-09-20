# F1 Race Engineering Dashboard

Aplicación Streamlit de análisis de Fórmula 1 con nivel de detalle de ingeniería de
pista. Combina la telemetría y el cronometraje de **FastF1** (2018 en adelante) con los
resultados y campeonatos de **Jolpica** (sucesor de Ergast, desde 1950), con filtros
cruzados en cascada por año, circuito, sesión, piloto y equipo. Incluye un predictor
probabilístico de carrera que suma el clima online de **Open-Meteo**.

## Puesta en marcha

```bash
pip install -r requirements.txt
```

```bash
python -m streamlit run app.py
```

La app abre en `http://localhost:8501`. La primera carga de una sesión descarga los
datos de la API de F1 y tarda entre 20 y 45 s; a partir de ahí el caché en disco la
deja en unos 4 s. El directorio `cache/` se crea solo al arrancar.

Probado con Python 3.14 en Windows (FastF1 3.8.3, Streamlit 1.64, Plotly 7.1), con las
dependencias instaladas de forma global.

> Se usa `python -m streamlit` porque en esta máquina pip instaló los ejecutables en
> `%APPDATA%\Python\Python314\Scripts`, que no está en el `PATH`. Si agregás ese
> directorio al `PATH`, `streamlit run app.py` funciona igual.

## Publicar en internet (Streamlit Community Cloud)

Gratis y sin tarjeta. El repositorio ya está listo: `cache/` y `.venv/` están excluidos.

1. Creá un repositorio vacío en [github.com/new](https://github.com/new) — puede ser
   privado, Streamlit Cloud también despliega repositorios privados.
2. Conectalo y subí el código:

   ```bash
   git remote add origin https://github.com/TU_USUARIO/TU_REPO.git
   git branch -M main
   git push -u origin main
   ```

3. Entrá en [share.streamlit.io](https://share.streamlit.io), *New app*, elegí el
   repositorio, rama `main`, archivo principal `app.py`.
4. En **Advanced settings** elegí **Python 3.13**. No uses una versión mayor: FastF1 y sus
   dependencias todavía no publican paquetes para 3.14.

Queda en `https://TU_USUARIO-TU_REPO.streamlit.app`, permanente y sin depender de tu máquina.

### Qué esperar del plan gratuito

- **Memoria**: 1 GB. Medido con dos sesiones completas en memoria: **451 MB de pico**.
  Entra con margen, pero por eso `load_session` guarda como mucho
  `MAX_CACHED_SESSIONS = 3` sesiones a la vez: cada una ocupa ~150 MB y seis desbordarían
  el contenedor.
- **Primera carga lenta**: el caché de FastF1 arranca vacío en el servidor, así que la
  primera sesión de cada circuito vuelve a tardar 20-45 s. Después va rápido, hasta que
  Streamlit reinicie el contenedor.
- **Límite de Jolpica compartido**: los 500 req/h son por dirección IP, así que **todos los
  visitantes comparten el mismo presupuesto**. Con varias personas usando a la vez, las
  vistas 7 y 8 pueden mostrar el aviso de límite. El cliente reintenta con backoff, pero no
  puede crear cuota que no existe.
- **La app se duerme** tras un rato sin visitas y tarda unos segundos en despertar.

## Las ocho vistas

| # | Vista | Qué muestra | Fuente |
|---|-------|-------------|--------|
| 1 | Resumen de sesión | KPIs (ganador/pole, vuelta rápida, vueltas, temperaturas, lluvia, ritmo del top 10) y clasificación con compuestos y paradas | FastF1 |
| 2 | Ritmo de carrera | Box/violin de tiempos, ritmo en aire limpio, corrección por combustible, evolución de pista | FastF1 |
| 3 | Neumáticos y estrategia | Gantt de stints, degradación con regresión e IC 95 %, comparador de estrategias, simulador undercut/overcut | FastF1 |
| 4 | Comparación de telemetría | Seis canales contra distancia, delta acumulado, mapa por velocidad y dominancia en 25 mini-sectores | FastF1 |
| 5 | Vueltas y sectores | Tiempos por sector, velocidades de trampa, vuelta ideal vs real, índice de consistencia | FastF1 |
| 6 | Carrera vuelta a vuelta | Posición, hueco al líder y al coche de adelante, con banderas/SC/VSC sombreados | FastF1 |
| 7 | Campeonato e histórico | Evolución de puntos, head-to-head entre compañeros, histórico de piloto y de circuito | Jolpica |
| 8 | Predictor de carrera | Probabilidades de pole, victoria, podio, puntos y abandono; perfil del circuito; clima online; backtest | Jolpica + Open-Meteo |

La vista 8 tiene su propia sección más abajo. Las vistas 1 a 6 necesitan FastF1 y por
tanto una temporada **2018 o posterior**. Con
una temporada anterior la app no se rompe: muestra un aviso explícito de telemetría no
disponible y la vista 7 sigue funcionando con datos reales desde 1950.

## Arquitectura

```
app.py                     entrada y layout de las ocho vistas
.streamlit/config.toml     tema oscuro (Streamlit no permite fijarlo desde codigo)
data/fastf1_loader.py      sesiones, vueltas y telemetría (caché de Streamlit + disco)
data/jolpica_client.py     cliente HTTP con rate limiting y backoff exponencial
data/weather_client.py     clima por circuito (Open-Meteo: pronóstico, archivo, climatología)
analysis/pace.py           filtrado de vueltas, aire limpio, combustible, sectores
analysis/tyres.py          stints, degradación por regresión, pit loss estimado
analysis/telemetry.py      alineación por distancia, delta, mini-sectores
analysis/strategy.py       posiciones, huecos, banderas, undercut, puntos acumulados
analysis/prediction.py     modelo Plackett-Luce, abandonos, simulación Monte Carlo, backtest
ui/theme.py                paleta, template de Plotly y CSS
ui/charts.py               constructores de figuras (Plotly puro)
ui/filters.py              filtros en cascada de la barra lateral
cache/                     caché en disco de FastF1 (en .gitignore)
```

Regla de separación, verificable con `grep -rn streamlit analysis/` (sin resultados):

- `analysis/` es código puro de pandas/numpy/scipy. No importa Streamlit ni dibuja.
- `ui/charts.py` solo construye figuras de Plotly; no lee datos ni calcula métricas.
- `app.py` solo orquesta: pide datos, métricas y figuras, y las coloca en la página.

## Barra lateral

Temporada, Gran Premio y sesión van **en cascada e inmediatas**: el año limita los
circuitos, el circuito limita las sesiones y la sesión limita los pilotos. Un piloto que
no corrió esa sesión nunca aparece en la lista.

El resto (filtro de vueltas, parámetros del modelo y selección de pilotos) vive dentro de
un formulario con el botón **Aplicar filtros**: podés mover seis controles y la app
recalcula una sola vez, al pulsar el botón, en lugar de recargar en cada clic.

## Tema

El tema oscuro se fija en `.streamlit/config.toml`. Es imprescindible: Streamlit no
expone ninguna API para elegir el tema desde código, y sin ese archivo los widgets
nativos se dibujan con el tema claro del sistema, con lo que sus etiquetas quedan gris
oscuro sobre el fondo oscuro de la app. Los colores están duplicados en `ui/theme.py`
(que gobierna el CSS y el template de Plotly); si cambiás uno, cambiá el otro.

## Caché y límites de las APIs

- **FastF1**: caché en disco obligatorio en `cache/`, habilitado al arrancar con
  `@st.cache_resource`. El objeto `Session` se cachea como recurso; los DataFrames
  derivados con `@st.cache_data`.
- **Jolpica**: 4 req/s y 500 req/h. Todas las llamadas pasan por `JolpicaClient`, que
  aplica un token bucket doble (ráfaga y presupuesto horario) y reintenta con backoff
  exponencial con jitter ante 429, 5xx y timeouts, respetando `Retry-After`. Las
  respuestas se cachean una hora con `@st.cache_data(ttl=3600)`. La API es lenta
  (~7 s por request): la primera visita a las vistas 7 y 8 puede tardar un minuto largo.
- **Open-Meteo**: gratuita y sin API key. El pronóstico llega a 16 días; más allá de eso
  la app usa la climatología realmente observada de años anteriores y lo declara en
  pantalla, en vez de inventar un pronóstico.

Si Jolpica falla o limita, la app muestra el error concreto en pantalla. **Nunca
rellena huecos con datos inventados, estimados o de otra sesión.**

## El predictor (vista 8)

### Qué hace

Estima **probabilidades**, no resultados: «Norris 34 % de ganar», nunca «gana Norris».
Para cada carrera devuelve probabilidad de pole, victoria, podio, puntos y abandono, la
posición esperada y la distribución completa de posiciones de cada piloto.

Funciona en dos modos, y el backtest los mide por separado porque no son comparables:

- **Después de la clasificación**: usa la parrilla real. Es el modo preciso.
- **Antes de la clasificación**: no existe parrilla, así que cada simulación genera la
  suya con un modelo de qualy aparte. La incertidumbre de la clasificación se arrastra
  al resultado en vez de darse por conocida.

### Cómo funciona

1. A cada piloto se le asigna una «fuerza» a partir de variables causales (solo datos
   anteriores a la carrera).
2. Esa fuerza alimenta un modelo **Plackett-Luce**, el estándar para órdenes de llegada.
3. Los coeficientes se ajustan por máxima verosimilitud con `scipy.optimize`. **Ningún
   peso está elegido a mano.**
4. La carrera se simula 20 000 veces: primero quién abandona (modelo logístico aparte),
   después el orden de los supervivientes con ruido Gumbel.

### Qué mira, y qué no puede mirar

| Pedido | Cómo entra |
|--------|-----------|
| Tipo de pista (Mónaco ≠ Imola) | Interacción `parrilla × índice de adelantamiento`, medido de los datos. Es la **2ª variable más informativa** del modelo, y aprende sola que donde se adelanta la parrilla pesa menos |
| Cómo vienen en tiempos | `quali_margin`: margen en segundos respecto de la pole, no el orden |
| Estrategia y ritmo de carrera | `form_race_gain` / `team_race_gain`: posiciones que piloto y equipo suelen ganar el domingo respecto de donde salen |
| Clima | Consulta online a Open-Meteo para la fecha y coordenadas del circuito |
| **Actualizaciones técnicas** | **No existe ninguna API pública de mejoras.** El modelo no las lee: las infiere de `team_trend`, la tendencia de rendimiento del equipo. Si un paquete funciona, aparece ahí |

Sobre el clima, un detalle que condiciona el diseño: en un modelo de ranking, una
variable idéntica para todos los pilotos de la carrera **se cancela matemáticamente** y
no puede alterar el orden esperado. Por eso la lluvia no entra como término principal
sino por los dos canales donde sí tiene efecto medible: cuántos abandonos provoca y
cuánto desorden añade. Ambos se calibran comparando carreras mojadas contra secas con
clima realmente observado. Si no hay suficientes carreras de cada tipo, la calibración
se deja neutra en vez de inventar un factor.

### Cuánto acierta de verdad

Backtest walk-forward sobre **81 carreras (2022-2026)**: para cada una se reentrena el
modelo usando solo las anteriores y se predice a ciegas.

Todo con 4 semillas distintas del Monte Carlo, para poder separar señal de ruido.

| Modo | Alcance | Modelo | Referencia | Ganador en top-3 | Brier |
|------|---------|--------|-----------|------------------|-------|
| Con parrilla | 81 carreras | 59,3 % | 58,0 % (parrilla) | 87,7 % | 0,027 |
| Con parrilla | solo 2026 | 57,1 % | **64,3 %** (parrilla) | 92,9 % | 0,024 |
| Antes de la qualy | 81 carreras | **48,8 %** ±3,1 | 45,7 % (campeonato) | 78,4 % | 0,038 |
| Antes de la qualy | solo 2026 | 42,9 % | **50,0 %** (campeonato) | 69,6 % | 0,037 |

### El ruido, primero

Repetir el backtest cambiando solo la semilla mueve el acierto hasta **±5 puntos**. Sobre
81 carreras, una diferencia de 2-3 carreras no significa nada. Las métricas estables son
el **Brier** y el **ganador entre los 3 primeros**. Cualquier comparación hay que leerla
con esa vara.

### Léelo con honestidad

- El modelo **no le gana a la parrilla** de forma convincente: +1,3 pp sobre 81 carreras,
  y en 2026 queda por debajo (57,1 % contra 64,3 %).
- Antes de la clasificación le saca +3,1 pp a «gana el líder del campeonato» sobre 81
  carreras, pero en 2026 **sigue por debajo** (42,9 % contra 50,0 %). Con Antonelli
  ganando 8 de 14, esa regla trivial es durísima de batir.
- Lo que sí aporta y una regla simple no puede: **probabilidades calibradas de todo el
  pelotón** y funcionar **antes de que exista una parrilla que copiar**.
- **Aviso de sobreajuste**: varias decisiones (ventanas, variables nuevas) se tomaron
  mirando este mismo backtest. Las que se conservaron tienen además una justificación de
  mecanismo, no solo un número, pero conviene leer estas cifras como el techo optimista.

### Qué pesa de verdad (medido, no supuesto)

Caída de verosimilitud al quitar cada variable del modelo de carrera:

| Variable | Aporte |
|----------|--------|
| `grid_score` (dónde sale) | −45,2 |
| **`grid_x_overtaking` (tipo de circuito)** | **−23,4** |
| `team_race_gain` | −10,9 |
| `form_race_gain` | −5,1 |
| `quali_margin` | −2,7 |

**El tipo de circuito es la segunda variable más potente del modelo**, solo por detrás de
la parrilla. No es decorativo.

Con la parrilla conocida las variables de forma aportan poco, y no es que la forma no
importe: **es que la parrilla ya la contiene**. Donde la forma reciente sí manda es en el
modelo de clasificación, donde `form_quali_margin` (margen de qualy de las últimas
carreras) es la variable número uno, con coeficiente 0,85.

### Cómo distingue a dos compañeros de equipo

Era un problema real: casi todas las variables de equipo (`team_form`, `team_trend`,
`team_race_gain`) valen **lo mismo** para los dos pilotos del mismo coche, así que no
podían separarlos. Antes de arreglarlo, el modelo daba a los dos Mercedes 14,2 % y 13,2 %
de victoria: un empate inútil.

Se agregaron dos variables por piloto:

- **`teammate_quali_edge` / `teammate_race_edge`**: cuánto le saca cada piloto a su propio
  compañero, restando la media del equipo en esa carrera. El coche se cancela y queda el
  piloto. Valores medidos antes de Bakú 2026: Verstappen **+0,49** sobre sus compañeros,
  Norris +0,17 sobre Piastri, y un matiz fino en Mercedes: Russell clasifica mejor
  (+0,08) pero **Antonelli corre mejor** (+0,08).
- **`season_points_rate`**: puntos por carrera en la temporada. `season_form` usa la
  posición de llegada, donde ganar y ser segundo se parecen; el campeonato no (25 contra
  18). Esa diferencia es la que separa a quien gana carreras de su compañero que las
  acompaña.

Efecto medido: el acierto antes de la clasificación en 2026 subió de **21,4 % a 42,9 %**,
y en Bakú los Mercedes dejaron de estar empatados (Antonelli 31,8 % de pole contra 16,9 %
de Russell). El precio fue perder acierto en el modo con parrilla en 2026.

### Dos bugs que falseaban el modo antes de la clasificación

Probando Mónaco y Las Vegas aparecieron dos fallos en la rama de simulación sin parrilla,
ambos corregidos:

1. **No se aplicaba la temperatura de calibración.** `RankingModel.strength()` multiplica
   las fuerzas por la temperatura ajustada (~2,7), pero la rama pre-clasificación usaba los
   coeficientes crudos. Resultado: probabilidades mucho más planas de lo que correspondía.
   Al arreglarlo, la probabilidad media asignada al ganador real pasó de **14,0 % a
   31,2 %** y el Brier de 0,0382 a **0,0337**.
2. **La interacción con el tipo de pista quedaba anulada.** La referencia de
   adelantamiento se recalculaba como la mediana de los pilotos de esa carrera — que
   comparten circuito, así que daba cero para todos. El término `grid_x_overtaking`
   valía cero en todas las simulaciones sin parrilla. Ahora la referencia es la mediana
   de todo el calendario y viaja como columna con los datos.

El segundo bug explicaba un síntoma concreto: Bakú y Las Vegas daban predicciones casi
idénticas pese a ser los extremos opuestos del calendario. Corregido, el modelo reparte
confianza según el circuito, sin haber agregado ninguna variable nueva:

| Circuito | Retención de parrilla | Favorito | Suma del top-3 |
|----------|----------------------|----------|----------------|
| Bakú | 0,86 (manda la parrilla) | 39,3 % | 71,1 % |
| Las Vegas | 0,45 (caótico) | 18,5 % | 42,4 % |

### Similitud entre circuitos: medida, mostrada, y fuera del modelo

Un coche no se comporta igual en todos lados, así que los resultados de Imola no deberían
pesar lo mismo que los de Mónaco para predecir un callejero. El modelo caracteriza cada
circuito con cifras medidas —duración de la vuelta, cuánto se adelanta, cuánto manda la
parrilla, abandonos— y calcula un parecido continuo entre todos.

Sale muy razonable: **Bakú** se agrupa con **Jeddah (0,79), Singapur (0,75) y Bahréin
(0,71)**, y se aleja de Las Vegas (0,03) e Interlagos (0,08). **Mónaco** se agrupa con
**México (0,83)**.

Se probó a usarlo para ponderar la forma de cada equipo por el parecido del circuito.
**Empeoró la predicción de forma consistente**: −3,5 pp antes de la clasificación y −5 pp
con parrilla, y el coeficiente salía **negativo**, al revés de lo que tendría sentido. Con
~100 carreras y ~10 equipos, la diferencia entre el promedio ponderado y el general es una
cantidad chica y ruidosa, y el ajuste termina modelando reversión a la media.

La similitud **se muestra en la app porque es informativa, pero no alimenta la
predicción**. Lo que sí entra del circuito es la interacción `parrilla × índice de
adelantamiento`, que es la segunda variable más potente del modelo.

### Un intento fallido, documentado

El caso de McLaren en 2026 es real: Norris promedia **P9,00 y cero victorias en las rondas
1-10**, y **P2,25 con dos victorias en las 11-14**, con el margen de qualy pasando de
0,647 % a 0,086 %. El modelo reacciona tarde a ese salto.

Se probaron tres cosas: acortar la ventana de temporada (se mantuvo, efecto dentro del
ruido), usar dos ventanas a la vez (falló: son colineales y el ajuste global prefirió la
larga) y una señal explícita de mejora `max(0, forma reciente − línea base)`, no lineal a
propósito. Esta última aportó **0,0** medido de tres formas y **se quitó**: una variable
que no mide nada no debería figurar como si modelara las actualizaciones técnicas.

Capturar un cambio de nivel a mitad de temporada con un modelo lineal global ajustado
sobre 106 carreras **no se logró**. El panel «peso de la forma reciente» permite
experimentar con las ventanas.

### Repetir el test

El panel de backtest permite elegir **alcance** (solo la temporada en curso o todas) y
**modo** (con parrilla real o antes de la clasificación), y muestra la tabla carrera por
carrera: qué dijo el modelo, qué probabilidad le dio al ganador real y qué pasó.

### Coste de la primera carga

La vista descarga 5 temporadas de resultados y clasificaciones de Jolpica: unas 50
consultas, **cerca de dos minutos la primera vez**. Después queda cacheada una hora. El
backtest tarda entre 10 y 40 s y está detrás de un botón.

## Qué es medición y qué es modelo

Todo valor que sale de un modelo aparece en la UI con la etiqueta **ESTIMACIÓN** y sus
supuestos a la vista:

- **Corrección por combustible**: resta un lastre lineal configurable (0.035 s/vuelta
  por defecto) para normalizar a coche en tanque vacío.
- **Degradación**: regresión lineal `tiempo = intercepto + pendiente × vida del
  neumático` por stint, con intervalo de confianza al 95 % y R². Sobre tiempos brutos
  la pendiente mezcla desgaste con aligeramiento del coche, por eso se puede ajustar
  sobre tiempos ya corregidos.
- **Pérdida en pit lane**: se estima con las vueltas reales de entrada y salida de
  boxes de la propia sesión, comparadas contra el ritmo de referencia del piloto. Es
  editable.
- **Undercut / overcut**: proyección, no predicción. Extrapola las rectas ajustadas y
  supone pista libre: no modela tráfico, banderas, errores ni cambios de condiciones.

El delta de telemetría, la vuelta ideal, los huecos y la dominancia por mini-sectores
**no** son estimaciones: salen directamente de los datos.

## Colores

Se usan los colores oficiales de equipo de `fastf1.plotting`, así que cada piloto
mantiene el mismo color en todas las vistas. Los compañeros de equipo comparten color y
se distinguen por línea sólida frente a punteada. Los compuestos usan el mapeo oficial
de Pirelli que publica FastF1.
