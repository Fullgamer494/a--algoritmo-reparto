"""Heurística admisible para la planificación de rutas de reparto mediante A*.

Demostración matemática de admisibilidad
----------------------------------------
Sea un estado de búsqueda $S = (l, t, U)$, donde $l$ es la ubicación actual,
$t$ es la hora actual en minutos y $U$ es el conjunto de paquetes pendientes.

Definimos la heurística:

$$
h(S) =
\\begin{cases}
0 & \\text{si } U = \\varnothing \\\\
\\min_{p_i \\in U,\\, \\tau(l, d_i) \\text{ disponible}} \\tau(l, d_i) & \\text{si } U \\neq \\varnothing
\\end{cases}
$$

donde $d_i = \\text{destination}(p_i)$ y $\\tau(l, d_i) \\ge 0$ es el tiempo de
desplazamiento directo desde $l$ hasta $d_i$. Si $U \\neq \\varnothing$ pero
ningún paquete pendiente tiene un traslado directo disponible desde $l$
($\\tau(l, d_i) = \\text{None}$ para todo $p_i \\in U$), entonces no existe
ninguna transición factible desde $S$ y $h(S) = +\\infty$.

**Prueba de que $h(S) \\le C^*(S)$ para todo estado $S$:**

1. Caso base ($U = \\varnothing$):
   El estado ya es objetivo, por lo que el costo óptimo restante es $C^*(S) = 0$
   y $h(S) = 0 \\le C^*(S)$.

2. Caso inductivo ($U \\neq \\varnothing$):
   Cualquier ruta completa y factible que parta de $S$ debe elegir como primera
   acción la entrega de algún paquete $p_k \\in U$ cuyo traslado $\\tau(l, d_k)$
   esté disponible. El costo incremental de esa primera transición es:

   $$
   c(S, p_k) = \\Delta t_k + \\alpha w_k \\max(0, f_k - r_k)
             = \\tau(l, d_k) + e_k + q_k + \\alpha w_k \\max(0, f_k - r_k)
   $$

   Dado que:
   - el tiempo de espera $e_k = \\max(0, o_k - a_k) \\ge 0$,
   - el tiempo de servicio $q_k \\ge 0$,
   - el parámetro $\\alpha \\ge 0$, el peso $w_k > 0$ y el retraso $d_k \\ge 0$,
   - todos los costos de las transiciones posteriores en la ruta también son no negativos,

   se cumple que para toda ruta factible que inicie entregando $p_k$:

   $$
   C^*(S) \\ge c(S, p_k) \\ge \\tau(l, d_k) \\ge \\min_{p_i \\in U} \\tau(l, d_i) = h(S)
   $$

   Por tanto, $h(S)$ nunca sobreestima el costo real mínimo restante $C^*(S)$ y
   es **admisible**.

Demostración matemática de consistencia (monotonicidad)
--------------------------------------------------------
Una heurística $h$ es consistente si para todo estado $S$ y todo sucesor válido
$S'$ alcanzado mediante la entrega de un paquete $p_k \\in U$ con costo de
transición $c(S, p_k)$, se cumple:

$$
h(S) \\le c(S, p_k) + h(S')
$$

Demostración:
1. Por definición de $h(S) = \\min_{p_i \\in U} \\tau(l, d_i)$, dado que $p_k \\in U$,
   se cumple inmediatamente:
   $$
   h(S) \\le \\tau(l, d_k)
   $$

2. Por la definición del costo de transición:
   $$
   c(S, p_k) = \\tau(l, d_k) + e_k + q_k + \\alpha w_k \\max(0, f_k - r_k)
   $$
   Dado que la espera $e_k \\ge 0$, el servicio $q_k \\ge 0$ y la penalización
   ponderada es no negativa, se tiene:
   $$
   c(S, p_k) \\ge \\tau(l, d_k) \\ge h(S)
   $$

3. Para cualquier estado sucesor $S'$, se cumple $h(S') \\ge 0$. Por tanto:
   $$
   h(S) \\le c(S, p_k) \\le c(S, p_k) + h(S')
   $$

**Conclusión sobre la desigualdad triangular:**
La consistencia de $h(S)$ se cumple de manera incondicional para cualquier matriz
de tiempos no negativos, **sin necesidad de exigir o asumir que la matriz cumpla
la desigualdad triangular**.

**Implicación para el algoritmo A*:**
Bajo esta heurística consistente, A* nunca expande un estado con un costo $g$
subóptimo, por lo que nunca se producen reaperturas con $h(S)$. No obstante, el
algoritmo A* mantiene el soporte de reaperturas en `best_g` para garantizar
robustez y optimalidad en caso de que se utilicen heurísticas arbitrarias
admisibles pero no consistentes.
"""

from __future__ import annotations

import math
from typing import Callable, TypeAlias

from delivery_agent.domain.problem import DeliveryProblem
from delivery_agent.domain.state import DeliveryState

HeuristicFn: TypeAlias = Callable[[DeliveryProblem, DeliveryState], float]


def min_travel_time_heuristic(problem: DeliveryProblem, state: DeliveryState) -> float:
    """Calcula h(S) = min_{p_i in U} tau(l, destination_i), o 0.0 si U está vacío.

    Ignora los tiempos de servicio, esperas y penalizaciones futuras (todos >= 0).
    Si un traslado tau(l, destination_i) es None (no disponible), se descarta del mínimo.
    Si U no está vacío y ningún paquete pendiente es alcanzable desde l, devuelve math.inf.
    """
    if state.is_goal:
        return 0.0

    min_travel: int | None = None
    for pkg_id in state.pending_packages:
        pkg = problem.get_package(pkg_id)
        travel = problem.travel_time(state.location, pkg.destination)
        if travel is None:
            continue
        if min_travel is None or travel < min_travel:
            min_travel = travel

    if min_travel is None:
        return math.inf

    return float(min_travel)


compute_heuristic: HeuristicFn = min_travel_time_heuristic

