"""Implementación del algoritmo de Dijkstra para planificación de rutas de reparto.

Dijkstra resuelve exactamente el mismo problema que A* sobre el mismo espacio de estados
S = (l, t, U) y con los mismos costos de transición no negativos c(S, p_i) >= 0.
Utiliza g(n) como prioridad de búsqueda exclusiva, equivalente a ejecutar A* con la
heurística nula h(n) = 0.
"""

from __future__ import annotations

from delivery_agent.algorithms.astar import SearchResult, solve_astar
from delivery_agent.domain.problem import DeliveryProblem
from delivery_agent.domain.state import DeliveryState


def zero_heuristic(problem: DeliveryProblem, state: DeliveryState) -> float:
    """Heurística nula que devuelve idénticamente 0 para todo estado."""
    return 0.0


def solve_dijkstra(
    problem: DeliveryProblem,
    initial_state: DeliveryState | None = None,
    *,
    alpha: float | None = None,
) -> SearchResult:
    """Ejecuta el algoritmo de Dijkstra para encontrar la ruta factible de costo mínimo.

    Reutiliza el motor de transiciones, la función de costos y la cola de prioridad de A*
    con h(n) = 0, garantizando optimalidad global para costos no negativos.
    """
    return solve_astar(
        problem,
        initial_state=initial_state,
        heuristic_fn=zero_heuristic,
        alpha=alpha,
        algorithm_name="Dijkstra",
    )

