"""Algoritmos de búsqueda y planificación de rutas."""

from delivery_agent.algorithms.astar import (
    SearchEvent,
    SearchMetrics,
    SearchResult,
    reconstruct_route,
    solve_astar,
)
from delivery_agent.algorithms.dijkstra import solve_dijkstra, zero_heuristic
from delivery_agent.algorithms.exhaustive import ExhaustiveResult, solve_exhaustive
from delivery_agent.algorithms.greedy import solve_greedy
from delivery_agent.algorithms.heuristics import (
    HeuristicFn,
    compute_heuristic,
    min_travel_time_heuristic,
)

__all__ = [
    "ExhaustiveResult",
    "HeuristicFn",
    "SearchEvent",
    "SearchMetrics",
    "SearchResult",
    "compute_heuristic",
    "min_travel_time_heuristic",
    "reconstruct_route",
    "solve_astar",
    "solve_dijkstra",
    "solve_exhaustive",
    "solve_greedy",
    "zero_heuristic",
]
