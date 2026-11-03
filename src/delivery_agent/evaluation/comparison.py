"""Módulo de comparación experimental y análisis de optimalidad y sensibilidad."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Sequence

from delivery_agent.algorithms.astar import SearchResult, solve_astar
from delivery_agent.algorithms.dijkstra import solve_dijkstra
from delivery_agent.algorithms.exhaustive import ExhaustiveResult, solve_exhaustive
from delivery_agent.algorithms.greedy import solve_greedy
from delivery_agent.core.cost import validate_alpha
from delivery_agent.domain.problem import DeliveryProblem

NUMERICAL_TOLERANCE = 1e-6


@dataclass(frozen=True, slots=True)
class ScenarioComparisonResult:
    """Resultado de la comparación exhaustiva entre algoritmos para un escenario y un alpha."""

    scenario_name: str
    alpha: float
    is_feasible: bool
    optimal_cost: float | None

    astar_solved: bool
    astar_cost: float | None
    astar_states_expanded: int

    dijkstra_solved: bool
    dijkstra_cost: float | None
    dijkstra_states_expanded: int

    exhaustive_solved: bool
    exhaustive_cost: float | None
    exhaustive_permutations: int

    greedy_solved: bool
    greedy_cost: float | None
    greedy_failed: bool
    greedy_suboptimal: bool
    greedy_absolute_gap: float | None
    greedy_percentage_gap: float | None

    optimality_verified: bool
    package_count: int

    def to_dict(self) -> dict[str, Any]:
        """Serializa la comparación a diccionario."""
        return asdict(self)


def compare_scenario_algorithms(
    problem: DeliveryProblem,
    alpha: float,
    *,
    run_exhaustive: bool = True,
) -> ScenarioComparisonResult:
    """Ejecuta y compara A*, Dijkstra, Greedy y Enumeración Exhaustiva bajo el mismo alpha y datos.

    Reglas de comparación:
    - Comprueba Costo_A* == Costo_Dijkstra == Costo_Exhaustivo dentro de NUMERICAL_TOLERANCE.
    - Si un escenario es inviable, comprueba que A*, Dijkstra y Exhaustiva coincidan.
    - Si Greedy no encuentra solución pero Exhaustiva sí, se registra como `greedy_failed=True`.
    - Si Greedy encuentra solución pero con mayor costo, se calcula la brecha absoluta y porcentual.
    """
    valid_alpha = validate_alpha(alpha)

    # 1. Ejecución de A*
    res_astar: SearchResult = solve_astar(problem, alpha=valid_alpha)

    # 2. Ejecución de Dijkstra
    res_dijkstra: SearchResult = solve_dijkstra(problem, alpha=valid_alpha)

    # 3. Ejecución de Greedy
    res_greedy: SearchResult = solve_greedy(problem, alpha=valid_alpha)

    # 4. Ejecución de Enumeración Exhaustiva (oráculo de validación)
    if run_exhaustive:
        res_exhaustive: ExhaustiveResult = solve_exhaustive(problem, alpha=valid_alpha)
        exh_solved = res_exhaustive.solved
        exh_cost = res_exhaustive.optimal_cost
        exh_perms = res_exhaustive.total_permutations_evaluated
    else:
        exh_solved = res_astar.solved
        exh_cost = res_astar.total_cost
        exh_perms = 0

    # Determinación de factibilidad y costo óptimo
    is_feasible = exh_solved
    optimal_cost = exh_cost

    # Verificación estricta de optimalidad
    optimality_verified = False
    if is_feasible:
        assert optimal_cost is not None
        astar_matches = res_astar.solved and res_astar.total_cost is not None and (
            abs(res_astar.total_cost - optimal_cost) <= NUMERICAL_TOLERANCE
        )
        dijkstra_matches = res_dijkstra.solved and res_dijkstra.total_cost is not None and (
            abs(res_dijkstra.total_cost - optimal_cost) <= NUMERICAL_TOLERANCE
        )
        optimality_verified = astar_matches and dijkstra_matches
    else:
        # Inviable: todos deben coincidir en no resuelto
        optimality_verified = (not res_astar.solved) and (not res_dijkstra.solved)

    # Análisis específico para Greedy
    greedy_solved = res_greedy.solved
    greedy_cost = res_greedy.total_cost
    greedy_failed = is_feasible and (not greedy_solved)
    greedy_suboptimal = False
    greedy_abs_gap: float | None = None
    greedy_pct_gap: float | None = None

    if is_feasible and greedy_solved and greedy_cost is not None and optimal_cost is not None:
        greedy_abs_gap = max(0.0, greedy_cost - optimal_cost)
        if greedy_abs_gap > NUMERICAL_TOLERANCE:
            greedy_suboptimal = True
            if optimal_cost > 0:
                greedy_pct_gap = (greedy_abs_gap / optimal_cost) * 100.0
            else:
                greedy_pct_gap = 0.0
        else:
            greedy_abs_gap = 0.0
            greedy_pct_gap = 0.0

    return ScenarioComparisonResult(
        scenario_name=problem.name,
        alpha=valid_alpha,
        is_feasible=is_feasible,
        optimal_cost=optimal_cost,
        astar_solved=res_astar.solved,
        astar_cost=res_astar.total_cost,
        astar_states_expanded=res_astar.metrics.states_expanded,
        dijkstra_solved=res_dijkstra.solved,
        dijkstra_cost=res_dijkstra.total_cost,
        dijkstra_states_expanded=res_dijkstra.metrics.states_expanded,
        exhaustive_solved=exh_solved,
        exhaustive_cost=exh_cost,
        exhaustive_permutations=exh_perms,
        greedy_solved=greedy_solved,
        greedy_cost=greedy_cost,
        greedy_failed=greedy_failed,
        greedy_suboptimal=greedy_suboptimal,
        greedy_absolute_gap=greedy_abs_gap,
        greedy_percentage_gap=greedy_pct_gap,
        optimality_verified=optimality_verified,
        package_count=len(problem.packages),
    )


def run_alpha_sensitivity_analysis(
    problem: DeliveryProblem,
    alphas: Sequence[float] = (0.0, 0.5, 1.0, 2.0),
    *,
    run_exhaustive: bool = True,
) -> tuple[ScenarioComparisonResult, ...]:
    """Evalúa el comportamiento de los algoritmos para un rango de valores de alpha."""
    results: list[ScenarioComparisonResult] = []
    for a in alphas:
        comp = compare_scenario_algorithms(
            problem,
            alpha=a,
            run_exhaustive=run_exhaustive,
        )
        results.append(comp)
    return tuple(results)

