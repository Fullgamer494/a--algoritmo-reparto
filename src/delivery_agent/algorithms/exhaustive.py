"""Algoritmo de enumeración exhaustiva de permutaciones para validación de optimalidad.

Genera y evalúa todas las secuencias posibles de entrega mediante `simulate_route`.
Sirve como oráculo de referencia exacto para escenarios pequeños (5 a 8 paquetes).
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import time
from typing import Sequence

from delivery_agent.algorithms.astar import SearchMetrics, SearchResult
from delivery_agent.core.cost import RouteCostSummary, summarize_route_costs, validate_alpha
from delivery_agent.core.transitions import TransitionResult, simulate_route
from delivery_agent.domain.problem import DeliveryProblem


@dataclass(frozen=True, slots=True)
class ExhaustiveResult:
    """Resultado detallado de la enumeración exhaustiva de secuencias."""

    solved: bool
    best_sequence: tuple[str, ...]
    optimal_cost: float | None
    transitions: tuple[TransitionResult, ...]
    cost_summary: RouteCostSummary | None
    total_permutations_evaluated: int
    feasible_permutations: int
    infeasible_permutations: int
    execution_time_seconds: float
    algorithm: str = "Exhaustive"

    @property
    def is_feasible(self) -> bool:
        """Indica si existe al menos una permutación factible."""
        return self.solved

    @property
    def package_order(self) -> tuple[str, ...]:
        """Orden de entrega de los paquetes en la mejor secuencia."""
        return self.best_sequence

    def to_search_result(self) -> SearchResult:
        """Convierte el resultado a la estructura común `SearchResult`."""
        metrics = SearchMetrics(
            states_generated=self.total_permutations_evaluated,
            states_expanded=self.feasible_permutations,
            infeasible_transitions_discarded=self.infeasible_permutations,
            dominated_or_repeated_discarded=0,
            stale_queue_entries=0,
            reopenings=0,
            max_queue_size=1,
            execution_time_seconds=self.execution_time_seconds,
        )
        final_state = self.transitions[-1].next_state if self.transitions else None
        return SearchResult(
            solved=self.solved,
            transitions=self.transitions,
            final_state=final_state,
            total_cost=self.optimal_cost,
            cost_summary=self.cost_summary,
            metrics=metrics,
            algorithm=self.algorithm,
        )


def solve_exhaustive(
    problem: DeliveryProblem,
    *,
    package_subset: Sequence[str] | None = None,
    alpha: float | None = None,
) -> ExhaustiveResult:
    """Evalúa exhaustivamente todas las permutaciones de paquetes para hallar la ruta óptima.

    Desempate determinista: en caso de igual costo objetivo, selecciona la tupla
    de IDs lexicográficamente menor.
    """
    start_clock = time.perf_counter()
    effective_alpha = validate_alpha(problem.alpha if alpha is None else alpha)

    if package_subset is None:
        target_package_ids = sorted(p.id for p in problem.packages)
    else:
        target_package_ids = sorted(package_subset)

    # Caso especial: escenario sin paquetes
    if not target_package_ids:
        elapsed_seconds = time.perf_counter() - start_clock
        summary = RouteCostSummary(
            total_elapsed_time=0,
            total_travel_time=0,
            total_wait_time=0,
            total_service_time=0,
            total_tardiness=0,
            weighted_tardiness=0.0,
            alpha=effective_alpha,
            total_penalty=0.0,
            objective_cost=0.0,
        )
        return ExhaustiveResult(
            solved=True,
            best_sequence=(),
            optimal_cost=0.0,
            transitions=(),
            cost_summary=summary,
            total_permutations_evaluated=1,
            feasible_permutations=1,
            infeasible_permutations=0,
            execution_time_seconds=elapsed_seconds,
        )

    total_permutations = 0
    feasible_count = 0
    infeasible_count = 0

    best_cost: float | None = None
    best_seq: tuple[str, ...] = ()
    best_steps: tuple[TransitionResult, ...] = ()
    best_summary: RouteCostSummary | None = None

    for perm in itertools.permutations(target_package_ids):
        total_permutations += 1
        steps = simulate_route(problem, perm, alpha=effective_alpha)
        if steps is None:
            infeasible_count += 1
            continue

        feasible_count += 1
        summary = summarize_route_costs(steps, alpha=effective_alpha)
        cost = summary.objective_cost

        # Desempate determinista: menor costo, luego menor tupla lexicográfica
        is_better = False
        if best_cost is None:
            is_better = True
        elif cost < best_cost - 1e-9:
            is_better = True
        elif abs(cost - best_cost) <= 1e-9 and perm < best_seq:
            is_better = True

        if is_better:
            best_cost = cost
            best_seq = perm
            best_steps = steps
            best_summary = summary

    elapsed_seconds = time.perf_counter() - start_clock
    solved = best_cost is not None

    return ExhaustiveResult(
        solved=solved,
        best_sequence=best_seq if solved else (),
        optimal_cost=best_cost if solved else None,
        transitions=best_steps if solved else (),
        cost_summary=best_summary if solved else None,
        total_permutations_evaluated=total_permutations,
        feasible_permutations=feasible_count,
        infeasible_permutations=infeasible_count,
        execution_time_seconds=elapsed_seconds,
    )

