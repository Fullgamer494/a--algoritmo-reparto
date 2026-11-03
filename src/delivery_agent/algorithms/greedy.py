"""Estrategia voraz: Vecino factible más cercano (Nearest Feasible Neighbor).

Esta estrategia selecciona en cada paso el paquete factible cuyo destino tenga el menor
tiempo de desplazamiento inmediato desde la ubicación actual.

ADVERTENCIA METODOLÓGICA:
Esta heurística constructiva no realiza búsqueda retrospectiva (backtracking) ni
exploración en árbol. Por tanto:
1. NO garantiza optimalidad global en el costo objetivo C(pi) = T(pi) + alpha * D_p(pi).
2. Puede quedar atrapada en un callejón sin salida (estado sin transiciones factibles)
   incluso cuando el problema admite soluciones globales válidas.
"""

from __future__ import annotations

import time

from delivery_agent.algorithms.astar import SearchMetrics, SearchResult
from delivery_agent.core.cost import summarize_route_costs, validate_alpha
from delivery_agent.core.transitions import TransitionResult, apply_transition
from delivery_agent.domain.problem import DeliveryProblem
from delivery_agent.domain.state import DeliveryState


def solve_greedy(
    problem: DeliveryProblem,
    initial_state: DeliveryState | None = None,
    *,
    alpha: float | None = None,
) -> SearchResult:
    """Ejecuta la estrategia voraz del vecino factible más cercano.

    En cada paso:
    - Evalúa las transiciones factibles para los paquetes pendientes.
    - Selecciona el paquete con menor `travel_time` inmediato.
    - Desempata de forma determinista por el identificador del paquete `package_id`.
    - Avanza al estado sucesor sin retroceso.
    """
    start_clock = time.perf_counter()
    effective_alpha = validate_alpha(problem.alpha if alpha is None else alpha)
    current_state = problem.initial_state if initial_state is None else initial_state

    transitions: list[TransitionResult] = []
    states_expanded = 0
    states_generated = 0
    infeasible_transitions_discarded = 0
    max_candidates = 0

    while not problem.is_goal(current_state):
        valid_successors: list[TransitionResult] = []
        for pkg_id in current_state.sorted_pending_ids():
            step = apply_transition(
                problem,
                current_state,
                pkg_id,
                alpha=effective_alpha,
            )
            if step is None:
                infeasible_transitions_discarded += 1
            else:
                valid_successors.append(step)
                states_generated += 1

        if not valid_successors:
            # Callejón sin salida: no hay sucesores factibles desde este estado
            elapsed_seconds = time.perf_counter() - start_clock
            metrics = SearchMetrics(
                states_generated=states_generated,
                states_expanded=states_expanded,
                infeasible_transitions_discarded=infeasible_transitions_discarded,
                dominated_or_repeated_discarded=0,
                stale_queue_entries=0,
                reopenings=0,
                max_queue_size=max_candidates,
                execution_time_seconds=elapsed_seconds,
            )
            return SearchResult(
                solved=False,
                transitions=tuple(transitions),
                final_state=current_state,
                total_cost=None,
                cost_summary=None,
                metrics=metrics,
                algorithm="Greedy",
            )

        if len(valid_successors) > max_candidates:
            max_candidates = len(valid_successors)

        # Criterio voraz: menor tiempo de traslado inmediato, desempate por ID de paquete
        chosen_step = min(
            valid_successors,
            key=lambda step: (step.travel_time, step.package_id),
        )

        transitions.append(chosen_step)
        states_expanded += 1
        current_state = chosen_step.next_state

    # Se entregaron todos los paquetes
    elapsed_seconds = time.perf_counter() - start_clock
    summary = summarize_route_costs(transitions, alpha=effective_alpha)
    metrics = SearchMetrics(
        states_generated=states_generated,
        states_expanded=states_expanded,
        infeasible_transitions_discarded=infeasible_transitions_discarded,
        dominated_or_repeated_discarded=0,
        stale_queue_entries=0,
        reopenings=0,
        max_queue_size=max_candidates,
        execution_time_seconds=elapsed_seconds,
    )
    return SearchResult(
        solved=True,
        transitions=tuple(transitions),
        final_state=current_state,
        total_cost=summary.objective_cost,
        cost_summary=summary,
        metrics=metrics,
        algorithm="Greedy",
    )

