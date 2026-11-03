"""Implementación del algoritmo A* para la planificación de rutas de reparto."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
import time
from typing import Mapping

from delivery_agent.core.cost import (
    RouteCostSummary,
    summarize_route_costs,
    validate_alpha,
)
from delivery_agent.core.heuristics import HeuristicFn, min_travel_time_heuristic
from delivery_agent.core.transitions import TransitionResult, apply_transition
from delivery_agent.domain.problem import DeliveryProblem
from delivery_agent.domain.state import DeliveryState


@dataclass(order=True, frozen=True, slots=True)
class _PriorityQueueEntry:
    """Entrada ordenada de la frontera OPEN en A*.

    Se ordena por:
    1. ``f`` = g(n) + h(n) (menor prioridad primero)
    2. ``g`` = costo acumulado conocido
    3. ``state`` = orden total determinista de DeliveryState
    4. ``entry_id`` = contador secuencial de inserción
    """

    f: float
    g: float
    state: DeliveryState
    entry_id: int


@dataclass(frozen=True, slots=True)
class SearchEvent:
    """Registro detallado de un evento de exploración durante la búsqueda."""

    expansion_index: int
    event_type: str  # "EXPAND", "GOAL", "STALE", "DISCARD_INFEASIBLE", "DISCARD_DOMINATED", "PUSH_SUCCESSOR"
    state: DeliveryState
    location: str
    current_time: int
    pending_packages: frozenset[str]
    g: float
    h: float
    f: float
    queue_size: int
    package_id: str | None = None
    next_state: DeliveryState | None = None
    incremental_cost: float | None = None
    discard_reason: str | None = None


@dataclass(frozen=True, slots=True)
class SearchMetrics:
    """Métricas internas de rendimiento y exploración de la búsqueda."""

    states_generated: int
    states_expanded: int
    infeasible_transitions_discarded: int
    dominated_or_repeated_discarded: int
    stale_queue_entries: int
    reopenings: int
    max_queue_size: int
    execution_time_seconds: float


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Resultado estructurado de la ejecución de un algoritmo de búsqueda."""

    solved: bool
    transitions: tuple[TransitionResult, ...]
    final_state: DeliveryState | None
    total_cost: float | None
    cost_summary: RouteCostSummary | None
    metrics: SearchMetrics
    algorithm: str = "A*"
    events: tuple[SearchEvent, ...] | None = None

    @property
    def is_feasible(self) -> bool:
        """Indica si se encontró una ruta factible que entrega todos los paquetes."""
        return self.solved

    @property
    def package_order(self) -> tuple[str, ...]:
        """Orden de entrega de los paquetes en la ruta solución."""
        return tuple(step.package_id for step in self.transitions)

    @property
    def arrival_times(self) -> tuple[int, ...]:
        """Horarios de llegada a cada destino en minutos."""
        return tuple(step.arrival_time for step in self.transitions)

    @property
    def wait_times(self) -> tuple[int, ...]:
        """Tiempos de espera en cada entrega en minutos."""
        return tuple(step.wait_time for step in self.transitions)

    @property
    def service_start_times(self) -> tuple[int, ...]:
        """Horarios de inicio de servicio de cada entrega en minutos."""
        return tuple(step.service_start_time for step in self.transitions)

    @property
    def completion_times(self) -> tuple[int, ...]:
        """Horarios de finalización de servicio de cada entrega en minutos."""
        return tuple(step.completion_time for step in self.transitions)

    @property
    def tardiness_values(self) -> tuple[int, ...]:
        """Retrasos respecto a la hora prometida de cada entrega en minutos."""
        return tuple(step.tardiness for step in self.transitions)

    @property
    def penalties(self) -> tuple[float, ...]:
        """Penalizaciones ponderadas de cada entrega."""
        return tuple(step.penalty for step in self.transitions)

    @property
    def incremental_costs(self) -> tuple[float, ...]:
        """Costos incrementales de cada transición."""
        return tuple(step.incremental_cost for step in self.transitions)


def reconstruct_route(
    parent_map: Mapping[DeliveryState, tuple[DeliveryState, TransitionResult]],
    goal_state: DeliveryState,
    start_state: DeliveryState,
) -> tuple[TransitionResult, ...]:
    """Reconstruye la secuencia ordenada de `TransitionResult` desde `start_state` hasta `goal_state`."""
    steps: list[TransitionResult] = []
    current = goal_state

    while current != start_state:
        if current not in parent_map:
            raise ValueError(
                f"No fue posible reconstruir la ruta: el estado {current} carece de padre registrado."
            )
        parent_state, transition = parent_map[current]
        steps.append(transition)
        current = parent_state

    steps.reverse()
    return tuple(steps)


def solve_astar(
    problem: DeliveryProblem,
    initial_state: DeliveryState | None = None,
    *,
    heuristic_fn: HeuristicFn | None = None,
    alpha: float | None = None,
    algorithm_name: str = "A*",
    track_events: bool = False,
) -> SearchResult:
    """Ejecuta el algoritmo A* para encontrar una ruta factible de costo mínimo.

    Características clave:
    - Utiliza una cola de prioridad mínima (`heapq`) ordenada por f(n) = g(n) + h(n).
    - Comprueba la condición de objetivo al extraer el estado de la cola, no al generarlo.
    - Gestiona estados repetidos mediante `best_g[state]` sin lista CLOSED permanente.
    - Permite la reapertura de estados previamente expandidos si se descubre un menor `g`.
    - Ignora entradas desactualizadas en la cola de prioridad.
    - Opcionalmente registra la traza cronológica de eventos (`track_events=True`) para visualización.
    """
    start_clock = time.perf_counter()

    start_state = problem.initial_state if initial_state is None else initial_state
    heuristic = min_travel_time_heuristic if heuristic_fn is None else heuristic_fn
    effective_alpha = validate_alpha(problem.alpha if alpha is None else alpha)

    open_heap: list[_PriorityQueueEntry] = []
    best_g: dict[DeliveryState, float] = {start_state: 0.0}
    parent_map: dict[DeliveryState, tuple[DeliveryState, TransitionResult]] = {}
    expanded_states: set[DeliveryState] = set()
    events_log: list[SearchEvent] | None = [] if track_events else None

    states_generated = 0
    states_expanded = 0
    infeasible_transitions_discarded = 0
    dominated_or_repeated_discarded = 0
    stale_queue_entries = 0
    reopenings = 0
    entry_counter = 0

    initial_h = heuristic(problem, start_state)
    heapq.heappush(
        open_heap,
        _PriorityQueueEntry(
            f=initial_h,
            g=0.0,
            state=start_state,
            entry_id=entry_counter,
        ),
    )
    max_queue_size = len(open_heap)

    while open_heap:
        current_entry = heapq.heappop(open_heap)
        current_state = current_entry.state
        current_g = current_entry.g
        current_f = current_entry.f

        # 1. Ignorar entradas obsoletas cuyo g es mayor que el mejor costo conocido
        known_best_g = best_g.get(current_state, math.inf)
        if current_g > known_best_g:
            stale_queue_entries += 1
            if events_log is not None:
                events_log.append(
                    SearchEvent(
                        expansion_index=states_expanded,
                        event_type="STALE",
                        state=current_state,
                        location=current_state.location,
                        current_time=current_state.current_time,
                        pending_packages=current_state.pending_packages,
                        g=current_g,
                        h=current_f - current_g,
                        f=current_f,
                        queue_size=len(open_heap),
                        discard_reason="g_greater_than_best_g",
                    )
                )
            continue

        # 2. Comprobar estado objetivo al extraer de la cola de prioridad
        if problem.is_goal(current_state):
            if events_log is not None:
                events_log.append(
                    SearchEvent(
                        expansion_index=states_expanded,
                        event_type="GOAL",
                        state=current_state,
                        location=current_state.location,
                        current_time=current_state.current_time,
                        pending_packages=current_state.pending_packages,
                        g=current_g,
                        h=0.0,
                        f=current_g,
                        queue_size=len(open_heap),
                    )
                )
            elapsed_seconds = time.perf_counter() - start_clock
            transitions = reconstruct_route(parent_map, current_state, start_state)
            summary = summarize_route_costs(transitions, alpha=effective_alpha)
            metrics = SearchMetrics(
                states_generated=states_generated,
                states_expanded=states_expanded,
                infeasible_transitions_discarded=infeasible_transitions_discarded,
                dominated_or_repeated_discarded=dominated_or_repeated_discarded,
                stale_queue_entries=stale_queue_entries,
                reopenings=reopenings,
                max_queue_size=max_queue_size,
                execution_time_seconds=elapsed_seconds,
            )
            return SearchResult(
                solved=True,
                transitions=transitions,
                final_state=current_state,
                total_cost=summary.objective_cost,
                cost_summary=summary,
                metrics=metrics,
                algorithm=algorithm_name,
                events=tuple(events_log) if events_log is not None else None,
            )

        # 3. Expandir estado actual generando sucesores mediante core/transitions.py
        states_expanded += 1
        expanded_states.add(current_state)
        curr_h = current_f - current_g

        if events_log is not None:
            events_log.append(
                SearchEvent(
                    expansion_index=states_expanded,
                    event_type="EXPAND",
                    state=current_state,
                    location=current_state.location,
                    current_time=current_state.current_time,
                    pending_packages=current_state.pending_packages,
                    g=current_g,
                    h=curr_h,
                    f=current_f,
                    queue_size=len(open_heap),
                )
            )

        for pkg_id in current_state.sorted_pending_ids():
            transition = apply_transition(
                problem,
                current_state,
                pkg_id,
                alpha=effective_alpha,
            )
            if transition is None:
                infeasible_transitions_discarded += 1
                if events_log is not None:
                    events_log.append(
                        SearchEvent(
                            expansion_index=states_expanded,
                            event_type="DISCARD_INFEASIBLE",
                            state=current_state,
                            location=current_state.location,
                            current_time=current_state.current_time,
                            pending_packages=current_state.pending_packages,
                            g=current_g,
                            h=curr_h,
                            f=current_f,
                            queue_size=len(open_heap),
                            package_id=pkg_id,
                            discard_reason="window_violated_or_unreachable",
                        )
                    )
                continue

            states_generated += 1
            next_state = transition.next_state
            tentative_g = current_g + transition.incremental_cost

            previous_best_g = best_g.get(next_state, math.inf)
            if tentative_g < previous_best_g:
                if next_state in expanded_states:
                    reopenings += 1

                best_g[next_state] = tentative_g
                parent_map[next_state] = (current_state, transition)
                h_next = heuristic(problem, next_state)
                f_next = tentative_g + h_next
                entry_counter += 1
                heapq.heappush(
                    open_heap,
                    _PriorityQueueEntry(
                        f=f_next,
                        g=tentative_g,
                        state=next_state,
                        entry_id=entry_counter,
                    ),
                )
                if len(open_heap) > max_queue_size:
                    max_queue_size = len(open_heap)

                if events_log is not None:
                    events_log.append(
                        SearchEvent(
                            expansion_index=states_expanded,
                            event_type="PUSH_SUCCESSOR",
                            state=current_state,
                            location=current_state.location,
                            current_time=current_state.current_time,
                            pending_packages=current_state.pending_packages,
                            g=tentative_g,
                            h=h_next,
                            f=f_next,
                            queue_size=len(open_heap),
                            package_id=pkg_id,
                            next_state=next_state,
                            incremental_cost=transition.incremental_cost,
                        )
                    )
            else:
                dominated_or_repeated_discarded += 1
                if events_log is not None:
                    events_log.append(
                        SearchEvent(
                            expansion_index=states_expanded,
                            event_type="DISCARD_DOMINATED",
                            state=current_state,
                            location=current_state.location,
                            current_time=current_state.current_time,
                            pending_packages=current_state.pending_packages,
                            g=tentative_g,
                            h=heuristic(problem, next_state),
                            f=tentative_g + heuristic(problem, next_state),
                            queue_size=len(open_heap),
                            package_id=pkg_id,
                            next_state=next_state,
                            incremental_cost=transition.incremental_cost,
                            discard_reason="tentative_g_not_better",
                        )
                    )

    # Cola vacía sin haber extraído un estado objetivo: no existe ruta factible
    elapsed_seconds = time.perf_counter() - start_clock
    metrics = SearchMetrics(
        states_generated=states_generated,
        states_expanded=states_expanded,
        infeasible_transitions_discarded=infeasible_transitions_discarded,
        dominated_or_repeated_discarded=dominated_or_repeated_discarded,
        stale_queue_entries=stale_queue_entries,
        reopenings=reopenings,
        max_queue_size=max_queue_size,
        execution_time_seconds=elapsed_seconds,
    )
    return SearchResult(
        solved=False,
        transitions=(),
        final_state=None,
        total_cost=None,
        cost_summary=None,
        metrics=metrics,
        algorithm=algorithm_name,
        events=tuple(events_log) if events_log is not None else None,
    )

