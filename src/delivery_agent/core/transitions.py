"""Motor de transiciones temporales y generación de estados sucesores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from delivery_agent.core.cost import evaluate_transition_cost
from delivery_agent.domain.location import LocationId
from delivery_agent.domain.package import Package
from delivery_agent.domain.problem import DeliveryProblem
from delivery_agent.domain.state import DeliveryState


@dataclass(frozen=True, slots=True)
class TransitionResult:
    """Resultado detallado e inmutable de aplicar una transición válida S -> S'."""

    next_state: DeliveryState
    package: Package
    package_id: str
    origin: LocationId
    destination: LocationId
    departure_time: int
    arrival_time: int
    travel_time: int
    wait_time: int
    service_start_time: int
    service_time: int
    completion_time: int
    elapsed_time: int
    window_start: int
    window_end: int
    promised_time: int
    tardiness: int
    priority: str
    priority_weight: float
    weighted_tardiness: float
    alpha: float
    penalty: float
    incremental_cost: float


def apply_transition(
    problem: DeliveryProblem,
    state: DeliveryState,
    package: Package | str,
    *,
    alpha: float | None = None,
) -> TransitionResult | None:
    """Intenta aplicar la transición de entregar `package` desde el estado `state`.

    Ecuaciones del modelo temporal (sección 6 del README):
    - Llegada: a_i = t + tau(l, destination_i)
    - Inicio del servicio: s_i = max(a_i, o_i)
    - Espera: e_i = max(0, o_i - a_i)
    - Finalización del servicio: f_i = s_i + q_i
    - Restricción obligatoria de ventana: f_i <= c_i
    - Estado sucesor: S' = (destination_i, f_i, U \\ {p_i})

    No modifica el estado original `state`.
    Si el traslado no está disponible (tau is None) o si el servicio termina después
    del cierre de la ventana (f_i > c_i), devuelve `None`.
    """
    pkg = problem.get_package(package) if isinstance(package, str) else package

    if pkg.id not in state.pending_packages:
        raise ValueError(
            f"El paquete '{pkg.id}' no se encuentra en el conjunto de paquetes pendientes del estado actual."
        )

    travel_time = problem.travel_time(state.location, pkg.destination)
    if travel_time is None:
        return None

    departure_time = state.current_time
    arrival_time = departure_time + travel_time
    service_start_time = max(arrival_time, pkg.window_start)
    wait_time = max(0, pkg.window_start - arrival_time)
    completion_time = service_start_time + pkg.service_time

    # Restricción obligatoria: el servicio completo debe finalizar en window_end o antes
    if completion_time > pkg.window_end:
        return None

    elapsed_time = completion_time - departure_time
    # Verificación de identidad temporal: delta_t == tau + espera + servicio
    assert elapsed_time == travel_time + wait_time + pkg.service_time

    effective_alpha = problem.alpha if alpha is None else alpha
    priority_weight = problem.priority_weight(pkg)
    cost_breakdown = evaluate_transition_cost(
        departure_time=departure_time,
        completion_time=completion_time,
        promised_time=pkg.promised_time,
        priority_weight=priority_weight,
        alpha=effective_alpha,
    )

    next_pending = state.pending_packages - frozenset((pkg.id,))
    next_state = DeliveryState(
        location=pkg.destination,
        current_time=completion_time,
        pending_packages=next_pending,
    )

    return TransitionResult(
        next_state=next_state,
        package=pkg,
        package_id=pkg.id,
        origin=state.location,
        destination=pkg.destination,
        departure_time=departure_time,
        arrival_time=arrival_time,
        travel_time=travel_time,
        wait_time=wait_time,
        service_start_time=service_start_time,
        service_time=pkg.service_time,
        completion_time=completion_time,
        elapsed_time=elapsed_time,
        window_start=pkg.window_start,
        window_end=pkg.window_end,
        promised_time=pkg.promised_time,
        tardiness=cost_breakdown.tardiness,
        priority=pkg.priority,
        priority_weight=priority_weight,
        weighted_tardiness=cost_breakdown.weighted_tardiness,
        alpha=cost_breakdown.alpha,
        penalty=cost_breakdown.penalty,
        incremental_cost=cost_breakdown.incremental_cost,
    )


def generate_successors(
    problem: DeliveryProblem,
    state: DeliveryState,
    *,
    alpha: float | None = None,
) -> tuple[TransitionResult, ...]:
    """Genera todas las transiciones factibles desde `state` en orden determinista de ID de paquete."""
    valid_transitions: list[TransitionResult] = []
    for pkg_id in state.sorted_pending_ids():
        result = apply_transition(problem, state, pkg_id, alpha=alpha)
        if result is not None:
            valid_transitions.append(result)
    return tuple(valid_transitions)


def simulate_route(
    problem: DeliveryProblem,
    package_sequence: Sequence[Package | str],
    *,
    alpha: float | None = None,
) -> tuple[TransitionResult, ...] | None:
    """Simula una secuencia de entregas desde el estado inicial del problema.

    Devuelve la tupla de `TransitionResult` si toda la secuencia es válida, o `None`
    si alguna entrega incumple su ventana de recepción o utiliza un traslado inexistente.
    """
    current_state = problem.initial_state
    steps: list[TransitionResult] = []

    for pkg in package_sequence:
        step = apply_transition(problem, current_state, pkg, alpha=alpha)
        if step is None:
            return None
        steps.append(step)
        current_state = step.next_state

    return tuple(steps)

