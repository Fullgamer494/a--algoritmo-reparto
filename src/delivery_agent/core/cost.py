"""Funciones de evaluación de retrasos, penalizaciones y costos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from delivery_agent.core.transitions import TransitionResult


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """Desglose del costo incremental de una transición."""

    elapsed_time: int
    tardiness: int
    priority_weight: float
    weighted_tardiness: float
    alpha: float
    penalty: float
    incremental_cost: float


@dataclass(frozen=True, slots=True)
class RouteCostSummary:
    """Resumen acumulado de tiempos y costos de una secuencia de transiciones."""

    total_elapsed_time: int
    total_travel_time: int
    total_wait_time: int
    total_service_time: int
    total_tardiness: int
    weighted_tardiness: float
    alpha: float
    total_penalty: float
    objective_cost: float


def validate_alpha(alpha: float) -> float:
    """Valida que el parámetro alpha sea un número no negativo."""
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError("El parámetro alpha debe ser numérico.")
    if alpha < 0:
        raise ValueError(f"El parámetro alpha no puede ser negativo ({alpha}).")
    return float(alpha)


def compute_tardiness(completion_time: int, promised_time: int) -> int:
    """Calcula el retraso respecto a la hora prometida:

    d_i = max(0, f_i - r_i)
    """
    if isinstance(completion_time, bool) or not isinstance(completion_time, int):
        raise ValueError("completion_time debe ser un entero en minutos.")
    if isinstance(promised_time, bool) or not isinstance(promised_time, int):
        raise ValueError("promised_time debe ser un entero en minutos.")
    return max(0, completion_time - promised_time)


def compute_weighted_tardiness(tardiness: int, priority_weight: float) -> float:
    """Calcula el retraso ponderado por el peso de prioridad: w_i * d_i."""
    if isinstance(tardiness, bool) or not isinstance(tardiness, int) or tardiness < 0:
        raise ValueError("El retraso (tardiness) debe ser un entero no negativo.")
    if (
        isinstance(priority_weight, bool)
        or not isinstance(priority_weight, (int, float))
        or priority_weight <= 0
    ):
        raise ValueError("El peso de prioridad (priority_weight) debe ser un número positivo.")
    return float(priority_weight) * float(tardiness)


def compute_penalty(tardiness: int, priority_weight: float, alpha: float) -> float:
    """Calcula la penalización ponderada:

    penalizacion = alpha * peso_prioridad * retraso
    """
    valid_alpha = validate_alpha(alpha)
    weighted = compute_weighted_tardiness(tardiness, priority_weight)
    return valid_alpha * weighted


def compute_incremental_cost(elapsed_time: int, penalty: float) -> float:
    """Calcula el costo incremental de una transición:

    costo_incremental = tiempo_transcurrido + penalizacion
    """
    if isinstance(elapsed_time, bool) or not isinstance(elapsed_time, int) or elapsed_time < 0:
        raise ValueError("El tiempo transcurrido (elapsed_time) debe ser un entero no negativo.")
    if isinstance(penalty, bool) or not isinstance(penalty, (int, float)) or penalty < 0:
        raise ValueError("La penalización (penalty) debe ser un número no negativo.")
    return float(elapsed_time) + float(penalty)


def evaluate_transition_cost(
    *,
    departure_time: int,
    completion_time: int,
    promised_time: int,
    priority_weight: float,
    alpha: float,
) -> CostBreakdown:
    """Evalúa todas las componentes de costo de una transición desde `departure_time` hasta `completion_time`."""
    if isinstance(departure_time, bool) or not isinstance(departure_time, int):
        raise ValueError("departure_time debe ser un entero en minutos.")
    if isinstance(completion_time, bool) or not isinstance(completion_time, int):
        raise ValueError("completion_time debe ser un entero en minutos.")
    if completion_time < departure_time:
        raise ValueError("La hora de finalización no puede ser anterior a la hora de salida.")

    valid_alpha = validate_alpha(alpha)
    elapsed_time = completion_time - departure_time
    tardiness = compute_tardiness(completion_time, promised_time)
    weighted_tardiness = compute_weighted_tardiness(tardiness, priority_weight)
    penalty = compute_penalty(tardiness, priority_weight, valid_alpha)
    incremental_cost = compute_incremental_cost(elapsed_time, penalty)

    return CostBreakdown(
        elapsed_time=elapsed_time,
        tardiness=tardiness,
        priority_weight=float(priority_weight),
        weighted_tardiness=weighted_tardiness,
        alpha=valid_alpha,
        penalty=penalty,
        incremental_cost=incremental_cost,
    )


def summarize_route_costs(
    transitions: Sequence[TransitionResult],
    alpha: float,
) -> RouteCostSummary:
    """Calcula el costo objetivo acumulado y el desglose temporal de una ruta:

    T(pi) = T_desplazamiento + T_espera + T_servicio
    D_p(pi) = sum(w_i * max(0, f_i - r_i))
    C(pi) = T(pi) + alpha * D_p(pi)
    """
    valid_alpha = validate_alpha(alpha)
    total_travel_time = sum(step.travel_time for step in transitions)
    total_wait_time = sum(step.wait_time for step in transitions)
    total_service_time = sum(step.service_time for step in transitions)
    total_elapsed_time = sum(step.elapsed_time for step in transitions)
    total_tardiness = sum(step.tardiness for step in transitions)
    weighted_tardiness = sum(step.weighted_tardiness for step in transitions)
    total_penalty = valid_alpha * weighted_tardiness
    objective_cost = float(total_elapsed_time) + total_penalty

    return RouteCostSummary(
        total_elapsed_time=total_elapsed_time,
        total_travel_time=total_travel_time,
        total_wait_time=total_wait_time,
        total_service_time=total_service_time,
        total_tardiness=total_tardiness,
        weighted_tardiness=weighted_tardiness,
        alpha=valid_alpha,
        total_penalty=total_penalty,
        objective_cost=objective_cost,
    )

