"""Módulo de métricas de calidad de ruta, rendimiento algorítmico y tiempos de ejecución."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import statistics
from typing import Any, Mapping, Sequence

from delivery_agent.algorithms.astar import SearchMetrics, SearchResult
from delivery_agent.algorithms.exhaustive import ExhaustiveResult
from delivery_agent.core.cost import RouteCostSummary, summarize_route_costs, validate_alpha
from delivery_agent.core.transitions import TransitionResult


@dataclass(frozen=True, slots=True)
class RouteQualityMetrics:
    """Métricas comunes de calidad de una ruta de entrega."""

    total_elapsed_time: int
    total_travel_time: int
    total_wait_time: int
    total_service_time: int
    total_tardiness: int
    weighted_tardiness: float
    delivery_count: int
    expected_packages: int
    on_promise_count: int
    on_promise_percentage: float
    window_compliance_count: int
    window_compliance_percentage: float
    alpha: float
    total_penalty: float
    objective_cost: float

    def to_dict(self) -> dict[str, Any]:
        """Serializa las métricas a un diccionario estándar."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExecutionTimingMetrics:
    """Estadísticas de tiempo de ejecución obtenidas tras múltiples repeticiones."""

    runs: int
    mean_seconds: float
    median_seconds: float
    min_seconds: float
    max_seconds: float
    stdev_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Serializa las métricas temporales a un diccionario estándar."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AlgorithmEvaluationReport:
    """Reporte integral de evaluación de un algoritmo sobre un escenario y valor de alpha."""

    scenario_name: str
    algorithm: str
    alpha: float
    solved: bool
    package_order: tuple[str, ...]
    quality: RouteQualityMetrics | None
    timing: ExecutionTimingMetrics
    search_metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serializa el reporte completo a un diccionario JSON-compatible."""
        return {
            "scenario_name": self.scenario_name,
            "algorithm": self.algorithm,
            "alpha": self.alpha,
            "solved": self.solved,
            "package_order": list(self.package_order),
            "quality": self.quality.to_dict() if self.quality else None,
            "timing": self.timing.to_dict(),
            "search_metrics": dict(self.search_metrics),
        }


def compute_route_quality_metrics(
    transitions: Sequence[TransitionResult],
    expected_packages: int,
    alpha: float,
) -> RouteQualityMetrics:
    """Calcula las métricas de calidad de ruta a partir de las transiciones ejecutadas.

    Conserva exactamente la formulación C(pi) = T(pi) + alpha * D_p(pi).
    """
    valid_alpha = validate_alpha(alpha)
    summary: RouteCostSummary = summarize_route_costs(transitions, alpha=valid_alpha)
    delivery_count = len(transitions)

    on_promise_count = sum(1 for step in transitions if step.tardiness == 0)
    window_compliance_count = sum(
        1
        for step in transitions
        if step.service_start_time >= step.window_start and step.completion_time <= step.window_end
    )

    if expected_packages > 0:
        on_promise_pct = (on_promise_count / expected_packages) * 100.0
        window_compliance_pct = (window_compliance_count / expected_packages) * 100.0
    else:
        on_promise_pct = 100.0
        window_compliance_pct = 100.0

    return RouteQualityMetrics(
        total_elapsed_time=summary.total_elapsed_time,
        total_travel_time=summary.total_travel_time,
        total_wait_time=summary.total_wait_time,
        total_service_time=summary.total_service_time,
        total_tardiness=summary.total_tardiness,
        weighted_tardiness=summary.weighted_tardiness,
        delivery_count=delivery_count,
        expected_packages=expected_packages,
        on_promise_count=on_promise_count,
        on_promise_percentage=round(on_promise_pct, 2),
        window_compliance_count=window_compliance_count,
        window_compliance_percentage=round(window_compliance_pct, 2),
        alpha=valid_alpha,
        total_penalty=summary.total_penalty,
        objective_cost=summary.objective_cost,
    )


def compute_timing_statistics(durations_seconds: Sequence[float]) -> ExecutionTimingMetrics:
    """Calcula media, mediana, mínimo, máximo y desviación estándar de una muestra de tiempos."""
    if not durations_seconds:
        raise ValueError("Se requiere al menos una medición de tiempo.")

    n = len(durations_seconds)
    mean_val = statistics.mean(durations_seconds)
    median_val = statistics.median(durations_seconds)
    min_val = min(durations_seconds)
    max_val = max(durations_seconds)
    stdev_val = statistics.stdev(durations_seconds) if n > 1 else 0.0

    return ExecutionTimingMetrics(
        runs=n,
        mean_seconds=mean_val,
        median_seconds=median_val,
        min_seconds=min_val,
        max_seconds=max_val,
        stdev_seconds=stdev_val,
    )


def extract_specific_search_metrics(
    result: SearchResult | ExhaustiveResult,
) -> dict[str, Any]:
    """Extrae las métricas de exploración específicas según la naturaleza del algoritmo."""
    if isinstance(result, ExhaustiveResult):
        return {
            "permutations_evaluated": result.total_permutations_evaluated,
            "feasible_permutations": result.feasible_permutations,
            "infeasible_permutations": result.infeasible_permutations,
        }

    # SearchResult (A*, Dijkstra o Greedy)
    if result.algorithm == "Greedy":
        completed = result.solved
        reason = "Ruta completada" if completed else "Sin sucesores factibles (callejón sin salida)"
        return {
            "steps_executed": len(result.transitions),
            "feasible_transitions_evaluated": result.metrics.states_generated,
            "infeasible_transitions_discarded": result.metrics.infeasible_transitions_discarded,
            "completed_route": completed,
            "termination_reason": reason,
        }

    # A* o Dijkstra
    return {
        "states_generated": result.metrics.states_generated,
        "states_expanded": result.metrics.states_expanded,
        "infeasible_transitions_discarded": result.metrics.infeasible_transitions_discarded,
        "dominated_or_repeated_discarded": result.metrics.dominated_or_repeated_discarded,
        "stale_queue_entries": result.metrics.stale_queue_entries,
        "reopenings": result.metrics.reopenings,
        "max_queue_size": result.metrics.max_queue_size,
    }

