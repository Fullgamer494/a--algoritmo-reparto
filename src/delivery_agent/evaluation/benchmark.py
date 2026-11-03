"""Módulo de benchmarking experimental, medición temporal de precisión y persistencia."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time
from typing import Any, Callable, Mapping, Sequence

from delivery_agent.algorithms.astar import SearchResult, solve_astar
from delivery_agent.algorithms.dijkstra import solve_dijkstra
from delivery_agent.algorithms.exhaustive import ExhaustiveResult, solve_exhaustive
from delivery_agent.algorithms.greedy import solve_greedy
from delivery_agent.core.cost import validate_alpha
from delivery_agent.domain.problem import DeliveryProblem
from delivery_agent.evaluation.metrics import (
    AlgorithmEvaluationReport,
    ExecutionTimingMetrics,
    compute_route_quality_metrics,
    compute_timing_statistics,
    extract_specific_search_metrics,
)
from delivery_agent.scenarios.loader import load_problem_from_json

ALGORITHM_DISPATCH: dict[str, Callable[[DeliveryProblem, float], SearchResult | ExhaustiveResult]] = {
    "A*": lambda p, a: solve_astar(p, alpha=a),
    "Dijkstra": lambda p, a: solve_dijkstra(p, alpha=a),
    "Greedy": lambda p, a: solve_greedy(p, alpha=a),
    "Exhaustive": lambda p, a: solve_exhaustive(p, alpha=a),
}


@dataclass(frozen=True, slots=True)
class BenchmarkSuiteResult:
    """Resultado agregado de una suite de benchmarking."""

    experiment_id: str
    reports: tuple[AlgorithmEvaluationReport, ...]
    output_dir: Path | None
    csv_path: Path | None
    json_path: Path | None
    metadata_path: Path | None


def measure_algorithm_execution(
    algorithm_name: str,
    problem: DeliveryProblem,
    alpha: float,
    *,
    repetitions: int = 10,
) -> tuple[SearchResult | ExhaustiveResult, ExecutionTimingMetrics]:
    """Mide con alta precisión el tiempo de ejecución de un algoritmo sin sesgos de I/O.

    Procedimiento:
    1. Ejecuta una corrida de calentamiento (warm-up) descartando su medición.
    2. Ejecuta `repetitions` repeticiones cronometradas con `time.perf_counter_ns()`.
    3. Devuelve el último resultado obtenido y las estadísticas temporales agregadas.
    """
    if algorithm_name not in ALGORITHM_DISPATCH:
        raise ValueError(
            f"Algoritmo desconocido: '{algorithm_name}'. Opciones válidas: {list(ALGORITHM_DISPATCH.keys())}"
        )

    runner = ALGORITHM_DISPATCH[algorithm_name]
    valid_alpha = validate_alpha(alpha)

    # 1. Warm-up run (calentamiento)
    sample_result = runner(problem, valid_alpha)

    # 2. Medición de repeticiones
    durations_seconds: list[float] = []
    for _ in range(max(1, repetitions)):
        t_start = time.perf_counter_ns()
        sample_result = runner(problem, valid_alpha)
        t_elapsed_ns = time.perf_counter_ns() - t_start
        durations_seconds.append(t_elapsed_ns / 1_000_000_000.0)

    timing_metrics = compute_timing_statistics(durations_seconds)
    return sample_result, timing_metrics


def evaluate_algorithm_on_scenario(
    algorithm_name: str,
    problem: DeliveryProblem,
    alpha: float,
    *,
    repetitions: int = 10,
) -> AlgorithmEvaluationReport:
    """Ejecuta y genera el reporte completo de evaluación para un algoritmo, escenario y alpha."""
    result, timing = measure_algorithm_execution(
        algorithm_name,
        problem,
        alpha=alpha,
        repetitions=repetitions,
    )

    quality = None
    if result.solved:
        quality = compute_route_quality_metrics(
            result.transitions,
            expected_packages=len(problem.packages),
            alpha=alpha,
        )

    search_metrics = extract_specific_search_metrics(result)
    pkg_order = tuple(step.package_id for step in result.transitions) if result.solved else ()

    return AlgorithmEvaluationReport(
        scenario_name=problem.name,
        algorithm=algorithm_name,
        alpha=validate_alpha(alpha),
        solved=result.solved,
        package_order=pkg_order,
        quality=quality,
        timing=timing,
        search_metrics=search_metrics,
    )


def export_benchmark_results(
    reports: Sequence[AlgorithmEvaluationReport],
    output_dir: str | Path,
    *,
    experiment_id: str | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> tuple[Path, Path, Path]:
    """Persiste los resultados de la evaluación en `summary.csv`, `details.json` y `metadata.json`."""
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    exp_id = experiment_id or f"benchmark_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    csv_path = target_dir / "summary.csv"
    json_path = target_dir / "details.json"
    metadata_path = target_dir / "metadata.json"

    # 1. Escritura de summary.csv
    csv_headers = [
        "scenario_name",
        "algorithm",
        "alpha",
        "solved",
        "package_order",
        "objective_cost",
        "total_elapsed_time",
        "total_travel_time",
        "total_wait_time",
        "total_service_time",
        "total_tardiness",
        "weighted_tardiness",
        "total_penalty",
        "delivery_count",
        "on_promise_count",
        "on_promise_percentage",
        "window_compliance_percentage",
        "mean_execution_time_seconds",
        "median_execution_time_seconds",
        "min_execution_time_seconds",
        "max_execution_time_seconds",
        "stdev_execution_time_seconds",
        "states_generated_or_permutations",
        "states_expanded_or_feasible",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(csv_headers)
        for r in reports:
            q = r.quality
            s_gen = r.search_metrics.get("states_generated") or r.search_metrics.get("permutations_evaluated") or ""
            s_exp = r.search_metrics.get("states_expanded") or r.search_metrics.get("feasible_permutations") or ""
            row = [
                r.scenario_name,
                r.algorithm,
                r.alpha,
                r.solved,
                "->".join(r.package_order) if r.package_order else "None",
                q.objective_cost if q else "",
                q.total_elapsed_time if q else "",
                q.total_travel_time if q else "",
                q.total_wait_time if q else "",
                q.total_service_time if q else "",
                q.total_tardiness if q else "",
                q.weighted_tardiness if q else "",
                q.total_penalty if q else "",
                q.delivery_count if q else "",
                q.on_promise_count if q else "",
                q.on_promise_percentage if q else "",
                q.window_compliance_percentage if q else "",
                f"{r.timing.mean_seconds:.8f}",
                f"{r.timing.median_seconds:.8f}",
                f"{r.timing.min_seconds:.8f}",
                f"{r.timing.max_seconds:.8f}",
                f"{r.timing.stdev_seconds:.8f}",
                s_gen,
                s_exp,
            ]
            writer.writerow(row)

    # 2. Escritura de details.json
    details_data = [r.to_dict() for r in reports]
    with json_path.open("w", encoding="utf-8") as f_json:
        json.dump(details_data, f_json, indent=2, ensure_ascii=False)

    # 3. Escritura de metadata.json
    scenarios_included = sorted(set(r.scenario_name for r in reports))
    algorithms_included = sorted(set(r.algorithm for r in reports))
    alphas_included = sorted(set(r.alpha for r in reports))
    metadata_data: dict[str, Any] = {
        "experiment_id": exp_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "system": {
            "os": platform.system(),
            "os_release": platform.release(),
            "python_version": platform.python_version(),
            "machine": platform.machine(),
        },
        "scenarios": scenarios_included,
        "algorithms": algorithms_included,
        "alphas": alphas_included,
        "total_experiments": len(reports),
    }
    if extra_metadata:
        metadata_data.update(extra_metadata)

    with metadata_path.open("w", encoding="utf-8") as f_meta:
        json.dump(metadata_data, f_meta, indent=2, ensure_ascii=False)

    return csv_path, json_path, metadata_path


def run_benchmark_suite(
    scenarios: Sequence[DeliveryProblem | Path | str],
    *,
    alphas: Sequence[float] = (0.0, 0.5, 1.0, 2.0),
    algorithms: Sequence[str] = ("A*", "Dijkstra", "Greedy", "Exhaustive"),
    repetitions: int = 10,
    output_dir: str | Path | None = "results/benchmarks",
    experiment_id: str | None = None,
) -> BenchmarkSuiteResult:
    """Ejecuta una batería completa y reproducible de benchmarks sobre múltiples escenarios y alphas."""
    loaded_problems: list[DeliveryProblem] = []
    for item in scenarios:
        if isinstance(item, DeliveryProblem):
            loaded_problems.append(item)
        else:
            loaded_problems.append(load_problem_from_json(item))

    reports: list[AlgorithmEvaluationReport] = []
    for prob in loaded_problems:
        for alpha_val in alphas:
            for algo in algorithms:
                report = evaluate_algorithm_on_scenario(
                    algo,
                    prob,
                    alpha=alpha_val,
                    repetitions=repetitions,
                )
                reports.append(report)

    csv_path, json_path, meta_path = None, None, None
    out_path = Path(output_dir) if output_dir else None
    if out_path:
        csv_path, json_path, meta_path = export_benchmark_results(
            reports,
            out_path,
            experiment_id=experiment_id,
            extra_metadata={"configured_repetitions": repetitions},
        )

    return BenchmarkSuiteResult(
        experiment_id=experiment_id or "benchmark_suite",
        reports=tuple(reports),
        output_dir=out_path,
        csv_path=csv_path,
        json_path=json_path,
        metadata_path=meta_path,
    )

