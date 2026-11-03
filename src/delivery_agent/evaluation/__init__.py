"""Módulo de evaluación experimental, métricas, benchmarking y comparación."""

from delivery_agent.evaluation.benchmark import (
    ALGORITHM_DISPATCH,
    BenchmarkSuiteResult,
    evaluate_algorithm_on_scenario,
    export_benchmark_results,
    measure_algorithm_execution,
    run_benchmark_suite,
)
from delivery_agent.evaluation.comparison import (
    NUMERICAL_TOLERANCE,
    ScenarioComparisonResult,
    compare_scenario_algorithms,
    run_alpha_sensitivity_analysis,
)
from delivery_agent.evaluation.metrics import (
    AlgorithmEvaluationReport,
    ExecutionTimingMetrics,
    RouteQualityMetrics,
    compute_route_quality_metrics,
    compute_timing_statistics,
    extract_specific_search_metrics,
)

__all__ = [
    "ALGORITHM_DISPATCH",
    "AlgorithmEvaluationReport",
    "BenchmarkSuiteResult",
    "ExecutionTimingMetrics",
    "NUMERICAL_TOLERANCE",
    "RouteQualityMetrics",
    "ScenarioComparisonResult",
    "compare_scenario_algorithms",
    "compute_route_quality_metrics",
    "compute_timing_statistics",
    "evaluate_algorithm_on_scenario",
    "export_benchmark_results",
    "extract_specific_search_metrics",
    "measure_algorithm_execution",
    "run_alpha_sensitivity_analysis",
    "run_benchmark_suite",
]

