"""Pruebas unitarias de la Etapa 5: sistema de evaluación, benchmarking y métricas."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from delivery_agent.algorithms.astar import solve_astar
from delivery_agent.algorithms.dijkstra import solve_dijkstra
from delivery_agent.algorithms.exhaustive import solve_exhaustive
from delivery_agent.algorithms.greedy import solve_greedy
from delivery_agent.core.transitions import simulate_route
from delivery_agent.domain.package import Package
from delivery_agent.domain.problem import DeliveryProblem
from delivery_agent.evaluation.benchmark import (
    evaluate_algorithm_on_scenario,
    export_benchmark_results,
    measure_algorithm_execution,
    run_benchmark_suite,
)
from delivery_agent.evaluation.comparison import (
    compare_scenario_algorithms,
    run_alpha_sensitivity_analysis,
)
from delivery_agent.evaluation.metrics import (
    compute_route_quality_metrics,
    compute_timing_statistics,
    extract_specific_search_metrics,
)
from delivery_agent.scenarios.loader import discover_scenario_files, load_problem_from_json


class TestEvaluationAndBenchmarking(unittest.TestCase):
    """Suite de pruebas para métricas, benchmarking, comparación y persistencia."""

    def setUp(self) -> None:
        self.scenarios_dir = PROJECT_ROOT / "data" / "scenarios"
        self.scen1_path = self.scenarios_dir / "escenario_01_base_5_paquetes.json"
        self.problem1 = load_problem_from_json(self.scen1_path)

    def test_01_metrics_aggregation_and_breakdown_coherence(self) -> None:
        """Comprueba la correcta agregación de métricas y la coherencia de C = T_total + alpha * D_p."""
        res = solve_astar(self.problem1, alpha=1.5)
        self.assertTrue(res.solved)

        metrics = compute_route_quality_metrics(
            res.transitions,
            expected_packages=len(self.problem1.packages),
            alpha=1.5,
        )

        self.assertEqual(metrics.delivery_count, 5)
        self.assertEqual(metrics.expected_packages, 5)
        self.assertEqual(metrics.window_compliance_percentage, 100.0)

        # Coherencia matemática exacta
        self.assertEqual(
            metrics.total_elapsed_time,
            metrics.total_travel_time + metrics.total_wait_time + metrics.total_service_time,
        )
        expected_obj = metrics.total_elapsed_time + 1.5 * metrics.weighted_tardiness
        self.assertAlmostEqual(metrics.objective_cost, expected_obj)
        self.assertAlmostEqual(metrics.total_penalty, 1.5 * metrics.weighted_tardiness)

    def test_02_timing_statistics_computation(self) -> None:
        """Verifica el cálculo de estadísticas temporales (media, mediana, min, max, stdev)."""
        samples = [0.010, 0.012, 0.011, 0.015, 0.010]
        stats = compute_timing_statistics(samples)

        self.assertEqual(stats.runs, 5)
        self.assertAlmostEqual(stats.min_seconds, 0.010)
        self.assertAlmostEqual(stats.max_seconds, 0.015)
        self.assertAlmostEqual(stats.median_seconds, 0.011)
        self.assertGreater(stats.stdev_seconds, 0.0)

        with self.assertRaises(ValueError):
            compute_timing_statistics([])

    def test_03_extract_specific_search_metrics_by_algorithm(self) -> None:
        """Verifica que las métricas específicas respeten la naturaleza de cada algoritmo."""
        res_a = solve_astar(self.problem1)
        m_a = extract_specific_search_metrics(res_a)
        self.assertIn("states_generated", m_a)
        self.assertIn("states_expanded", m_a)
        self.assertIn("max_queue_size", m_a)

        res_g = solve_greedy(self.problem1)
        m_g = extract_specific_search_metrics(res_g)
        self.assertIn("steps_executed", m_g)
        self.assertIn("completed_route", m_g)
        self.assertIn("termination_reason", m_g)

        res_e = solve_exhaustive(self.problem1)
        m_e = extract_specific_search_metrics(res_e)
        self.assertIn("permutations_evaluated", m_e)
        self.assertIn("feasible_permutations", m_e)

    def test_04_comparability_and_optimality_across_four_algorithms(self) -> None:
        """Comprueba que A* y Dijkstra coincidan exactamente con la enumeración exhaustiva."""
        comparison = compare_scenario_algorithms(self.problem1, alpha=1.0)

        self.assertTrue(comparison.is_feasible)
        self.assertTrue(comparison.optimality_verified)
        self.assertTrue(comparison.astar_solved)
        self.assertTrue(comparison.dijkstra_solved)
        self.assertTrue(comparison.exhaustive_solved)

        assert comparison.optimal_cost is not None
        self.assertAlmostEqual(comparison.astar_cost or 0.0, comparison.optimal_cost)
        self.assertAlmostEqual(comparison.dijkstra_cost or 0.0, comparison.optimal_cost)

        # A* debe expandir <= estados que Dijkstra
        self.assertLessEqual(comparison.astar_states_expanded, comparison.dijkstra_states_expanded)

    def test_05_infeasible_scenario_unanimously_detected(self) -> None:
        """Verifica que en el escenario 8 (sin solución), A*, Dijkstra y Exhaustiva reporten inviabilidad."""
        scen8_path = self.scenarios_dir / "escenario_08_sin_solucion_5_paquetes.json"
        prob8 = load_problem_from_json(scen8_path)

        comp8 = compare_scenario_algorithms(prob8, alpha=1.0)
        self.assertFalse(comp8.is_feasible)
        self.assertTrue(comp8.optimality_verified)
        self.assertFalse(comp8.astar_solved)
        self.assertFalse(comp8.dijkstra_solved)
        self.assertFalse(comp8.exhaustive_solved)
        self.assertIsNone(comp8.optimal_cost)

    def test_06_greedy_failure_and_suboptimality_identification(self) -> None:
        """Verifica que el sistema identifique fallos en Greedy y calcule la brecha de suboptimalidad."""
        # Escenario 6: trampa para Greedy (callejón sin salida)
        scen6_path = self.scenarios_dir / "escenario_06_greedy_suboptimo_6_paquetes.json"
        prob6 = load_problem_from_json(scen6_path)

        comp6 = compare_scenario_algorithms(prob6, alpha=2.0)
        self.assertTrue(comp6.is_feasible)
        self.assertFalse(comp6.greedy_solved)
        self.assertTrue(comp6.greedy_failed)  # Fallo de la estrategia, no del problema

        # Escenario 3: Greedy encuentra ruta pero con costo subóptimo
        scen3_path = self.scenarios_dir / "escenario_03_prioridades_promesas_6_paquetes.json"
        prob3 = load_problem_from_json(scen3_path)
        comp3 = compare_scenario_algorithms(prob3, alpha=2.0)

        self.assertTrue(comp3.greedy_solved)
        self.assertTrue(comp3.greedy_suboptimal)
        self.assertGreater(comp3.greedy_absolute_gap or 0.0, 0.0)
        self.assertGreater(comp3.greedy_percentage_gap or 0.0, 0.0)

    def test_07_alpha_sensitivity_analysis(self) -> None:
        """Verifica la ejecución del análisis de sensibilidad para alpha in [0.0, 0.5, 1.0, 2.0]."""
        alphas = (0.0, 0.5, 1.0, 2.0)
        results = run_alpha_sensitivity_analysis(
            self.problem1,
            alphas=alphas,
            run_exhaustive=True,
        )

        self.assertEqual(len(results), 4)
        for idx, alpha_val in enumerate(alphas):
            r = results[idx]
            self.assertEqual(r.alpha, alpha_val)
            self.assertTrue(r.optimality_verified)
            self.assertIsNotNone(r.optimal_cost)

    def test_08_benchmark_export_csv_json_and_metadata(self) -> None:
        """Verifica la persistencia de summary.csv, details.json y metadata.json."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "benchmarks"

            rep1 = evaluate_algorithm_on_scenario("A*", self.problem1, alpha=1.0, repetitions=2)
            rep2 = evaluate_algorithm_on_scenario("Greedy", self.problem1, alpha=1.0, repetitions=2)

            csv_p, json_p, meta_p = export_benchmark_results(
                [rep1, rep2],
                output_dir=out_dir,
                experiment_id="test_exp_01",
            )

            self.assertTrue(csv_p.is_file())
            self.assertTrue(json_p.is_file())
            self.assertTrue(meta_p.is_file())

            # Validar CSV
            with csv_p.open("r", encoding="utf-8") as f_csv:
                reader = list(csv.reader(f_csv))
                self.assertEqual(len(reader), 3)  # 1 header + 2 filas
                self.assertIn("A*", reader[1])
                self.assertIn("Greedy", reader[2])

            # Validar JSON
            with json_p.open("r", encoding="utf-8") as f_json:
                data = json.load(f_json)
                self.assertEqual(len(data), 2)
                self.assertEqual(data[0]["algorithm"], "A*")

            # Validar metadata
            with meta_p.open("r", encoding="utf-8") as f_meta:
                meta = json.load(f_meta)
                self.assertEqual(meta["experiment_id"], "test_exp_01")
                self.assertIn("A*", meta["algorithms"])

    def test_09_reproducibility_of_results(self) -> None:
        """Comprueba que dos ejecuciones independientes sobre el mismo escenario arrojen idénticos resultados."""
        res_a1 = solve_astar(self.problem1, alpha=1.0)
        res_a2 = solve_astar(self.problem1, alpha=1.0)

        self.assertEqual(res_a1.package_order, res_a2.package_order)
        self.assertEqual(res_a1.total_cost, res_a2.total_cost)
        self.assertEqual(res_a1.metrics.states_expanded, res_a2.metrics.states_expanded)
        self.assertEqual(res_a1.metrics.states_generated, res_a2.metrics.states_generated)

    def test_10_separation_of_simulated_time_and_cpu_execution_time(self) -> None:
        """Comprueba que el tiempo simulado (minutos del día) esté estrictamente separado del tiempo de CPU (segundos)."""
        result, timing = measure_algorithm_execution(
            "A*",
            self.problem1,
            alpha=1.0,
            repetitions=3,
        )
        self.assertTrue(result.solved)
        assert result.cost_summary is not None

        simulated_minutes = result.cost_summary.total_elapsed_time
        cpu_seconds = timing.median_seconds

        # El tiempo simulado de reparto es en minutos enteros de la jornada (e.g. 51 min)
        self.assertIsInstance(simulated_minutes, int)
        self.assertGreater(simulated_minutes, 0)

        # El tiempo de ejecución real en CPU es un float muy pequeño en segundos (e.g. < 0.1 s)
        self.assertIsInstance(cpu_seconds, float)
        self.assertLess(cpu_seconds, 1.0)


if __name__ == "__main__":
    unittest.main()

