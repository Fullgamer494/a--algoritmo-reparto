"""Pruebas unitarias de la Etapa 4: algoritmos de comparación (Dijkstra, Greedy y Exhaustiva)."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from delivery_agent.algorithms.astar import solve_astar
from delivery_agent.algorithms.dijkstra import solve_dijkstra
from delivery_agent.algorithms.exhaustive import solve_exhaustive
from delivery_agent.algorithms.greedy import solve_greedy
from delivery_agent.core.cost import summarize_route_costs
from delivery_agent.core.transitions import simulate_route
from delivery_agent.domain.package import Package
from delivery_agent.domain.problem import DeliveryProblem
from delivery_agent.scenarios.loader import load_problem_from_json


class TestComparisonAlgorithms(unittest.TestCase):
    """Suite de pruebas para Dijkstra, Greedy y Enumeración Exhaustiva."""

    def setUp(self) -> None:
        self.base_scenario_path = (
            PROJECT_ROOT / "data" / "scenarios" / "escenario_base_5_paquetes.json"
        )
        self.base_problem = load_problem_from_json(self.base_scenario_path)

    def _verify_route_validity(
        self,
        problem: DeliveryProblem,
        transitions: tuple,
        expected_cost: float,
        alpha: float,
    ) -> None:
        """Comprueba que la ruta respete ventanas, coincida con simulate_route y tenga el costo esperado."""
        seq = [t.package_id for t in transitions]
        simulated = simulate_route(problem, seq, alpha=alpha)
        self.assertIsNotNone(simulated)
        assert simulated is not None
        self.assertEqual(simulated, transitions)

        for step in transitions:
            self.assertGreaterEqual(step.service_start_time, step.window_start)
            self.assertLessEqual(step.completion_time, step.window_end)

        summary = summarize_route_costs(simulated, alpha=alpha)
        self.assertAlmostEqual(summary.objective_cost, expected_cost, places=6)

    def test_01_dijkstra_matches_astar_optimal_cost(self) -> None:
        """Dijkstra encuentra exactamente el mismo costo óptimo que A* en el escenario base."""
        res_astar = solve_astar(self.base_problem)
        res_dijkstra = solve_dijkstra(self.base_problem)

        self.assertTrue(res_astar.solved)
        self.assertTrue(res_dijkstra.solved)
        self.assertIsNotNone(res_astar.total_cost)
        self.assertIsNotNone(res_dijkstra.total_cost)

        assert res_astar.total_cost is not None and res_dijkstra.total_cost is not None
        self.assertAlmostEqual(res_dijkstra.total_cost, res_astar.total_cost)

        self._verify_route_validity(
            self.base_problem,
            res_dijkstra.transitions,
            res_dijkstra.total_cost,
            self.base_problem.alpha,
        )

        # La heurística admisible min_travel_time de A* debe permitirle expandir <= estados que Dijkstra
        self.assertLessEqual(
            res_astar.metrics.states_expanded,
            res_dijkstra.metrics.states_expanded,
        )

    def test_02_astar_and_dijkstra_match_exhaustive_oracle(self) -> None:
        """A* y Dijkstra coinciden exactamente con la enumeración exhaustiva en costo y factibilidad."""
        res_astar = solve_astar(self.base_problem)
        res_dijkstra = solve_dijkstra(self.base_problem)
        res_exhaustive = solve_exhaustive(self.base_problem)

        self.assertTrue(res_exhaustive.solved)
        self.assertIsNotNone(res_exhaustive.optimal_cost)
        assert res_exhaustive.optimal_cost is not None
        assert res_astar.total_cost is not None
        assert res_dijkstra.total_cost is not None

        # 5 paquetes = 5! = 120 permutaciones evaluadas
        self.assertEqual(res_exhaustive.total_permutations_evaluated, 120)
        self.assertEqual(res_exhaustive.feasible_permutations, 120)
        self.assertEqual(res_exhaustive.infeasible_permutations, 0)

        self.assertAlmostEqual(res_astar.total_cost, res_exhaustive.optimal_cost)
        self.assertAlmostEqual(res_dijkstra.total_cost, res_exhaustive.optimal_cost)

        self._verify_route_validity(
            self.base_problem,
            res_exhaustive.transitions,
            res_exhaustive.optimal_cost,
            self.base_problem.alpha,
        )

        # Probar además en un escenario con ventanas ajustadas donde existan permutaciones inviables
        p1 = Package("P1", "L1", 480, 500, 495, "high", 5)
        p2 = Package("P2", "L2", 480, 515, 510, "medium", 5)
        p3 = Package("P3", "L3", 480, 600, 550, "low", 5)
        tight_problem = DeliveryProblem(
            name="ventanas_ajustadas",
            start_location="D0",
            start_time=480,
            packages=(p1, p2, p3),
            locations=("D0", "L1", "L2", "L3"),
            travel_times={
                "D0": {"D0": 0, "L1": 10, "L2": 15, "L3": 20},
                "L1": {"D0": 10, "L1": 0, "L2": 10, "L3": 15},
                "L2": {"D0": 15, "L1": 10, "L2": 0, "L3": 10},
                "L3": {"D0": 20, "L1": 15, "L2": 10, "L3": 0},
            },
        )
        res_tight_astar = solve_astar(tight_problem)
        res_tight_dijkstra = solve_dijkstra(tight_problem)
        res_tight_exh = solve_exhaustive(tight_problem)

        self.assertTrue(res_tight_exh.solved)
        assert res_tight_exh.optimal_cost is not None
        assert res_tight_astar.total_cost is not None
        assert res_tight_dijkstra.total_cost is not None
        self.assertEqual(res_tight_exh.total_permutations_evaluated, 6)
        self.assertGreater(res_tight_exh.feasible_permutations, 0)
        self.assertGreater(res_tight_exh.infeasible_permutations, 0)
        self.assertAlmostEqual(res_tight_astar.total_cost, res_tight_exh.optimal_cost)
        self.assertAlmostEqual(res_tight_dijkstra.total_cost, res_tight_exh.optimal_cost)

    def test_03_greedy_finds_feasible_route_on_easy_scenario(self) -> None:
        """Greedy puede encontrar una solución factible cuando no hay ventanas conflictivas."""
        p1 = Package("P1", "L1", 480, 600, 550, "medium", 5)
        p2 = Package("P2", "L2", 480, 600, 550, "low", 5)
        problem = DeliveryProblem(
            name="greedy_facil",
            start_location="D0",
            start_time=480,
            packages=(p1, p2),
            locations=("D0", "L1", "L2"),
            travel_times={
                "D0": {"D0": 0, "L1": 8, "L2": 15},
                "L1": {"D0": 8, "L1": 0, "L2": 10},
                "L2": {"D0": 15, "L1": 10, "L2": 0},
            },
        )
        res_greedy = solve_greedy(problem)
        self.assertTrue(res_greedy.solved)
        self.assertEqual(res_greedy.package_order, ("P1", "P2"))
        self.assertIsNotNone(res_greedy.total_cost)
        assert res_greedy.total_cost is not None
        self._verify_route_validity(problem, res_greedy.transitions, res_greedy.total_cost, problem.alpha)

    def test_04_greedy_produces_suboptimal_solution(self) -> None:
        """Greedy produce una solución con costo mayor que el óptimo al guiarse solo por traslado local."""
        # D0 -> L_near toma 5 min, D0 -> L_far toma 10 min.
        # P_far tiene ventana y promesa estrecha (promesa 495, prioridad high).
        # Greedy va a L_near (5 < 10) y luego llega tarde a P_far, penalizando severamente.
        # A*, Dijkstra y Exhaustive eligen ir primero a L_far, logrando menor costo global.
        p_near = Package("P_near", "L_near", 480, 600, 580, "low", 5)
        p_far = Package("P_far", "L_far", 480, 600, 495, "high", 5)
        problem = DeliveryProblem(
            name="greedy_suboptimo",
            start_location="D0",
            start_time=480,
            packages=(p_near, p_far),
            locations=("D0", "L_near", "L_far"),
            travel_times={
                "D0": {"D0": 0, "L_near": 5, "L_far": 10},
                "L_near": {"D0": 5, "L_near": 0, "L_far": 10},
                "L_far": {"D0": 10, "L_near": 10, "L_far": 0},
            },
            alpha=1.0,
        )

        res_greedy = solve_greedy(problem)
        res_astar = solve_astar(problem)
        res_dijkstra = solve_dijkstra(problem)
        res_exh = solve_exhaustive(problem)

        self.assertTrue(res_greedy.solved)
        self.assertTrue(res_astar.solved)
        self.assertTrue(res_dijkstra.solved)
        self.assertTrue(res_exh.solved)

        assert res_greedy.total_cost is not None
        assert res_astar.total_cost is not None
        assert res_dijkstra.total_cost is not None
        assert res_exh.optimal_cost is not None

        # Orden de entregas
        self.assertEqual(res_greedy.package_order, ("P_near", "P_far"))
        self.assertEqual(res_astar.package_order, ("P_far", "P_near"))
        self.assertEqual(res_dijkstra.package_order, ("P_far", "P_near"))
        self.assertEqual(res_exh.best_sequence, ("P_far", "P_near"))

        # El costo óptimo es 30.0, mientras que el voraz es 55.0
        self.assertAlmostEqual(res_astar.total_cost, 30.0)
        self.assertAlmostEqual(res_dijkstra.total_cost, 30.0)
        self.assertAlmostEqual(res_exh.optimal_cost, 30.0)
        self.assertAlmostEqual(res_greedy.total_cost, 55.0)
        self.assertGreater(res_greedy.total_cost, res_astar.total_cost)

    def test_05_greedy_fails_in_dead_end_while_others_succeed(self) -> None:
        """Greedy queda atrapado en un callejón sin salida (inviable) mientras A*, Dijkstra y Exhaustive hallan la ruta."""
        # D0 -> L1 = 5, D0 -> L2 = 12.
        # P1 en L1 tiene ventana amplia [480, 600].
        # P2 en L2 tiene ventana temprana [480, 500].
        # L1 -> L2 toma 20 min.
        # Si Greedy elige P1 por cercanía (5 < 12), termina en L1 a las 490.
        # Desde L1 hacia L2 llega a las 490 + 20 = 510 > window_end(500) -> INVIABLE.
        # Por tanto Greedy se queda sin sucesores.
        # Sin embargo, la ruta P2 -> P1 es perfectamente factible:
        # D0 -> L2 llega 492, servicio 5 -> 497 <= 500. L2 -> L1 toma 10 min -> llega 507, servicio 5 -> 512 <= 600.
        p1 = Package("P1", "L1", 480, 600, 550, "low", 5)
        p2 = Package("P2", "L2", 480, 500, 497, "high", 5)
        problem = DeliveryProblem(
            name="greedy_callejon_sin_salida",
            start_location="D0",
            start_time=480,
            packages=(p1, p2),
            locations=("D0", "L1", "L2"),
            travel_times={
                "D0": {"D0": 0, "L1": 5, "L2": 12},
                "L1": {"D0": 5, "L1": 0, "L2": 20},
                "L2": {"D0": 12, "L1": 10, "L2": 0},
            },
        )

        res_greedy = solve_greedy(problem)
        res_astar = solve_astar(problem)
        res_dijkstra = solve_dijkstra(problem)
        res_exh = solve_exhaustive(problem)

        # Greedy falla (se queda sin ruta completa)
        self.assertFalse(res_greedy.solved)
        self.assertIsNone(res_greedy.total_cost)
        self.assertEqual(len(res_greedy.transitions), 1)  # Entregó P1 y se bloqueó

        # Los algoritmos óptimos sí encuentran la solución válida
        self.assertTrue(res_astar.solved)
        self.assertTrue(res_dijkstra.solved)
        self.assertTrue(res_exh.solved)
        self.assertEqual(res_astar.package_order, ("P2", "P1"))
        self.assertEqual(res_dijkstra.package_order, ("P2", "P1"))
        self.assertEqual(res_exh.best_sequence, ("P2", "P1"))
        assert res_astar.total_cost is not None
        self._verify_route_validity(problem, res_astar.transitions, res_astar.total_cost, problem.alpha)

    def test_06_all_four_approaches_handle_infeasible_scenario(self) -> None:
        """Los 4 algoritmos detectan y reportan correctamente cuando un escenario no tiene solución factible."""
        p1 = Package("P1", "L1", 480, 490, 485, "low", 5)
        p2 = Package("P2", "L2", 480, 490, 485, "low", 5)
        problem = DeliveryProblem(
            name="todos_inviable",
            start_location="D0",
            start_time=480,
            packages=(p1, p2),
            locations=("D0", "L1", "L2"),
            travel_times={
                "D0": {"D0": 0, "L1": 10, "L2": 10},
                "L1": {"D0": 10, "L1": 0, "L2": 15},
                "L2": {"D0": 10, "L1": 15, "L2": 0},
            },
        )

        res_astar = solve_astar(problem)
        res_dijkstra = solve_dijkstra(problem)
        res_greedy = solve_greedy(problem)
        res_exh = solve_exhaustive(problem)

        self.assertFalse(res_astar.solved)
        self.assertFalse(res_dijkstra.solved)
        self.assertFalse(res_greedy.solved)
        self.assertFalse(res_exh.solved)

        self.assertIsNone(res_astar.total_cost)
        self.assertIsNone(res_dijkstra.total_cost)
        self.assertIsNone(res_greedy.total_cost)
        self.assertIsNone(res_exh.optimal_cost)

        self.assertEqual(res_exh.feasible_permutations, 0)
        self.assertEqual(res_exh.infeasible_permutations, 2)

    def test_07_all_four_approaches_handle_empty_scenario(self) -> None:
        """Los 4 algoritmos resuelven un escenario vacío (0 paquetes) con costo cero y sin pasos."""
        problem = DeliveryProblem(
            name="todos_vacio",
            start_location="D0",
            start_time=480,
            packages=(),
            locations=("D0",),
            travel_times={"D0": {"D0": 0}},
        )

        res_astar = solve_astar(problem)
        res_dijkstra = solve_dijkstra(problem)
        res_greedy = solve_greedy(problem)
        res_exh = solve_exhaustive(problem)

        self.assertTrue(res_astar.solved)
        self.assertTrue(res_dijkstra.solved)
        self.assertTrue(res_greedy.solved)
        self.assertTrue(res_exh.solved)

        self.assertEqual(res_astar.total_cost, 0.0)
        self.assertEqual(res_dijkstra.total_cost, 0.0)
        self.assertEqual(res_greedy.total_cost, 0.0)
        self.assertEqual(res_exh.optimal_cost, 0.0)

        self.assertEqual(res_astar.package_order, ())
        self.assertEqual(res_dijkstra.package_order, ())
        self.assertEqual(res_greedy.package_order, ())
        self.assertEqual(res_exh.best_sequence, ())

    def test_08_comparison_with_different_alpha_values(self) -> None:
        """Los costos entre algoritmos son consistentes y comparables al variar el parámetro alpha."""
        for alpha_val in (0.5, 1.0, 2.0):
            res_astar = solve_astar(self.base_problem, alpha=alpha_val)
            res_dijkstra = solve_dijkstra(self.base_problem, alpha=alpha_val)
            res_exh = solve_exhaustive(self.base_problem, alpha=alpha_val)

            self.assertTrue(res_astar.solved)
            self.assertTrue(res_dijkstra.solved)
            self.assertTrue(res_exh.solved)

            assert res_astar.total_cost is not None
            assert res_dijkstra.total_cost is not None
            assert res_exh.optimal_cost is not None

            self.assertAlmostEqual(res_astar.total_cost, res_exh.optimal_cost)
            self.assertAlmostEqual(res_dijkstra.total_cost, res_exh.optimal_cost)

    def test_09_exhaustive_result_to_search_result_interoperability(self) -> None:
        """La conversión de ExhaustiveResult a SearchResult es transparente y compatible."""
        exh = solve_exhaustive(self.base_problem)
        search_res = exh.to_search_result()

        self.assertEqual(search_res.algorithm, "Exhaustive")
        self.assertEqual(search_res.solved, exh.solved)
        self.assertEqual(search_res.total_cost, exh.optimal_cost)
        self.assertEqual(search_res.package_order, exh.best_sequence)
        self.assertEqual(search_res.transitions, exh.transitions)
        self.assertEqual(search_res.metrics.states_generated, exh.total_permutations_evaluated)


if __name__ == "__main__":
    unittest.main()
