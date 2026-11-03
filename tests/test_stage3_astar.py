"""Pruebas unitarias de la Etapa 3: heurística admisible y algoritmo A*."""

from __future__ import annotations

import itertools
from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from delivery_agent.algorithms.astar import solve_astar
from delivery_agent.algorithms.heuristics import min_travel_time_heuristic
from delivery_agent.core.cost import summarize_route_costs
from delivery_agent.core.transitions import simulate_route
from delivery_agent.domain.package import Package
from delivery_agent.domain.problem import DeliveryProblem
from delivery_agent.domain.state import DeliveryState
from delivery_agent.scenarios.loader import load_problem_from_json


def _brute_force_optimal_cost(problem: DeliveryProblem) -> float | None:
    """Auxiliar de prueba para verificar optimalidad exacta en escenarios pequeños."""
    pkg_ids = [pkg.id for pkg in problem.packages]
    if not pkg_ids:
        return 0.0

    best_cost: float | None = None
    for perm in itertools.permutations(pkg_ids):
        steps = simulate_route(problem, perm)
        if steps is None:
            continue
        cost = summarize_route_costs(steps, alpha=problem.alpha).objective_cost
        if best_cost is None or cost < best_cost:
            best_cost = cost
    return best_cost


class TestHeuristicsAndAStar(unittest.TestCase):
    """Suite de pruebas para la heurística y el algoritmo A* (Etapa 3)."""

    def _verify_solution_integrity(
        self,
        problem: DeliveryProblem,
        result_transitions: tuple,
        expected_total_cost: float,
    ) -> None:
        """Verifica que la ruta reconstruida coincida con simulate_route y cumpla todas las ventanas."""
        package_order = [step.package_id for step in result_transitions]
        self.assertEqual(len(package_order), len(problem.packages))
        self.assertEqual(set(package_order), {p.id for p in problem.packages})

        simulated = simulate_route(problem, package_order)
        self.assertIsNotNone(simulated)
        assert simulated is not None
        self.assertEqual(simulated, result_transitions)

        for step in result_transitions:
            self.assertGreaterEqual(step.service_start_time, step.window_start)
            self.assertLessEqual(step.completion_time, step.window_end)

        sim_summary = summarize_route_costs(simulated, alpha=problem.alpha)
        self.assertAlmostEqual(sim_summary.objective_cost, expected_total_cost)

    def test_01_single_package_scenario(self) -> None:
        pkg = Package("P1", "L1", 480, 600, 500, "high", 5)
        problem = DeliveryProblem(
            name="un_paquete",
            start_location="D0",
            start_time=480,
            packages=(pkg,),
            locations=("D0", "L1"),
            travel_times={
                "D0": {"D0": 0, "L1": 12},
                "L1": {"D0": 12, "L1": 0},
            },
            alpha=1.0,
        )

        result = solve_astar(problem)
        self.assertTrue(result.solved)
        self.assertEqual(result.package_order, ("P1",))
        self.assertAlmostEqual(result.total_cost or 0.0, 17.0)
        self.assertEqual(result.metrics.states_expanded, 1)
        self.assertEqual(result.metrics.states_generated, 1)
        self._verify_solution_integrity(problem, result.transitions, result.total_cost or 0.0)

    def test_02_multiple_packages_base_scenario(self) -> None:
        scenario_path = (
            PROJECT_ROOT / "data" / "scenarios" / "escenario_base_5_paquetes.json"
        )
        problem = load_problem_from_json(scenario_path)
        result = solve_astar(problem)

        self.assertTrue(result.solved)
        assert result.total_cost is not None
        self._verify_solution_integrity(problem, result.transitions, result.total_cost)

        oracle_cost = _brute_force_optimal_cost(problem)
        self.assertIsNotNone(oracle_cost)
        assert oracle_cost is not None
        self.assertAlmostEqual(result.total_cost, oracle_cost)

        # Verificar que la heurística es 0 en el estado objetivo y no sobreestima el costo óptimo
        assert result.final_state is not None
        self.assertEqual(min_travel_time_heuristic(problem, result.final_state), 0.0)
        self.assertLessEqual(
            min_travel_time_heuristic(problem, problem.initial_state),
            result.total_cost,
        )

    def test_03_scenario_with_waiting_time(self) -> None:
        # Sale a las 480, llega a L1 en 490, pero la ventana abre en 510 -> espera 20 min
        pkg = Package("P1", "L1", 510, 600, 520, "medium", 5)
        problem = DeliveryProblem(
            name="con_espera",
            start_location="D0",
            start_time=480,
            packages=(pkg,),
            locations=("D0", "L1"),
            travel_times={
                "D0": {"D0": 0, "L1": 10},
                "L1": {"D0": 10, "L1": 0},
            },
        )
        result = solve_astar(problem)
        self.assertTrue(result.solved)
        self.assertEqual(result.wait_times, (20,))
        self.assertEqual(result.service_start_times, (510,))
        self.assertEqual(result.completion_times, (515,))
        self.assertAlmostEqual(result.total_cost or 0.0, 35.0)
        self._verify_solution_integrity(problem, result.transitions, result.total_cost or 0.0)

    def test_04_scenario_with_tardiness_and_different_priorities(self) -> None:
        # Dos paquetes con igual geometría, pero P_high tiene prioridad alta (w=3) y P_low baja (w=1)
        p_low = Package("P_low", "L1", 480, 600, 495, "low", 5)
        p_high = Package("P_high", "L2", 480, 600, 495, "high", 5)
        problem = DeliveryProblem(
            name="retrasos_y_prioridades",
            start_location="D0",
            start_time=480,
            packages=(p_low, p_high),
            locations=("D0", "L1", "L2"),
            travel_times={
                "D0": {"D0": 0, "L1": 10, "L2": 10},
                "L1": {"D0": 10, "L1": 0, "L2": 10},
                "L2": {"D0": 10, "L1": 10, "L2": 0},
            },
            alpha=1.0,
        )
        result = solve_astar(problem)
        self.assertTrue(result.solved)
        # Debe entregar primero P_high (sin retraso) y luego P_low (con retraso ponderado por w=1 en vez de w=3)
        self.assertEqual(result.package_order, ("P_high", "P_low"))
        self.assertEqual(result.tardiness_values, (0, 15))
        self.assertEqual(result.penalties, (0.0, 15.0))
        self.assertAlmostEqual(result.total_cost or 0.0, 30.0 + 15.0)
        self._verify_solution_integrity(problem, result.transitions, result.total_cost or 0.0)

    def test_05_nearest_neighbor_is_not_optimal_cost(self) -> None:
        # L_near está a 5 min de D0, L_far está a 10 min de D0.
        # Si va primero a L_near (voraz), llega tarde a P_far (prioridad alta, promesa 495) con gran penalización.
        # Si va primero a L_far, cumple ambas promesas sin retraso y obtiene menor costo total.
        p_near = Package("P_near", "L_near", 480, 600, 580, "low", 5)
        p_far = Package("P_far", "L_far", 480, 600, 495, "high", 5)
        problem = DeliveryProblem(
            name="voraz_no_optimo",
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
        result = solve_astar(problem)
        self.assertTrue(result.solved)
        self.assertEqual(result.package_order, ("P_far", "P_near"))
        self.assertAlmostEqual(result.total_cost or 0.0, 30.0)

        greedy_steps = simulate_route(problem, ["P_near", "P_far"])
        assert greedy_steps is not None
        greedy_cost = summarize_route_costs(greedy_steps, alpha=1.0).objective_cost
        self.assertAlmostEqual(greedy_cost, 55.0)
        self.assertLess(result.total_cost or 0.0, greedy_cost)
        self._verify_solution_integrity(problem, result.transitions, result.total_cost or 0.0)

    def test_06_multiple_candidate_states_and_state_deduplication(self) -> None:
        # Caso A: el camino óptimo hacia X = ("L1", 500, {"P3"}) se explora primero (g=25)
        # y el segundo camino hacia X (g=35 >= 25) se descarta inmediatamente como dominado.
        p1_low = Package("P1", "L1", 480, 600, 495, "low", 5)
        p2_high = Package("P2", "L1", 480, 600, 495, "high", 5)
        p3 = Package("P3", "L2", 480, 600, 520, "medium", 5)
        problem_dominated = DeliveryProblem(
            name="candidatos_dominados",
            start_location="D0",
            start_time=480,
            packages=(p1_low, p2_high, p3),
            locations=("D0", "L1", "L2"),
            travel_times={
                "D0": {"D0": 0, "L1": 10, "L2": 20},
                "L1": {"D0": 10, "L1": 0, "L2": 8},
                "L2": {"D0": 20, "L1": 8, "L2": 0},
            },
            alpha=1.0,
        )
        res_dom = solve_astar(problem_dominated)
        self.assertTrue(res_dom.solved)
        self.assertEqual(res_dom.package_order, ("P2", "P1", "P3"))
        self.assertAlmostEqual(res_dom.total_cost or 0.0, 38.0)
        self.assertGreaterEqual(res_dom.metrics.dominated_or_repeated_discarded, 1)
        self._verify_solution_integrity(
            problem_dominated, res_dom.transitions, res_dom.total_cost or 0.0
        )

        # Caso B: el camino más costoso hacia X (g=35, f=43) se genera primero en OPEN y luego
        # se descubre el camino mejor (g=25, f=33). Como P3 tiene retraso en L2 (f_goal=62 > 43),
        # la entrada obsoleta de X (f=43) se extrae antes que el objetivo y se descarta como stale.
        p1_high = Package("P1", "L1", 480, 600, 495, "high", 5)
        p2_low = Package("P2", "L1", 480, 600, 495, "low", 5)
        p3_tight = Package("P3", "L2", 480, 600, 505, "high", 5)
        problem_stale = DeliveryProblem(
            name="candidatos_stale",
            start_location="D0",
            start_time=480,
            packages=(p1_high, p2_low, p3_tight),
            locations=("D0", "L1", "L2"),
            travel_times={
                "D0": {"D0": 0, "L1": 10, "L2": 20},
                "L1": {"D0": 10, "L1": 0, "L2": 8},
                "L2": {"D0": 20, "L1": 8, "L2": 0},
            },
            alpha=1.0,
        )
        res_stale = solve_astar(problem_stale)
        self.assertTrue(res_stale.solved)
        self.assertEqual(res_stale.package_order, ("P1", "P2", "P3"))
        self.assertAlmostEqual(res_stale.total_cost or 0.0, 62.0)
        self.assertGreaterEqual(res_stale.metrics.stale_queue_entries, 1)
        self._verify_solution_integrity(
            problem_stale, res_stale.transitions, res_stale.total_cost or 0.0
        )

    def test_07_infeasible_scenario_returns_no_solution(self) -> None:
        # Dos paquetes cuyas ventanas estrechas en ubicaciones lejanas son mutuamente excluyentes
        p1 = Package("P1", "L1", 480, 495, 490, "high", 5)
        p2 = Package("P2", "L2", 480, 495, 490, "high", 5)
        problem = DeliveryProblem(
            name="sin_solucion",
            start_location="D0",
            start_time=480,
            packages=(p1, p2),
            locations=("D0", "L1", "L2"),
            travel_times={
                "D0": {"D0": 0, "L1": 10, "L2": 10},
                "L1": {"D0": 10, "L1": 0, "L2": 20},
                "L2": {"D0": 10, "L1": 20, "L2": 0},
            },
        )
        result = solve_astar(problem)
        self.assertFalse(result.solved)
        self.assertFalse(result.is_feasible)
        self.assertEqual(result.transitions, ())
        self.assertIsNone(result.final_state)
        self.assertIsNone(result.total_cost)
        self.assertIsNone(result.cost_summary)
        self.assertGreater(result.metrics.infeasible_transitions_discarded, 0)

    def test_08_empty_packages_scenario_returns_zero_cost_solution(self) -> None:
        problem = DeliveryProblem(
            name="sin_paquetes",
            start_location="D0",
            start_time=480,
            packages=(),
            locations=("D0",),
            travel_times={"D0": {"D0": 0}},
        )
        result = solve_astar(problem)
        self.assertTrue(result.solved)
        self.assertEqual(result.transitions, ())
        self.assertEqual(result.package_order, ())
        self.assertEqual(result.final_state, DeliveryState("D0", 480, frozenset()))
        self.assertEqual(result.total_cost, 0.0)
        self.assertEqual(result.metrics.states_expanded, 0)
        self.assertEqual(result.metrics.states_generated, 0)

    def test_09_two_packages_at_same_location(self) -> None:
        p1 = Package("P1", "L1", 480, 550, 500, "medium", 5)
        p2 = Package("P2", "L1", 480, 550, 495, "high", 5)
        problem = DeliveryProblem(
            name="misma_ubicacion",
            start_location="D0",
            start_time=480,
            packages=(p1, p2),
            locations=("D0", "L1"),
            travel_times={
                "D0": {"D0": 0, "L1": 10},
                "L1": {"D0": 10, "L1": 0},
            },
        )
        result = solve_astar(problem)
        self.assertTrue(result.solved)
        # Entrega primero P2 (promesa 495, prioridad alta) y luego P1 (promesa 500) -> 0 retraso en ambos
        self.assertEqual(result.package_order, ("P2", "P1"))
        self.assertEqual(result.completion_times, (495, 500))
        self.assertEqual(result.tardiness_values, (0, 0))
        self.assertAlmostEqual(result.total_cost or 0.0, 20.0)
        self._verify_solution_integrity(problem, result.transitions, result.total_cost or 0.0)

    def test_10_asymmetric_travel_times(self) -> None:
        # L1 -> L2 toma 5 min, pero L2 -> L1 toma 40 min.
        p1 = Package("P1", "L1", 480, 600, 550, "low", 5)
        p2 = Package("P2", "L2", 480, 600, 550, "low", 5)
        problem = DeliveryProblem(
            name="asimetrico",
            start_location="D0",
            start_time=480,
            packages=(p1, p2),
            locations=("D0", "L1", "L2"),
            travel_times={
                "D0": {"D0": 0, "L1": 10, "L2": 10},
                "L1": {"D0": 10, "L1": 0, "L2": 5},
                "L2": {"D0": 10, "L1": 40, "L2": 0},
            },
        )
        result = solve_astar(problem)
        self.assertTrue(result.solved)
        self.assertEqual(result.package_order, ("P1", "P2"))
        self.assertAlmostEqual(result.total_cost or 0.0, 25.0)
        self._verify_solution_integrity(problem, result.transitions, result.total_cost or 0.0)

    def test_11_goal_is_selected_when_popped_not_when_generated(self) -> None:
        # Desde D0 (t=480):
        # - Ir primero a P1 (D0->L1 = 5, servicio = 5) tiene g=10, h(L1)=5 -> f = 15.0.
        # - Ir primero a P2 (D0->L2 = 12, servicio = 5) tiene g=17, h(L2)=5 -> f = 22.0.
        # Por tanto, A* extrae y expande primero S_P1 (f=15 < 22) y GENERA un estado objetivo G_1 (P1->P2).
        # Sin embargo, en P1->P2, P2 termina en t=500 > promesa 497 con prioridad alta y alpha=10,
        # dando un costo de G_1 = 20 + 10*3*3 = 110.0.
        # Como A* solo termina al EXTRAER el objetivo de la cola, G_1 (f=110) queda en espera,
        # A* extrae S_P2 (f=22), genera G_2 (P2->P1) con costo 27.0 y extrae G_2 como solución óptima.
        p1 = Package("P1", "L1", 480, 600, 520, "low", 5)
        p2 = Package("P2", "L2", 480, 600, 497, "high", 5)
        problem = DeliveryProblem(
            name="objetivo_al_extraer",
            start_location="D0",
            start_time=480,
            packages=(p1, p2),
            locations=("D0", "L1", "L2"),
            travel_times={
                "D0": {"D0": 0, "L1": 5, "L2": 12},
                "L1": {"D0": 5, "L1": 0, "L2": 5},
                "L2": {"D0": 12, "L1": 5, "L2": 0},
            },
            alpha=10.0,
        )
        result = solve_astar(problem)
        self.assertTrue(result.solved)
        self.assertEqual(result.package_order, ("P2", "P1"))
        self.assertAlmostEqual(result.total_cost or 0.0, 27.0)
        self._verify_solution_integrity(problem, result.transitions, result.total_cost or 0.0)

    def test_12_state_reopening_when_better_g_is_found(self) -> None:
        # P1 y P2 en L1 (servicio 5 cada uno), P3 en L2 (traslado L1->L2 = 25, servicio = 5).
        # Tanto (P1 -> P2) como (P2 -> P1) llegan al mismo estado intermedio:
        # X = DeliveryState(location="L1", current_time=500, pending_packages=frozenset({"P3"}))
        # - Camino (P2 -> P1) llega a X con g = 35.0 (P1 tiene retraso 5 * w=3 = 15).
        # - Camino (P1 -> P2) llega a X con g = 25.0 (P2 tiene retraso 5 * w=1 = 5).
        p1 = Package("P1", "L1", 480, 600, 495, "high", 5)
        p2 = Package("P2", "L1", 480, 600, 495, "low", 5)
        p3 = Package("P3", "L2", 480, 600, 560, "medium", 5)
        problem = DeliveryProblem(
            name="reapertura_de_estado",
            start_location="D0",
            start_time=480,
            packages=(p1, p2, p3),
            locations=("D0", "L1", "L2"),
            travel_times={
                "D0": {"D0": 0, "L1": 10, "L2": 40},
                "L1": {"D0": 10, "L1": 0, "L2": 25},
                "L2": {"D0": 40, "L1": 25, "L2": 0},
            },
            alpha=1.0,
        )

        # Heurística admisible pero no consistente:
        # En S_P1 (pendientes {"P2", "P3"}), el costo óptimo restante real es 10 (P2) + 30 (P3) = 40.0.
        # Asignamos h(S_P1) = 22.0 <= 40.0 (admisible) y 0.0 al resto.
        # Así:
        # 1) A* expande S_P2 (f = 15 + 0 = 15), que genera X con g = 35, f = 35.
        # 2) Como f(X) = 35 < f(S_P1) = 15 + 22 = 37, A* extrae y EXPANDE X con g = 35 (generando Goal con f = 65).
        # 3) A continuación A* extrae S_P1 (f = 37 < 65), descubre X con g = 25 < 35 y REABRE X.
        def admissible_inconsistent_heuristic(
            prob: DeliveryProblem, state: DeliveryState
        ) -> float:
            if state.pending_packages == frozenset({"P2", "P3"}):
                return 22.0
            return 0.0

        result = solve_astar(problem, heuristic_fn=admissible_inconsistent_heuristic)
        self.assertTrue(result.solved)
        self.assertEqual(result.package_order, ("P1", "P2", "P3"))
        self.assertAlmostEqual(result.total_cost or 0.0, 55.0)
        self.assertEqual(result.metrics.reopenings, 1)
        self._verify_solution_integrity(problem, result.transitions, result.total_cost or 0.0)


if __name__ == "__main__":
    unittest.main()
