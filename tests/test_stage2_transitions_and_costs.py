"""Pruebas unitarias de la Etapa 2: motor de transiciones y evaluación de costos."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from delivery_agent.core.cost import (
    compute_incremental_cost,
    compute_penalty,
    compute_tardiness,
    compute_weighted_tardiness,
    evaluate_transition_cost,
    summarize_route_costs,
)
from delivery_agent.core.transitions import (
    apply_transition,
    generate_successors,
    simulate_route,
)
from delivery_agent.domain.package import Package
from delivery_agent.domain.problem import DeliveryProblem
from delivery_agent.domain.state import DeliveryState


def _build_test_problem(
    packages: list[Package],
    *,
    start_time: int = 480,
    alpha: float = 1.0,
    allow_zero_alpha: bool = False,
    travel_times: dict[str, dict[str, int | None]] | None = None,
) -> DeliveryProblem:
    locations = ["D0", "L1", "L2", "L3"]
    default_matrix: dict[str, dict[str, int | None]] = {
        "D0": {"D0": 0, "L1": 10, "L2": 20, "L3": None},
        "L1": {"D0": 15, "L1": 0, "L2": 8, "L3": 12},
        "L2": {"D0": 7, "L1": 18, "L2": 0, "L3": 6},
        "L3": {"D0": 25, "L1": 12, "L2": 6, "L3": 0},
    }
    return DeliveryProblem(
        name="escenario_pruebas_etapa2",
        start_location="D0",
        start_time=start_time,
        packages=tuple(packages),
        locations=tuple(locations),
        travel_times=travel_times or default_matrix,
        alpha=alpha,
        allow_zero_alpha=allow_zero_alpha,
    )


class TestTransitionsAndCostEngine(unittest.TestCase):
    """Casos de prueba del motor de transiciones y las fórmulas de costo."""

    def test_arrival_before_window_starts_adds_wait_time(self) -> None:
        # Sale a las 08:00 (480), traslado D0->L1 = 10 -> llega 490.
        # Ventana abre a las 08:20 (500) -> espera 10 min, inicia 500, termina 505.
        p1 = Package(
            id="P1",
            destination="L1",
            window_start=500,
            window_end=600,
            promised_time=540,
            priority="high",
            service_time=5,
        )
        problem = _build_test_problem([p1])
        initial = problem.initial_state

        result = apply_transition(problem, initial, "P1")
        self.assertIsNotNone(result)
        assert result is not None

        self.assertEqual(result.origin, "D0")
        self.assertEqual(result.destination, "L1")
        self.assertEqual(result.departure_time, 480)
        self.assertEqual(result.travel_time, 10)
        self.assertEqual(result.arrival_time, 490)
        self.assertEqual(result.wait_time, 10)
        self.assertEqual(result.service_start_time, 500)
        self.assertEqual(result.service_time, 5)
        self.assertEqual(result.completion_time, 505)
        self.assertEqual(result.elapsed_time, 25)
        self.assertEqual(result.tardiness, 0)
        self.assertEqual(result.penalty, 0.0)
        self.assertEqual(result.incremental_cost, 25.0)
        self.assertEqual(result.next_state, DeliveryState("L1", 505, frozenset()))
        # Estado original sin mutaciones
        self.assertEqual(initial.pending_packages, frozenset({"P1"}))
        self.assertEqual(initial.current_time, 480)

    def test_arrival_exactly_at_window_start_has_zero_wait(self) -> None:
        # Sale 480 + traslado 10 = llega 490 == window_start (490)
        p1 = Package(
            id="P1",
            destination="L1",
            window_start=490,
            window_end=600,
            promised_time=510,
            priority="medium",
            service_time=5,
        )
        problem = _build_test_problem([p1])
        result = apply_transition(problem, problem.initial_state, p1)
        self.assertIsNotNone(result)
        assert result is not None

        self.assertEqual(result.arrival_time, 490)
        self.assertEqual(result.wait_time, 0)
        self.assertEqual(result.service_start_time, 490)
        self.assertEqual(result.completion_time, 495)
        self.assertEqual(result.elapsed_time, 15)

    def test_completion_exactly_at_window_end_is_valid(self) -> None:
        # Sale 480 + traslado 10 = llega 490, servicio 5 -> termina en 495 == window_end
        p1 = Package(
            id="P1",
            destination="L1",
            window_start=485,
            window_end=495,
            promised_time=490,
            priority="low",
            service_time=5,
        )
        problem = _build_test_problem([p1])
        result = apply_transition(problem, problem.initial_state, "P1")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.completion_time, 495)
        self.assertEqual(result.window_end, 495)

    def test_completion_after_window_end_is_discarded(self) -> None:
        # Sale 480 + traslado 10 = llega 490, servicio 6 -> termina en 496 > window_end (495)
        p1 = Package(
            id="P1",
            destination="L1",
            window_start=485,
            window_end=495,
            promised_time=490,
            priority="low",
            service_time=6,
        )
        problem = _build_test_problem([p1])
        result = apply_transition(problem, problem.initial_state, "P1")
        self.assertIsNone(result)
        self.assertEqual(generate_successors(problem, problem.initial_state), ())

    def test_early_vs_late_delivery_relative_to_promised_time(self) -> None:
        # Llega 490, servicio 5 -> finaliza 495.
        p_early = Package(
            id="P_early",
            destination="L1",
            window_start=480,
            window_end=550,
            promised_time=500,  # Finaliza en 495 <= 500 -> retraso 0
            priority="medium",
            service_time=5,
        )
        p_late = Package(
            id="P_late",
            destination="L1",
            window_start=480,
            window_end=550,
            promised_time=485,  # Finaliza en 495 > 485 -> retraso 10
            priority="medium",
            service_time=5,
        )
        problem = _build_test_problem([p_early, p_late], alpha=1.0)
        res_early = apply_transition(problem, problem.initial_state, "P_early")
        res_late = apply_transition(problem, problem.initial_state, "P_late")

        assert res_early is not None and res_late is not None
        self.assertEqual(res_early.tardiness, 0)
        self.assertEqual(res_early.penalty, 0.0)
        self.assertEqual(res_early.incremental_cost, 15.0)

        self.assertEqual(res_late.tardiness, 10)
        # priority medium = 2, alpha = 1.0 -> penalty = 20.0, incremental = 15 + 20 = 35.0
        self.assertEqual(res_late.penalty, 20.0)
        self.assertEqual(res_late.incremental_cost, 35.0)

    def test_different_priorities_with_same_tardiness(self) -> None:
        # Tres paquetes hacia L1 con idéntico horario (finalización = 495, promesa = 485 -> retraso = 10)
        p_low = Package("P_low", "L1", 480, 550, 485, "low", 5)
        p_med = Package("P_med", "L1", 480, 550, 485, "medium", 5)
        p_high = Package("P_high", "L1", 480, 550, 485, "high", 5)

        problem = _build_test_problem([p_low, p_med, p_high], alpha=1.0)
        r_low = apply_transition(problem, problem.initial_state, "P_low")
        r_med = apply_transition(problem, problem.initial_state, "P_med")
        r_high = apply_transition(problem, problem.initial_state, "P_high")

        assert r_low is not None and r_med is not None and r_high is not None
        self.assertEqual(r_low.tardiness, 10)
        self.assertEqual(r_med.tardiness, 10)
        self.assertEqual(r_high.tardiness, 10)

        self.assertEqual(r_low.penalty, 10.0)
        self.assertEqual(r_med.penalty, 20.0)
        self.assertEqual(r_high.penalty, 30.0)
        self.assertTrue(r_low.incremental_cost < r_med.incremental_cost < r_high.incremental_cost)

    def test_different_alpha_values_and_negative_alpha_rejection(self) -> None:
        # Finaliza 495, promesa 485 -> retraso 10, prioridad high (3) -> weighted_tardiness = 30
        pkg = Package("P1", "L1", 480, 550, 485, "high", 5)
        problem = _build_test_problem([pkg], alpha=1.0)

        r_zero = apply_transition(problem, problem.initial_state, "P1", alpha=0.0)
        r_half = apply_transition(problem, problem.initial_state, "P1", alpha=0.5)
        r_one = apply_transition(problem, problem.initial_state, "P1", alpha=1.0)
        r_two = apply_transition(problem, problem.initial_state, "P1", alpha=2.0)

        assert r_zero is not None and r_half is not None and r_one is not None and r_two is not None
        self.assertAlmostEqual(r_zero.incremental_cost, 15.0)
        self.assertAlmostEqual(r_half.incremental_cost, 15.0 + 0.5 * 30.0)
        self.assertAlmostEqual(r_one.incremental_cost, 15.0 + 1.0 * 30.0)
        self.assertAlmostEqual(r_two.incremental_cost, 15.0 + 2.0 * 30.0)

        with self.assertRaises(ValueError):
            apply_transition(problem, problem.initial_state, "P1", alpha=-0.1)

    def test_two_packages_at_same_location_account_both_service_times(self) -> None:
        # Dos paquetes en L1: P1 (servicio 5) y P2 (servicio 7)
        p1 = Package("P1", "L1", 480, 600, 500, "high", 5)
        p2 = Package("P2", "L1", 480, 600, 500, "medium", 7)
        problem = _build_test_problem([p1, p2])

        steps = simulate_route(problem, ["P1", "P2"])
        self.assertIsNotNone(steps)
        assert steps is not None
        self.assertEqual(len(steps), 2)

        step1, step2 = steps
        # Paso 1: D0 -> L1 (traslado 10, servicio 5, termina 495)
        self.assertEqual(step1.travel_time, 10)
        self.assertEqual(step1.service_time, 5)
        self.assertEqual(step1.completion_time, 495)

        # Paso 2: L1 -> L1 (traslado 0, servicio 7, inicia 495, termina 502)
        self.assertEqual(step2.origin, "L1")
        self.assertEqual(step2.destination, "L1")
        self.assertEqual(step2.travel_time, 0)
        self.assertEqual(step2.service_time, 7)
        self.assertEqual(step2.completion_time, 502)
        self.assertTrue(step2.next_state.is_goal)

        summary = summarize_route_costs(steps, alpha=problem.alpha)
        self.assertEqual(summary.total_travel_time, 10)
        self.assertEqual(summary.total_wait_time, 0)
        self.assertEqual(summary.total_service_time, 12)
        self.assertEqual(summary.total_elapsed_time, 22)

    def test_asymmetric_travel_times_and_unavailable_edges(self) -> None:
        # En la matriz de prueba:
        # L1 -> L2 cuesta 8 min, pero L2 -> L1 cuesta 18 min.
        # D0 -> L3 es None (inviable directamente desde D0).
        p1 = Package("P1", "L1", 480, 600, 550, "low", 5)
        p2 = Package("P2", "L2", 480, 600, 550, "low", 5)
        p3 = Package("P3", "L3", 480, 600, 550, "low", 5)
        problem = _build_test_problem([p1, p2, p3])

        # D0 -> L3 directo no está disponible
        self.assertIsNone(apply_transition(problem, problem.initial_state, "P3"))

        # Ruta P1 (L1) -> P2 (L2): L1->L2 toma 8 min
        route_12 = simulate_route(problem, ["P1", "P2"])
        # Ruta P2 (L2) -> P1 (L1): L2->L1 toma 18 min
        route_21 = simulate_route(problem, ["P2", "P1"])

        assert route_12 is not None and route_21 is not None
        self.assertEqual(route_12[1].travel_time, 8)
        self.assertEqual(route_21[1].travel_time, 18)
        self.assertNotEqual(route_12[1].travel_time, route_21[1].travel_time)

    def test_route_cost_coherence_with_readme_equations(self) -> None:
        # Verificar g(S_goal) = sum(c(S, p_i)) == T(pi) + alpha * D_p(pi)
        p1 = Package("P1", "L1", 500, 600, 502, "high", 5)   # llega 490, espera 10, termina 505 -> retraso 3, w=3
        p2 = Package("P2", "L2", 500, 600, 515, "medium", 5)  # sale 505, traslado 8 -> llega 513, termina 518 -> retraso 3, w=2
        problem = _build_test_problem([p1, p2], alpha=1.5)

        steps = simulate_route(problem, ["P1", "P2"])
        assert steps is not None
        summary = summarize_route_costs(steps, alpha=problem.alpha)

        # T(pi) = T_desplazamiento (10+8=18) + T_espera (10+0=10) + T_servicio (5+5=10) = 38
        self.assertEqual(
            summary.total_elapsed_time,
            summary.total_travel_time + summary.total_wait_time + summary.total_service_time,
        )
        self.assertEqual(summary.total_elapsed_time, steps[-1].completion_time - problem.start_time)

        # D_p(pi) = 3*3 + 2*3 = 15
        self.assertAlmostEqual(summary.weighted_tardiness, 15.0)
        # C(pi) = 38 + 1.5 * 15 = 60.5
        self.assertAlmostEqual(summary.objective_cost, 60.5)
        self.assertAlmostEqual(
            sum(step.incremental_cost for step in steps),
            summary.objective_cost,
        )


if __name__ == "__main__":
    unittest.main()

