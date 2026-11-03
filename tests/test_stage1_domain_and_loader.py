"""Pruebas unitarias de la Etapa 1: dominio, estado, problema, tiempo y cargador JSON."""

from __future__ import annotations

import copy
import heapq
import json
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from delivery_agent.domain.location import Location, validate_travel_matrix
from delivery_agent.domain.package import DEFAULT_PRIORITY_WEIGHTS, Package
from delivery_agent.domain.problem import DeliveryProblem
from delivery_agent.domain.state import DeliveryState, SearchState
from delivery_agent.scenarios.loader import (
    format_minutes_to_hhmm,
    load_problem_from_dict,
    load_problem_from_json,
    parse_time_to_minutes,
)


class TestTimeConversion(unittest.TestCase):
    """Pruebas para la conversión entre HH:MM y minutos desde las 00:00."""

    def test_readme_reference_examples(self) -> None:
        self.assertEqual(parse_time_to_minutes("08:00"), 480)
        self.assertEqual(parse_time_to_minutes("08:30"), 510)
        self.assertEqual(parse_time_to_minutes("09:00"), 540)
        self.assertEqual(parse_time_to_minutes("12:00"), 720)

    def test_boundary_times_and_roundtrip(self) -> None:
        self.assertEqual(parse_time_to_minutes("00:00"), 0)
        self.assertEqual(parse_time_to_minutes("23:59"), 1439)
        self.assertEqual(format_minutes_to_hhmm(480), "08:00")
        self.assertEqual(format_minutes_to_hhmm(510), "08:30")
        self.assertEqual(format_minutes_to_hhmm(720), "12:00")

    def test_invalid_time_formats_raise_value_error(self) -> None:
        for invalid in ("8:00", "24:00", "08:60", "-01:00", "ab:cd", "", "0800", -10, 1500):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    parse_time_to_minutes(invalid)

        with self.assertRaises(ValueError):
            parse_time_to_minutes(True)

        with self.assertRaises(ValueError):
            format_minutes_to_hhmm(-5)


class TestLocationAndTravelMatrix(unittest.TestCase):
    """Pruebas del modelo de ubicaciones y matriz dirigida de traslados."""

    def test_location_entity_validation(self) -> None:
        loc = Location(id="D0", name="Depósito central")
        self.assertEqual(loc.id, "D0")
        with self.assertRaises(ValueError):
            Location(id="   ")

    def test_asymmetric_matrix_and_unavailable_routes_allowed_and_immutable(self) -> None:
        locations = ["D0", "L1", "L2"]
        matrix = {
            "D0": {"D0": 0, "L1": 10, "L2": None},
            "L1": {"D0": 15, "L1": 0, "L2": 7},
            "L2": {"D0": 8, "L1": 12, "L2": 0},
        }
        validated = validate_travel_matrix(locations, matrix)
        self.assertEqual(validated["D0"]["L1"], 10)
        self.assertEqual(validated["L1"]["D0"], 15)
        self.assertIsNone(validated["D0"]["L2"])

        with self.assertRaises(TypeError):
            validated["D0"]["L1"] = 999  # type: ignore[index]

    def test_negative_travel_time_or_non_zero_diagonal_rejected(self) -> None:
        locations = ["D0", "L1"]
        with self.assertRaises(ValueError):
            validate_travel_matrix(
                locations,
                {
                    "D0": {"D0": 0, "L1": -5},
                    "L1": {"D0": 10, "L1": 0},
                },
            )

        with self.assertRaises(ValueError):
            validate_travel_matrix(
                locations,
                {
                    "D0": {"D0": 3, "L1": 10},
                    "L1": {"D0": 10, "L1": 0},
                },
            )


class TestPackageModel(unittest.TestCase):
    """Pruebas del modelo de paquetes, ventanas, promesas y prioridades."""

    def test_valid_package_and_tardiness(self) -> None:
        pkg = Package(
            id="P1",
            destination="L1",
            window_start=500,
            window_end=600,
            promised_time=540,
            priority="high",
            service_time=5,
        )
        self.assertEqual(pkg.compute_tardiness(530), 0)
        self.assertEqual(pkg.compute_tardiness(540), 0)
        self.assertEqual(pkg.compute_tardiness(555), 15)
        self.assertEqual(pkg.resolve_priority_weight(DEFAULT_PRIORITY_WEIGHTS), 3.0)

    def test_promised_time_outside_window_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Package(
                id="P_early_promise",
                destination="L1",
                window_start=500,
                window_end=600,
                promised_time=490,
                priority="medium",
                service_time=5,
            )

        with self.assertRaises(ValueError):
            Package(
                id="P_late_promise",
                destination="L1",
                window_start=500,
                window_end=600,
                promised_time=610,
                priority="medium",
                service_time=5,
            )

    def test_window_shorter_than_service_time_or_negative_times_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Package(
                id="P_impossible",
                destination="L1",
                window_start=500,
                window_end=503,
                promised_time=502,
                priority="low",
                service_time=5,
            )

        with self.assertRaises(ValueError):
            Package(
                id="P_neg_service",
                destination="L1",
                window_start=500,
                window_end=600,
                promised_time=540,
                priority="low",
                service_time=-1,
            )


class TestDeliveryState(unittest.TestCase):
    """Pruebas de la representación, identidad, inmutabilidad y comparación de DeliveryState."""

    def test_state_identity_immutability_and_alias(self) -> None:
        self.assertIs(DeliveryState, SearchState)
        s1 = DeliveryState.create("L1", 510, ["P2", "P1"])
        s2 = DeliveryState.create("L1", 510, ["P1", "P2"])
        s_diff_time = DeliveryState.create("L1", 515, ["P1", "P2"])

        self.assertEqual(s1, s2)
        self.assertEqual(hash(s1), hash(s2))
        self.assertNotEqual(s1, s_diff_time)
        self.assertFalse(s1.is_goal)
        self.assertEqual(s1.sorted_pending_ids(), ("P1", "P2"))

        best_g: dict[DeliveryState, float] = {s1: 42.0}
        self.assertIn(s2, best_g)
        self.assertNotIn(s_diff_time, best_g)

    def test_total_ordering_with_disjoint_pending_sets_for_heapq(self) -> None:
        # Dos estados con misma ubicación y hora pero conjuntos pendientes disjuntos:
        # En frozenset estándar, {"P1"} < {"P2"} es False y {"P2"} < {"P1"} es False.
        # DeliveryState debe ofrecer un orden total estricto y determinista.
        state_a = DeliveryState.create("L1", 500, ["P2"])
        state_b = DeliveryState.create("L1", 500, ["P1"])

        self.assertTrue(state_b < state_a)
        self.assertFalse(state_a < state_b)

        heap: list[tuple[float, float, DeliveryState]] = []
        heapq.heappush(heap, (100.0, 50.0, state_a))
        heapq.heappush(heap, (100.0, 50.0, state_b))

        first = heapq.heappop(heap)[2]
        second = heapq.heappop(heap)[2]
        self.assertEqual(first, state_b)
        self.assertEqual(second, state_a)

    def test_goal_state_when_pending_is_empty(self) -> None:
        goal = DeliveryState.create("L5", 600, [])
        self.assertTrue(goal.is_goal)


class TestProblemAndLoader(unittest.TestCase):
    """Pruebas del modelo general del problema, función de costo y cargador JSON."""

    def setUp(self) -> None:
        self.scenario_path = (
            PROJECT_ROOT / "data" / "scenarios" / "escenario_base_5_paquetes.json"
        )
        with self.scenario_path.open("r", encoding="utf-8") as f:
            self.raw_data = json.load(f)

    def test_load_base_scenario_from_json_and_immutability(self) -> None:
        problem = load_problem_from_json(self.scenario_path, enforce_package_range=True)
        self.assertEqual(problem.name, "escenario_base_5_paquetes")
        self.assertEqual(problem.start_location, "D0")
        self.assertEqual(problem.start_time, 480)
        self.assertEqual(problem.alpha, 1.0)
        self.assertEqual(len(problem.packages), 5)
        self.assertEqual(problem.travel_time("D0", "L1"), 10)

        initial = problem.initial_state
        self.assertEqual(initial.location, "D0")
        self.assertEqual(initial.current_time, 480)
        self.assertEqual(initial.pending_packages, frozenset({"P1", "P2", "P3", "P4", "P5"}))
        self.assertFalse(problem.is_goal(initial))

        with self.assertRaises(TypeError):
            problem.priority_weights["high"] = 99  # type: ignore[index]
        with self.assertRaises(TypeError):
            problem.travel_times["D0"]["L1"] = 99  # type: ignore[index]

    def test_multiple_packages_at_same_location_supported(self) -> None:
        raw = copy.deepcopy(self.raw_data)
        # Hacer que P2 también tenga destino L1 con su propia ventana y servicio
        raw["packages"][1]["destination"] = "L1"
        raw["packages"][1]["service_time"] = 8
        problem = load_problem_from_dict(raw)

        l1_packages = problem.packages_at_location("L1")
        self.assertEqual(tuple(p.id for p in l1_packages), ("P1", "P2"))
        self.assertEqual(l1_packages[0].service_time, 5)
        self.assertEqual(l1_packages[1].service_time, 8)
        self.assertEqual(len(problem.initial_state.pending_packages), 5)

    def test_cost_function_considers_elapsed_time_promise_priority_and_alpha(self) -> None:
        problem = load_problem_from_json(self.scenario_path)
        cost_high = problem.transition_cost(elapsed_time=20, package="P1", completion_time=550)
        cost_low = problem.transition_cost(elapsed_time=20, package="P3", completion_time=580)

        self.assertAlmostEqual(cost_high, 50.0)
        self.assertAlmostEqual(cost_low, 30.0)
        self.assertGreater(cost_high, cost_low)
        self.assertAlmostEqual(
            problem.objective_cost(total_elapsed_time=120, weighted_tardiness=35.0),
            155.0,
        )

    def test_alpha_validation(self) -> None:
        raw_zero = copy.deepcopy(self.raw_data)
        raw_zero["alpha"] = 0.0
        with self.assertRaises(ValueError):
            load_problem_from_dict(raw_zero, allow_zero_alpha=False)

        control_problem = load_problem_from_dict(raw_zero, allow_zero_alpha=True)
        self.assertEqual(control_problem.alpha, 0.0)

    def test_loader_detects_missing_fields_duplicates_invalid_windows_and_negative_times(self) -> None:
        # 1. Campo obligatorio faltante a nivel escenario
        missing_start = copy.deepcopy(self.raw_data)
        del missing_start["start_location"]
        with self.assertRaises(ValueError):
            load_problem_from_dict(missing_start)

        # 2. Campo obligatorio faltante en paquete
        missing_pkg_field = copy.deepcopy(self.raw_data)
        del missing_pkg_field["packages"][0]["promised_time"]
        with self.assertRaises(ValueError):
            load_problem_from_dict(missing_pkg_field)

        # 3. IDs de paquetes duplicados
        dup_pkg = copy.deepcopy(self.raw_data)
        dup_pkg["packages"][1]["id"] = "P1"
        with self.assertRaises(ValueError):
            load_problem_from_dict(dup_pkg)

        # 4. IDs de ubicaciones duplicados
        dup_loc = copy.deepcopy(self.raw_data)
        dup_loc["locations"].append("L1")
        with self.assertRaises(ValueError):
            load_problem_from_dict(dup_loc)

        # 5. Ventana que cierra antes de la hora de salida del repartidor
        window_before_start = copy.deepcopy(self.raw_data)
        window_before_start["packages"][0]["window_start"] = "07:00"
        window_before_start["packages"][0]["promised_time"] = "07:30"
        window_before_start["packages"][0]["window_end"] = "07:50"
        with self.assertRaises(ValueError):
            load_problem_from_dict(window_before_start)

        # 6. Tiempo de traslado negativo en matriz JSON
        neg_travel = copy.deepcopy(self.raw_data)
        neg_travel["travel_times"]["D0"]["L1"] = -3
        with self.assertRaises(ValueError):
            load_problem_from_dict(neg_travel)

    def test_loader_rejects_malformed_json_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_file = Path(tmp_dir) / "bad.json"
            bad_file.write_text("{ invalid json ", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_problem_from_json(bad_file)


if __name__ == "__main__":
    unittest.main()
