"""Suite de pruebas unitarias para la lógica del gestor de escenarios de la GUI (Etapa 6.5)."""

from __future__ import annotations

import copy
import os
from pathlib import Path
import tempfile
import unittest

from delivery_agent.algorithms.astar import solve_astar
from delivery_agent.domain.problem import DeliveryProblem
from delivery_agent.gui.scenario_manager import ScenarioManager, format_minutes_to_hhmm


class TestStage65ScenarioManagerLogic(unittest.TestCase):
    """Pruebas de la capa lógica y de estado desacoplada de la interfaz gráfica."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.manager = ScenarioManager()
        # Redirigir carpeta de personalizados a la temporal para pruebas
        self.manager.CUSTOM_SCENARIOS_DIR = Path(self.temp_dir.name) / "custom"
        self.manager.CUSTOM_SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_01_create_manual_scenario(self) -> None:
        """Verifica la creación de un escenario manual desde cero con valores válidos."""
        self.manager.new_scenario(name="escenario_prueba_manual")
        self.assertEqual(self.manager.name, "escenario_prueba_manual")
        self.assertIn(self.manager.start_location, self.manager.locations)
        self.assertEqual(self.manager.start_time, 480)
        self.assertEqual(len(self.manager.packages), 5)
        self.assertEqual(self.manager.validate(), [])

    def test_02_edit_initial_state(self) -> None:
        """Verifica la modificación del punto y hora de salida del vehículo."""
        self.manager.start_location = "L2"
        self.manager.start_time = 510  # 08:30

        # Para que los paquetes sigan siendo válidos, ajustar ventanas para no cerrar antes de 510
        for p in self.manager.packages:
            p["window_start"] = max(p["window_start"], 510)
            p["window_end"] = max(p["window_end"], 650)
            p["promised_time"] = max(p["promised_time"], 510)

        self.assertEqual(self.manager.validate(), [])
        problem = self.manager.to_delivery_problem()
        self.assertEqual(problem.start_location, "L2")
        self.assertEqual(problem.start_time, 510)

    def test_03_add_and_remove_packages(self) -> None:
        """Verifica la adición y eliminación dinámica de paquetes."""
        init_count = len(self.manager.packages)
        new_pkg = {
            "id": "P_EXTRA",
            "destination": "L1",
            "window_start": 480,
            "window_end": 700,
            "promised_time": 600,
            "priority": "alta",
            "service_time": 8,
        }
        self.manager.add_package(new_pkg)
        self.assertEqual(len(self.manager.packages), init_count + 1)
        self.assertTrue(any(p["id"] == "P_EXTRA" for p in self.manager.packages))

        # Eliminar
        self.manager.remove_package("P_EXTRA")
        self.assertEqual(len(self.manager.packages), init_count)
        self.assertFalse(any(p["id"] == "P_EXTRA" for p in self.manager.packages))

    def test_04_modify_priorities_and_weights(self) -> None:
        """Verifica la edición de pesos numéricos de prioridad y rechazo de pesos negativos."""
        self.manager.priority_weights["baja"] = 1.5
        self.manager.priority_weights["media"] = 3.5
        self.manager.priority_weights["alta"] = 7.0
        self.assertEqual(self.manager.validate(), [])

        # Peso inválido
        self.manager.priority_weights["alta"] = -1.0
        errors = self.manager.validate()
        self.assertTrue(any("alta" in err and "> 0" in err for err in errors))

    def test_05_modify_alpha(self) -> None:
        """Verifica la modificación de alpha y rechazo de valores negativos."""
        self.manager.alpha = 2.5
        self.assertEqual(self.manager.validate(), [])

        self.manager.alpha = 0.0  # Permitido en experimentos de control
        self.assertEqual(self.manager.validate(), [])

        self.manager.alpha = -0.5
        errors = self.manager.validate()
        self.assertTrue(any("alpha" in err for err in errors))

    def test_06_edit_directed_matrix_and_asymmetry(self) -> None:
        """Verifica la edición de la matriz de traslados soportando asimetría y conexiones None."""
        self.manager.set_travel_time("L1", "L2", 15)
        self.manager.set_travel_time("L2", "L1", 25)  # Asimétrico
        self.manager.set_travel_time("L3", "L4", None)  # Ruta inexistente

        self.assertEqual(self.manager.travel_times["L1"]["L2"], 15)
        self.assertEqual(self.manager.travel_times["L2"]["L1"], 25)
        self.assertIsNone(self.manager.travel_times["L3"]["L4"])

        # Diagonal siempre debe ser 0
        self.assertEqual(self.manager.travel_times["L1"]["L1"], 0)

    def test_07_multiple_packages_same_destination(self) -> None:
        """Verifica que se permitan múltiples paquetes asignados a la misma ubicación."""
        self.manager.add_package({
            "id": "P_COMPARTIDO",
            "destination": "L1",  # P1 también va a L1
            "window_start": 480,
            "window_end": 700,
            "promised_time": 600,
            "priority": "media",
            "service_time": 10,
        })
        self.assertEqual(self.manager.validate(), [])
        problem = self.manager.to_delivery_problem()
        pkgs_at_l1 = problem.packages_at_location("L1")
        self.assertEqual(len(pkgs_at_l1), 2)
        pkg_ids = {p.id for p in pkgs_at_l1}
        self.assertIn("P1", pkg_ids)
        self.assertIn("P_COMPARTIDO", pkg_ids)

    def test_08_detect_invalid_data(self) -> None:
        """Verifica la detección granular de violaciones de validación."""
        # 1. Identificador de paquete duplicado
        invalid_manager = copy.deepcopy(self.manager)
        invalid_manager.packages.append(copy.deepcopy(invalid_manager.packages[0]))
        self.assertTrue(any("duplicado" in e for e in invalid_manager.validate()))

        # 2. Destino inexistente
        invalid_manager2 = copy.deepcopy(self.manager)
        invalid_manager2.packages[0]["destination"] = "L_FANTASMA"
        self.assertTrue(any("no existe" in e for e in invalid_manager2.validate()))

        # 3. Fin de ventana anterior al inicio
        invalid_manager3 = copy.deepcopy(self.manager)
        invalid_manager3.packages[0]["window_end"] = invalid_manager3.packages[0]["window_start"] - 10
        self.assertTrue(any("anterior al inicio" in e for e in invalid_manager3.validate()))

        # 4. Hora prometida fuera de ventana
        invalid_manager4 = copy.deepcopy(self.manager)
        invalid_manager4.packages[0]["promised_time"] = invalid_manager4.packages[0]["window_end"] + 15
        self.assertTrue(any("fuera de la ventana" in e for e in invalid_manager4.validate()))

    def test_09_save_and_load_custom_json(self) -> None:
        """Verifica que el guardado y posterior carga conserven exactamente todas las propiedades."""
        self.manager.name = "mi_escenario_personalizado"
        self.manager.alpha = 1.75
        saved_file = self.manager.save_as_custom("test_custom.json")

        self.assertTrue(saved_file.exists())
        self.assertTrue(saved_file.is_file())

        new_mgr = ScenarioManager()
        new_mgr.load_from_json(saved_file)

        self.assertEqual(new_mgr.name, "mi_escenario_personalizado")
        self.assertEqual(new_mgr.alpha, 1.75)
        self.assertEqual(len(new_mgr.packages), len(self.manager.packages))
        self.assertEqual(new_mgr.locations, self.manager.locations)
        self.assertEqual(new_mgr.travel_times, self.manager.travel_times)

    def test_10_execute_astar_on_custom_scenario(self) -> None:
        """Verifica la resolución con A* sobre un escenario configurado manualmente."""
        res = self.manager.run_algorithm("A*", alpha=1.0)
        self.assertTrue(res.is_feasible)
        self.assertEqual(len(res.transitions), len(self.manager.packages))
        self.assertIsNotNone(res.total_cost)

    def test_11_compare_results_with_direct_algorithm_execution(self) -> None:
        """Comprueba que la ejecución vía GUI produzca exactamente el mismo resultado que la llamada directa."""
        # 1. Ejecución vía manager
        mgr_result = self.manager.run_algorithm("A*", alpha=1.0)

        # 2. Ejecución directa sobre DeliveryProblem
        problem = self.manager.to_delivery_problem()
        direct_result = solve_astar(problem, alpha=1.0)

        self.assertEqual(mgr_result.total_cost, direct_result.total_cost)
        self.assertEqual(mgr_result.package_order, direct_result.package_order)
        self.assertEqual(mgr_result.arrival_times, direct_result.arrival_times)
        self.assertEqual(mgr_result.metrics.states_expanded, direct_result.metrics.states_expanded)

    def test_12_custom_scenarios_do_not_alter_official_benchmark(self) -> None:
        """Comprueba que guardar escenarios personalizados nunca altere data/scenarios/ ni summary.csv."""
        official_files_before = set(self.manager.list_official_scenarios())
        summary_path = Path("results/benchmarks/summary.csv")
        summary_mtime_before = summary_path.stat().st_mtime if summary_path.exists() else None

        # Guardar varios personalizados
        self.manager.save_as_custom("custom_1.json")
        self.manager.save_as_custom("escenario_01.json")  # Debe prefijar custom_ para no colisionar

        official_files_after = set(self.manager.list_official_scenarios())
        self.assertEqual(official_files_before, official_files_after)

        if summary_path.exists():
            summary_mtime_after = summary_path.stat().st_mtime
            self.assertEqual(summary_mtime_before, summary_mtime_after)


if __name__ == "__main__":
    unittest.main()

