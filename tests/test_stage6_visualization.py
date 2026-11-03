"""Suite de pruebas unitarias para el sistema de visualización (Etapa 6)."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

# Forzar backend no interactivo antes de cargar matplotlib
os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp()
import matplotlib
matplotlib.use("Agg")

from PIL import Image

from delivery_agent.algorithms.astar import solve_astar
from delivery_agent.algorithms.dijkstra import solve_dijkstra
from delivery_agent.domain.package import Package
from delivery_agent.domain.problem import DeliveryProblem
from delivery_agent.scenarios.loader import load_problem_from_json
from delivery_agent.visualization.benchmark_visualizer import (
    generate_all_benchmark_charts,
    load_benchmark_data,
    plot_cost_sensitivity_alpha,
    plot_elapsed_time_sensitivity_alpha,
    plot_execution_time_comparison,
    plot_greedy_suboptimality_gap,
    plot_max_queue_size_comparison,
    plot_objective_cost_comparison,
    plot_states_expanded_comparison,
    plot_weighted_tardiness_sensitivity_alpha,
)
from delivery_agent.visualization.route_visualizer import (
    animate_route,
    compute_deterministic_layout,
    render_route_static,
)
from delivery_agent.visualization.search_visualizer import (
    animate_search_process,
    render_search_tree_static,
)


class TestStage6Visualization(unittest.TestCase):
    """Pruebas de visualización de rutas, procesos de búsqueda y gráficas experimentales."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.test_dir = Path(cls.temp_dir.name)

        # Cargar escenario base 1
        scenario_path = Path("data/scenarios/escenario_01_base_5_paquetes.json")
        cls.problem = load_problem_from_json(scenario_path)

        # Resolver con A* registrando eventos
        cls.astar_result = solve_astar(cls.problem, alpha=1.0, track_events=True)

        # Cargar escenario infactible 8
        infeasible_path = Path("data/scenarios/escenario_08_sin_solucion_5_paquetes.json")
        cls.infeasible_problem = load_problem_from_json(infeasible_path)
        cls.infeasible_result = solve_astar(cls.infeasible_problem, alpha=1.0, track_events=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def test_01_deterministic_layout_generation(self) -> None:
        """Verifica que el cálculo de posiciones 2D sea determinista y cubra todas las ubicaciones."""
        pos1 = compute_deterministic_layout(self.problem)
        pos2 = compute_deterministic_layout(self.problem)

        self.assertEqual(pos1, pos2)
        self.assertEqual(set(pos1.keys()), set(self.problem.locations))
        # El origen debe estar en (0.0, 0.0)
        self.assertEqual(pos1[self.problem.start_location], (0.0, 0.0))

    def test_02_render_route_static_feasible(self) -> None:
        """Verifica la generación correcta del PNG estático de la ruta factible."""
        out_png = self.test_dir / "static_route_feasible.png"
        res_path = render_route_static(self.problem, self.astar_result, out_png)

        self.assertTrue(res_path.exists())
        self.assertGreater(res_path.stat().st_size, 5000)

        with Image.open(res_path) as img:
            self.assertEqual(img.format, "PNG")
            self.assertGreater(img.width, 1000)
            self.assertGreater(img.height, 500)

    def test_03_render_route_static_infeasible(self) -> None:
        """Verifica que un escenario infactible genere correctamente un PNG descriptivo."""
        out_png = self.test_dir / "static_route_infeasible.png"
        res_path = render_route_static(self.infeasible_problem, self.infeasible_result, out_png)

        self.assertTrue(res_path.exists())
        self.assertGreater(res_path.stat().st_size, 5000)

    def test_04_animate_route_gif_creation(self) -> None:
        """Verifica que la animación del recorrido se compile en un archivo GIF animado."""
        out_gif = self.test_dir / "animated_route.gif"
        res_path = animate_route(self.problem, self.astar_result, out_gif, fps=2)

        self.assertTrue(res_path.exists())
        self.assertGreater(res_path.stat().st_size, 10000)

        with Image.open(res_path) as img:
            self.assertEqual(img.format, "GIF")
            self.assertTrue(getattr(img, "is_animated", False))
            self.assertGreater(img.n_frames, 5)

    def test_05_render_search_tree_static(self) -> None:
        """Verifica que el diagrama estático del árbol de búsqueda se genere adecuadamente."""
        out_png = self.test_dir / "static_search_tree.png"
        res_path = render_search_tree_static(self.problem, self.astar_result, out_png)

        self.assertTrue(res_path.exists())
        self.assertGreater(res_path.stat().st_size, 10000)

        with Image.open(res_path) as img:
            self.assertEqual(img.format, "PNG")

    def test_06_animate_search_process_gif(self) -> None:
        """Verifica la animación GIF del proceso de exploración y toma de decisiones de A*."""
        out_gif = self.test_dir / "animated_search.gif"
        res_path = animate_search_process(self.problem, self.astar_result, out_gif, fps=4, max_frames=20)

        self.assertTrue(res_path.exists())
        self.assertGreater(res_path.stat().st_size, 10000)

        with Image.open(res_path) as img:
            self.assertEqual(img.format, "GIF")
            self.assertTrue(getattr(img, "is_animated", False))
            self.assertGreater(img.n_frames, 3)

    def test_07_track_events_consistency_with_standard_astar(self) -> None:
        """Comprueba que track_events=True no altere el costo óptimo ni el comportamiento de A*."""
        result_fast = solve_astar(self.problem, alpha=1.0, track_events=False)
        self.assertIsNone(result_fast.events)

        self.assertEqual(self.astar_result.total_cost, result_fast.total_cost)
        self.assertEqual(self.astar_result.package_order, result_fast.package_order)
        self.assertEqual(self.astar_result.metrics.states_expanded, result_fast.metrics.states_expanded)
        self.assertIsNotNone(self.astar_result.events)
        self.assertGreater(len(self.astar_result.events), 0)

    def test_08_load_benchmark_data(self) -> None:
        """Verifica la carga del archivo summary.csv y details.json."""
        summary_path = Path("results/benchmarks/summary.csv")
        details_path = Path("results/benchmarks/details.json")

        self.assertTrue(summary_path.exists(), "Debe existir summary.csv para ejecutar el test.")
        data = load_benchmark_data(summary_path, details_path)

        self.assertEqual(len(data), 128)
        first_row = data[0]
        self.assertIn("scenario_name", first_row)
        self.assertIn("algorithm", first_row)
        self.assertIn("alpha", first_row)
        self.assertIn("objective_cost", first_row)
        self.assertIn("max_queue_size", first_row)

    def test_09_generate_all_benchmark_charts(self) -> None:
        """Verifica la generación de la suite completa de las 8 gráficas experimentales."""
        summary_path = Path("results/benchmarks/summary.csv")
        details_path = Path("results/benchmarks/details.json")
        out_charts_dir = self.test_dir / "charts"

        generated = generate_all_benchmark_charts(
            summary_csv_path=summary_path,
            output_dir=out_charts_dir,
            details_json_path=details_path,
        )

        self.assertEqual(len(generated), 8)
        expected_names = [
            "01_objective_cost_comparison.png",
            "02_execution_time_comparison.png",
            "03_states_expanded_comparison.png",
            "04_max_queue_size_comparison.png",
            "05_cost_sensitivity_alpha.png",
            "06_elapsed_time_sensitivity_alpha.png",
            "07_weighted_tardiness_sensitivity_alpha.png",
            "08_greedy_suboptimality_gap.png",
        ]
        actual_names = [p.name for p in generated]
        self.assertEqual(actual_names, expected_names)

        for p in generated:
            self.assertTrue(p.exists())
            self.assertGreater(p.stat().st_size, 10000)


if __name__ == "__main__":
    unittest.main()

