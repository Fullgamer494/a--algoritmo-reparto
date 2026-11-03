"""Pruebas de la capa visual y controles de la interfaz gráfica interactiva (Etapa 6.5)."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import tkinter as tk
import unittest

# Matplotlib backend TkAgg
os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp()
import matplotlib
matplotlib.use("TkAgg")

from delivery_agent.gui.app import DeliveryAgentApp


class TestStage65GuiPresentation(unittest.TestCase):
    """Pruebas de los widgets, eventos y pestañas de la interfaz gráfica de usuario."""

    def setUp(self) -> None:
        self.root = tk.Tk()
        self.root.withdraw()  # Ocultar la ventana para ejecución desatendida
        self.app = DeliveryAgentApp(self.root)

    def tearDown(self) -> None:
        try:
            self.root.destroy()
        except Exception:
            pass

    def test_01_gui_initialization_and_tabs(self) -> None:
        """Comprueba que la ventana y las cuatro pestañas principales se inicialicen correctamente."""
        self.assertIsNotNone(self.app.notebook)
        tab_names = [self.app.notebook.tab(i, "text").strip() for i in range(4)]
        self.assertEqual(len(tab_names), 4)
        self.assertTrue(any("Configuración" in t for t in tab_names))
        self.assertTrue(any("Paquetes" in t for t in tab_names))
        self.assertTrue(any("Ubicaciones" in t for t in tab_names))
        self.assertTrue(any("Ejecución" in t for t in tab_names))

    def test_02_tab_switching(self) -> None:
        """Verifica la navegación entre pestañas del Notebook."""
        for i in range(4):
            self.app.notebook.select(i)
            self.root.update_idletasks()
            self.assertEqual(self.app.notebook.index(self.app.notebook.select()), i)

    def test_03_load_official_scenario_updates_ui(self) -> None:
        """Verifica que al cargar un escenario oficial la interfaz refleje sus datos."""
        official_scenarios = self.app.manager.list_official_scenarios()
        self.assertGreater(len(official_scenarios), 0)

        # Cargar escenario 1
        scen_1 = official_scenarios[0]
        self.app.manager.load_from_json(scen_1)
        self.app.refresh_all_views()
        self.root.update_idletasks()

        self.assertEqual(self.app.scenario_name_var.get(), self.app.manager.name)
        self.assertEqual(len(self.app.pkg_tree.get_children()), len(self.app.manager.packages))
        self.assertEqual(self.app.loc_listbox.size(), len(self.app.manager.locations))

    def test_04_execute_astar_and_update_results_and_canvas(self) -> None:
        """Verifica que la ejecución de A* pueble el resumen, tabla de pasos y canvas embebido."""
        self.app.notebook.select(3)  # Pestaña D
        self.app.exec_alg_combo.set("A*")
        self.app.on_run_single_algorithm()
        self.root.update_idletasks()

        self.assertIsNotNone(self.app.last_result)
        self.assertTrue(self.app.last_result.is_feasible)

        # Verificar resumen en la caja de texto
        summary_content = self.app.result_summary_box.get("1.0", tk.END)
        self.assertIn("A*", summary_content)
        self.assertIn("Costo Objetivo", summary_content)

        # Verificar tabla de pasos
        steps_count = len(self.app.steps_tree.get_children())
        self.assertEqual(steps_count, len(self.app.last_result.transitions))

        # Verificar controles de reproducción
        self.assertGreater(len(self.app.playback_frames), 0)
        self.app.playback_next()
        self.app.playback_prev()
        self.app.playback_last()
        self.app.playback_first()

    def test_05_execute_comparison_updates_comparison_table(self) -> None:
        """Verifica que el botón de comparar los 4 algoritmos pueble la tabla comparativa."""
        self.app.notebook.select(3)
        self.app.on_run_comparison()
        self.root.update_idletasks()

        comp_rows = self.app.comp_tree.get_children()
        self.assertEqual(len(comp_rows), 4)


if __name__ == "__main__":
    unittest.main()

