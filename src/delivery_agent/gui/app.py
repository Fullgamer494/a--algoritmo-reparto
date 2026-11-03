"""Interfaz gráfica de usuario interactiva (Tkinter/ttk) para el agente de reparto."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp()

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt

from delivery_agent.algorithms.astar import SearchResult
from delivery_agent.core.transitions import TransitionResult
from delivery_agent.gui.scenario_manager import ScenarioManager, format_minutes_to_hhmm
from delivery_agent.scenarios.loader import parse_time_to_minutes
from delivery_agent.visualization.route_visualizer import (
    _render_single_animation_frame,
    compute_deterministic_layout,
    render_route_static,
)
from delivery_agent.visualization.search_visualizer import render_search_tree_static


class DeliveryAgentApp:
    """Ventana principal de la interfaz gráfica del agente de reparto."""

    def __init__(self, root: tk.Tk | None = None) -> None:
        self.root = root or tk.Tk()
        self.root.title("Agente de Planificación de Reparto Inteligente - Control de Escenarios (FIA)")
        self.root.geometry("1300x840")
        self.root.minsize(1080, 720)

        # Gestor de escenarios en memoria
        self.manager = ScenarioManager()

        # Resultados de ejecución
        self.last_result: SearchResult | None = None
        self.last_algorithm: str = "A*"
        self.comparison_results: dict[str, SearchResult] | None = None

        # Variables de reproducción paso a paso
        self.playback_frames: list[tuple[int, str, int, str, TransitionResult | None, set[str]]] = []
        self.current_frame_idx: int = 0
        self.is_playing: bool = False
        self.playback_timer: str | None = None

        self._setup_styles()
        self._build_top_toolbar()
        self._build_notebook()
        self._build_status_bar()

        # Cargar datos iniciales en la interfaz
        self.refresh_all_views()

    def _setup_styles(self) -> None:
        """Configura temas y estilos visuales consistentes con ttk."""
        self.style = ttk.Style(self.root)
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass

        # Colores de la interfaz
        self.style.configure(".", font=("Segoe UI", 10))
        self.style.configure("TNotebook", background="#0f172a")
        self.style.configure("TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=[14, 6])
        self.style.map("TNotebook.Tab", background=[("selected", "#1e293b"), ("!selected", "#0f172a")],
                                      foreground=[("selected", "#38bdf8"), ("!selected", "#94a3b8")])
        self.style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"), foreground="#0f172a")
        self.style.configure("Section.TLabelframe", padding=10)
        self.style.configure("Section.TLabelframe.Label", font=("Segoe UI", 10, "bold"), foreground="#1e293b")
        self.style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"))

    def _build_top_toolbar(self) -> None:
        """Construye la barra superior de acciones globales de gestión de escenarios."""
        toolbar = ttk.Frame(self.root, padding=6)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="+ Nuevo Escenario", command=self.on_new_scenario).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="📂 Cargar Oficial", command=self.on_load_official_dialog).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="📁 Cargar Personalizado", command=self.on_load_custom_dialog).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="💾 Guardar como Personalizado", command=self.on_save_as_custom).pack(side=tk.LEFT, padx=3)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Button(toolbar, text="📥 Importar JSON...", command=self.on_import_json).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="📤 Exportar JSON...", command=self.on_export_json).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="↺ Restablecer Valores", command=self.on_reset_scenario).pack(side=tk.LEFT, padx=3)

    def _build_notebook(self) -> None:
        """Crea el contenedor de pestañas principales (A, B, C, D)."""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # Tab A: Configuración General
        self.tab_config = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(self.tab_config, text=" A. Configuración General ")
        self._build_tab_config()

        # Tab B: Administración de Paquetes
        self.tab_packages = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(self.tab_packages, text=" B. Administración de Paquetes ")
        self._build_tab_packages()

        # Tab C: Ubicaciones y Matriz de Tiempos
        self.tab_locations = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(self.tab_locations, text=" C. Ubicaciones y Matriz de Tiempos ")
        self._build_tab_locations()

        # Tab D: Ejecución y Resultados
        self.tab_execution = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(self.tab_execution, text=" D. Ejecución y Resultados ")
        self._build_tab_execution()

    def _build_status_bar(self) -> None:
        """Barra de estado inferior."""
        self.status_bar = ttk.Frame(self.root, padding=4)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_label = ttk.Label(self.status_bar, text="Listo.", relief=tk.SUNKEN, anchor=tk.W, font=("Segoe UI", 9))
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.official_badge = ttk.Label(self.status_bar, text="", relief=tk.SUNKEN, font=("Segoe UI", 9, "bold"))
        self.official_badge.pack(side=tk.RIGHT, padx=4)

    def set_status(self, message: str, is_error: bool = False) -> None:
        """Actualiza el mensaje en la barra de estado."""
        self.status_label.config(text=message, foreground="#ef4444" if is_error else "#0f172a")

    # =========================================================================
    # TAB A: CONFIGURACIÓN GENERAL
    # =========================================================================

    def _build_tab_config(self) -> None:
        grid_frame = ttk.Frame(self.tab_config)
        grid_frame.pack(fill=tk.BOTH, expand=True)

        # Panel izquierdo: Parámetros del escenario
        left_box = ttk.LabelFrame(grid_frame, text=" Parámetros Generales del Problema ", style="Section.TLabelframe")
        left_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8), pady=4)

        row = 0
        ttk.Label(left_box, text="Escenario Seleccionado:").grid(row=row, column=0, sticky=tk.W, pady=6)
        self.scenario_combo = ttk.Combobox(left_box, state="readonly", width=42)
        self.scenario_combo.grid(row=row, column=1, sticky=tk.W, pady=6)
        self.scenario_combo.bind("<<ComboboxSelected>>", self.on_scenario_combo_selected)
        row += 1

        ttk.Label(left_box, text="Nombre del Escenario:").grid(row=row, column=0, sticky=tk.W, pady=6)
        self.scenario_name_var = tk.StringVar()
        ttk.Entry(left_box, textvariable=self.scenario_name_var, width=44).grid(row=row, column=1, sticky=tk.W, pady=6)
        row += 1

        ttk.Label(left_box, text="Ubicación Inicial (Depósito):").grid(row=row, column=0, sticky=tk.W, pady=6)
        self.start_loc_combo = ttk.Combobox(left_box, state="readonly", width=18)
        self.start_loc_combo.grid(row=row, column=1, sticky=tk.W, pady=6)
        row += 1

        ttk.Label(left_box, text="Hora de Partida (HH:MM):").grid(row=row, column=0, sticky=tk.W, pady=6)
        self.start_time_var = tk.StringVar()
        ttk.Entry(left_box, textvariable=self.start_time_var, width=18).grid(row=row, column=1, sticky=tk.W, pady=6)
        row += 1

        ttk.Label(left_box, text="Cantidad de Paquetes:").grid(row=row, column=0, sticky=tk.W, pady=6)
        self.pkg_count_label = ttk.Label(left_box, text="0 paquetes", font=("Segoe UI", 10, "bold"))
        self.pkg_count_label.grid(row=row, column=1, sticky=tk.W, pady=6)
        row += 1

        ttk.Label(left_box, text="Parámetro Alpha (α):").grid(row=row, column=0, sticky=tk.W, pady=6)
        self.alpha_var = tk.StringVar()
        ttk.Entry(left_box, textvariable=self.alpha_var, width=18).grid(row=row, column=1, sticky=tk.W, pady=6)
        row += 1

        # Pesos de prioridad
        prio_box = ttk.LabelFrame(left_box, text=" Pesos Numéricos de Prioridad (w_i) ", padding=8)
        prio_box.grid(row=row, column=0, columnspan=2, sticky=tk.EW, pady=12)

        ttk.Label(prio_box, text="Baja (low):").grid(row=0, column=0, padx=4, pady=4)
        self.prio_low_var = tk.StringVar()
        ttk.Entry(prio_box, textvariable=self.prio_low_var, width=10).grid(row=0, column=1, padx=4, pady=4)

        ttk.Label(prio_box, text="Media (medium):").grid(row=0, column=2, padx=4, pady=4)
        self.prio_med_var = tk.StringVar()
        ttk.Entry(prio_box, textvariable=self.prio_med_var, width=10).grid(row=0, column=3, padx=4, pady=4)

        ttk.Label(prio_box, text="Alta (high):").grid(row=0, column=4, padx=4, pady=4)
        self.prio_high_var = tk.StringVar()
        ttk.Entry(prio_box, textvariable=self.prio_high_var, width=10).grid(row=0, column=5, padx=4, pady=4)

        row += 1
        btn_apply_config = ttk.Button(left_box, text="Aplicar Cambios de Configuración", command=self.apply_tab_config)
        btn_apply_config.grid(row=row, column=0, columnspan=2, pady=10, sticky=tk.EW)

        # Panel derecho: Auditoría y validación en vivo
        right_box = ttk.LabelFrame(grid_frame, text=" Validación del Escenario ", style="Section.TLabelframe")
        right_box.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=4)

        self.validation_text = tk.Text(right_box, wrap=tk.WORD, font=("Consolas", 10), bg="#1e293b", fg="#f8fafc", relief=tk.FLAT)
        self.validation_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    def refresh_tab_config(self) -> None:
        """Actualiza los controles de la pestaña de configuración con los datos del manager."""
        # Poblar combo de escenarios
        official_files = self.manager.list_official_scenarios()
        custom_files = self.manager.list_custom_scenarios()

        options = [f"[Oficial] {f.name}" for f in official_files] + [f"[Personalizado] {f.name}" for f in custom_files]
        self.scenario_combo["values"] = options

        if self.manager.source_path:
            tag = "[Oficial]" if self.manager.is_official else "[Personalizado]"
            match_name = f"{tag} {self.manager.source_path.name}"
            if match_name in options:
                self.scenario_combo.set(match_name)
        else:
            self.scenario_combo.set("")

        self.scenario_name_var.set(self.manager.name)
        self.start_loc_combo["values"] = self.manager.locations
        self.start_loc_combo.set(self.manager.start_location)
        self.start_time_var.set(format_minutes_to_hhmm(self.manager.start_time))
        self.alpha_var.set(str(self.manager.alpha))

        # Pesos
        self.prio_low_var.set(str(self.manager.priority_weights.get("low", self.manager.priority_weights.get("baja", 1.0))))
        self.prio_med_var.set(str(self.manager.priority_weights.get("medium", self.manager.priority_weights.get("media", 2.0))))
        self.prio_high_var.set(str(self.manager.priority_weights.get("high", self.manager.priority_weights.get("alta", 3.0))))

        is_off, msg = self.manager.is_official_range()
        self.pkg_count_label.config(text=msg, foreground="#10b981" if is_off else "#f59e0b")
        self.official_badge.config(
            text="✓ Rango Oficial (5-8)" if is_off else "⚠ Modo Prueba",
            foreground="#10b981" if is_off else "#f59e0b",
        )

        self.run_live_validation()

    def apply_tab_config(self) -> bool:
        """Aplica los cambios introducidos en la pestaña de configuración general."""
        try:
            self.manager.name = self.scenario_name_var.get().strip() or "escenario_personalizado"
            start_loc = self.start_loc_combo.get().strip()
            if start_loc in self.manager.locations:
                self.manager.start_location = start_loc

            self.manager.start_time = parse_time_to_minutes(self.start_time_var.get().strip())
            self.manager.alpha = float(self.alpha_var.get().strip())

            w_low = float(self.prio_low_var.get().strip())
            w_med = float(self.prio_med_var.get().strip())
            w_high = float(self.prio_high_var.get().strip())
            self.manager.priority_weights["low"] = w_low
            self.manager.priority_weights["baja"] = w_low
            self.manager.priority_weights["medium"] = w_med
            self.manager.priority_weights["media"] = w_med
            self.manager.priority_weights["high"] = w_high
            self.manager.priority_weights["alta"] = w_high

            self.set_status("Configuración general aplicada con éxito.")
            self.refresh_all_views()
            return True
        except Exception as exc:
            messagebox.showerror("Error en Configuración", f"Datos inválidos: {exc}")
            self.set_status(f"Error al aplicar configuración: {exc}", is_error=True)
            return False

    def run_live_validation(self) -> bool:
        """Ejecuta la validación del escenario y actualiza la caja de texto."""
        errors = self.manager.validate()
        self.validation_text.delete("1.0", tk.END)

        if not errors:
            self.validation_text.insert(
                tk.END,
                "✓ ESCENARIO VÁLIDO\n\n"
                "- Identificadores únicos y coherentes.\n"
                "- Todas las ventanas temporales son válidas.\n"
                "- Horas prometidas dentro de los rangos.\n"
                "- Matriz de tiempos completa con diagonal cero.\n"
                "- Listo para ejecutar algoritmos de búsqueda.\n"
            )
            return True
        else:
            self.validation_text.insert(
                tk.END,
                f"✗ SE DETECTARON {len(errors)} ERROR(ES) DE VALIDACIÓN:\n\n"
            )
            for err in errors:
                self.validation_text.insert(tk.END, f" • {err}\n")
            return False

    # =========================================================================
    # TAB B: ADMINISTRACIÓN DE PAQUETES
    # =========================================================================

    def _build_tab_packages(self) -> None:
        paned = ttk.PanedWindow(self.tab_packages, orient=tk.VERTICAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # Parte superior: Tabla de paquetes
        table_frame = ttk.Frame(paned)
        paned.add(table_frame, weight=3)

        columns = ("id", "destination", "window_start", "window_end", "promised_time", "priority", "service_time")
        self.pkg_tree = ttk.Treeview(table_frame, columns=columns, show="headings", selectmode="browse")

        self.pkg_tree.heading("id", text="ID Paquete")
        self.pkg_tree.heading("destination", text="Destino")
        self.pkg_tree.heading("window_start", text="Ventana Inicio")
        self.pkg_tree.heading("window_end", text="Ventana Fin")
        self.pkg_tree.heading("promised_time", text="Hora Prometida")
        self.pkg_tree.heading("priority", text="Prioridad")
        self.pkg_tree.heading("service_time", text="Servicio (min)")

        self.pkg_tree.column("id", width=100, anchor=tk.CENTER)
        self.pkg_tree.column("destination", width=110, anchor=tk.CENTER)
        self.pkg_tree.column("window_start", width=120, anchor=tk.CENTER)
        self.pkg_tree.column("window_end", width=120, anchor=tk.CENTER)
        self.pkg_tree.column("promised_time", width=130, anchor=tk.CENTER)
        self.pkg_tree.column("priority", width=110, anchor=tk.CENTER)
        self.pkg_tree.column("service_time", width=110, anchor=tk.CENTER)

        tree_scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.pkg_tree.yview)
        self.pkg_tree.configure(yscrollcommand=tree_scroll.set)

        self.pkg_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.pkg_tree.bind("<<TreeviewSelect>>", self.on_pkg_selected)

        # Parte inferior: Formulario de edición de paquetes
        form_box = ttk.LabelFrame(paned, text=" Editor de Paquete ", style="Section.TLabelframe", padding=10)
        paned.add(form_box, weight=2)

        f_grid = ttk.Frame(form_box)
        f_grid.pack(fill=tk.X, expand=True)

        ttk.Label(f_grid, text="ID:").grid(row=0, column=0, padx=4, pady=4, sticky=tk.W)
        self.f_pkg_id_var = tk.StringVar()
        ttk.Entry(f_grid, textvariable=self.f_pkg_id_var, width=12).grid(row=0, column=1, padx=4, pady=4)

        ttk.Label(f_grid, text="Destino:").grid(row=0, column=2, padx=4, pady=4, sticky=tk.W)
        self.f_pkg_dest_combo = ttk.Combobox(f_grid, state="readonly", width=12)
        self.f_pkg_dest_combo.grid(row=0, column=3, padx=4, pady=4)

        ttk.Label(f_grid, text="Prioridad:").grid(row=0, column=4, padx=4, pady=4, sticky=tk.W)
        self.f_pkg_prio_combo = ttk.Combobox(f_grid, state="readonly", values=["baja", "media", "alta"], width=12)
        self.f_pkg_prio_combo.set("media")
        self.f_pkg_prio_combo.grid(row=0, column=5, padx=4, pady=4)

        ttk.Label(f_grid, text="Ventana Inicio:").grid(row=1, column=0, padx=4, pady=4, sticky=tk.W)
        self.f_pkg_wstart_var = tk.StringVar(value="08:00")
        ttk.Entry(f_grid, textvariable=self.f_pkg_wstart_var, width=12).grid(row=1, column=1, padx=4, pady=4)

        ttk.Label(f_grid, text="Ventana Fin:").grid(row=1, column=2, padx=4, pady=4, sticky=tk.W)
        self.f_pkg_wend_var = tk.StringVar(value="10:00")
        ttk.Entry(f_grid, textvariable=self.f_pkg_wend_var, width=12).grid(row=1, column=3, padx=4, pady=4)

        ttk.Label(f_grid, text="Hora Prometida:").grid(row=1, column=4, padx=4, pady=4, sticky=tk.W)
        self.f_pkg_prom_var = tk.StringVar(value="09:00")
        ttk.Entry(f_grid, textvariable=self.f_pkg_prom_var, width=12).grid(row=1, column=5, padx=4, pady=4)

        ttk.Label(f_grid, text="Servicio (min):").grid(row=2, column=0, padx=4, pady=4, sticky=tk.W)
        self.f_pkg_serv_var = tk.StringVar(value="5")
        ttk.Entry(f_grid, textvariable=self.f_pkg_serv_var, width=12).grid(row=2, column=1, padx=4, pady=4)

        # Botonera del formulario
        btn_bar = ttk.Frame(form_box)
        btn_bar.pack(fill=tk.X, pady=8)

        ttk.Button(btn_bar, text="+ Agregar Paquete", command=self.on_add_package).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_bar, text="✎ Actualizar Seleccionado", command=self.on_update_package).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_bar, text="✕ Eliminar Seleccionado", command=self.on_delete_package).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_bar, text="Limpiar Formulario", command=self.on_clear_package_form).pack(side=tk.LEFT, padx=4)

    def refresh_tab_packages(self) -> None:
        """Refresca la tabla de paquetes."""
        for item in self.pkg_tree.get_children():
            self.pkg_tree.delete(item)

        for p in self.manager.packages:
            self.pkg_tree.insert(
                "",
                tk.END,
                iid=p["id"],
                values=(
                    p["id"],
                    p["destination"],
                    format_minutes_to_hhmm(p["window_start"]),
                    format_minutes_to_hhmm(p["window_end"]),
                    format_minutes_to_hhmm(p["promised_time"]),
                    p["priority"],
                    p["service_time"],
                ),
            )

        self.f_pkg_dest_combo["values"] = self.manager.locations
        if self.manager.locations and not self.f_pkg_dest_combo.get():
            self.f_pkg_dest_combo.set(self.manager.locations[0])

    def on_pkg_selected(self, event: Any) -> None:
        """Puebla el formulario con el paquete seleccionado en el árbol."""
        selected = self.pkg_tree.selection()
        if not selected:
            return
        pkg_id = selected[0]
        pkg = next((p for p in self.manager.packages if p["id"] == pkg_id), None)
        if pkg:
            self.f_pkg_id_var.set(pkg["id"])
            self.f_pkg_dest_combo.set(pkg["destination"])
            self.f_pkg_wstart_var.set(format_minutes_to_hhmm(pkg["window_start"]))
            self.f_pkg_wend_var.set(format_minutes_to_hhmm(pkg["window_end"]))
            self.f_pkg_prom_var.set(format_minutes_to_hhmm(pkg["promised_time"]))
            self.f_pkg_prio_combo.set(pkg["priority"])
            self.f_pkg_serv_var.set(str(pkg["service_time"]))

    def on_add_package(self) -> None:
        try:
            pkg_data = {
                "id": self.f_pkg_id_var.get().strip(),
                "destination": self.f_pkg_dest_combo.get().strip(),
                "window_start": parse_time_to_minutes(self.f_pkg_wstart_var.get().strip()),
                "window_end": parse_time_to_minutes(self.f_pkg_wend_var.get().strip()),
                "promised_time": parse_time_to_minutes(self.f_pkg_prom_var.get().strip()),
                "priority": self.f_pkg_prio_combo.get().strip(),
                "service_time": int(self.f_pkg_serv_var.get().strip()),
            }
            self.manager.add_package(pkg_data)
            self.set_status(f"Paquete '{pkg_data['id']}' agregado exitosamente.")
            self.refresh_all_views()
        except Exception as exc:
            messagebox.showerror("Error al agregar paquete", str(exc))

    def on_update_package(self) -> None:
        selected = self.pkg_tree.selection()
        if not selected:
            messagebox.showwarning("Selección requerida", "Seleccione un paquete en la tabla para actualizar.")
            return
        old_id = selected[0]
        try:
            pkg_data = {
                "id": self.f_pkg_id_var.get().strip(),
                "destination": self.f_pkg_dest_combo.get().strip(),
                "window_start": parse_time_to_minutes(self.f_pkg_wstart_var.get().strip()),
                "window_end": parse_time_to_minutes(self.f_pkg_wend_var.get().strip()),
                "promised_time": parse_time_to_minutes(self.f_pkg_prom_var.get().strip()),
                "priority": self.f_pkg_prio_combo.get().strip(),
                "service_time": int(self.f_pkg_serv_var.get().strip()),
            }
            self.manager.update_package(old_id, pkg_data)
            self.set_status(f"Paquete '{old_id}' actualizado.")
            self.refresh_all_views()
        except Exception as exc:
            messagebox.showerror("Error al actualizar paquete", str(exc))

    def on_delete_package(self) -> None:
        selected = self.pkg_tree.selection()
        if not selected:
            messagebox.showwarning("Selección requerida", "Seleccione un paquete en la tabla para eliminar.")
            return
        pkg_id = selected[0]
        if messagebox.askyesno("Confirmar eliminación", f"¿Desea eliminar el paquete '{pkg_id}'?"):
            try:
                self.manager.remove_package(pkg_id)
                self.set_status(f"Paquete '{pkg_id}' eliminado.")
                self.on_clear_package_form()
                self.refresh_all_views()
            except Exception as exc:
                messagebox.showerror("Error", str(exc))

    def on_clear_package_form(self) -> None:
        self.f_pkg_id_var.set("")
        self.f_pkg_wstart_var.set("08:00")
        self.f_pkg_wend_var.set("10:00")
        self.f_pkg_prom_var.set("09:00")
        self.f_pkg_prio_combo.set("media")
        self.f_pkg_serv_var.set("5")

    # =========================================================================
    # TAB C: UBICACIONES Y MATRIZ DE TIEMPOS
    # =========================================================================

    def _build_tab_locations(self) -> None:
        paned = ttk.PanedWindow(self.tab_locations, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # Panel izquierdo: Gestión de ubicaciones
        loc_box = ttk.LabelFrame(paned, text=" Ubicaciones de la Red ", style="Section.TLabelframe", padding=8)
        paned.add(loc_box, weight=1)

        self.loc_listbox = tk.Listbox(loc_box, font=("Segoe UI", 10), selectmode=tk.SINGLE, height=12)
        self.loc_listbox.pack(fill=tk.BOTH, expand=True, pady=4)

        entry_frame = ttk.Frame(loc_box)
        entry_frame.pack(fill=tk.X, pady=4)
        self.new_loc_var = tk.StringVar()
        ttk.Entry(entry_frame, textvariable=self.new_loc_var, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(entry_frame, text="+ Agregar", command=self.on_add_location).pack(side=tk.LEFT, padx=2)

        ttk.Button(loc_box, text="✕ Eliminar Ubicación", command=self.on_remove_location).pack(fill=tk.X, pady=2)
        ttk.Button(loc_box, text="✎ Renombrar Ubicación", command=self.on_rename_location).pack(fill=tk.X, pady=2)
        ttk.Button(loc_box, text="★ Establecer como Depósito", command=self.on_set_as_depot).pack(fill=tk.X, pady=2)

        # Panel derecho: Matriz de tiempos editable
        matrix_box = ttk.LabelFrame(paned, text=" Matriz de Tiempos de Desplazamiento (minutos) ", style="Section.TLabelframe", padding=8)
        paned.add(matrix_box, weight=3)

        info_lbl = ttk.Label(
            matrix_box,
            text="* Matriz dirigida (admite asimetría). Diagonal debe ser 0. Use '-' o vacío para ruta no disponible.",
            font=("Segoe UI", 9, "italic"),
            foreground="#475569",
        )
        info_lbl.pack(anchor=tk.W, pady=(0, 6))

        # Contenedor scrollable para la grilla
        canvas_container = ttk.Frame(matrix_box)
        canvas_container.pack(fill=tk.BOTH, expand=True)

        self.matrix_canvas = tk.Canvas(canvas_container, bg="#1e293b", highlightthickness=0)
        self.matrix_scroll_y = ttk.Scrollbar(canvas_container, orient=tk.VERTICAL, command=self.matrix_canvas.yview)
        self.matrix_scroll_x = ttk.Scrollbar(canvas_container, orient=tk.HORIZONTAL, command=self.matrix_canvas.xview)

        self.matrix_grid_frame = ttk.Frame(self.matrix_canvas)
        self.matrix_canvas.create_window((0, 0), window=self.matrix_grid_frame, anchor="nw")

        self.matrix_canvas.configure(xscrollcommand=self.matrix_scroll_x.set, yscrollcommand=self.matrix_scroll_y.set)
        self.matrix_grid_frame.bind("<Configure>", lambda e: self.matrix_canvas.configure(scrollregion=self.matrix_canvas.bbox("all")))

        self.matrix_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.matrix_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.matrix_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)

        self.matrix_entries: dict[tuple[str, str], tk.StringVar] = {}

        btn_mat_bar = ttk.Frame(matrix_box)
        btn_mat_bar.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(btn_mat_bar, text="Aplicar Cambios a la Matriz", command=self.on_apply_matrix).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_mat_bar, text="Revertir Matriz", command=self.refresh_tab_locations).pack(side=tk.LEFT, padx=4)

    def refresh_tab_locations(self) -> None:
        """Actualiza la lista de ubicaciones y reconstruye la grilla de la matriz."""
        self.loc_listbox.delete(0, tk.END)
        for loc in self.manager.locations:
            prefix = "★ [Depósito] " if loc == self.manager.start_location else "   "
            self.loc_listbox.insert(tk.END, f"{prefix}{loc}")

        # Limpiar grilla anterior
        for widget in self.matrix_grid_frame.winfo_children():
            widget.destroy()

        self.matrix_entries.clear()
        locations = self.manager.locations

        # Encabezado de columnas
        ttk.Label(self.matrix_grid_frame, text="Origen \\ Destino", font=("Segoe UI", 9, "bold"), background="#334155", foreground="#ffffff", padding=4).grid(row=0, column=0, padx=1, pady=1, sticky="nsew")

        for col_idx, dest in enumerate(locations, start=1):
            ttk.Label(self.matrix_grid_frame, text=dest, font=("Segoe UI", 9, "bold"), background="#334155", foreground="#ffffff", padding=4).grid(row=0, column=col_idx, padx=1, pady=1, sticky="nsew")

        # Filas
        for row_idx, origin in enumerate(locations, start=1):
            ttk.Label(self.matrix_grid_frame, text=origin, font=("Segoe UI", 9, "bold"), background="#334155", foreground="#ffffff", padding=4).grid(row=row_idx, column=0, padx=1, pady=1, sticky="nsew")

            for col_idx, dest in enumerate(locations, start=1):
                raw_val = self.manager.travel_times.get(origin, {}).get(dest)
                str_val = "0" if origin == dest else ("-" if raw_val is None else str(raw_val))

                var = tk.StringVar(value=str_val)
                self.matrix_entries[(origin, dest)] = var

                if origin == dest:
                    lbl = tk.Label(self.matrix_grid_frame, text="0", bg="#475569", fg="#94a3b8", width=7, relief=tk.FLAT)
                    lbl.grid(row=row_idx, column=col_idx, padx=1, pady=1, sticky="nsew")
                else:
                    ent = tk.Entry(self.matrix_grid_frame, textvariable=var, width=7, justify=tk.CENTER, bg="#0f172a", fg="#f8fafc", insertbackground="#38bdf8")
                    ent.grid(row=row_idx, column=col_idx, padx=1, pady=1, sticky="nsew")

    def on_add_location(self) -> None:
        loc_id = self.new_loc_var.get().strip()
        if not loc_id:
            messagebox.showwarning("Atención", "Ingrese un nombre de ubicación.")
            return
        try:
            self.manager.add_location(loc_id)
            self.new_loc_var.set("")
            self.set_status(f"Ubicación '{loc_id}' añadida a la red.")
            self.refresh_all_views()
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def on_remove_location(self) -> None:
        sel_idx = self.loc_listbox.curselection()
        if not sel_idx:
            messagebox.showwarning("Atención", "Seleccione una ubicación en la lista.")
            return
        loc_id = self.manager.locations[sel_idx[0]]
        if messagebox.askyesno("Confirmar", f"¿Desea eliminar la ubicación '{loc_id}'?"):
            try:
                self.manager.remove_location(loc_id)
                self.set_status(f"Ubicación '{loc_id}' eliminada.")
                self.refresh_all_views()
            except Exception as exc:
                messagebox.showerror("Error", str(exc))

    def on_rename_location(self) -> None:
        sel_idx = self.loc_listbox.curselection()
        if not sel_idx:
            messagebox.showwarning("Atención", "Seleccione una ubicación en la lista para renombrar.")
            return
        old_id = self.manager.locations[sel_idx[0]]
        new_id = filedialog.askstring if hasattr(filedialog, "askstring") else None
        # Usar diálogo simple de tkinter
        from tkinter import simpledialog
        new_id = simpledialog.askstring("Renombrar Ubicación", f"Nuevo nombre para '{old_id}':", initialvalue=old_id)
        if new_id and new_id.strip():
            try:
                self.manager.rename_location(old_id, new_id.strip())
                self.set_status(f"Ubicación '{old_id}' renombrada a '{new_id}'.")
                self.refresh_all_views()
            except Exception as exc:
                messagebox.showerror("Error", str(exc))

    def on_set_as_depot(self) -> None:
        sel_idx = self.loc_listbox.curselection()
        if not sel_idx:
            messagebox.showwarning("Atención", "Seleccione una ubicación en la lista.")
            return
        loc_id = self.manager.locations[sel_idx[0]]
        self.manager.start_location = loc_id
        self.set_status(f"Ubicación '{loc_id}' establecida como depósito inicial.")
        self.refresh_all_views()

    def on_apply_matrix(self) -> None:
        """Lee los valores de la grilla y los aplica al gestor de escenarios."""
        try:
            for (u, v), var in self.matrix_entries.items():
                if u == v:
                    self.manager.set_travel_time(u, v, 0)
                    continue
                val_str = var.get().strip().lower()
                if val_str in ("", "-", "none", "null"):
                    self.manager.set_travel_time(u, v, None)
                else:
                    self.manager.set_travel_time(u, v, int(val_str))
            self.set_status("Matriz de tiempos actualizada correctamente.")
            self.run_live_validation()
        except Exception as exc:
            messagebox.showerror("Error en Matriz", f"Valor no numérico o inválido en la matriz: {exc}")

    # =========================================================================
    # TAB D: EJECUCIÓN Y RESULTADOS
    # =========================================================================

    def _build_tab_execution(self) -> None:
        # Barra superior de controles de ejecución
        ctrl_bar = ttk.Frame(self.tab_execution, padding=6)
        ctrl_bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(ctrl_bar, text="Algoritmo:", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=4)
        self.exec_alg_combo = ttk.Combobox(ctrl_bar, state="readonly", values=["A*", "Dijkstra", "Greedy", "Exhaustive"], width=14)
        self.exec_alg_combo.set("A*")
        self.exec_alg_combo.pack(side=tk.LEFT, padx=4)

        ttk.Button(ctrl_bar, text="▶ Ejecutar Algoritmo", command=self.on_run_single_algorithm, style="Primary.TButton").pack(side=tk.LEFT, padx=6)
        ttk.Button(ctrl_bar, text="📊 Comparar los 4 Algoritmos", command=self.on_run_comparison).pack(side=tk.LEFT, padx=6)

        ttk.Separator(ctrl_bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Label(ctrl_bar, text="Vista:", font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=2)
        self.view_mode_var = tk.StringVar(value="route")
        ttk.Radiobutton(ctrl_bar, text="Ruta Planificada", variable=self.view_mode_var, value="route", command=self.update_embedded_plot).pack(side=tk.LEFT, padx=3)
        ttk.Radiobutton(ctrl_bar, text="Árbol de Búsqueda A*", variable=self.view_mode_var, value="search", command=self.update_embedded_plot).pack(side=tk.LEFT, padx=3)

        # Panel principal dividido (Split View)
        paned = ttk.PanedWindow(self.tab_execution, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Panel izquierdo: Resultados numéricos y tablas
        left_results_frame = ttk.Frame(paned)
        paned.add(left_results_frame, weight=2)

        # Banner de resumen
        self.result_summary_box = tk.Text(left_results_frame, height=9, font=("Consolas", 10), bg="#0f172a", fg="#f8fafc", relief=tk.FLAT)
        self.result_summary_box.pack(fill=tk.X, padx=4, pady=4)

        # Sub-pestañas para pasos de la ruta vs comparativa
        res_notebook = ttk.Notebook(left_results_frame)
        res_notebook.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Tabla de pasos
        steps_tab = ttk.Frame(res_notebook)
        res_notebook.add(steps_tab, text=" Pasos de la Ruta ")
        step_cols = ("step", "package", "origin", "dest", "arrival", "wait", "start", "end", "tardiness", "penalty")
        self.steps_tree = ttk.Treeview(steps_tab, columns=step_cols, show="headings", height=8)

        self.steps_tree.heading("step", text="#")
        self.steps_tree.heading("package", text="Paquete")
        self.steps_tree.heading("origin", text="Origen")
        self.steps_tree.heading("dest", text="Destino")
        self.steps_tree.heading("arrival", text="Llegada")
        self.steps_tree.heading("wait", text="Espera")
        self.steps_tree.heading("start", text="Inicio Serv")
        self.steps_tree.heading("end", text="Fin Entrega")
        self.steps_tree.heading("tardiness", text="Retraso")
        self.steps_tree.heading("penalty", text="Penalización")

        for col, w in zip(step_cols, [35, 65, 55, 55, 60, 55, 75, 75, 60, 80]):
            self.steps_tree.column(col, width=w, anchor=tk.CENTER)

        s_scroll = ttk.Scrollbar(steps_tab, orient=tk.VERTICAL, command=self.steps_tree.yview)
        self.steps_tree.configure(yscrollcommand=s_scroll.set)
        self.steps_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        s_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Tabla comparativa
        comp_tab = ttk.Frame(res_notebook)
        res_notebook.add(comp_tab, text=" Tabla Comparativa ")
        comp_cols = ("alg", "solved", "cost", "elapsed", "tardiness", "expanded", "cpu_ms")
        self.comp_tree = ttk.Treeview(comp_tab, columns=comp_cols, show="headings", height=8)

        self.comp_tree.heading("alg", text="Algoritmo")
        self.comp_tree.heading("solved", text="Factible")
        self.comp_tree.heading("cost", text="Costo Objetivo J(π)")
        self.comp_tree.heading("elapsed", text="Duración (min)")
        self.comp_tree.heading("tardiness", text="Retraso Ponderado")
        self.comp_tree.heading("expanded", text="Expandidos")
        self.comp_tree.heading("cpu_ms", text="Tiempo CPU (ms)")

        for c, w in zip(comp_cols, [80, 60, 110, 90, 110, 80, 95]):
            self.comp_tree.column(c, width=w, anchor=tk.CENTER)

        c_scroll = ttk.Scrollbar(comp_tab, orient=tk.VERTICAL, command=self.comp_tree.yview)
        self.comp_tree.configure(yscrollcommand=c_scroll.set)
        self.comp_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        c_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Panel derecho: Visualización gráfica Matplotlib incrustada
        right_plot_frame = ttk.Frame(paned)
        paned.add(right_plot_frame, weight=3)

        # Barra de control de reproducción paso a paso
        playback_bar = ttk.Frame(right_plot_frame, padding=4)
        playback_bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(playback_bar, text="|< Inicio", width=8, command=self.playback_first).pack(side=tk.LEFT, padx=2)
        ttk.Button(playback_bar, text="< Anterior", width=8, command=self.playback_prev).pack(side=tk.LEFT, padx=2)
        self.play_btn = ttk.Button(playback_bar, text="▶ Reproducir", width=12, command=self.playback_toggle_play)
        self.play_btn.pack(side=tk.LEFT, padx=3)
        ttk.Button(playback_bar, text="Siguiente >", width=9, command=self.playback_next).pack(side=tk.LEFT, padx=2)
        ttk.Button(playback_bar, text="Fin >|", width=8, command=self.playback_last).pack(side=tk.LEFT, padx=2)

        self.playback_info_lbl = ttk.Label(playback_bar, text="Paso 0/0", font=("Segoe UI", 9, "bold"), foreground="#0f172a")
        self.playback_info_lbl.pack(side=tk.RIGHT, padx=6)

        # Canvas Matplotlib embebido
        self.fig, self.ax = plt.subplots(figsize=(6.5, 5.5), facecolor="#0f172a")
        self.ax.set_facecolor("#1e293b")
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def on_run_single_algorithm(self) -> None:
        """Ejecuta el algoritmo seleccionado y actualiza las vistas de resultados."""
        if not self.apply_tab_config():
            return

        errors = self.manager.validate()
        if errors:
            messagebox.showerror(
                "Escenario Inválido",
                "Corrija los siguientes errores antes de ejecutar:\n\n" + "\n".join(errors[:5])
            )
            return

        alg_name = self.exec_alg_combo.get().strip()
        self.last_algorithm = alg_name
        self.set_status(f"Ejecutando {alg_name}...")
        self.root.update_idletasks()

        try:
            # Rastrear eventos en A* para visualización interactiva
            track_events = (alg_name == "A*")
            result = self.manager.run_algorithm(alg_name, track_events=track_events)
            self.last_result = result
            self.comparison_results = None

            self._display_execution_results(result, alg_name)
            self._prepare_playback_frames(result)
            self.update_embedded_plot()
            self.set_status(f"Ejecución de {alg_name} completada. Factible: {result.is_feasible}.")
        except Exception as exc:
            messagebox.showerror("Error en Búsqueda", str(exc))
            self.set_status(f"Error en {alg_name}: {exc}", is_error=True)

    def on_run_comparison(self) -> None:
        """Ejecuta los cuatro algoritmos sobre el mismo escenario y presenta la comparativa."""
        if not self.apply_tab_config():
            return
        errors = self.manager.validate()
        if errors:
            messagebox.showerror("Escenario Inválido", "No se puede ejecutar debido a errores de validación.")
            return

        self.set_status("Ejecutando comparativa entre A*, Dijkstra, Greedy y Exhaustiva...")
        self.root.update_idletasks()

        try:
            results = self.manager.run_comparison()
            self.comparison_results = results
            # Usar A* como resultado principal para la gráfica
            self.last_result = results.get("A*")
            self.last_algorithm = "A*"

            self._display_comparison_table(results)
            if self.last_result:
                self._display_execution_results(self.last_result, "A*")
                self._prepare_playback_frames(self.last_result)
                self.update_embedded_plot()

            self.set_status("Comparativa de 4 algoritmos finalizada con éxito.")
        except Exception as exc:
            messagebox.showerror("Error en Comparativa", str(exc))
            self.set_status(f"Error en comparativa: {exc}", is_error=True)

    def _display_execution_results(self, result: SearchResult, alg_name: str) -> None:
        """Puebla el cuadro de resumen y la tabla de pasos."""
        self.result_summary_box.delete("1.0", tk.END)
        summary = result.cost_summary

        lines = [
            f" Algoritmo: {alg_name} | Factible: {'SÍ (Solución Óptima)' if result.is_feasible else 'NO'}",
            f" Ruta:      {' -> '.join([self.manager.start_location] + list(result.package_order)) if result.is_feasible else 'Sin solución'}",
        ]

        if result.is_feasible and summary is not None:
            lines.extend([
                f" Costo Objetivo J(π): {summary.objective_cost:.2f} (α={summary.alpha})",
                f" Duración Total:      {summary.total_elapsed_time} min ({format_minutes_to_hhmm(summary.total_elapsed_time)})",
                f" Tiempos:             Viaje={summary.total_travel_time}m | Espera={summary.total_wait_time}m | Servicio={summary.total_service_time}m",
                f" Retrasos:            Total={summary.total_tardiness}m | Ponderado={summary.weighted_tardiness:.2f} | Penalización={summary.total_penalty:.2f}",
                f" Eficiencia Motor:    Expandidos={result.metrics.states_expanded} | Generados={result.metrics.states_generated} | Cola Máx={result.metrics.max_queue_size} | CPU={result.metrics.execution_time_seconds * 1000.0:.2f} ms",
            ])
        else:
            lines.extend([
                " No se encontró ninguna ruta que satisfaga todas las restricciones de ventana.",
                f" Nodos expandidos hasta agotar: {result.metrics.states_expanded} | Descartados: {result.metrics.infeasible_transitions_discarded}",
                f" Tiempo de búsqueda: {result.metrics.execution_time_seconds * 1000.0:.2f} ms",
            ])

        self.result_summary_box.insert(tk.END, "\n".join(lines))

        # Pasos
        for item in self.steps_tree.get_children():
            self.steps_tree.delete(item)

        if result.is_feasible:
            for idx, step in enumerate(result.transitions, start=1):
                self.steps_tree.insert(
                    "",
                    tk.END,
                    values=(
                        idx,
                        step.package_id,
                        step.origin,
                        step.destination,
                        format_minutes_to_hhmm(step.arrival_time),
                        f"{step.wait_time}m",
                        format_minutes_to_hhmm(step.service_start_time),
                        format_minutes_to_hhmm(step.departure_time),
                        f"{step.tardiness}m",
                        f"{step.penalty:.2f}",
                    ),
                )

    def _display_comparison_table(self, results: dict[str, SearchResult]) -> None:
        """Puebla la tabla de resultados comparativos."""
        for item in self.comp_tree.get_children():
            self.comp_tree.delete(item)

        for alg, res in results.items():
            cost_str = f"{res.total_cost:.2f}" if res.total_cost is not None else "N/F"
            elap_str = f"{res.cost_summary.total_elapsed_time}m" if res.cost_summary else "N/F"
            tard_str = f"{res.cost_summary.weighted_tardiness:.2f}" if res.cost_summary else "N/F"
            exp_val = getattr(res.metrics, "states_expanded", "-")
            cpu_val = f"{res.metrics.execution_time_seconds * 1000.0:.2f}"

            self.comp_tree.insert(
                "",
                tk.END,
                values=(
                    alg,
                    "SÍ" if res.is_feasible else "NO",
                    cost_str,
                    elap_str,
                    tard_str,
                    exp_val,
                    cpu_val,
                ),
            )

    # =========================================================================
    # REPRODUCCIÓN PASO A PASO Y MATPLOTLIB EMBEBIDO
    # =========================================================================

    def _prepare_playback_frames(self, result: SearchResult) -> None:
        """Construye los estados de reproducción paso a paso para la animación."""
        self.playback_frames.clear()
        self.current_frame_idx = 0
        if not result.is_feasible or not result.transitions:
            return

        completed: set[str] = set()
        # Cuadro 0: Inicial
        self.playback_frames.append((0, "INIT", self.manager.start_time, self.manager.start_location, None, set(completed)))

        for idx, step in enumerate(result.transitions):
            # Fase viaje
            self.playback_frames.append((idx, "TRAVEL", step.arrival_time, step.destination, step, set(completed)))
            # Fase espera
            if step.wait_time > 0:
                self.playback_frames.append((idx, "WAIT", step.service_start_time, step.destination, step, set(completed)))
            # Fase servicio
            self.playback_frames.append((idx, "SERVICE", step.departure_time, step.destination, step, set(completed)))
            # Fase finalización
            completed.add(step.package_id)
            self.playback_frames.append((idx, "COMPLETED", step.departure_time, step.destination, step, set(completed)))

    def update_embedded_plot(self) -> None:
        """Renderiza la gráfica seleccionada (Ruta o Árbol A*) dentro del Canvas de Tkinter."""
        self.fig.clear()

        if self.last_result is None:
            ax = self.fig.add_subplot(111)
            ax.set_facecolor("#1e293b")
            ax.text(0.5, 0.5, "Ejecute un algoritmo para visualizar la ruta planificada.", color="#94a3b8", ha="center", va="center", fontsize=11)
            ax.axis("off")
            self.canvas.draw()
            return

        mode = self.view_mode_var.get()

        if mode == "search":
            # Vista de árbol de búsqueda A*
            if not self.last_result.events:
                ax = self.fig.add_subplot(111)
                ax.set_facecolor("#1e293b")
                ax.text(0.5, 0.5, "El trazado de eventos solo está disponible para ejecuciones con A*.", color="#f59e0b", ha="center", va="center", fontsize=11)
                ax.axis("off")
            else:
                # Renderizar árbol temporalmente a archivo y mostrar o pintar directamente
                temp_png = Path(tempfile.gettempdir()) / "_embedded_search.png"
                problem = self.manager.to_delivery_problem()
                render_search_tree_static(problem, self.last_result, temp_png, dpi=110)
                img = plt.imread(str(temp_png))
                ax = self.fig.add_subplot(111)
                ax.imshow(img)
                ax.axis("off")
        else:
            # Vista de ruta paso a paso
            problem = self.manager.to_delivery_problem()
            pos = compute_deterministic_layout(problem)

            if not self.playback_frames:
                # Renderizar estático
                temp_png = Path(tempfile.gettempdir()) / "_embedded_route.png"
                render_route_static(problem, self.last_result, temp_png, dpi=110)
                img = plt.imread(str(temp_png))
                ax = self.fig.add_subplot(111)
                ax.imshow(img)
                ax.axis("off")
            else:
                # Dibujar cuadro activo del reproductor
                frame_data = self.playback_frames[self.current_frame_idx]
                step_idx, phase, cur_time, ag_loc, cur_step, completed = frame_data

                pil_img = _render_single_animation_frame(
                    problem=problem,
                    result=self.last_result,
                    pos=pos,
                    current_step_idx=step_idx,
                    phase=phase,
                    current_time=cur_time,
                    agent_loc=ag_loc,
                    completed_pkgs=completed,
                    current_step=cur_step,
                )
                ax = self.fig.add_subplot(111)
                ax.imshow(pil_img)
                ax.axis("off")

                total_frames = len(self.playback_frames)
                self.playback_info_lbl.config(text=f"Cuadro {self.current_frame_idx + 1}/{total_frames} ({phase})")

        self.fig.tight_layout()
        self.canvas.draw()

    def playback_first(self) -> None:
        self.current_frame_idx = 0
        self.update_embedded_plot()

    def playback_last(self) -> None:
        if self.playback_frames:
            self.current_frame_idx = len(self.playback_frames) - 1
            self.update_embedded_plot()

    def playback_prev(self) -> None:
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.update_embedded_plot()

    def playback_next(self) -> None:
        if self.current_frame_idx < len(self.playback_frames) - 1:
            self.current_frame_idx += 1
            self.update_embedded_plot()

    def playback_toggle_play(self) -> None:
        if self.is_playing:
            self.is_playing = False
            self.play_btn.config(text="▶ Reproducir")
            if self.playback_timer is not None:
                self.root.after_cancel(self.playback_timer)
                self.playback_timer = None
        else:
            if not self.playback_frames:
                return
            self.is_playing = True
            self.play_btn.config(text="❚❚ Pausa")
            self._playback_loop()

    def _playback_loop(self) -> None:
        if not self.is_playing:
            return
        if self.current_frame_idx < len(self.playback_frames) - 1:
            self.current_frame_idx += 1
            self.update_embedded_plot()
            self.playback_timer = self.root.after(700, self._playback_loop)
        else:
            self.is_playing = False
            self.play_btn.config(text="▶ Reproducir")
            self.playback_timer = None

    # =========================================================================
    # ACCIONES DE BARRA SUPERIOR Y ARCHIVOS
    # =========================================================================

    def refresh_all_views(self) -> None:
        """Actualiza integralmente todas las pestañas de la interfaz."""
        self.refresh_tab_config()
        self.refresh_tab_packages()
        self.refresh_tab_locations()

    def on_new_scenario(self) -> None:
        """Crea un nuevo escenario en blanco desde cero."""
        if messagebox.askyesno("Nuevo Escenario", "¿Desea iniciar un nuevo escenario desde cero? Se perderán cambios no guardados."):
            self.manager.new_scenario()
            self.last_result = None
            self.set_status("Nuevo escenario en blanco inicializado.")
            self.refresh_all_views()
            self.update_embedded_plot()

    def on_scenario_combo_selected(self, event: Any) -> None:
        """Maneja la selección rápida en el combobox de escenarios."""
        val = self.scenario_combo.get().strip()
        if not val:
            return
        if val.startswith("[Oficial]"):
            name = val.replace("[Oficial]", "").strip()
            path = self.manager.OFFICIAL_SCENARIOS_DIR / name
        else:
            name = val.replace("[Personalizado]", "").strip()
            path = self.manager.CUSTOM_SCENARIOS_DIR / name

        if path.exists():
            try:
                self.manager.load_from_json(path)
                self.last_result = None
                self.set_status(f"Escenario '{path.name}' cargado con éxito.")
                self.refresh_all_views()
                self.update_embedded_plot()
            except Exception as exc:
                messagebox.showerror("Error al cargar", str(exc))

    def on_load_official_dialog(self) -> None:
        """Abre un diálogo de selección para cargar un escenario oficial."""
        initial_dir = str(self.manager.OFFICIAL_SCENARIOS_DIR.resolve())
        path = filedialog.askopenfilename(
            initialdir=initial_dir,
            title="Seleccionar Escenario Oficial",
            filetypes=[("Archivos JSON", "*.json")],
        )
        if path:
            try:
                self.manager.load_from_json(path)
                self.last_result = None
                self.set_status(f"Escenario oficial '{Path(path).name}' cargado.")
                self.refresh_all_views()
                self.update_embedded_plot()
            except Exception as exc:
                messagebox.showerror("Error al cargar", str(exc))

    def on_load_custom_dialog(self) -> None:
        """Abre un diálogo para cargar un escenario personalizado."""
        initial_dir = str(self.manager.CUSTOM_SCENARIOS_DIR.resolve())
        path = filedialog.askopenfilename(
            initialdir=initial_dir,
            title="Seleccionar Escenario Personalizado",
            filetypes=[("Archivos JSON", "*.json")],
        )
        if path:
            try:
                self.manager.load_from_json(path)
                self.last_result = None
                self.set_status(f"Escenario personalizado '{Path(path).name}' cargado.")
                self.refresh_all_views()
                self.update_embedded_plot()
            except Exception as exc:
                messagebox.showerror("Error al cargar", str(exc))

    def on_save_as_custom(self) -> None:
        """Guarda el escenario en la carpeta aislada data/scenarios/custom/."""
        self.apply_tab_config()
        try:
            target = self.manager.save_as_custom()
            messagebox.showinfo("Guardado Exitoso", f"Escenario personalizado guardado en:\n{target}")
            self.set_status(f"Escenario guardado como '{target.name}'.")
            self.refresh_tab_config()
        except Exception as exc:
            messagebox.showerror("Error al guardar", str(exc))

    def on_import_json(self) -> None:
        """Importa un archivo JSON arbitrario de escenario."""
        path = filedialog.askopenfilename(
            title="Importar Escenario JSON",
            filetypes=[("Archivos JSON", "*.json"), ("Todos los archivos", "*.*")],
        )
        if path:
            try:
                self.manager.load_from_json(path)
                self.last_result = None
                self.set_status(f"Escenario importado desde '{Path(path).name}'.")
                self.refresh_all_views()
                self.update_embedded_plot()
            except Exception as exc:
                messagebox.showerror("Error de Importación", str(exc))

    def on_export_json(self) -> None:
        """Exporta el escenario actual a cualquier ruta seleccionada por el usuario."""
        self.apply_tab_config()
        default_name = f"{self.manager.name}.json"
        path = filedialog.asksaveasfilename(
            title="Exportar Escenario JSON",
            defaultextension=".json",
            initialfile=default_name,
            filetypes=[("Archivos JSON", "*.json")],
        )
        if path:
            try:
                self.manager.save_to_json(path)
                messagebox.showinfo("Exportación Exitosa", f"Escenario exportado correctamente a:\n{path}")
                self.set_status(f"Escenario exportado a '{Path(path).name}'.")
            except Exception as exc:
                messagebox.showerror("Error al exportar", str(exc))

    def on_reset_scenario(self) -> None:
        """Restablece los valores al escenario oficial actual o a un nuevo escenario."""
        if self.manager.source_path and self.manager.source_path.exists():
            self.manager.load_from_json(self.manager.source_path)
            self.set_status(f"Valores restablecidos desde '{self.manager.source_path.name}'.")
        else:
            self.manager.new_scenario()
            self.set_status("Valores restablecidos a valores por defecto.")
        self.last_result = None
        self.refresh_all_views()
        self.update_embedded_plot()

    def run(self) -> None:
        """Inicia el bucle de eventos de la interfaz de escritorio."""
        self.root.mainloop()
