"""Visualizador del árbol de búsqueda y la dinámica interna del algoritmo A*."""

from __future__ import annotations

import io
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image

from delivery_agent.algorithms.astar import SearchEvent, SearchResult
from delivery_agent.domain.problem import DeliveryProblem
from delivery_agent.domain.state import DeliveryState


def _state_key(state: DeliveryState) -> str:
    """Genera una clave corta legible para representar un estado de entrega."""
    pkgs = ",".join(sorted(state.pending_packages)) or "Ø"
    return f"{state.location}@{state.current_time}m[{pkgs}]"


def render_search_tree_static(
    problem: DeliveryProblem,
    result: SearchResult,
    output_path: str | Path,
    title: str | None = None,
    dpi: int = 200,
) -> Path:
    """Genera un diagrama estático de alta resolución del árbol de búsqueda explorado.

    Muestra los estados generados y expandidos, las podas por infactibilidad o
    dominancia, los valores heurísticos f = g + h y el camino solución óptimo.

    Args:
        problem: Problema de reparto evaluado.
        result: Resultado de búsqueda obtenido con A* (debe tener `events`).
        output_path: Ruta del archivo PNG de salida.
        title: Título opcional para el gráfico.
        dpi: Resolución en puntos por pulgada.

    Returns:
        Ruta absoluta del archivo PNG generado.
    """
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_tree, ax_hud) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(16, 9),
        gridspec_kw={"width_ratios": [2.3, 1.0]},
    )

    fig.patch.set_facecolor("#0f172a")
    ax_tree.set_facecolor("#1e293b")
    ax_hud.set_facecolor("#1e293b")

    events = result.events or ()
    total_packages = len(problem.packages)

    # Construir grafo de búsqueda dirigido a partir de los eventos
    G = nx.DiGraph()
    node_labels: dict[str, str] = {}
    node_colors: dict[str, str] = {}
    node_layers: dict[str, int] = {}
    node_details: dict[str, dict[str, float | str]] = {}

    # Estado inicial
    init_state = problem.initial_state
    init_key = _state_key(init_state)
    G.add_node(init_key)
    node_layers[init_key] = 0
    node_colors[init_key] = "#38bdf8"  # Sky 400
    node_labels[init_key] = f"{problem.start_location}\nt=0\ng=0"
    node_details[init_key] = {"g": 0.0, "h": 0.0, "f": 0.0, "status": "ORIGEN"}

    # Conjunto de estados que pertenecen a la ruta óptima encontrada
    solution_states: set[str] = {init_key}
    curr_state = init_state
    for step in result.transitions:
        curr_state = step.next_state
        solution_states.add(_state_key(curr_state))

    # Procesar eventos para poblar el grafo
    for ev in events:
        s_key = _state_key(ev.state)
        depth = total_packages - len(ev.pending_packages)
        node_layers[s_key] = depth

        if ev.event_type == "EXPAND":
            if s_key not in node_colors:
                node_colors[s_key] = "#818cf8"  # Indigo 400
            node_labels[s_key] = f"{ev.location}\nt={ev.current_time}\nf={ev.f:.0f}"
            node_details[s_key] = {"g": ev.g, "h": ev.h, "f": ev.f, "status": "EXPANDIDO"}

        elif ev.event_type == "PUSH_SUCCESSOR" and ev.next_state is not None:
            succ_key = _state_key(ev.next_state)
            succ_depth = total_packages - len(ev.next_state.pending_packages)
            G.add_edge(s_key, succ_key, package=ev.package_id, color="#64748b")
            node_layers[succ_key] = succ_depth
            if succ_key not in node_colors:
                node_colors[succ_key] = "#fbbf24"  # Amber 400 (en cola)
            node_labels[succ_key] = f"{ev.next_state.location}\nt={ev.next_state.current_time}\nf={ev.f:.0f}"
            node_details[succ_key] = {"g": ev.g, "h": ev.h, "f": ev.f, "status": "COLA"}

        elif ev.event_type == "GOAL":
            node_colors[s_key] = "#10b981"  # Emerald 500
            node_labels[s_key] = f"META\nt={ev.current_time}\nC={ev.g:.1f}"
            node_details[s_key] = {"g": ev.g, "h": ev.h, "f": ev.f, "status": "META"}

    # Resaltar estados y aristas de la solución óptima
    for s_key in solution_states:
        if s_key in G:
            node_colors[s_key] = "#10b981"

    # Disposición jerárquica por capas de profundidad (paquetes entregados)
    pos: dict[str, tuple[float, float]] = {}
    layer_nodes: dict[int, list[str]] = {}
    for node, layer in node_layers.items():
        layer_nodes.setdefault(layer, []).append(node)

    max_layer = max(layer_nodes.keys()) if layer_nodes else 0
    for layer, nodes in layer_nodes.items():
        count = len(nodes)
        y = max_layer - layer  # Arriba hacia abajo
        for i, node in enumerate(sorted(nodes)):
            x = (i - (count - 1) / 2.0) * (2.2 / max(1, count * 0.4))
            pos[node] = (x, y)

    # Si hay nodos sin posición calculada, asignarles layout por defecto
    if len(pos) < len(G):
        spring_pos = nx.spring_layout(G, seed=42)
        for n in G.nodes():
            if n not in pos:
                pos[n] = (spring_pos[n][0] * 5.0, spring_pos[n][1] * 5.0)

    # Dibujar aristas
    for u, v, data in G.edges(data=True):
        is_solution_edge = (u in solution_states and v in solution_states)
        edge_col = "#10b981" if is_solution_edge else "#475569"
        edge_lw = 2.8 if is_solution_edge else 1.0
        edge_alpha = 1.0 if is_solution_edge else 0.4

        ux, uy = pos[u]
        vx, vy = pos[v]
        ax_tree.annotate(
            "",
            xy=(vx, vy),
            xytext=(ux, uy),
            arrowprops=dict(
                arrowstyle="-|>",
                color=edge_col,
                lw=edge_lw,
                alpha=edge_alpha,
                shrinkA=14,
                shrinkB=14,
            ),
        )

        # Etiqueta del paquete sobre la arista
        pkg_id = data.get("package", "")
        if pkg_id and is_solution_edge:
            mid_x = (ux + vx) / 2.0
            mid_y = (uy + vy) / 2.0
            ax_tree.text(
                mid_x,
                mid_y,
                pkg_id,
                color="#ffffff",
                fontsize=8,
                weight="bold",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#10b981", edgecolor="#0f172a", alpha=0.9),
            )

    # Dibujar nodos
    for node in G.nodes():
        if node in pos:
            x, y = pos[node]
            c = node_colors.get(node, "#64748b")
            is_sol = node in solution_states
            size = 550 if is_sol else 380
            ax_tree.scatter([x], [y], s=size, c=[c], edgecolors="#ffffff" if is_sol else "#334155", linewidths=2 if is_sol else 1, zorder=4)
            lbl = node_labels.get(node, node[:6])
            ax_tree.text(
                x,
                y,
                lbl,
                color="#0f172a" if c in ("#fbbf24", "#38bdf8", "#10b981") else "#f8fafc",
                fontsize=7.5,
                weight="bold",
                ha="center",
                va="center",
                zorder=5,
            )

    ax_tree.axis("off")
    ax_tree.set_title(
        title or f"Árbol de Búsqueda A* - Nodos Explorados ({len(G.nodes())})",
        color="#f8fafc",
        fontsize=13,
        weight="bold",
        pad=16,
    )

    # Panel lateral de métricas del algoritmo A*
    ax_hud.axis("off")
    metrics = result.metrics
    summary = result.cost_summary

    hud_lines = [
        "═══════════════════════════════════",
        "  MÉTRICAS DEL ALGORITMO A*",
        "═══════════════════════════════════",
        f" Algoritmo:         {result.algorithm}",
        f" Estado:            {'SOLUCIONADO' if result.is_feasible else 'NO FACTIBLE'}",
        f" Eventos Rastreados:{len(events)}",
        "───────────────────────────────────",
        "  EXPLORACIÓN Y EFICIENCIA",
        "───────────────────────────────────",
        f" Estados Expandidos:{metrics.states_expanded}",
        f" Estados Generados: {metrics.states_generated}",
        f" Máx. Tamaño Cola:  {metrics.max_queue_size}",
        f" Re-aperturas:      {metrics.reopenings}",
        f" Entradas Obsoletas:{metrics.stale_queue_entries}",
        "───────────────────────────────────",
        "  PODAS Y DESCARTES",
        "───────────────────────────────────",
        f" Podas Ventana (inf): {metrics.infeasible_transitions_discarded}",
        f" Podas Dominancia:    {metrics.dominated_or_repeated_discarded}",
        "───────────────────────────────────",
        "  SOLUCIÓN HALLADA",
        "───────────────────────────────────",
    ]

    if result.is_feasible and summary is not None:
        route_str = " -> ".join([problem.start_location] + list(result.package_order))
        hud_lines.extend([
            f" Ruta:              {route_str}",
            f" Costo Objetivo:    {summary.objective_cost:.2f}",
            f" Tiempo Total:      {summary.total_elapsed_time} min",
            f" Retraso Ponderado: {summary.weighted_tardiness:.2f}",
            f" Penalización (α):  {summary.total_penalty:.2f}",
        ])
    else:
        hud_lines.append(" Sin solución factible.")

    ax_hud.text(
        0.05,
        0.95,
        "\n".join(hud_lines),
        color="#e2e8f0",
        fontsize=9.2,
        family="monospace",
        va="top",
        ha="left",
    )

    plt.tight_layout()
    fig.savefig(out_file, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    return out_file


def animate_search_process(
    problem: DeliveryProblem,
    result: SearchResult,
    output_path: str | Path,
    fps: int = 2,
    max_frames: int = 60,
    loop: int = 0,
) -> Path:
    """Genera una animación paso a paso (GIF) que ilustra el proceso dinámico de A*.

    Muestra de forma interactiva la selección de estados en la frontera OPEN,
    la evaluación de acciones sucesoras, la detección de violaciones de ventana
    temporal y dominancia de costos, y la extracción final de la meta óptima.

    Args:
        problem: Problema de reparto.
        result: Resultado de búsqueda con eventos (`result.events`).
        output_path: Ruta del archivo GIF de salida.
        fps: Cuadros por segundo.
        max_frames: Límite máximo de cuadros para mantener tamaño y tiempo razonables.
        loop: Repeticiones del GIF (0 = continuo).

    Returns:
        Ruta absoluta del archivo GIF generado.
    """
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    events = result.events or ()
    if not events:
        # Si no hay eventos registrados, generar imagen estática
        png_path = out_file.with_suffix(".png")
        render_search_tree_static(problem, result, png_path)
        img = Image.open(png_path)
        img.save(out_file, format="GIF")
        return out_file

    # Seleccionar subconjunto de eventos clave si exceden max_frames
    selected_events: list[SearchEvent]
    if len(events) <= max_frames:
        selected_events = list(events)
    else:
        # Priorizar eventos de tipo EXPAND y GOAL
        expand_and_goal = [e for e in events if e.event_type in ("EXPAND", "GOAL")]
        step = max(1, len(expand_and_goal) // (max_frames - 2))
        selected_events = expand_and_goal[::step]
        if events[-1] not in selected_events:
            selected_events.append(events[-1])

    frames: list[Image.Image] = []

    # Estado acumulativo de la búsqueda durante la animación
    expanded_states: set[str] = set()
    frontier_states: set[str] = set()
    discarded_infeasible_count = 0
    discarded_dominated_count = 0

    init_key = _state_key(problem.initial_state)
    frontier_states.add(init_key)

    total_pkgs = len(problem.packages)

    for frame_idx, ev in enumerate(selected_events, start=1):
        s_key = _state_key(ev.state)

        if ev.event_type == "EXPAND":
            frontier_states.discard(s_key)
            expanded_states.add(s_key)
        elif ev.event_type == "PUSH_SUCCESSOR" and ev.next_state is not None:
            succ_key = _state_key(ev.next_state)
            frontier_states.add(succ_key)
        elif ev.event_type == "DISCARD_INFEASIBLE":
            discarded_infeasible_count += 1
        elif ev.event_type == "DISCARD_DOMINATED":
            discarded_dominated_count += 1

        # Crear figura de este cuadro
        fig, (ax_graph, ax_hud) = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(15, 8),
            gridspec_kw={"width_ratios": [2.2, 1.0]},
        )
        fig.patch.set_facecolor("#0f172a")
        ax_graph.set_facecolor("#1e293b")
        ax_hud.set_facecolor("#1e293b")

        # Dibujar gráfico representativo de la frontera vs explorados
        # Disponer nodos en un espacio bidimensional (x: tiempo, y: paquetes pendientes)
        active_x = ev.current_time
        active_y = len(ev.pending_packages)

        # Fondo con nodos expandidos y en frontera
        ax_graph.scatter(
            [0],
            [total_pkgs],
            s=500,
            c=["#38bdf8"],
            marker="s",
            edgecolors="#ffffff",
            linewidths=2,
            label="Origen (D0)",
            zorder=3,
        )

        ax_graph.scatter(
            [ev.current_time],
            [len(ev.pending_packages)],
            s=900,
            c=["#f59e0b" if ev.event_type != "GOAL" else "#10b981"],
            marker="*",
            edgecolors="#ffffff",
            linewidths=2.5,
            label=f"Estado Activo ({ev.event_type})",
            zorder=6,
        )

        ax_graph.text(
            ev.current_time,
            len(ev.pending_packages) - 0.28,
            f"{ev.location}\ng={ev.g:.1f}, h={ev.h:.1f}\nf={ev.f:.1f}",
            color="#f8fafc",
            fontsize=9.5,
            weight="bold",
            ha="center",
            va="top",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#0f172a", edgecolor="#f59e0b", alpha=0.9),
            zorder=7,
        )

        ax_graph.set_xlim(-5, max(120, ev.current_time + 40))
        ax_graph.set_ylim(-0.5, total_pkgs + 0.8)
        ax_graph.set_xlabel("Tiempo Transcurrido (minutos)", color="#cbd5e1", fontsize=11, labelpad=8)
        ax_graph.set_ylabel("Paquetes Pendientes por Entregar", color="#cbd5e1", fontsize=11, labelpad=8)
        ax_graph.tick_params(colors="#94a3b8")
        ax_graph.grid(True, linestyle="--", alpha=0.2, color="#64748b")
        ax_graph.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="#e2e8f0", loc="upper right")
        ax_graph.set_title(
            f"Búsqueda A* - Evento {frame_idx}/{len(selected_events)}: {ev.event_type}",
            color="#f8fafc",
            fontsize=13,
            weight="bold",
            pad=14,
        )

        # Panel HUD lateral
        ax_hud.axis("off")
        hud_lines = [
            "═══════════════════════════════════",
            "  TRAZADOR DE BÚSQUEDA A*",
            "═══════════════════════════════════",
            f" Evento:            {ev.event_type}",
            f" Expansión #:       {ev.expansion_index}",
            f" Ubicación:         {ev.location}",
            f" Reloj Actual:      {ev.current_time} min",
            f" Pendientes:        {len(ev.pending_packages)} / {total_pkgs}",
            "───────────────────────────────────",
            "  FUNCIÓN DE EVALUACIÓN",
            "───────────────────────────────────",
            f" g(n) [Costo real]: {ev.g:.2f}",
            f" h(n) [Heurística]: {ev.h:.2f}",
            f" f(n) = g + h:      {ev.f:.2f}",
            "───────────────────────────────────",
            "  ESTADO DE LA FRONTERA",
            "───────────────────────────────────",
            f" Cola OPEN actual:  {ev.queue_size}",
            f" Nodos Expandidos:  {len(expanded_states)}",
            f" Podas Ventana:     {discarded_infeasible_count}",
            f" Podas Dominancia:  {discarded_dominated_count}",
        ]

        if ev.package_id is not None:
            hud_lines.extend([
                "───────────────────────────────────",
                "  ACCIÓN EVALUADA",
                "───────────────────────────────────",
                f" Paquete:           {ev.package_id}",
                f" Razón / Info:      {ev.discard_reason or 'Factible'}",
            ])

        ax_hud.text(
            0.05,
            0.95,
            "\n".join(hud_lines),
            color="#e2e8f0",
            fontsize=9.2,
            family="monospace",
            va="top",
            ha="left",
        )

        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)
        frames.append(img.copy())

    # Repetir el cuadro final para pausar al final de la animación
    if frames:
        frames.extend([frames[-1].copy()] * 3)

    duration_ms = int(1000 / max(1, fps))
    frames[0].save(
        out_file,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=loop,
    )

    return out_file
