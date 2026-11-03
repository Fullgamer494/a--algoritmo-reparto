"""Visualizador de rutas y topología abstracta para el agente de reparto."""

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
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image

from delivery_agent.algorithms.astar import SearchResult
from delivery_agent.core.transitions import TransitionResult
from delivery_agent.domain.problem import DeliveryProblem


def _format_time(minutes: int) -> str:
    """Convierte minutos enteros a formato HH:MM."""
    hh = minutes // 60
    mm = minutes % 60
    return f"{hh:02d}:{mm:02d}"


def compute_deterministic_layout(
    problem: DeliveryProblem,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """Calcula una disposición 2D abstracta y determinista para las ubicaciones.

    Sitúa el depósito de origen en el centro (0, 0) y distribuye el resto
    de ubicaciones uniformemente en una circunferencia concéntrica, garantizando
    una representación visual reproducible, limpia y sin solapamientos.
    """
    origin = problem.start_location
    other_locations = [loc for loc in sorted(problem.locations) if loc != origin]
    positions: dict[str, tuple[float, float]] = {origin: (0.0, 0.0)}

    total_others = len(other_locations)
    if total_others == 0:
        return positions

    radius = 1.0
    for idx, loc in enumerate(other_locations):
        angle = (2.0 * math.pi * idx / total_others) + (math.pi / 2.0)
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        positions[loc] = (round(x, 4), round(y, 4))

    return positions


def render_route_static(
    problem: DeliveryProblem,
    result: SearchResult,
    output_path: str | Path,
    title: str | None = None,
    dpi: int = 200,
) -> Path:
    """Genera una imagen estática de alta resolución que visualiza la ruta completa.

    Representa la red de ubicaciones, los tiempos de desplazamiento entre nodos,
    la secuencia ordenada de entrega mediante aristas destacadas y un panel HUD
    con los indicadores de calidad del servicio y costo de la función objetivo.

    Args:
        problem: Definición formal del problema de reparto.
        result: Resultado de búsqueda obtenido por el algoritmo.
        output_path: Ruta de destino del archivo PNG generado.
        title: Título personalizado opcional para la gráfica.
        dpi: Densidad de píxeles por pulgada.

    Returns:
        Ruta absoluta del archivo PNG guardado.
    """
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    pos = compute_deterministic_layout(problem)
    fig, (ax_graph, ax_hud) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(15, 8),
        gridspec_kw={"width_ratios": [2.2, 1.0]},
    )

    fig.patch.set_facecolor("#0f172a")  # Slate 900
    ax_graph.set_facecolor("#1e293b")   # Slate 800
    ax_hud.set_facecolor("#1e293b")

    # Mapeo de ubicación a paquete entregado en dicha ubicación
    loc_to_pkg: dict[str, list[str]] = {}
    for pkg in problem.packages:
        loc_to_pkg.setdefault(pkg.destination, []).append(pkg.id)

    # Dibujar aristas base del grafo (desplazamientos posibles)
    for u in problem.locations:
        for v in problem.locations:
            if u != v:
                ux, uy = pos[u]
                vx, vy = pos[v]
                ax_graph.annotate(
                    "",
                    xy=(vx, vy),
                    xytext=(ux, uy),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color="#475569",  # Slate 600
                        lw=0.8,
                        alpha=0.35,
                        shrinkA=18,
                        shrinkB=18,
                    ),
                )

    # Si se encontró solución factible, trazar la ruta con aristas destacadas
    if result.is_feasible and result.transitions:
        route_edges: list[tuple[str, str, int, str]] = []
        for step_idx, step in enumerate(result.transitions, start=1):
            route_edges.append((step.origin, step.destination, step_idx, step.package_id))

        colors = ["#38bdf8", "#818cf8", "#c084fc", "#f472b6", "#fb923c", "#facc15"]
        for u, v, step_num, pkg_id in route_edges:
            ux, uy = pos[u]
            vx, vy = pos[v]
            col = colors[(step_num - 1) % len(colors)]
            
            # Arco curvado para distinguir idas y vueltas
            connectionstyle = "arc3,rad=0.12" if u != problem.start_location and v != problem.start_location else "arc3,rad=0.08"
            ax_graph.annotate(
                "",
                xy=(vx, vy),
                xytext=(ux, uy),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=col,
                    lw=2.8,
                    mutation_scale=16,
                    shrinkA=18,
                    shrinkB=18,
                    connectionstyle=connectionstyle,
                ),
            )
            # Etiqueta de paso en el punto medio del segmento
            mid_x = (ux + vx) / 2.0
            mid_y = (uy + vy) / 2.0
            ax_graph.text(
                mid_x,
                mid_y,
                f"#{step_num} ({pkg_id})",
                color="#ffffff",
                fontsize=9,
                weight="bold",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=col, edgecolor="#0f172a", alpha=0.9),
                zorder=5,
            )

    # Dibujar nodos de ubicaciones
    visited_locations = {step.destination: idx for idx, step in enumerate(result.transitions, start=1)}

    for loc, (x, y) in pos.items():
        is_origin = (loc == problem.start_location)
        if is_origin:
            node_color = "#10b981"  # Emerald 500
            node_size = 650
            node_shape = "s"
            label = f"{loc}\n[Depósito]"
        else:
            order_badge = f"#{visited_locations[loc]} " if loc in visited_locations else ""
            pkgs_str = ",".join(loc_to_pkg.get(loc, []))
            node_color = "#6366f1" if loc in visited_locations else "#64748b"  # Indigo vs Slate 500
            node_size = 500
            node_shape = "o"
            label = f"{order_badge}{loc}\n({pkgs_str})"

        ax_graph.scatter(
            [x],
            [y],
            s=node_size,
            c=[node_color],
            marker=node_shape,
            edgecolors="#ffffff",
            linewidths=2,
            zorder=4,
        )
        ax_graph.text(
            x,
            y - 0.16,
            label,
            color="#f8fafc",
            fontsize=9,
            weight="bold",
            ha="center",
            va="top",
            zorder=5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#1e293b", edgecolor=node_color, alpha=0.85),
        )

    ax_graph.set_xlim(-1.45, 1.45)
    ax_graph.set_ylim(-1.45, 1.45)
    ax_graph.set_aspect("equal")
    ax_graph.axis("off")
    ax_graph.set_title(
        title or f"Ruta Planificada - Algoritmo: {result.algorithm}",
        color="#f8fafc",
        fontsize=14,
        weight="bold",
        pad=16,
    )

    # Panel lateral de métricas (HUD)
    ax_hud.axis("off")
    summary = result.cost_summary
    
    hud_lines = [
        "═══════════════════════════════════",
        "  RESUMEN DE PLANIFICACIÓN",
        "═══════════════════════════════════",
        f" Algoritmo:         {result.algorithm}",
        f" Factible:          {'SÍ' if result.is_feasible else 'NO'}",
        f" Paquetes:          {len(result.transitions)} / {len(problem.packages)}",
    ]

    if result.is_feasible and summary is not None:
        route_str = " -> ".join([problem.start_location] + list(result.package_order))
        on_promise_count = sum(1 for step in result.transitions if step.tardiness == 0)
        total_delivered = len(result.transitions)
        on_promise_pct = (on_promise_count / max(1, total_delivered)) * 100.0

        hud_lines.extend([
            f" Secuencia:         {route_str}",
            "───────────────────────────────────",
            "  INDICADORES OPERATIVOS",
            "───────────────────────────────────",
            f" Costo Objetivo:    {summary.objective_cost:.2f}",
            f" Duración Total:    {summary.total_elapsed_time} min ({_format_time(summary.total_elapsed_time)})",
            f" Tiempo en Viaje:   {summary.total_travel_time} min",
            f" Tiempo de Espera:  {summary.total_wait_time} min",
            f" Tiempo de Servicio:{summary.total_service_time} min",
            "───────────────────────────────────",
            "  CALIDAD Y PUNTUALIDAD",
            "───────────────────────────────────",
            f" Retraso Total:     {summary.total_tardiness} min",
            f" Retraso Ponderado: {summary.weighted_tardiness:.2f}",
            f" Penalización (α):  {summary.total_penalty:.2f} (α={summary.alpha})",
            f" A tiempo (Promesa):{on_promise_count}/{total_delivered} ({on_promise_pct:.1f}%)",
            f" En ventana:        {total_delivered}/{len(problem.packages)} (100.0%)",
            "───────────────────────────────────",
            "  RENDIMIENTO DEL MOTOR",
            "───────────────────────────────────",
            f" Nodos Expandidos:  {result.metrics.states_expanded}",
            f" Nodos Generados:   {result.metrics.states_generated}",
            f" Cola Máxima:       {result.metrics.max_queue_size}",
            f" Tiempo Ejecución:  {result.metrics.execution_time_seconds * 1000.0:.2f} ms",
        ])
    else:
        hud_lines.extend([
            "───────────────────────────────────",
            "  ESTADO DE LA BÚSQUEDA",
            "───────────────────────────────────",
            " No se halló solución factible.",
            " Las ventanas de entrega impiden",
            " completar la ruta con los",
            " parámetros configurados.",
            "───────────────────────────────────",
            f" Nodos Expandidos:  {result.metrics.states_expanded}",
            f" Nodos Podados:     {result.metrics.infeasible_transitions_discarded}",
            f" Tiempo Ejecución:  {result.metrics.execution_time_seconds * 1000.0:.2f} ms",
        ])

    ax_hud.text(
        0.05,
        0.95,
        "\n".join(hud_lines),
        color="#e2e8f0",
        fontsize=9.5,
        family="monospace",
        va="top",
        ha="left",
    )

    plt.tight_layout()
    fig.savefig(out_file, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    return out_file


def _render_single_animation_frame(
    problem: DeliveryProblem,
    result: SearchResult,
    pos: dict[str, tuple[float, float]],
    current_step_idx: int,
    phase: str,  # "TRAVEL", "WAIT", "SERVICE", "COMPLETED", "INIT"
    current_time: int,
    agent_loc: str,
    completed_pkgs: set[str],
    current_step: TransitionResult | None = None,
) -> Image.Image:
    """Renderiza un cuadro individual de la animación en memoria como PIL Image."""
    fig, (ax_graph, ax_hud) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(15, 8),
        gridspec_kw={"width_ratios": [2.2, 1.0]},
    )

    fig.patch.set_facecolor("#0f172a")
    ax_graph.set_facecolor("#1e293b")
    ax_hud.set_facecolor("#1e293b")

    loc_to_pkg: dict[str, list[str]] = {}
    for pkg in problem.packages:
        loc_to_pkg.setdefault(pkg.destination, []).append(pkg.id)

    # Dibujar aristas base del grafo
    for u in problem.locations:
        for v in problem.locations:
            if u != v:
                ux, uy = pos[u]
                vx, vy = pos[v]
                ax_graph.annotate(
                    "",
                    xy=(vx, vy),
                    xytext=(ux, uy),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color="#475569",
                        lw=0.7,
                        alpha=0.3,
                        shrinkA=16,
                        shrinkB=16,
                    ),
                )

    # Dibujar aristas de pasos completados previamente
    colors = ["#38bdf8", "#818cf8", "#c084fc", "#f472b6", "#fb923c", "#facc15"]
    for idx in range(current_step_idx):
        prev_step = result.transitions[idx]
        u, v = prev_step.origin, prev_step.destination
        ux, uy = pos[u]
        vx, vy = pos[v]
        col = colors[idx % len(colors)]
        ax_graph.annotate(
            "",
            xy=(vx, vy),
            xytext=(ux, uy),
            arrowprops=dict(
                arrowstyle="-|>",
                color=col,
                lw=2.5,
                alpha=0.75,
                shrinkA=18,
                shrinkB=18,
                connectionstyle="arc3,rad=0.1",
            ),
        )

    # Dibujar arista del paso activo si está en tránsito
    if current_step is not None and phase in ("TRAVEL", "WAIT", "SERVICE", "COMPLETED"):
        u, v = current_step.origin, current_step.destination
        ux, uy = pos[u]
        vx, vy = pos[v]
        edge_col = "#f59e0b" if phase == "TRAVEL" else "#10b981"
        ax_graph.annotate(
            "",
            xy=(vx, vy),
            xytext=(ux, uy),
            arrowprops=dict(
                arrowstyle="-|>",
                color=edge_col,
                lw=3.8,
                mutation_scale=20,
                shrinkA=18,
                shrinkB=18,
                connectionstyle="arc3,rad=0.1",
            ),
        )

    # Dibujar nodos de ubicaciones
    for loc, (x, y) in pos.items():
        is_origin = (loc == problem.start_location)
        is_delivered = any(pkg_id in completed_pkgs for pkg_id in loc_to_pkg.get(loc, []))

        if is_origin:
            node_color = "#10b981"
            node_size = 600
            node_shape = "s"
            label = f"{loc}\n[Depósito]"
        elif is_delivered:
            node_color = "#10b981"  # Verde entregado
            node_size = 500
            node_shape = "o"
            pkgs = ",".join(loc_to_pkg.get(loc, []))
            label = f"{loc} ✓\n({pkgs})"
        else:
            node_color = "#64748b"
            node_size = 450
            node_shape = "o"
            pkgs = ",".join(loc_to_pkg.get(loc, []))
            label = f"{loc}\n({pkgs})"

        ax_graph.scatter(
            [x],
            [y],
            s=node_size,
            c=[node_color],
            marker=node_shape,
            edgecolors="#ffffff",
            linewidths=2,
            zorder=4,
        )
        ax_graph.text(
            x,
            y - 0.16,
            label,
            color="#f8fafc",
            fontsize=9,
            weight="bold",
            ha="center",
            va="top",
            zorder=5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#1e293b", edgecolor=node_color, alpha=0.85),
        )

    # Dibujar agente en su posición actual (con halo brillante)
    ag_x, ag_y = pos[agent_loc]
    ax_graph.scatter(
        [ag_x],
        [ag_y],
        s=1200,
        c=["#f43f5e"],  # Rose 500
        marker="*",
        edgecolors="#ffffff",
        linewidths=2.5,
        zorder=8,
    )
    ax_graph.text(
        ag_x,
        ag_y + 0.14,
        "REPARTIDOR",
        color="#f43f5e",
        fontsize=9,
        weight="bold",
        ha="center",
        va="bottom",
        zorder=9,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#0f172a", edgecolor="#f43f5e", alpha=0.9),
    )

    ax_graph.set_xlim(-1.45, 1.45)
    ax_graph.set_ylim(-1.45, 1.45)
    ax_graph.set_aspect("equal")
    ax_graph.axis("off")
    ax_graph.set_title(
        f"Simulación de Reparto: {result.algorithm} - Paso {current_step_idx + (1 if current_step else 0)}/{len(result.transitions)}",
        color="#f8fafc",
        fontsize=13,
        weight="bold",
        pad=16,
    )

    # Panel lateral de simulación (HUD)
    ax_hud.axis("off")
    phase_labels = {
        "INIT": "EN ORIGEN (PREPARACIÓN)",
        "TRAVEL": "DESPLAZAMIENTO EN RUTA",
        "WAIT": "ESPERA DE VENTANA",
        "SERVICE": "REALIZANDO SERVICIO",
        "COMPLETED": "ENTREGA COMPLETADA",
    }
    phase_colors = {
        "INIT": "#38bdf8",
        "TRAVEL": "#fbbf24",
        "WAIT": "#fb923c",
        "SERVICE": "#a855f7",
        "COMPLETED": "#34d399",
    }

    pending_count = len(problem.packages) - len(completed_pkgs)
    hud_lines = [
        "═══════════════════════════════════",
        "  SIMULADOR DE REPARTO EN VIVO",
        "═══════════════════════════════════",
        f" Reloj Global:      {current_time} min ({_format_time(current_time)})",
        f" Fase Actual:       {phase_labels.get(phase, phase)}",
        f" Ubicación Agente:  {agent_loc}",
        "───────────────────────────────────",
        "  ESTADO DE PAQUETES",
        "───────────────────────────────────",
        f" Entregados:        {len(completed_pkgs)} / {len(problem.packages)}",
        f" Pendientes:        {pending_count}",
    ]

    if current_step is not None:
        pkg = problem.get_package(current_step.package_id)
        hud_lines.extend([
            "───────────────────────────────────",
            "  DETALLE DEL PASO ACTUAL",
            "───────────────────────────────────",
            f" Paquete:           {pkg.id} (Destino: {pkg.destination})",
            f" Ventana Horaria:   [{pkg.window_start} - {pkg.window_end}] min ({_format_time(pkg.window_start)} - {_format_time(pkg.window_end)})",
            f" Hora Prometida:    {pkg.promised_time} min ({_format_time(pkg.promised_time)})",
            f" Prioridad (ρ):     {pkg.priority}",
            f" Viaje:             {current_step.travel_time} min ({current_step.origin} -> {current_step.destination})",
            f" Llegada:           {current_step.arrival_time} min ({_format_time(current_step.arrival_time)})",
            f" Espera en Destino: {current_step.wait_time} min",
            f" Inicio Servicio:   {current_step.service_start_time} min",
            f" Duración Servicio: {current_step.service_time} min",
            f" Fin de Entrega:    {current_step.completion_time} min ({_format_time(current_step.completion_time)})",
            f" Retraso:           {current_step.tardiness} min",
            f" Penalización Paso: {current_step.penalty:.2f}",
        ])
    else:
        hud_lines.extend([
            "───────────────────────────────────",
            "  DETALLE DEL PASO ACTUAL",
            "───────────────────────────────────",
            " Inicio del recorrido en depósito.",
            " El vehículo se prepara para salir.",
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
    fig.savefig(buf, format="png", dpi=130, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf)
    # Copia profunda de la imagen para desacoplar del buffer en memoria
    return image.copy()


def animate_route(
    problem: DeliveryProblem,
    result: SearchResult,
    output_path: str | Path,
    fps: int = 1,
    loop: int = 0,
) -> Path:
    """Genera una animación paso a paso (GIF) que ilustra el recorrido del repartidor.

    Desglosa cada paso en las cuatro fases especificadas en el protocolo de la etapa 6:
    1. Desplazamiento hacia el siguiente destino (tiempo de desplazamiento).
    2. Espera en el destino (si la llegada ocurre antes de la apertura de la ventana).
    3. Ejecución del servicio (duración del servicio).
    4. Finalización de la entrega (actualización del reloj y del estado del paquete).

    Args:
        problem: Definición del problema.
        result: Resultado de la búsqueda.
        output_path: Ruta de salida del archivo GIF.
        fps: Cuadros por segundo de la animación (por defecto 1 fps para inspección clara).
        loop: Número de repeticiones (0 = bucle infinito).

    Returns:
        Ruta absoluta del archivo GIF generado.
    """
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if not result.is_feasible or not result.transitions:
        # Si no es factible, generar un cuadro estático de estado no factible
        static_png = out_file.with_suffix(".png")
        render_route_static(problem, result, static_png, title="Ruta No Factible")
        img = Image.open(static_png)
        img.save(out_file, format="GIF")
        return out_file

    pos = compute_deterministic_layout(problem)
    frames: list[Image.Image] = []

    # Cuadro inicial: En depósito antes de partir
    completed_pkgs: set[str] = set()
    initial_frame = _render_single_animation_frame(
        problem=problem,
        result=result,
        pos=pos,
        current_step_idx=0,
        phase="INIT",
        current_time=problem.start_time,
        agent_loc=problem.start_location,
        completed_pkgs=completed_pkgs,
        current_step=None,
    )
    frames.append(initial_frame)

    # Iterar sobre cada transición del plan de reparto
    for step_idx, step in enumerate(result.transitions):
        # 1. Fase de desplazamiento: el vehículo llega al destino
        travel_frame = _render_single_animation_frame(
            problem=problem,
            result=result,
            pos=pos,
            current_step_idx=step_idx,
            phase="TRAVEL",
            current_time=step.arrival_time,
            agent_loc=step.destination,
            completed_pkgs=completed_pkgs,
            current_step=step,
        )
        frames.append(travel_frame)

        # 2. Fase de espera: si llegó antes de la apertura de ventana
        if step.wait_time > 0:
            wait_frame = _render_single_animation_frame(
                problem=problem,
                result=result,
                pos=pos,
                current_step_idx=step_idx,
                phase="WAIT",
                current_time=step.service_start_time,
                agent_loc=step.destination,
                completed_pkgs=completed_pkgs,
                current_step=step,
            )
            frames.append(wait_frame)

        # 3. Fase de servicio: el paquete está siendo atendido
        service_frame = _render_single_animation_frame(
            problem=problem,
            result=result,
            pos=pos,
            current_step_idx=step_idx,
            phase="SERVICE",
            current_time=step.service_start_time,
            agent_loc=step.destination,
            completed_pkgs=completed_pkgs,
            current_step=step,
        )
        frames.append(service_frame)

        # 4. Fase de finalización: entrega confirmada
        completed_pkgs.add(step.package_id)
        completion_frame = _render_single_animation_frame(
            problem=problem,
            result=result,
            pos=pos,
            current_step_idx=step_idx,
            phase="COMPLETED",
            current_time=step.completion_time,
            agent_loc=step.destination,
            completed_pkgs=completed_pkgs,
            current_step=step,
        )
        frames.append(completion_frame)

    # Cuadro final con pausa extra (repetir el último cuadro 2 veces)
    if frames:
        frames.extend([frames[-1].copy(), frames[-1].copy()])

    # Guardar como GIF animado mediante Pillow
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
