"""Generador de gráficas experimentales y visualizaciones del benchmark."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import tempfile
from typing import Any

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _scenario_display_name(raw_name: str) -> str:
    """Acorta el nombre del escenario para una lectura limpia en ejes y leyendas."""
    name = raw_name.replace("escenario_", "Esc. ")
    name = name.replace("_5_paquetes", "")
    name = name.replace("_6_paquetes", "")
    name = name.replace("_paquetes", "")
    return name


def load_benchmark_data(
    summary_csv_path: str | Path,
    details_json_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Carga y procesa los registros del archivo summary.csv y opcionalmente details.json."""
    csv_file = Path(summary_csv_path).resolve()
    if not csv_file.exists():
        raise FileNotFoundError(f"No se encontró el archivo de benchmark: {csv_file}")

    rows: list[dict[str, Any]] = []
    with open(csv_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            parsed: dict[str, Any] = {
                "scenario_name": r["scenario_name"],
                "algorithm": r["algorithm"],
                "alpha": float(r["alpha"]),
                "solved": r["solved"].strip().lower() == "true",
                "package_order": r["package_order"],
                "objective_cost": float(r["objective_cost"]) if r["objective_cost"] else None,
                "total_elapsed_time": int(r["total_elapsed_time"]) if r["total_elapsed_time"] else None,
                "total_travel_time": int(r["total_travel_time"]) if r["total_travel_time"] else None,
                "total_wait_time": int(r["total_wait_time"]) if r["total_wait_time"] else None,
                "total_service_time": int(r["total_service_time"]) if r["total_service_time"] else None,
                "total_tardiness": int(r["total_tardiness"]) if r["total_tardiness"] else None,
                "weighted_tardiness": float(r["weighted_tardiness"]) if r["weighted_tardiness"] else None,
                "total_penalty": float(r["total_penalty"]) if r["total_penalty"] else None,
                "mean_execution_time_seconds": float(r["mean_execution_time_seconds"]) if r["mean_execution_time_seconds"] else 0.0,
                "states_expanded": int(r["states_expanded_or_feasible"]) if r["states_expanded_or_feasible"] else 0,
                "states_generated": int(r["states_generated_or_permutations"]) if r["states_generated_or_permutations"] else 0,
                "max_queue_size": 0,
            }
            rows.append(parsed)

    # Si se proporciona details.json, enriquecer con max_queue_size
    if details_json_path is not None:
        json_file = Path(details_json_path).resolve()
        if json_file.exists():
            with open(json_file, mode="r", encoding="utf-8") as jf:
                details_data = json.load(jf)
                # Crear índice rápido por (scenario_name, algorithm, alpha)
                queue_map: dict[tuple[str, str, float], int] = {}
                for entry in details_data:
                    sc = entry.get("scenario_name")
                    alg = entry.get("algorithm")
                    al = float(entry.get("alpha", 0.0))
                    sm = entry.get("search_metrics")
                    if sm and "max_queue_size" in sm:
                        queue_map[(sc, alg, al)] = int(sm["max_queue_size"])

                for row in rows:
                    key = (row["scenario_name"], row["algorithm"], row["alpha"])
                    if key in queue_map:
                        row["max_queue_size"] = queue_map[key]

    return rows


def plot_objective_cost_comparison(
    data: list[dict[str, Any]],
    output_path: str | Path,
    alpha_filter: float = 1.0,
    dpi: int = 200,
) -> Path:
    """Gráfica 1: Comparación del costo objetivo J(π) entre algoritmos (α=1.0)."""
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    filtered = [r for r in data if abs(r["alpha"] - alpha_filter) < 1e-6]
    scenarios = sorted(list({r["scenario_name"] for r in filtered}))
    algorithms = ["A*", "Dijkstra", "Greedy", "Exhaustive"]
    colors = {"A*": "#3b82f6", "Dijkstra": "#6366f1", "Greedy": "#f59e0b", "Exhaustive": "#10b981"}

    x = np.arange(len(scenarios))
    width = 0.2
    fig, ax = plt.subplots(figsize=(14, 6.5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    for i, alg in enumerate(algorithms):
        alg_costs = []
        for sc in scenarios:
            match = next((r for r in filtered if r["scenario_name"] == sc and r["algorithm"] == alg), None)
            if match and match["solved"] and match["objective_cost"] is not None:
                alg_costs.append(match["objective_cost"])
            else:
                alg_costs.append(0.0)

        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, alg_costs, width, label=alg, color=colors[alg], edgecolor="#ffffff", linewidth=0.5)

        # Anotación en barras infactibles
        for bar_idx, (sc, cost) in enumerate(zip(scenarios, alg_costs)):
            match = next((r for r in filtered if r["scenario_name"] == sc and r["algorithm"] == alg), None)
            if not match or not match["solved"]:
                bar_x = x[bar_idx] + offset
                ax.text(bar_x, 4.0, "N/F", color="#ef4444", fontsize=7.5, ha="center", weight="bold", rotation=90)

    ax.set_ylabel(f"Costo Función Objetivo J(π) [α={alpha_filter}]", color="#e2e8f0", fontsize=11, labelpad=8)
    ax.set_title(f"Comparación de Costo Objetivo por Escenario y Algoritmo (α={alpha_filter})", color="#f8fafc", fontsize=13, weight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([_scenario_display_name(sc) for sc in scenarios], color="#cbd5e1", rotation=25, ha="right", fontsize=9.5)
    ax.tick_params(colors="#94a3b8")
    ax.grid(True, linestyle="--", alpha=0.15, color="#64748b", axis="y")
    ax.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="#e2e8f0")

    plt.tight_layout()
    fig.savefig(out_file, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_file


def plot_execution_time_comparison(
    data: list[dict[str, Any]],
    output_path: str | Path,
    alpha_filter: float = 1.0,
    dpi: int = 200,
) -> Path:
    """Gráfica 2: Comparación del tiempo medio de ejecución (escala logarítmica)."""
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    filtered = [r for r in data if abs(r["alpha"] - alpha_filter) < 1e-6]
    scenarios = sorted(list({r["scenario_name"] for r in filtered}))
    algorithms = ["A*", "Dijkstra", "Greedy", "Exhaustive"]
    colors = {"A*": "#3b82f6", "Dijkstra": "#6366f1", "Greedy": "#f59e0b", "Exhaustive": "#10b981"}

    x = np.arange(len(scenarios))
    width = 0.2
    fig, ax = plt.subplots(figsize=(14, 6.5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    for i, alg in enumerate(algorithms):
        times = []
        for sc in scenarios:
            match = next((r for r in filtered if r["scenario_name"] == sc and r["algorithm"] == alg), None)
            val = match["mean_execution_time_seconds"] * 1000.0 if match else 0.001
            times.append(max(0.001, val))

        offset = (i - 1.5) * width
        ax.bar(x + offset, times, width, label=alg, color=colors[alg], edgecolor="#ffffff", linewidth=0.5)

    ax.set_yscale("log")
    ax.set_ylabel("Tiempo Medio de Ejecución [milisegundos] (Escala Log)", color="#e2e8f0", fontsize=11, labelpad=8)
    ax.set_title(f"Tiempo de Cómputo por Escenario y Algoritmo (α={alpha_filter})", color="#f8fafc", fontsize=13, weight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([_scenario_display_name(sc) for sc in scenarios], color="#cbd5e1", rotation=25, ha="right", fontsize=9.5)
    ax.tick_params(colors="#94a3b8")
    ax.grid(True, linestyle="--", alpha=0.15, color="#64748b", which="both", axis="y")
    ax.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="#e2e8f0")

    plt.tight_layout()
    fig.savefig(out_file, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_file


def plot_states_expanded_comparison(
    data: list[dict[str, Any]],
    output_path: str | Path,
    alpha_filter: float = 1.0,
    dpi: int = 200,
) -> Path:
    """Gráfica 3: Nodos expandidos (A* vs Dijkstra vs Greedy), mostrando la eficiencia de A*."""
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    filtered = [r for r in data if abs(r["alpha"] - alpha_filter) < 1e-6]
    scenarios = sorted(list({r["scenario_name"] for r in filtered}))
    algorithms = ["A*", "Dijkstra"]
    colors = {"A*": "#38bdf8", "Dijkstra": "#818cf8"}

    x = np.arange(len(scenarios))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 6.5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    for i, alg in enumerate(algorithms):
        expanded = []
        for sc in scenarios:
            match = next((r for r in filtered if r["scenario_name"] == sc and r["algorithm"] == alg), None)
            expanded.append(match["states_expanded"] if match else 0)

        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, expanded, width, label=alg, color=colors[alg], edgecolor="#ffffff", linewidth=0.5)

        for bar, val in zip(bars, expanded):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2.0, val + 2, f"{val}", ha="center", va="bottom", color="#e2e8f0", fontsize=8.5)

    ax.set_ylabel("Cantidad de Estados Expandidos", color="#e2e8f0", fontsize=11, labelpad=8)
    ax.set_title(f"Eficiencia de Búsqueda: Nodos Expandidos A* (Heurística) vs Dijkstra (Ciega)", color="#f8fafc", fontsize=13, weight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([_scenario_display_name(sc) for sc in scenarios], color="#cbd5e1", rotation=25, ha="right", fontsize=9.5)
    ax.tick_params(colors="#94a3b8")
    ax.grid(True, linestyle="--", alpha=0.15, color="#64748b", axis="y")
    ax.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="#e2e8f0")

    plt.tight_layout()
    fig.savefig(out_file, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_file


def plot_max_queue_size_comparison(
    data: list[dict[str, Any]],
    output_path: str | Path,
    alpha_filter: float = 1.0,
    dpi: int = 200,
) -> Path:
    """Gráfica 4: Tamaño máximo de la cola de prioridad OPEN (huella de memoria)."""
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    filtered = [r for r in data if abs(r["alpha"] - alpha_filter) < 1e-6]
    scenarios = sorted(list({r["scenario_name"] for r in filtered}))
    algorithms = ["A*", "Dijkstra"]
    colors = {"A*": "#38bdf8", "Dijkstra": "#a855f7"}

    x = np.arange(len(scenarios))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 6.5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    for i, alg in enumerate(algorithms):
        q_sizes = []
        for sc in scenarios:
            match = next((r for r in filtered if r["scenario_name"] == sc and r["algorithm"] == alg), None)
            q_sizes.append(match["max_queue_size"] if match else 0)

        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, q_sizes, width, label=alg, color=colors[alg], edgecolor="#ffffff", linewidth=0.5)

        for bar, val in zip(bars, q_sizes):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2.0, val + 2, f"{val}", ha="center", va="bottom", color="#e2e8f0", fontsize=8.5)

    ax.set_ylabel("Tamaño Máximo de la Cola (Frontera OPEN)", color="#e2e8f0", fontsize=11, labelpad=8)
    ax.set_title(f"Uso de Memoria: Tamaño Máximo de Cola OPEN (A* vs Dijkstra, α={alpha_filter})", color="#f8fafc", fontsize=13, weight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([_scenario_display_name(sc) for sc in scenarios], color="#cbd5e1", rotation=25, ha="right", fontsize=9.5)
    ax.tick_params(colors="#94a3b8")
    ax.grid(True, linestyle="--", alpha=0.15, color="#64748b", axis="y")
    ax.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="#e2e8f0")

    plt.tight_layout()
    fig.savefig(out_file, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_file


def plot_cost_sensitivity_alpha(
    data: list[dict[str, Any]],
    output_path: str | Path,
    algorithm: str = "A*",
    dpi: int = 200,
) -> Path:
    """Gráfica 5: Sensibilidad del costo objetivo respecto al parámetro de penalización alpha."""
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    scenarios = sorted(list({r["scenario_name"] for r in data}))
    palette = ["#38bdf8", "#818cf8", "#c084fc", "#f472b6", "#fb923c", "#facc15", "#34d399", "#ef4444"]

    alphas = [0.0, 0.5, 1.0, 2.0]

    for idx, sc in enumerate(scenarios):
        costs = []
        for al in alphas:
            match = next((r for r in data if r["scenario_name"] == sc and r["algorithm"] == algorithm and abs(r["alpha"] - al) < 1e-6), None)
            if match and match["solved"] and match["objective_cost"] is not None:
                costs.append(match["objective_cost"])
            else:
                costs.append(np.nan)

        if not all(np.isnan(c) for c in costs):
            col = palette[idx % len(palette)]
            ax.plot(alphas, costs, marker="o", lw=2.2, label=_scenario_display_name(sc), color=col)

    ax.set_xlabel("Parámetro de Ponderación α", color="#e2e8f0", fontsize=11, labelpad=8)
    ax.set_ylabel("Costo Objetivo Total J(π)", color="#e2e8f0", fontsize=11, labelpad=8)
    ax.set_title(f"Sensibilidad del Costo Objetivo Total vs Parámetro α ({algorithm})", color="#f8fafc", fontsize=13, weight="bold", pad=12)
    ax.set_xticks(alphas)
    ax.tick_params(colors="#94a3b8")
    ax.grid(True, linestyle="--", alpha=0.15, color="#64748b")
    ax.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="#e2e8f0", loc="upper left", bbox_to_anchor=(1.02, 1))

    plt.tight_layout()
    fig.savefig(out_file, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return out_file


def plot_elapsed_time_sensitivity_alpha(
    data: list[dict[str, Any]],
    output_path: str | Path,
    algorithm: str = "A*",
    dpi: int = 200,
) -> Path:
    """Gráfica 6: Sensibilidad del tiempo total transcurrido (duración operativa) vs alpha."""
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    scenarios = sorted(list({r["scenario_name"] for r in data}))
    palette = ["#38bdf8", "#818cf8", "#c084fc", "#f472b6", "#fb923c", "#facc15", "#34d399", "#ef4444"]
    alphas = [0.0, 0.5, 1.0, 2.0]

    for idx, sc in enumerate(scenarios):
        durations = []
        for al in alphas:
            match = next((r for r in data if r["scenario_name"] == sc and r["algorithm"] == algorithm and abs(r["alpha"] - al) < 1e-6), None)
            if match and match["solved"] and match["total_elapsed_time"] is not None:
                durations.append(match["total_elapsed_time"])
            else:
                durations.append(np.nan)

        if not all(np.isnan(d) for d in durations):
            col = palette[idx % len(palette)]
            ax.plot(alphas, durations, marker="s", lw=2.2, label=_scenario_display_name(sc), color=col)

    ax.set_xlabel("Parámetro de Ponderación α", color="#e2e8f0", fontsize=11, labelpad=8)
    ax.set_ylabel("Tiempo Total Transcurrido [minutos]", color="#e2e8f0", fontsize=11, labelpad=8)
    ax.set_title(f"Compromiso Operativo: Tiempo Total de Ruta vs Parámetro α ({algorithm})", color="#f8fafc", fontsize=13, weight="bold", pad=12)
    ax.set_xticks(alphas)
    ax.tick_params(colors="#94a3b8")
    ax.grid(True, linestyle="--", alpha=0.15, color="#64748b")
    ax.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="#e2e8f0", loc="upper left", bbox_to_anchor=(1.02, 1))

    plt.tight_layout()
    fig.savefig(out_file, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return out_file


def plot_weighted_tardiness_sensitivity_alpha(
    data: list[dict[str, Any]],
    output_path: str | Path,
    algorithm: str = "A*",
    dpi: int = 200,
) -> Path:
    """Gráfica 7: Sensibilidad del retraso ponderado sum(rho_i * T_i) respecto a alpha."""
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    scenarios = sorted(list({r["scenario_name"] for r in data}))
    palette = ["#38bdf8", "#818cf8", "#c084fc", "#f472b6", "#fb923c", "#facc15", "#34d399", "#ef4444"]
    alphas = [0.0, 0.5, 1.0, 2.0]

    for idx, sc in enumerate(scenarios):
        tardiness = []
        for al in alphas:
            match = next((r for r in data if r["scenario_name"] == sc and r["algorithm"] == algorithm and abs(r["alpha"] - al) < 1e-6), None)
            if match and match["solved"] and match["weighted_tardiness"] is not None:
                tardiness.append(match["weighted_tardiness"])
            else:
                tardiness.append(np.nan)

        if not all(np.isnan(t) for t in tardiness):
            col = palette[idx % len(palette)]
            ax.plot(alphas, tardiness, marker="^", lw=2.2, label=_scenario_display_name(sc), color=col)

    ax.set_xlabel("Parámetro de Ponderación α", color="#e2e8f0", fontsize=11, labelpad=8)
    ax.set_ylabel("Retraso Ponderado Total Σ(ρ_i · T_i) [min]", color="#e2e8f0", fontsize=11, labelpad=8)
    ax.set_title(f"Mitigación del Retraso Ponderado según el Valor de α ({algorithm})", color="#f8fafc", fontsize=13, weight="bold", pad=12)
    ax.set_xticks(alphas)
    ax.tick_params(colors="#94a3b8")
    ax.grid(True, linestyle="--", alpha=0.15, color="#64748b")
    ax.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="#e2e8f0", loc="upper left", bbox_to_anchor=(1.02, 1))

    plt.tight_layout()
    fig.savefig(out_file, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return out_file


def plot_greedy_suboptimality_gap(
    data: list[dict[str, Any]],
    output_path: str | Path,
    alpha_filter: float = 1.0,
    dpi: int = 200,
) -> Path:
    """Gráfica 8: Brecha de suboptimizad porcentual del algoritmo Greedy respecto al óptimo."""
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    filtered = [r for r in data if abs(r["alpha"] - alpha_filter) < 1e-6]
    scenarios = sorted(list({r["scenario_name"] for r in filtered}))

    gaps: list[float] = []
    labels: list[str] = []
    annotations: list[str] = []

    for sc in scenarios:
        opt_match = next((r for r in filtered if r["scenario_name"] == sc and r["algorithm"] == "A*"), None)
        greedy_match = next((r for r in filtered if r["scenario_name"] == sc and r["algorithm"] == "Greedy"), None)

        labels.append(_scenario_display_name(sc))
        if opt_match and opt_match["solved"] and opt_match["objective_cost"] is not None:
            opt_cost = opt_match["objective_cost"]
            if greedy_match and greedy_match["solved"] and greedy_match["objective_cost"] is not None:
                gr_cost = greedy_match["objective_cost"]
                gap = ((gr_cost - opt_cost) / max(1e-6, opt_cost)) * 100.0
                gaps.append(max(0.0, gap))
                annotations.append(f"+{gap:.1f}%")
            else:
                # Greedy no encontró solución
                gaps.append(0.0)
                annotations.append("Greedy N/F")
        else:
            # Ni siquiera óptimo factible
            gaps.append(0.0)
            annotations.append("Sin Sol.")

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    x = np.arange(len(scenarios))
    bars = ax.bar(x, gaps, color="#f59e0b", edgecolor="#ffffff", linewidth=0.6, width=0.55)

    for bar, ann in zip(bars, annotations):
        h = bar.get_height()
        y_pos = h + 1.2 if h > 0 else 1.2
        col = "#facc15" if "%" in ann else "#ef4444"
        ax.text(bar.get_x() + bar.get_width() / 2.0, y_pos, ann, ha="center", va="bottom", color=col, fontsize=9, weight="bold")

    ax.set_ylabel("Brecha de Suboptimidad [%] vs A* (Óptimo)", color="#e2e8f0", fontsize=11, labelpad=8)
    ax.set_title(f"Brecha de Suboptimidad de Vecino Cercano (Greedy) vs A* (α={alpha_filter})", color="#f8fafc", fontsize=13, weight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="#cbd5e1", rotation=25, ha="right", fontsize=9.5)
    ax.tick_params(colors="#94a3b8")
    ax.grid(True, linestyle="--", alpha=0.15, color="#64748b", axis="y")

    plt.tight_layout()
    fig.savefig(out_file, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_file


def generate_all_benchmark_charts(
    summary_csv_path: str | Path,
    output_dir: str | Path,
    details_json_path: str | Path | None = None,
) -> list[Path]:
    """Genera la suite completa de las 8 gráficas experimentales del proyecto."""
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_benchmark_data(summary_csv_path, details_json_path)
    generated: list[Path] = []

    generated.append(plot_objective_cost_comparison(data, out_dir / "01_objective_cost_comparison.png"))
    generated.append(plot_execution_time_comparison(data, out_dir / "02_execution_time_comparison.png"))
    generated.append(plot_states_expanded_comparison(data, out_dir / "03_states_expanded_comparison.png"))
    generated.append(plot_max_queue_size_comparison(data, out_dir / "04_max_queue_size_comparison.png"))
    generated.append(plot_cost_sensitivity_alpha(data, out_dir / "05_cost_sensitivity_alpha.png"))
    generated.append(plot_elapsed_time_sensitivity_alpha(data, out_dir / "06_elapsed_time_sensitivity_alpha.png"))
    generated.append(plot_weighted_tardiness_sensitivity_alpha(data, out_dir / "07_weighted_tardiness_sensitivity_alpha.png"))
    generated.append(plot_greedy_suboptimality_gap(data, out_dir / "08_greedy_suboptimality_gap.png"))

    return generated

