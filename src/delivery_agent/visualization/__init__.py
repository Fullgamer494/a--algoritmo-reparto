"""Módulo de visualización estática, interactiva y experimental para el agente de reparto."""

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

__all__ = [
    "compute_deterministic_layout",
    "render_route_static",
    "animate_route",
    "render_search_tree_static",
    "animate_search_process",
    "load_benchmark_data",
    "plot_objective_cost_comparison",
    "plot_execution_time_comparison",
    "plot_states_expanded_comparison",
    "plot_max_queue_size_comparison",
    "plot_cost_sensitivity_alpha",
    "plot_elapsed_time_sensitivity_alpha",
    "plot_weighted_tardiness_sensitivity_alpha",
    "plot_greedy_suboptimality_gap",
    "generate_all_benchmark_charts",
]

