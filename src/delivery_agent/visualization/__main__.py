"""Punto de entrada de línea de comandos (CLI) para la visualización del agente de reparto."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from delivery_agent.algorithms.astar import solve_astar
from delivery_agent.algorithms.dijkstra import solve_dijkstra
from delivery_agent.algorithms.exhaustive import solve_exhaustive
from delivery_agent.algorithms.greedy import solve_greedy
from delivery_agent.scenarios.loader import load_problem_from_json as load_scenario
from delivery_agent.visualization.benchmark_visualizer import generate_all_benchmark_charts
from delivery_agent.visualization.route_visualizer import animate_route, render_route_static
from delivery_agent.visualization.search_visualizer import animate_search_process, render_search_tree_static


def build_parser() -> argparse.ArgumentParser:
    """Configura el parser de argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Herramienta de visualización del agente de reparto (Etapa 6)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Genera las 8 figuras comparativas del benchmark a partir de summary.csv",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="results/benchmarks/summary.csv",
        help="Ruta al archivo summary.csv con los resultados consolidados",
    )
    parser.add_argument(
        "--details-json",
        type=str,
        default="results/benchmarks/details.json",
        help="Ruta al archivo details.json con métricas detalladas de búsqueda",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default="results/figures",
        help="Directorio de destino para las figuras del benchmark",
    )

    parser.add_argument(
        "--scenario",
        type=str,
        help="Ruta al archivo JSON del escenario a simular y visualizar",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="astar",
        choices=["astar", "dijkstra", "greedy", "exhaustive"],
        help="Algoritmo para planificar la ruta del escenario",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Ponderación alpha de la penalización por retraso",
    )
    parser.add_argument(
        "--route-png",
        type=str,
        help="Ruta del archivo PNG estático de la ruta planificada",
    )
    parser.add_argument(
        "--route-gif",
        type=str,
        help="Ruta del archivo GIF animado del recorrido del vehículo",
    )
    parser.add_argument(
        "--search-png",
        type=str,
        help="Ruta del archivo PNG estático del árbol de búsqueda (solo A*)",
    )
    parser.add_argument(
        "--search-gif",
        type=str,
        help="Ruta del archivo GIF animado del proceso de búsqueda (solo A*)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Genera todos los artefactos de visualización para el escenario indicado",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Función principal CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    actions_executed = 0

    if args.benchmark:
        print(f"[*] Generando suite de 8 gráficas de benchmark desde {args.summary_csv}...")
        charts = generate_all_benchmark_charts(
            summary_csv_path=args.summary_csv,
            output_dir=args.figures_dir,
            details_json_path=args.details_json,
        )
        print(f"[+] {len(charts)} gráficas generadas con éxito en: {args.figures_dir}")
        for c in charts:
            print(f"    - {c.name}")
        actions_executed += 1

    if args.scenario:
        scenario_path = Path(args.scenario).resolve()
        if not scenario_path.exists():
            print(f"[-] Error: No existe el escenario: {scenario_path}", file=sys.stderr)
            return 1

        print(f"[*] Cargando escenario: {scenario_path.name}")
        problem = load_scenario(scenario_path)

        track_events = bool(args.search_png or args.search_gif or args.all)
        print(f"[*] Resolviendo con algoritmo '{args.algorithm}' (alpha={args.alpha}, track_events={track_events})...")

        if args.algorithm == "astar":
            result = solve_astar(problem, alpha=args.alpha, track_events=track_events)
        elif args.algorithm == "dijkstra":
            result = solve_dijkstra(problem, alpha=args.alpha)
        elif args.algorithm == "greedy":
            result = solve_greedy(problem, alpha=args.alpha)
        elif args.algorithm == "exhaustive":
            result = solve_exhaustive(problem, alpha=args.alpha)
        else:
            print(f"[-] Algoritmo no soportado: {args.algorithm}", file=sys.stderr)
            return 1

        print(f"[+] Búsqueda finalizada. Factible: {result.is_feasible}, Costo: {result.total_cost}")

        stem = scenario_path.stem
        route_png_path = args.route_png or (f"results/routes/{stem}_route.png" if args.all else None)
        route_gif_path = args.route_gif or (f"results/routes/{stem}_route.gif" if args.all else None)
        search_png_path = args.search_png or (f"results/search/{stem}_search.png" if args.all and args.algorithm == "astar" else None)
        search_gif_path = args.search_gif or (f"results/search/{stem}_search.gif" if args.all and args.algorithm == "astar" else None)

        if route_png_path:
            p = render_route_static(problem, result, route_png_path)
            print(f"[+] Ruta estática generada: {p}")
            actions_executed += 1

        if route_gif_path:
            p = animate_route(problem, result, route_gif_path)
            print(f"[+] Animación de ruta generada: {p}")
            actions_executed += 1

        if search_png_path and result.events:
            p = render_search_tree_static(problem, result, search_png_path)
            print(f"[+] Árbol de búsqueda estático generado: {p}")
            actions_executed += 1

        if search_gif_path and result.events:
            p = animate_search_process(problem, result, search_gif_path)
            print(f"[+] Animación de búsqueda generada: {p}")
            actions_executed += 1

    if actions_executed == 0:
        parser.print_help()
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
