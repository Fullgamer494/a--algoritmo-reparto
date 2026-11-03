"""Punto de entrada principal del agente de reparto (CLI y lanzador de GUI)."""

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sistema de Planificación de Reparto Inteligente mediante A* (FIA)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gui", action="store_true", help="Lanza la interfaz gráfica interactiva (Tkinter)")
    parser.add_argument("--scenario", type=str, help="Ruta al archivo JSON del escenario a resolver")
    parser.add_argument("--algorithm", type=str, default="astar", choices=["astar", "dijkstra", "greedy", "exhaustive"], help="Algoritmo de planificación")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ponderación alpha de la penalización")
    parser.add_argument("--benchmark", action="store_true", help="Genera las figuras del benchmark comparativo")

    args = parser.parse_args(argv)

    if args.gui:
        from delivery_agent.gui.app import DeliveryAgentApp
        app = DeliveryAgentApp()
        app.run()
        return 0

    if args.benchmark or args.scenario:
        from delivery_agent.visualization.__main__ import main as vis_main
        return vis_main(argv)

    # Si no se especificaron argumentos, sugerir --gui o mostrar ayuda
    parser.print_help()
    print("\n[Tip] Ejecute con '--gui' para iniciar la interfaz gráfica interactiva.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

