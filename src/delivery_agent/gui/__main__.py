"""Punto de entrada de línea de comandos para iniciar la interfaz gráfica interactiva."""

import sys
from delivery_agent.gui.app import DeliveryAgentApp


def main() -> int:
    """Inicia la aplicación de escritorio Tkinter."""
    app = DeliveryAgentApp()
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())

