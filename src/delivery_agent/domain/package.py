"""Modelo de datos para los paquetes a entregar."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from delivery_agent.domain.location import LocationId

DEFAULT_PRIORITY_WEIGHTS: dict[str, int] = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "baja": 1,
    "media": 2,
    "alta": 3,
}


@dataclass(frozen=True, slots=True)
class Package:
    """Representa un paquete con su destino, ventana obligatoria, promesa y prioridad.

    Todos los valores temporales se expresan en minutos enteros desde las 00:00
    de la jornada de planificación.
    """

    id: str
    destination: LocationId
    window_start: int
    window_end: int
    promised_time: int
    priority: str
    service_time: int

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("El id del paquete debe ser una cadena no vacía.")
        if not isinstance(self.destination, str) or not self.destination.strip():
            raise ValueError(f"El destino del paquete '{self.id}' debe ser una cadena no vacía.")

        for field_name, value in (
            ("window_start", self.window_start),
            ("window_end", self.window_end),
            ("promised_time", self.promised_time),
            ("service_time", self.service_time),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"El campo '{field_name}' del paquete '{self.id}' debe ser un entero en minutos."
                )

        if self.window_start < 0 or self.window_end < 0 or self.promised_time < 0:
            raise ValueError(
                f"Los tiempos del paquete '{self.id}' no pueden ser negativos."
            )

        if self.service_time < 0:
            raise ValueError(
                f"El tiempo de servicio del paquete '{self.id}' no puede ser negativo."
            )

        if self.window_start > self.window_end:
            raise ValueError(
                f"Ventana inválida en paquete '{self.id}': window_start ({self.window_start}) "
                f"> window_end ({self.window_end})."
            )

        if not (self.window_start <= self.promised_time <= self.window_end):
            raise ValueError(
                f"La hora prometida ({self.promised_time}) del paquete '{self.id}' "
                f"debe cumplir window_start ({self.window_start}) <= promised_time <= window_end ({self.window_end})."
            )

        if self.window_start + self.service_time > self.window_end:
            raise ValueError(
                f"El paquete '{self.id}' no puede servirse dentro de su propia ventana: "
                f"window_start ({self.window_start}) + service_time ({self.service_time}) > window_end ({self.window_end})."
            )

        if not isinstance(self.priority, str) or not self.priority.strip():
            raise ValueError(f"La prioridad del paquete '{self.id}' debe ser una cadena no vacía.")

    def compute_tardiness(self, completion_time: int) -> int:
        """Calcula el retraso respecto a la hora prometida: d_i = max(0, f_i - r_i)."""
        if isinstance(completion_time, bool) or not isinstance(completion_time, int):
            raise ValueError("La hora de finalización debe ser un entero en minutos.")
        return max(0, completion_time - self.promised_time)

    def resolve_priority_weight(self, priority_weights: Mapping[str, int | float]) -> float:
        """Obtiene el peso numérico asociado al nivel de prioridad del paquete."""
        if self.priority in priority_weights:
            weight = priority_weights[self.priority]
        elif self.priority.lower() in priority_weights:
            weight = priority_weights[self.priority.lower()]
        else:
            raise ValueError(
                f"La prioridad '{self.priority}' del paquete '{self.id}' no existe en priority_weights."
            )

        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight <= 0:
            raise ValueError(
                f"El peso de la prioridad '{self.priority}' debe ser un número positivo."
            )
        return float(weight)

