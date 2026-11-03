"""Representación inmutable del estado de búsqueda S = (l, t, U)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, TypeAlias

from delivery_agent.domain.location import LocationId


@dataclass(frozen=True, slots=True)
class DeliveryState:
    """Estado del espacio de búsqueda: S = (location, current_time, pending_packages).

    La identidad del estado está compuesta exclusivamente por:
    - ``location``: ubicación actual (l)
    - ``current_time``: hora actual en minutos enteros (t)
    - ``pending_packages``: conjunto inmutable de IDs de paquetes pendientes (U)

    El costo acumulado ``g`` no forma parte de la identidad del estado.
    No se fusionan estados con horas distintas.

    Implementa orden total determinista basado en ``(current_time, location, sorted_pending_ids)``
    para evitar el orden parcial de subconjuntos de ``frozenset`` al desempatar en ``heapq``.
    """

    location: LocationId
    current_time: int
    pending_packages: frozenset[str]

    def __post_init__(self) -> None:
        if not isinstance(self.location, str) or not self.location.strip():
            raise ValueError("La ubicación del estado debe ser una cadena no vacía.")
        if isinstance(self.current_time, bool) or not isinstance(self.current_time, int):
            raise ValueError("La hora actual del estado debe ser un entero en minutos.")
        if self.current_time < 0:
            raise ValueError("La hora actual del estado no puede ser negativa.")
        if not isinstance(self.pending_packages, frozenset):
            object.__setattr__(self, "pending_packages", frozenset(self.pending_packages))
        for pkg_id in self.pending_packages:
            if not isinstance(pkg_id, str) or not pkg_id.strip():
                raise ValueError("Todos los identificadores en pending_packages deben ser cadenas no vacías.")

    @classmethod
    def create(
        cls,
        location: LocationId,
        current_time: int,
        pending_packages: Iterable[str],
    ) -> DeliveryState:
        """Construye un estado convirtiendo la colección de paquetes pendientes a frozenset."""
        return cls(
            location=location,
            current_time=current_time,
            pending_packages=frozenset(pending_packages),
        )

    @property
    def is_goal(self) -> bool:
        """Indica si el estado es objetivo (U = vacío, todos los paquetes entregados)."""
        return len(self.pending_packages) == 0

    def sorted_pending_ids(self) -> tuple[str, ...]:
        """Devuelve los IDs de paquetes pendientes en orden lexicográfico determinista."""
        return tuple(sorted(self.pending_packages))

    def sort_key(self) -> tuple[int, LocationId, tuple[str, ...]]:
        """Clave canónica de orden total determinista para colas de prioridad."""
        return (self.current_time, self.location, self.sorted_pending_ids())

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, DeliveryState):
            return NotImplemented
        return self.sort_key() < other.sort_key()

    def __le__(self, other: object) -> bool:
        if not isinstance(other, DeliveryState):
            return NotImplemented
        return self.sort_key() <= other.sort_key()

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, DeliveryState):
            return NotImplemented
        return self.sort_key() > other.sort_key()

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, DeliveryState):
            return NotImplemented
        return self.sort_key() >= other.sort_key()


SearchState: TypeAlias = DeliveryState
