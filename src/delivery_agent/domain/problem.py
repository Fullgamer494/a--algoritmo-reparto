"""Modelo general del problema de planificación de reparto."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from delivery_agent.domain.location import (
    Location,
    LocationId,
    TravelTimeMatrix,
    validate_travel_matrix,
)
from delivery_agent.domain.package import DEFAULT_PRIORITY_WEIGHTS, Package
from delivery_agent.domain.state import DeliveryState


@dataclass(frozen=True, slots=True)
class DeliveryProblem:
    """Define una instancia completa y validada del problema de reparto monoagente.

    Incluye:
    - Punto y hora de salida (`start_location`, `start_time`).
    - Parámetro `alpha` de la función objetivo C(pi) = T(pi) + alpha * D_p(pi).
    - Pesos de prioridad configurables (`priority_weights`).
    - Ubicaciones y matriz dirigida de tiempos de traslado (`travel_times`).
    - Paquetes a entregar (`packages`).
    """

    name: str
    start_location: LocationId
    start_time: int
    packages: tuple[Package, ...]
    locations: tuple[LocationId, ...]
    travel_times: TravelTimeMatrix
    alpha: float = 1.0
    priority_weights: Mapping[str, int | float] = field(
        default_factory=lambda: dict(DEFAULT_PRIORITY_WEIGHTS)
    )
    allow_zero_alpha: bool = False
    _package_map: Mapping[str, Package] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("El nombre del escenario debe ser una cadena no vacía.")

        if not isinstance(self.start_location, str) or not self.start_location.strip():
            raise ValueError("La ubicación inicial (start_location) debe ser una cadena no vacía.")

        if isinstance(self.start_time, bool) or not isinstance(self.start_time, int) or self.start_time < 0:
            raise ValueError("La hora inicial (start_time) debe ser un entero no negativo en minutos.")

        if isinstance(self.alpha, bool) or not isinstance(self.alpha, (int, float)):
            raise ValueError("El parámetro alpha debe ser numérico.")

        if self.allow_zero_alpha:
            if self.alpha < 0:
                raise ValueError("El parámetro alpha no puede ser negativo.")
        else:
            if self.alpha <= 0:
                raise ValueError(
                    "El parámetro alpha debe ser mayor que cero en la configuración normal "
                    "(use allow_zero_alpha=True solo para experimentos de control con alpha=0)."
                )
        object.__setattr__(self, "alpha", float(self.alpha))

        if not isinstance(self.priority_weights, Mapping) or not self.priority_weights:
            raise ValueError("priority_weights debe ser un mapeo no vacío.")

        normalized_weights: dict[str, float] = {}
        for level, weight in self.priority_weights.items():
            if not isinstance(level, str) or not level.strip():
                raise ValueError("Las claves de priority_weights deben ser cadenas no vacías.")
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight <= 0:
                raise ValueError(
                    f"El peso para la prioridad '{level}' debe ser un número positivo."
                )
            normalized_weights[level] = float(weight)
        object.__setattr__(self, "priority_weights", MappingProxyType(normalized_weights))

        if not isinstance(self.locations, tuple):
            object.__setattr__(self, "locations", tuple(self.locations))

        validated_matrix = validate_travel_matrix(self.locations, self.travel_times)
        object.__setattr__(self, "travel_times", validated_matrix)

        location_set = set(self.locations)
        if self.start_location not in location_set:
            raise ValueError(
                f"La ubicación inicial '{self.start_location}' no pertenece a la lista de ubicaciones declaradas."
            )

        if not isinstance(self.packages, tuple):
            object.__setattr__(self, "packages", tuple(self.packages))

        pkg_map: dict[str, Package] = {}
        for pkg in self.packages:
            if not isinstance(pkg, Package):
                raise ValueError("Todos los elementos de 'packages' deben ser instancias de Package.")
            if pkg.id in pkg_map:
                raise ValueError(f"Identificador de paquete duplicado: '{pkg.id}'.")
            if pkg.destination not in location_set:
                raise ValueError(
                    f"El destino '{pkg.destination}' del paquete '{pkg.id}' no existe en 'locations'."
                )
            if pkg.window_end < self.start_time:
                raise ValueError(
                    f"La ventana del paquete '{pkg.id}' cierra ({pkg.window_end}) "
                    f"antes de la hora de salida del repartidor ({self.start_time})."
                )
            pkg.resolve_priority_weight(normalized_weights)
            pkg_map[pkg.id] = pkg

        object.__setattr__(self, "_package_map", MappingProxyType(pkg_map))

    @property
    def initial_state(self) -> DeliveryState:
        """Devuelve el estado inicial S_0 = (l_0, t_0, P)."""
        return DeliveryState(
            location=self.start_location,
            current_time=self.start_time,
            pending_packages=frozenset(self._package_map.keys()),
        )

    def is_goal(self, state: DeliveryState) -> bool:
        """Comprueba si el estado corresponde al estado objetivo (U = vacío)."""
        return state.is_goal

    def get_package(self, package_id: str) -> Package:
        """Obtiene un paquete por su identificador único."""
        try:
            return self._package_map[package_id]
        except KeyError as exc:
            raise KeyError(f"No existe el paquete con id '{package_id}'.") from exc

    def packages_at_location(self, location_id: LocationId) -> tuple[Package, ...]:
        """Devuelve todos los paquetes cuyo destino es ``location_id``, preservando el orden."""
        if location_id not in self.travel_times:
            raise KeyError(f"Ubicación desconocida: '{location_id}'.")
        return tuple(pkg for pkg in self.packages if pkg.destination == location_id)

    @property
    def location_entities(self) -> tuple[Location, ...]:
        """Devuelve las ubicaciones del problema como objetos Location."""
        return tuple(Location(id=loc_id) for loc_id in self.locations)

    def travel_time(self, origin: LocationId, destination: LocationId) -> int | None:
        """Devuelve el tiempo de traslado tau(origin, destination) en minutos o None si no está disponible."""
        if origin not in self.travel_times:
            raise KeyError(f"Ubicación de origen desconocida: '{origin}'.")
        if destination not in self.travel_times[origin]:
            raise KeyError(f"Ubicación de destino desconocida: '{destination}'.")
        return self.travel_times[origin][destination]

    def priority_weight(self, package: Package | str) -> float:
        """Obtiene el peso de prioridad w_i para un paquete o su ID."""
        pkg = self.get_package(package) if isinstance(package, str) else package
        return pkg.resolve_priority_weight(self.priority_weights)

    def transition_cost(
        self,
        elapsed_time: int,
        package: Package | str,
        completion_time: int,
    ) -> float:
        """Calcula el costo de una transición individual según la ecuación de la sección 7.1:

        c(S, p_i) = (f_i - t) + alpha * w_i * max(0, f_i - r_i)
        donde `elapsed_time` = f_i - t.
        """
        from delivery_agent.core.cost import compute_incremental_cost, compute_penalty

        pkg = self.get_package(package) if isinstance(package, str) else package
        tardiness = pkg.compute_tardiness(completion_time)
        weight = self.priority_weight(pkg)
        penalty = compute_penalty(tardiness, weight, self.alpha)
        return compute_incremental_cost(elapsed_time, penalty)

    def objective_cost(self, total_elapsed_time: int, weighted_tardiness: float) -> float:
        """Calcula el costo objetivo total: C(pi) = T(pi) + alpha * D_p(pi)."""
        from delivery_agent.core.cost import compute_incremental_cost, validate_alpha

        if isinstance(weighted_tardiness, bool) or not isinstance(weighted_tardiness, (int, float)) or weighted_tardiness < 0:
            raise ValueError("weighted_tardiness debe ser un número no negativo.")
        valid_alpha = validate_alpha(self.alpha)
        return compute_incremental_cost(total_elapsed_time, valid_alpha * float(weighted_tardiness))

    def validate_package_count_range(
        self, min_packages: int = 5, max_packages: int = 8
    ) -> bool:
        """Indica si la cantidad de paquetes está dentro del rango principal [min_packages, max_packages]."""
        return min_packages <= len(self.packages) <= max_packages
