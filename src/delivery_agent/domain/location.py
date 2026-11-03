"""Módulo de modelo de ubicaciones y matriz de tiempos de traslado."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, TypeAlias

LocationId: TypeAlias = str
TravelTimeMatrix: TypeAlias = Mapping[LocationId, Mapping[LocationId, int | None]]


@dataclass(frozen=True, slots=True)
class Location:
    """Representa una ubicación discreta dentro del escenario de reparto."""

    id: LocationId
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("El identificador de la ubicación debe ser una cadena no vacía.")


def validate_travel_matrix(
    locations: tuple[LocationId, ...] | list[LocationId],
    travel_times: Mapping[LocationId, Mapping[LocationId, int | None]],
) -> TravelTimeMatrix:
    """Valida y normaliza la matriz dirigida de tiempos de traslado como estructura inmutable.

    Reglas:
    - Incluye todas las ubicaciones declaradas.
    - La diagonal debe ser exactamente 0.
    - Los traslados imposibles se representan explícitamente como ``None``.
    - Ningún tiempo de traslado puede ser negativo ni booleano.
    - No se asume simetría (M[i][j] puede diferir de M[j][i]).
    """
    if not locations:
        raise ValueError("La lista de ubicaciones no puede estar vacía.")

    unique_locations = list(dict.fromkeys(locations))
    if len(unique_locations) != len(locations):
        raise ValueError("La lista de ubicaciones contiene identificadores duplicados.")

    for loc in unique_locations:
        if not isinstance(loc, str) or not loc.strip():
            raise ValueError("Todos los identificadores de ubicación deben ser cadenas no vacías.")

    if not isinstance(travel_times, Mapping):
        raise ValueError("La matriz de tiempos de traslado debe ser un mapeo.")

    location_set = set(unique_locations)
    matrix_keys = set(travel_times.keys())

    missing_rows = location_set - matrix_keys
    if missing_rows:
        raise ValueError(
            f"Faltan filas en la matriz de tiempos para las ubicaciones: {sorted(missing_rows)}"
        )

    extra_rows = matrix_keys - location_set
    if extra_rows:
        raise ValueError(
            f"La matriz de tiempos contiene filas para ubicaciones no declaradas: {sorted(extra_rows)}"
        )

    normalized: dict[LocationId, Mapping[LocationId, int | None]] = {}
    for origin in unique_locations:
        row = travel_times[origin]
        if not isinstance(row, Mapping):
            raise ValueError(f"La fila de tiempos para '{origin}' debe ser un mapeo.")

        row_keys = set(row.keys())
        missing_cols = location_set - row_keys
        if missing_cols:
            raise ValueError(
                f"Faltan destinos desde '{origin}' hacia: {sorted(missing_cols)}"
            )

        extra_cols = row_keys - location_set
        if extra_cols:
            raise ValueError(
                f"La fila '{origin}' contiene destinos no declarados: {sorted(extra_cols)}"
            )

        inner_row: dict[LocationId, int | None] = {}
        for destination in unique_locations:
            value = row[destination]
            if value is None:
                if origin == destination:
                    raise ValueError(
                        f"El traslado diagonal de '{origin}' a sí mismo debe ser 0, no None."
                    )
                inner_row[destination] = None
                continue

            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"El tiempo de traslado de '{origin}' a '{destination}' debe ser un entero en minutos o None."
                )

            if value < 0:
                raise ValueError(
                    f"El tiempo de traslado de '{origin}' a '{destination}' no puede ser negativo ({value})."
                )

            if origin == destination and value != 0:
                raise ValueError(
                    f"La diagonal de la matriz debe ser 0; se recibió {value} en '{origin}' -> '{destination}'."
                )

            inner_row[destination] = value

        normalized[origin] = MappingProxyType(inner_row)

    return MappingProxyType(normalized)
