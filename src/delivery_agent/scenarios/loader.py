"""Cargador y validador de escenarios JSON y conversión de tiempo."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from delivery_agent.domain.package import DEFAULT_PRIORITY_WEIGHTS, Package
from delivery_agent.domain.problem import DeliveryProblem

_HHMM_PATTERN = re.compile(r"^(?P<hours>\d{2}):(?P<minutes>\d{2})$")


def parse_time_to_minutes(value: str | int) -> int:
    """Convierte una hora en formato 'HH:MM' (o un entero en minutos) a minutos desde las 00:00.

    Ejemplos:
    - "08:00" -> 480
    - "08:30" -> 510
    - "09:00" -> 540
    - "12:00" -> 720
    """
    if isinstance(value, bool):
        raise ValueError("El valor de tiempo no puede ser booleano.")

    if isinstance(value, int):
        if value < 0 or value >= 24 * 60:
            raise ValueError(
                f"El tiempo en minutos ({value}) debe estar dentro de la jornada [0, 1439]."
            )
        return value

    if not isinstance(value, str):
        raise ValueError(
            f"Formato de hora inválido ({type(value).__name__}): se esperaba 'HH:MM' o entero."
        )

    cleaned = value.strip()
    match = _HHMM_PATTERN.match(cleaned)
    if not match:
        raise ValueError(
            f"Formato de hora inválido '{value}'. Se esperaba exactamente 'HH:MM' (ej. '08:00')."
        )

    hours = int(match.group("hours"))
    minutes = int(match.group("minutes"))

    if not (0 <= hours <= 23):
        raise ValueError(f"Hora fuera de rango en '{value}': las horas deben estar entre 00 y 23.")
    if not (0 <= minutes <= 59):
        raise ValueError(f"Minutos fuera de rango en '{value}': los minutos deben estar entre 00 y 59.")

    return hours * 60 + minutes


def format_minutes_to_hhmm(minutes: int) -> str:
    """Convierte minutos enteros desde las 00:00 al formato legible 'HH:MM'."""
    if isinstance(minutes, bool) or not isinstance(minutes, int):
        raise ValueError("Los minutos deben ser un número entero.")
    if minutes < 0:
        raise ValueError("Los minutos no pueden ser negativos.")

    hours, mins = divmod(minutes, 60)
    return f"{hours:02d}:{mins:02d}"


def load_problem_from_dict(
    data: Mapping[str, Any],
    *,
    allow_zero_alpha: bool = False,
    enforce_package_range: bool = False,
) -> DeliveryProblem:
    """Construye y valida un ``DeliveryProblem`` a partir de un diccionario deserializado de JSON."""
    if not isinstance(data, Mapping):
        raise ValueError("El contenido del escenario debe ser un objeto JSON (diccionario).")

    required_fields = (
        "name",
        "start_location",
        "start_time",
        "locations",
        "travel_times",
        "packages",
    )
    for field_name in required_fields:
        if field_name not in data:
            raise ValueError(f"Falta el campo obligatorio '{field_name}' en el escenario.")

    name = data["name"]
    start_location = data["start_location"]
    start_time = parse_time_to_minutes(data["start_time"])
    alpha = data.get("alpha", 1.0)

    raw_weights = data.get("priority_weights")
    if raw_weights is None:
        priority_weights: Mapping[str, int | float] = dict(DEFAULT_PRIORITY_WEIGHTS)
    else:
        if not isinstance(raw_weights, Mapping) or not raw_weights:
            raise ValueError("El campo 'priority_weights' debe ser un objeto JSON no vacío.")
        priority_weights = dict(raw_weights)

    raw_locations = data["locations"]
    if not isinstance(raw_locations, list) or not raw_locations:
        raise ValueError("El campo 'locations' debe ser una lista no vacía de identificadores.")

    raw_travel_times = data["travel_times"]
    if not isinstance(raw_travel_times, Mapping):
        raise ValueError("El campo 'travel_times' debe ser un objeto JSON con la matriz de tiempos.")

    raw_packages = data["packages"]
    if not isinstance(raw_packages, list) or not raw_packages:
        raise ValueError("El campo 'packages' debe ser una lista no vacía de paquetes.")

    required_pkg_fields = (
        "id",
        "destination",
        "window_start",
        "window_end",
        "promised_time",
        "priority",
        "service_time",
    )
    packages: list[Package] = []
    for idx, pkg_data in enumerate(raw_packages):
        if not isinstance(pkg_data, Mapping):
            raise ValueError(f"El paquete en la posición {idx} debe ser un objeto JSON.")
        for pkg_field in required_pkg_fields:
            if pkg_field not in pkg_data:
                raise ValueError(
                    f"Falta el campo obligatorio '{pkg_field}' en el paquete de la posición {idx}."
                )

        pkg = Package(
            id=pkg_data["id"],
            destination=pkg_data["destination"],
            window_start=parse_time_to_minutes(pkg_data["window_start"]),
            window_end=parse_time_to_minutes(pkg_data["window_end"]),
            promised_time=parse_time_to_minutes(pkg_data["promised_time"]),
            priority=pkg_data["priority"],
            service_time=pkg_data["service_time"],
        )
        packages.append(pkg)

    problem = DeliveryProblem(
        name=name,
        start_location=start_location,
        start_time=start_time,
        packages=tuple(packages),
        locations=tuple(raw_locations),
        travel_times=raw_travel_times,
        alpha=alpha,
        priority_weights=priority_weights,
        allow_zero_alpha=allow_zero_alpha,
    )

    if enforce_package_range and not problem.validate_package_count_range(5, 8):
        raise ValueError(
            f"El escenario '{problem.name}' tiene {len(problem.packages)} paquetes; "
            "se exigía entre 5 y 8 paquetes."
        )

    return problem


def load_problem_from_json(
    path: str | Path,
    *,
    allow_zero_alpha: bool = False,
    enforce_package_range: bool = False,
) -> DeliveryProblem:
    """Lee un archivo JSON desde disco, convierte tiempos a minutos y valida el escenario."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de escenario: '{file_path}'.")

    with file_path.open("r", encoding="utf-8") as handle:
        try:
            raw_data = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON inválido en el archivo '{file_path}': {exc}") from exc

    return load_problem_from_dict(
        raw_data,
        allow_zero_alpha=allow_zero_alpha,
        enforce_package_range=enforce_package_range,
    )


def discover_scenario_files(directory: str | Path) -> tuple[Path, ...]:
    """Descubre y devuelve todos los archivos JSON de escenarios en un directorio ordenados por nombre."""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"El directorio no existe: '{dir_path}'")
    return tuple(sorted(dir_path.glob("*.json")))

