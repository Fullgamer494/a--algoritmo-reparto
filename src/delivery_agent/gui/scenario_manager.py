"""Gestor de estado, validación y ejecución para la interfaz gráfica del agente de reparto."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

from delivery_agent.algorithms.astar import SearchResult, solve_astar
from delivery_agent.algorithms.dijkstra import solve_dijkstra
from delivery_agent.algorithms.exhaustive import solve_exhaustive
from delivery_agent.algorithms.greedy import solve_greedy
from delivery_agent.domain.package import DEFAULT_PRIORITY_WEIGHTS, Package
from delivery_agent.domain.problem import DeliveryProblem
from delivery_agent.scenarios.loader import parse_time_to_minutes


def format_minutes_to_hhmm(minutes: int) -> str:
    """Convierte minutos enteros a representación de hora 'HH:MM'."""
    hh = (minutes // 60) % 24
    mm = minutes % 60
    return f"{hh:02d}:{mm:02d}"


class ScenarioManager:
    """Administra el estado en memoria de un escenario editable y desacoplado de la UI.

    Permite crear escenarios desde cero, cargar escenarios oficiales o personalizados,
    editar paquetes, ubicaciones y la matriz de tiempos, ejecutar algoritmos y validar
    todas las restricciones antes de la ejecución.
    """

    OFFICIAL_SCENARIOS_DIR = Path("data/scenarios")
    CUSTOM_SCENARIOS_DIR = Path("data/scenarios/custom")

    def __init__(self) -> None:
        self.CUSTOM_SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)
        self.new_scenario()

    def new_scenario(self, name: str = "nuevo_escenario_personalizado") -> None:
        """Inicializa un nuevo escenario en blanco con valores por defecto válidos."""
        self.name: str = name
        self.start_location: str = "D0"
        self.start_time: int = 480  # 08:00
        self.alpha: float = 1.0
        self.priority_weights: dict[str, float] = {
            "low": 1.0,
            "medium": 2.0,
            "high": 3.0,
            "baja": 1.0,
            "media": 2.0,
            "alta": 3.0,
        }
        self.locations: list[str] = ["D0", "L1", "L2", "L3", "L4", "L5"]
        self.packages: list[dict[str, Any]] = [
            {
                "id": "P1",
                "destination": "L1",
                "window_start": 480,  # 08:00
                "window_end": 600,    # 10:00
                "promised_time": 540, # 09:00
                "priority": "alta",
                "service_time": 5,
            },
            {
                "id": "P2",
                "destination": "L2",
                "window_start": 490,  # 08:10
                "window_end": 630,    # 10:30
                "promised_time": 570, # 09:30
                "priority": "media",
                "service_time": 5,
            },
            {
                "id": "P3",
                "destination": "L3",
                "window_start": 500,  # 08:20
                "window_end": 660,    # 11:00
                "promised_time": 600, # 10:00
                "priority": "baja",
                "service_time": 5,
            },
            {
                "id": "P4",
                "destination": "L4",
                "window_start": 510,  # 08:30
                "window_end": 690,    # 11:30
                "promised_time": 615, # 10:15
                "priority": "media",
                "service_time": 5,
            },
            {
                "id": "P5",
                "destination": "L5",
                "window_start": 520,  # 08:40
                "window_end": 720,    # 12:00
                "promised_time": 630, # 10:30
                "priority": "alta",
                "service_time": 5,
            },
        ]
        self._init_default_travel_times()
        self.source_path: Path | None = None
        self.is_official: bool = False

    def _init_default_travel_times(self) -> None:
        """Crea una matriz de tiempos básica coherente entre las ubicaciones existentes."""
        self.travel_times: dict[str, dict[str, int | None]] = {}
        for u in self.locations:
            self.travel_times[u] = {}
            for v in self.locations:
                if u == v:
                    self.travel_times[u][v] = 0
                else:
                    # Tiempo proporcional estándar por defecto
                    self.travel_times[u][v] = 10

    def list_official_scenarios(self) -> list[Path]:
        """Devuelve los archivos oficiales (escenario_01 a escenario_08) ordenados."""
        if not self.OFFICIAL_SCENARIOS_DIR.exists():
            return []
        files = [
            f for f in sorted(self.OFFICIAL_SCENARIOS_DIR.glob("escenario_*.json"))
            if f.is_file() and not f.name.startswith("custom")
        ]
        return files

    def list_custom_scenarios(self) -> list[Path]:
        """Devuelve los escenarios personalizados guardados en data/scenarios/custom/."""
        if not self.CUSTOM_SCENARIOS_DIR.exists():
            return []
        return sorted([f for f in self.CUSTOM_SCENARIOS_DIR.glob("*.json") if f.is_file()])

    def load_from_dict(self, data: dict[str, Any], source_path: Path | None = None) -> None:
        """Carga los datos de un escenario desde un diccionario de especificación."""
        self.name = str(data.get("name", "escenario_sin_nombre"))
        self.start_location = str(data.get("start_location", "D0"))

        raw_time = data.get("start_time", 480)
        self.start_time = parse_time_to_minutes(raw_time)

        self.alpha = float(data.get("alpha", 1.0))

        raw_weights = data.get("priority_weights", DEFAULT_PRIORITY_WEIGHTS)
        self.priority_weights = {str(k): float(v) for k, v in raw_weights.items()}

        raw_locations = data.get("locations", [])
        self.locations = [str(loc) for loc in raw_locations]
        if self.start_location not in self.locations and self.start_location:
            self.locations.insert(0, self.start_location)

        # Cargar paquetes
        raw_packages = data.get("packages", [])
        self.packages = []
        for pkg_data in raw_packages:
            pkg_dict: dict[str, Any] = {
                "id": str(pkg_data.get("id", "")),
                "destination": str(pkg_data.get("destination", "")),
                "window_start": parse_time_to_minutes(pkg_data.get("window_start", 0)),
                "window_end": parse_time_to_minutes(pkg_data.get("window_end", 0)),
                "promised_time": parse_time_to_minutes(pkg_data.get("promised_time", 0)),
                "priority": str(pkg_data.get("priority", "media")),
                "service_time": int(pkg_data.get("service_time", 5)),
            }
            self.packages.append(pkg_dict)

        # Cargar matriz de tiempos
        raw_travel = data.get("travel_times", {})
        self.travel_times = {}
        for u in self.locations:
            self.travel_times[u] = {}
            for v in self.locations:
                if u == v:
                    self.travel_times[u][v] = 0
                else:
                    val = raw_travel.get(u, {}).get(v)
                    if val is None or val == "" or str(val).lower() in ("none", "null", "-"):
                        self.travel_times[u][v] = None
                    else:
                        self.travel_times[u][v] = int(val)

        self.source_path = source_path
        if source_path is not None:
            self.is_official = (
                source_path.parent.resolve() == self.OFFICIAL_SCENARIOS_DIR.resolve()
                and source_path.name.startswith("escenario_")
            )
        else:
            self.is_official = False

    def load_from_json(self, path: Path | str) -> None:
        """Carga y valida un archivo JSON de escenario."""
        file_path = Path(path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"No existe el archivo de escenario: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.load_from_dict(data, source_path=file_path)

    def to_dict(self) -> dict[str, Any]:
        """Exporta el escenario actual a un diccionario compatible con el esquema JSON oficial."""
        return {
            "name": self.name,
            "start_location": self.start_location,
            "start_time": format_minutes_to_hhmm(self.start_time),
            "alpha": self.alpha,
            "priority_weights": copy.deepcopy(self.priority_weights),
            "locations": list(self.locations),
            "packages": [
                {
                    "id": p["id"],
                    "destination": p["destination"],
                    "window_start": format_minutes_to_hhmm(p["window_start"]),
                    "window_end": format_minutes_to_hhmm(p["window_end"]),
                    "promised_time": format_minutes_to_hhmm(p["promised_time"]),
                    "priority": p["priority"],
                    "service_time": p["service_time"],
                }
                for p in self.packages
            ],
            "travel_times": copy.deepcopy(self.travel_times),
        }

    def save_to_json(self, path: Path | str) -> Path:
        """Guarda el escenario en la ruta especificada."""
        file_path = Path(path).resolve()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return file_path

    def save_as_custom(self, filename: str | None = None) -> Path:
        """Guarda el escenario en la carpeta aislada data/scenarios/custom/.

        Impide expresamente sobreescribir los escenarios oficiales de benchmark.
        """
        self.CUSTOM_SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = filename or f"{self.name}.json"
        if not safe_name.endswith(".json"):
            safe_name += ".json"

        # Impedir colisión de nombres con los 8 oficiales
        if safe_name.startswith("escenario_0") or safe_name.startswith("escenario_base"):
            safe_name = f"custom_{safe_name}"

        target = self.CUSTOM_SCENARIOS_DIR / safe_name
        self.save_to_json(target)
        self.source_path = target
        self.is_official = False
        return target

    # =========================================================================
    # Operaciones sobre Ubicaciones
    # =========================================================================

    def add_location(self, loc_id: str) -> None:
        """Añade una nueva ubicación y actualiza la matriz de tiempos."""
        clean_id = loc_id.strip()
        if not clean_id:
            raise ValueError("El identificador de la ubicación no puede estar vacío.")
        if clean_id in self.locations:
            raise ValueError(f"La ubicación '{clean_id}' ya existe.")

        self.locations.append(clean_id)
        self.travel_times[clean_id] = {}
        for other in self.locations:
            if other == clean_id:
                self.travel_times[clean_id][clean_id] = 0
            else:
                self.travel_times[clean_id][other] = 10
                self.travel_times[other][clean_id] = 10

    def remove_location(self, loc_id: str) -> None:
        """Elimina una ubicación existente y limpia sus filas/columnas en la matriz."""
        if loc_id not in self.locations:
            raise ValueError(f"La ubicación '{loc_id}' no existe.")
        if len(self.locations) <= 1:
            raise ValueError("No se puede eliminar la única ubicación restante.")

        # Verificar si hay paquetes asignados
        assigned_pkgs = [p["id"] for p in self.packages if p["destination"] == loc_id]
        if assigned_pkgs:
            raise ValueError(
                f"No se puede eliminar '{loc_id}' porque tiene paquetes asignados: {assigned_pkgs}. "
                "Reasigne o elimine estos paquetes primero."
            )

        if self.start_location == loc_id:
            # Reasignar origen al primer elemento restante
            other_locs = [l for l in self.locations if l != loc_id]
            self.start_location = other_locs[0]

        self.locations.remove(loc_id)
        self.travel_times.pop(loc_id, None)
        for row in self.travel_times.values():
            row.pop(loc_id, None)

    def rename_location(self, old_id: str, new_id: str) -> None:
        """Renombra una ubicación y actualiza paquetes, origen y matriz."""
        new_clean = new_id.strip()
        if not new_clean:
            raise ValueError("El nuevo identificador no puede estar vacío.")
        if old_id not in self.locations:
            raise ValueError(f"La ubicación '{old_id}' no existe.")
        if new_clean in self.locations and new_clean != old_id:
            raise ValueError(f"Ya existe una ubicación con el nombre '{new_clean}'.")

        if old_id == new_clean:
            return

        # Actualizar lista
        idx = self.locations.index(old_id)
        self.locations[idx] = new_clean

        # Actualizar origen
        if self.start_location == old_id:
            self.start_location = new_clean

        # Actualizar paquetes
        for p in self.packages:
            if p["destination"] == old_id:
                p["destination"] = new_clean

        # Actualizar matriz
        new_matrix: dict[str, dict[str, int | None]] = {}
        for u in self.locations:
            old_u = old_id if u == new_clean else u
            new_matrix[u] = {}
            for v in self.locations:
                old_v = old_id if v == new_clean else v
                new_matrix[u][v] = self.travel_times.get(old_u, {}).get(old_v)

        self.travel_times = new_matrix

    def set_travel_time(self, origin: str, destination: str, minutes: int | None) -> None:
        """Establece el tiempo de desplazamiento entre dos ubicaciones (admite None)."""
        if origin not in self.locations:
            raise ValueError(f"Origen desconocido: '{origin}'.")
        if destination not in self.locations:
            raise ValueError(f"Destino desconocido: '{destination}'.")

        if origin == destination:
            self.travel_times[origin][destination] = 0
            return

        if minutes is not None and minutes < 0:
            raise ValueError("El tiempo de viaje no puede ser negativo.")

        self.travel_times[origin][destination] = minutes

    # =========================================================================
    # Operaciones sobre Paquetes
    # =========================================================================

    def add_package(self, pkg_dict: dict[str, Any]) -> None:
        """Agrega un paquete validando que su identificador no exista previamente."""
        pkg_id = str(pkg_dict.get("id", "")).strip()
        if not pkg_id:
            raise ValueError("El id del paquete no puede estar vacío.")
        if any(p["id"] == pkg_id for p in self.packages):
            raise ValueError(f"Ya existe un paquete con id '{pkg_id}'.")

        dest = str(pkg_dict.get("destination", "")).strip()
        if dest not in self.locations:
            raise ValueError(f"El destino '{dest}' no pertenece a las ubicaciones existentes.")

        new_pkg = {
            "id": pkg_id,
            "destination": dest,
            "window_start": int(pkg_dict.get("window_start", 0)),
            "window_end": int(pkg_dict.get("window_end", 0)),
            "promised_time": int(pkg_dict.get("promised_time", 0)),
            "priority": str(pkg_dict.get("priority", "media")).lower(),
            "service_time": int(pkg_dict.get("service_time", 5)),
        }
        self.packages.append(new_pkg)

    def update_package(self, pkg_id: str, updated_dict: dict[str, Any]) -> None:
        """Actualiza un paquete existente por su identificador."""
        target_idx = next((i for i, p in enumerate(self.packages) if p["id"] == pkg_id), None)
        if target_idx is None:
            raise ValueError(f"No existe el paquete '{pkg_id}' para actualizar.")

        new_id = str(updated_dict.get("id", pkg_id)).strip()
        if not new_id:
            raise ValueError("El id del paquete no puede estar vacío.")
        if new_id != pkg_id and any(p["id"] == new_id for p in self.packages):
            raise ValueError(f"Ya existe otro paquete con id '{new_id}'.")

        dest = str(updated_dict.get("destination", self.packages[target_idx]["destination"])).strip()
        if dest not in self.locations:
            raise ValueError(f"El destino '{dest}' no existe en las ubicaciones actuales.")

        self.packages[target_idx] = {
            "id": new_id,
            "destination": dest,
            "window_start": int(updated_dict.get("window_start", self.packages[target_idx]["window_start"])),
            "window_end": int(updated_dict.get("window_end", self.packages[target_idx]["window_end"])),
            "promised_time": int(updated_dict.get("promised_time", self.packages[target_idx]["promised_time"])),
            "priority": str(updated_dict.get("priority", self.packages[target_idx]["priority"])).lower(),
            "service_time": int(updated_dict.get("service_time", self.packages[target_idx]["service_time"])),
        }

    def remove_package(self, pkg_id: str) -> None:
        """Elimina un paquete por su identificador."""
        initial_len = len(self.packages)
        self.packages = [p for p in self.packages if p["id"] != pkg_id]
        if len(self.packages) == initial_len:
            raise ValueError(f"No se encontró el paquete con id '{pkg_id}'.")

    # =========================================================================
    # Validación y Construcción de DeliveryProblem
    # =========================================================================

    def validate(self) -> list[str]:
        """Realiza una validación exhaustiva de todas las variables del escenario.

        Devuelve una lista de cadenas con los errores encontrados. Si la lista
        está vacía, el escenario es válido y apto para ejecución.
        """
        errors: list[str] = []

        if not self.name.strip():
            errors.append("Configuración General: El nombre del escenario no puede estar vacío.")

        if self.alpha < 0:
            errors.append(f"Configuración General: El parámetro alpha ({self.alpha}) debe ser >= 0.")

        for p_key, weight in self.priority_weights.items():
            if weight <= 0:
                errors.append(f"Prioridades: El peso para '{p_key}' ({weight}) debe ser estrictamente > 0.")

        if not self.locations:
            errors.append("Ubicaciones: Debe existir al menos una ubicación declarada.")

        if self.start_location not in self.locations:
            errors.append(f"Ubicación Inicial: '{self.start_location}' no existe entre las ubicaciones.")

        if self.start_time < 0:
            errors.append(f"Hora Inicial: La hora inicial ({self.start_time} min) debe ser >= 0.")

        if not self.packages:
            errors.append("Paquetes: Debe haber al menos un paquete para planificar.")

        # Validaciones individuales de paquetes
        seen_pkg_ids: set[str] = set()
        for idx, p in enumerate(self.packages, start=1):
            p_id = p["id"]
            if not p_id:
                errors.append(f"Paquete #{idx}: Identificador vacío.")
            elif p_id in seen_pkg_ids:
                errors.append(f"Paquete '{p_id}': Identificador duplicado.")
            seen_pkg_ids.add(p_id)

            if p["destination"] not in self.locations:
                errors.append(f"Paquete '{p_id}': Destino '{p['destination']}' no existe en las ubicaciones.")

            if p["service_time"] < 0:
                errors.append(f"Paquete '{p_id}': Tiempo de servicio negativo ({p['service_time']}).")

            if p["window_start"] < 0:
                errors.append(f"Paquete '{p_id}': Inicio de ventana negativo ({p['window_start']}).")

            if p["window_end"] < p["window_start"]:
                errors.append(
                    f"Paquete '{p_id}': Fin de ventana ({p['window_end']}) anterior al inicio ({p['window_start']})."
                )

            if p["window_end"] - p["window_start"] < p["service_time"]:
                errors.append(
                    f"Paquete '{p_id}': La ventana [{p['window_start']}, {p['window_end']}] dura menos "
                    f"que el tiempo de servicio ({p['service_time']})."
                )

            if p["window_end"] < self.start_time:
                errors.append(
                    f"Paquete '{p_id}': La ventana cierra ({p['window_end']}) antes de la hora de partida ({self.start_time})."
                )

            if not (p["window_start"] <= p["promised_time"] <= p["window_end"]):
                errors.append(
                    f"Paquete '{p_id}': Hora prometida ({p['promised_time']}) fuera de la ventana "
                    f"[{p['window_start']}, {p['window_end']}]."
                )

        # Validación de la matriz de tiempos
        for u in self.locations:
            if u not in self.travel_times:
                errors.append(f"Matriz de Tiempos: Falta la fila de origen para '{u}'.")
                continue
            for v in self.locations:
                if v not in self.travel_times[u]:
                    errors.append(f"Matriz de Tiempos: Falta la celda ({u} -> {v}).")
                    continue
                val = self.travel_times[u][v]
                if u == v and val != 0:
                    errors.append(f"Matriz de Tiempos: La diagonal ({u} -> {u}) debe ser exactamente 0.")
                elif val is not None and val < 0:
                    errors.append(f"Matriz de Tiempos: Tiempo negativo en ({u} -> {v}): {val} min.")

        return errors

    def is_official_range(self) -> tuple[bool, str]:
        """Comprueba si la cantidad de paquetes está dentro del rango oficial de benchmark (5 a 8)."""
        count = len(self.packages)
        if 5 <= count <= 8:
            return True, f"{count} paquetes (Rango oficial de benchmark: 5 a 8)"
        return False, f"{count} paquetes (Configuración de prueba/ajena al benchmark oficial [5-8])"

    def to_delivery_problem(self) -> DeliveryProblem:
        """Convierte el estado actual en una instancia inmutable de DeliveryProblem.

        Lanza ValueError si la validación encuentra inconsistencias.
        """
        errors = self.validate()
        if errors:
            raise ValueError("No se puede construir DeliveryProblem debido a errores de validación:\n" + "\n".join(errors))

        # Crear instancias de Package
        package_objs: list[Package] = []
        for p in self.packages:
            pkg = Package(
                id=p["id"],
                destination=p["destination"],
                window_start=p["window_start"],
                window_end=p["window_end"],
                promised_time=p["promised_time"],
                priority=p["priority"],
                service_time=p["service_time"],
            )
            package_objs.append(pkg)

        # Matriz validada
        matrix_copy = copy.deepcopy(self.travel_times)

        return DeliveryProblem(
            name=self.name,
            start_location=self.start_location,
            start_time=self.start_time,
            packages=tuple(package_objs),
            locations=tuple(self.locations),
            travel_times=matrix_copy,
            alpha=self.alpha,
            priority_weights=copy.deepcopy(self.priority_weights),
            allow_zero_alpha=True,
        )

    # =========================================================================
    # Ejecución de Algoritmos
    # =========================================================================

    def run_algorithm(
        self,
        algorithm_name: str,
        alpha: float | None = None,
        track_events: bool = False,
    ) -> SearchResult:
        """Ejecuta un algoritmo seleccionado sobre el escenario configurado."""
        problem = self.to_delivery_problem()
        eff_alpha = self.alpha if alpha is None else alpha
        alg_key = algorithm_name.strip().lower()

        if "a*" in alg_key or "astar" in alg_key:
            return solve_astar(problem, alpha=eff_alpha, track_events=track_events)
        elif "dijkstra" in alg_key:
            return solve_dijkstra(problem, alpha=eff_alpha)
        elif "greedy" in alg_key or "vecino" in alg_key:
            return solve_greedy(problem, alpha=eff_alpha)
        elif "exhaustiv" in alg_key:
            return solve_exhaustive(problem, alpha=eff_alpha).to_search_result()
        else:
            raise ValueError(f"Algoritmo no reconocido: '{algorithm_name}'")

    def run_comparison(self, alpha: float | None = None) -> dict[str, SearchResult]:
        """Ejecuta los cuatro algoritmos sobre el mismo escenario y devuelve sus resultados."""
        eff_alpha = self.alpha if alpha is None else alpha
        problem = self.to_delivery_problem()

        results: dict[str, SearchResult] = {
            "A*": solve_astar(problem, alpha=eff_alpha, track_events=False),
            "Dijkstra": solve_dijkstra(problem, alpha=eff_alpha),
            "Greedy": solve_greedy(problem, alpha=eff_alpha),
            "Exhaustive": solve_exhaustive(problem, alpha=eff_alpha).to_search_result(),
        }
        return results

