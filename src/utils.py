import math


def gerar_grade_bbox(
    bbox: list[float], step: float
) -> list[tuple[float, float, float, float]]:
    """Divide uma bbox em células menores."""
    min_lon, min_lat, max_lon, max_lat = bbox
    celulas = []

    lon = min_lon
    while lon < max_lon - 1e-9:
        next_lon = min(lon + step, max_lon)
        lat = min_lat
        while lat < max_lat - 1e-9:
            next_lat = min(lat + step, max_lat)
            celulas.append((
                round(lon, 6), round(lat, 6),
                round(next_lon, 6), round(next_lat, 6),
            ))
            lat = next_lat
        lon = next_lon

    return celulas


def distancia_haversine(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Distância geodésica aproximada, em metros, entre duas coordenadas."""
    raio_terra_m = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2)
        * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return raio_terra_m * c
