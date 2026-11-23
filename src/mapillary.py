import sys
from typing import Any

import requests

from .config import (
    API_LIMIT,
    MAPILLARY_API_URL,
    MAPILLARY_FIELDS,
    START_CAPTURED_AT,
)


def consultar_celula(
    session: requests.Session,
    celula: tuple[float, float, float, float],
    headers: dict[str, str],
) -> list[dict[str, Any]]:
    """Consulta a API Mapillary para uma célula da área de interesse."""
    min_lon, min_lat, max_lon, max_lat = celula
    bbox_str = f"{min_lon},{min_lat},{max_lon},{max_lat}"

    params: dict[str, str | int] | None = {
        "bbox": bbox_str,
        "fields": MAPILLARY_FIELDS,
        "is_pano": "false",
        "start_captured_at": START_CAPTURED_AT,
        "limit": API_LIMIT,
    }

    url: str | None = MAPILLARY_API_URL
    registros: list[dict[str, Any]] = []
    urls_visitadas: set[str] = set()

    while url:
        if url in urls_visitadas:
            print(f"  [!] Paginação repetida detectada para {bbox_str}.", file=sys.stderr)
            break
        urls_visitadas.add(url)

        try:
            response = session.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"Falha ao consultar célula {bbox_str}: {exc}") from exc

        for item in payload.get("data", []):
            if item.get("is_pano") is True:
                continue

            image_id = str(item.get("id", "")).strip()
            thumb_url = item.get("thumb_1024_url")
            coordinates = item.get("computed_geometry", {}).get("coordinates", [])

            if not image_id or not thumb_url or len(coordinates) != 2:
                continue

            try:
                longitude, latitude = float(coordinates[0]), float(coordinates[1])
            except (TypeError, ValueError):
                continue

            registros.append({
                "image_id": image_id,
                "thumb_1024_url": str(thumb_url),
                "longitude": longitude,
                "latitude": latitude,
                "compass_angle": item.get("computed_compass_angle"),
                "sequence_id": str(item.get("sequence") or "sem_sequencia"),
                "captured_at": item.get("captured_at", 0),
            })

        url = payload.get("paging", {}).get("next")
        params = None

    return registros


def baixar_imagem(
    session: requests.Session, url: str, destino
) -> bool:
    """Baixa a imagem, preservando arquivos já existentes e não vazios."""
    if destino.is_file() and destino.stat().st_size > 0:
        return True

    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        if not response.content:
            print(f"  [!] Download vazio: {destino.name}", file=sys.stderr)
            return False
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_bytes(response.content)
        return True
    except requests.RequestException as exc:
        print(f"  [!] Falha ao baixar {destino.name}: {exc}", file=sys.stderr)
        return False
