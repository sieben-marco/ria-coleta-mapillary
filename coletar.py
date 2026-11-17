from __future__ import annotations

import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

MAPILLARY_ACCESS_TOKEN = os.getenv("MAPILLARY_ACCESS_TOKEN", "").strip()
PLACEHOLDER_TOKENS = {"", "MLY|seu_token_aqui", "seu_token_aqui"}

# Bounding Box: Porto Alegre (Centro Histórico / Cidade Baixa / Bom Fim / Floresta)
# Formato: [min_lon, min_lat, max_lon, max_lat]
POA_BBOX: list[float] = [-51.2350, -30.0450, -51.2050, -30.0250]
STEP: float = 0.008  # Abaixo do limite de 0.01° da Graph API v4

API_URL = "https://graph.mapillary.com/images"
FIELDS = "id,thumb_2048_url,computed_geometry,computed_compass_angle,captured_at,sequence,is_pano"
START_CAPTURED_AT = "2023-01-01T00:00:00Z"
LIMIT = 50
MIN_DISTANCE_METERS = 15.0
SEED = 42

OUTPUT_DIR = Path("dataset_poa_mapillary")
IMAGES_DIR = OUTPUT_DIR / "images"
METADATA_CSV = OUTPUT_DIR / "metadados_poa.csv"


def validar_token(token: str) -> None:
    """Valida se o token do Mapillary foi configurado corretamente no .env."""
    if token in PLACEHOLDER_TOKENS:
        print(
            "ERRO: MAPILLARY_ACCESS_TOKEN ausente ou com valor placeholder.\n"
            "Configure seu token válido da Mapillary Graph API v4 no arquivo .env\n"
            "Exemplo: MAPILLARY_ACCESS_TOKEN=MLY|1234567890...",
            file=sys.stderr,
        )
        sys.exit(1)


def gerar_grade_bbox(
    bbox: list[float], step: float = STEP
) -> list[tuple[float, float, float, float]]:
    """Divide a Bounding Box principal em células menores que o limite da API v4 (0.01°)."""
    min_lon, min_lat, max_lon, max_lat = bbox
    celulas: list[tuple[float, float, float, float]] = []

    lon = min_lon
    while lon < max_lon - 1e-9:
        next_lon = min(lon + step, max_lon)
        lat = min_lat
        while lat < max_lat - 1e-9:
            next_lat = min(lat + step, max_lat)
            celulas.append((round(lon, 6), round(lat, 6), round(next_lon, 6), round(next_lat, 6)))
            lat = next_lat
        lon = next_lon

    return celulas


def distancia_haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calcula a distância geodésica em metros entre dois pontos (fórmula de Haversine)."""
    raio_terra_m = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return raio_terra_m * c


def consultar_celula(
    session: requests.Session,
    celula: tuple[float, float, float, float],
    headers: dict[str, str],
) -> list[dict[str, Any]]:
    """Consulta a Graph API v4 do Mapillary para uma célula com suporte a paginação."""
    min_lon, min_lat, max_lon, max_lat = celula
    bbox_str = f"{min_lon},{min_lat},{max_lon},{max_lat}"

    params: dict[str, str | int] | None = {
        "bbox": bbox_str,
        "fields": FIELDS,
        "is_pano": "false",
        "start_captured_at": START_CAPTURED_AT,
        "limit": LIMIT,
    }

    url: str | None = API_URL
    registros: list[dict[str, Any]] = []

    while url:
        try:
            response = session.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            print(f"  [!] Erro ao consultar célula {bbox_str}: {exc}", file=sys.stderr)
            break

        for item in payload.get("data", []):
            if item.get("is_pano") is True:
                continue

            image_id = str(item.get("id", "")).strip()
            thumb_url = item.get("thumb_2048_url")
            geom = item.get("computed_geometry", {})
            coords = geom.get("coordinates", [])

            if not image_id or not thumb_url or len(coords) != 2:
                continue

            lon, lat = float(coords[0]), float(coords[1])
            registros.append(
                {
                    "image_id": image_id,
                    "thumb_2048_url": str(thumb_url),
                    "longitude": lon,
                    "latitude": lat,
                    "compass_angle": item.get("computed_compass_angle"),
                    "sequence_id": str(item.get("sequence", "sem_sequencia")),
                    "captured_at": item.get("captured_at", 0),
                }
            )

        url = payload.get("paging", {}).get("next")
        params = None  # A URL de paginação já contém todos os query params

    return registros


def deduplicar_espacialmente(
    registros: list[dict[str, Any]], distancia_min_m: float = MIN_DISTANCE_METERS
) -> list[dict[str, Any]]:
    """
    Ordena metadados por (sequence_id, captured_at) e descarta frames na mesma
    sequência que estejam a menos de `distancia_min_m` metros do último frame aceito.
    """
    # Remover eventuais IDs duplicados nas bordas entre células da grade
    unicos_por_id: dict[str, dict[str, Any]] = {r["image_id"]: r for r in registros}
    lista_ordenada = sorted(
        unicos_por_id.values(),
        key=lambda r: (str(r["sequence_id"]), int(r["captured_at"] or 0), str(r["image_id"])),
    )

    aceitos: list[dict[str, Any]] = []
    ultimo_por_seq: dict[str, tuple[float, float]] = {}

    for item in lista_ordenada:
        seq_id = str(item["sequence_id"])
        lat = float(item["latitude"])
        lon = float(item["longitude"])

        if seq_id in ultimo_por_seq:
            ult_lat, ult_lon = ultimo_por_seq[seq_id]
            dist = distancia_haversine(ult_lat, ult_lon, lat, lon)
            if dist < distancia_min_m:
                continue

        aceitos.append(item)
        ultimo_por_seq[seq_id] = (lat, lon)

    return aceitos


def atribuir_split_por_sequencia(
    registros: list[dict[str, Any]], seed: int = SEED
) -> dict[str, str]:
    """
    Divide as sequências de forma determinística (70% train, 20% val, 10% test)
    agrupando estritamente por sequence_id para evitar vazamento de dados (data leakage).
    """
    sequencias = sorted({str(r["sequence_id"]) for r in registros})
    rng = random.Random(seed)
    rng.shuffle(sequencias)

    n_total = len(sequencias)
    n_train = int(round(n_total * 0.70))
    n_val = int(round(n_total * 0.20))

    # Garantir que os limites não ultrapassem n_total
    if n_train + n_val > n_total:
        n_val = max(0, n_total - n_train)

    mapa_split: dict[str, str] = {}
    for idx, seq_id in enumerate(sequencias):
        if idx < n_train:
            mapa_split[seq_id] = "train"
        elif idx < n_train + n_val:
            mapa_split[seq_id] = "val"
        else:
            mapa_split[seq_id] = "test"

    return mapa_split


def baixar_imagem(session: requests.Session, url: str, destino: Path) -> bool:
    """Baixa a imagem em 2048px caso ainda não exista em disco."""
    if destino.exists() and destino.stat().st_size > 0:
        return True

    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        destino.write_bytes(resp.content)
        return True
    except requests.RequestException as exc:
        print(f"  [!] Falha ao baixar {destino.name}: {exc}", file=sys.stderr)
        return False


def main() -> None:
    validar_token(MAPILLARY_ACCESS_TOKEN)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"OAuth {MAPILLARY_ACCESS_TOKEN}"}

    celulas = gerar_grade_bbox(POA_BBOX, STEP)
    print(f"Iniciando coleta em Porto Alegre ({len(celulas)} células na grade)...")

    todos_registros: list[dict[str, Any]] = []
    with requests.Session() as session:
        for idx, celula in enumerate(celulas, start=1):
            print(f"[{idx}/{len(celulas)}] Consultando célula {celula}...")
            encontrados = consultar_celula(session, celula, headers)
            print(f"  -> {len(encontrados)} imagens brutas retornadas.")
            todos_registros.extend(encontrados)
            time.sleep(0.2)

        if not todos_registros:
            print("Nenhuma imagem encontrada para os critérios informados.")
            return

        filtrados = deduplicar_espacialmente(todos_registros, MIN_DISTANCE_METERS)
        print(
            f"\nTotal bruto: {len(todos_registros)} | "
            f"Após deduplicação espacial (>= {MIN_DISTANCE_METERS}m): {len(filtrados)}"
        )

        mapa_split = atribuir_split_por_sequencia(filtrados, seed=SEED)

        linhas_csv: list[dict[str, Any]] = []
        for idx, item in enumerate(filtrados, start=1):
            image_id = item["image_id"]
            filename = f"{image_id}.jpg"
            caminho_img = IMAGES_DIR / filename

            if idx % 25 == 0 or idx == len(filtrados):
                print(f"Baixando imagens: {idx}/{len(filtrados)}...")

            sucesso = baixar_imagem(session, item["thumb_2048_url"], caminho_img)
            if not sucesso:
                continue

            seq_id = str(item["sequence_id"])
            linhas_csv.append(
                {
                    "image_id": image_id,
                    "filename": filename,
                    "longitude": item["longitude"],
                    "latitude": item["latitude"],
                    "compass_angle": item["compass_angle"],
                    "sequence_id": seq_id,
                    "captured_at": item["captured_at"],
                    "split": mapa_split[seq_id],
                }
            )

    colunas = [
        "image_id",
        "filename",
        "longitude",
        "latitude",
        "compass_angle",
        "sequence_id",
        "captured_at",
        "split",
    ]
    df = pd.DataFrame(linhas_csv, columns=colunas)
    df.to_csv(METADATA_CSV, index=False)

    print(f"\nMetadados salvos em: {METADATA_CSV} ({len(df)} registros)")
    if not df.empty:
        resumo = (
            df.groupby("split")
            .agg(
                imagens=("image_id", "count"),
                sequencias=("sequence_id", "nunique"),
            )
            .reindex(["train", "val", "test"], fill_value=0)
        )
        resumo["percentual_imagens (%)"] = (resumo["imagens"] / len(df) * 100).round(1)
        print("\n=== Resumo Estatístico por Split ===")
        print(resumo.to_string())


if __name__ == "__main__":
    main()
