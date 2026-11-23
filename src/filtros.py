import random
from typing import Any

from .config import MIN_DISTANCE_METERS, RANDOM_SEED
from .utils import distancia_haversine


def deduplicar_espacialmente(
    registros: list[dict[str, Any]],
    distancia_min_m: float = MIN_DISTANCE_METERS,
) -> list[dict[str, Any]]:
    """Remove IDs repetidos e frames próximos dentro da mesma sequência."""
    unicos_por_id = {registro["image_id"]: registro for registro in registros}
    ordenados = sorted(
        unicos_por_id.values(),
        key=lambda item: (
            str(item["sequence_id"]),
            int(item["captured_at"] or 0),
            str(item["image_id"]),
        ),
    )

    aceitos = []
    ultimo_por_sequencia: dict[str, tuple[float, float]] = {}

    for item in ordenados:
        sequence_id = str(item["sequence_id"])
        latitude = float(item["latitude"])
        longitude = float(item["longitude"])

        if sequence_id in ultimo_por_sequencia:
            ultima_lat, ultima_lon = ultimo_por_sequencia[sequence_id]
            distancia = distancia_haversine(
                ultima_lat, ultima_lon, latitude, longitude
            )
            if distancia < distancia_min_m:
                continue

        aceitos.append(item)
        ultimo_por_sequencia[sequence_id] = (latitude, longitude)

    return aceitos


def atribuir_split_por_sequencia(
    registros: list[dict[str, Any]], seed: int = RANDOM_SEED
) -> dict[str, str]:
    """Atribui train/val/test por sequência para reduzir data leakage."""
    sequencias = sorted({str(item["sequence_id"]) for item in registros})
    rng = random.Random(seed)
    rng.shuffle(sequencias)

    total = len(sequencias)
    n_train = int(round(total * 0.70))
    n_val = int(round(total * 0.20))
    if n_train + n_val > total:
        n_val = max(0, total - n_train)

    return {
        sequence_id: (
            "train" if index < n_train
            else "val" if index < n_train + n_val
            else "test"
        )
        for index, sequence_id in enumerate(sequencias)
    }
