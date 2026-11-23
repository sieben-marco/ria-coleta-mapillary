import sys
import time

import pandas as pd
import requests

from .config import (
    API_LIMIT,
    BBOX_STEP,
    DATASET_DIR,
    IMAGES_DIR,
    MAPILLARY_ACCESS_TOKEN,
    METADATA_CSV,
    PLACEHOLDER_TOKENS,
    POA_BBOX,
    MIN_DISTANCE_METERS,
    RANDOM_SEED,
)
from .filtros import atribuir_split_por_sequencia, deduplicar_espacialmente
from .mapillary import baixar_imagem, consultar_celula
from .utils import gerar_grade_bbox


def validar_token() -> None:
    if MAPILLARY_ACCESS_TOKEN in PLACEHOLDER_TOKENS:
        raise SystemExit(
            "ERRO: MAPILLARY_ACCESS_TOKEN ausente ou inválido. "
            "Configure o token no arquivo .env."
        )


def main() -> None:
    validar_token()
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    headers = {"Authorization": f"OAuth {MAPILLARY_ACCESS_TOKEN}"}
    celulas = gerar_grade_bbox(POA_BBOX, BBOX_STEP)
    print(f"Iniciando coleta em Porto Alegre ({len(celulas)} células)...")

    todos_registros = []
    with requests.Session() as session:
        for indice, celula in enumerate(celulas, start=1):
            print(f"[{indice}/{len(celulas)}] Consultando célula {celula}...")
            encontrados = consultar_celula(session, celula, headers)
            print(f"  -> {len(encontrados)} imagens retornadas.")
            todos_registros.extend(encontrados)
            time.sleep(0.2)

        if not todos_registros:
            print("Nenhuma imagem encontrada para os critérios informados.")
            pd.DataFrame(columns=[
                "image_id", "filename", "longitude", "latitude",
                "compass_angle", "sequence_id", "captured_at", "split",
            ]).to_csv(METADATA_CSV, index=False)
            return

        filtrados = deduplicar_espacialmente(todos_registros, MIN_DISTANCE_METERS)
        print(
            f"\\nTotal bruto: {len(todos_registros)} | "
            f"Após deduplicação: {len(filtrados)}"
        )
        mapa_split = atribuir_split_por_sequencia(filtrados, seed=RANDOM_SEED)

        linhas_csv = []
        for indice, item in enumerate(filtrados, start=1):
            filename = f"{item['image_id']}.jpg"
            destino = IMAGES_DIR / filename

            if indice % 25 == 0 or indice == len(filtrados):
                print(f"Baixando imagens: {indice}/{len(filtrados)}...")

            if not baixar_imagem(session, item["thumb_1024_url"], destino):
                continue

            sequence_id = str(item["sequence_id"])
            linhas_csv.append({
                "image_id": item["image_id"],
                "filename": filename,
                "longitude": item["longitude"],
                "latitude": item["latitude"],
                "compass_angle": item["compass_angle"],
                "sequence_id": sequence_id,
                "captured_at": item["captured_at"],
                "split": mapa_split[sequence_id],
            })

    columns = [
        "image_id", "filename", "longitude", "latitude",
        "compass_angle", "sequence_id", "captured_at", "split",
    ]
    dataframe = pd.DataFrame(linhas_csv, columns=columns)
    dataframe.to_csv(METADATA_CSV, index=False)
    print(f"\\nMetadados salvos: {METADATA_CSV} ({len(dataframe)} registros)")

    if not dataframe.empty:
        resumo = dataframe.groupby("split").agg(
            imagens=("image_id", "count"),
            sequencias=("sequence_id", "nunique"),
        ).reindex(["train", "val", "test"], fill_value=0)
        resumo["percentual_imagens (%)"] = (
            resumo["imagens"] / len(dataframe) * 100
        ).round(1)
        print("\\n=== Resumo por split ===")
        print(resumo.to_string())


if __name__ == "__main__":
    main()
