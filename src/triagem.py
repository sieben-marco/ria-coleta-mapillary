import hashlib
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import torch
from ultralytics import YOLOWorld
from ultralytics.utils import WEIGHTS_DIR

from .config import (
    CLIP_VIT_B32_SHA256,
    CLIP_VIT_B32_URL,
    COM_POSTES_DIR,
    CONF_THRESHOLD,
    IMAGE_SIZE,
    METADATA_CSV,
    METADATA_FILTERED_CSV,
    MIN_NORMALIZED_HEIGHT,
    MODEL_WEIGHTS,
    SEM_POSTES_DIR,
    TARGET_CLASSES,
    IMAGES_DIR,
)


def detectar_dispositivo() -> str:
    """Seleciona MPS, CUDA ou CPU, conforme disponibilidade."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def calcular_sha256(caminho: Path) -> str:
    hasher = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            hasher.update(bloco)
    return hasher.hexdigest()


def garantir_pesos_clip(max_tentativas: int = 8) -> Path:
    """Baixa/retoma o CLIP e valida o arquivo pelo SHA256 esperado."""
    clip_dir = Path(WEIGHTS_DIR) / "clip"
    clip_dir.mkdir(parents=True, exist_ok=True)
    destino = clip_dir / "ViT-B-32.pt"

    if destino.is_file() and calcular_sha256(destino) == CLIP_VIT_B32_SHA256:
        return destino

    with requests.Session() as session:
        tamanho_esperado = 0
        try:
            head = session.head(CLIP_VIT_B32_URL, timeout=30, allow_redirects=True)
            head.raise_for_status()
            tamanho_esperado = int(head.headers.get("Content-Length", 0))
        except requests.RequestException:
            pass

        for tentativa in range(1, max_tentativas + 1):
            bytes_atuais = destino.stat().st_size if destino.is_file() else 0

            if tamanho_esperado and bytes_atuais >= tamanho_esperado:
                if calcular_sha256(destino) == CLIP_VIT_B32_SHA256:
                    return destino
                print("  [!] CLIP corrompido; reiniciando download.")
                destino.unlink(missing_ok=True)
                bytes_atuais = 0

            headers = {}
            if bytes_atuais:
                headers["Range"] = f"bytes={bytes_atuais}-"
                print(
                    f"Retomando CLIP a partir de {bytes_atuais / (1024 * 1024):.1f} MiB "
                    f"(tentativa {tentativa}/{max_tentativas})..."
                )
            else:
                print(f"Baixando CLIP (tentativa {tentativa}/{max_tentativas})...")

            try:
                with session.get(
                    CLIP_VIT_B32_URL, headers=headers, stream=True, timeout=60
                ) as response:
                    if response.status_code == 416:
                        destino.unlink(missing_ok=True)
                        continue
                    response.raise_for_status()
                    modo = (
                        "ab" if response.status_code == 206 and bytes_atuais else "wb"
                    )
                    with destino.open(modo) as arquivo:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                arquivo.write(chunk)
            except requests.RequestException as exc:
                print(f"  [!] Falha de rede: {exc}. Tentando novamente...", file=sys.stderr)
                time.sleep(2)
                continue

            if destino.is_file() and calcular_sha256(destino) == CLIP_VIT_B32_SHA256:
                print("Pesos CLIP verificados (SHA256 válido).")
                return destino

            tamanho_final = destino.stat().st_size if destino.is_file() else 0
            if tamanho_esperado and tamanho_final < tamanho_esperado:
                continue
            destino.unlink(missing_ok=True)

    raise RuntimeError(
        "Não foi possível validar o SHA256 do CLIP após "
        f"{max_tentativas} tentativas."
    )


def imagem_possui_poste_valido(model: YOLOWorld, caminho_imagem: Path, device: str) -> bool:
    resultados = model.predict(
        source=str(caminho_imagem),
        conf=CONF_THRESHOLD,
        imgsz=IMAGE_SIZE,
        device=device,
        verbose=False,
    )

    for resultado in resultados:
        if resultado.boxes is None or len(resultado.boxes) == 0:
            continue
        for xywhn in resultado.boxes.xywhn:
            if float(xywhn[3]) >= MIN_NORMALIZED_HEIGHT:
                return True
    return False


def main() -> None:
    if not METADATA_CSV.is_file():
        raise SystemExit(
            f"ERRO: metadados não encontrados em {METADATA_CSV}. "
            "Execute 'uv run python coletar.py' primeiro."
        )

    COM_POSTES_DIR.mkdir(parents=True, exist_ok=True)
    SEM_POSTES_DIR.mkdir(parents=True, exist_ok=True)

    device = detectar_dispositivo()
    print(f"Dispositivo de inferência: {device}")

    if not MODEL_WEIGHTS.is_file():
        raise SystemExit(
            f"ERRO: pesos YOLO-World não encontrados: {MODEL_WEIGHTS}. "
            "Coloque yolov8s-worldv2.pt na raiz do projeto."
        )

    print(f"Carregando modelo {MODEL_WEIGHTS.name}...")
    model = YOLOWorld(str(MODEL_WEIGHTS))
    garantir_pesos_clip()
    model.set_classes(TARGET_CLASSES)

    dataframe = pd.read_csv(METADATA_CSV)
    if dataframe.empty:
        print("O CSV está vazio. Nenhuma imagem para filtrar.")
        dataframe.iloc[0:0].to_csv(METADATA_FILTERED_CSV, index=False)
        return

    registros_aprovados = []
    aprovadas = 0
    rejeitadas = 0
    total = len(dataframe)

    for progresso, (_, row) in enumerate(dataframe.iterrows(), start=1):
        filename = str(row["filename"])
        origem = IMAGES_DIR / filename

        if not origem.is_file():
            print(f"  [!] Imagem ausente, ignorando: {origem}", file=sys.stderr)
            continue

        possui_poste = imagem_possui_poste_valido(model, origem, device)
        if possui_poste:
            shutil.copy2(origem, COM_POSTES_DIR / filename)
            registros_aprovados.append(row)
            aprovadas += 1
        else:
            shutil.copy2(origem, SEM_POSTES_DIR / filename)
            rejeitadas += 1

        if progresso % 25 == 0 or progresso == total:
            print(
                f"Processadas: {progresso}/{total} | "
                f"Com postes: {aprovadas} | Sem postes: {rejeitadas}"
            )

    aprovados_df = pd.DataFrame(registros_aprovados, columns=dataframe.columns)
    aprovados_df.to_csv(METADATA_FILTERED_CSV, index=False)

    print("\\n=== Resumo da triagem YOLO-World ===")
    print(f"Total avaliado : {aprovadas + rejeitadas}")
    print(f"Com postes     : {aprovadas} -> {COM_POSTES_DIR}")
    print(f"Sem postes     : {rejeitadas} -> {SEM_POSTES_DIR}")
    print(f"Metadados      : {METADATA_FILTERED_CSV}")

    if not aprovados_df.empty and "split" in aprovados_df.columns:
        print("\\nDistribuição por split:")
        print(aprovados_df["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
