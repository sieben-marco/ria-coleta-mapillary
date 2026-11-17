from __future__ import annotations

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

DATASET_DIR = Path("dataset_poa_mapillary")
IMAGES_DIR = DATASET_DIR / "images"
COM_POSTES_DIR = DATASET_DIR / "com_postes"
SEM_POSTES_DIR = DATASET_DIR / "sem_postes"
METADATA_IN_CSV = DATASET_DIR / "metadados_poa.csv"
METADATA_OUT_CSV = DATASET_DIR / "metadados_com_postes.csv"

MODEL_WEIGHTS = "yolov8s-worldv2.pt"
CLASSES_ALVO = ["utility pole", "electricity pole", "light pole"]
CONF_THRESHOLD = 0.18
IMGSZ = 640
MIN_HEIGHT_NORM = 0.15

CLIP_VIT_B32_SHA256 = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
CLIP_VIT_B32_URL = (
    f"https://openaipublic.azureedge.net/clip/models/{CLIP_VIT_B32_SHA256}/ViT-B-32.pt"
)


def detectar_dispositivo() -> str:
    """Detecta automaticamente o acelerador de hardware disponível (MPS, CUDA ou CPU)."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def calcular_sha256(caminho: Path) -> str:
    """Calcula o hash SHA256 de um arquivo em blocos para baixo consumo de memória."""
    hasher = hashlib.sha256()
    with caminho.open("rb") as f:
        for bloco in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(bloco)
    return hasher.hexdigest()


def garantir_pesos_clip(max_tentativas: int = 8) -> Path:
    """
    Garante que o arquivo ViT-B-32.pt do CLIP esteja íntegro em WEIGHTS_DIR / 'clip'.
    Suporta retomada de download via HTTP Range caso a conexão caia antes de 100%,
    evitando o RuntimeError de SHA256 do downloader padrão do pacote clip.
    """
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
            tamanho_esperado = 0

        for tentativa in range(1, max_tentativas + 1):
            bytes_atuais = destino.stat().st_size if destino.is_file() else 0

            if tamanho_esperado > 0 and bytes_atuais >= tamanho_esperado:
                if calcular_sha256(destino) == CLIP_VIT_B32_SHA256:
                    return destino
                print("  [!] Arquivo CLIP corrompido detectado; reiniciando download do zero...")
                destino.unlink(missing_ok=True)
                bytes_atuais = 0

            headers: dict[str, str] = {}
            if bytes_atuais > 0:
                headers["Range"] = f"bytes={bytes_atuais}-"
                mb_atuais = bytes_atuais / (1024 * 1024)
                print(
                    f"Retomando download do CLIP ViT-B-32.pt a partir de {mb_atuais:.1f} MiB "
                    f"(tentativa {tentativa}/{max_tentativas})..."
                )
            else:
                print(
                    f"Baixando pesos do CLIP ViT-B-32.pt "
                    f"(tentativa {tentativa}/{max_tentativas})..."
                )

            try:
                with session.get(
                    CLIP_VIT_B32_URL, headers=headers, stream=True, timeout=60
                ) as resp:
                    if resp.status_code == 416:
                        destino.unlink(missing_ok=True)
                        continue
                    resp.raise_for_status()

                    modo = "ab" if resp.status_code == 206 and bytes_atuais > 0 else "wb"
                    with destino.open(modo) as out_file:
                        for chunk in resp.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                out_file.write(chunk)
            except requests.RequestException as exc:
                print(
                    f"  [!] Oscilação de rede durante download do CLIP ({exc}). Retentando...",
                    file=sys.stderr,
                )
                time.sleep(2)
                continue

            if destino.is_file() and calcular_sha256(destino) == CLIP_VIT_B32_SHA256:
                print("Pesos do CLIP (ViT-B-32.pt) verificados com sucesso (SHA256 válido).")
                return destino

            # Se ainda estiver menor que o tamanho esperado, a conexão fechou antes do fim (EOF)
            tamanho_final = destino.stat().st_size if destino.is_file() else 0
            if tamanho_esperado > 0 and tamanho_final < tamanho_esperado:
                continue

            # Se chegou ao tamanho total mas o hash não bateu, apaga e recomeça
            destino.unlink(missing_ok=True)

    print(
        "ERRO: Não foi possível validar o SHA256 do modelo CLIP ViT-B-32.pt após "
        f"{max_tentativas} tentativas.",
        file=sys.stderr,
    )
    sys.exit(1)


def imagem_possui_poste_valido(model: YOLOWorld, caminho_imagem: Path, device: str) -> bool:
    """
    Executa inferência Zero-Shot com YOLO-World e aprova a imagem somente se
    houver ao menos uma detecção com altura normalizada (h_norm) >= 0.15.
    """
    resultados = model.predict(
        source=str(caminho_imagem),
        conf=CONF_THRESHOLD,
        imgsz=IMGSZ,
        device=device,
        verbose=False,
    )

    for result in resultados:
        if result.boxes is None or len(result.boxes) == 0:
            continue
        for xywhn in result.boxes.xywhn:
            h_norm = float(xywhn[3])
            if h_norm >= MIN_HEIGHT_NORM:
                return True

    return False


def main() -> None:
    if not METADATA_IN_CSV.exists():
        print(
            f"ERRO: Arquivo de metadados não encontrado em {METADATA_IN_CSV}.\n"
            "Execute 'uv run coletar.py' antes de rodar a filtragem.",
            file=sys.stderr,
        )
        sys.exit(1)

    COM_POSTES_DIR.mkdir(parents=True, exist_ok=True)
    SEM_POSTES_DIR.mkdir(parents=True, exist_ok=True)

    device = detectar_dispositivo()
    print(f"Dispositivo de inferência selecionado: {device}")

    print(f"Carregando modelo Zero-Shot {MODEL_WEIGHTS}...")
    model = YOLOWorld(MODEL_WEIGHTS)
    garantir_pesos_clip()
    model.set_classes(CLASSES_ALVO)

    df = pd.read_csv(METADATA_IN_CSV)
    if df.empty:
        print("O arquivo de metadados está vazio. Nenhuma imagem para filtrar.")
        return

    registros_aprovados = []
    total = len(df)
    aprovadas = 0
    rejeitadas = 0

    for idx, row in df.iterrows():
        filename = str(row["filename"])
        caminho_origem = IMAGES_DIR / filename

        if not caminho_origem.exists():
            print(f"  [!] Imagem ausente em disco, ignorando: {caminho_origem}", file=sys.stderr)
            continue

        possui_poste = imagem_possui_poste_valido(model, caminho_origem, device=device)

        if possui_poste:
            shutil.copy2(caminho_origem, COM_POSTES_DIR / filename)
            registros_aprovados.append(row)
            aprovadas += 1
        else:
            shutil.copy2(caminho_origem, SEM_POSTES_DIR / filename)
            rejeitadas += 1

        progresso = int(idx) + 1
        if progresso % 25 == 0 or progresso == total:
            print(
                f"Processadas: {progresso}/{total} | "
                f"Com postes: {aprovadas} | Sem postes: {rejeitadas}"
            )

    df_aprovados = pd.DataFrame(registros_aprovados, columns=df.columns)
    df_aprovados.to_csv(METADATA_OUT_CSV, index=False)

    print("\n=== Resumo da Triagem YOLO-World ===")
    print(f"Total avaliado : {aprovadas + rejeitadas}")
    print(f"Com postes     : {aprovadas} -> {COM_POSTES_DIR}")
    print(f"Sem postes     : {rejeitadas} -> {SEM_POSTES_DIR}")
    print(f"Metadados      : {METADATA_OUT_CSV}")

    if not df_aprovados.empty and "split" in df_aprovados.columns:
        print("\nDistribuição por split (imagens aprovadas):")
        print(df_aprovados["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
