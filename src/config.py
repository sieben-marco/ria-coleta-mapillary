from pathlib import Path
import os

from dotenv import load_dotenv

PROJECT_ROOT = Path.cwd()
load_dotenv(PROJECT_ROOT / ".env")

MAPILLARY_ACCESS_TOKEN = os.getenv("MAPILLARY_ACCESS_TOKEN", "").strip()
PLACEHOLDER_TOKENS = {"", "MLY|seu_token_aqui", "seu_token_aqui"}

# Porto Alegre: Centro Histórico / Cidade Baixa / Bom Fim / Floresta.
# Ordem: [min_lon, min_lat, max_lon, max_lat].

# AREA JARDIM ITU-SABARA
POA_BBOX = [-51.17801,-30.03878,-51.09904,-30.00185]
# AREA MATHIAS VELHO
# POA_BBOX = [-51.23569,-29.91097,-51.18178,-29.89252]
# AREA MOINHOS DE VENTO INDEPENDENCIA
# POA_BBOX = [-51.212157,-30.026783,-51.204207,-30.024479]

BBOX_STEP = 0.008

MAPILLARY_API_URL = "https://graph.mapillary.com/images"
MAPILLARY_FIELDS = (
    "id,thumb_1024_url,computed_geometry,computed_compass_angle,"
    "captured_at,sequence,is_pano"
)
START_CAPTURED_AT = "2023-01-01T00:00:00Z"
API_LIMIT = 200

MIN_DISTANCE_METERS = 15.0
RANDOM_SEED = 42

DATASET_DIR = PROJECT_ROOT / "dataset_poa_mapillary"
IMAGES_DIR = DATASET_DIR / "images"
METADATA_CSV = DATASET_DIR / "metadados_poa.csv"
COM_POSTES_DIR = DATASET_DIR / "com_postes"
SEM_POSTES_DIR = DATASET_DIR / "sem_postes"
METADATA_FILTERED_CSV = DATASET_DIR / "metadados_com_postes.csv"

# Mantenha o arquivo de pesos na raiz do projeto.
MODEL_WEIGHTS = PROJECT_ROOT / "yolov8s-worldv2.pt"
TARGET_CLASSES = ["utility pole", "electricity pole", "light pole"]
CONF_THRESHOLD = 0.18
IMAGE_SIZE = 640
MIN_NORMALIZED_HEIGHT = 0.15

CLIP_VIT_B32_SHA256 = (
    "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
)
CLIP_VIT_B32_URL = (
    f"https://openaipublic.azureedge.net/clip/models/"
    f"{CLIP_VIT_B32_SHA256}/ViT-B-32.pt"
)
