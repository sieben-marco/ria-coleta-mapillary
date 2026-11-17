# AGENTS.md — Especificação Técnica e Regras para Agentes de IA

Este documento é a **Fonte Única da Verdade (Single Source of Truth)** para todos os agentes de IA (`Claude Code`, `Gemini CLI`, `Antigravity CLI / agy`, `Cursor`, etc.) que atuam neste repositório.

---

## 1. Contexto e Problema Urbano (Porto Alegre / RS)

O projeto integra uma **Residência em Inteligência Artificial** voltada à fiscalização automatizada de infraestrutura urbana em **Porto Alegre (RS)**, com foco no acúmulo irregular de fiação elétrica e de telecomunicações ("fios mortos" e cabos rompidos) em postes urbanos.

### Alinhamento com os Objetivos de Desenvolvimento Sustentável (ONU)
- **ODS 9 (Indústria, Inovação e Infraestrutura):** Modernização do monitoramento de infraestrutura crítica de energia e telecomunicações.
- **ODS 11 (Cidades e Comunidades Sustentáveis):** Redução da poluição visual, prevenção de acidentes com cabos baixos/energizados e melhoria da segurança viária e de pedestres.
- **ODS 12 (Consumo e Produção Responsáveis):** Gestão de resíduos sólidos tecnológicos (cabos obsoletos abandonados pelas operadoras) e responsabilização compartilhada.

---

## 2. Arquitetura de Visão Computacional

### 2.1. Modelo de Produção (`YOLO11s-seg` em `1024x1024`)
- **Tarefa:** Segmentação de Instâncias (*Instance Segmentation*). Caixas delimitadoras (*Bounding Boxes*) são insuficientes porque cabos atravessam a imagem diagonalmente e incluem grande área de fundo (céu, árvores, prédios). Máscaras de polígonos permitem medir a área exata ocupada pela massa de cabos.
- **Arquitetura:** `YOLO11s-seg` treinado na resolução **`1024x1024`** (`imgsz=1024`), essencial para preservar a espessura de cabos finos e feixes distantes.
- **Coleta em Alta Resolução:** As imagens são baixadas do Mapillary em `2048px` (`thumb_2048_url`) com descarte estrito de panoramas 360° (`is_pano=false`) para evitar distorções equiretangulares.

### 2.2. Classes-Alvo do Dataset Final
| ID | Classe | Descrição |
| :--- | :--- | :--- |
| `0` | `poste` | Estrutura vertical principal do poste urbano (concreto, madeira ou metálico). |
| `1` | `aglomerado_fios` | Massa densa, emaranhado ou rolo de cabos/reserva técnica acumulada no poste ou nos vãos adjacentes. |
| `2` | `cabo_solto` | Fio ou cabo partido, pendurado verticalmente ou caído em direção à calçada/via. |

### 2.3. Pós-Processamento Geométrico Determinístico
A classificação de criticidade do poste **não** depende de julgamento subjetivo de LLMs/VLMs, mas sim de regras geométricas determinísticas calculadas sobre as máscaras segmentadas:

1. **Índice de Acúmulo:**
   $$\text{Índice de Acúmulo} = \frac{\text{Área}(\text{aglomerado\_fios})}{\text{Área}(\text{poste})}$$
   Normaliza a densidade da fiação pela área aparente do poste na imagem, tornando a métrica invariante à distância entre a câmera e o poste.
2. **Verificação Vertical de Risco (`cabo_solto`):**
   Analisa a projeção vertical da máscara de `cabo_solto` em relação à metade inferior da altura do `poste` (zona de circulação de pedestres e veículos), disparando alerta prioritário de segurança.

### 2.4. Papel Estrito do `YOLOWorld` e Exclusão de VLMs
- **`YOLOWorld` (`yolov8s-worldv2.pt`):** Utilizado **exclusivamente** no script `filtrar_postes.py` como filtro *offline* Zero-Shot de pré-triagem para descartar imagens de rua sem postes visíveis (`h_norm >= 0.15`) **antes** da etapa humana de anotação assistida no **CVAT + SAM 2**.
- **Fora de Escopo (VLMs):** Modelos de Linguagem e Visão (*Vision-Language Models*) estão **explicitamente fora do escopo** do pipeline de produção por custo, latência e falta de precisão geométrica pixel a pixel.

---

## 3. Pipeline de Dados e Prevenção de Vazamento (*Data Leakage*)

1. **Grade Geográfica (`coletar.py`):**
   - Bounding Box de Porto Alegre (`POA_BBOX = [-51.2350, -30.0450, -51.2050, -30.0250]`, cobrindo Centro Histórico, Cidade Baixa, Bom Fim e Floresta) subdividida em células de `0.008°` (respeitando o teto de `0.01°` da Mapillary Graph API v4).
2. **Deduplicação Espacial por Haversine:**
   - Frames de uma mesma sequência (`sequence_id`) capturados a menos de **`15.0` metros** do último frame aceito são descartados **antes do download** para evitar redundância de imagens em semáforos ou congestionamentos.
3. **Particionamento Agrupado por `sequence_id` (`70/20/10`):**
   - O split determinístico (`seed=42`, `train`: 70%, `val`: 20%, `test`: 10%) é realizado **estritamente no nível de `sequence_id`**.
   - **Regra Inviolável:** Nunca dividir imagens aleatoriamente sem agrupar por `sequence_id`, pois fotos consecutivas da mesma rua mostrariam o mesmo poste em `train` e `val`/`test`.

---

## 4. Regras de Engenharia e Ferramental

1. **Gerenciamento de Pacotes e Execução (`uv`):**
   - **Sempre** utilize `uv add <pacote>`, `uv add --dev <pacote>`, `uv sync` e `uv run <script>.py`.
   - **Proibido** usar `pip install`, `poetry` ou ativar ambientes virtuais manualmente nos scripts de automação.
2. **Portabilidade Multiplataforma (`pyproject.toml`):**
   - **Linux (`sys_platform == 'linux'`):** `torch` e `torchvision` devem vir **exclusivamente** do índice `pytorch-cpu` (`https://download.pytorch.org/whl/cpu`), evitando o download de gigabytes de bibliotecas CUDA desnecessárias em máquinas AMD/CPU.
   - **macOS (`darwin`) e Windows (`win32`):** Utilizam o PyPI padrão (habilitando aceleração `mps` no Apple Silicon e `cuda`/`cpu` no Windows).
   - O código deve sempre selecionar o dispositivo via `detectar_dispositivo()` (`mps` -> `cuda` -> `cpu`).
3. **Qualidade de Código e Formatação (`ruff`):**
   - Todo código Python deve respeitar `line-length = 100`, `target-version = "py312"` e regras `["E", "F", "I", "UP", "B"]`.
   - Antes de concluir qualquer alteração, execute obrigatoriamente:
     ```bash
     uv run ruff check --fix .
     uv run ruff format .
     ```
4. **Segurança e Controle de Versão (`.gitignore`):**
   - **Jamais** faça commit de `.env` (tokens da API Mapillary), pesos de modelos (`*.pt`), ambientes virtuais (`.venv/`) ou diretórios de dados (`dataset_poa_mapillary/`).
