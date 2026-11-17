@AGENTS.md

# Regra de Projeto para Antigravity CLI (`agy`) e Gemini CLI

Adote [`AGENTS.md`](../../AGENTS.md) como a especificação principal e Fonte Única da Verdade deste repositório.

Regras obrigatórias:
1. Utilize exclusivamente `uv` (`uv add`, `uv sync`, `uv run`) para gerenciamento de pacotes e execução.
2. Mantenha `uv run ruff check .` e `uv run ruff format --check .` sempre limpos (`line-length = 100`, `py312`).
3. Preserve a configuração multiplataforma do PyTorch no `pyproject.toml` (`pytorch-cpu` apenas para `sys_platform == 'linux'`).
4. Preserve o particionamento `70/20/10` agrupado por `sequence_id` (`seed=42`) e a deduplicação espacial via Haversine (`15.0m`).
5. Nunca adicione VLMs ao pipeline de produção; o modelo final alvo é `YOLO11s-seg` (`1024x1024`) com pós-processamento geométrico determinístico sobre as classes `poste`, `aglomerado_fios` e `cabo_solto`.
