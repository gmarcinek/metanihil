poetry run python pipeline/structure/run_structure_pipeline.py docs/owu2.pdf
poetry run writer-pipeline docs/owu2.pdf

GET /document/full - Pełny dokument jako plain text
GET /document/stats - Statystyki dokumentu (słowa, znaki, rozdziały)
GET /document/toc - Spis treści z statusami
GET /document/progress - Progress pisania (ile % gotowe)
