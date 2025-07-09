poetry run writer-pipeline docs/TOC_short.md
poetry run writer-pipeline docs/TOC_short.md --batch-size 5 --max-iterations 100 --author "Stanisław Lem" --title "Meta Nihilizm III ciego stopnia"

### SUMARY QA TASK

poetry run luigi --module pipeline.writer.tasks.final_qa_task FinalQATask --toc-path "docs/TOC_short.md" --local-scheduler

GET /document/full - Pełny dokument jako plain text
GET /document/stats - Statystyki dokumentu (słowa, znaki, rozdziały)
GET /document/toc - Spis treści z statusami
GET /document/progress - Progress pisania (ile % gotowe)
