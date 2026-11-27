# NYXVAULT raw training data folder

This folder is for temporary incoming uploads only.

The canonical structured project dataset is:
`training/data/code_projects_seed.jsonl`

New programming/game examples are validated against `training/data_schema.py` and `training/curriculum.py`, then merged into that canonical file. Do not train directly from temporary/raw uploads.

The November 27, 2026 Claude source file used older field names (`task`, `code`, `difficulty`) and did not match the NYXVAULT schema. It was converted, validated, and merged into the canonical project dataset. The source and converted copies are retained in Git history; the active copy is in `training/data/code_projects_seed.jsonl`.
