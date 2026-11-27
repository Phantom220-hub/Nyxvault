# Central training-data directory

This directory is the canonical home for NYXVAULT training input datasets and dataset-related artifacts.

- `programming_seed.jsonl`: 103-record starter corpus (89 legacy Q&A and 14 structured records).
- `code_projects_seed.jsonl`: 18 original structured coding tasks, including a tested calculator, Python algorithms, Godot 4 gameplay systems, Kotlin architecture and a JavaScript Canvas game.
- `raw/`: collected, user-owned, or properly licensed source material before preparation.
- `processed/`: train/validation text and preparation manifest.
- `reports/`: dataset-validation reports.

Keep model checkpoints and export binaries outside this directory. Do not mix generated model weights with source datasets. Keep attribution/license metadata alongside the source data it describes.

The CPU smoke test uses synthetic tokens and intentionally does not read or alter these datasets.
