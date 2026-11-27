# NYXVAULT training data

This folder contains the canonical, schema-validated JSONL datasets used by the project.

## Canonical project dataset

- `code_projects_seed.jsonl`: programming and project examples across Python, GDScript, Kotlin, and JavaScript. The Godot 4 examples from the November 27, 2026 SFT batch have been merged into this file.
- `programming_seed.jsonl`: general programming examples (includes legacy question/answer entries and structured records).

New structured project examples should be validated against `training/data_schema.py` and `training/curriculum.py`, then added to `code_projects_seed.jsonl` when they are project/code examples.

Validation:

```bash
python training/validate_data.py training/data/code_projects_seed.jsonl --exec-tests
```

Only run executable checks on trusted code. GDScript examples are schema-validated here; execute them in Godot separately before treating them as runtime-verified.
