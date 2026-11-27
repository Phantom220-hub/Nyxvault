# NYXVAULT training system

All model source code and training data live in this repository. The current model itself has one canonical location: `NYXVAULT_MODEL.zip` at the root of `main` (Git LFS).

## Canonical model and continuous training

- `NYXVAULT_MODEL.zip` contains the current `model.pt`, matching tokenizer, Android ONNX export, combined training corpus, manifest, inference checks and immutable `base_step_600_backup.pt`.
- `training/finetune_instructions.py` performs resumable SFT and restores optimizer/RNG state from the canonical checkpoint.
- `.github/workflows/model-inference-test.yml` is the only active model-training workflow. It reads the root bundle from Git LFS, adds validated data from `training/data/*.jsonl`, resumes directly from the bundle's `model.pt`, exports and checks ONNX inference, then updates the same `NYXVAULT_MODEL.zip` in `main`.
- Push new JSONL training examples to `training/data/`, or start the workflow with GitHub Actions `workflow_dispatch`. Do not start separate SFT pipelines or publish independent checkpoint artifacts/releases.

## Source code

- `model.py`: decoder-only GPT-style Transformer.
- `train.py`: base next-token training.
- `train_tokenizer.py`: ByteLevel BPE tokenizer training.
- `prepare_data.py`: prepares base-training text.
- `evaluate.py`: validation metrics and prompt checks.
- `finetune_instructions.py`: supervised instruction tuning with prompt-loss masking and resumable optimizer state.
- `export_mobile.py`: exports the current checkpoint and matching Android tokenizer.
- `data_schema.py` and `curriculum.py`: canonical instruction-data schema and topic validation.

The Colab/Kaggle notebooks are legacy research utilities, not the canonical model-update process. The old release-upload helper is retired. Do not resume from historical Release assets or Actions artifacts.

## Training data

Validated instruction datasets are kept under `training/data/*.jsonl`. The canonical bundle preserves the combined, de-duplicated corpus so later training continues with earlier examples as well as newly committed records.

## Run checks

```bash
python training/validate_data.py training/data/programming_seed.jsonl
python -m unittest discover -s training/tests -v
```

Only run `--exec-tests` on trusted dataset code. The validator uses time and resource limits but is not a complete sandbox.

## Storage rules

There is one model file in the visible repository root: `NYXVAULT_MODEL.zip`. It is tracked by Git LFS; the pointer file in a normal Git clone is not the full 200 MB archive. Use `git lfs pull` to download its actual contents.

Every training run must use the previous `model.pt` from this archive as its resume checkpoint, preserve `base_step_600_backup.pt`, and replace the same archive in `main`. Never commit raw model binaries to ordinary Git history, create separate model directories, or publish separate checkpoints as workflow artifacts/releases.
