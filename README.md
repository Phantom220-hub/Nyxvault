# NYXVAULT 1.0 — one AI, one Android app

**Project:** NYXVAULT — Nyxvault Studio  
**Owner:** Акрамзод Анас

NYXVAULT is a single offline-first Android assistant. Its one brain combines verified local knowledge retrieval, evidence-based answers, math tools, learning lessons, local memory, optional web research, and an optional locally-run Transformer language model.

## One product, one brain

- **Android app:** the only user-facing NYXVAULT application.
- **Knowledge and reasoning:** one local knowledge file, ranked retrieval, evidence reasoning and math tools.
- **Transformer:** the trainable generative model. After a trained ONNX model and `tokenizer_mobile.json` are imported, the same Android brain calls it when retrieval and deterministic tools cannot answer.
- **Training tools:** Python scripts under `training/` prepare data, train the one Transformer, evaluate it and export Android files. They are build tools, not another chatbot.
- **Memory and Academy:** conversation memory, long-term notes and lessons are services used by the same brain.

The old standalone Python chat, browser chat and character-level RNN are retired. Their duplicate interfaces and legacy model artifacts are no longer part of the product.

## Build Android APK

Requires JDK 17 and Gradle 8.9:

```bash
gradle --no-daemon :app:testDebugUnitTest
gradle --no-daemon :app:assembleDebug
```

GitHub Actions runs Kotlin tests and builds the debug APK.

## One canonical model

The current trained NYXVAULT model is stored in exactly one place: **`NYXVAULT_MODEL.zip` at the repository root on `main`**, tracked with Git LFS.

Download the repository with Git LFS enabled, then run `git lfs pull` to retrieve the full archive. It contains the canonical resumable `model.pt`, Android `model.onnx`, matching desktop/mobile tokenizers, the full combined training corpus, a manifest, real inference-check results, and the immutable original `base_step_600_backup.pt`.

The binary archive is intentionally a single Git LFS object rather than many visible checkpoint files and folders. Do not use historical Release assets or old Actions artifacts as the current model.

## Continue training the same model

The only active model-training workflow is `.github/workflows/model-inference-test.yml`. It reads `NYXVAULT_MODEL.zip` from `main`, resumes from its `model.pt` **including optimizer state**, adds validated JSONL data from `training/data/`, runs another SFT stage, exports and checks Android ONNX inference, then replaces the same archive in `main`.

Add new training records under `training/data/*.jsonl`, or manually dispatch the canonical workflow. The file `training/RUSSIAN_SFT_START.txt` is the manual push trigger. There are no active separate chat, Python, Godot or Russian SFT workflows.

## Programming dataset format and checks

New programming examples use the structured JSONL schema in `training/data_schema.py`. Each record includes an ID, curriculum level, language, topic, task type, instruction, expected solution, and license. Optional Python tests can be stored in the `tests` field. The eight-level curriculum is defined in `training/curriculum.py`. The additional `training/data/code_projects_seed.jsonl` contains original examples for complete code tasks, Godot 4 game mechanics, a browser game and project structure.

The data preparation script supports both this new schema and the existing legacy `question` / `answer` records. It preserves source-code indentation and URLs while normalizing line endings:

```bash
python training/prepare_data.py --input training/data/raw --knowledge data/knowledge.jsonl --out training/data/processed
```

Validate the programming corpus and run its automated checks:

```bash
python training/validate_data.py training/data/programming_seed.jsonl
python training/validate_data.py training/data/code_projects_seed.jsonl --exec-tests
python training/validate_data.py training/data/raw --exec-tests
python -m unittest discover -s training/tests -v
```

Only use `--exec-tests` with trusted training data. It starts a subprocess with a timeout and, on POSIX systems, CPU/address-space limits; it is not a complete security sandbox.

## Connect the trained model on Android

In NYXVAULT, tap **MODEL: OFF**. Import `nyxvault.onnx`, then import `tokenizer_mobile.json`. The indicator changes to **MODEL: ON** when both files load successfully. Both files remain in app-private storage; no model files or personal knowledge are uploaded.

## Data and privacy

- Knowledge, long-term facts and recent conversation history are stored locally on the device.
- WEB research is OFF by default and makes network requests only after the user explicitly enables it.
- Android speech recognition depends on the speech service installed on the device.
- The Transformer runs locally through ONNX Runtime after its files are imported.
- The Transformer is experimental. Verify generated code and important facts; lower validation loss does not guarantee factual accuracy.


Before committing to a long training run, the notebook now runs a short end-to-end CPU check on the prepared corpus:

```bash
python training/pipeline_smoke_test.py --data training/data/processed --tokenizer artifacts/tokenizer.json --steps 3 --block-size 32
```

This invokes the actual `training/train.py` script with a small model, checks finite train/validation loss, writes a checkpoint, and resumes from that checkpoint. It verifies training-pipeline mechanics only; it does not measure coding ability. The normal model defaults are 256 dimensions, 8 attention heads, 6 layers and a 512-token context.

## Instruction fine-tuning (SFT)

`training/finetune_instructions.py` performs instruction fine-tuning with prompt-loss masking, validation splits and resumable optimizer state. The current model's training history, latest checkpoint, tokenizer, Android export and combined training corpus are kept together in the root-level `NYXVAULT_MODEL.zip`.

Historical notebooks remain as research references only. They are not the active model-update path and must not publish independent checkpoints or releases.

SFT source files:
- `training/data_schema.py` — dataset validation and canonical prompt formatting.
- `training/finetune_instructions.py` — resumable instruction fine-tuning.
- `training/export_mobile.py` — matching ONNX and Android tokenizer export.
- `.github/workflows/model-inference-test.yml` — only active continuous-training workflow.

## Complete repository map

All active NYXVAULT source code, app code, training code, seed datasets, notebooks, tests, schemas, evaluation scripts, and export tools live in this **one repository** on `main`.

- `app/` — the Android application and its local intelligence services.
- `data/` — the app's knowledge corpus.
- `training/` — the base Transformer, tokenizer, base-training notebook, 5,000-record instruction-data generator, SFT trainer, inference script, validation tools, tests, and mobile export.
- `training/data/` — the canonical programming seed datasets and structured project examples.
- `artifacts/` — artifact policy and instructions.
- `legacy/` — source files preserved from earlier prototype branches. They are archived and are not part of the active build.
- `MODEL_CARD.md` — model architecture, data, observed validation results and limitations.

## Canonical model workflow

1. Retrieve the single current bundle from `main/NYXVAULT_MODEL.zip` using Git LFS.
2. Add validated JSONL examples to `training/data/`, or manually start the canonical workflow.
3. The workflow resumes from the previous bundle's `model.pt`; it does not restart from the base model or an old release checkpoint.
4. It rebuilds the same archive with the updated checkpoint, Android export, combined corpus, manifest, inference checks and preserved base backup.
5. The updated archive replaces the old one at the same path in `main`.

Old Release assets and Actions artifacts are migration history only. They are not used as the current model source.

## Archived branches

Earlier Python/NumPy prototypes from `neural-integration` and `training/programming-focus` are preserved under `legacy/` for history. Their old workflows are deliberately stored outside the root `.github/workflows/` path and cannot run as active Actions workflows. The earlier `dev` / `feature/unified-nyxvault-brain` programming seed file is an exact subset of the current 103-record seed file, so it is not duplicated.

