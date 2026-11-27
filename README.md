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

## Train the one Transformer

Open the [NYXVAULT training notebook](https://colab.research.google.com/github/Phantom220-hub/Nyxvault/blob/main/training/nyxvault_train.ipynb) in Colab or Kaggle with a GPU. The notebook uses the main branch, prepares the corpus, trains the tokenizer, runs a short CPU integration check against the prepared text, and then trains the full Transformer, evaluates it, and exports:
- `nyxvault.onnx`
- `tokenizer_mobile.json`

The model is not bundled into the APK because trained weights are produced by the GPU training run. The app remains useful without it through local knowledge, math and reasoning tools.


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

After the base model has a saved full checkpoint such as `/content/nyxvault-output/checkpoints/step_600.pt`, open the [NYXVAULT SFT Colab notebook](https://colab.research.google.com/github/Phantom220-hub/Nyxvault/blob/main/training/nyxvault_sft.ipynb). It downloads the current `main` source, generates a deterministic 5,000-record starter instruction corpus (1,000 each for Python, JavaScript/TypeScript, algorithms, debugging and Godot/GDScript), validates its schema and Python examples, then runs a separate 100-step SFT stage.

SFT source files:
- `training/generate_instruction_dataset.py` — deterministic dataset generator.
- `training/finetune_instructions.py` — instruction fine-tuning with prompt-loss masking, stratified validation and resumable `sft_step_N.pt` checkpoints.
- `training/generate_instruction.py` — instruction-formatted inference.

The 5,000 records are template-authored parameter variations, not 5,000 separately human-reviewed tasks. The corpus is generated in Colab and is not duplicated as a large generated JSONL file in GitHub. The base checkpoint `step_600.pt` is read-only during SFT; SFT results are saved under `/content/nyxvault-output/sft_checkpoints`. Save `nyxvault-output` to Google Drive after every stage because Colab's `/content` storage is temporary.

