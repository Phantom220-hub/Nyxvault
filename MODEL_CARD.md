# NYXVAULT model card

## Project purpose

NYXVAULT is an experimental programming-focused language model and Android application. Its Transformer is initialized from random weights; no third-party pretrained model weights or hosted inference API are used in the training pipeline.

## Active base Transformer

The current training configuration in `training/train.py` is:

- Architecture: decoder-only GPT-style Transformer
- Trainable parameters: 6,918,144 with the current 8,000-token vocabulary
- Width: 256
- Attention heads: 8
- Transformer blocks: 6
- Context length: 512 tokens
- Tokenizer: ByteLevel BPE, vocabulary size 8,000
- Training target: next-token prediction
- Base training target: 5,000 optimizer update steps
- Checkpoint format: PyTorch `.pt`

The architecture and parameter count describe the current configured model, not a claim that it matches the capability of a large pretrained coding model.

## Observed base-training checkpoint

The project owner reported these checkpoints in a Google Colab Tesla T4 run:

- `step_500.pt`: resumable checkpoint with optimizer and RNG state
- `step_600.pt`: resumable checkpoint with optimizer and RNG state
- The same held-out validation file was measured at loss 9.21411 for step 500 and 5.87435 for step 600 (reported relative decrease 36.246%).

These numbers are user-reported Colab results, not independently reproduced by this repository CI. Validation loss measures next-token prediction on that dataset; it does not prove factual accuracy or coding competence.

## Training data

The active repository includes:
- `training/data/programming_seed.jsonl`: 103 starter programming records.
- `training/data/code_projects_seed.jsonl`: 18 original structured coding/project records.
- `data/knowledge.jsonl`: local app knowledge corpus.
- `training/generate_instruction_dataset.py`: creates 5,000 instruction examples, distributed equally across Python, JavaScript/TypeScript, algorithms, debugging, and Godot/GDScript.
- `training/finetune_instructions.py`: supervised instruction fine-tuning (SFT) with prompt-token loss masking and resumable SFT checkpoints.

The 5,000 SFT records are deterministic parameter variations from authored templates. They are not 5,000 separately written and human-reviewed tasks. The current generator's Python examples have automated assertions; JavaScript, TypeScript, and GDScript examples require runtime validation in their respective toolchains before being treated as fully verified.

Wikipedia collection is optional and subject to source rate limits. A 429 response means the source is rate-limiting requests; skipped pages must not be represented as collected training data. Review `DATA_SOURCES.md` and the generated attribution manifest for source-specific terms.

## Consolidated model lineage

The current canonical bundle is `NYXVAULT_MODEL.zip` at the root of `main`, tracked through Git LFS. It contains one resumable `model.pt`, the matching tokenizer, Android ONNX export, combined training corpus, manifest, inference-check output and the original `base_step_600_backup.pt`.

The consolidation used three compatible SFT checkpoints that shared `sft_step_50.pt` as their parent:
- Chat/Python basics: `sft_step_100.pt`.
- Expanded chat and Python projects: `sft_step_350.pt`.
- Russian conversation: `sft_step_300.pt`.

Their task-vector deltas from the common parent were averaged, then jointly fine-tuned for 100 additional optimizer updates on a validated, de-duplicated corpus of 2,961 records. The final checkpoint is the result of that joint stage; it is not a literal sequential addition of every sibling's step counter.

The initial consolidated checkpoint has an internal joint-stage step of 100. The base step-600 checkpoint is preserved separately inside the same archive as a read-only backup.

## Active training process

`.github/workflows/model-inference-test.yml` is the only active model-training workflow. It resumes directly from the previous bundle's `model.pt` and optimizer state, extends the canonical training corpus with validated JSONL records from `training/data/`, performs a new SFT stage, exports ONNX and updates the same LFS archive in `main`. Historical Release assets and workflow artifacts are migration inputs only.

## Limitations

- The model is small and experimental.
- A lower validation loss does not establish that generated code is correct.
- Instruction following must be evaluated separately from next-token loss.
- Generated code must be run against tests and reviewed before use.
- SFT template variations can contain correlated examples and do not replace a broad, carefully curated coding corpus.

## Model artifacts

The single current model package is `main/NYXVAULT_MODEL.zip` (Git LFS). Retrieve the full archive with `git lfs pull`. It includes `model.pt`, `model.onnx`, matching `tokenizer.json` and `tokenizer_mobile.json`, `training_corpus.jsonl`, `manifest.json`, `inference_test.json`, and `base_step_600_backup.pt`.

## Observed inference quality

The first consolidated ONNX inference checks produced non-empty output for greeting, tokenizer, calculator and Godot prompts, but the generated text was still incoherent and did not pass a semantic-quality check. These were technical export/generation checks, not proof of useful conversation or programming ability. The model remains experimental and needs substantially more high-quality training and evaluation before it can be considered reliable.
