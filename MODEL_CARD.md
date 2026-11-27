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

## Training stages

1. Base language-model training: `training/nyxvault_train.ipynb` and `training/train.py`.
2. Instruction dataset generation: `training/generate_instruction_dataset.py`.
3. Instruction fine-tuning: `training/nyxvault_sft.ipynb` and `training/finetune_instructions.py`.
4. Instruction inference: `training/generate_instruction.py`.
5. Mobile export: `training/export_mobile.py`.

## Limitations

- The model is small and experimental.
- A lower validation loss does not establish that generated code is correct.
- Instruction following must be evaluated separately from next-token loss.
- Generated code must be run against tests and reviewed before use.
- SFT template variations can contain correlated examples and do not replace a broad, carefully curated coding corpus.

## Model artifacts

Large binary checkpoints and exported models are stored as GitHub Release assets rather than ordinary Git blobs. Use `training/publish_release_assets.py` to publish a ZIP archive from Colab. Do not commit access tokens, personal conversations, or private local notes.
