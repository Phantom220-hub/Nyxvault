# NYXVAULT training system

All active training components are maintained in this directory and the parent repository. Run them from the repository root.

## Base language model

- `model.py`: decoder-only GPT-style Transformer.
- `train.py`: resumable next-token training, validation loss, best weights, and optimizer checkpoints.
- `train_tokenizer.py`: ByteLevel BPE tokenizer training.
- `prepare_data.py`: prepares train/validation text from the canonical JSONL and raw-data folders.
- `evaluate.py`: validation metrics and fixed prompt smoke checks.
- `export_mobile.py`: exports the trained model for the Android runtime.
- `nyxvault_train.ipynb`: Colab/Kaggle base-training workflow.

## Instruction tuning

- `generate_instruction_dataset.py`: reproducibly generates 5,000 instruction records across five programming categories.
- `finetune_instructions.py`: supervised fine-tuning with prompt-loss masking, a deterministic stratified validation split, and resumable SFT checkpoints.
- `generate_instruction.py`: generates answers using the instruction format.
- `nyxvault_sft.ipynb`: restores saved model artifacts from Google Drive, fetches current code from GitHub main, generates/validates the dataset, runs SFT, packages artifacts, and can publish the archive as a GitHub Release asset.
- `publish_release_assets.py`: securely uploads an archive to a GitHub Release using `GITHUB_TOKEN` from the environment.

## Data

The canonical source datasets are:
- `data/programming_seed.jsonl`
- `data/code_projects_seed.jsonl`

The instruction SFT corpus is generated from its source generator on Colab. Its generated copy, dataset report, tokenizer and model checkpoints are packed into the final artifact ZIP; the ZIP can be attached to a Release in this same repository.

## Run checks

```bash
python training/validate_data.py training/data/programming_seed.jsonl
python training/validate_data.py training/data/code_projects_seed.jsonl --exec-tests
python -m unittest discover -s training/tests -v
```

Only run `--exec-tests` on trusted dataset code. The validator uses time and resource limits but is not a complete sandbox.

## Storage policy

Do not commit `.pt`, `.onnx`, `.npz`, local processed corpora, Colab output folders, or access tokens into ordinary Git commits. Keep the complete output folder on Google Drive during training and publish the final ZIP through GitHub Releases when ready.
