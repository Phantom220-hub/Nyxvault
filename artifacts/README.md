# NYXVAULT canonical model

The only current model distribution and resumable checkpoint is the root-level `NYXVAULT_MODEL.zip` on `main`, tracked through Git LFS.

Download the repository with Git LFS enabled, then run `git lfs pull` if the archive is only an LFS pointer. Do not use old workflow artifacts or separate Release assets as the current model.

The single archive contains:
- `model.pt`: canonical PyTorch checkpoint, including optimizer state for continued SFT.
- `model.onnx`: Android inference model.
- `tokenizer.json` and `tokenizer_mobile.json`: matching tokenizer assets.
- `training_corpus.jsonl`: combined validated training corpus.
- `manifest.json`: checkpoint lineage, step count, hashes and data report.
- `inference_test.json`: actual generated responses from the packaged ONNX model.
- `base_step_600_backup.pt`: immutable original base checkpoint backup.

All future training must resume from `model.pt` inside the latest `main/NYXVAULT_MODEL.zip`, then replace that same archive in `main`. Do not create parallel checkpoints, model folders, workflow artifacts or new model releases. Historical releases are retained only as migration inputs/rollback history; they are not current model sources.
