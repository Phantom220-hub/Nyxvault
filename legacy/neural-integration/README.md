# Archived neural-integration prototype

This directory preserves the earlier Python prototype from the `neural-integration` branch in the same NYXVAULT repository.

It is archived for historical reference and is **not part of the active training or Android build pipeline**. The prototype combined:
- a small local TF-IDF retrieval index;
- a NumPy character-level MLP trained from random initialization;
- an interactive command-line app with Chat Mode and Teacher Mode.

The active project is the Android-first NYXVAULT implementation at the repository root. Its current Transformer training pipeline is under `training/`, with the base Colab notebook and instruction-SFT notebook. Do not run archived `.github/workflows/train.yml` as an active workflow; it is stored here only as history.

Personal sample knowledge records from that old branch were intentionally not copied into this archive. The current curated datasets in `training/data/` are the active source of training examples.
