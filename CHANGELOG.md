# Changelog

## 1.1.0 — One canonical model bundle
- Consolidated the shared SFT branches (chat/Python and Russian conversation) into one checkpoint, then jointly fine-tuned the merged model on a single validated corpus.
- Added one root-level `NYXVAULT_MODEL.zip` in `main`, tracked via Git LFS, containing the canonical resumable model, Android ONNX export, matching tokenizers, combined dataset, manifest, inference checks and immutable base checkpoint backup.
- Replaced parallel SFT workflows with one continuous workflow that always resumes from the previous canonical `model.pt` and updates the same bundle.
- Removed obsolete workflows and release-publishing path that created separate model outputs.
- Recorded that current generation remains incoherent in semantic checks; technical export success is not equivalent to useful answers.


## 1.0.0 — Unified NYXVAULT
- Replaced separate Android/Python/RNN identities with one Android product and one NyxvaultBrain coordinator.
- Unified math solving, evidence-based knowledge retrieval, Academy lessons, local memory and optional Transformer generation behind the same assistant.
- Added Android ONNX Runtime and ByteLevel BPE inference; trained model files can be imported from the MODEL menu.
- Synchronized the complete knowledge dataset into Android assets.
- Retired the separate Python terminal/browser chat and the legacy character-level RNN and its generated artifacts.
- Kept one Transformer training/evaluation/export pipeline under `training/`.
- Added tests for the unified brain and CI checks for ONNX + tokenizer export.

## 0.7.0
- Offline-first Android assistant with local question/answer knowledge, Teacher Mode, conversation memory, math tools and optional web research.
- Experimental Python retrieval and character-level RNN prototypes.
