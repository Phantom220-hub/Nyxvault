# Blockers and unverified items

- Device-level testing on a Samsung Galaxy A16 is not available from GitHub Actions; ANR, actual memory use, microphone/TTS behavior, and UI flows remain unverified until tested on-device.
- Android password migration from legacy plaintext preferences is implemented but needs a real upgrade-path test on an existing installation.
- Python retrieval quality and the proposed 50+ Russian control-question benchmark are not yet measured.
- BM25, Russian Snowball stemming, character n-gram typo matching, full fact editing, expanded math tools, puzzle bank, chess engine, settings screens, and full README rewrite remain outstanding unless explicitly covered by later commits.
- No local or cloud LLM is included. A genuinely large context window and general-purpose reasoning are therefore unavailable.

## From-scratch LLM training
- Training scripts and Kaggle notebook have been added on dev. No training has been run by the assistant; no model metrics exist yet.
- Corpus currently includes project knowledge and Wikimedia API text only when the user launches the notebook. Tajik essays written by the user are not included until supplied/authorized.
- Android integration is not implemented; model export is experimental and must be tested after training.
