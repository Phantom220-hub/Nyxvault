# NYXVAULT — from-scratch learning assistant

A small, transparent experimental AI project. It does **not** call an external model or API and does not download pretrained weights.

## Two modes

1. **Chat mode** — answers from the knowledge/examples already approved in the local data file. It never writes to training data and never changes weights.
2. **Teacher mode** — password-protected. The owner can add question/answer examples and run training. The password is configured locally with the `NYXVAULT_TEACHER_PASSWORD` environment variable; never commit it.

## Honest scope

This is a learning project, not a ChatGPT-level model. The first version uses a small model trained from random initialization and a transparent local knowledge store. It will only know what its training data teaches it. A genuinely capable general language model requires large datasets and substantial compute; GitHub Actions is for tests/builds, not unlimited training.

## Run locally

```bash
python -m pip install -r requirements.txt
export NYXVAULT_TEACHER_PASSWORD='choose-a-long-private-password'
python app.py
```

On Android/Termux, set the environment variable in your local shell before running. Do not send the password in chat or store it in GitHub.

## Safety and privacy

- Keep this repository **private** if it will contain personal training examples.
- Never put passwords, API keys, private conversations, or personal data in Git.
- Training examples are stored locally in `data/knowledge.jsonl`.
- Chat mode is read-only. Teacher mode requires a password each time it is entered.
