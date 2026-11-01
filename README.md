# NYXVAULT

**Created by:** Nyxvault Studio  
**Owner:** Акрамзод Анас

NYXVAULT is an experimental local AI project built without external chat models, hosted inference APIs, or pretrained model weights.

## Two separate modes

- **Chat Mode:** read-only conversation using the locally trained knowledge index. It does not add examples or change the model.
- **Teacher Mode:** requires the owner-set password. It accepts approved question/answer examples and rebuilds the local model when `/train` is run.

## What "training" means in this version

The first learning engine is a transparent TF-IDF retrieval model trained from scratch on the project's own examples. It learns a vocabulary and weighted question vectors from `data/knowledge.jsonl`. It is **not yet a generative neural language model**, so it cannot freely reason or speak like a large assistant. That requires a much larger corpus and a separate neural-network training stage.

The starter corpus emphasizes Python, C++, Godot, GDScript, GitHub, Android development, and the project's anime-style 3D RPG interests.

## Run

Python 3.10+:

```bash
python train.py
```

Set a private password locally, then start:

Linux/macOS/Termux:
```bash
export NYXVAULT_TEACHER_PASSWORD='your-long-private-password'
python app.py
```

Windows PowerShell:
```powershell
$env:NYXVAULT_TEACHER_PASSWORD='your-long-private-password'
python app.py
```

In Teacher Mode, add examples as:

```
How do I make a player move in Godot? => Use CharacterBody3D...
```

Then run `/train` to rebuild the index. Chat Mode never trains or writes to the corpus.

## Privacy

Do not commit the Teacher Mode password, private conversations, personal data, or secret keys. Keep the repository private if you plan to add personal training material. The owner identity above is included as requested.
