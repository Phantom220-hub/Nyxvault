# NYXVAULT

**Created by:** Nyxvault Studio  
**Owner:** Акрамзод Анас

NYXVAULT is an experimental local AI project built without external chat models, hosted inference APIs, or pretrained model weights.

## Two separate modes

- **Chat Mode:** read-only conversation using the locally trained knowledge index. It does not add examples or change the model.
- **Teacher Mode:** requires the owner-set password. It accepts approved question/answer examples and rebuilds both local models when `/train` is run.

## What "training" means in this version

The project now contains two learning components: a transparent TF-IDF retrieval index and a small character-level neural network (MLP) trained from random weights using NumPy and the project's own corpus. This is an educational mini-model, not a modern large language model; generated text may be inaccurate or incoherent. It does not use external models, APIs, or pretrained weights.

The starter corpus emphasizes Python, C++, Godot, GDScript, GitHub, Android development, and the project's anime-style 3D RPG interests.

## Run

Python 3.10+ (NumPy is required for neural training):

```bash
python -m pip install -r requirements.txt
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

Then run `/train` to rebuild the retrieval index and retrain the small neural network. Chat Mode never trains or writes to the corpus. If the retrieval index has no suitable answer and a neural model exists, it can show a clearly labelled experimental generated response.

## Privacy

Do not commit the Teacher Mode password, private conversations, personal data, or secret keys. Keep the repository private if you plan to add personal training material. The owner identity above is included as requested.
