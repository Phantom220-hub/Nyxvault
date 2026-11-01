# NYXVAULT

**Created by:** Nyxvault Studio  
**Owner:** Акрамзод Анас

NYXVAULT is an experimental local AI project built without external chat models, hosted inference APIs, or pretrained model weights.

## Two separate modes

- **Chat Mode:** read-only conversation using the locally trained knowledge index. It does not add examples or change the model.
- **Teacher Mode:** requires the owner-set password. It accepts approved question/answer examples and rebuilds both local models when `/train` is run.

## What "training" means in this version

The project now contains two learning components: a transparent TF-IDF retrieval index and a character-level vanilla RNN trained from random weights using NumPy and the project's own corpus. This is an educational mini-model with a bounded training workload for modest devices, not a modern large language model; generated text may be inaccurate or incoherent. It does not use external models, APIs, or pretrained weights.

The local corpus now includes programming fundamentals, Python, C++, algorithms, Git/GitHub, testing, Godot 3D, Android optimization, game graphics, and neural-network basics. Chat Mode keeps up to four recent exchanges in temporary session context; `/clear` removes that context.

## Run

Python 3.10+ (NumPy is required for neural training):

```bash
python -m pip install -r requirements.txt
python train.py
```

### Browser interface (Android / desktop)\n\nStart the local web app:\n\n```bash\npython web_app.py\n```\n\nOpen `http://127.0.0.1:8765` in the same device's browser. The server binds only to localhost and keeps chat history in temporary session memory. To enable Teacher Mode, set the password environment variable before starting the server.\n\nSet a private password locally, then start:

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
