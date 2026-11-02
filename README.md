# NYXVAULT 0.7.0

**Project:** NYXVAULT — Nyxvault Studio  
**Owner:** Акрамзод Анас

NYXVAULT is an experimental offline-first Android assistant and a separate Python learning prototype. It is a retrieval-based system, **not** a general-purpose large language model and not comparable to Claude or ChatGPT.

## Android app

- Native Kotlin Android application.
- Local question/answer knowledge base, copied to app-private storage on first launch.
- Teacher Mode for adding local knowledge.
- Teacher password is stored as a PBKDF2-HMAC-SHA256 hash with a random salt. Existing plaintext passwords are migrated after a successful login.
- Local long-term facts and a rolling conversation history.
- Basic retrieval, limited linear-equation and arithmetic tools.
- Speech recognition and Android Text-to-Speech, depending on installed device services.
- Web research is **OFF by default**. Tap the WEB toggle to enable it explicitly; turn it off again to return to offline-only chat.

## Python prototype

The root Python scripts are a separate prototype. They use a local TF-IDF retrieval index and an experimental character-level RNN trained with NumPy. The RNN may generate incoherent text and is not embedded in the Android application.

## Build Android APK

Requires JDK 17 and Gradle 8.9:

```bash
gradle --no-daemon :app:testDebugUnitTest
gradle --no-daemon :app:assembleDebug
```

The GitHub Actions workflow runs Kotlin unit tests before assembling the debug APK. The artifact is named `NYXVAULT-Android-APK`.

## Run Python tests and training

Python 3.12 and NumPy:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python train.py
```

## Privacy and limitations

- Android knowledge, facts, and conversation history are stored in the app's private local storage.
- Web research makes network requests only after the user explicitly enables WEB.
- Android voice recognition may use an external system speech service; offline availability depends on the device.
- No cloud LLM, local LLM, Stockfish engine, 200-puzzle bank, general equation solver, or semantic memory system is included.
- No device-level ANR, memory, microphone, or TTS verification is claimed. See [BLOCKERS.md](BLOCKERS.md) for outstanding work.

## Add knowledge

Open Teacher Mode in the Android app, set a password, then add a question and a verified answer. The Android knowledge base remains local to that installation. Python training data is stored separately in `data/knowledge.jsonl`.
