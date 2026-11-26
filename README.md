# NYXVAULT 1.0 — one AI, one Android app

**Project:** NYXVAULT — Nyxvault Studio  
**Owner:** Акрамзод Анас

NYXVAULT is a single offline-first Android assistant. Its one brain combines verified local knowledge retrieval, evidence-based answers, math tools, learning lessons, local memory, optional web research, and an optional locally-run Transformer language model.

## One product, one brain

- **Android app:** the only user-facing NYXVAULT application.
- **Knowledge and reasoning:** one local knowledge file, ranked retrieval, evidence reasoning and math tools.
- **Transformer:** the trainable generative model. After a trained ONNX model and `tokenizer_mobile.json` are imported, the same Android brain calls it when retrieval and deterministic tools cannot answer.
- **Training tools:** Python scripts under `training/` prepare data, train the one Transformer, evaluate it and export Android files. They are build tools, not another chatbot.
- **Memory and Academy:** conversation memory, long-term notes and lessons are services used by the same brain.

The old standalone Python chat, browser chat and character-level RNN are retired. Their duplicate interfaces and legacy model artifacts are no longer part of the product.

## Build Android APK

Requires JDK 17 and Gradle 8.9:

```bash
gradle --no-daemon :app:testDebugUnitTest
gradle --no-daemon :app:assembleDebug
```

GitHub Actions runs Kotlin tests and builds the debug APK.

## Train the one Transformer

Open the [NYXVAULT training notebook](https://colab.research.google.com/github/Phantom220-hub/Nyxvault/blob/dev/training/nyxvault_train.ipynb) in Colab or Kaggle with a GPU. The notebook prepares the programming-first corpus, trains the tokenizer and Transformer, evaluates it, and exports:
- `nyxvault.onnx`
- `tokenizer_mobile.json`

The model is not bundled into the APK because trained weights are produced by the GPU training run. The app remains useful without it through local knowledge, math and reasoning tools.

## Connect the trained model on Android

In NYXVAULT, tap **MODEL: OFF**. Import `nyxvault.onnx`, then import `tokenizer_mobile.json`. The indicator changes to **MODEL: ON** when both files load successfully. Both files remain in app-private storage; no model files or personal knowledge are uploaded.

## Data and privacy

- Knowledge, long-term facts and recent conversation history are stored locally on the device.
- WEB research is OFF by default and makes network requests only after the user explicitly enables it.
- Android speech recognition depends on the speech service installed on the device.
- The Transformer runs locally through ONNX Runtime after its files are imported.
- The Transformer is experimental. Verify generated code and important facts; lower validation loss does not guarantee factual accuracy.
