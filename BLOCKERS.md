# Current limitations and verification

- The Android app is the only user-facing NYXVAULT product. Retrieval, math, evidence reasoning, Academy, memory and optional Transformer inference are coordinated by NyxvaultBrain.
- Transformer training still requires a GPU session in Colab/Kaggle. No completed full-size Transformer checkpoint is claimed until a training archive with metrics exists.
- Android model import and ONNX inference must be tested on a physical device, including memory use, speed and output quality.
- Generated answers are experimental and must be checked; perplexity is not a factuality score.
- Device-level microphone, TTS, Android lifecycle and ANR checks require a real device.
- WEB research is optional and disabled by default.
