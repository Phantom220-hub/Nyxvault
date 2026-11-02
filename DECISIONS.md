# Engineering decisions

## 2026-11-02 — autonomous 0.7.0 work
- Keep `main` unchanged until the `dev` branch passes automated tests and APK assembly.
- Store Teacher Mode credentials as PBKDF2WithHmacSHA256 hashes with a per-install random 128-bit salt and 120,000 iterations; compare hashes with a constant-time comparison.
- Migrate an existing plaintext Teacher Mode password only after a successful login, then remove the legacy preference.
- Disable web research on fresh installs. The user must explicitly tap the WEB toggle to enable it; this avoids network access during ordinary chat.
- Do not add a cloud LLM or paid service. Keep the default app offline-first.
- Keep the experimental Python RNN clearly labelled as an educational prototype, not a conversational LLM.
- Use JUnit for pure Kotlin logic tests and Python unittest for the Python retrieval/training layer.
