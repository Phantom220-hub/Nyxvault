# Changelog

## 0.7.0 — 2026-11-02
- Teacher Mode password now uses PBKDF2-HMAC-SHA256 with a unique random salt.
- Existing plaintext Teacher Mode password is migrated on successful login and removed from preferences.
- Web research is disabled on fresh installs and requires an explicit WEB toggle.
- Added JUnit tests for normalization, retrieval ranking, and supported linear equations.
- Android CI runs Kotlin unit tests before building the APK and runs on main and dev.
- Python CI also runs on main and dev.
- Added DECISIONS.md and BLOCKERS.md.
- Rewrote README to separate Android functionality from the Python prototype and document current limitations.

### Not included in 0.7.0
- BM25 and typo-tolerant n-gram retrieval in Python.
- 50+ question measured quality benchmark.
- Complete Kotlin/Python unit-test coverage.
- Expanded math tools, 200+ puzzles, chess engine, full settings, local/cloud LLM.
- On-device performance and upgrade migration tests.
