# NYXVAULT CPU inference server

This service runs the exported ONNX model and tokenizer on the server CPU. The browser does not download either file.

## API

- GET /health — requires Authorization: Bearer <NYXVAULT_API_TOKEN>.
- POST /generate — requires the same bearer token and accepts JSON fields prompt, max_new_tokens, temperature, top_k, stop.
- Response fields: text, tokens, prompt_tokens.

The app reads only nyxvault.onnx and tokenizer.json from the public GitHub Release using HTTP range requests. It does not fetch the entire ZIP when the host supports byte ranges. Model files are cached on the server's local disk and the ONNX session stays in process memory.

## Temporary Colab CPU test

Open NYXVAULT_CPU_SERVER.ipynb in Colab. It installs CPU dependencies, fetches the model files on the remote Colab runtime (not on your phone), starts the API and creates a temporary Cloudflare Quick Tunnel. It prints a URL and a bearer token to copy into the website's server-side Secrets.

The Colab runtime and tunnel are temporary. They stop when the runtime disconnects or expires, and a new runtime downloads model files again. This is a test host, not an always-on production server.

Keep the API token private. Never paste it into chat, GitHub, notebook source, or browser JavaScript.

## Persistent host

Run with Python 3.11 or 3.12:

    pip install -r requirements.txt
    export NYXVAULT_API_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
    uvicorn app:app --host 0.0.0.0 --port 8000

Configure these server-side secrets on the website:
- NYXVAULT_BACKEND_URL: base URL of this API, without a trailing slash.
- NYXVAULT_BACKEND_TOKEN: the same value as NYXVAULT_API_TOKEN.

The Release archive must contain nyxvault-output/nyxvault.onnx and nyxvault-output/tokenizer.json. Successful inference does not mean the model's answers or generated code are reliable; NYXVAULT is still an experimental small model.
