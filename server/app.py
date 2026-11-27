from __future__ import annotations

import hmac
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from remotezip import RemoteZip
from tokenizers import Tokenizer

ARCHIVE_URL = os.getenv(
    "NYXVAULT_ARCHIVE_URL",
    "https://github.com/Phantom220-hub/Nyxvault/releases/download/nyxvault-weights-600/NYXVAULT_ALL_FILES.zip",
)
MODEL_DIR = Path(os.getenv("NYXVAULT_MODEL_DIR", "/tmp/nyxvault-model"))
MODEL_PATH = MODEL_DIR / "nyxvault.onnx"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.json"
API_TOKEN = os.getenv("NYXVAULT_API_TOKEN", "")
BLOCK_SIZE = 512
MAX_PROMPT_CHARS = 12_000
MAX_NEW_TOKENS = 160
TOP_K_DEFAULT = 40

session: ort.InferenceSession | None = None
tokenizer: Tokenizer | None = None
inference_lock = threading.Lock()


def require_token(authorization: str | None = Header(default=None)) -> None:
    if not API_TOKEN:
        raise HTTPException(status_code=503, detail="NYXVAULT_API_TOKEN is not configured")
    expected = f"Bearer {API_TOKEN}"
    if authorization is None or not hmac.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")


def find_entry(names: list[str], filename: str) -> str:
    matches = [name for name in names if name == filename or name.endswith("/" + filename)]
    if not matches:
        raise RuntimeError(f"The release archive does not contain {filename}")
    return next((name for name in matches if name.endswith("nyxvault-output/" + filename)), matches[0])


def prepare_model_files() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.is_file() and TOKENIZER_PATH.is_file():
        return

    print("NYXVAULT: reading the public GitHub release with HTTP range requests", flush=True)
    try:
        with RemoteZip(ARCHIVE_URL) as archive:
            names = archive.namelist()
            onnx_name = find_entry(names, "nyxvault.onnx")
            tokenizer_name = find_entry(names, "tokenizer.json")
            for archive_name, destination in (
                (onnx_name, MODEL_PATH),
                (tokenizer_name, TOKENIZER_PATH),
            ):
                if destination.is_file():
                    continue
                print(f"NYXVAULT: extracting {archive_name}", flush=True)
                payload = archive.read(archive_name)
                temporary = destination.with_suffix(destination.suffix + ".part")
                temporary.write_bytes(payload)
                temporary.replace(destination)
    except Exception as exc:
        raise RuntimeError(
            "Could not read the model files from the GitHub Release using HTTP range requests. "
            "Check that the Release asset exists and supports byte-range requests."
        ) from exc

    if not MODEL_PATH.is_file() or not TOKENIZER_PATH.is_file():
        raise RuntimeError("Model setup did not produce both nyxvault.onnx and tokenizer.json")


def load_model() -> None:
    global session, tokenizer
    if session is not None and tokenizer is not None:
        return

    prepare_model_files()
    print("NYXVAULT: loading tokenizer", flush=True)
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    print("NYXVAULT: loading ONNX Runtime CPU session", flush=True)

    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, min(2, os.cpu_count() or 1))
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    loaded = ort.InferenceSession(
        str(MODEL_PATH),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    input_names = [item.name for item in loaded.get_inputs()]
    output_names = [item.name for item in loaded.get_outputs()]
    if "input_ids" not in input_names:
        raise RuntimeError(f"ONNX model is missing input_ids; inputs={input_names}")
    if "logits" not in output_names:
        raise RuntimeError(f"ONNX model is missing logits; outputs={output_names}")

    probe_ids = tok.encode("Привет").ids
    if not probe_ids:
        raise RuntimeError("Tokenizer produced no tokens for the startup test")
    logits = loaded.run(
        ["logits"],
        {"input_ids": np.asarray([probe_ids], dtype=np.int64)},
    )[0]
    if logits.ndim != 3 or logits.shape[1] != len(probe_ids):
        raise RuntimeError(f"Unexpected ONNX logits shape: {logits.shape}")

    session = loaded
    tokenizer = tok
    print(
        f"NYXVAULT ready: vocab={tok.get_vocab_size()}, "
        f"model={MODEL_PATH.stat().st_size / (1024 * 1024):.1f} MiB",
        flush=True,
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not API_TOKEN or len(API_TOKEN) < 24:
        raise RuntimeError("Set NYXVAULT_API_TOKEN to a random secret of at least 24 characters")
    load_model()
    yield


app = FastAPI(title="NYXVAULT inference API", version="1.0.0", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)
    max_new_tokens: int = Field(default=80, ge=1, le=MAX_NEW_TOKENS)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_k: int = Field(default=TOP_K_DEFAULT, ge=1, le=8000)
    stop: list[str] = Field(default_factory=list, max_length=8)


class GenerateResponse(BaseModel):
    text: str
    tokens: int
    prompt_tokens: int


def sample_next(logits: np.ndarray, temperature: float, top_k: int) -> int:
    values = np.asarray(logits, dtype=np.float64)
    if temperature <= 0:
        return int(np.argmax(values))
    k = max(1, min(int(top_k), values.size))
    if k < values.size:
        indices = np.argpartition(values, -k)[-k:]
        selected = values[indices] / temperature
        selected -= np.max(selected)
        probabilities = np.exp(selected)
        probabilities /= probabilities.sum()
        return int(np.random.choice(indices, p=probabilities))
    values = values / temperature
    values -= np.max(values)
    probabilities = np.exp(values)
    probabilities /= probabilities.sum()
    return int(np.random.choice(np.arange(values.size), p=probabilities))


def generate_text(request: GenerateRequest) -> GenerateResponse:
    if session is None or tokenizer is None:
        raise RuntimeError("Model is not loaded")
    ids = tokenizer.encode(request.prompt).ids
    if not ids:
        raise ValueError("Prompt tokenization returned no tokens")
    ids = ids[-BLOCK_SIZE:]
    prompt_tokens = len(ids)
    generated: list[int] = []
    eos_id = tokenizer.token_to_id("[EOS]")
    stop_strings = [value for value in request.stop if value]
    text = ""

    for _ in range(request.max_new_tokens):
        context = ids[-BLOCK_SIZE:]
        with inference_lock:
            logits = session.run(
                ["logits"],
                {"input_ids": np.asarray([context], dtype=np.int64)},
            )[0]
        if logits.ndim != 3 or logits.shape[0] != 1:
            raise RuntimeError(f"Unexpected logits output shape: {logits.shape}")
        next_id = sample_next(logits[0, -1, :], request.temperature, request.top_k)
        if eos_id is not None and next_id == eos_id:
            break
        ids.append(next_id)
        generated.append(next_id)
        text = tokenizer.decode(generated)
        stop_positions = [text.find(stop) for stop in stop_strings if stop in text]
        if stop_positions:
            text = text[: min(stop_positions)]
            break

    return GenerateResponse(
        text=text.strip(),
        tokens=len(generated),
        prompt_tokens=prompt_tokens,
    )


@app.get("/")
def root():
    return {"service": "NYXVAULT inference API", "health": "/health", "generate": "POST /generate"}


@app.get("/health", dependencies=[Depends(require_token)])
def health():
    return {
        "status": "ready" if session is not None and tokenizer is not None else "loading",
        "model": "NYXVAULT step 600 ONNX",
        "device": "CPU",
    }


@app.post("/generate", response_model=GenerateResponse, dependencies=[Depends(require_token)])
def generate(request: GenerateRequest):
    try:
        return generate_text(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"NYXVAULT generation failed: {exc}") from exc
