"""A small character-level vanilla RNN trained from random weights with NumPy.

This is an educational recurrent network, not a modern large language model.
It uses only the project's local JSONL corpus and does not download model weights.
"""
import json
from pathlib import Path

import numpy as np

DATA = Path("data/knowledge.jsonl")
MODEL = Path("data/neural_model.npz")
SEQ_LEN = 32
HIDDEN = 64
EPOCHS = 30
LEARNING_RATE = 0.08
MAX_CORPUS_CHARS = 50000


def load_corpus(data_path=DATA):
    path = Path(data_path)
    if not path.exists():
        return ""
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            question, answer = row.get("question"), row.get("answer")
            if question and answer:
                samples.append(f"Вопрос: {question.strip()}\nОтвет: {answer.strip()}\n\n")
        except (ValueError, TypeError, AttributeError):
            continue
    # A bounded corpus keeps training predictable on phones and free CI runners.
    return "".join(samples)[:MAX_CORPUS_CHARS]


def _softmax(logits):
    shifted = logits - np.max(logits, axis=0, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=0, keepdims=True)


def train_neural(data_path=DATA, model_path=MODEL, epochs=EPOCHS):
    corpus = load_corpus(data_path)
    if len(corpus) < SEQ_LEN + 2:
        raise ValueError("Недостаточно обучающего текста: добавь больше примеров.")
    chars = sorted(set(corpus))
    to_id = {char: i for i, char in enumerate(chars)}
    encoded = np.array([to_id[c] for c in corpus], dtype=np.int32)
    vocab = len(chars)
    rng = np.random.default_rng(20261101)

    # Vanilla RNN: h[t] = tanh(Wxh*x[t] + Whh*h[t-1] + bh)
    scale = 1.0 / np.sqrt(HIDDEN)
    wxh = (rng.standard_normal((HIDDEN, vocab)) * scale).astype(np.float32)
    whh = (rng.standard_normal((HIDDEN, HIDDEN)) * scale).astype(np.float32)
    why = (rng.standard_normal((vocab, HIDDEN)) * scale).astype(np.float32)
    bh = np.zeros((HIDDEN, 1), dtype=np.float32)
    by = np.zeros((vocab, 1), dtype=np.float32)

    # Deterministic, bounded sequence training.
    starts = np.arange(0, len(encoded) - SEQ_LEN - 1, max(1, SEQ_LEN // 2))
    final_loss = 0.0
    completed_epochs = max(1, int(epochs))
    for epoch in range(completed_epochs):
        order = rng.permutation(starts)
        epoch_loss = 0.0
        batches = 0
        for start in order:
            h_prev = np.zeros((HIDDEN, 1), dtype=np.float32)
            xs, hs, ps, targets = {}, {-1: h_prev}, {}, {}
            loss = 0.0
            for t in range(SEQ_LEN):
                x_id = int(encoded[start + t])
                y_id = int(encoded[start + t + 1])
                x = np.zeros((vocab, 1), dtype=np.float32)
                x[x_id, 0] = 1.0
                h = np.tanh(wxh @ x + whh @ hs[t - 1] + bh)
                p = _softmax(why @ h + by)
                loss -= float(np.log(p[y_id, 0] + 1e-9))
                xs[t], hs[t], ps[t], targets[t] = x, h, p, y_id

            d_wxh = np.zeros_like(wxh)
            d_whh = np.zeros_like(whh)
            d_why = np.zeros_like(why)
            d_bh = np.zeros_like(bh)
            d_by = np.zeros_like(by)
            dh_next = np.zeros((HIDDEN, 1), dtype=np.float32)
            for t in reversed(range(SEQ_LEN)):
                dy = ps[t].copy()
                dy[targets[t], 0] -= 1.0
                d_why += dy @ hs[t].T
                d_by += dy
                dh = why.T @ dy + dh_next
                dh_raw = (1.0 - hs[t] * hs[t]) * dh
                d_bh += dh_raw
                d_wxh += dh_raw @ xs[t].T
                d_whh += dh_raw @ hs[t - 1].T
                dh_next = whh.T @ dh_raw

            for grad in (d_wxh, d_whh, d_why, d_bh, d_by):
                np.clip(grad, -5.0, 5.0, out=grad)
            wxh -= LEARNING_RATE * d_wxh
            whh -= LEARNING_RATE * d_whh
            why -= LEARNING_RATE * d_why
            bh -= LEARNING_RATE * d_bh
            by -= LEARNING_RATE * d_by
            epoch_loss += loss / SEQ_LEN
            batches += 1
        final_loss = epoch_loss / max(1, batches)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_path,
        chars=np.array(chars),
        wxh=wxh, whh=whh, why=why, bh=bh, by=by,
        seq_len=np.array([SEQ_LEN], dtype=np.int32),
        hidden=np.array([HIDDEN], dtype=np.int32),
    )
    return {
        "characters": len(corpus),
        "vocabulary": vocab,
        "epochs": completed_epochs,
        "loss": round(final_loss, 4),
        "model": str(model_path),
        "architecture": "vanilla_rnn",
    }


class NeuralBrain:
    def __init__(self, model_path=MODEL):
        self.data = np.load(model_path, allow_pickle=False)
        self.chars = self.data["chars"].tolist()
        self.lookup = {char: i for i, char in enumerate(self.chars)}
        self.hidden_size = int(self.data["hidden"][0])

    def generate(self, prompt="Ответ:", max_new=160, temperature=0.8, top_k=8):
        wxh, whh = self.data["wxh"], self.data["whh"]
        why, bh, by = self.data["why"], self.data["bh"], self.data["by"]
        vocab = len(self.chars)
        h = np.zeros((self.hidden_size, 1), dtype=np.float32)
        output = []
        prompt_ids = [self.lookup.get(char, None) for char in prompt]
        # Prime the recurrent state with the entire prompt.
        for char_id in prompt_ids:
            if char_id is None:
                h = np.zeros_like(h)
                continue
            x = np.zeros((vocab, 1), dtype=np.float32)
            x[char_id, 0] = 1.0
            h = np.tanh(wxh @ x + whh @ h + bh)

        rng = np.random.default_rng()
        for _ in range(max(0, int(max_new))):
            logits = (why @ h + by).ravel() / max(0.15, float(temperature))
            logits -= logits.max()
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum()
            if 0 < top_k < vocab:
                keep = np.argpartition(probabilities, -top_k)[-top_k:]
                filtered = np.zeros_like(probabilities)
                filtered[keep] = probabilities[keep]
                probabilities = filtered / filtered.sum()
            next_id = int(rng.choice(vocab, p=probabilities))
            output.append(self.chars[next_id])
            x = np.zeros((vocab, 1), dtype=np.float32)
            x[next_id, 0] = 1.0
            h = np.tanh(wxh @ x + whh @ h + bh)
            if self.chars[next_id] == "\n":
                break
        return "".join(output).strip()
