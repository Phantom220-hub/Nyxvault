"""NYXVAULT's first locally trained retrieval brain.

No remote model, pretrained weights, external API, or network access.
The model learns a TF-IDF index from approved local examples.
"""
import json
import math
import re
from pathlib import Path

DATA = Path("data/knowledge.jsonl")
INDEX = Path("data/model_index.json")


def tokenize(text: str) -> list[str]:
    text = text.lower().replace("ё", "е")
    return re.findall(r"[a-zа-я0-9_+#.]+", text, flags=re.IGNORECASE)


def load_examples(path: Path = DATA) -> list[dict]:
    if not path.exists():
        return []
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            if row.get("question") and row.get("answer"):
                result.append({"question": str(row["question"]), "answer": str(row["answer"])})
        except (ValueError, TypeError, AttributeError):
            continue
    return result


def train(data_path: Path = DATA, index_path: Path = INDEX) -> dict:
    """Train a transparent TF-IDF retrieval model from scratch on local examples."""
    rows = load_examples(data_path)
    docs = [set(tokenize(row["question"])) for row in rows]
    n = max(1, len(docs))
    vocab = sorted(set().union(*docs)) if docs else []
    idf = {word: math.log(1 + n / (1 + sum(word in doc for doc in docs))) for word in vocab}
    vectors = []
    for row, doc in zip(rows, docs):
        weights = {word: idf[word] for word in doc}
        norm = math.sqrt(sum(value * value for value in weights.values())) or 1.0
        vectors.append({
            "question": row["question"],
            "answer": row["answer"],
            "weights": {word: value / norm for word, value in weights.items()}
        })
    index_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format": 1, "examples": len(rows), "vocabulary": vocab, "vectors": vectors}
    index_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return {"examples": len(rows), "vocabulary": len(vocab), "index": str(index_path)}


class Nyxvault:
    def __init__(self, data_path: Path = DATA, index_path: Path = INDEX):
        self.data_path, self.index_path = Path(data_path), Path(index_path)
        if not self.index_path.exists():
            train(self.data_path, self.index_path)
        self.index = json.loads(self.index_path.read_text(encoding="utf-8"))

    def teach(self, question: str, answer: str) -> None:
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        row = {"question": question.strip(), "answer": answer.strip()}
        with self.data_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    def answer(self, question: str) -> str:
        q = set(tokenize(question))
        if not q:
            return "Напиши вопрос — я постараюсь найти подходящий ответ."
        vocab = set(self.index.get("vocabulary", []))
        q &= vocab
        if not q:
            return "Я пока не знаю этого достаточно хорошо. В режиме обучения добавь вопрос и правильный ответ."
        idf = {}
        vectors = self.index.get("vectors", [])
        # Query weights use the same inverse-document-frequency values reconstructed from the trained index.
        n = max(1, len(vectors))
        for word in q:
            df = sum(word in row["weights"] for row in vectors)
            idf[word] = math.log(1 + n / (1 + df))
        norm = math.sqrt(sum(v * v for v in idf.values())) or 1.0
        query = {w: v / norm for w, v in idf.items()}
        best, best_score = None, 0.0
        for row in vectors:
            score = sum(weight * row["weights"].get(word, 0.0) for word, weight in query.items())
            if score > best_score:
                best, best_score = row, score
        if best and best_score >= 0.16:
            return best["answer"]
        return "Я пока не знаю точного ответа. Обучи меня этому в Teacher Mode — я сохраню пример после переобучения."
