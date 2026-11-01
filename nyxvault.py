"""Transparent local knowledge engine. No pretrained model, API, or network access."""
import json
import re
from pathlib import Path


def normalize(text: str) -> list[str]:
    return re.findall(r"[\w']+", text.lower(), flags=re.UNICODE)


class KnowledgeBase:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def examples(self) -> list[dict[str, str]]:
        if not self.path.exists():
            return []
        rows = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                    if isinstance(row.get("question"), str) and isinstance(row.get("answer"), str):
                        rows.append(row)
                except (json.JSONDecodeError, AttributeError):
                    continue
        return rows

    def add(self, question: str, answer: str) -> None:
        row = {"question": question.strip(), "answer": answer.strip()}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def answer(self, question: str) -> str:
        query = set(normalize(question))
        if not query:
            return "Пока я не знаю, как ответить. Научи меня в режиме учителя."
        best_score = 0.0
        best_answer = None
        for row in self.examples():
            candidate = set(normalize(row["question"]))
            if not candidate:
                continue
            score = len(query & candidate) / max(1, len(query | candidate))
            if score > best_score:
                best_score, best_answer = score, row["answer"]
        if best_answer is not None and best_score >= 0.20:
            return best_answer
        return "Я пока не знаю ответа. Перейди в режим учителя и добавь пример."
