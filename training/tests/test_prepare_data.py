import json
import tempfile
import unittest
from pathlib import Path

from training.prepare_data import build_dataset

FENCE = chr(96) * 3


class PrepareDataCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.knowledge = self.root / "knowledge.jsonl"
        self.out = self.root / "out"

    def tearDown(self):
        self.tmp.cleanup()

    def write_jsonl(self, path, records):
        path.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
            encoding="utf-8",
        )

    def structured(self, record_id, name, topic="variables-and-types"):
        return {
            "id": record_id,
            "level": 1,
            "language": "python",
            "topic": topic,
            "type": "write_code",
            "instruction": f"Напиши функцию, которая возвращает приветствие для пользователя {name}.",
            "output": (
                "План: принять имя, сформировать строку и вернуть результат.\n\n"
                + FENCE + "python\n"
                "def greet(name: str) -> str:\n"
                "    return f'Привет, {name}!'\n"
                + FENCE
            ),
            "license": "original",
            "tests": ["assert greet('Анас') == 'Привет, Анас!'"],
            "source": "unit test",
        }

    def test_accepts_legacy_and_structured_records(self):
        records = [
            {"question": f"Старый вопрос номер {i}", "answer": f"Это подробный старый ответ номер {i} по программированию."}
            for i in range(1, 4)
        ]
        records.append(self.structured("structured-001", "Анас"))
        self.write_jsonl(self.raw / "mixed.jsonl", records)
        self.write_jsonl(
            self.knowledge,
            [{"question": "Вопрос из старой базы", "answer": "Подробный ответ из старой базы знаний."}],
        )
        manifest = build_dataset(
            self.raw, self.knowledge, self.out,
            val_ratio=0.20, min_chars=20, min_documents=2, seed=7,
        )
        train = (self.out / "train.txt").read_text(encoding="utf-8")
        validation = (self.out / "validation.txt").read_text(encoding="utf-8")
        combined = train + "\n" + validation
        for marker in (
            "Вопрос: Старый вопрос номер", "### Level: 1",
            "### Topic: variables-and-types", "### Task type: write_code",
            "### Language: python", "### Task", "### Solution",
            "### Verification tests",
        ):
            self.assertIn(marker, combined)
        self.assertEqual(manifest["formats_seen"], {"legacy": 4, "structured": 1})
        self.assertEqual(manifest["documents"], 5)
        self.assertEqual(manifest["train_documents"] + manifest["validation_documents"], 5)

    def test_rejects_invalid_structured_record(self):
        bad = self.structured("bad-003", "Неверный", topic="not-a-real-topic")
        self.write_jsonl(self.raw / "bad.jsonl", [bad])
        with self.assertRaisesRegex(ValueError, "Only 0 usable documents"):
            build_dataset(self.raw, self.knowledge, self.out, min_chars=10, min_documents=2)

    def test_deduplicates_legacy_records(self):
        same = {"question": "Один и тот же вопрос", "answer": "Один и тот же ответ"}
        other = {"question": "Другой вопрос", "answer": "Другой достаточно подробный ответ"}
        self.write_jsonl(self.raw / "duplicates.jsonl", [same, same, other])
        manifest = build_dataset(self.raw, self.knowledge, self.out, min_chars=10, min_documents=2)
        self.assertEqual(manifest["documents"], 2)
        self.assertEqual(manifest["duplicate_records"], 1)

    def test_preserves_python_indentation(self):
        first = self.structured("structured-indent", "Тест")
        first["output"] = (
            FENCE + "python\ndef outer():\n"
            "    def inner():\n        return 7\n"
            "    return inner()\n" + FENCE
        )
        second = self.structured("structured-indent-2", "Другой")
        self.write_jsonl(self.raw / "indent.jsonl", [first, second])
        build_dataset(self.raw, self.knowledge, self.out, min_chars=10, min_documents=2)
        combined = (
            (self.out / "train.txt").read_text(encoding="utf-8")
            + (self.out / "validation.txt").read_text(encoding="utf-8")
        )
        self.assertIn("    def inner():\n        return 7", combined)

    def test_preserves_urls_in_code_text(self):
        (self.raw / "urls.txt").write_text(
            'Example source code with a URL that must survive normalization.\n'
            'url = "https://example.com/api"\nprint(url)\n',
            encoding="utf-8",
        )
        (self.raw / "other.txt").write_text(
            "Another sufficiently long reference document for this test.", encoding="utf-8"
        )
        build_dataset(self.raw, self.knowledge, self.out, min_chars=10, min_documents=2)
        combined = (
            (self.out / "train.txt").read_text(encoding="utf-8")
            + (self.out / "validation.txt").read_text(encoding="utf-8")
        )
        self.assertIn("https://example.com/api", combined)


if __name__ == "__main__":
    unittest.main()
