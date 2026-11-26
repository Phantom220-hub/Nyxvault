import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = PROJECT_ROOT / "training" / "validate_data.py"
FENCE = chr(96) * 3


class ValidateDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / "records.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def record(self, code="def double(value):\n    return value * 2", tests=None):
        return {
            "id": "python-double-001",
            "level": 1,
            "language": "python",
            "topic": "functions-basics",
            "type": "write_code",
            "instruction": "Напиши функцию double, которая возвращает удвоенное число.",
            "output": (
                "Решение: функция умножает значение на два.\n\n"
                + FENCE + "python\n" + code + "\n" + FENCE
            ),
            "license": "original",
            "tests": tests or ["assert double(3) == 6", "assert double(-2) == -4"],
            "source": "automated test",
        }

    def write(self, records):
        self.source.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
            encoding="utf-8",
        )

    def run_validator(self, *args):
        return subprocess.run(
            [sys.executable, str(VALIDATOR), str(self.source), *args],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            timeout=20,
        )

    def test_valid_record_and_execution_tests_pass(self):
        self.write([self.record()])
        clean_out = self.root / "clean.jsonl"
        result = self.run_validator("--exec-tests", "--clean-out", str(clean_out))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("валидных: 1", result.stdout)
        cleaned = json.loads(clean_out.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(cleaned["id"], "python-double-001")

    def test_python_syntax_error_is_rejected(self):
        self.write([self.record(code="def double(value)\n    return value * 2")])
        result = self.run_validator()
        self.assertEqual(result.returncode, 1)
        self.assertIn("E031", result.stdout)

    def test_legacy_question_answer_records_remain_supported(self):
        legacy = {
            "question": "Как удвоить число?",
            "answer": (
                "Функция умножает число на два.\n"
                + FENCE + "python\ndef double(value):\n"
                "    return value * 2\n" + FENCE
            ),
            "source": "NYXVAULT legacy seed",
        }
        self.write([legacy])
        clean_out = self.root / "legacy-clean.jsonl"
        result = self.run_validator("--clean-out", str(clean_out))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("legacy: 1", result.stdout)
        cleaned = json.loads(clean_out.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(cleaned["question"], legacy["question"])
        self.assertEqual(cleaned["source"], legacy["source"])

    def test_duplicate_id_and_task_are_rejected(self):
        first = self.record()
        second = self.record()
        second["output"] += "\nДополнительное объяснение."
        self.write([first, second])
        result = self.run_validator()
        self.assertEqual(result.returncode, 1)
        self.assertIn("E041", result.stdout)
        self.assertIn("E040", result.stdout)


if __name__ == "__main__":
    unittest.main()
