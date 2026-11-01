import tempfile
import unittest
from pathlib import Path

from nyxvault import Nyxvault, train


class NyxvaultTests(unittest.TestCase):
    def test_train_and_retrieve(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / "knowledge.jsonl"
            index = root / "model_index.json"
            data.write_text(
                '{"question":"Как создать игру в Godot","answer":"Используй игровой движок Godot."}\n',
                encoding="utf-8",
            )
            stats = train(data, index)
            self.assertEqual(stats["examples"], 1)
            bot = Nyxvault(data, index)
            self.assertIn("Godot", bot.answer("Как создать игру в Godot?"))

    def test_unknown_question_does_not_invent(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / "knowledge.jsonl"
            index = root / "model_index.json"
            data.write_text(
                '{"question":"Как создать игру в Godot","answer":"Используй Godot."}\n',
                encoding="utf-8",
            )
            train(data, index)
            bot = Nyxvault(data, index)
            self.assertIn("пока не знаю", bot.answer("Расскажи о квантовой хромодинамике"))

if __name__ == "__main__":
    unittest.main()
