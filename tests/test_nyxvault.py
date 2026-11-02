import tempfile
import unittest
from pathlib import Path

from nyxvault import Nyxvault, train
from neural_brain import NeuralBrain, train_neural


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

    def test_neural_model_trains_and_generates(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / "knowledge.jsonl"
            model = root / "neural_model.npz"
            data.write_text(
                '{"question":"Привет","answer":"Я NYXVAULT."}\n'
                '{"question":"Что такое Python","answer":"Python — язык программирования."}\n',
                encoding="utf-8",
            )
            stats = train_neural(data, model, epochs=2)
            self.assertTrue(model.exists())
            self.assertGreater(stats["characters"], 10)
            self.assertEqual(stats["architecture"], "vanilla_rnn")
            bot = NeuralBrain(model)
            self.assertIsInstance(bot.generate("Ответ:", max_new=8), str)
            self.assertFalse(bot.looks_plausible("зювтоп шшщщк лллпп", data))
            self.assertTrue(bot.looks_plausible("Привет Я NYXVAULT", data))


    def test_neural_training_resumes_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / "knowledge.jsonl"
            model = root / "neural_model.npz"
            data.write_text(
                '{"question":"Привет","answer":"Я NYXVAULT."}\\n'
                '{"question":"Что такое Python","answer":"Python — язык программирования."}\\n',
                encoding="utf-8",
            )
            first = train_neural(data, model, epochs=1)
            second = train_neural(data, model, epochs=1)
            self.assertFalse(first["resumed"])
            self.assertTrue(second["resumed"])
            self.assertGreater(second["optimizer_steps"], first["optimizer_steps"])
            self.assertEqual(second["epochs_total"], 2)

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
