import unittest

from training.chat_eval import score_answer


class ChatEvaluationTests(unittest.TestCase):
    def test_good_answer_passes_basic_rubric(self):
        result = score_answer(
            "Привет! Переменная хранит значение или данные программы.",
            {"must_include_any": [["перем", "значен", "данн"]]},
        )
        self.assertEqual(result["heuristic_score"], 100)
        self.assertTrue(all(result["checks"].values()))

    def test_empty_answer_fails(self):
        result = score_answer("", {"must_include_any": [["привет"]]})
        self.assertFalse(result["checks"]["non_empty"])
        self.assertEqual(result["heuristic_score"], 20)

    def test_repetitive_answer_is_flagged(self):
        result = score_answer(
            "да да да да да да да да",
            {"must_include_any": [["да"]]},
        )
        self.assertFalse(result["checks"]["not_obviously_repetitive"])

    def test_prompt_template_leak_is_flagged(self):
        result = score_answer(
            "### Solution\nПривет!",
            {"must_include_any": [["привет"]]},
        )
        self.assertFalse(result["checks"]["no_prompt_template_leak"])


    def test_every_required_concept_group_must_be_present(self):
        result = score_answer(
            "Переменная хранит значение.",
            {"must_include_groups": [["переменная"], ["значение"], ["пример"]]},
        )
        self.assertFalse(result["checks"]["contains_expected_concept_words"])
        self.assertEqual(len(result["missing_expected_groups"]), 1)

    def test_code_answer_requires_the_expected_fence(self):
        result = score_answer(
            "def square(x): return x ** 2",
            {
                "must_include_groups": [["def "], ["return"], ["** 2"]],
                "expected_code_language": "python",
            },
        )
        self.assertFalse(result["checks"]["expected_code_format"])


if __name__ == "__main__":
    unittest.main()
