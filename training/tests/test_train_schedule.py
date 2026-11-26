import unittest

import torch

from training.train import learning_rate_at_step, resolve_precision


class TrainingScheduleTests(unittest.TestCase):
    def test_warmup_starts_below_base_lr_and_reaches_it(self):
        base_lr = 0.001
        values = [
            learning_rate_at_step(step, 20, base_lr, 4, 0.1)
            for step in range(20)
        ]
        self.assertAlmostEqual(values[0], base_lr / 4)
        self.assertAlmostEqual(values[3], base_lr)
        self.assertAlmostEqual(values[4], base_lr)
        self.assertTrue(all(a >= b for a, b in zip(values[4:], values[5:])))
        self.assertGreaterEqual(values[-1], base_lr * 0.1)
        self.assertLess(values[-1], base_lr * 0.11)

    def test_schedule_rejects_invalid_parameters(self):
        with self.assertRaises(ValueError):
            learning_rate_at_step(0, 0, 0.001, 0, 0.1)
        with self.assertRaises(ValueError):
            learning_rate_at_step(0, 10, 0.001, 10, 0.1)
        with self.assertRaises(ValueError):
            learning_rate_at_step(0, 10, 0.001, 0, 1.1)

    def test_schedule_clamps_out_of_range_step(self):
        expected = learning_rate_at_step(9, 10, 0.001, 0, 0.1)
        actual = learning_rate_at_step(1000, 10, 0.001, 0, 0.1)
        self.assertAlmostEqual(actual, expected)

    def test_cpu_precision_never_enables_cuda_scaler(self):
        dtype, scaler_enabled = resolve_precision("auto", torch.device("cpu"))
        self.assertIsNone(dtype)
        self.assertFalse(scaler_enabled)

    def test_fp32_precision_disables_autocast(self):
        dtype, scaler_enabled = resolve_precision("fp32", torch.device("cuda"))
        self.assertIsNone(dtype)
        self.assertFalse(scaler_enabled)


if __name__ == "__main__":
    unittest.main()
