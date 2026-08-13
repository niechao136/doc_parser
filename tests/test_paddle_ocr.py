import unittest

from src.ocr.paddle_ocr import DATA_DIR, run_single_tier


class PaddleOcrRegressionTests(unittest.TestCase):
    def test_cpu_inference_succeeds_for_sample_image(self):
        summary = run_single_tier([DATA_DIR / "evaluate.png"])

        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(summary["fail_count"], 0)


if __name__ == "__main__":
    unittest.main()