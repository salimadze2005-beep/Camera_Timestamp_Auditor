import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from postprocess_timestamp_results import choose_candidate
from run_easyocr import position_boxes


class TimestampLogicTests(unittest.TestCase):
    def test_exact_timestamp(self):
        dt, delta, status, _ = choose_candidate(
            "2026-06-08 14:31:22",
            "2026-06-08T14:31:22+05:00",
            "Asia/Yekaterinburg",
            10.0,
            allow_reference_repair=False,
        )
        self.assertIsNotNone(dt)
        self.assertEqual(status, "OK")
        self.assertLessEqual(delta, 0.01)

    def test_fallback_bbox_inside_image(self):
        import numpy as np
        image = np.zeros((1000, 1600, 3), dtype=np.uint8)
        boxes = position_boxes(image, "bottom-right")
        self.assertGreaterEqual(len(boxes), 3)
        for _, (x1, y1, x2, y2) in boxes:
            self.assertTrue(0 <= x1 < x2 <= 1600)
            self.assertTrue(0 <= y1 < y2 <= 1000)


if __name__ == "__main__":
    unittest.main()
