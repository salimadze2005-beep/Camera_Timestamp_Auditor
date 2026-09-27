import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from postprocess_timestamp_results import main


class PostprocessCsvTests(unittest.TestCase):
    def test_fills_empty_parsed_time_column(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "input.csv"
            output = Path(temp_dir) / "output.csv"
            pd.DataFrame([{
                "reference_time": "2026-06-08T14:31:22+05:00",
                "timezone": "Asia/Yekaterinburg",
                "ocr_text_ready_model": "2026-06-08 14:31:22",
                "parsed_time": "",
                "delta_minutes": "",
                "status": "PARSE_FAILED",
                "status_reason": "",
            }]).to_csv(source, index=False)
            argv = ["postprocess", "--in-csv", str(source), "--out-csv", str(output)]
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                main()
            result = pd.read_csv(output)
            self.assertEqual(result.loc[0, "status"], "OK")
            self.assertEqual(result.loc[0, "parsed_time"], "2026-06-08 14:31:22")


if __name__ == "__main__":
    unittest.main()
