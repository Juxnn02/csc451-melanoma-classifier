import csv
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "app.py"


class AppTests(unittest.TestCase):
    def test_missing_checkpoint_still_offers_image_check(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "missing.pt")
            with patch.dict(os.environ, {"CSC451_MODEL_PATH": path}):
                page = AppTest.from_file(str(APP), default_timeout=15).run()
        self.assertFalse(page.exception)
        self.assertIn("Mount a trained checkpoint", page.info[0].value)
        self.assertEqual(page.button[0].label, "Check image")
        self.assertFalse(page.get("progress"))

    def test_dataset_audit_displays_manifest_counts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "images").mkdir()
            (root / "images" / "sample.jpg").touch()
            with (root / "manifest.csv").open("w", encoding="utf-8", newline="") as output:
                writer = csv.writer(output)
                writer.writerow(["isic_id", "relative_path", "proposed_class", "team_review", "patient_id", "linked_patient_id"])
                writer.writerow(["ISIC_0000001", "images/sample.jpg", "Melanoma", "pending", "", ""])
            with patch.dict(os.environ, {"CSC451_DATA_DIR": folder, "CSC451_MODEL_PATH": str(root / "missing.pt")}):
                page = AppTest.from_file(str(APP), default_timeout=15).run()
                page.radio[0].set_value("Dataset audit").run()
        self.assertFalse(page.exception)
        self.assertEqual({metric.label: metric.value for metric in page.metric},
                         {"Manifest rows": "1", "Pending review": "1", "Missing patient linkage": "1"})


if __name__ == "__main__":
    unittest.main()
