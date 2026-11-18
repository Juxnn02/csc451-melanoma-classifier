import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from torch import nn

from evaluate import evaluate
from labels import CLASS_NAMES


class OrderedScores(nn.Module):
    validation_metrics = {}

    def forward(self, images):
        return torch.eye(len(CLASS_NAMES), dtype=images.dtype)


class HoldoutTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest = self.root / "split.csv"
        self.rows = []
        for split in ("train", "val", "test"):
            for index, label in enumerate(CLASS_NAMES):
                path = self.root / f"{split}-{index}.png"
                if split == "test":
                    Image.new("RGB", (12, 12)).save(path)
                else:
                    path.write_bytes(b"Evaluation must never decode training or validation files.")
                self.rows.append((path.name, label, f"{split}-patient-{index}", split))

    def write_manifest(self, rows):
        with self.manifest.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("relative_path", "proposed_class", "group_id", "split"))
            writer.writerows(rows)

    def test_held_out_metrics_use_only_test_images(self):
        self.write_manifest(self.rows)
        with patch("evaluate.load_model", return_value=OrderedScores()):
            report = evaluate(self.root / "checkpoint.pt", self.manifest)
        count = len(CLASS_NAMES)
        self.assertEqual(report["split_counts"], {"train": count, "val": count, "test": count})
        self.assertEqual(report["holdout"]["correct_top1"], count)
        self.assertEqual(report["holdout"]["correct_top3"], count)
        self.assertEqual(report["holdout"]["macro_f1"], 1.0)
        self.assertEqual(
            report["holdout"]["confusion_matrix_actual_rows"],
            [[int(row == column) for column in range(count)] for row in range(count)],
        )
        self.assertNotIn("verified_frozen_tensors", report)

    def test_requires_test_support_for_each_class_before_loading_checkpoint(self):
        self.write_manifest(self.rows[:-1])
        with patch("evaluate.load_model") as load_model:
            with self.assertRaisesRegex(ValueError, "test split must contain all five classes"):
                evaluate(self.root / "checkpoint.pt", self.manifest)
        load_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
