import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from labels import CLASS_NAMES
from model import load_model, predict
from train import classification_metrics, load_splits, train


class TrainingTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.manifest = self.root / "reviewed.csv"
        self.rows = []
        for split in ("train", "val"):
            for index, label in enumerate(CLASS_NAMES):
                path = f"{split}_{index}.png"
                color = (index * 40, 20 if split == "train" else 100, 120)
                Image.new("RGB", (224, 224), color).save(self.root / path)
                self.rows.append((path, label, f"group_{split}_{index}", split))
        self.write_manifest()

    def write_manifest(self):
        with self.manifest.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("relative_path", "proposed_class", "group_id", "split"))
            writer.writerows(self.rows)

    def test_rejects_unreviewed_manifest(self):
        self.manifest.write_text("relative_path,proposed_class,team_review\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing columns: group_id, split"):
            load_splits(self.manifest)

    def test_rejects_groups_crossing_into_held_out_test_set(self):
        self.rows.append(("held_out.png", CLASS_NAMES[0], self.rows[0][2], "test"))
        with (self.root / "held_out.png").open("wb") as file:
            file.write(b"Image contents are unnecessary to validate groups.")
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "group crosses splits"):
            load_splits(self.manifest)

    def test_metrics_match_known_predictions(self):
        metrics = classification_metrics(list(range(5)), [0, 0, 2, 3, 3])
        self.assertAlmostEqual(metrics["accuracy"], 3 / 5)
        self.assertAlmostEqual(metrics["melanoma_recall"], 1.0)
        self.assertAlmostEqual(metrics["macro_f1"], (2 / 3 + 1 + 2 / 3) / 5)

    def test_offline_training_writes_usable_checkpoint_and_history(self):
        output = self.root / "models" / "test.pt"
        history = self.root / "models" / "history.json"
        self.rows.append(("held_out.png", CLASS_NAMES[0], "group_test", "test"))
        (self.root / "held_out.png").write_bytes(b"The trainer must never decode the held-out image.")
        self.write_manifest()
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        train(argparse.Namespace(
            manifest=self.manifest, output=output, history=history, device="cpu", seed=451,
            batch_size=5, epochs=1, pretrained=False, fine_tune_layer4=False, baselines=False,
        ))
        run = json.loads(history.read_text(encoding="utf-8"))
        self.assertEqual(run["best_epoch"], 1)
        self.assertGreater(run["epochs"][0]["train_loss"], 0)
        with Image.open(self.root / "val_0.png") as sample:
            results = predict(load_model(output), sample)
        self.assertEqual(len(results), 3)
        self.assertTrue(all(label in CLASS_NAMES and 0 <= score <= 1 for label, score in results))


if __name__ == "__main__":
    unittest.main()
