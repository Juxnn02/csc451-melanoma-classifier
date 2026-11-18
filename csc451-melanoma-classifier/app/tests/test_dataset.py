import csv
import tempfile
import unittest
from pathlib import Path

from dataset import audit_dataset, resolve_image_path


FIELDS = (
    "isic_id",
    "proposed_class",
    "relative_path",
    "team_review",
    "patient_id",
    "linked_patient_id",
)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def write_manifest(self, *rows):
        with (self.root / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as target:
            writer = csv.writer(target)
            writer.writerow(FIELDS)
            writer.writerows(rows)

    def test_candidate_counts_include_linked_patient_ids_and_missing_files(self):
        images = self.root / "images"
        images.mkdir()
        (images / "one.jpg").touch()
        (images / "three.jpg").touch()
        self.write_manifest(
            ("first", "Melanoma", "images/one.jpg", "pending", "", "linked"),
            ("second", "Benign keratosis", "images/two.jpg", "approved", "", ""),
            ("third", "Melanoma", "images/three.jpg", "rejected", "direct", ""),
        )

        result = audit_dataset(self.root)

        self.assertEqual(result.total_rows, 3)
        self.assertEqual(result.class_counts["Melanoma"], 2)
        self.assertEqual(result.class_counts["Basal cell carcinoma"], 0)
        self.assertEqual(result.pending_review, 1)
        self.assertEqual(result.approved_rows, 1)
        self.assertEqual(result.rejected_rows, 1)
        self.assertEqual(result.missing_files, 1)
        self.assertEqual(result.missing_patient_linkage, 1)
        self.assertEqual(result.invalid_rows, 0)

    def test_invalid_rows_and_duplicate_ids_or_paths_are_excluded(self):
        self.write_manifest(
            ("same", "Melanoma", "first.jpg", "pending", "", ""),
            ("same", "Benign keratosis", "second.jpg", "pending", "", ""),
            ("third", "Melanoma", "first.jpg", "pending", "", ""),
            ("escape", "Melanoma", "../outside.jpg", "pending", "", ""),
            ("unknown", "Other", "other.jpg", "pending", "", ""),
            ("invalid status", "Melanoma", "status.jpg", "done", "", ""),
            ("conflicting", "Melanoma", "linkage.jpg", "pending", "a", "b"),
            ("short", "Melanoma"),
        )

        result = audit_dataset(self.root)

        self.assertEqual(result.total_rows, 8)
        self.assertEqual(result.invalid_rows, 8)
        self.assertEqual(result.pending_review, 0)
        self.assertEqual(result.class_counts["Melanoma"], 0)
        self.assertEqual(len(result.issues), 8)

    def test_original_pilot_manifest_does_not_need_linked_patient_column(self):
        (self.root / "pilot.jpg").touch()
        with (self.root / "manifest.csv").open("w", newline="", encoding="utf-8") as target:
            writer = csv.writer(target)
            writer.writerow(FIELDS[:-1])
            writer.writerow(("pilot", "Melanoma", "pilot.jpg", "pending", "direct"))

        result = audit_dataset(self.root)
        self.assertEqual((result.total_rows, result.missing_patient_linkage, result.invalid_rows), (1, 0, 0))

    def test_malformed_header_is_reported(self):
        (self.root / "manifest.csv").write_text("relative_path,team_review\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Manifest columns"):
            audit_dataset(self.root)

    def test_relative_paths_reject_escaping_spelling(self):
        for path in ("/etc/file.jpg", "../file.jpg", "images/../file.jpg", "C:/file.jpg", "images\\file.jpg", "file.jpg:stream"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                resolve_image_path(self.root, path)

    def test_relative_paths_reject_symlink_to_outside_directory(self):
        dataset_dir = self.root / "dataset"
        dataset_dir.mkdir()
        outside = self.root / "outside.jpg"
        outside.touch()
        try:
            (dataset_dir / "image.jpg").symlink_to(outside)
        except OSError as error:
            self.skipTest(f"Symlinks unavailable: {error}")

        with self.assertRaisesRegex(ValueError, "outside"):
            resolve_image_path(dataset_dir, "image.jpg")

    def test_missing_manifest_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            audit_dataset(self.root)


if __name__ == "__main__":
    unittest.main()
