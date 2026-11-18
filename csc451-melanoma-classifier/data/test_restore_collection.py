import csv
import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import restore_collection as restore


PAYLOAD = b"Test bytes with a known checksum"


def record(**changes):
    image_id = "ISIC_0000001"
    relative_path = f"images/{image_id}.jpg"
    return {
        "isic_id": image_id,
        "relative_path": relative_path,
        "original_url": f"{restore.ARCHIVE}/{relative_path}",
        "size_bytes": str(len(PAYLOAD)),
        "file_sha256": hashlib.sha256(PAYLOAD).hexdigest(),
    } | changes


class RestoreCollectionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.manifest = self.root / "private.csv"
        self.output = self.root / "restored"

    def write_manifest(self, *rows):
        with self.manifest.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)

    def test_restore_and_resume_existing_verified_file(self):
        self.write_manifest(record())
        with patch.object(restore, "_open_image", return_value=io.BytesIO(PAYLOAD)) as request:
            self.assertEqual(restore.restore_collection(self.manifest, self.output), (1, 0))
            request.assert_called_once_with(record()["original_url"])
        self.assertEqual((self.output / record()["relative_path"]).read_bytes(), PAYLOAD)
        self.assertEqual((self.output / "manifest.csv").read_bytes(), self.manifest.read_bytes())
        with patch.object(restore, "_open_image") as request:
            self.assertEqual(restore.restore_collection(self.manifest, self.output), (0, 1))
            request.assert_not_called()

    def test_invalid_source_path_and_duplicates_fail_before_writing(self):
        for rows in (
            (record(original_url="https://example.invalid/image.jpg"),),
            (record(relative_path="../original.jpg"),),
            (record(), record()),
        ):
            with self.subTest(rows=rows):
                self.write_manifest(*rows)
                with patch.object(restore, "_open_image") as request:
                    with self.assertRaises(ValueError):
                        restore.restore_collection(self.manifest, self.output)
                    request.assert_not_called()
                self.assertFalse(self.output.exists())

    def test_invalid_download_leaves_no_image_manifest_or_partial_file(self):
        self.write_manifest(record())
        with patch.object(restore, "_open_image", return_value=io.BytesIO(b"x" * len(PAYLOAD))):
            with self.assertRaisesRegex(ValueError, "do not match"):
                restore.restore_collection(self.manifest, self.output)
        self.assertEqual(list((self.output / "images").iterdir()), [])
        self.assertFalse((self.output / "manifest.csv").exists())
        self.assertFalse((self.output / ".restore-collection.lock").exists())

    def test_legacy_manifest_hash_name_is_supported(self):
        row = record()
        row["sha256"] = row.pop("file_sha256")
        self.write_manifest(row)
        with patch.object(restore, "_open_image", return_value=io.BytesIO(PAYLOAD)):
            self.assertEqual(restore.restore_collection(self.manifest, self.output), (1, 0))

    def test_conflicting_existing_file_is_preserved(self):
        self.write_manifest(record())
        image = self.output / record()["relative_path"]
        image.parent.mkdir(parents=True)
        image.write_bytes(b"existing file")
        with patch.object(restore, "_open_image") as request:
            with self.assertRaisesRegex(ValueError, "conflicting local file"):
                restore.restore_collection(self.manifest, self.output)
            request.assert_not_called()
        self.assertEqual(image.read_bytes(), b"existing file")

    def test_concurrent_restore_rejects_an_already_reserved_output(self):
        self.write_manifest(record())
        (self.output / ".restore-collection.lock").mkdir(parents=True)
        with patch.object(restore, "_open_image") as request:
            with self.assertRaisesRegex(ValueError, "already reserved"):
                restore.restore_collection(self.manifest, self.output)
            request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
