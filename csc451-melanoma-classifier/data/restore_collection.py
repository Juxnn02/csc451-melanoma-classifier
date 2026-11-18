"""Restore the exact image bytes listed in a privately shared ISIC manifest."""

import argparse
import csv
import hashlib
import io
import re
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, build_opener


ARCHIVE = "https://isic-archive.s3.us-east-1.amazonaws.com"
MAX_IMAGE_BYTES = 20 * 1024 * 1024
BLOCK_BYTES = 128 * 1024


@dataclass(frozen=True)
class ImageRecord:
    line: int
    relative_path: str
    url: str
    size: int
    sha256: str


def parse_manifest(contents: bytes) -> list[ImageRecord]:
    required = {"isic_id", "relative_path", "original_url", "size_bytes"}
    records = []
    seen = set()
    try:
        reader = csv.DictReader(io.StringIO(contents.decode("utf-8-sig"), newline=""), strict=True)
        fields = reader.fieldnames or []
        hash_field = "file_sha256" if "file_sha256" in fields else "sha256"
        if len(fields) != len(set(fields)) or not (required | {hash_field}).issubset(fields):
            raise ValueError("Manifest columns are missing or repeated.")

        for row in reader:
            line = reader.line_num
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Manifest line {line} has the wrong number of fields.")
            image_id = row["isic_id"]
            relative_path = f"images/{image_id}.jpg"
            url = f"{ARCHIVE}/{relative_path}"
            if (
                not re.fullmatch(r"ISIC_[0-9]{7}", image_id)
                or row["relative_path"] != relative_path
                or row["original_url"] != url
            ):
                raise ValueError(f"Manifest line {line} must point to its exact ISIC archive JPEG.")
            try:
                size = int(row["size_bytes"])
            except ValueError:
                raise ValueError(f"Manifest line {line} has an invalid byte count.") from None
            digest = row[hash_field].lower()
            if not 0 < size <= MAX_IMAGE_BYTES or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"Manifest line {line} has an invalid byte count or SHA-256.")
            if image_id in seen:
                raise ValueError(f"Manifest line {line} repeats an image ID.")
            seen.add(image_id)
            records.append(ImageRecord(line, relative_path, url, size, digest))
    except (csv.Error, UnicodeError) as error:
        raise ValueError("Manifest must be a well-formed UTF-8 CSV.") from error
    if not records:
        raise ValueError("Manifest has no image records.")
    return records


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _open_image(url: str):
    return build_opener(_NoRedirect()).open(url, timeout=30)


@contextmanager
def _atomic_file(destination: Path):
    with NamedTemporaryFile(dir=destination.parent, suffix=".part", delete=False) as stream:
        staged = Path(stream.name)
        try:
            yield stream
            stream.close()
            staged.replace(destination)
        finally:
            stream.close()
            staged.unlink(missing_ok=True)


def _verified_image(record: ImageRecord, source, target=None) -> bool:
    digest = hashlib.sha256()
    size = 0
    for block in iter(lambda: source.read(BLOCK_BYTES), b""):
        size += len(block)
        if size > record.size:
            return False
        digest.update(block)
        if target is not None:
            target.write(block)
    return size == record.size and digest.hexdigest() == record.sha256


def restore_image(record: ImageRecord, output: Path) -> bool:
    """Return whether this image was downloaded rather than already verified."""
    destination = output / record.relative_path
    if destination.is_symlink():
        raise ValueError(f"Manifest line {record.line} points to a symlink in the output.")
    try:
        if destination.exists():
            if destination.is_file():
                with destination.open("rb") as local:
                    if _verified_image(record, local):
                        return False
            raise ValueError(f"Manifest line {record.line} has a conflicting local file. "
                             "Use another output directory or remove the conflicting file.")
        with _open_image(record.url) as response, _atomic_file(destination) as target:
            if not _verified_image(record, response, target):
                raise ValueError(f"Archive contents for manifest line {record.line} do not match the recorded bytes.")
    except HTTPError as error:
        raise ValueError(f"Archive returned HTTP {error.code} for manifest line {record.line}.") from error
    except OSError as error:
        raise ValueError(f"Could not restore manifest line {record.line} ({type(error).__name__}).") from error
    return True


def restore_collection(manifest: Path, output: Path) -> tuple[int, int]:
    contents = manifest.read_bytes()
    records = parse_manifest(contents)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    lock = output / ".restore-collection.lock"
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise ValueError(
            "A restore already reserved this output folder. After an interrupted run, "
            "remove the empty .restore-collection.lock directory once no restore is running."
        ) from error
    try:
        return _populate(output, records, contents)
    finally:
        lock.rmdir()


def _populate(output: Path, records: list[ImageRecord], contents: bytes) -> tuple[int, int]:
    local_manifest = output / "manifest.csv"
    if local_manifest.is_symlink() or (local_manifest.exists() and local_manifest.read_bytes() != contents):
        raise ValueError("Output already contains a different manifest. Choose a different output directory.")
    images = output / "images"
    if images.resolve() != images or images.is_symlink():
        raise ValueError("The output images directory may not be a link.")
    images.mkdir(exist_ok=True)

    downloads = ThreadPoolExecutor(max_workers=8)
    try:
        restored = sum(downloads.map(lambda record: restore_image(record, output), records))
    finally:
        downloads.shutdown(cancel_futures=True)
    if not local_manifest.exists():
        with _atomic_file(local_manifest) as target:
            target.write(contents)
    return restored, len(records) - restored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Private manifest CSV")
    parser.add_argument("--output", type=Path, required=True, help="Private destination folder")
    args = parser.parse_args()
    try:
        restored, existing = restore_collection(args.manifest, args.output)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Restore failed: {error}\n")
    print(f"Verified {restored + existing:,} images ({restored:,} downloaded, {existing:,} already present).")


if __name__ == "__main__":
    main()
