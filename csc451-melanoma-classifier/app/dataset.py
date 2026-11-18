"""Audit the local candidate manifest without reading image contents."""

import csv
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from labels import CLASS_NAMES


REQUIRED_COLUMNS = {
    "isic_id",
    "relative_path",
    "proposed_class",
    "team_review",
    "patient_id",
}
REVIEW_STATES = {"pending", "approved", "rejected"}


@dataclass(frozen=True)
class DatasetAudit:
    total_rows: int
    class_counts: dict[str, int]
    pending_review: int
    approved_rows: int
    rejected_rows: int
    missing_files: int
    missing_patient_linkage: int
    invalid_rows: int
    issues: tuple[str, ...]


@dataclass(frozen=True)
class _Candidate:
    line: int
    image_id: str
    image_path: Path
    image_exists: bool
    class_name: str
    review: str
    has_patient_linkage: bool


def _path_parts(relative_path: str) -> list[str]:
    parts = relative_path.split("/")
    if ":" in relative_path or "\\" in relative_path or any(
        part in {"", ".", ".."} for part in parts
    ):
        raise ValueError("Image paths must be relative and use forward slashes.")
    return parts


def _confined_path(root: Path, path: Path) -> Path:
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError) as error:
        raise ValueError("The image path could not be resolved.") from error
    if not resolved.is_relative_to(root):
        raise ValueError("The image path points outside the dataset directory.")
    return resolved


def resolve_image_path(dataset_dir: Path, relative_path: str) -> Path:
    """Resolve a manifest path and confine it to the dataset directory."""
    root = dataset_dir.resolve()
    result = _confined_path(root, root.joinpath(*_path_parts(relative_path)))
    if result == root:
        raise ValueError("The image path must point to a file inside the dataset directory.")
    return result


def _audit_image_path(root: Path, path: str, directories: dict) -> tuple[Path, bool]:
    parts = _path_parts(path)
    parent = root.joinpath(*parts[:-1])
    # Scan each directory once to keep audits fast on Docker Desktop bind mounts.
    if parent not in directories:
        resolved_parent = _confined_path(root, parent)
        try:
            with os.scandir(resolved_parent) as entries:
                directories[parent] = (resolved_parent, {os.path.normcase(entry.name): entry for entry in entries})
        except (FileNotFoundError, NotADirectoryError):
            directories[parent] = (resolved_parent, {})
    resolved_parent, entries = directories[parent]
    entry = entries.get(os.path.normcase(parts[-1]))
    if entry is None:
        return resolved_parent / parts[-1], False
    if entry.is_symlink():
        resolved = resolve_image_path(root, path)
        return resolved, resolved.is_file()
    return Path(entry.path), entry.is_file(follow_symlinks=False)


def _candidate(line: int, row: dict[str, str], dataset_dir: Path, directories: dict) -> _Candidate:
    image_id = row["isic_id"].strip()
    class_name = row["proposed_class"].strip()
    review = row["team_review"].strip().lower()
    if not image_id:
        raise ValueError("Missing image ID.")
    if class_name not in CLASS_NAMES:
        raise ValueError("Unsupported or missing proposed class.")
    if review not in REVIEW_STATES:
        raise ValueError("Review status must be pending, approved, or rejected.")

    direct_id = row["patient_id"].strip()
    linked_id = row.get("linked_patient_id", "").strip()
    if direct_id and linked_id and direct_id != linked_id:
        raise ValueError("Conflicting patient linkage.")
    path, exists = _audit_image_path(dataset_dir, row["relative_path"].strip(), directories)
    return _Candidate(
        line,
        image_id,
        path,
        exists,
        class_name,
        review,
        bool(direct_id or linked_id),
    )


def audit_dataset(dataset_dir: str | Path) -> DatasetAudit:
    """Count candidate records, local files, review states, and linkage gaps."""
    root = Path(dataset_dir).resolve(strict=True)
    manifest = resolve_image_path(root, "manifest.csv")
    candidates = []
    directories = {}
    issues = []
    total_rows = 0
    try:
        with manifest.open(newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source, strict=True)
            fields = reader.fieldnames or []
            if len(fields) != len(set(fields)) or not REQUIRED_COLUMNS.issubset(fields):
                raise ValueError("Manifest columns are missing or repeated.")
            for row in reader:
                total_rows += 1
                try:
                    if None in row or any(value is None for value in row.values()):
                        raise ValueError("Row has the wrong number of fields.")
                    candidates.append(_candidate(reader.line_num, row, root, directories))
                except ValueError as error:
                    issues.append(f"Line {reader.line_num}: {error}")
    except (csv.Error, UnicodeError) as error:
        raise ValueError("The manifest must be a well-formed UTF-8 CSV.") from error

    image_ids = Counter(row.image_id for row in candidates)
    paths = Counter(row.image_path for row in candidates)
    counts = dict.fromkeys(CLASS_NAMES, 0)
    review_counts = Counter()
    missing_files = 0
    missing_patient_linkage = 0
    for row in candidates:
        if image_ids[row.image_id] > 1 or paths[row.image_path] > 1:
            issues.append(f"Line {row.line}: Duplicate image ID or path.")
            continue
        counts[row.class_name] += 1
        review_counts[row.review] += 1
        missing_files += not row.image_exists
        missing_patient_linkage += not row.has_patient_linkage

    return DatasetAudit(
        total_rows=total_rows,
        class_counts=counts,
        pending_review=review_counts["pending"],
        approved_rows=review_counts["approved"],
        rejected_rows=review_counts["rejected"],
        missing_files=missing_files,
        missing_patient_linkage=missing_patient_linkage,
        invalid_rows=len(issues),
        issues=tuple(issues[:10]),
    )
