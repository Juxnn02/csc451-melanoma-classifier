"""Re-create the local CSC 451 engineering pilot from the private archive manifest."""

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory


# This order and seed determine the original pilot's sample and split.
CLASSES = (
    "Squamous cell carcinoma",
    "Benign keratosis",
    "Basal cell carcinoma",
    "Melanoma",
    "Benign melanocytic nevus",
)
QUOTAS = {"train": 32, "val": 10, "test": 8}
SEED = 451
UNIQUE_FIELDS = ("lesion_id", "isic_id", "file_sha256", "pixel_sha256")
REQUIRED_COLUMNS = {
    *UNIQUE_FIELDS, "patient_id", "proposed_class", "diagnosis_confirm_type",
    "image_type", "attribution", "copyright_license", "team_review",
    "size_bytes", "width_px", "height_px", "relative_path",
}


def _patient(row):
    direct = row["patient_id"].strip()
    linked = row.get("linked_patient_id", "").strip()
    return "" if direct and linked and direct != linked else direct or linked


def _read_manifest(source):
    with (source / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        fields = reader.fieldnames or []
        if len(set(fields)) != len(fields) or not REQUIRED_COLUMNS.issubset(fields):
            raise ValueError("The archive manifest has missing or repeated columns.")
        rows = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Row {reader.line_num} does not match the manifest columns.")
            rows.append(row)
    return rows


def _select(rows):
    used_patients = set()
    selected = []
    rng = random.Random(SEED)
    for label in CLASSES:
        choices = [
            (group, row) for row in rows
            if row["proposed_class"] == label
            and (group := _patient(row))
            and row["diagnosis_confirm_type"] == "histopathology"
            and row["image_type"] == "dermoscopic"
            and row["attribution"] != "MILK study team"
            and row["copyright_license"] in {"CC-0", "CC-BY"}
            and row["team_review"] != "rejected"
            and 0 < int(row["size_bytes"]) <= 20 * 1024 * 1024
            and 0 < int(row["width_px"]) * int(row["height_px"]) <= 32_000_000
        ]
        choices.sort(key=lambda choice: choice[1]["isic_id"])
        rng.shuffle(choices)
        for split, quota in QUOTAS.items():
            included = 0
            for group, row in choices:
                if group in used_patients:
                    continue
                used_patients.add(group)
                selected.append((split, group, row))
                included += 1
                if included == quota:
                    break
            if included != quota:
                raise ValueError(f"{label} only supplied {included}/{quota} images for {split}.")

    for field in UNIQUE_FIELDS:
        if len({row[field] for _, _, row in selected if row[field]}) != len(selected):
            raise ValueError(f"The selected records have missing or duplicate {field} values.")
    return selected


def _copy_image(source, images, row):
    relative = row["relative_path"]
    parts = relative.split("/")
    if ":" in relative or "\\" in relative or any(part in {"", ".", ".."} for part in parts):
        raise ValueError("Image paths must be relative and use forward slashes.")
    original = source.joinpath(*parts).resolve(strict=True)
    if not original.is_relative_to(source) or not original.is_file():
        raise ValueError("The image path must point to a file inside the archive folder.")
    if original.stat().st_size != int(row["size_bytes"]):
        raise ValueError(f"File size differs from the archive manifest for {original.name}.")
    image = original.read_bytes()
    if hashlib.sha256(image).hexdigest() != row["file_sha256"]:
        raise ValueError(f"File checksum differs from the archive manifest for {original.name}.")
    with (images / original.name).open("xb") as stream:
        stream.write(image)
    return f"images/{original.name}"


def prepare_pilot(source, output):
    source = source.resolve(strict=True)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Pilot output already exists at {output}.")
    selected = _select(_read_manifest(source))
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        stage = Path(temporary) / "pilot"
        images = stage / "images"
        images.mkdir(parents=True)
        (stage / "models").mkdir()
        with (stage / "pilot-split.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("relative_path", "proposed_class", "group_id", "split"))
            for split, group, row in selected:
                relative_path = _copy_image(source, images, row)
                writer.writerow((relative_path, row["proposed_class"], group, split))
        summary = {
            "purpose": "Local engineering pilot with source approval and review pending",
            "selection_seed": SEED,
            "distinct_patients": len(selected),
            "distinct_lesions": len(selected),
            "split_counts": dict(Counter(split for split, _, _ in selected)),
            "per_class_counts_by_split": QUOTAS,
            "source_attributions": Counter(row["attribution"] for _, _, row in selected),
            "source_counts_by_class": {
                label: Counter(
                    row["attribution"] for _, _, row in selected if row["proposed_class"] == label
                ) for label in CLASSES
            },
            "licenses": Counter(row["copyright_license"] for _, _, row in selected),
        }
        (stage / "pilot-selection.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        if output.exists():
            raise FileExistsError(f"Pilot output already exists at {output}.")
        stage.rename(output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Folder containing manifest.csv")
    parser.add_argument("--output", required=True, type=Path, help="New private pilot folder")
    args = parser.parse_args()
    try:
        summary = prepare_pilot(args.source, args.output)
    except (csv.Error, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary["split_counts"]))
    print(f"Created pilot-split.csv in {args.output.resolve()}")


if __name__ == "__main__":
    main()
