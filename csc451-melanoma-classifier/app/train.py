"""Fit an app checkpoint and measure performance on the reviewed validation split."""

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dataset import resolve_image_path
from labels import CLASS_NAMES
from model import build_model, decode_image, make_checkpoint, preprocess


CLASS_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
SPLITS = ("train", "val", "test")
REVIEWED_COLUMNS = {"relative_path", "proposed_class", "group_id", "split"}


@dataclass(frozen=True)
class Example:
    path: Path
    label: int


def load_splits(manifest: Path) -> dict[str, list[Example]]:
    """Check the team's existing split assignments before reading image contents."""
    dataset_dir = manifest.parent.resolve()
    splits: dict[str, list[Example]] = {name: [] for name in SPLITS}
    group_splits: dict[str, str] = {}
    assigned_paths: set[Path] = set()
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream, strict=True)
        fields = rows.fieldnames or []
        missing = REVIEWED_COLUMNS.difference(fields)
        if missing:
            raise ValueError(f"Reviewed manifest is missing columns: {', '.join(sorted(missing))}.")
        if len(fields) != len(set(fields)):
            raise ValueError("Reviewed manifest has duplicate column names.")
        for row in rows:
            if None in row or any(row[column] is None for column in REVIEWED_COLUMNS):
                raise ValueError(f"Row {rows.line_num} does not match the manifest columns.")
            split = (row["split"] or "").strip()
            group_id = (row["group_id"] or "").strip()
            label = (row["proposed_class"] or "").strip()
            if split not in SPLITS or not group_id or label not in CLASS_INDEX:
                raise ValueError(f"Row {rows.line_num} requires a valid split, group ID and class.")
            if group_splits.setdefault(group_id, split) != split:
                raise ValueError(f"A group crosses splits at row {rows.line_num}.")
            path = resolve_image_path(dataset_dir, (row["relative_path"] or "").strip())
            if not path.is_file() or path in assigned_paths:
                raise ValueError(f"Missing or duplicate image at row {rows.line_num}.")
            assigned_paths.add(path)
            splits[split].append(Example(path, CLASS_INDEX[label]))

    for split in ("train", "val"):
        represented = {example.label for example in splits[split]}
        if len(represented) != len(CLASS_NAMES):
            raise ValueError(f"The {split} split must contain all {len(CLASS_NAMES)} classes.")
    return splits


class ReviewedImages(Dataset):
    def __init__(self, examples: list[Example]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        with decode_image(example.path.read_bytes()) as image:
            return preprocess(image), example.label


def classification_metrics(actual: list[int], predicted: list[int]) -> dict[str, float]:
    support = Counter(actual)
    predicted_counts = Counter(predicted)
    correct = Counter(label for label, guess in zip(actual, predicted, strict=True) if label == guess)
    f1 = [2 * correct[index] / (support[index] + predicted_counts[index]) for index in CLASS_INDEX.values()]
    melanoma = CLASS_INDEX["Melanoma"]
    return {
        "accuracy": sum(correct.values()) / len(actual),
        "macro_f1": sum(f1) / len(CLASS_NAMES),
        "melanoma_recall": correct[melanoma] / support[melanoma],
    }


def baseline_metrics(train: list[Example], val: list[Example]) -> dict:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def pixels(examples):
        result = []
        for example in examples:
            with decode_image(example.path.read_bytes()) as image:
                resized = image.resize((32, 32), Image.Resampling.BILINEAR)
                result.append(np.asarray(resized, dtype=np.float32).ravel())
        return np.stack(result)

    train_labels = [example.label for example in train]
    val_labels = [example.label for example in val]
    majority = Counter(train_labels).most_common(1)[0][0]
    logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300))
    logistic.fit(pixels(train), train_labels)
    return {
        "majority_class": classification_metrics(val_labels, [majority] * len(val)),
        "logistic_regression": classification_metrics(val_labels, logistic.predict(pixels(val)).tolist()),
    }


def validate(model, loader, loss_fn, device):
    model.eval()
    actual, predicted = [], []
    total_loss = 0.0
    with torch.inference_mode():
        for images, labels in loader:
            scores = model(images.to(device))
            total_loss += loss_fn(scores, labels.to(device)).item() * len(labels)
            actual.extend(labels.tolist())
            predicted.extend(scores.argmax(dim=1).tolist())
    return total_loss / len(actual), classification_metrics(actual, predicted)


def save_checkpoint(checkpoint, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(checkpoint, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def train(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Use --device cpu for the Docker image.")
    device = torch.device(args.device)
    splits = load_splits(args.manifest)
    output_paths = {path.resolve() for path in (args.output, args.history) if path is not None}
    if any(example.path in output_paths for examples in splits.values() for example in examples):
        raise ValueError("The checkpoint and history paths need to be separate from the dataset images.")
    torch.manual_seed(args.seed)
    train_loader = DataLoader(ReviewedImages(splits["train"]), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ReviewedImages(splits["val"]), batch_size=args.batch_size)
    model = build_model(pretrained=args.pretrained).to(device)
    if args.pretrained:
        model.requires_grad_(False)
        model.fc.requires_grad_(True)
        if args.fine_tune_layer4:
            model.layer4.requires_grad_(True)
    optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    report = {"epochs": []}
    if args.baselines:
        report["baselines"] = baseline_metrics(splits["train"], splits["val"])
        print(json.dumps({"validation_baselines": report["baselines"]}), flush=True)

    best_score = (-1.0, -1.0)
    for epoch in range(1, args.epochs + 1):
        model.train(not args.pretrained)
        if args.pretrained:
            model.fc.train()
            if args.fine_tune_layer4:
                model.layer4.train()
        total_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        val_loss, metrics = validate(model, val_loader, loss_fn, device)
        result = {"epoch": epoch, "train_loss": total_loss / len(splits["train"]), "val_loss": val_loss, **metrics}
        print(json.dumps(result), flush=True)
        report["epochs"].append(result)
        score = (metrics["macro_f1"], metrics["melanoma_recall"])
        if score > best_score:
            best_score = score
            report["best_epoch"] = epoch
            save_checkpoint(make_checkpoint(model, validation_metrics=metrics), args.output)

    if args.history:
        args.history.parent.mkdir(parents=True, exist_ok=True)
        args.history.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved best validation checkpoint to {args.output}", flush=True)


def positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Reviewed split CSV")
    parser.add_argument("--output", type=Path, required=True, help="Output checkpoint path")
    parser.add_argument("--history", type=Path, help="Optional JSON learning history")
    parser.add_argument("--epochs", type=positive_integer, default=1)
    parser.add_argument("--batch-size", type=positive_integer, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=451)
    parser.add_argument("--pretrained", action="store_true", help="Load ImageNet weights and freeze the backbone")
    parser.add_argument("--fine-tune-layer4", action="store_true", help="Also fine-tune layer4 with --pretrained")
    parser.add_argument("--baselines", action="store_true", help="Evaluate majority class and logistic regression")
    args = parser.parse_args()
    if args.fine_tune_layer4 and not args.pretrained:
        parser.error("--fine-tune-layer4 requires --pretrained.")
    paths = [path.resolve() for path in (args.manifest, args.output, args.history) if path is not None]
    if len(set(paths)) != len(paths):
        parser.error("The manifest, checkpoint and history need separate paths.")
    try:
        train(args)
    except (csv.Error, OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
