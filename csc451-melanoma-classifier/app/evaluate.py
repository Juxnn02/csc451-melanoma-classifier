"""Evaluate a saved checkpoint once on the patient-disjoint test split."""

import argparse
import csv
import json
import pickle
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from labels import CLASS_NAMES
from model import build_model, load_model
from train import ReviewedImages, classification_metrics, load_splits


FINE_TUNED_PREFIXES = ("layer4.", "fc.")


def _verify_layer4_run(network, initial_weights):
    try:
        pretrained = torch.load(initial_weights, weights_only=True, map_location="cpu")
        if not isinstance(pretrained, dict):
            raise ValueError("Expected a ResNet-18 state dictionary for the initial weights.")
        reference = build_model()
        result = reference.load_state_dict(
            {name: tensor for name, tensor in pretrained.items() if not name.startswith("fc.")},
            strict=False,
        )
    except (AttributeError, EOFError, RuntimeError, pickle.UnpicklingError) as error:
        raise ValueError("Could not read the initial ResNet-18 weights.") from error
    if set(result.missing_keys) != {"fc.weight", "fc.bias"} or result.unexpected_keys:
        raise ValueError("The initial weights must contain the complete ResNet-18 backbone.")

    initial = reference.state_dict()
    state = network.state_dict()
    frozen = [name for name in state if not name.startswith(FINE_TUNED_PREFIXES)]
    if not all(torch.equal(state[name], initial[name]) for name in frozen):
        raise ValueError("A frozen backbone tensor differs from the initial weights.")
    if not any(not torch.equal(tensor, initial[name]) for name, tensor in state.items()
               if name.startswith("layer4.") and tensor.is_floating_point()):
        raise ValueError("The final residual stage matches the initial weights.")
    return {
        "verified_frozen_tensors": len(frozen),
        "fine_tuned_parameters": sum(
            parameter.numel() for name, parameter in network.named_parameters()
            if name.startswith(FINE_TUNED_PREFIXES)
        ),
    }


def evaluate(checkpoint, manifest, initial_weights=None):
    splits = load_splits(manifest)
    test = splits["test"]
    if {sample.label for sample in test} != set(range(len(CLASS_NAMES))):
        raise ValueError("The test split must contain all five classes.")
    network = load_model(checkpoint)
    verification = _verify_layer4_run(network, initial_weights) if initial_weights else {}

    by_class = [[0] * len(CLASS_NAMES) for _ in CLASS_NAMES]
    actual, predicted = [], []
    top3_hits = 0
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    with torch.inference_mode():
        for images, labels in DataLoader(ReviewedImages(test), batch_size=16):
            logits = network(images)
            if not torch.isfinite(logits).all():
                raise ValueError("The checkpoint produced non-finite test scores.")
            indices = logits.topk(3, dim=1).indices
            choices = indices[:, 0].tolist()
            loss += loss_fn(logits, labels).item()
            top3_hits += indices.eq(labels.unsqueeze(1)).any(dim=1).sum().item()
            batch_labels = labels.tolist()
            actual.extend(batch_labels)
            predicted.extend(choices)
            for label, guess in zip(batch_labels, choices, strict=True):
                by_class[label][guess] += 1

    return {
        "purpose": "Held-out evaluation on the supplied patient-disjoint manifest",
        "split_counts": {split: len(records) for split, records in splits.items()},
        "class_names": list(CLASS_NAMES),
        "holdout": {
            **classification_metrics(actual, predicted),
            "loss": loss / len(test),
            "correct_top1": sum(right == guess for right, guess in zip(actual, predicted, strict=True)),
            "correct_top3": top3_hits,
            "top3_accuracy": top3_hits / len(test),
            "confusion_matrix_actual_rows": by_class,
        },
        "checkpoint_validation_metrics": network.validation_metrics,
        **verification,
        "total_parameters": sum(parameter.numel() for parameter in network.parameters()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New JSON results file")
    parser.add_argument(
        "--initial-weights", type=Path,
        help="Verify the layer4 pilot against the original ImageNet ResNet-18 weights",
    )
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise FileExistsError(f"Evaluation output already exists at {args.output}.")
        report = evaluate(args.checkpoint, args.manifest, args.initial_weights)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
    except (csv.Error, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
