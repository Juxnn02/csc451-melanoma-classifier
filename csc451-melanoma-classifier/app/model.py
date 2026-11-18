"""Shared preprocessing and checkpoint format for the lesion classifier."""

import io
import pickle
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from labels import CLASS_NAMES

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 32_000_000
FORMAT_VERSION = 1
_transform = ResNet18_Weights.DEFAULT.transforms()


def decode_image(payload: bytes) -> Image.Image:
    """Decode a single JPEG or PNG without trusting the filename or media type."""
    if not payload or len(payload) > MAX_UPLOAD_BYTES:
        raise ValueError("Choose a JPEG or PNG image no larger than 20 MiB.")
    try:
        with Image.open(io.BytesIO(payload), formats=("JPEG", "PNG")) as source:
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise ValueError("Choose an image with at most 32 million pixels.")
            if getattr(source, "is_animated", False):
                raise ValueError("Choose a single still image.")
            return ImageOps.exif_transpose(source).convert("RGB")
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ValueError("Choose an image with at most 32 million pixels.") from error
    except (OSError, SyntaxError) as error:
        raise ValueError("The file could not be read as a JPEG or PNG image.") from error


def preprocess(image: Image.Image) -> torch.Tensor:
    """Resize and normalize RGB pixels using the same transform as training."""
    return _transform(image.convert("RGB"))


def build_model(*, pretrained: bool = False) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model


def _validate_metrics(metrics: object) -> dict[str, float]:
    if not isinstance(metrics, dict) or any(
        not isinstance(name, str)
        or type(value) not in (int, float)
        or not 0 <= value <= 1
        for name, value in metrics.items()
    ):
        raise ValueError("Validation metrics must be named scores between 0 and 1.")
    return {name: float(value) for name, value in metrics.items()}


def make_checkpoint(
    model: nn.Module, *, validation_metrics: dict[str, float] | None = None
) -> dict:
    """Build the checkpoint dictionary for torch.save after a training run."""
    checkpoint = {
        "format_version": FORMAT_VERSION,
        "class_names": list(CLASS_NAMES),
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
    }
    if validation_metrics is not None:
        checkpoint["validation_metrics"] = _validate_metrics(validation_metrics)
    return checkpoint


def load_model(path: str | Path) -> nn.Module:
    """Load a local checkpoint supplied by the project team for CPU inference."""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except FileNotFoundError:
        raise
    except (OSError, EOFError, RuntimeError, pickle.UnpicklingError) as error:
        raise ValueError("The file could not be read as a model checkpoint.") from error

    if (
        not isinstance(checkpoint, dict)
        or type(checkpoint.get("format_version")) is not int
        or checkpoint["format_version"] != FORMAT_VERSION
        or checkpoint.get("class_names") != list(CLASS_NAMES)
    ):
        raise ValueError("The checkpoint version or class order does not match this app.")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict) or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in state_dict.items()
    ):
        raise ValueError("The checkpoint must contain a state_dict of model tensors.")

    metrics = _validate_metrics(checkpoint.get("validation_metrics", {}))
    model = build_model()
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise ValueError("The checkpoint tensor shapes do not match the five-class ResNet-18.") from error
    model.validation_metrics = metrics
    return model.eval()


@torch.inference_mode()
def predict(model: nn.Module, image: Image.Image, top_k: int = 3) -> list[tuple[str, float]]:
    """Return the most likely classes and their scores in descending order."""
    if not 1 <= top_k <= len(CLASS_NAMES):
        raise ValueError("Choose between one and five classes to display.")
    logits = model(preprocess(image).unsqueeze(0))
    if logits.shape != (1, len(CLASS_NAMES)) or not torch.isfinite(logits).all():
        raise ValueError("The model produced invalid scores for this image.")
    scores, indices = logits[0].softmax(dim=0).topk(top_k)
    return [(CLASS_NAMES[index], score) for score, index in zip(scores.tolist(), indices.tolist())]
