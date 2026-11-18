import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

import model


class ImageTests(unittest.TestCase):
    def test_decode_and_preprocess_png(self):
        buffer = io.BytesIO()
        Image.new("L", (37, 42), 160).save(buffer, format="PNG")

        image = model.decode_image(buffer.getvalue())

        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (37, 42))
        tensor = model.preprocess(image)
        self.assertEqual(tensor.shape, (3, 224, 224))
        self.assertEqual(tensor.dtype, torch.float32)

    def test_reject_unreadable_files_and_apply_decode_limits(self):
        with self.assertRaisesRegex(ValueError, "could not be read"):
            model.decode_image(b"invalid image bytes")

        buffer = io.BytesIO()
        Image.new("RGB", (16, 16)).save(buffer, format="JPEG")
        payload = buffer.getvalue()
        with patch.object(model, "MAX_UPLOAD_BYTES", len(payload) - 1):
            with self.assertRaisesRegex(ValueError, "no larger"):
                model.decode_image(payload)
        with patch.object(model, "MAX_IMAGE_PIXELS", 255):
            with self.assertRaisesRegex(ValueError, "pixels"):
                model.decode_image(payload)


class CheckpointTests(unittest.TestCase):
    def test_round_trip_and_rank_cpu_predictions(self):
        # Fixed logits exercise the serving path without a trained weights file.
        network = model.build_model()
        with torch.no_grad():
            network.fc.weight.zero_()
            network.fc.bias.copy_(torch.tensor([0.0, 4.0, -2.0, 1.0, -1.0]))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.pt"
            torch.save(model.make_checkpoint(network, validation_metrics={"macro_f1": 0.25}), path)
            loaded = model.load_model(path)

        self.assertFalse(loaded.training)
        self.assertEqual(next(loaded.parameters()).device.type, "cpu")
        self.assertEqual(loaded.validation_metrics, {"macro_f1": 0.25})
        results = model.predict(loaded, Image.new("RGB", (80, 50)))
        self.assertEqual([name for name, _ in results], [
            "Basal cell carcinoma", "Benign melanocytic nevus", "Melanoma",
        ])
        self.assertGreater(results[0][1], results[1][1])
        self.assertGreater(results[1][1], results[2][1])

    def test_reject_checkpoint_with_wrong_class_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong-order.pt"
            torch.save({
                "format_version": model.FORMAT_VERSION,
                "class_names": list(reversed(model.CLASS_NAMES)),
                "state_dict": {},
            }, path)
            with self.assertRaisesRegex(ValueError, "class order"):
                model.load_model(path)

    def test_reject_wrong_tensor_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong-shapes.pt"
            torch.save({
                "format_version": model.FORMAT_VERSION,
                "class_names": list(model.CLASS_NAMES),
                "state_dict": {"fc.bias": torch.zeros(4)},
            }, path)
            with self.assertRaisesRegex(ValueError, "tensor shapes"):
                model.load_model(path)


if __name__ == "__main__":
    unittest.main()
