import os
from pathlib import Path

import streamlit as st

from dataset import audit_dataset
from model import decode_image, load_model, predict


@st.cache_resource(show_spinner="Loading model")
def cached_model(path, file_version):
    """Cache one CPU model per checkpoint version."""
    return load_model(path)


def render_prediction():
    checkpoint = Path(os.environ.get("CSC451_MODEL_PATH", "/models/resnet18.pt"))

    st.header("Test an image")
    st.write("Upload a dermoscopic JPEG or PNG up to 20 MB.")
    model = None
    try:
        if checkpoint.is_file():
            stat = checkpoint.stat()
            model = cached_model(str(checkpoint), (stat.st_mtime_ns, stat.st_size))
        else:
            st.info("Mount a trained checkpoint to see predictions. You can still validate an image here.")
    except (OSError, ValueError):
        st.error("The checkpoint could not be loaded. Check that it came from the CSC 451 training command.")

    with st.form("image_input"):
        upload = st.file_uploader("Dermoscopy image", type=["jpg", "jpeg", "png"])
        submitted = st.form_submit_button("Classify image" if model is not None else "Check image",
                                          type="primary")

    if submitted and upload is None:
        st.warning("Choose an image first.")
    if submitted and upload is not None:
        try:
            image = decode_image(upload.getvalue())
        except ValueError as error:
            st.error(str(error))
        else:
            left, right = st.columns(2, gap="large")
            with left:
                st.image(image, caption=f"{image.width:,} × {image.height:,} pixels", width="stretch")
            with right:
                if model is None:
                    st.success("The image opened successfully.")
                else:
                    st.subheader("Top 3 model scores")
                    try:
                        scores = predict(model, image)
                    except (RuntimeError, ValueError):
                        st.error("The model could not score this image.")
                    else:
                        for label, score in scores:
                            st.progress(score, text=f"{label} · {score:.1%}")

    with st.expander("About this model"):
        st.write("The team’s five-class model uses ResNet-18 with 224 × 224 inputs. Image quality "
                 "and differences between cameras or skin tones can affect how it performs.")
        if model is None:
            st.caption("Evaluation scores will appear when a checkpoint with validation metrics is mounted.")
        else:
            metrics = getattr(model, "validation_metrics", {}) or {}
            for name, label in (
                ("accuracy", "Validation accuracy"),
                ("macro_f1", "Validation macro-F1"),
                ("melanoma_recall", "Validation melanoma recall"),
            ):
                if name in metrics:
                    st.write(f"**{label}** {metrics[name]:.1%}")


def render_dataset_audit():
    dataset_dir = Path(os.environ.get("CSC451_DATA_DIR", "/data"))

    st.header("Dataset audit")
    st.write("Check the candidate manifest, image paths, review status, and patient linkage.")
    try:
        audit = audit_dataset(dataset_dir)
    except OSError:
        st.info("Mount the collection folder containing manifest.csv and images/ to run the audit.")
        return
    except ValueError as error:
        st.error(f"Could not read the manifest. {error}")
        return

    left, middle, right = st.columns(3)
    left.metric("Manifest rows", f"{audit.total_rows:,}")
    middle.metric("Pending review", f"{audit.pending_review:,}")
    right.metric("Missing patient linkage", f"{audit.missing_patient_linkage:,}")
    st.table([{"Proposed class": label, "Records": count} for label, count in audit.class_counts.items()])
    st.caption(f"{audit.approved_rows:,} approved · {audit.rejected_rows:,} rejected · "
               f"{audit.missing_files:,} missing files · {audit.invalid_rows:,} invalid rows")
    if audit.issues:
        with st.expander("Sample manifest issues"):
            for issue in audit.issues:
                st.text(issue)


def main():
    st.set_page_config(page_title="CSC 451 Skin Lesion Classifier", layout="centered")
    st.title("Skin lesion classifier")
    st.caption("CSC 451 · Educational use only. Model scores cannot diagnose or rule out cancer.")
    view = st.radio("Project view", ["Image test", "Dataset audit"], horizontal=True,
                    label_visibility="collapsed")
    if view == "Image test":
        render_prediction()
    else:
        render_dataset_audit()


if __name__ == "__main__":
    main()
