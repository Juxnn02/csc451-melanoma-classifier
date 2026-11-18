# Data collection and review

The collection contains **4,000 candidate dermoscopic images**, including 20 pilot images. Each of the five classes has 800 candidates. The archive supplies the clinical diagnoses. The instructor's approval of the source and the team's independent label and inclusion review are pending.

Keep the image files, patient linkage, private manifests, and checkpoints in team-only storage. The [project README](../README.md) links the team's sample folder. Use a separate private folder for the full collection and model weights.

## Source and provenance

The images came from individual records in the [ISIC Archive](https://registry.opendata.aws/isic-archive/). The 20-image pilot used the [Hospital Italiano de Buenos Aires collection](https://doi.org/10.34970/587329). The remaining images were selected against the [March 9, 2026 archive metadata snapshot](https://isic-archive.s3.us-east-1.amazonaws.com/snapshots/ISIC_metadata.parquet) and retrieved as full-size ISIC JPEGs on November 17. The file hashes are in the private manifest.

The manifest records the ISIC ID, acquisition type, archive diagnosis and confirmation method, patient and lesion linkage, license and attribution, original URL, dimensions, size, and file and decoded pixel hashes. The data audit found 1,722 records with unresolved patient IDs. The archive records describe the source diagnoses and per-image Creative Commons terms. The [HIBA study](https://pmc.ncbi.nlm.nih.gov/articles/PMC10584927/) reports an ethics board waiver of individual consent for its de-identified release. Course acceptance remains pending.

The broad archive includes images that appear in curated releases. The ID audit found 435 images in the [MILK10k release](https://api.isic-archive.com/doi/milk10k/) and zero matching IDs in its [separate benchmark test set](https://api.isic-archive.com/doi/milk10k-benchmark/). The engineering pilot excluded MILK study team records. The overlap audit covered those two releases.

## Candidate label mapping

Labels were proposed from the recorded diagnosis and histopathology confirmation. The team still needs to verify inclusion and mapping independently on the same 100 records, calculate agreement, and document how disagreements are resolved. The pilot used the following exact archive terms.

| Project class | Archive diagnosis |
| --- | --- |
| Melanoma | Melanoma Invasive, Melanoma, NOS |
| Basal cell carcinoma | Basal cell carcinoma |
| Squamous cell carcinoma | Squamous cell carcinoma, Invasive or NOS |
| Benign melanocytic nevus | Nevus |
| Benign keratosis | Seborrheic keratosis |

Actinic keratosis, SCC in situ, keratoacanthoma-type SCC, and conflicting lesion diagnoses or patient links were excluded from the expansion.

## Collection counts

| Stage | Records |
| --- | ---: |
| Archive snapshot rows screened | 549,590 |
| Dermoscopic records | 121,744 |
| Records with a mapped diagnosis | 70,373 |
| Histopathology and expected benign or malignant status | 32,565 |
| Lesion ID, usable license, attribution, and subtype scope | 27,045 |
| Selected and verified candidates, including the pilot | 4,000 |

Selection used one image per recorded lesion, favored records with patient linkage, and checked JPEG decoding, dimensions, file hashes, and decoded pixel hashes. Four file-level exclusions occurred during the download checks. Ruler marks, ink, hair, and scope borders appeared in a thumbnail spot check. Near-duplicate review, image quality decisions, and the final cleaning counts are still pending.

## Restore the private collection

Get the full collection folder or its private `manifest.csv` from the team. With Python 3.10 or newer, run from the project folder containing `app/` and `data/`.

```powershell
python data/restore_collection.py --manifest "C:\path\to\private\manifest.csv" --output "C:\path\to\private\isic-dermoscopy-4000"
```

The restore script downloads the recorded ISIC JPEGs to `images/`, verifies byte counts and SHA-256 hashes, and copies the private CSV as `manifest.csv`. It skips existing files that match the manifest. Allow about 2.48 GB for the full JPEG collection. Use the [Docker app](../app/README.md) to audit the restored folder.

## Splits and preprocessing

The engineering pilot used 250 images, one view per known patient, with 160 training, 50 validation, and 40 held out. Each class has 32 training, 10 validation, and 8 held-out images. The [pilot report](../docs/pilot-results.md) documents the observed results and limits.

For the later course experiment, assign each patient and related duplicate group to one split before fitting scalers or balancing the training set. Resolve missing patient linkage before claiming patient disjointness. The trainer and app share ImageNet preprocessing with a 224 × 224 center crop. If needed for Colab storage, keep smaller processed copies in the private working folder and preserve the archive originals.
