# Data

Image data is **not** stored in this public repository. It lives in the team's private Google Drive
folder (`MyDrive/CSC451_project/`). This file documents where it came from and every step that changed it.

## Source
- **ISIC Archive** (International Skin Imaging Collaboration), https://www.isic-archive.com, public API
  `https://api.isic-archive.com/api/v2/`. Individually selected records, **dermoscopic** images only.
- We do **not** use the archive's challenge collections or any pre-made train/test split; we make our own split.

## Fields recorded for every image
image ID, source, retrieval date, license, attribution, diagnosis (`diagnosis_1`/`_2`/`_3`) and its evidence
(`diagnosis_confirm_type`), patient ID, lesion ID, modality (`image_type`), skin-tone annotation where available
(`fitzpatrick_skin_type`), body site, original width/height.

## Label mapping (most specific diagnosis available, first matching rule wins)
| Class | Rule on the archive's diagnosis text |
|---|---|
| melanoma | contains "melanoma" |
| bcc | contains "basal cell" |
| scc | contains "squamous cell" |
| nevus | contains "nevus"/"nevi" |
| benign_keratosis | seborrheic keratosis, solar lentigo, lichen planus-like keratosis, pigmented benign keratosis |
| **excluded** | actinic keratosis, indeterminate, uncertain, atypical, collision lesions |

**Agreement check:** two members independently review the label mapping and inclusion decision for the same
100 records. Agreement: _[x / 100]_. Disagreements and how they were resolved: _[fill in]_

## Cleaning log (every exclusion has a reason and a count)
| Step | Removed | Remaining | Reason |
|---|---|---|---|
| Records retrieved | — | | |
| Missing or conflicting diagnosis | | | |
| Excluded diagnoses (actinic, uncertain…) | | | |
| Not dermoscopic | | | |
| Corrupt or unreadable files | | | |
| Exact and near duplicates | | | |
| Unusable images (blur, non-lesion…) | | | |
| Class caps / balancing (training split only) | | | |
| **Final** | | | |

## Final class counts
| Class | Train | Val | Test | Total |
|---|---|---|---|---|
| melanoma | | | | |
| bcc | | | | |
| scc | | | | |
| nevus | | | | |
| benign_keratosis | | | | |

## Split (made before any training, scaling or balancing)
70/15/15 by **patient**: all images of one patient stay in one split. If a patient ID is missing, the lesion ID is
used, and the remaining risk is disclosed. Duplicate groups and repeated visits stay together.

## Working set
Every image is center-cropped to square and resized **once** to 224×224 RGB; originals are preserved.
Total working set stays under 1 GB.
