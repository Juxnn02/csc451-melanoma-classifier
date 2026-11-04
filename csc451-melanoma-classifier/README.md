# Skin Lesion Classification (CSC 451 Final Project)

> **Educational project, not a medical device.** The model classifies a single dermoscopic image into one of five
> categories for learning and research. It does not diagnose a person, and it cannot rule out cancer.
> Anyone worried about a skin spot should see a dermatologist.

**Team:** Denis Borta, Juan Capelo, Tavoy Arnett · CSC 451 Fundamentals of Deep Learning, Fall 2026

| Status | Milestone |
|---|---|
| ☐ | Proposal, repo, 20 raw samples, pilot timing (Wed 11/04) |
| ☐ | Progress presentation (Wed 11/18) |
| ☐ | Stop training (11/30) |
| ☐ | Final presentation + live demo (Mon 12/7) |

## Problem and intended user
An educational dashboard for students and researchers: upload one **dermoscopic** lesion image and see the model's
top-3 scores over five categories. We emphasize **melanoma recall** (a missed melanoma is the costly error),
honest comparison with simple baselines, and evaluation across the skin-tone groups that the data supports.
Ordinary phone photos are outside the initial scope.

| Class | Label group | Planned images |
|---|---|---|
| Melanoma | Malignant | 300–500 |
| Basal cell carcinoma | Malignant | 300–500 |
| Squamous cell carcinoma | Malignant | 300–500 |
| Benign melanocytic nevus | Benign mole | 300–500 |
| Benign keratosis | Benign lesion | 300–500 |

Actinic keratosis and mixed or uncertain diagnoses are excluded, not relabeled as benign.

## Data
Full details, the cleaning log and the label-mapping rules are in [`data/README.md`](data/README.md).

- **Source:** individually selected [ISIC Archive] dermoscopic images only.
- **Raw samples:** 20 raw samples (4 per class) are in a private Google Drive folder: https://drive.google.com/drive/folders/1CiY8FonLVLWGpRWi8mduOkxCdu79h6dy?usp=drive_link.
  Images are **not** committed to this public repo.

## Model
- **Baselines:** (1) majority class; (2) logistic regression on 32×32 RGB pixels, scaling fitted on training data only.
- **Neural network:** ImageNet-pretrained ResNet-18 with a 5-output head (11,179,077 parameters), 224×224 inputs.
  Start with a frozen backbone, then test fine-tuning the last block (`layer4`).
- **Why:** small enough for free Colab, strong with a few hundred images per class (transfer learning).

## Training setup
- Free Colab T4 (Kaggle as backup); checkpoints saved to Drive every epoch.
- Batch size 32, AdamW, cross-entropy, up to 10 epochs with validation-based early stopping, mixed precision.
- **Pilot timing (11/04):** _[paste from `notebooks/00_pilot_timing.ipynb`]_


## Experiment log
One change at a time, picked on the validation set (macro-F1 and melanoma recall).

| Run | What changed | Key settings | Best val macro-F1 | Val melanoma recall | Epochs | Train time |
|---|---|---|---|---|---|---|
| R0 | Baseline: majority class | — | | | — | — |
| R1 | Baseline: logistic regression, 32×32 | | | | — | |
| R2 | ResNet-18, frozen backbone | lr 1e-4, bs 32 | | | | |
| R3 | | | | | | |

## Results
_Test set used once, at the end._
- Test metrics vs. both baselines: accuracy, per-class precision/recall/F1, macro-F1: _[fill in]_
- Confusion matrix: _[figure]_
- Learning curves (train vs. validation loss): _[figure]_
- Final configuration, 3 seeds: macro-F1 _mean ± std_, melanoma recall _mean ± std_
- Skin-tone subgroups (where reliable labels exist): group sizes, support, missingness, metrics with uncertainty
- Fresh-data test (30+ new examples collected after training): _[fill in, explain any drop]_

## Error analysis and limitations
_[10–20 misclassified images, especially missed melanomas, and the patterns behind them]_

Known limitations to address: dermoscopic images only; skin-tone coverage in public archives is uneven, so sparse
groups cannot establish equal performance; scores are not personal cancer probabilities.

## How to run
```bash
pip install -r requirements.txt
# pilot / data: open notebooks/ in Google Colab with a T4 runtime and Run all
streamlit run app/app.py   # after training (app not built yet)
```

## Deployed app (optional bonus)
_[Cloud Run URL]_


## AI tool use
Per the course policy, every AI tool and what it was used for:
- Claude: drafted the repository starter (folder layout, README skeleton, requirements, `.gitignore`) and the pilot
  notebook that pulls the raw samples and times one epoch. The team reviewed, ran and is responsible for all of it.
- _[add others]_
