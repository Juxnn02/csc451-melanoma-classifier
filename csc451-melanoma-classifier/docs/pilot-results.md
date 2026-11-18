# CSC 451 ResNet-18 pilot results

This engineering pilot used the diagnoses supplied by the source archive. Instructor approval for the source and team label review are pending. The model is for testing the course application.

## Data and training

The local pilot included 250 dermoscopic images from 250 distinct patient groups and lesions. The patient-disjoint splits contained 160 training images, 50 validation images, and 40 held-out images. Each split included all five diagnostic classes, with 32 training, 10 validation, and 8 held-out images per class. The pilot used CC BY and CC0 records and excluded images attributed to the MILK study team.

The run started with ImageNet weights and fine-tuned ResNet-18's final residual stage and classifier head for three CPU epochs. It used AdamW with a learning rate of 0.0001, cross-entropy loss, a batch size of 16, and selection and training seeds of 451. The run enabled training for 8,396,293 of 11,179,077 parameters. Evaluation verified that 90 earlier tensors matched the initialization. The trainer selected epoch 2 by validation macro-F1, with melanoma recall as the tie-breaker.

## Validation comparison

| Model | Accuracy | Macro-F1 | Melanoma recall |
| --- | ---: | ---: | ---: |
| Majority class | 20% | 0.067 | 0% |
| Logistic regression on 32 × 32 RGB pixels | 40% | 0.414 | 20% |
| ResNet-18, epoch 2 | 54% | 0.525 | 60% |

### Learning history

| Epoch | Training loss | Validation loss | Validation accuracy | Validation macro-F1 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.650 | 1.406 | 36% | 0.357 |
| 2 | 0.712 | 1.305 | 54% | 0.525 |
| 3 | 0.414 | 1.254 | 44% | 0.439 |

## Engineering holdout

The selected checkpoint achieved **55% top-1 accuracy (22/40)**, **90% top-3 accuracy (36/40)**, and **0.506 macro-F1**. A separate browser check uploaded a held-out JPEG to the local Docker app and verified that the mounted checkpoint returned three model scores.

| Source class | Correct top-1 predictions | Recall |
| --- | ---: | ---: |
| Melanoma | 6/8 | 75.0% |
| Basal cell carcinoma | 1/8 | 12.5% |
| Squamous cell carcinoma | 8/8 | 100.0% |
| Benign melanocytic nevus | 5/8 | 62.5% |
| Benign keratosis | 2/8 | 25.0% |

With eight held-out images per class, one prediction changes that class’s recall by 12.5 percentage points. Hospital Italiano de Buenos Aires supplied 48 of the 50 squamous cell carcinoma images, so this within-archive evaluation leaves cross-site generalization untested. The balanced class counts mean aggregate accuracy describes that class mix. Clinical performance remains unmeasured.

See the [Docker app guide](../app/README.md) for running training and loading the checkpoint. The [aggregate metrics](pilot-metrics.json) include the full learning history and confusion matrix. Its rows are source classes and columns are predicted classes, both in the listed class order.

Checkpoint SHA-256 `a0903767aa550d24b9cf2602e8b551edc9c9440991628f0af75a7eb3cc5a67df`.
