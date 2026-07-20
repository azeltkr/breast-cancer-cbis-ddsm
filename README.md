# Deep Learning for Breast Cancer Detection (CBIS-DDSM)

A feasibility prototype for a final-year project on detecting breast cancer in
mammographic images with convolutional neural networks. It implements the first
stages of a deep-learning pipeline on the **CBIS-DDSM** dataset — data preparation,
a baseline CNN, and an initial VGG16 transfer-learning experiment — and evaluates
both with metrics appropriate to a clinical screening task.

Final-year project for CM3070, BSc Computer Science (Machine Learning & AI),
University of London. Template B: Data Science and Machine Learning.

## Problem framing

The task is **binary classification**: mammograms are labelled malignant or
non-malignant (benign and normal cases merged into the negative class). This mirrors
the clinical priority at the screening stage — a **missed malignancy (false negative)
is far more costly than a false alarm** — so the evaluation foregrounds sensitivity
and the false-negative count rather than raw accuracy. Performance is measured against
the per-image **AUC-ROC of 0.88** reported by Shen et al. (2019) on the same dataset.

## Results (prototype)

Evaluated on a held-out test set of 572 images:

| Model | AUC-ROC | Sensitivity | Specificity | False negatives |
|---|---|---|---|---|
| Baseline CNN (from scratch) | **0.610** | 0.658 | 0.537 | 88 / 257 |
| VGG16 (frozen base) | 0.437 | 0.778 | 0.219 | 57 / 257 |

**The headline finding: a small CNN trained from scratch beat a frozen pretrained
VGG16 — which scored *below random* (AUC 0.437).** This is the interesting result, not
a disappointing one:

- VGG16's ImageNet features describe everyday colour photographs; greyscale
  mammograms carry their signal in subtle texture, not bold natural-image shapes.
  Freezing the entire convolutional base forced the model to describe mammograms with
  ill-suited features, and its predictions collapsed toward 0.5 for nearly every image.
- This reproduces, on the project's own data, the caution from Tajbakhsh et al. (2016)
  that **fine-tuning, not feature-freezing, is what unlocks transfer learning** in
  medical imaging.
- Reporting AUC *alongside* sensitivity and specificity earned its keep: VGG16's
  sensitivity of 0.778 looks acceptable in isolation, but its specificity of 0.219 and
  sub-random AUC reveal that "sensitivity" came from a bias toward predicting positive,
  not genuine discrimination. A single headline metric would have hidden this.

The prototype's purpose was feasibility, not hitting the benchmark — and it clearly
points to its own next steps.

## Approach

1. **Data prep** — CBIS-DDSM JPEGs joined to pathology labels via a patient/side/view
   key, yielding 2,857 labelled images (1,575 non-malignant, 1,282 malignant). Decoded,
   resized to 128×128, normalised to [0,1]; corrupt/truncated files skipped at load
   time. Stratified 80/20 split, light flip augmentation, class weighting.
2. **Baseline CNN** — three conv blocks → global average pooling → dropout-regularised
   dense head, trained from scratch to beat the naive AUC = 0.5 floor.
3. **VGG16 transfer** — ImageNet-pretrained VGG16 with the convolutional base frozen,
   topped by the same style of head.
4. **Evaluation** — AUC-ROC (primary), sensitivity, specificity, confusion matrices,
   ROC curves, and training-history plots.

## Next steps (for the full project)

Fine-tune (unfreeze VGG16's upper blocks) rather than freeze · train at 224×224 to
preserve microcalcification detail · more epochs with early stopping · decision-threshold
tuning for the clinical operating point · ResNet50 as a contingency architecture · a
FastAPI endpoint to serve the trained model.

## Dataset

[CBIS-DDSM (Curated Breast Imaging Subset of DDSM)](https://www.cancerimagingarchive.net/collection/cbis-ddsm/),
an open, expert-curated benchmark from The Cancer Imaging Archive, used here via a
public [Kaggle JPEG mirror](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset).
The data is fully anonymised and publicly available under TCIA's Data Usage Policy — no
identifiable patient information is involved. The dataset is **not** included in this repo;
the notebook downloads it at runtime via the Kaggle API (you'll need your own
`kaggle.json` credentials).

## Running

Open the notebook in Google Colab (a GPU runtime is recommended — this was developed on
a T4). The first cell installs the Kaggle CLI and downloads the dataset. Run cells top to
bottom. Two cells that display raw mammogram thumbnails have had their outputs cleared;
the code is intact and will regenerate them on a run.

## Note

This is research-grade experimentation, not a clinical tool. It is not intended for
diagnosing patients, and every result is experimental.

## Techniques

TensorFlow/Keras · `tf.data` pipelines · CNN from scratch · VGG16 transfer learning ·
class weighting for imbalance · data augmentation · AUC-ROC / sensitivity / specificity ·
confusion-matrix and ROC analysis · benchmarking against published literature
