# Multimodal Deep Learning for Archaeological Artifact Identification

Deep Learning (22AIE304) course project by Harsitha Senthilkumar, K Shanjana Priya and Varshni Kannan.

Given a museum artifact's **image** and **catalogue metadata**, the system:

1. classifies it into one of 6 known categories (bowl, vase, figure, coin, lamp, cylinder seal), or flags it as **Unknown** if it belongs to a category never seen in training;
2. retrieves the **Top-5 most similar known artifacts** with FAISS as supporting evidence.

Data: [The Metropolitan Museum of Art Open Access](https://github.com/metmuseum/openaccess) (public-domain records from archaeological departments).

## Pipeline

| Step | Script | What it does |
| --- | --- | --- |
| 1 | `inspect_met.py` | Reads MetObjects.csv, keeps archaeological departments, lists candidate categories |
| 2 | `check_images.py` | Samples each category through the MET API to check image availability |
| 3 | `build_dataset.py` | Builds a leakage-safe split (70/15/15 per known class; unseen classes only in the unknown test set) with 14 automated checks |
| 4 | `extract_features.py` | Frozen ViT-B/16 (ImageNet) gives a 768-d feature per image |
| 5 | `metadata_features.py` | Encodes Department, Culture, Period, dates and Medium into 125 features (fitted on train only) |
| 6 | `train.py` | Trains the image-only, metadata-only and fused models (`models.py`) |
| 7 | `open_set.py` | Mahalanobis-distance Known/Unknown detection (threshold from validation) + softmax baseline |
| 8 | `retrieval.py` | FAISS cosine-similarity Top-K search over training embeddings |
| 9 | `evaluate.py` | Final table, 95% bootstrap confidence intervals, paired differences, plots |
| 10 | `demo.py` | Runs the full system on one image or MET object |

## Results (test set: 360 known + 450 unknown artifacts)

| Model | Known accuracy | Open-set AUROC (Mahalanobis) | Softmax AUROC |
| --- | --- | --- | --- |
| Image only | 0.947 | 0.716 | 0.867 |
| Metadata only | 0.806 | 0.709 | 0.733 |
| Image + Metadata | 0.950 | 0.777 | 0.863 |

- Fusion improves unknown detection over image only by +0.061 AUROC (95% CI 0.044 to 0.077).
- FAISS retrieval: Precision@1 = 0.956, Precision@5 = 0.948.
- Limitation: unseen jugs are hard to separate from vases (per-category AUROC 0.46).

## Setup

```bash
pip install -r requirements.txt
```

`data/MetObjects.csv` (~300 MB) and the downloaded images are not included because of their size. `inspect_met.py` downloads the CSV automatically. The split files in `data/` record exactly which objects were used, and the extracted features, trained models and all results are included, so `open_set.py` and `evaluate.py` run without the images. Re-download the images (via `build_dataset.py`) before running `extract_features.py`, `retrieval.py` or `demo.py`.

## Run

```bash
python extract_features.py
python metadata_features.py
python train.py --mode image
python train.py --mode metadata
python train.py --mode multimodal
python open_set.py
python retrieval.py
python evaluate.py
python demo.py --sample unknown_test
python demo.py --image path/to/photo.jpg
```
