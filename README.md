# Tichu Card Detection

Computer-vision pipeline for detecting and classifying the 56 cards in a Tichu
deck.


## Repository Layout

```text
configs/
  classes.names            existing ordered class names
  classes.yaml             canonical class IDs
  label_studio/            card-corner labeling configuration

data/
  raw/
    card_scans/             original scan videos
    card_videos/            original recordings
  external/dtd/             third-party texture images
  interim/                  extracted media, card crops, and caches
  synthetic/                color, mixed, and monochrome scene datasets

scripts/                    existing conversion and generation scripts
notebooks/                  existing exploration and training notebooks
models/pretrained/          generic YOLO initialization weights
models/trained/             weights from the current training runs
reports/training_runs/      metrics and plots from the current training runs
artifacts/test/             existing generated test outputs
```


## Using Label Studio
Label Studio is used to manually annotate datasets. It is installed in a seperate python virtual environment.

#### Install and launch
```powershell
python -m venv .venv-label-studio
.venv-label-studio/Scripts/activate

pip install -U label-studio
label-studio
```

#### Preannotating with a yolo model
```powershell
$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED = "true"
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT = (Resolve-Path "data/raw").Path

python scripts/preannotate_label_studio.py `
  --images "data/raw/<session_id>/images" `
  --weights "models/trained/<run>/weights/best.pt" `
  --output "data/annotations/label_studio/<session_id>/yolo-predictions.json"

$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED = "true"
label-studio
```

In Label Studio:

1. Open Settings → Cloud Storage.
2. Select Add Source Storage → Local Files.
3. Use:

Storage title: