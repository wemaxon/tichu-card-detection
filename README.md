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
