# Label Studio

Label Studio project configurations belong here. Treat the native Label Studio
JSON export as the annotation source of truth.

Recommended paths:

```text
data/raw/<session_id>/videos/
data/annotations/<project>/label_studio/
data/processed/<dataset_version>/
```

For detector/classifier training, select representative frames from each video
and label the complete rank-and-suit corner unit. Preserve readability and
occlusion attributes in the native export, then generate YOLO labels and
classifier crops with scripts.
