# 3-Class Fire/Smoke Detection (YOLO11)

This repository contains a lightweight training pipeline to adapt a YOLO11 fire/smoke detector to **3 classes**:

- `fire`
- `smoke`
- `controlled_fire`

It is designed for small datasets (like your ~180 originals augmented to ~500 images) by using transfer learning and conservative augmentation.

## 1) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Prepare dataset in YOLO format

Use this structure:

```text
dataset_root/
  images/
    train/
    val/
    test/      # optional
  labels/
    train/
    val/
    test/      # optional
```

Each `labels/*.txt` file uses YOLO annotations:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Class IDs for this project must be:

- `0 = fire`
- `1 = smoke`
- `2 = controlled_fire`

Edit `configs/dataset_3class.yaml` with your real dataset path.

## 3) Train

```bash
python src/train_three_class.py \
  --data configs/dataset_3class.yaml \
  --model yolo11n.pt \
  --epochs 120 \
  --batch 8 \
  --imgsz 640
```

### Why these defaults for a small dataset?

- Starts from pretrained YOLO11 weights (transfer learning).
- Freezes early layers (`--freeze 10`) to reduce overfitting.
- Uses `AdamW` with lower LR and early stopping (`--patience 25`).
- Keeps augmentation moderate (mixup, color jitter, small geometric changes).

If overfitting appears, try:

- smaller model (`yolo11n.pt`),
- stronger freeze (e.g. `--freeze 15`),
- fewer epochs or higher patience,
- cleaner validation split.

## 4) Inference

```bash
python src/predict_three_class.py \
  --weights runs/fire3class/yolo11n_transfer/weights/best.pt \
  --source /path/to/image_or_video \
  --conf 0.25
```

Predictions are saved under `runs/detect/` by default.

## 5) Suggested next improvements

- Add hard-negative samples (non-fire scenes that resemble smoke/flames).
- Keep `controlled_fire` examples diverse (campfires, stoves, industrial flares, etc.).
- Run k-fold cross-validation due to limited data volume.
