# 3-Class Fire/Smoke Detection (YOLO11)

This project adapts the original fire/smoke detector to **3 classes**:

- `controlled_fire`
- `fire`
- `smoke`

The defaults are tuned for small datasets (~500 augmented images) using transfer learning.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## What is left for you to do (from your YOLOv8 dataset)

If your dataset is already in YOLO format, you only need to do these steps:

1. Put your dataset in this structure:

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

2. Ensure class IDs are exactly:
   - `0 = controlled_fire`
   - `1 = fire`
   - `2 = smoke`

3. Edit `configs/dataset_3class.yaml` and set `path:` to your `dataset_root`.

4. Run dataset validation:

```bash
python src/check_dataset.py --dataset-root /absolute/path/to/dataset_root
```

5. Start training.

## Train (new 3-class model)

```bash
python src/train_three_class.py \
  --data configs/dataset_3class.yaml \
  --model yolo11n.pt \
  --epochs 120 \
  --batch 8 \
  --imgsz 640
```

Outputs are saved to:

```text
runs/fire3class/yolo11n_transfer/
```

Best model path:

```text
runs/fire3class/yolo11n_transfer/weights/best.pt
```

## How to call the tuned model later

### 1) Batch/file inference (save predictions)

```bash
python src/predict_three_class.py \
  --weights runs/fire3class/yolo11n_transfer/weights/best.pt \
  --source /path/to/image_or_video_or_folder \
  --conf 0.25
```

### 2) Realtime mode (close to original repo style)

```bash
python src/realtime_infer.py \
  --weights runs/fire3class/yolo11n_transfer/weights/best.pt \
  --source 0 \
  --conf 0.25
```

- `--source 0` = default webcam.
- You can also pass a video path or RTSP/HTTP stream URL.

### 3) Ultralytics CLI (if you prefer original-style command usage)

```bash
yolo predict \
  model=runs/fire3class/yolo11n_transfer/weights/best.pt \
  source=/path/to/image_or_video_or_folder \
  conf=0.25
```

## Notes for small datasets

- Keep validation clean and representative.
- If overfitting appears, increase `--freeze` (e.g., 15), reduce epochs, or switch to `yolo11n.pt` if using larger checkpoints.
- Improve class balance, especially for underrepresented classes (especially `controlled_fire`).
