# Project Layout

Training-related files now live **only** under `model_training/`.

## Where things are

- Training guide: `model_training/README.md`
- Training code: `model_training/src/`
- Dataset config: `model_training/configs/`
- Python dependencies (shared): `requirements.txt` (repo root)

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then follow the full workflow here:

- `model_training/README.md`

## Common commands (from repo root)

```bash
python model_training/src/train_three_class.py --data model_training/configs/dataset_3class.yaml --model yolo11n.pt
python model_training/src/predict_three_class.py --weights runs/fire3class/yolo11n_transfer/weights/best.pt --source /path/to/image.jpg
python model_training/src/realtime_infer.py --weights runs/fire3class/yolo11n_transfer/weights/best.pt --source 0
```
