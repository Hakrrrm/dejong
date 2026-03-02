# Fire Detection Project

This repo has two parts:

- `model_training/` → train/fine-tune YOLO for `controlled_fire`, `fire`, `smoke`
- `classification/` → run-time video risk classification

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Classification quick start (single-window)

Each command run analyzes **one selected time window** and outputs **one JSON + one JPG**.

```bash
python classification/analyze_video.py \
  --video /path/to/video.mp4 \
  --weights classification_model.pt \
  --start-seconds 0 \
  --analyze-seconds 10 \
  --sample-fps 2 \
  --conf 0.25 \
  --results-dir results \
  --run-label incident_cam01_0000_0010 \
  --camera-id cam_01 \
  --location-type warehouse
```

Outputs:

- `results/<run-label>/metrics.json`
- `results/<run-label>/top_fire.jpg`

For full flag/docs, see `classification/README.md`.

## OpenAI setup (optional)

```bash
cp .env.example .env
# fallback if dotfiles are hidden: cp env.example .env
```

Set key in `.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

If no key is provided, classifier still runs in local-only mode.
Use `--demo-mode` to force local-only.

## Training docs

All model training/fine-tuning documentation remains in:

- `model_training/README.md`


## Image-model comparison quick test

To compare `classification_model.pt` vs `best_nano_111.pt` on a sample image folder:

```bash
python classification/compare_models_on_images.py \
  --source-dir /path/to/sample_images \
  --improved-weights classification_model.pt \
  --baseline-weights best_nano_111.pt \
  --output-dir test_results
```

This writes annotated outputs per model under `test_results/` with filenames that include the model name.
