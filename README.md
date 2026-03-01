# Fire Detection Project

This repository has two clearly separated areas:

- `model_training/` → everything related to training and fine-tuning your 3-class model.
- `classification/` → runtime video analysis using your trained model (`classification_model.pt`).

`requirements.txt` stays at repo root so dependencies can evolve in one place.

---

## Repository structure

- `model_training/README.md` — full training/fine-tuning guide.
- `model_training/src/` — training and inference helper scripts.
- `model_training/configs/` — dataset config templates.
- `classification/analyze_video.py` — interval-based video analysis and metrics output.
- `requirements.txt` — shared dependencies.

---

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Video classification module (new)

The new video module is designed for **post-training usage** with your tuned model file:

- expected model path: `classification_model.pt` (repo root), or pass `--weights` explicitly.
- input: video file.
- output: JSON metrics report (default: `classification/video_metrics.json`).

### Why video is handled differently

Compared to a single image, video provides temporal signals that can improve decision quality:

- **Flicker behavior**: abrupt frame-to-frame confidence changes can indicate active flames.
- **Spread behavior**: increasing detected fire area over time can indicate escalation.
- **Sampling windows**: process short clips (e.g., 10s), skip a gap (e.g., 10s), then process again to reduce compute while still tracking trends.

### Implemented interval logic

The script supports this pattern:

- analyze for `clip_seconds` (default 10s),
- skip for `break_seconds` (default 10s),
- repeat until video ends.

Within active windows, frames are sampled at `sample_fps` (default 2 fps).

### Command

```bash
python classification/analyze_video.py \
  --video /path/to/video.mp4 \
  --weights classification_model.pt \
  --clip-seconds 10 \
  --break-seconds 10 \
  --sample-fps 2 \
  --conf 0.25 \
  --json-out classification/video_metrics.json \
  --interval-json-dir classification/interval_metrics \
  --top-fire-frame-out classification/top_fire_frame.jpg
```

### Returned metrics (end of run)

The JSON contains:

- `input`: video/weights/fps/frame size metadata.
- `sampling`: clip/break/sample settings and sampled frame count.
- `summary.counts`: detection counts per class (`controlled_fire`, `fire`, `smoke`).
- `summary.max_confidence`: peak confidence per class.
- `summary.mean_confidence`: average confidence per class.
- `summary.video_behavior_signals.fire_flicker_score`: mean abs delta of consecutive fire confidences.
- `summary.video_behavior_signals.fire_spread_score`: range of normalized fire bbox area over the video.
- `summary.aggregate_relative_confidence`: normalized final confidence for `controlled_fire`, `fire`, and `smoke`.
- `summary.top_fire_frame`: path, timestamp, and computed fire-frame score for the highest-risk sampled frame.
- `scoring_formula`: explicit formulas used for aggregate confidence and frame-level fire score.
- `interval_outputs`: list of per-interval JSON files written during run.
- `detections`: per-detection rows with timestamp, class, confidence, bbox, and area ratio.

These outputs are ready to feed into downstream contextual assessment logic (including OpenAI API decision layers).

---

## Model training docs (kept intact)

All training documentation remains in:

- `model_training/README.md`

Start there for:

- dataset setup,
- fine-tuning from your existing `.pt`,
- running image/realtime model checks,
- and producing `best.pt` (which you can copy/rename to `classification_model.pt` for runtime use).
