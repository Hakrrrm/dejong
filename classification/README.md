# Classification Module

This folder contains runtime-only video analysis tools.

## Script

- `analyze_video.py`
  - Uses `classification_model.pt` by default.
  - Analyzes video in ON/OFF windows (default: 10s ON / 10s OFF).
  - Computes per-class metrics + temporal fire behavior signals.
  - Saves JSON output to `classification/video_metrics.json` by default.

## Example

```bash
python classification/analyze_video.py \
  --video /path/to/video.mp4 \
  --weights classification_model.pt \
  --clip-seconds 10 \
  --break-seconds 10 \
  --sample-fps 2 \
  --conf 0.25
```

## Aggregate relative confidence

The script now outputs `summary.aggregate_relative_confidence` with final normalized confidence for:

- `controlled_fire`
- `fire`
- `smoke`

It also emits a `scoring_formula` section in JSON with the exact equations used.

## Highest fire-score frame logging

The highest-risk sampled frame is saved to `classification/top_fire_frame.jpg` by default
(or custom `--top-fire-frame-out`).

The JSON also includes:

- `summary.top_fire_frame.path`
- `summary.top_fire_frame.timestamp_s`
- `summary.top_fire_frame.fire_frame_score`
