# Classification Module

This folder contains runtime-only video analysis tools.

## Script

- `analyze_video.py`
  - Uses `classification_model.pt` by default.
  - Analyzes video in ON/OFF windows (default: 10s ON / 10s OFF).
  - Computes per-class metrics + temporal fire behavior signals.
  - Saves:
    - one overall JSON (`classification/video_metrics.json` by default)
    - one JSON per active interval (`classification/interval_metrics/` by default)
    - one highest fire-score frame image per interval (`classification/interval_top_fire_frames/` by default)
    - one highest fire-score frame image across the full run (`classification/top_fire_frame.jpg` by default)

## Test with a video

```bash
python classification/analyze_video.py \
  --video /path/to/test_video.mp4 \
  --weights classification_model.pt \
  --clip-seconds 10 \
  --break-seconds 10 \
  --sample-fps 2 \
  --conf 0.25 \
  --json-out classification/video_metrics.json \
  --interval-json-dir classification/interval_metrics \
  --interval-top-frame-dir classification/interval_top_fire_frames \
  --top-fire-frame-out classification/top_fire_frame.jpg
```

After run:

- Open `classification/video_metrics.json` for run-wide aggregates.
- Open `classification/interval_metrics/incident_interval_XXXX_*.json` for each active clip.
- Open `classification/interval_top_fire_frames/incident_interval_XXXX_*_top_fire.jpg` for each interval's top fire-score frame.
- Open `classification/top_fire_frame.jpg` for the highest computed fire-score frame across the full run.

## Interval labeling

Intervals are labeled from incident start time:

- first analyzed 10s clip => `interval_0001` (`000000s` to `000010s`)
- second analyzed 10s clip => `interval_0002` (`000020s` to `000030s`, with default 10s break)
- third analyzed 10s clip => `interval_0003`, and so on.

## Aggregate relative confidence math

Per interval (and also overall), final confidence is calculated for:

- `controlled_fire`
- `fire`
- `smoke`

### Inputs

- `mean_controlled`, `mean_fire`, `mean_smoke` = mean detection confidence by class.
- `fire_spread_score` = `max(fire_bbox_area_ratio) - min(fire_bbox_area_ratio)`.
- `fire_flicker_score` = mean absolute delta between consecutive fire confidences.

### Raw scores

- `controlled_raw = 0.85*mean_controlled + 0.10*(1-clamp(fire_spread*3)) + 0.05*(1-clamp(fire_flicker*4))`
- `fire_raw = (0.70*mean_fire + 0.20*clamp(fire_spread*3) + 0.10*clamp(fire_flicker*4))*(1-0.25*mean_controlled)`
- `smoke_raw = 0.85*mean_smoke + 0.15*mean_fire`

### Normalization

- `final[class] = raw[class] / (controlled_raw + fire_raw + smoke_raw)`

These values are saved under `summary.aggregate_relative_confidence`.

## Frame-level fire score math

For each sampled frame, fire severity score is:

- `fire_frame_score = max(fire_conf*(0.6 + 0.4*clamp(area_ratio*5))) - 0.35*max(controlled_fire_conf)`

The script saves the frame with highest `fire_frame_score` and reports:

- `summary.top_fire_frame.path`
- `summary.top_fire_frame.timestamp_s`
- `summary.top_fire_frame.fire_frame_score`


## Per-interval top fire frame in JSON

Each interval JSON now includes:

- `summary.top_fire_frame.path`
- `summary.top_fire_frame.timestamp_s`
- `summary.top_fire_frame.fire_frame_score`

The overall JSON (`video_metrics.json`) also includes these values in `interval_outputs[]`.
