# Classification Module

This folder contains runtime-only video analysis tools.

## Script

- `analyze_video.py`
  - Uses `classification_model.pt` by default.
  - Analyzes video in ON/OFF windows (default: 10s ON / 10s OFF).
  - Writes compact aggregate JSON only (no per-box timestamp dump).
  - Saves:
    - one overall JSON (`classification/video_metrics.json` by default)
    - one compact JSON per active interval (`classification/interval_metrics/`)
    - one highest fire-score frame JPG per interval (`classification/interval_top_fire_frames/`)
    - one highest fire-score frame JPG across the full run (`classification/top_fire_frame.jpg`)

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

## Clear interval labeling and JSON/JPG matching

For each interval, a shared base name is used in both files:

- JSON: `classification/interval_metrics/incident_interval_0001_000000s_000010s.json`
- JPG: `classification/interval_top_fire_frames/incident_interval_0001_000000s_000010s_top_fire.jpg`

This guarantees that each interval JSON maps clearly to its interval top-fire frame image.

Each interval JSON also contains:

- `interval.label`
- `interval.base_name`
- `summary.top_fire_frame.path`

## JSON content policy (compact)

### Overall JSON

Contains only:

- `input`
- `sampling`
- `summary` (aggregate run metrics)
- `scoring_formula`
- `interval_outputs` (pointers to each interval JSON/JPG + interval aggregate confidence)

### Per-interval JSON

Contains only:

- `interval` metadata (label/index/timing/base_name)
- `sampling` config
- `summary` aggregate metrics:
  - counts, mean/max confidence
  - flicker/spread
  - aggregate relative confidence
  - interval top-fire-frame info

No per-box detection arrays are written.

## Rebalanced confidence math

The confidence model is rebalanced so real fires are less likely to be over-labeled as `controlled_fire`.

- `controlled_raw = (0.45*mean_controlled + 0.15*(1-spread_n) + 0.10*(1-flicker_n) + 0.30*(1-mean_fire))*(1-0.35*mean_smoke)`
- `fire_raw = (0.60*mean_fire + 0.20*spread_n + 0.15*flicker_n + 0.05*mean_smoke)*(1-0.15*mean_controlled)`
- `smoke_raw = 0.75*mean_smoke + 0.20*mean_fire + 0.05*flicker_n`
- `final[class] = raw[class] / (controlled_raw + fire_raw + smoke_raw)`

with:

- `spread_n = clamp(fire_spread_score * 3)`
- `flicker_n = clamp(fire_flicker_score * 4)`

## Frame-level fire score math

- `fire_frame_score = max(fire_conf*(0.7 + 0.3*clamp(area_ratio*5))) - 0.20*max(controlled_fire_conf)`
