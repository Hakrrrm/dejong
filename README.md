# Fire Detection Project

This repository has two areas:

- `model_training/` → training/fine-tuning workflow for `controlled_fire`, `fire`, `smoke`
- `classification/` → interval-based runtime risk analysis for videos

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## OpenAI setup (optional)

> If `.env.example` is not visible in your GitHub file list, use `env.example` (same content).


OpenAI is optional and only used for uncertain intervals.

1. Copy env template:

```bash
cp .env.example .env
# if dotfiles are hidden in your UI: cp env.example .env
```

2. Set your key in `.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

If key is missing, analyzer runs in local-only mode and records `openai.used: false`.

## Classification command

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
  --interval-top-frame-dir classification/interval_top_fire_frames \
  --top-fire-frame-out classification/top_fire_frame.jpg \
  --timeline-out classification/timeline.json \
  --camera-id cam_01 \
  --location-type warehouse \
  --demo-mode
```

`--demo-mode` forces local-only run (no OpenAI calls).

## What classification outputs

- `classification/video_metrics.json` (overall compact aggregate)
- `classification/timeline.json` (full per-interval decision timeline)
- one compact interval JSON per interval in `classification/interval_metrics/`
- one top-fire JPG per interval in `classification/interval_top_fire_frames/`
- one global top-fire JPG in `classification/top_fire_frame.jpg`

JSON is compact/aggregate-only (no per-box timestamp dump).

### Key per-interval numeric fields

In each interval JSON:

- `summary.aggregate_relative_confidence.{controlled_fire,fire,smoke}`
- `summary.risk_numbers.dangerous_fire_index`
- `summary.risk_numbers.fire_vs_controlled_gap`
- `summary.risk_numbers.fire_to_controlled_ratio`
- `decision.local_score`
- `decision.final_score`
- `decision.decision_confidence`
- `decision.scenario_rank`

## Timeline summary fields

`timeline.json` includes top-level:

- `summary.max_risk`
- `summary.time_to_first_escalation`
- `summary.scenario_counts`

## Model training docs

Training docs stay in:

- `model_training/README.md`


## OpenAI model switch behavior

Configured in `classification/configs/scoring.yaml`:
- normal uncertain intervals use `gpt-4o-mini`
- higher-risk uncertain intervals can switch to `gpt-4o` based on `high_risk_switch_threshold`


## Troubleshooting

If you see `ModuleNotFoundError: No module named 'dotenv'`, install dependencies first:

```bash
pip install -r requirements.txt
```

The analyzer can still run in local-only mode without OpenAI key integration if `python-dotenv` is missing, but installing requirements is recommended.
