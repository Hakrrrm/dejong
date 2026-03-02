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


OpenAI is optional and used for uncertain intervals plus forced verification of any interval that would be locally classified as Emergency. Context is now prioritized in blending (`w_context=0.72` with minimum context-weight floor `0.70`).

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
  --results-dir results \
  --timeline-out results/timeline.json \
  --camera-id cam_01 \
  --location-type warehouse \
  --demo-mode
```

`--demo-mode` forces local-only run (no OpenAI calls).

## What classification outputs

- `results/timeline.json` (full per-interval decision timeline)
- one folder per interval under `results/` containing:
  - `interval_metrics.json`
  - `top_fire.jpg` (or fallback first sampled frame if no fire box is detected)

JSON is compact/aggregate-only (no per-box timestamp dump), and only interval artifacts are saved.

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
- higher-risk uncertain intervals can switch to `gpt-4o` based on `high_risk_switch_threshold` (now more aggressive at `0.55`)

## Exact OpenAI trigger criteria and score ranges

OpenAI is called when intervals are uncertain OR locally Emergency. Uncertainty is triggered when any of these are true:
- `0.20 <= dangerous_fire_index <= 0.90`
- `abs(fire_vs_controlled_gap) < 0.24`
- `smoke >= 0.08` and `controlled_fire >= 0.30`
- `flicker_normalized >= 0.45` and `0.08 <= spread_normalized <= 0.75`

Scenario rank thresholds (with hysteresis):
- Emergency enter `>= 0.46`, stay while `>= 0.40`
- Hazard enter `>= 0.22`, stay while `>= 0.16`
- Elevated Risk for low-risk smoke-heavy scenes with minimal visible fire (`score >= 0.08`, `smoke >= 0.10`, `fire <= 0.12`); No Fire Risk for very low scores (`<= 0.04`)

## Console output behavior

`classification/analyze_video.py` prints one compact decision line per interval only:
- interval label
- `openai_used`
- `scenario`
- `final_score`
- `decision_confidence`

Full metrics remain in JSON outputs.

## Troubleshooting

If you see `ModuleNotFoundError: No module named 'dotenv'`, install dependencies first:

```bash
pip install -r requirements.txt
```

The analyzer can still run in local-only mode without OpenAI key integration if `python-dotenv` is missing, but installing requirements is recommended.
