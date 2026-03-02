# Classification Module

This module runs interval-based video risk analysis and writes compact outputs for downstream decision logic.

## What it outputs

For each run it saves:

- `video_metrics.json` (overall aggregate)
- one JSON per interval in `classification/interval_metrics/`
- one top-fire image per interval in `classification/interval_top_fire_frames/`
- one overall top-fire image (`classification/top_fire_frame.jpg`)
- `timeline.json` (full decision timeline)

No per-box detection dump is written to JSON.

## OpenAI integration behavior (uncertain intervals only)

OpenAI is called only when an interval is considered uncertain by `is_uncertain(...)` rules from `classification/configs/scoring.yaml`. The uncertainty band has been widened so OpenAI is invoked more frequently as a tie-breaker.

If uncertain and API key exists:
- interval top-fire frame + metrics are sent to OpenAI.
- OpenAI returns machine-readable:
  - `context_score`
  - `scenario`
  - `confidence`
  - `rationale`

If uncertain and API key is missing:
- OpenAI is skipped.
- output records local-only mode.

If not uncertain:
- OpenAI is skipped by design.

### Exact uncertainty trigger criteria (from `classification/configs/scoring.yaml`)

An interval triggers OpenAI reasoning when **any one** of these is true:

1. **Danger index mid-band**
   - `0.25 <= dangerous_fire_index <= 0.85`
2. **Fire vs controlled is near tie**
   - `abs(fire_vs_controlled_gap) < 0.18`
3. **Smoke present while controlled fire dominates**
   - `smoke >= 0.10` **and** `controlled_fire >= 0.35`
4. **High flicker + moderate spread conflict**
   - `flicker_normalized >= 0.50` **and** `0.10 <= spread_normalized <= 0.70`

If none of the above are true, OpenAI is not called for that interval.

### Model selection for OpenAI

OpenAI model choice is configurable in `classification/configs/scoring.yaml`:

- default uncertain intervals: `openai.model` (default `gpt-4o-mini`)
- higher-risk uncertain intervals: `openai.high_risk_model` (default `gpt-4o`) when local score exceeds `openai.high_risk_switch_threshold` (`0.65`)


## Configure environment

### 1) Create `.env`

Copy `.env.example` to `.env` and fill your key:

```bash
cp .env.example .env
# if dotfiles are hidden in your UI: cp env.example .env
```

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 2) Set env var manually (optional)

- macOS/Linux:
  ```bash
  export OPENAI_API_KEY="..."
  ```
- PowerShell:
  ```powershell
  $env:OPENAI_API_KEY="..."
  ```

## Run with a test video

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
  --top-fire-frame-out classification/top_fire_frame.jpg \
  --timeline-out classification/timeline.json \
  --camera-id cam_01 \
  --location-type warehouse \
  --demo-mode
```

`--demo-mode` forces local-only behavior even if an API key is present.

## Where to read the interval aggregate numbers

In each interval JSON:

- `summary.aggregate_relative_confidence.controlled_fire`
- `summary.aggregate_relative_confidence.fire`
- `summary.aggregate_relative_confidence.smoke`
- `summary.risk_numbers.dangerous_fire_index`
- `summary.risk_numbers.fire_vs_controlled_gap`
- `summary.risk_numbers.fire_to_controlled_ratio`
- `decision.local_score`
- `decision.final_score`
- `decision.decision_confidence`
- `decision.scenario_rank`

## Timeline output

`timeline.json` includes full sequence and summary:

- `event_id`
- `interval_seconds`
- `timeline[]` entries with:
  - aggregate confidences
  - risk numbers
  - openai block
  - decision block
- top-level summary:
  - `max_risk`
  - `time_to_first_escalation`
  - `scenario_counts`

## Scoring overview

`local_score` uses these configured weights:

- `0.55 * dangerous_fire_index`
- `0.30 * spread_normalized`
- `0.25 * smoke`
- `0.20 * max(fire_vs_controlled_gap, 0)`
- `0.18 * min(fire_to_controlled_ratio, 5)/5`
- penalty: `0.04 * flicker_normalized * max(0, 1-fire)`

Then:
- if OpenAI used: `final_score = (1-w)*local + w*context`, with `w=0.50` (optionally multiplied by OpenAI confidence).
- if OpenAI skipped: `final_score = local_score`.

`decision_confidence` is separate from detection confidence and measures consistency/decisiveness of the combined signals.

### Exact scoring level ranges and hysteresis

Configured thresholds:

- **Emergency**
  - enter when `final_score >= 0.66`
  - remain Emergency while `final_score >= 0.58`
- **Hazard**
  - enter when `final_score >= 0.40` (and not Emergency)
  - remain Hazard while `final_score >= 0.34`
- **Elevated Risk**
  - default fallback for lower scores (`>= 0.15` target band; implementation currently always falls back to this rank)

## Console output behavior

The CLI now prints only compact per-interval decisions (no full metrics dump), one line per interval:

- interval label
- `openai_used=true/false`
- `scenario`
- `final_score`
- `decision_confidence`

Use JSON outputs for full structured data.

## Troubleshooting

If you hit `ModuleNotFoundError: No module named 'dotenv'`:

```bash
pip install -r requirements.txt
```

`python-dotenv` is used for `.env` loading. If unavailable, the analyzer now falls back to local-only environment behavior, but full install is recommended.
