# Classification Module

This module runs interval-based video risk analysis and writes compact outputs for downstream decision logic.

## What it outputs

For each run it saves:

- `results/timeline.json` (full decision timeline)
- one folder per interval under `results/`, each containing:
  - `interval_metrics.json`
  - `top_fire.jpg` (or fallback first sampled frame if no fire box is detected)

No overall `video_metrics.json` and no overall top-fire image are written.
No per-box detection dump is written to JSON.

## OpenAI integration behavior (uncertain intervals + local emergency verification)

OpenAI is called when either: (1) an interval is uncertain by `is_uncertain(...)`, or (2) local-only scoring would classify the interval as `Emergency` (forced contextual verification).

When OpenAI is used, context now takes priority in blending: `w_context=0.72` with a minimum effective context weight floor (`min_context_weight=0.70`). If OpenAI scenario is `Emergency`, final score is clamped to be at least the OpenAI `context_score`.

If triggered (uncertain or local Emergency) and API key exists:
- interval top-fire frame + metrics are sent to OpenAI.
- OpenAI returns machine-readable:
  - `context_score`
  - `scenario` (`No Fire Risk`, `Elevated Risk`, `Hazard`, `Emergency`)
  - `confidence`
  - `rationale`

If triggered and API key is missing:
- OpenAI is skipped.
- output records local-only mode.

If not uncertain and not local Emergency:
- OpenAI is skipped by design.

### Exact uncertainty trigger criteria (from `classification/configs/scoring.yaml`)

An interval triggers OpenAI reasoning when **any one** of these is true:

0. **Local emergency verification trigger**
   - local-only scenario rank is `Emergency` before OpenAI blending

1. **Danger index mid-band**
   - `0.20 <= dangerous_fire_index <= 0.90`
2. **Fire vs controlled is near tie**
   - `abs(fire_vs_controlled_gap) < 0.24`
3. **Smoke present while controlled fire dominates**
   - `smoke >= 0.08` **and** `controlled_fire >= 0.30`
4. **High flicker + moderate spread conflict**
   - `flicker_normalized >= 0.45` **and** `0.08 <= spread_normalized <= 0.75`

If none of the above are true, OpenAI is not called for that interval unless local-only rank is Emergency.

### Model selection for OpenAI

OpenAI model choice is configurable in `classification/configs/scoring.yaml`:

- default uncertain intervals: `openai.model` (default `gpt-4o-mini`)
- higher-risk uncertain intervals: `openai.high_risk_model` (default `gpt-4o`) when local score exceeds `openai.high_risk_switch_threshold` (`0.55`)


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
  --results-dir results \
  --timeline-out results/timeline.json \
  --camera-id cam_01 \
  --location-type warehouse \
  --demo-mode
```

`--demo-mode` forces local-only behavior even if an API key is present.

## Where to read the interval aggregate numbers

In each interval folder (`results/incident_interval_.../`), `interval_metrics.json` includes:

- `openai.eligible` (whether interval met criteria to call OpenAI)
- `openai.trigger_reason` (`uncertainty`, `emergency_verification`, or `none`)
- `openai.note` (explicit skip reason such as `demo_mode` or missing key)

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

- `0.62 * dangerous_fire_index`
- `0.36 * spread_normalized`
- `0.28 * smoke`
- `0.24 * max(fire_vs_controlled_gap, 0)`
- `0.22 * min(fire_to_controlled_ratio, 5)/5`
- penalty: `0.03 * flicker_normalized * max(0, 1-fire)`

Then:
- if OpenAI used: `final_score = (1-w)*local + w*context`, with `w=0.72` and minimum context-priority floor (`min_context_weight=0.70`).
- if OpenAI skipped: `final_score = local_score`.

`decision_confidence` is separate from detection confidence and measures consistency/decisiveness of the combined signals.

### Exact scoring level ranges and hysteresis

Configured thresholds:

- **Emergency** (uncontrolled growth / strongly dangerous)
  - enter when `final_score >= 0.54`
  - remain Emergency while `final_score >= 0.46`
- **Hazard** (fire present and concerning, potentially controlled but significant)
  - enter when `final_score >= 0.26` (and not Emergency)
  - remain Hazard while `final_score >= 0.20`
- **Elevated Risk** (smoke or weak fire evidence, low certainty of active/uncontrolled fire)
  - for low-risk smoke-heavy scenes with minimal visible fire: `final_score >= 0.08`, `smoke >= 0.10`, and `fire <= 0.12`
- **No Fire Risk**
  - when `final_score <= 0.04`

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
