"""Interval-based video fire analysis with optional OpenAI context tie-breaker.

Outputs:
- overall compact JSON (--json-out)
- per-interval compact JSON (--interval-json-dir)
- one top-fire JPG per interval (--interval-top-frame-dir)
- one global top-fire JPG (--top-fire-frame-out)
- consolidated timeline JSON (--timeline-out)
"""

from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import cv2
import yaml
try:
    from dotenv import load_dotenv
except ImportError:  # optional in local-only runtime
    def load_dotenv(*_args, **_kwargs):
        return False
from ultralytics import YOLO



def _bootstrap_import_paths() -> None:
    """Make direct script execution robust on Windows/Linux shells."""
    import sys

    this_dir = Path(__file__).resolve().parent
    repo_root = this_dir.parent

    for candidate in (repo_root, this_dir):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


_bootstrap_import_paths()

try:
    from classification.src.openai_reasoner.client import get_openai_client
    from classification.src.openai_reasoner.reasoner import reason_with_openai
    from classification.src.scoring import (
        LocalScoreWeights,
        ScenarioThresholds,
        UncertaintyThresholds,
        assign_scenario_rank,
        compute_decision_confidence,
        compute_local_score,
        is_uncertain,
    )
except ModuleNotFoundError:
    try:
        from src.openai_reasoner.client import get_openai_client
        from src.openai_reasoner.reasoner import reason_with_openai
        from src.scoring import (
            LocalScoreWeights,
            ScenarioThresholds,
            UncertaintyThresholds,
            assign_scenario_rank,
            compute_decision_confidence,
            compute_local_score,
            is_uncertain,
        )
    except ModuleNotFoundError:
        from openai_reasoner.client import get_openai_client
        from openai_reasoner.reasoner import reason_with_openai
        from scoring import (
            LocalScoreWeights,
            ScenarioThresholds,
            UncertaintyThresholds,
            assign_scenario_rank,
            compute_decision_confidence,
            compute_local_score,
            is_uncertain,
        )

CLASS_NAMES = {0: "controlled_fire", 1: "fire", 2: "smoke"}


@dataclass
class IntervalMeta:
    index: int
    start_s: float
    end_s: float


@dataclass
class AggregateStats:
    counts: dict[str, int] = field(default_factory=lambda: {"controlled_fire": 0, "fire": 0, "smoke": 0})
    sum_conf: dict[str, float] = field(default_factory=lambda: {"controlled_fire": 0.0, "fire": 0.0, "smoke": 0.0})
    max_conf: dict[str, float] = field(default_factory=lambda: {"controlled_fire": 0.0, "fire": 0.0, "smoke": 0.0})
    fire_conf_series: list[float] = field(default_factory=list)
    fire_area_series: list[float] = field(default_factory=list)
    num_detections_total: int = 0
    sampled_frames: int = 0


@dataclass
class FrameSignal:
    class_name: str
    confidence: float
    bbox_area_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze video in intervals with classification_model.pt")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=Path("classification_model.pt"))
    parser.add_argument("--clip-seconds", type=float, default=10.0)
    parser.add_argument("--break-seconds", type=float, default=10.0)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--json-out", type=Path, default=Path("classification/video_metrics.json"))
    parser.add_argument("--interval-json-dir", type=Path, default=Path("classification/interval_metrics"))
    parser.add_argument("--top-fire-frame-out", type=Path, default=Path("classification/top_fire_frame.jpg"))
    parser.add_argument("--interval-top-frame-dir", type=Path, default=Path("classification/interval_top_fire_frames"))
    parser.add_argument("--timeline-out", type=Path, default=Path("classification/timeline.json"))
    parser.add_argument("--scoring-config", type=Path, default=Path("classification/configs/scoring.yaml"))
    parser.add_argument("--camera-id", type=str, default="unknown_camera")
    parser.add_argument("--location-type", type=str, default="unknown_location")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--demo-mode", action="store_true", help="Force local-only demo mode (skip OpenAI even if key exists)")
    return parser.parse_args()


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _safe_mean(total: float, count: int) -> float:
    return total / count if count > 0 else 0.0


def _in_active_window(t: float, clip_s: float, break_s: float) -> bool:
    period = clip_s + break_s
    if period <= 0:
        return True
    return (t % period) < clip_s


def _interval_meta_for_time(t: float, clip_s: float, break_s: float) -> IntervalMeta:
    period = clip_s + break_s
    if period <= 0:
        return IntervalMeta(index=1, start_s=0.0, end_s=clip_s)
    idx0 = int(t // period)
    start_s = idx0 * period
    end_s = start_s + clip_s
    return IntervalMeta(index=idx0 + 1, start_s=start_s, end_s=end_s)


def _area_ratio(xyxy: list[float], frame_w: int, frame_h: int) -> float:
    x1, y1, x2, y2 = xyxy
    box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return box_area / float(max(1, frame_w * frame_h))


def _fire_frame_score(frame_signals: list[FrameSignal]) -> float:
    fire = [d for d in frame_signals if d.class_name == "fire"]
    controlled = [d for d in frame_signals if d.class_name == "controlled_fire"]
    if not fire:
        return 0.0
    fire_strength = max(d.confidence * (0.7 + 0.3 * _clamp01(d.bbox_area_ratio * 5.0)) for d in fire)
    controlled_penalty = max((d.confidence for d in controlled), default=0.0) * 0.20
    return _clamp01(fire_strength - controlled_penalty)


def _update_stats(stats: AggregateStats, frame_signals: list[FrameSignal]) -> None:
    stats.sampled_frames += 1
    for d in frame_signals:
        stats.num_detections_total += 1
        stats.counts[d.class_name] += 1
        stats.sum_conf[d.class_name] += d.confidence
        stats.max_conf[d.class_name] = max(stats.max_conf[d.class_name], d.confidence)
        if d.class_name == "fire":
            stats.fire_conf_series.append(d.confidence)
            stats.fire_area_series.append(d.bbox_area_ratio)


def _compute_behavior(stats: AggregateStats) -> tuple[float, float]:
    fire_flicker_score = 0.0
    if len(stats.fire_conf_series) > 1:
        deltas = [
            abs(stats.fire_conf_series[i] - stats.fire_conf_series[i - 1])
            for i in range(1, len(stats.fire_conf_series))
        ]
        fire_flicker_score = mean(deltas)

    fire_spread_score = 0.0
    if len(stats.fire_area_series) > 1:
        fire_spread_score = max(stats.fire_area_series) - min(stats.fire_area_series)

    return fire_flicker_score, fire_spread_score


def _compute_aggregate_confidence(
    mean_controlled: float,
    mean_fire: float,
    mean_smoke: float,
    fire_spread_score: float,
    fire_flicker_score: float,
) -> dict[str, float]:
    spread_n = _clamp01(fire_spread_score * 3.0)
    flicker_n = _clamp01(fire_flicker_score * 4.0)

    controlled_raw = (
        0.45 * mean_controlled
        + 0.15 * (1.0 - spread_n)
        + 0.10 * (1.0 - flicker_n)
        + 0.30 * (1.0 - mean_fire)
    )
    controlled_raw *= 1.0 - 0.35 * mean_smoke

    fire_raw = 0.60 * mean_fire + 0.20 * spread_n + 0.15 * flicker_n + 0.05 * mean_smoke
    fire_raw *= 1.0 - 0.15 * mean_controlled

    smoke_raw = 0.75 * mean_smoke + 0.20 * mean_fire + 0.05 * flicker_n

    total = controlled_raw + fire_raw + smoke_raw
    if total <= 1e-12:
        return {"controlled_fire": 0.0, "fire": 0.0, "smoke": 0.0}

    return {
        "controlled_fire": _clamp01(controlled_raw / total),
        "fire": _clamp01(fire_raw / total),
        "smoke": _clamp01(smoke_raw / total),
    }


def _compute_risk_numbers(aggregate: dict[str, float], fire_spread_score: float, fire_flicker_score: float) -> dict[str, float]:
    fire = aggregate["fire"]
    controlled = aggregate["controlled_fire"]
    smoke = aggregate["smoke"]
    spread_n = _clamp01(fire_spread_score * 3.0)
    flicker_n = _clamp01(fire_flicker_score * 4.0)

    fire_vs_controlled_gap = fire - controlled
    fire_to_controlled_ratio = fire / max(controlled, 1e-6)
    dangerous_fire_index = _clamp01(0.55 * fire + 0.20 * smoke + 0.15 * spread_n + 0.10 * flicker_n)

    return {
        "dangerous_fire_index": dangerous_fire_index,
        "fire_vs_controlled_gap": fire_vs_controlled_gap,
        "fire_to_controlled_ratio": fire_to_controlled_ratio,
        "spread_normalized": spread_n,
        "flicker_normalized": flicker_n,
    }


def _summarize_stats(stats: AggregateStats) -> dict:
    mean_controlled = _safe_mean(stats.sum_conf["controlled_fire"], stats.counts["controlled_fire"])
    mean_fire = _safe_mean(stats.sum_conf["fire"], stats.counts["fire"])
    mean_smoke = _safe_mean(stats.sum_conf["smoke"], stats.counts["smoke"])

    fire_flicker_score, fire_spread_score = _compute_behavior(stats)
    aggregate_relative_confidence = _compute_aggregate_confidence(
        mean_controlled=mean_controlled,
        mean_fire=mean_fire,
        mean_smoke=mean_smoke,
        fire_spread_score=fire_spread_score,
        fire_flicker_score=fire_flicker_score,
    )
    risk_numbers = _compute_risk_numbers(
        aggregate=aggregate_relative_confidence,
        fire_spread_score=fire_spread_score,
        fire_flicker_score=fire_flicker_score,
    )

    return {
        "num_detections_total": stats.num_detections_total,
        "sampled_frames": stats.sampled_frames,
        "counts": stats.counts,
        "max_confidence": stats.max_conf,
        "mean_confidence": {
            "controlled_fire": mean_controlled,
            "fire": mean_fire,
            "smoke": mean_smoke,
        },
        "video_behavior_signals": {
            "fire_flicker_score": fire_flicker_score,
            "fire_spread_score": fire_spread_score,
        },
        "aggregate_relative_confidence": aggregate_relative_confidence,
        "risk_numbers": risk_numbers,
    }


def _load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Scoring config not found: {path}")
    return yaml.safe_load(path.read_text())


def _as_uncertainty(cfg: dict) -> UncertaintyThresholds:
    return UncertaintyThresholds(**cfg["uncertainty"])


def _as_local_weights(cfg: dict) -> LocalScoreWeights:
    return LocalScoreWeights(**cfg["local_score_weights"])


def _as_scenario_thresholds(cfg: dict) -> ScenarioThresholds:
    return ScenarioThresholds(**cfg["scenario_thresholds"])


def analyze_video(args: argparse.Namespace) -> dict:
    load_dotenv()

    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    cfg = _load_config(args.scoring_config)
    uncertainty_cfg = _as_uncertainty(cfg)
    local_weights = _as_local_weights(cfg)
    scenario_thresholds = _as_scenario_thresholds(cfg)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_s = (total_frames / fps) if fps > 0 else 0.0

    model = YOLO(str(args.weights))
    sample_stride = max(1, int(round(fps / max(args.sample_fps, 0.01))))

    overall_stats = AggregateStats()
    interval_buckets: dict[int, dict] = {}

    top_fire_frame_score = -1.0
    top_fire_frame_timestamp = None
    top_fire_frame = None

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t = frame_idx / fps if fps > 0 else 0.0
        active = _in_active_window(t, args.clip_seconds, args.break_seconds)
        should_sample = active and (frame_idx % sample_stride == 0)

        if should_sample:
            meta = _interval_meta_for_time(t, args.clip_seconds, args.break_seconds)
            bucket = interval_buckets.setdefault(
                meta.index,
                {
                    "meta": meta,
                    "stats": AggregateStats(),
                    "top_fire_frame_score": -1.0,
                    "top_fire_frame_timestamp": None,
                    "top_fire_frame": None,
                },
            )

            results = model.predict(source=frame, conf=args.conf, verbose=False, device=args.device)
            frame_signals: list[FrameSignal] = []

            if results:
                r = results[0]
                if r.boxes is not None:
                    boxes = r.boxes
                    for i in range(len(boxes)):
                        cls_id = int(boxes.cls[i].item())
                        conf = float(boxes.conf[i].item())
                        xyxy = [float(v) for v in boxes.xyxy[i].tolist()]
                        class_name = CLASS_NAMES.get(cls_id, str(cls_id))
                        frame_signals.append(
                            FrameSignal(
                                class_name=class_name,
                                confidence=conf,
                                bbox_area_ratio=_area_ratio(xyxy, frame_w, frame_h),
                            )
                        )

            _update_stats(overall_stats, frame_signals)
            _update_stats(bucket["stats"], frame_signals)

            frame_fire_score = _fire_frame_score(frame_signals)
            if frame_fire_score > top_fire_frame_score:
                top_fire_frame_score = frame_fire_score
                top_fire_frame_timestamp = t
                top_fire_frame = frame.copy()

            if frame_fire_score > bucket["top_fire_frame_score"]:
                bucket["top_fire_frame_score"] = frame_fire_score
                bucket["top_fire_frame_timestamp"] = t
                bucket["top_fire_frame"] = frame.copy()

        frame_idx += 1

    cap.release()

    top_fire_frame_path = None
    if top_fire_frame is not None:
        args.top_fire_frame_out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.top_fire_frame_out), top_fire_frame)
        top_fire_frame_path = str(args.top_fire_frame_out)

    overall_summary = _summarize_stats(overall_stats)
    overall_summary["top_fire_frame"] = {
        "path": top_fire_frame_path,
        "timestamp_s": top_fire_frame_timestamp,
        "fire_frame_score": max(0.0, top_fire_frame_score),
    }

    args.interval_json_dir.mkdir(parents=True, exist_ok=True)
    args.interval_top_frame_dir.mkdir(parents=True, exist_ok=True)

    openai_client = None if args.demo_mode else get_openai_client()
    openai_enabled = (openai_client is not None) and (not args.demo_mode)
    runtime_mode = "demo_local" if args.demo_mode or not openai_enabled else "openai_enabled"
    context_cfg = cfg["context_weighting"]
    w_context = float(context_cfg["w_context"])
    scale_by_openai_confidence = bool(context_cfg.get("scale_by_openai_confidence", True))

    timeline_entries = []
    interval_outputs = []
    prev_rank = None

    for idx in sorted(interval_buckets.keys()):
        bucket = interval_buckets[idx]
        meta: IntervalMeta = bucket["meta"]
        interval_label = f"interval_{meta.index:04d}"
        interval_base = f"incident_{interval_label}_{int(meta.start_s):06d}s_{int(meta.end_s):06d}s"

        interval_top_fire_frame_path = None
        if bucket["top_fire_frame"] is not None:
            interval_top_frame_path = args.interval_top_frame_dir / f"{interval_base}_top_fire.jpg"
            cv2.imwrite(str(interval_top_frame_path), bucket["top_fire_frame"])
            interval_top_fire_frame_path = str(interval_top_frame_path)

        interval_summary = _summarize_stats(bucket["stats"])
        interval_summary["top_fire_frame"] = {
            "path": interval_top_fire_frame_path,
            "timestamp_s": bucket["top_fire_frame_timestamp"],
            "fire_frame_score": max(0.0, bucket["top_fire_frame_score"]),
        }

        agg = interval_summary["aggregate_relative_confidence"]
        risk = interval_summary["risk_numbers"]

        local_score = compute_local_score(agg, risk, local_weights)
        uncertain = is_uncertain(interval_summary, uncertainty_cfg)
        local_pre_openai_rank = assign_scenario_rank(local_score, prev_rank, scenario_thresholds, agg)
        emergency_needs_verification = local_pre_openai_rank == "Emergency"
        should_call_openai = uncertain or emergency_needs_verification

        openai_payload = {"used": False, "context_score": 0.0, "scenario": None, "confidence": 0.0, "rationale": [], "note": ""}

        if should_call_openai and openai_enabled and interval_top_fire_frame_path is not None:
            model_name = cfg["openai"]["model"]
            if local_score >= float(cfg["openai"].get("high_risk_switch_threshold", 1.0)):
                model_name = cfg["openai"].get("high_risk_model", model_name)

            metadata = {
                "interval_label": interval_label,
                "camera_id": args.camera_id,
                "location_type": args.location_type,
                "start_s": meta.start_s,
                "end_s": meta.end_s,
                "openai_trigger_reason": "emergency_verification" if emergency_needs_verification else "uncertainty",
            }
            openai_result = reason_with_openai(
                client=openai_client,
                model=model_name,
                image_path=Path(interval_top_fire_frame_path),
                metadata=metadata,
                metrics={"aggregate_relative_confidence": agg, "risk_numbers": risk, "local_pre_openai_rank": local_pre_openai_rank},
            )
            openai_payload = {"used": True, **openai_result}
        elif should_call_openai and not openai_enabled:
            if emergency_needs_verification:
                openai_payload["note"] = "local Emergency flagged; OpenAI verification skipped in demo/local-only mode"
            else:
                openai_payload["note"] = "uncertain interval; OpenAI skipped in demo/local-only mode"
        else:
            openai_payload["note"] = "interval not uncertain and not local-Emergency; OpenAI skipped"

        if openai_payload["used"]:
            context_weight = w_context
            if scale_by_openai_confidence:
                context_weight *= float(openai_payload["confidence"])
            final_score = (1.0 - context_weight) * local_score + context_weight * float(openai_payload["context_score"])
        else:
            final_score = local_score

        decision_confidence = compute_decision_confidence(
            aggregate_relative_confidence=agg,
            risk_numbers=risk,
            openai_output_optional=openai_payload if openai_payload["used"] else None,
        )
        scenario_rank = assign_scenario_rank(final_score, prev_rank, scenario_thresholds, agg)
        prev_rank = scenario_rank

        decision = {
            "local_score": _clamp01(local_score),
            "final_score": _clamp01(final_score),
            "decision_confidence": decision_confidence,
            "scenario_rank": scenario_rank,
        }

        interval_payload = {
            "interval": {
                "label": interval_label,
                "index": meta.index,
                "start_s": meta.start_s,
                "end_s": meta.end_s,
                "base_name": interval_base,
            },
            "sampling": {
                "clip_seconds": args.clip_seconds,
                "break_seconds": args.break_seconds,
                "sample_fps": args.sample_fps,
                "confidence_threshold": args.conf,
            },
            "summary": interval_summary,
            "openai": openai_payload,
            "decision": decision,
        }

        interval_path = args.interval_json_dir / f"{interval_base}.json"
        interval_path.write_text(json.dumps(interval_payload, indent=2))

        interval_outputs.append(
            {
                "label": interval_label,
                "index": meta.index,
                "base_name": interval_base,
                "json_path": str(interval_path),
                "top_fire_frame_path": interval_top_fire_frame_path,
                "aggregate_relative_confidence": agg,
                "risk_numbers": risk,
                "openai": openai_payload,
                "decision": decision,
            }
        )

        timeline_entries.append(
            {
                "interval_index": meta.index - 1,
                "start_time": meta.start_s,
                "end_time": meta.end_s,
                "aggregate_relative_confidence": agg,
                "risk_numbers": risk,
                "openai": openai_payload,
                "decision": decision,
            }
        )

    metrics = {
        "input": {
            "video": str(args.video),
            "weights": str(args.weights),
            "duration_seconds": duration_s,
            "fps": fps,
            "frame_size": {"width": frame_w, "height": frame_h},
            "camera_id": args.camera_id,
            "location_type": args.location_type,
            "runtime_mode": runtime_mode,
        },
        "sampling": {
            "clip_seconds": args.clip_seconds,
            "break_seconds": args.break_seconds,
            "sample_fps": args.sample_fps,
            "sampled_frames": overall_stats.sampled_frames,
            "confidence_threshold": args.conf,
        },
        "summary": overall_summary,
        "interval_outputs": interval_outputs,
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(metrics, indent=2))

    # timeline.json
    scenario_counts = {"Emergency": 0, "Hazard": 0, "Elevated Risk": 0, "No Fire Risk": 0}
    first_escalation = None
    max_risk = 0.0
    for row in timeline_entries:
        rank = row["decision"]["scenario_rank"]
        scenario_counts[rank] = scenario_counts.get(rank, 0) + 1
        max_risk = max(max_risk, row["decision"]["final_score"])
        if first_escalation is None and rank in {"Emergency", "Hazard"}:
            first_escalation = row["start_time"]

    timeline = {
        "event_id": f"evt_{uuid.uuid4().hex[:12]}",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "interval_seconds": args.clip_seconds,
        "timeline": timeline_entries,
        "summary": {
            "max_risk": max_risk,
            "time_to_first_escalation": first_escalation,
            "scenario_counts": scenario_counts,
            "openai_enabled": openai_enabled,
            "runtime_mode": runtime_mode,
        },
    }
    args.timeline_out.parent.mkdir(parents=True, exist_ok=True)
    args.timeline_out.write_text(json.dumps(timeline, indent=2))

    return metrics


def main() -> None:
    args = parse_args()
    metrics = analyze_video(args)
    print("Analysis complete")
    print(f"Runtime mode: {metrics['input']['runtime_mode']}")
    print(f"Timeline saved to: {args.timeline_out}")

    for row in metrics.get("interval_outputs", []):
        openai_used = bool(row.get("openai", {}).get("used", False))
        decision = row.get("decision", {})
        print(
            f"{row.get('label', 'interval')}: "
            f"openai_used={openai_used}, "
            f"scenario={decision.get('scenario_rank')}, "
            f"final_score={decision.get('final_score', 0.0):.3f}, "
            f"decision_confidence={decision.get('decision_confidence', 0.0):.3f}"
        )


if __name__ == "__main__":
    main()
