"""Interval-based video fire analysis using a trained YOLO classification model.

This script is designed for post-training usage and defaults to `classification_model.pt`.
It samples the video in active windows (e.g., 10s ON / 10s OFF), extracts frame-level
signals, and aggregates end-of-video metrics useful for risk scoring and downstream APIs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import cv2
from ultralytics import YOLO

CLASS_NAMES = {
    0: "controlled_fire",
    1: "fire",
    2: "smoke",
}


@dataclass
class FrameDetection:
    timestamp_s: float
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]
    bbox_area_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze video in intervals with classification_model.pt")
    parser.add_argument("--video", type=Path, required=True, help="Path to input video")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("classification_model.pt"),
        help="Path to trained model weights (default: classification_model.pt)",
    )
    parser.add_argument("--clip-seconds", type=float, default=10.0, help="Active analysis window in seconds")
    parser.add_argument("--break-seconds", type=float, default=10.0, help="Skip interval between active windows")
    parser.add_argument("--sample-fps", type=float, default=2.0, help="Frames/sec sampled during active windows")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("classification/video_metrics.json"),
        help="Where to save the final metrics JSON",
    )
    parser.add_argument(
        "--top-fire-frame-out",
        type=Path,
        default=Path("classification/top_fire_frame.jpg"),
        help="Where to save the sampled frame with the highest computed fire score",
    )
    parser.add_argument("--device", type=str, default="0", help="Device id or cpu")
    return parser.parse_args()


def _in_active_window(t: float, clip_s: float, break_s: float) -> bool:
    period = clip_s + break_s
    if period <= 0:
        return True
    return (t % period) < clip_s


def _area_ratio(xyxy: list[float], frame_w: int, frame_h: int) -> float:
    x1, y1, x2, y2 = xyxy
    box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return box_area / float(max(1, frame_w * frame_h))


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _fire_frame_score(frame_dets: list[FrameDetection]) -> float:
    """Compute a per-frame fire severity score in [0, 1].

    Uses only sampled detections from one frame and emphasizes fire confidence + area,
    while reducing score when controlled fire is dominant.
    """
    fire = [d for d in frame_dets if d.class_name == "fire"]
    controlled = [d for d in frame_dets if d.class_name == "controlled_fire"]

    if not fire:
        return 0.0

    fire_strength = max(d.confidence * (0.6 + 0.4 * _clamp01(d.bbox_area_ratio * 5.0)) for d in fire)
    controlled_penalty = max((d.confidence for d in controlled), default=0.0) * 0.35
    return _clamp01(fire_strength - controlled_penalty)


def _compute_aggregate_confidence(
    controlled_mean: float,
    fire_mean: float,
    smoke_mean: float,
    fire_spread_score: float,
    fire_flicker_score: float,
) -> dict[str, float]:
    """Return normalized final confidence values for controlled_fire / fire / smoke.

    Design:
    - controlled_fire: strong controlled detections, reduced by aggressive spread/flicker.
    - fire: strong fire detections, boosted by spread/flicker, reduced by strong controlled signal.
    - smoke: mostly based on smoke signal, lightly boosted by fire presence.
    """
    controlled_raw = controlled_mean * 0.85 + (1.0 - _clamp01(fire_spread_score * 3.0)) * 0.10 + (
        1.0 - _clamp01(fire_flicker_score * 4.0)
    ) * 0.05
    fire_raw = fire_mean * 0.70 + _clamp01(fire_spread_score * 3.0) * 0.20 + _clamp01(fire_flicker_score * 4.0) * 0.10
    fire_raw *= 1.0 - controlled_mean * 0.25
    smoke_raw = smoke_mean * 0.85 + fire_mean * 0.15

    total = controlled_raw + fire_raw + smoke_raw
    if total <= 1e-12:
        return {
            "controlled_fire": 0.0,
            "fire": 0.0,
            "smoke": 0.0,
        }

    return {
        "controlled_fire": _clamp01(controlled_raw / total),
        "fire": _clamp01(fire_raw / total),
        "smoke": _clamp01(smoke_raw / total),
    }


def analyze_video(args: argparse.Namespace) -> dict:
    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

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

    frame_idx = 0
    sampled_frames = 0
    detections: list[FrameDetection] = []

    top_fire_frame_score = -1.0
    top_fire_frame_timestamp = None
    top_fire_frame = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t = frame_idx / fps if fps > 0 else 0.0
        active = _in_active_window(t, args.clip_seconds, args.break_seconds)
        should_sample = active and (frame_idx % sample_stride == 0)

        if should_sample:
            sampled_frames += 1
            results = model.predict(
                source=frame,
                conf=args.conf,
                verbose=False,
                device=args.device,
            )
            frame_detections: list[FrameDetection] = []

            if results:
                r = results[0]
                if r.boxes is not None:
                    boxes = r.boxes
                    for i in range(len(boxes)):
                        cls_id = int(boxes.cls[i].item())
                        conf = float(boxes.conf[i].item())
                        xyxy = [float(v) for v in boxes.xyxy[i].tolist()]
                        name = CLASS_NAMES.get(cls_id, str(cls_id))
                        frame_detections.append(
                            FrameDetection(
                                timestamp_s=t,
                                class_id=cls_id,
                                class_name=name,
                                confidence=conf,
                                bbox_xyxy=xyxy,
                                bbox_area_ratio=_area_ratio(xyxy, frame_w, frame_h),
                            )
                        )

            detections.extend(frame_detections)

            frame_fire_score = _fire_frame_score(frame_detections)
            if frame_fire_score > top_fire_frame_score:
                top_fire_frame_score = frame_fire_score
                top_fire_frame_timestamp = t
                top_fire_frame = frame.copy()

        frame_idx += 1

    cap.release()

    def class_items(class_name: str) -> list[FrameDetection]:
        return [d for d in detections if d.class_name == class_name]

    fire_items = class_items("fire")
    smoke_items = class_items("smoke")
    controlled_items = class_items("controlled_fire")

    fire_conf_series = [d.confidence for d in fire_items]
    fire_area_series = [d.bbox_area_ratio for d in fire_items]

    fire_flicker_score = 0.0
    if len(fire_conf_series) > 1:
        deltas = [abs(fire_conf_series[i] - fire_conf_series[i - 1]) for i in range(1, len(fire_conf_series))]
        fire_flicker_score = mean(deltas)

    fire_spread_score = 0.0
    if len(fire_area_series) > 1:
        fire_spread_score = max(fire_area_series) - min(fire_area_series)

    mean_controlled = _safe_mean([d.confidence for d in controlled_items])
    mean_fire = _safe_mean([d.confidence for d in fire_items])
    mean_smoke = _safe_mean([d.confidence for d in smoke_items])
    aggregate_confidence = _compute_aggregate_confidence(
        controlled_mean=mean_controlled,
        fire_mean=mean_fire,
        smoke_mean=mean_smoke,
        fire_spread_score=fire_spread_score,
        fire_flicker_score=fire_flicker_score,
    )

    top_fire_frame_path = None
    if top_fire_frame is not None:
        args.top_fire_frame_out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.top_fire_frame_out), top_fire_frame)
        top_fire_frame_path = str(args.top_fire_frame_out)

    metrics = {
        "input": {
            "video": str(args.video),
            "weights": str(args.weights),
            "duration_seconds": duration_s,
            "fps": fps,
            "frame_size": {"width": frame_w, "height": frame_h},
        },
        "sampling": {
            "clip_seconds": args.clip_seconds,
            "break_seconds": args.break_seconds,
            "sample_fps": args.sample_fps,
            "sampled_frames": sampled_frames,
            "confidence_threshold": args.conf,
        },
        "class_order": ["controlled_fire", "fire", "smoke"],
        "summary": {
            "num_detections_total": len(detections),
            "counts": {
                "controlled_fire": len(controlled_items),
                "fire": len(fire_items),
                "smoke": len(smoke_items),
            },
            "max_confidence": {
                "controlled_fire": max([d.confidence for d in controlled_items], default=0.0),
                "fire": max([d.confidence for d in fire_items], default=0.0),
                "smoke": max([d.confidence for d in smoke_items], default=0.0),
            },
            "mean_confidence": {
                "controlled_fire": mean_controlled,
                "fire": mean_fire,
                "smoke": mean_smoke,
            },
            "video_behavior_signals": {
                "fire_flicker_score": fire_flicker_score,
                "fire_spread_score": fire_spread_score,
            },
            "aggregate_relative_confidence": aggregate_confidence,
            "top_fire_frame": {
                "path": top_fire_frame_path,
                "timestamp_s": top_fire_frame_timestamp,
                "fire_frame_score": max(0.0, top_fire_frame_score),
            },
        },
        "scoring_formula": {
            "controlled_fire": "0.85*mean_controlled + 0.10*(1-clamp(fire_spread*3)) + 0.05*(1-clamp(fire_flicker*4))",
            "fire": "(0.70*mean_fire + 0.20*clamp(fire_spread*3) + 0.10*clamp(fire_flicker*4))*(1-0.25*mean_controlled)",
            "smoke": "0.85*mean_smoke + 0.15*mean_fire",
            "normalization": "final_confidence[class] = raw[class] / sum(raw_controlled_fire, raw_fire, raw_smoke)",
            "fire_frame_score": "max(fire_conf*(0.6+0.4*clamp(area_ratio*5))) - 0.35*max(controlled_fire_conf)",
        },
        "detections": [
            {
                "timestamp_s": d.timestamp_s,
                "class_id": d.class_id,
                "class_name": d.class_name,
                "confidence": d.confidence,
                "bbox_xyxy": d.bbox_xyxy,
                "bbox_area_ratio": d.bbox_area_ratio,
            }
            for d in detections
        ],
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    args = parse_args()
    metrics = analyze_video(args)
    print("Analysis complete")
    print(f"Metrics saved to: {args.json_out}")
    if metrics["summary"]["top_fire_frame"]["path"]:
        print(f"Top fire frame saved to: {metrics['summary']['top_fire_frame']['path']}")
    print(json.dumps(metrics["summary"], indent=2))


if __name__ == "__main__":
    main()
