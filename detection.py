"""
detection.py — YOLOv8 + ByteTrack with:
  • Kalman-smoothed distance estimation
  • Pixel-velocity to real-world velocity conversion
  • Per-track distance history for approach rate calculation
  • Frame persistence counter for false-positive suppression
"""

import time
import numpy as np
import threading
from collections import defaultdict
from ultralytics import YOLO
import config as cfg


_model = YOLO("yolov8n.pt")


# Warm-up — prevents first-frame inference lag
_model(np.zeros((480, 640, 3), dtype=np.uint8), verbose=False)
_infer_lock = threading.Lock()

# ── Real-world object heights (cm) ────────────────────────────────────────────
HEIGHTS_CM = {
    "person": 170,  "bicycle": 100,   "car": 150,    "motorcycle": 120,
    "bus": 300,     "truck": 250,     "traffic light": 250, "stop sign": 90,
    "dog": 55,      "cat": 30,        "bird": 25,
    "chair": 80,    "couch": 85,      "dining table": 75,   "bed": 55,
    "toilet": 70,   "tv": 60,         "laptop": 25,
    "cell phone": 15, "book": 22,     "cup": 12,     "bottle": 25,
    "backpack": 50, "handbag": 30,    "umbrella": 90, "suitcase": 65,
    "fire hydrant": 60, "bench": 50,  "potted plant": 40,
}

OBJECT_PRIORITY = {
    "person": 1,  "car": 1,    "truck": 1,  "bus": 1,  "motorcycle": 1,
    "bicycle": 2, "traffic light": 2, "stop sign": 2, "fire hydrant": 2,
    "dog": 3,     "cat": 3,
    "chair": 5,   "couch": 5,  "dining table": 5, "bed": 5,
}

# ── Per-track state ───────────────────────────────────────────────────────────
_dist_history: dict[int, list] = defaultdict(list)
_pix_history:  dict[int, list] = defaultdict(list)
_seen_count:   dict[int, int]  = defaultdict(int)

_HIST_LEN = 12


def _smooth_distance(raw_m: float, tid: int, now: float) -> float:
    hist = _dist_history[tid]
    if not hist:
        hist.append((now, raw_m))
        return raw_m
    alpha = 0.35
    smoothed = alpha * raw_m + (1 - alpha) * hist[-1][1]
    hist.append((now, smoothed))
    if len(hist) > _HIST_LEN:
        hist.pop(0)
    return round(smoothed, 2)


def _approach_rate(tid: int) -> float:
    hist = _dist_history.get(tid, [])
    if len(hist) < 4:
        return 0.0
    times = [h[0] for h in hist]
    dists = [h[1] for h in hist]
    dt = times[-1] - times[0]
    if dt < 0.1:
        return 0.0
    n = len(hist)
    t_mean = sum(times) / n
    d_mean = sum(dists) / n
    num = sum((times[i] - t_mean) * (dists[i] - d_mean) for i in range(n))
    den = sum((times[i] - t_mean) ** 2 for i in range(n))
    if den == 0:
        return 0.0
    slope = num / den
    return round(-slope, 3)


def _pixel_motion(tid: int, cx: float, cy: float) -> str:
    hist = _pix_history[tid]
    hist.append((cx, cy))
    if len(hist) > _HIST_LEN:
        hist.pop(0)
    if len(hist) < 5:
        return "unknown"
    dx = hist[-1][0] - hist[0][0]
    dy = hist[-1][1] - hist[0][1]
    spd = (dx**2 + dy**2) ** 0.5
    if spd < 12:
        return "stationary"
    if abs(dy) > abs(dx) * 1.3:
        return "approaching" if dy > 0 else "receding"
    return "moving right" if dx > 0 else "moving left"


def _raw_distance(box_h_px: float, label: str) -> float | None:
    real_h = HEIGHTS_CM.get(label, 100)
    if box_h_px < 6:
        return None
    return (real_h * cfg.FOCAL_LENGTH) / (box_h_px * 100)


def _position(cx: float, w: int) -> str:
    t = w / 3
    return "left" if cx < t else ("center" if cx < 2 * t else "right")


def detect_objects(frame: np.ndarray) -> list[dict]:
    h, w = frame.shape[:2]
    now = time.perf_counter()

    with _infer_lock:
        results = _model.track(
            frame,
            persist=True,
            verbose=False,
            conf=cfg.DETECTION_CONF,
            iou=0.45,
            tracker="bytetrack.yaml",
        )

    seen: dict[tuple, dict] = {}

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls    = int(box.cls[0])
            label  = _model.names[cls]
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bh  = y2 - y1
            cx  = (x1 + x2) / 2
            cy  = (y1 + y2) / 2
            tid = int(box.id[0]) if box.id is not None else -1

            raw_d = _raw_distance(bh, label)
            dist  = None
            approach_rate = 0.0
            motion = "unknown"

            if tid >= 0:
                _seen_count[tid] += 1
                if raw_d is not None:
                    dist = _smooth_distance(raw_d, tid, now)
                    approach_rate = _approach_rate(tid)
                motion = _pixel_motion(tid, cx, cy)

            det = {
                "label":         label,
                "confidence":    round(conf, 2),
                "distance":      dist,
                "position":      _position(cx, w),
                "motion":        motion,
                "approach_rate": approach_rate,
                "track_id":      tid,
                "frames_seen":   _seen_count.get(tid, 1),
                "bbox":          (int(x1), int(y1), int(x2), int(y2)),
                "priority":      OBJECT_PRIORITY.get(label, 10),
            }

            key = (label, det["position"])
            if key not in seen or conf > seen[key]["confidence"]:
                seen[key] = det

    return sorted(seen.values(), key=lambda d: (d["priority"], d["distance"] or 99))


def cleanup_stale_tracks(active_ids: set[int]):
    for store in (_dist_history, _pix_history, _seen_count):
        stale = [k for k in store if k not in active_ids]
        for k in stale:
            del store[k]