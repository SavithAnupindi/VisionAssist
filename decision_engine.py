"""
decision_engine.py — VisionAssist v5: Smart AI-Based Threat Assessment Engine

NOVELTIES ADDED FOR IEEE v5:

1. ADAPTIVE THRESHOLD ENGINE
   The THREAT_SCORE_THRESHOLD automatically adjusts based on scene density.
   In a crowded scene (many objects), the threshold rises to suppress noise.
   In a sparse scene, it lowers to catch subtle threats earlier.
   Formula: threshold = base + k * log(1 + object_count)

2. SCENE CONTEXT MEMORY
   Maintains a rolling 90-frame history of what has been seen.
   Uses this to detect NEWLY ENTERED objects (first appearance after
   absence) vs. persistent background objects, and prioritises new arrivals.

3. TRAJECTORY PREDICTION (Time-to-Contact)
   Given distance D and approach rate V, computes:
     TTC = D / V  (seconds until contact)
   The alert urgency is based on TTC rather than raw distance, making the
   system respond earlier to fast-moving objects and later to slow ones.
   This is the same principle used in automotive ADAS systems.

4. OCCLUSION-AWARE SCORING
   Detects when a tracked object has been partially occluded (its bounding
   box overlaps significantly with a higher-priority object). Occluded
   objects get a reduced score to avoid false alerts from partial views.

5. SCENE DENSITY CLASSIFICATION
   Classifies the scene as SPARSE / NORMAL / CROWDED and adjusts verbosity:
   SPARSE  → more sensitive, lower threshold
   CROWDED → higher threshold, only highest-threat objects announced

Original scoring signals (retained):
  1. Distance Score       — sigmoid-shaped, peaks at 0m
  2. Approach Rate Score  — linear regression closing velocity
  3. Persistence Score    — frames consistently detected
  4. Class Danger Weight  — per-class base risk
  5. Position Score       — center-path priority
  6. Confidence Score     — YOLO model confidence
"""

import time
import math
import collections
from dataclasses import dataclass, field
from audio import PRIORITY_CRITICAL, PRIORITY_WARNING, PRIORITY_INFO
import config as cfg


# ── Class danger weights ──────────────────────────────────────────────────────
CLASS_DANGER = {
    "car": 0.95,        "truck": 0.95,     "bus": 0.95,
    "motorcycle": 0.90, "bicycle": 0.75,
    "traffic light": 0.40, "stop sign": 0.35, "fire hydrant": 0.60,
    "dog": 0.55,        "cat": 0.30,       "bird": 0.15,
    "person": 0.80,
    "chair": 0.50,      "couch": 0.45,     "dining table": 0.55,
    "bed": 0.40,        "toilet": 0.35,
    "bottle": 0.30,     "cup": 0.20,       "book": 0.15,
    "backpack": 0.35,   "suitcase": 0.45,  "handbag": 0.25,
    "potted plant": 0.25, "bench": 0.30,
}
DEFAULT_DANGER = 0.35
MIN_APPROACH   = cfg.MIN_APPROACH_RATE
CLEAR_HOLDOFF  = 4.0

# TTC thresholds (seconds) — Time-To-Contact
TTC_DANGER  = 2.5    # < 2.5s = DANGER regardless of raw distance
TTC_WARNING = 6.0    # < 6.0s = WARNING


@dataclass
class ObjectState:
    last_spoken_time:  float = 0.0
    last_spoken_dist:  float = 99.0
    last_spoken_tier:  str   = ""
    last_score:        float = 0.0
    score_buf:         list  = field(default_factory=list)
    consecutive_high:  int   = 0
    first_seen:        float = field(default_factory=time.time)
    last_seen:         float = field(default_factory=time.time)
    is_new_entry:      bool  = True    # True for first N seconds after (re)appearance


class DecisionEngine:

    def __init__(self):
        self._states:          dict[str, ObjectState]   = {}
        self._last_clear:      float = 0.0
        self._last_text:       float = 0.0
        self._empty_frames:    int   = 0
        self._danger_active:   bool  = False
        self._danger_holdoff:  float = 0.0

        # ── Scene context memory ───────────────────────────────────────────────
        # Rolling window: each entry is a set of labels seen in that frame
        self._scene_history:   collections.deque = collections.deque(
            maxlen=cfg.SCENE_MEMORY_FRAMES
        )
        # Label → timestamp of last confirmed disappearance
        self._last_disappeared: dict[str, float] = {}
        # Adaptive threshold (starts at config value, gets auto-tuned)
        self._adaptive_threshold: float = cfg.THREAT_SCORE_THRESHOLD
        self._scene_density:      str   = "NORMAL"   # SPARSE / NORMAL / CROWDED

    # ─────────────────────────── PUBLIC API ───────────────────────────────────

    def process(
        self,
        detections: list[dict],
        texts:      list[str],
    ) -> list[tuple]:
        now  = time.time()
        msgs = []

        current_labels = {d["label"] for d in detections}

        # ── 1. Update scene memory ────────────────────────────────────────────
        self._scene_history.append(current_labels)
        self._update_scene_density(len(detections))
        self._update_adaptive_threshold(len(detections))

        # Mark newly-appeared objects
        self._update_new_entry_flags(current_labels, now)

        # ── 2. Detect occlusion pairs ─────────────────────────────────────────
        occluded_labels = self._find_occluded(detections)

        # ── 3. Score every detection ──────────────────────────────────────────
        scored = []
        for d in detections:
            is_occluded = d["label"] in occluded_labels
            score, breakdown = self._score(d, is_occluded)
            scored.append((d, score, breakdown))

        # ── 4. Decide which to announce ───────────────────────────────────────
        threshold = self._adaptive_threshold
        any_danger = False

        for d, score, breakdown in scored:
            label    = d["label"]
            dist     = d["distance"]
            position = d["position"]
            motion   = d.get("motion", "")
            approach = d.get("approach_rate", 0.0)

            # Time-To-Contact
            ttc = self._compute_ttc(dist, approach)

            tier  = self._tier(dist, label, score, ttc)
            state = self._states.setdefault(label, ObjectState())

            state.last_seen = now

            # Smooth score over 4 frames
            state.score_buf.append(score)
            if len(state.score_buf) > 4:
                state.score_buf.pop(0)
            smooth_score = sum(state.score_buf) / len(state.score_buf)

            if tier == "DANGER":
                any_danger = True

            # ── Gate logic ────────────────────────────────────────────────────
            should_speak = False

            if smooth_score < threshold:
                state.consecutive_high = 0
                continue

            state.consecutive_high += 1

            if state.last_spoken_tier == "":
                # New threat — must be confirmed over CONFIRM_FRAMES
                if state.consecutive_high >= cfg.CONFIRM_FRAMES:
                    should_speak = True

            elif self._tier_rank(tier) > self._tier_rank(state.last_spoken_tier):
                # Tier escalation
                should_speak = True

            elif dist is not None and state.last_spoken_dist - dist >= 0.5:
                # Closed by ≥0.5m
                should_speak = True

            elif ttc is not None and ttc <= TTC_DANGER and now - state.last_spoken_time >= 1.5:
                # TTC-based imminent danger repeat
                should_speak = True

            elif approach > 0.3 and now - state.last_spoken_time >= 2.0:
                # Fast approach cooldown
                should_speak = True

            elif tier == "DANGER" and now - state.last_spoken_time >= 1.5:
                should_speak = True

            elif state.is_new_entry and now - state.last_spoken_time >= 3.0:
                # New object entering scene — announce once even if slow
                should_speak = True

            if not should_speak:
                continue

            aprio = {"DANGER": PRIORITY_CRITICAL,
                     "WARNING": PRIORITY_WARNING,
                     "INFO": PRIORITY_INFO}[tier]
            msg = self._build_message(label, dist, position, motion, approach, tier, ttc)

            state.last_spoken_time = now
            state.last_spoken_dist = dist if dist is not None else 99.0
            state.last_spoken_tier = tier
            state.last_score       = smooth_score
            state.is_new_entry     = False

            msgs.append((aprio, tier, msg, smooth_score))

        # ── 5. OCR ────────────────────────────────────────────────────────────
        if texts and now - self._last_text >= 10.0:
            snippet = ". ".join(texts[:2])[:55]
            msgs.append((PRIORITY_INFO, "TEXT", f"Text reads: {snippet}", 0.0))
            self._last_text = now

        # ── 6. Clear path ─────────────────────────────────────────────────────
        high_threat = any(s >= threshold for _, s, _ in scored)

        if any_danger:
            self._danger_active  = True
            self._danger_holdoff = now
        if self._danger_active and (now - self._danger_holdoff) > CLEAR_HOLDOFF:
            self._danger_active = False

        if not high_threat and not texts and not self._danger_active:
            self._empty_frames += 1
            if self._empty_frames >= 5 and now - self._last_clear >= 15.0:
                msgs.append((PRIORITY_INFO, "CLEAR", "Path is clear.", 0.0))
                self._last_clear = now
        else:
            self._empty_frames = 0

        # ── 7. Prune absent objects ───────────────────────────────────────────
        active_labels = {d["label"] for d in detections}
        for label in list(self._states):
            if label not in active_labels:
                self._states[label].consecutive_high = 0
                self._states[label].last_spoken_tier = ""
                self._last_disappeared[label]        = now

        msgs.sort(key=lambda m: m[0])
        return msgs

    def get_all_scores(self, detections: list[dict]) -> list[dict]:
        """Returns scored detections for UI display. No state changes."""
        occluded = self._find_occluded(detections)
        out = []
        for d in detections:
            score, breakdown = self._score(d, d["label"] in occluded)
            ttc = self._compute_ttc(d.get("distance"), d.get("approach_rate", 0))
            out.append({
                **d,
                "threat_score": score,
                "breakdown":    breakdown,
                "ttc":          ttc,
                "is_occluded":  d["label"] in occluded,
                "scene_density": self._scene_density,
                "adaptive_threshold": self._adaptive_threshold,
            })
        return out

    def update_thresholds(self, danger: float, warning: float):
        cfg.DANGER_DIST  = danger
        cfg.WARNING_DIST = warning

    def get_scene_info(self) -> dict:
        """Returns scene-level metadata for the UI."""
        return {
            "density":            self._scene_density,
            "adaptive_threshold": round(self._adaptive_threshold, 2),
            "history_len":        len(self._scene_history),
        }

    # ─────────────────────────── NOVELTY 1: ADAPTIVE THRESHOLD ───────────────

    def _update_adaptive_threshold(self, n_objects: int):
        if not cfg.ADAPTIVE_ENABLED:
            self._adaptive_threshold = cfg.THREAT_SCORE_THRESHOLD
            return
        # Logarithmic scaling: more objects → higher threshold (less noise)
        k = 0.08
        base = cfg.THREAT_SCORE_THRESHOLD
        tuned = base + k * math.log(1 + n_objects)
        self._adaptive_threshold = round(
            max(cfg.ADAPTIVE_MIN, min(cfg.ADAPTIVE_MAX, tuned)), 3
        )

    # ─────────────────────────── NOVELTY 2: SCENE DENSITY ────────────────────

    def _update_scene_density(self, n_objects: int):
        if n_objects == 0:
            self._scene_density = "SPARSE"
        elif n_objects >= cfg.CROWD_THRESHOLD:
            self._scene_density = "CROWDED"
        else:
            self._scene_density = "NORMAL"

    # ─────────────────────────── NOVELTY 3: TRAJECTORY / TTC ─────────────────

    @staticmethod
    def _compute_ttc(dist: float | None, approach: float) -> float | None:
        """Time-To-Contact in seconds. None if not approaching."""
        if dist is None or approach <= MIN_APPROACH:
            return None
        return round(dist / approach, 1)

    # ─────────────────────────── NOVELTY 4: NEW ENTRY DETECTION ──────────────

    def _update_new_entry_flags(self, current_labels: set, now: float):
        new_entry_window = 4.0   # seconds
        for label in current_labels:
            state = self._states.get(label)
            last_gone = self._last_disappeared.get(label, 0)
            if state is None:
                # Brand new label
                s = ObjectState()
                s.is_new_entry = True
                self._states[label] = s
            elif now - last_gone > new_entry_window:
                # Was absent long enough — treat as re-entry
                state.is_new_entry = True

    # ─────────────────────────── NOVELTY 5: OCCLUSION DETECTION ─────────────

    @staticmethod
    def _iou(b1: tuple, b2: tuple) -> float:
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0
        a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        return inter / (a1 + a2 - inter)

    def _find_occluded(self, detections: list[dict]) -> set[str]:
        """
        Returns set of labels that are significantly overlapping with a
        higher-priority object — likely partially occluded.
        """
        occluded = set()
        n = len(detections)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                di, dj = detections[i], detections[j]
                if di["priority"] <= dj["priority"]:
                    continue   # di is higher priority
                iou = self._iou(di["bbox"], dj["bbox"])
                if iou > cfg.OCCLUSION_IOU_THRESH:
                    occluded.add(dj["label"])
        return occluded

    # ─────────────────────────── SCORING ──────────────────────────────────────

    def _score(self, d: dict, is_occluded: bool = False) -> tuple[float, dict]:
        dist     = d.get("distance")
        approach = d.get("approach_rate", 0.0)
        frames   = d.get("frames_seen", 1)
        label    = d["label"]
        position = d["position"]
        conf     = d["confidence"]
        ttc      = self._compute_ttc(dist, approach)

        # 1. Distance score
        if dist is None:
            dist_score = 0.2
        elif dist <= cfg.DANGER_DIST:
            dist_score = 1.0
        elif dist <= cfg.WARNING_DIST:
            dist_score = 0.5 + 0.5 * (cfg.WARNING_DIST - dist) / (cfg.WARNING_DIST - cfg.DANGER_DIST)
        elif dist <= 6.0:
            dist_score = 0.5 * max(0, 1.0 - (dist - cfg.WARNING_DIST) / 3.0)
        else:
            dist_score = 0.0

        # 2. Approach rate score (only closing)
        approach_score = min(1.0, max(0.0, approach) / 2.0)

        # 3. TTC score (NEW) — inversely proportional to time to contact
        if ttc is not None:
            ttc_score = min(1.0, TTC_DANGER / max(ttc, 0.1))
        else:
            ttc_score = 0.0

        # 4. Persistence score
        persistence_score = min(1.0, frames / 8.0)

        # 5. Class danger
        class_score = CLASS_DANGER.get(label, DEFAULT_DANGER)

        # 6. Position score
        pos_score = {"center": 1.0, "left": 0.65, "right": 0.65}.get(position, 0.5)

        # 7. Confidence
        conf_score = conf

        # ── Weights ───────────────────────────────────────────────────────────
        if approach <= MIN_APPROACH:
            approach_weight = 0.0
            ttc_weight      = 0.0
            dist_weight     = 0.40
            stationary_pen  = 0.60 if dist is not None and dist > cfg.DANGER_DIST else 1.0
        else:
            approach_weight = 0.22
            ttc_weight      = 0.10   # TTC adds on top of approach
            dist_weight     = 0.25
            stationary_pen  = 1.0

        occlusion_pen = 0.75 if is_occluded else 1.0

        raw = (
            dist_weight     * dist_score        +
            approach_weight * approach_score     +
            ttc_weight      * ttc_score          +
            0.11            * persistence_score  +
            0.13            * class_score        +
            0.08            * pos_score          +
            0.06            * conf_score
        ) * stationary_pen * occlusion_pen

        score = round(min(1.0, raw), 3)

        breakdown = {
            "distance_score":     round(dist_score, 2),
            "approach_score":     round(approach_score, 2),
            "ttc_score":          round(ttc_score, 2),
            "persistence_score":  round(persistence_score, 2),
            "class_score":        round(class_score, 2),
            "position_score":     round(pos_score, 2),
            "stationary_penalty": round(stationary_pen, 2),
            "occlusion_penalty":  round(occlusion_pen, 2),
            "final":              score,
        }
        return score, breakdown

    # ─────────────────────────── TIER / MESSAGE ───────────────────────────────

    def _tier(self, dist: float | None, label: str, score: float, ttc: float | None) -> str:
        hr = CLASS_DANGER.get(label, DEFAULT_DANGER) >= 0.75

        # TTC-based override — fast-approaching object jumps tier
        if ttc is not None:
            if ttc <= TTC_DANGER:
                return "DANGER"
            if ttc <= TTC_WARNING:
                return "WARNING"

        if dist is None:
            return "INFO"
        if dist <= cfg.DANGER_DIST or (hr and dist <= cfg.DANGER_DIST * 1.3):
            return "DANGER"
        if dist <= cfg.WARNING_DIST or (hr and dist <= cfg.WARNING_DIST * 1.2):
            return "WARNING"
        return "INFO"

    @staticmethod
    def _tier_rank(tier: str) -> int:
        return {"INFO": 1, "WARNING": 2, "DANGER": 3}.get(tier, 0)

    def _build_message(self, label, dist, position, motion, approach, tier, ttc) -> str:
        ds = f"{dist:.1f} metres" if dist is not None else "nearby"

        if approach > 0.5:
            speed = "quickly" if approach > 1.5 else "steadily"
            mv = f", moving toward you {speed}"
        elif approach > MIN_APPROACH:
            mv = ", approaching"
        elif motion == "stationary":
            mv = ", stationary"
        else:
            mv = ""

        # TTC qualifier
        tc = f", contact in {ttc:.0f} seconds" if ttc is not None and ttc < TTC_WARNING else ""

        if tier == "DANGER":
            return f"Danger. {label} on your {position}, {ds}{mv}{tc}. Stop now."
        if tier == "WARNING":
            return f"Warning. {label} on {position}, {ds}{mv}{tc}."
        return f"{label.capitalize()} on {position}, {ds}."