"""config.py — Central configuration for VisionAssist v5 (Pi-Laptop Split Mode)."""

# ── Network (Pi → Laptop streaming) ──────────────────────────────────────────
UDP_PORT            = 5000        # Must match Pi's --port argument
STREAM_TIMEOUT_S    = 5.0         # Seconds without packet = disconnected

# ── Camera (used only in direct-camera / fallback mode) ──────────────────────
CAMERA_INDEX   = 0
FRAME_W        = 640
FRAME_H        = 480
TARGET_FPS     = 20               # Match Pi stream rate

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME        = "yolov8m.pt"
FOCAL_LENGTH      = 615
DETECTION_CONF    = 0.45
OCR_EVERY_N       = 10            # Slightly less frequent OCR to ease laptop load

# ── Smart Decision Engine ─────────────────────────────────────────────────────
DANGER_DIST           = 1.0
WARNING_DIST          = 2.0
CONFIRM_FRAMES        = 3
MIN_APPROACH_RATE     = 0.08
THREAT_SCORE_THRESHOLD = 0.55

# ── Scene Context Engine ──────────────────────────────────────────────────────
SCENE_MEMORY_FRAMES   = 90
CROWD_THRESHOLD       = 3
OCCLUSION_IOU_THRESH  = 0.30

# ── Adaptive Threshold Engine ─────────────────────────────────────────────────
ADAPTIVE_ENABLED      = True
ADAPTIVE_MIN          = 0.35
ADAPTIVE_MAX          = 0.75

# ── UI ────────────────────────────────────────────────────────────────────────
WIN_W   = 1280
WIN_H   = 800

# Colour palette
C_BG      = "#07090F"
C_SURFACE = "#0D1117"
C_PANEL   = "#111827"
C_CARD    = "#161F2E"
C_BORDER  = "#1E2D42"
C_ACCENT  = "#38BDF8"
C_ACCENT2 = "#0EA5E9"
C_DANGER  = "#EF4444"
C_WARN    = "#F59E0B"
C_OK      = "#22C55E"
C_PURPLE  = "#A78BFA"
C_TEXT    = "#CBD5E1"
C_MUTED   = "#475569"
C_WHITE   = "#F1F5F9"
C_VIDEO   = "#7C3AED"
C_PI      = "#E84393"          # Pink accent for Pi-stream mode indicator