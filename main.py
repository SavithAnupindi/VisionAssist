"""
main.py — VisionAssist v5 (Pi-Laptop Split Architecture)

Architecture:
  Raspberry Pi  →  UDP stream  →  Laptop (this file)
                                   ├── YOLOv8m detection
                                   ├── ByteTrack tracking
                                   ├── Threat / Decision Engine
                                   ├── OCR (pytesseract)
                                   ├── TTS audio alerts
                                   └── Tkinter GUI display

Source modes:
  1. Pi Stream   — receives JPEG frames from the Pi over UDP (primary)
  2. Local Camera — direct OpenCV capture (fallback / demo without Pi)
  3. Video File  — pre-recorded video for testing

Run:
  python main.py                  # defaults to Pi Stream mode
  python main.py --source camera  # local webcam
  python main.py --source video   # prompts for file
"""

import argparse
import cv2
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import platform
import time
import collections
import numpy as np
from PIL import Image, ImageTk
import sys
from typing import Optional
import os

import config as cfg
from frame_receiver import FrameReceiver
from detection import detect_objects, cleanup_stale_tracks
from decision_engine import DecisionEngine
from audio import speak, clear_queue, PRIORITY_CRITICAL, PRIORITY_WARNING, PRIORITY_INFO

# ─────────────────────────────────────────────────────────────────────────────
_MONO = "Consolas" if platform.system() == "Windows" else (
    "Menlo" if platform.system() == "Darwin" else "DejaVu Sans Mono")
_SANS = "Segoe UI" if platform.system() == "Windows" else (
    "SF Pro Display" if platform.system() == "Darwin" else "Liberation Sans")


# ─────────────────────────────────────────────────────────────────────────────
# Frame annotation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tier_bgr(tier):
    return {"DANGER": (40, 50, 239), "WARNING": (20, 158, 245), "INFO": (30, 190, 90)}.get(tier, (80, 80, 80))


def annotate(frame, scored, texts):
    for d in scored:
        x1, y1, x2, y2 = d["bbox"]
        dist     = d["distance"]
        score    = d.get("threat_score", 0.0)
        approach = d.get("approach_rate", 0.0)
        ttc      = d.get("ttc")
        occ      = d.get("is_occluded", False)
        label    = d["label"]

        tier = "DANGER" if (dist is not None and dist <= cfg.DANGER_DIST) else \
               "WARNING" if (dist is not None and dist <= cfg.WARNING_DIST) else "INFO"
        if ttc is not None:
            if ttc <= 2.5:  tier = "DANGER"
            elif ttc <= 6:  tier = "WARNING"

        color = _tier_bgr(tier)
        if tier == "DANGER":
            tint = frame.copy()
            cv2.rectangle(tint, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(tint, 0.15, frame, 0.85, 0, frame)

        thick = 3 if tier == "DANGER" else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)

        bw = x2 - x1
        cv2.rectangle(frame, (x1, y2 - 4), (x2, y2), (25, 25, 35), -1)
        cv2.rectangle(frame, (x1, y2 - 4), (x1 + max(2, int(bw * score)), y2), color, -1)

        ds      = f"{dist:.1f}m" if dist else "?"
        arr     = "↑" if approach > cfg.MIN_APPROACH_RATE else ("↓" if approach < -0.05 else "·")
        tc      = f" T:{ttc:.0f}s" if ttc and ttc < 8 else ""
        occ_m   = " [OCC]" if occ else ""
        cap_txt = f" {label}  {ds} {arr} {int(score * 100)}%{tc}{occ_m} "

        (tw, th), _ = cv2.getTextSize(cap_txt, cv2.FONT_HERSHEY_DUPLEX, 0.41, 1)
        py = max(y1 - 1, th + 8)
        cv2.rectangle(frame, (x1, py - th - 6), (x1 + tw, py + 2), color, -1)
        cv2.putText(frame, cap_txt, (x1, py - 1), cv2.FONT_HERSHEY_DUPLEX, 0.41, (6, 6, 10), 1, cv2.LINE_AA)

    if texts:
        strip = "  ·  ".join(texts)[:85]
        cv2.rectangle(frame, (0, frame.shape[0] - 28), (frame.shape[1], frame.shape[0]), (8, 10, 18), -1)
        cv2.putText(frame, f" OCR › {strip}", (6, frame.shape[0] - 8),
                    cv2.FONT_HERSHEY_DUPLEX, 0.40, (170, 130, 255), 1, cv2.LINE_AA)
    return frame


def make_heatmap(shape, scored):
    h, w = shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)
    for d in scored:
        x1, y1, x2, y2 = d["bbox"]
        intensity = d.get("threat_score", 0.0)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        r = max(30, int((x2 - x1) * 0.5))
        cv2.circle(heat, (cx, cy), r, intensity, -1)
    heat = cv2.GaussianBlur(heat, (71, 71), 0)
    heat = np.clip(heat * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(heat, cv2.COLORMAP_JET)


# ─────────────────────────────────────────────────────────────────────────────
# OCR (lazy import — only loads tesseract if available)
# ─────────────────────────────────────────────────────────────────────────────

_ocr_available = False
try:
    import pytesseract
    _ocr_available = True
except ImportError:
    pass


def run_ocr(frame: np.ndarray) -> list[str]:
    if not _ocr_available:
        return []
    try:
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        raw   = pytesseract.image_to_string(bw, config="--psm 6 --oem 1")
        words = [w.strip() for w in raw.split() if len(w.strip()) >= 3]
        return words[:6]
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────────────────────────────────────

class App:
    def __init__(self, root: tk.Tk, source_mode: str = "pi"):
        self.root        = root
        self.source_mode = source_mode   # "pi" | "camera" | "video"
        self.root.title("VisionAssist v5 — Pi-Laptop Split Mode")
        self.root.configure(bg=cfg.C_BG)
        self.root.geometry(f"{cfg.WIN_W}x{cfg.WIN_H}")
        self.root.minsize(1050, 680)

        # State
        self.running       = False
        self.paused        = False
        self.engine        = DecisionEngine()
        self.frame_n       = 0
        self.fps_buf       = collections.deque(maxlen=30)
        self.last_texts: list[str] = []
        self._video_fps    = cfg.TARGET_FPS

        # Pi stream receiver (created on start)
        self._rx = None  # type: Optional[FrameReceiver]

        # Local camera / video fallback
        self._cap = None  # type: Optional[cv2.VideoCapture]
        self.video_path = None  # type: Optional[str]
        self.video_total_frames: int  = 0
        self.video_pos_var = tk.IntVar(value=0)

        # Session stats
        self.stat_total_alerts  = 0
        self.stat_danger_count  = 0
        self.stat_warn_count    = 0
        self.stat_score_history = collections.deque(maxlen=300)

        # Settings vars
        self.heatmap_on = tk.BooleanVar(value=False)
        self.ocr_on     = tk.BooleanVar(value=True)
        self.speech_on  = tk.BooleanVar(value=True)
        self.danger_v   = tk.DoubleVar(value=cfg.DANGER_DIST)
        self.warn_v     = tk.DoubleVar(value=cfg.WARNING_DIST)
        self.conf_v     = tk.DoubleVar(value=cfg.DETECTION_CONF)
        self.thresh_v   = tk.DoubleVar(value=cfg.THREAT_SCORE_THRESHOLD)
        self.cam_v      = tk.IntVar(value=cfg.CAMERA_INDEX)
        self.speed_v    = tk.DoubleVar(value=1.0)
        self.port_v     = tk.IntVar(value=cfg.UDP_PORT)

        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

        # Auto-start if Pi mode
        if self.source_mode == "pi":
            self.root.after(400, self.start)

    # ─────────────────────────── BUILD ────────────────────────────────────────

    def _build(self):
        self._ttk_styles()
        self._build_header()
        tk.Frame(self.root, bg=cfg.C_BORDER, height=1).pack(fill="x")
        body = tk.Frame(self.root, bg=cfg.C_BG)
        body.pack(fill="both", expand=True)
        self._build_left(body)
        self._build_right(body)
        self.flash = tk.Label(self.root, text="", height=1,
                              font=(_SANS, 11, "bold"), bg=cfg.C_BG, fg=cfg.C_BG)
        self.flash.pack(fill="x", side="bottom")

    def _ttk_styles(self):
        s = ttk.Style()
        s.theme_use("clam")
        for w in ("TNotebook", "TNotebook.Tab", "TFrame", "TLabel", "TScrollbar"):
            s.configure(w, background=cfg.C_PANEL, foreground=cfg.C_TEXT,
                        bordercolor=cfg.C_BORDER, lightcolor=cfg.C_PANEL,
                        darkcolor=cfg.C_PANEL, troughcolor=cfg.C_SURFACE,
                        selectbackground=cfg.C_ACCENT)
        s.configure("TNotebook.Tab", padding=[10, 4], font=(_SANS, 9))
        s.map("TNotebook.Tab",
              background=[("selected", cfg.C_CARD)],
              foreground=[("selected", cfg.C_ACCENT)])

    def _build_header(self):
        hdr = tk.Frame(self.root, bg=cfg.C_BG, height=52)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        lf = tk.Frame(hdr, bg=cfg.C_BG)
        lf.pack(side="left", padx=16, pady=10)
        tk.Label(lf, text="◉", font=(_SANS, 18, "bold"), fg=cfg.C_ACCENT, bg=cfg.C_BG).pack(side="left")
        tk.Label(lf, text=" VisionAssist", font=(_SANS, 16, "bold"), fg=cfg.C_WHITE, bg=cfg.C_BG).pack(side="left")
        tk.Label(lf, text="  v5 · Pi-Laptop", font=(_SANS, 9), fg=cfg.C_MUTED, bg=cfg.C_BG).pack(side="left", pady=5)

        # Status pill
        sf = tk.Frame(hdr, bg=cfg.C_CARD, highlightbackground=cfg.C_BORDER, highlightthickness=1)
        sf.pack(side="right", padx=12, pady=12)
        self.s_dot = tk.Label(sf, text="●", font=(_SANS, 9), fg=cfg.C_DANGER, bg=cfg.C_CARD, padx=6)
        self.s_dot.pack(side="left")
        self.s_lbl = tk.Label(sf, text="OFFLINE", font=(_SANS, 9, "bold"), fg=cfg.C_MUTED,
                              bg=cfg.C_CARD, padx=6, pady=4)
        self.s_lbl.pack(side="left")

        # Source mode badge
        src_colors = {"pi": cfg.C_PI, "camera": cfg.C_ACCENT, "video": cfg.C_VIDEO}
        src_text   = {"pi": "π PI STREAM", "camera": "CAMERA", "video": "VIDEO"}
        self.mode_lbl = tk.Label(hdr, text=src_text.get(self.source_mode, "CAMERA"),
                                  font=(_MONO, 9, "bold"),
                                  fg=src_colors.get(self.source_mode, cfg.C_ACCENT),
                                  bg=cfg.C_BG, padx=8)
        self.mode_lbl.pack(side="right", padx=4)

    def _build_left(self, parent):
        lp = tk.Frame(parent, bg=cfg.C_BG)
        lp.pack(side="left", fill="both", padx=(12, 6), pady=10)

        # Video canvas
        self.vid = tk.Label(lp, bg="#050709", width=cfg.FRAME_W, height=cfg.FRAME_H,
                            text="Waiting for Pi stream…\n\nMake sure the Pi is running:\npython3 stream_sender.py --host <THIS PC IP>",
                            fg=cfg.C_MUTED, font=(_SANS, 11), anchor="center")
        self.vid.pack()

        # Scene density bar
        self.scene_bar = tk.Label(lp, text="Scene: —", font=(_MONO, 9),
                                   fg=cfg.C_MUTED, bg=cfg.C_BG, anchor="w")
        self.scene_bar.pack(fill="x", pady=(4, 0))

        # Top stats row
        stats_row = tk.Frame(lp, bg=cfg.C_BG)
        stats_row.pack(fill="x", pady=6)
        self._make_stat(stats_row, "FPS",     "w_fps",   "—")
        self._make_stat(stats_row, "OBJECTS", "w_objs",  "—")
        self._make_stat(stats_row, "NEAREST", "w_near",  "—")
        self._make_stat(stats_row, "THREAT",  "w_score", "—%")
        self._make_stat(stats_row, "THRESH",  "w_thresh","—")

        # Control buttons
        btn_row = tk.Frame(lp, bg=cfg.C_BG)
        btn_row.pack(fill="x", pady=(4, 0))
        bkw = {"font": (_SANS, 10, "bold"), "relief": "flat", "padx": 14, "pady": 7, "cursor": "hand2"}
        self.start_btn = tk.Button(btn_row, text="▶  Start", bg=cfg.C_OK, fg="#0A1A0A",
                                    command=self.start, **bkw)
        self.start_btn.pack(side="left", padx=(0, 4))
        self.stop_btn = tk.Button(btn_row, text="■  Stop", bg=cfg.C_DANGER, fg=cfg.C_WHITE,
                                   command=self.stop, state="disabled", **bkw)
        self.stop_btn.pack(side="left", padx=(0, 4))
        self.pause_btn = tk.Button(btn_row, text="⏸  Pause", bg=cfg.C_WARN, fg="#1A0E00",
                                    command=self._pause, state="disabled", **bkw)
        self.pause_btn.pack(side="left", padx=(0, 4))
        tk.Button(btn_row, text="🔇  Mute", bg=cfg.C_CARD, fg=cfg.C_TEXT,
                  command=self._mute, **bkw).pack(side="left")

        # Scrubber (video mode only)
        self.scrub_frame = tk.Frame(lp, bg=cfg.C_BG)
        self.scrub_frame.pack(fill="x", pady=(4, 0))
        self.scrub_lbl = tk.Label(self.scrub_frame, text="", font=(_MONO, 9),
                                   fg=cfg.C_MUTED, bg=cfg.C_BG)
        self.scrub_lbl.pack(side="right")
        self.scrub = ttk.Scale(self.scrub_frame, from_=0, to=1,
                                variable=self.video_pos_var, orient="horizontal")
        self.scrub.pack(fill="x", side="left", expand=True)
        self.scrub_frame.pack_forget()   # hidden unless video mode

    def _make_stat(self, parent, label, attr, init):
        f = tk.Frame(parent, bg=cfg.C_CARD, highlightbackground=cfg.C_BORDER, highlightthickness=1)
        f.pack(side="left", expand=True, fill="x", padx=2)
        tk.Label(f, text=label, font=(_MONO, 7), fg=cfg.C_MUTED, bg=cfg.C_CARD).pack(pady=(4, 0))
        w = tk.Label(f, text=init, font=(_MONO, 12, "bold"), fg=cfg.C_ACCENT, bg=cfg.C_CARD)
        w.pack(pady=(0, 4))
        setattr(self, attr, w)

    def _build_right(self, parent):
        rp = tk.Frame(parent, bg=cfg.C_PANEL, highlightbackground=cfg.C_BORDER, highlightthickness=1)
        rp.pack(side="right", fill="both", expand=True, padx=(0, 12), pady=10)

        nb = ttk.Notebook(rp)
        nb.pack(fill="both", expand=True)

        # ── Tab 1: Alert log ──
        log_tab = tk.Frame(nb, bg=cfg.C_PANEL)
        nb.add(log_tab, text="  Alerts  ")
        self.log = tk.Text(log_tab, state="disabled", bg=cfg.C_SURFACE, fg=cfg.C_TEXT,
                           font=(_MONO, 9), wrap="word", relief="flat",
                           padx=8, pady=6, insertbackground=cfg.C_ACCENT)
        sb = ttk.Scrollbar(log_tab, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log.pack(fill="both", expand=True)
        for tag, col in [("DANGER", cfg.C_DANGER), ("WARNING", cfg.C_WARN),
                          ("INFO", cfg.C_OK), ("CLEAR", cfg.C_ACCENT),
                          ("TIME", cfg.C_MUTED), ("TEXT", cfg.C_PURPLE)]:
            self.log.tag_configure(tag, foreground=col)
        tk.Button(log_tab, text="Clear log", font=(_SANS, 8), fg=cfg.C_MUTED,
                  bg=cfg.C_PANEL, relief="flat", command=self._clear).pack(pady=4)

        # ── Tab 2: Threat scores ──
        score_tab = tk.Frame(nb, bg=cfg.C_PANEL)
        nb.add(score_tab, text="  Threats  ")
        score_canvas = tk.Canvas(score_tab, bg=cfg.C_PANEL, highlightthickness=0)
        score_sb = ttk.Scrollbar(score_tab, command=score_canvas.yview)
        score_canvas.configure(yscrollcommand=score_sb.set)
        score_sb.pack(side="right", fill="y")
        score_canvas.pack(fill="both", expand=True)
        self.score_frame = tk.Frame(score_canvas, bg=cfg.C_PANEL)
        score_canvas.create_window((0, 0), window=self.score_frame, anchor="nw")
        self.score_frame.bind("<Configure>", lambda e: score_canvas.configure(
            scrollregion=score_canvas.bbox("all")))

        # ── Tab 3: Settings ──
        set_tab = tk.Frame(nb, bg=cfg.C_PANEL)
        nb.add(set_tab, text="  Settings  ")
        self._build_settings(set_tab)

        # ── Tab 4: Session stats ──
        stat_tab = tk.Frame(nb, bg=cfg.C_PANEL)
        nb.add(stat_tab, text="  Session  ")
        self._build_session_tab(stat_tab)

    def _build_settings(self, parent):
        pad = {"padx": 12, "pady": 4}

        def row(label, widget_fn):
            f = tk.Frame(parent, bg=cfg.C_PANEL)
            f.pack(fill="x", **pad)
            tk.Label(f, text=label, font=(_SANS, 9), fg=cfg.C_MUTED,
                     bg=cfg.C_PANEL, width=20, anchor="w").pack(side="left")
            widget_fn(f)

        # Pi Network settings
        tk.Label(parent, text="─── Pi Stream ───", font=(_MONO, 9), fg=cfg.C_PI,
                 bg=cfg.C_PANEL).pack(pady=(14, 2))
        row("UDP Port", lambda f: ttk.Entry(f, textvariable=self.port_v, width=8).pack(side="left"))

        # Show local IP hint
        import socket as _sock
        try:
            s = _sock.socket(_sock.AF_INET, _sock.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
        except Exception:
            local_ip = "unknown"
        tk.Label(parent, text=f"  This PC IP: {local_ip}  (tell the Pi to stream here)",
                 font=(_MONO, 8), fg=cfg.C_ACCENT, bg=cfg.C_PANEL).pack(pady=(0, 6))

        tk.Label(parent, text="─── Detection ───", font=(_MONO, 9), fg=cfg.C_ACCENT,
                 bg=cfg.C_PANEL).pack(pady=(10, 2))
        row("Danger dist (m)", lambda f: ttk.Scale(f, variable=self.danger_v, from_=0.3,
            to=3.0, orient="h", length=140).pack(side="left"))
        row("Warning dist (m)", lambda f: ttk.Scale(f, variable=self.warn_v, from_=0.5,
            to=6.0, orient="h", length=140).pack(side="left"))
        row("Conf threshold", lambda f: ttk.Scale(f, variable=self.conf_v, from_=0.2,
            to=0.9, orient="h", length=140).pack(side="left"))
        row("Threat threshold", lambda f: ttk.Scale(f, variable=self.thresh_v, from_=0.2,
            to=0.9, orient="h", length=140).pack(side="left"))

        tk.Label(parent, text="─── Display ───", font=(_MONO, 9), fg=cfg.C_ACCENT,
                 bg=cfg.C_PANEL).pack(pady=(10, 2))
        row("Heatmap overlay", lambda f: ttk.Checkbutton(f, variable=self.heatmap_on).pack(side="left"))
        row("OCR enabled", lambda f: ttk.Checkbutton(f, variable=self.ocr_on).pack(side="left"))
        row("Speech enabled", lambda f: ttk.Checkbutton(f, variable=self.speech_on).pack(side="left"))
        if self.source_mode == "video":
            row("Playback speed", lambda f: ttk.Scale(f, variable=self.speed_v,
                from_=0.25, to=4.0, orient="h", length=140).pack(side="left"))

        tk.Button(parent, text="Apply Settings", font=(_SANS, 10, "bold"),
                  bg=cfg.C_ACCENT2, fg=cfg.C_WHITE, relief="flat", padx=14, pady=6,
                  command=self._apply).pack(pady=14)

    def _build_session_tab(self, parent):
        pad = {"padx": 16, "pady": 6}

        def stat_row(label, attr, color=cfg.C_WHITE):
            f = tk.Frame(parent, bg=cfg.C_PANEL)
            f.pack(fill="x", **pad)
            tk.Label(f, text=label, font=(_SANS, 10), fg=cfg.C_MUTED,
                     bg=cfg.C_PANEL).pack(side="left")
            w = tk.Label(f, text="0", font=(_MONO, 12, "bold"), fg=color, bg=cfg.C_PANEL)
            w.pack(side="right")
            setattr(self, attr, w)

        tk.Label(parent, text="Session Statistics", font=(_SANS, 12, "bold"),
                 fg=cfg.C_WHITE, bg=cfg.C_PANEL).pack(pady=(16, 8))
        stat_row("Total alerts issued", "ss_total")
        stat_row("Danger alerts",       "ss_danger", cfg.C_DANGER)
        stat_row("Warning alerts",      "ss_warn",   cfg.C_WARN)
        stat_row("Unique object types", "ss_objects", cfg.C_ACCENT)

    # ─────────────────────────── SESSION CONTROL ──────────────────────────────

    def start(self):
        if self.running:
            return

        if self.source_mode == "pi":
            # Start UDP receiver
            self._rx = FrameReceiver(port=self.port_v.get(), timeout=cfg.STREAM_TIMEOUT_S)
            self._rx.start()
            self._log("INFO", f"Pi stream receiver started on UDP port {self.port_v.get()}")
            self._log("INFO", "Waiting for frames from Raspberry Pi…")

        elif self.source_mode == "camera":
            cap = cv2.VideoCapture(self.cam_v.get(), cv2.CAP_ANY)
            if not cap.isOpened():
                messagebox.showerror("Camera Error", f"Cannot open camera index {self.cam_v.get()}")
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg.FRAME_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.FRAME_H)
            cap.set(cv2.CAP_PROP_FPS,          cfg.TARGET_FPS)
            cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
            self._cap = cap

        elif self.source_mode == "video":
            path = filedialog.askopenfilename(
                title="Select video file",
                filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
            )
            if not path:
                return
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                messagebox.showerror("Video Error", f"Cannot open: {path}")
                return
            self.video_path          = path
            self.video_total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._video_fps          = cap.get(cv2.CAP_PROP_FPS) or 25
            self._cap                = cap
            self.scrub.configure(to=max(1, self.video_total_frames))
            dur = int(self.video_total_frames / self._video_fps)
            self.video_dur_str = f"{dur//60}:{dur%60:02d}"
            self.scrub_frame.pack(fill="x", pady=(4, 0))
            self.mode_lbl.config(text=f"VIDEO · {path.split('/')[-1][:20]}", fg=cfg.C_VIDEO)

        self.running = True
        self.paused  = False
        self.frame_n = 0

        self.s_dot.config(fg=cfg.C_OK)
        self.s_lbl.config(text="LIVE", fg=cfg.C_OK)
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.pause_btn.config(state="normal", text="⏸  Pause")

        threading.Thread(target=self._loop, daemon=True, name="vision-loop").start()

    def stop(self):
        self.running = False
        self.paused  = False
        clear_queue()

        if self._rx:
            self._rx.stop()
            self._rx = None
        if self._cap:
            self._cap.release()
            self._cap = None

        self.vid.config(image="", text="Session stopped.\nPress ▶ Start to begin.",
                        fg=cfg.C_MUTED)
        self.s_dot.config(fg=cfg.C_DANGER)
        self.s_lbl.config(text="OFFLINE", fg=cfg.C_MUTED)
        self.flash.config(text="", bg=cfg.C_BG, fg=cfg.C_BG)
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.pause_btn.config(state="disabled")
        if self.source_mode == "video":
            self.scrub_frame.pack_forget()
        self._log("INFO", "Session stopped.")

    def _pause(self):
        if not self.running:
            return
        self.paused = not self.paused
        self.pause_btn.config(text="▶  Resume" if self.paused else "⏸  Pause")
        self.s_lbl.config(text="PAUSED" if self.paused else "LIVE",
                          fg=cfg.C_WARN if self.paused else cfg.C_OK)
        self.s_dot.config(fg=cfg.C_WARN if self.paused else cfg.C_OK)

    def _mute(self):
        self.speech_on.set(not self.speech_on.get())
        self._log("INFO", "Speech " + ("enabled" if self.speech_on.get() else "muted"))

    def _clear(self):
        self.log.config(state="normal")
        self.log.delete("1.0", "end")
        self.log.config(state="disabled")

    def _apply(self):
        cfg.THREAT_SCORE_THRESHOLD = self.thresh_v.get()
        self.engine.update_thresholds(self.danger_v.get(), self.warn_v.get())
        cfg.DETECTION_CONF = self.conf_v.get()
        self._log("INFO", f"Settings applied — danger {self.danger_v.get():.1f}m  "
                          f"warn {self.warn_v.get():.1f}m  "
                          f"threshold {self.thresh_v.get():.2f}")

    def _quit(self):
        self.stop()
        self.root.destroy()

    # ─────────────────────────── MAIN LOOP ────────────────────────────────────

    def _loop(self):
        frame_interval = 1.0 / max(cfg.TARGET_FPS, 1)

        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue

            t0 = time.perf_counter()
            frame = self._get_frame()

            if frame is None:
                # Pi not yet sending — show waiting message
                if self.source_mode == "pi":
                    rx_ok = self._rx and self._rx.is_connected()
                    if not rx_ok:
                        time.sleep(0.1)
                        continue
                else:
                    self.root.after(0, self._cam_error)
                    break

            self.frame_n += 1

            # Detection
            detections = detect_objects(frame)
            active_ids = {d["track_id"] for d in detections if d["track_id"] >= 0}
            cleanup_stale_tracks(active_ids)

            # OCR (every N frames)
            if self.ocr_on.get() and self.frame_n % cfg.OCR_EVERY_N == 0:
                self.last_texts = run_ocr(frame)

            # Scoring + decision
            scored     = self.engine.get_all_scores(detections)
            messages   = self.engine.process(detections, self.last_texts)
            scene_info = self.engine.get_scene_info()

            # TTS
            if self.speech_on.get():
                for prio, tag, msg, *_ in messages:
                    speak(msg, prio)

            # Annotate
            display = annotate(frame.copy(), scored, self.last_texts)
            if self.heatmap_on.get() and scored:
                hm = make_heatmap(frame.shape, scored)
                display = cv2.addWeighted(display, 0.65, hm, 0.35, 0)

            # Overlay Pi stream indicator
            if self.source_mode == "pi" and self._rx:
                rx_fps = self._rx.fps()
                cv2.putText(display, f"Pi stream {rx_fps:.0f}fps",
                            (4, 16), cv2.FONT_HERSHEY_DUPLEX, 0.38, (232, 67, 147), 1, cv2.LINE_AA)

            # FPS
            elapsed = time.perf_counter() - t0
            self.fps_buf.append(elapsed)
            fps = 1.0 / (sum(self.fps_buf) / len(self.fps_buf))

            # Stats
            for prio, tag, msg, *_ in messages:
                self.stat_total_alerts += 1
                if tag == "DANGER":  self.stat_danger_count += 1
                if tag == "WARNING": self.stat_warn_count   += 1
            peak = max((d.get("threat_score", 0) for d in scored), default=0)
            self.stat_score_history.append(peak)

            self.root.after(0, self._ui_update, display, scored, messages, fps, scene_info)

            # Pace loop (no sleeping in Pi mode — governed by frame arrival rate)
            if self.source_mode != "pi":
                spent = time.perf_counter() - t0
                wait  = frame_interval - spent
                if wait > 0:
                    time.sleep(wait)

    def _get_frame(self):  # returns Optional[np.ndarray]
        """Fetch next frame from the appropriate source."""
        if self.source_mode == "pi":
            return self._rx.get_frame() if self._rx else None
        else:
            if self._cap is None:
                return None
            ret, frame = self._cap.read()
            if not ret:
                return None
            if frame.shape[:2] != (cfg.FRAME_H, cfg.FRAME_W):
                frame = cv2.resize(frame, (cfg.FRAME_W, cfg.FRAME_H))
            return frame

    def _cam_error(self):
        self._log("DANGER", "Camera feed lost. Press Start to reconnect.")
        self.stop()

    def _update_scrubber(self, pos: int):
        self.video_pos_var.set(pos)
        fps_v = self._video_fps or 30
        elapsed = int(pos / fps_v)
        mins, secs = divmod(elapsed, 60)
        self.scrub_lbl.config(text=f"{mins}:{secs:02d} / {self.video_dur_str}")

    # ─────────────────────────── UI UPDATE ────────────────────────────────────

    def _ui_update(self, display, scored, messages, fps, scene_info):
        if not self.running:
            return

        # Frame
        rgb   = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self.vid.imgtk = imgtk
        self.vid.configure(image=imgtk, text="")

        # Stats
        nearest  = min((d["distance"] for d in scored if d["distance"] is not None), default=None)
        peak     = max((d.get("threat_score", 0) for d in scored), default=0)
        adap_thr = scene_info["adaptive_threshold"]
        sc_col   = cfg.C_DANGER if peak >= 0.75 else (cfg.C_WARN if peak >= 0.45 else cfg.C_OK)

        self.w_fps.config(text=f"{fps:.0f}")
        self.w_objs.config(text=str(len(scored)))
        self.w_near.config(text=f"{nearest:.1f}m" if nearest else "—")
        self.w_score.config(text=f"{int(peak * 100)}%", fg=sc_col)
        self.w_thresh.config(text=f"{adap_thr:.2f}")

        # Scene bar
        density = scene_info["density"]
        den_col = {"SPARSE": cfg.C_OK, "NORMAL": cfg.C_ACCENT, "CROWDED": cfg.C_WARN}[density]
        approaching = [d for d in scored if d.get("approach_rate", 0) > cfg.MIN_APPROACH_RATE]
        rx_info = f"  ·  Pi {self._rx.fps():.0f}fps" if (self.source_mode == "pi" and self._rx) else ""
        self.scene_bar.config(
            text=f"Scene: {density}  ·  Threshold: {adap_thr:.2f}  ·  {len(approaching)} approaching{rx_info}",
            fg=den_col
        )

        # Alerts
        for item in messages:
            self._log(item[1], item[2])

        # Threat panel
        self._update_score_panel(scored)

        # Session stats
        self.ss_total.config(text=str(self.stat_total_alerts))
        self.ss_danger.config(text=str(self.stat_danger_count))
        self.ss_warn.config(text=str(self.stat_warn_count))
        self.ss_objects.config(text=str(len({d["label"] for d in scored})))

        # Flash bar
        has_danger = any(i[1] == "DANGER"  for i in messages)
        has_warn   = any(i[1] == "WARNING" for i in messages)
        if has_danger:
            self.flash.config(text="  ⚠  DANGER — STOP IMMEDIATELY  ⚠  ",
                              bg=cfg.C_DANGER, fg="#FFFFFF")
        elif has_warn:
            self.flash.config(text="  ⚡  WARNING — OBSTACLE NEARBY  ⚡  ",
                              bg=cfg.C_WARN, fg="#1A0E00")
        else:
            self.flash.config(text="", bg=cfg.C_BG, fg=cfg.C_BG)

    def _update_score_panel(self, scored):
        for w in self.score_frame.winfo_children():
            w.destroy()
        if not scored:
            tk.Label(self.score_frame, text="No objects detected.",
                     font=(_MONO, 10), fg=cfg.C_MUTED, bg=cfg.C_PANEL).pack(pady=20)
            return
        for d in scored:
            label    = d["label"]
            score    = d.get("threat_score", 0.0)
            dist     = d["distance"]
            approach = d.get("approach_rate", 0.0)
            ttc      = d.get("ttc")
            occ      = d.get("is_occluded", False)
            bkdn     = d.get("breakdown", {})

            sc = cfg.C_DANGER if score >= 0.75 else (cfg.C_WARN if score >= 0.45 else cfg.C_OK)
            row = tk.Frame(self.score_frame, bg=cfg.C_CARD,
                           highlightbackground=cfg.C_BORDER, highlightthickness=1)
            row.pack(fill="x", pady=2)
            top = tk.Frame(row, bg=cfg.C_CARD)
            top.pack(fill="x", padx=10, pady=(6, 2))

            lbl_txt = label.upper() + (" [OCCLUDED]" if occ else "")
            tk.Label(top, text=lbl_txt, font=(_SANS, 10, "bold"),
                     fg=cfg.C_MUTED if occ else cfg.C_WHITE, bg=cfg.C_CARD).pack(side="left")

            arr_txt = f"  ↑ {approach:.2f}m/s" if approach > cfg.MIN_APPROACH_RATE \
                      else (f"  ↓ receding" if approach < -0.05 else "  · still")
            tk.Label(top, text=arr_txt, font=(_MONO, 8),
                     fg=cfg.C_ACCENT if approach > cfg.MIN_APPROACH_RATE else cfg.C_MUTED,
                     bg=cfg.C_CARD).pack(side="left")

            if ttc is not None:
                ttc_col = cfg.C_DANGER if ttc <= 2.5 else (cfg.C_WARN if ttc <= 6 else cfg.C_MUTED)
                tk.Label(top, text=f" TTC {ttc:.1f}s ", font=(_MONO, 8, "bold"),
                         fg=ttc_col, bg=cfg.C_CARD).pack(side="left", padx=4)

            ds = f"{dist:.1f}m" if dist else "?"
            tk.Label(top, text=f"{ds}    {int(score * 100)}%",
                     font=(_MONO, 10, "bold"), fg=sc, bg=cfg.C_CARD).pack(side="right")

            bar_f = tk.Frame(row, bg=cfg.C_SURFACE, height=5)
            bar_f.pack(fill="x", padx=10, pady=(0, 3))
            bar_f.pack_propagate(False)
            bar_f.update_idletasks()
            bw = bar_f.winfo_reqwidth() or 200
            fw = max(2, int(bw * score))
            tk.Frame(bar_f, bg=sc, width=fw, height=5).place(x=0, y=0)

            if bkdn:
                mini = tk.Frame(row, bg=cfg.C_CARD)
                mini.pack(fill="x", padx=10, pady=(0, 5))
                for short, key in [("dist", "distance_score"), ("appr", "approach_score"),
                                   ("ttc", "ttc_score"), ("pers", "persistence_score"),
                                   ("pen", "stationary_penalty")]:
                    v = bkdn.get(key, 0)
                    tk.Label(mini, text=f"{short}:{int(v * 100)}",
                             font=(_MONO, 7), fg=cfg.C_MUTED, bg=cfg.C_CARD, padx=3).pack(side="left")

    # ─────────────────────────── LOG ──────────────────────────────────────────

    def _log(self, tag: str, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.log.config(state="normal")
        self.log.insert("end", f"[{ts}] ", "TIME")
        self.log.insert("end", f"[{tag}]", tag)
        self.log.insert("end", f"  {msg}\n")
        self.log.see("end")
        self.log.config(state="disabled")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VisionAssist v5 — Pi-Laptop Split")
    parser.add_argument("--source", choices=["pi", "camera", "video"], default="pi",
                        help="Frame source: pi (UDP stream), camera (local webcam), video (file)")
    args = parser.parse_args()

    root = tk.Tk()
    App(root, source_mode=args.source)
    root.mainloop()