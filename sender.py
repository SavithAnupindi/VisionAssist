"""
stream_sender.py — VisionAssist Pi Side (USB Camera, Hardened)

Run on Raspberry Pi:
    python3 stream_sender.py --host <LAPTOP_IP> --port 5000

You can also pass a hostname if it resolves:
    python3 stream_sender.py --host Savith.local --port 5000

Requirements (install once on Pi):
    pip install opencv-python-headless numpy

Protocol (matches frame_receiver.py exactly):
    Each UDP packet:
        MAGIC       4 bytes  b"VAPI"
        frame_id    2 bytes  big-endian uint16  (wraps at 65535)
        chunk_idx   2 bytes  big-endian uint16
        total_chunks 2 bytes big-endian uint16  (0 = heartbeat sentinel)
        data_len    2 bytes  big-endian uint16
        payload     N bytes  JPEG slice
"""

import argparse
import socket
import struct
import time
import sys
import traceback

import cv2
import numpy as np

# ── Constants (must match frame_receiver.py) ──────────────────────────────────
MAGIC        = b"VAPI"
CHUNK_SIZE   = 60000          # bytes per UDP payload — safely under 65507 MTU
HEARTBEAT_S  = 2.0            # keepalive interval in seconds

# ── Stream settings ───────────────────────────────────────────────────────────
FRAME_W      = 640
FRAME_H      = 480
TARGET_FPS   = 20             # Pi 3B+ reliable ceiling with USB cam + encode
JPEG_QUALITY = 70             # 70 is sharp enough, keeps packets small


# ─────────────────────────────────────────────────────────────────────────────
# Camera
# ─────────────────────────────────────────────────────────────────────────────

def open_camera() -> cv2.VideoCapture:
    """
    Try V4L2 backend first (native on Pi), fall back to auto-detect.
    Searches indices 0-3. Raises RuntimeError if no camera found.
    """
    backends = [cv2.CAP_V4L2, cv2.CAP_ANY]

    for idx in range(4):
        for backend in backends:
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                cap.release()
                continue

            # Push settings — camera may ignore some, that is fine
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
            cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)

            # Keep internal buffer at 1 frame — avoids stale-frame latency
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            # Warm-up read to confirm the camera actually delivers frames
            for _ in range(3):
                ret, _ = cap.read()
                if ret:
                    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    backend_name = "V4L2" if backend == cv2.CAP_V4L2 else "AUTO"
                    print(f"[pi] Camera {idx} ({backend_name}) opened — {w}x{h} @ {fps:.0f}fps")
                    return cap

            cap.release()

    raise RuntimeError("No USB camera found on indices 0-3.")


# ─────────────────────────────────────────────────────────────────────────────
# UDP streamer
# ─────────────────────────────────────────────────────────────────────────────

class FrameStreamer:
    """
    Encodes frames as JPEG and sends them as chunked UDP packets.

    The host name/IP is resolved ONCE at construction — not on every send.
    This eliminates repeated DNS lookups when passing a hostname like "Savith".
    """

    def __init__(self, host: str, port: int):
        # Resolve hostname → IP address once, fail fast if unreachable
        try:
            info      = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_DGRAM)
            self._dst = info[0][4]   # (ip_str, port) tuple
        except socket.gaierror as e:
            raise RuntimeError(
                f"Cannot resolve host '{host}': {e}\n"
                f"  → Pass the laptop's numeric IP (e.g. 192.168.1.42) to avoid DNS issues."
            ) from e

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Request a large send buffer; OS may grant less — that is acceptable
        try:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
        except OSError:
            pass   # Some systems cap this — streaming still works

        self._frame_id     = 0
        self._encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]

        print(f"[pi] UDP destination resolved → {self._dst[0]}:{self._dst[1]}")

    def send_frame(self, frame: np.ndarray) -> bool:
        """
        JPEG-encode frame and send in CHUNK_SIZE slices.
        Returns True on success, False if encode fails.
        """
        ok, buf = cv2.imencode(".jpg", frame, self._encode_params)
        if not ok:
            return False

        data         = buf.tobytes()
        total_chunks = (len(data) + CHUNK_SIZE - 1) // CHUNK_SIZE
        frame_id     = self._frame_id & 0xFFFF   # wraps at 65535

        for idx in range(total_chunks):
            chunk  = data[idx * CHUNK_SIZE : (idx + 1) * CHUNK_SIZE]
            header = MAGIC + struct.pack(">HHHH", frame_id, idx, total_chunks, len(chunk))
            try:
                self._sock.sendto(header + chunk, self._dst)
            except OSError as e:
                print(f"[pi] Send error chunk {idx}/{total_chunks}: {e}")
                return False

        self._frame_id += 1
        return True

    def send_heartbeat(self):
        """Lightweight keepalive — total_chunks=0 tells the receiver it is not a frame."""
        pkt = MAGIC + struct.pack(">HHHH", 0xFFFF, 0, 0, 0)
        try:
            self._sock.sendto(pkt, self._dst)
        except OSError:
            pass

    def close(self):
        self._sock.close()


# ─────────────────────────────────────────────────────────────────────────────
# FPS tracker
# ─────────────────────────────────────────────────────────────────────────────

class FPSTracker:
    """Rolling-window FPS — never divides by zero."""

    def __init__(self, window: float = 5.0):
        self._window  = window
        self._times: list[float] = []

    def tick(self):
        now = time.time()
        self._times.append(now)
        cutoff = now - self._window
        self._times = [t for t in self._times if t >= cutoff]

    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / span if span > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def run(host: str, port: int):
    # ── Camera ────────────────────────────────────────────────────────────────
    try:
        cap = open_camera()
    except RuntimeError as e:
        print(f"[pi] FATAL: {e}")
        sys.exit(1)

    # ── Streamer (resolves host once here) ────────────────────────────────────
    try:
        streamer = FrameStreamer(host, port)
    except RuntimeError as e:
        print(f"[pi] FATAL: {e}")
        cap.release()
        sys.exit(1)

    frame_interval = 1.0 / TARGET_FPS
    last_heartbeat = time.time()
    fps_tracker    = FPSTracker(window=5.0)
    last_fps_print = time.time()

    print(f"[pi] Streaming {FRAME_W}x{FRAME_H} @ target {TARGET_FPS}fps  quality={JPEG_QUALITY}%")
    print(f"[pi] Press Ctrl+C to stop.")

    try:
        while True:
            t0 = time.perf_counter()

            # ── Capture ───────────────────────────────────────────────────────
            ret, frame = cap.read()

            if not ret:
                # Camera stalled — attempt single reconnect before giving up
                print("[pi] Frame read failed — attempting camera reconnect…")
                cap.release()
                time.sleep(1.0)
                try:
                    cap = open_camera()
                    print("[pi] Camera reconnected.")
                    continue
                except RuntimeError:
                    print("[pi] Reconnect failed. Exiting.")
                    break

            # Resize only if camera ignored our settings
            if frame.shape[:2] != (FRAME_H, FRAME_W):
                frame = cv2.resize(frame, (FRAME_W, FRAME_H), interpolation=cv2.INTER_LINEAR)

            # ── Send ──────────────────────────────────────────────────────────
            streamer.send_frame(frame)
            fps_tracker.tick()

            # ── Heartbeat ─────────────────────────────────────────────────────
            now = time.time()
            if now - last_heartbeat >= HEARTBEAT_S:
                streamer.send_heartbeat()
                last_heartbeat = now

            # ── FPS report every 5 s ──────────────────────────────────────────
            if now - last_fps_print >= 5.0:
                print(f"[pi] Stream FPS: {fps_tracker.fps():.1f}")
                last_fps_print = now

            # ── Pace to TARGET_FPS ────────────────────────────────────────────
            elapsed    = time.perf_counter() - t0
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[pi] Stopped by user.")
    except Exception:
        traceback.print_exc()
    finally:
        cap.release()
        streamer.close()
        print("[pi] Resources released.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VisionAssist — Raspberry Pi USB Camera Streamer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 stream_sender.py --host 192.168.1.42 --port 5000
  python3 stream_sender.py --host Savith.local  --port 5000

Tip: use the numeric IP for guaranteed connectivity.
Find your laptop IP with:
  Windows → ipconfig
  Linux   → ip addr show
  macOS   → ifconfig
        """
    )
    parser.add_argument("--host", required=True,
                        help="Laptop IP address or hostname")
    parser.add_argument("--port", type=int, default=5000,
                        help="UDP port (default: 5000, must match laptop)")
    args = parser.parse_args()
    run(args.host, args.port)