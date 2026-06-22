"""
frame_receiver.py — VisionAssist Laptop Side
UDP frame receiver that reassembles chunked JPEG frames sent by the Pi.

Runs in a background thread and exposes the latest complete frame
via get_frame() — zero-copy via a shared buffer protected by a lock.

Compatible with Python 3.7+
"""

import socket
import struct
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional

import cv2
import numpy as np

MAGIC      = b"VAPI"
HEADER_LEN = 4 + 2 + 2 + 2 + 2   # magic + frame_id + chunk_idx + total_chunks + data_len


class FrameReceiver:
    """
    Binds a UDP socket on `port`, reassembles chunked JPEG frames from the Pi,
    and makes the latest decoded BGR frame available via get_frame().

    Usage:
        rx = FrameReceiver(port=5000)
        rx.start()
        ...
        frame = rx.get_frame()   # np.ndarray (BGR) or None
        rx.stop()
    """

    def __init__(self, port: int = 5000, timeout: float = 5.0):
        self._port    = port
        self._timeout = timeout  # seconds without data before marking disconnected

        self._frame   = None                  # type: Optional[np.ndarray]
        self._lock    = threading.Lock()
        self._running = False
        self._thread  = None                  # type: Optional[threading.Thread]

        self._last_rx   = 0.0                 # timestamp of last received packet
        self._fps_buf   = []                  # type: List[float]
        self._frame_cnt = 0

        # In-flight chunk buffers: frame_id → {chunk_idx: bytes, "_total": int}
        self._pending = defaultdict(dict)     # type: Dict[int, dict]

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self._recv_loop, daemon=True, name="udp-rx"
        )
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def get_frame(self):
        # type: () -> Optional[np.ndarray]
        """Return the latest complete BGR frame, or None if none received yet."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def is_connected(self):
        # type: () -> bool
        """True if a packet was received within the timeout window."""
        return self._running and (time.time() - self._last_rx) < self._timeout

    def fps(self):
        # type: () -> float
        """Estimated frames-per-second based on recent frame arrivals."""
        now = time.time()
        self._fps_buf = [t for t in self._fps_buf if now - t < 2.0]
        return len(self._fps_buf) / 2.0

    def frame_count(self):
        # type: () -> int
        return self._frame_cnt

    # ── Internal ───────────────────────────────────────────────────────────────

    def _recv_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        sock.settimeout(1.0)   # 1s poll so we can check self._running
        try:
            sock.bind(("0.0.0.0", self._port))
            print("[rx] Listening on UDP port {}".format(self._port))
        except OSError as e:
            print("[rx] Bind failed on port {}: {}".format(self._port, e))
            self._running = False
            return

        while self._running:
            try:
                data, _ = sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                break

            if len(data) < HEADER_LEN:
                continue

            if data[:4] != MAGIC:
                continue

            frame_id, chunk_idx, total_chunks, data_len = struct.unpack(
                ">HHHH", data[4:12]
            )
            self._last_rx = time.time()

            # Heartbeat packet — total_chunks == 0 is the sentinel
            if total_chunks == 0:
                continue

            payload = data[HEADER_LEN: HEADER_LEN + data_len]
            bucket  = self._pending[frame_id]
            bucket[chunk_idx] = payload
            bucket.setdefault("_total", total_chunks)

            # Check if all chunks for this frame have arrived
            if len(bucket) - 1 == bucket["_total"]:   # -1 accounts for "_total" key
                chunks = [bucket[i] for i in range(bucket["_total"])]
                jpeg   = b"".join(chunks)
                del self._pending[frame_id]

                # Evict stale incomplete frames using modular arithmetic so
                # frame_id wraparound at 65535 → 0 is handled correctly
                stale = [
                    fid for fid in list(self._pending)
                    if ((frame_id - fid) & 0xFFFF) > 10
                ]
                for fid in stale:
                    del self._pending[fid]

                self._decode_and_store(jpeg)

        sock.close()

    def _decode_and_store(self, jpeg_bytes):
        # type: (bytes) -> None
        arr   = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return
        with self._lock:
            self._frame = frame
        self._frame_cnt += 1
        self._fps_buf.append(time.time())