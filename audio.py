"""
audio.py — Thread-safe TTS with Windows SAPI5 fix.
Fresh engine per utterance eliminates the silent-after-first-alert bug.
"""

import threading
import queue
import time

PRIORITY_CRITICAL = 0
PRIORITY_WARNING  = 1
PRIORITY_INFO     = 2

_q       = queue.PriorityQueue()
_counter = 0
_lock    = threading.Lock()

_RATES = {
    PRIORITY_CRITICAL: 170,
    PRIORITY_WARNING:  158,
    PRIORITY_INFO:     150,
}


def _make_engine():
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty("volume", 1.0)
    voices = engine.getProperty("voices") or []
    for v in voices:
        name = (v.name or "").lower()
        vid  = (v.id  or "").lower()
        if any(k in name or k in vid for k in
               ("zira", "david", "hazel", "english", "en-us", "en_us")):
            engine.setProperty("voice", v.id)
            break
    return engine


def _worker():
    while True:
        try:
            priority, _seq, text = _q.get(timeout=0.5)
        except queue.Empty:
            continue

        if priority >= PRIORITY_INFO and not _q.empty():
            try:
                next_item = _q.queue[0]
                if next_item[0] < PRIORITY_INFO:
                    _q.task_done()
                    continue
            except (IndexError, AttributeError):
                pass

        engine = None
        try:
            engine = _make_engine()
            engine.setProperty("rate", _RATES.get(priority, 155))
            engine.say(text)
            engine.runAndWait()
        except Exception as exc:
            print(f"[audio] TTS error: {exc}")
            time.sleep(0.1)
        finally:
            if engine is not None:
                try:
                    engine.stop()
                except Exception:
                    pass
                del engine

        _q.task_done()


threading.Thread(target=_worker, daemon=True, name="tts-worker").start()


def speak(text: str, priority: int = PRIORITY_INFO):
    global _counter
    with _lock:
        _counter += 1
        seq = _counter
    _q.put((priority, seq, text))


def speak_urgent(text: str):
    speak(text, PRIORITY_CRITICAL)


def speak_warning(text: str):
    speak(text, PRIORITY_WARNING)


def clear_queue():
    while not _q.empty():
        try:
            _q.get_nowait()
            _q.task_done()
        except queue.Empty:
            break