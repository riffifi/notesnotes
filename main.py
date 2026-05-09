#!/usr/bin/env python3
"""
notes.py — real-time note display (clean + styled + stable)
"""

import sys, time, math, shutil, subprocess
from pathlib import Path
from datetime import datetime
import numpy as np
import sounddevice as sd
from collections import deque

# ── config ─────────────────────────────────────────────
SAMPLE_RATE  = 44100
BLOCK_SIZE   = 8192
OVERLAP      = 2048
MIN_FREQ     = 60.0
MAX_FREQ     = 1400.0
RMS_THRESH   = 0.008
MIN_CONF     = 0.70
SMOOTH       = 3
HOLD         = 0.08
MAX_HIST     = 8

SESSIONS_DIR = Path.home() / ".notesnotes"
REPO_URL     = "https://github.com/riffifi/notesnotes.git"

CHROMATIC = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

# ── ANSI colors ────────────────────────────────────────
R   = "\033[0m"
B   = "\033[1m"

def fg(r,g,b): return f"\033[38;2;{r};{g};{b}m"

GLOW  = fg(255, 210, 120)
CREAM = fg(240, 235, 220)
SOFT  = fg(180, 170, 150)
DIM   = fg(120, 110, 100)
FADE  = fg(70, 65, 60)

# ── big ASCII letters ──────────────────────────────────
BIG = {
    'A': ["  ▄▄▄  "," █   █ "," █   █ "," █████ "," █   █ "," █   █ "," █   █ ","       "],
    'B': [" ████  "," █   █ "," █   █ "," ████  "," █   █ "," █   █ "," ████  ","       "],
    'C': ["  ████ "," █     "," █     "," █     "," █     "," █     ","  ████ ","       "],
    'D': [" ████  "," █   █ "," █   █ "," █   █ "," █   █ "," █   █ "," ████  ","       "],
    'E': [" █████ "," █     "," █     "," ████  "," █     "," █     "," █████ ","       "],
    'F': [" █████ "," █     "," █     "," ████  "," █     "," █     "," █     ","       "],
    'G': ["  ████ "," █     "," █     "," █  ██ "," █   █ "," █   █ ","  ████ ","       "],
}

SHARP = [
    " ▗▖  ",
    " ███ ",
    " ▝▘  ",
    " ███ ",
    " ▗▖  ",
    "     ",
    "     ",
    "     ",
]

# ── note detection ─────────────────────────────────────
def closest_note(freq):
    midi = round(12 * math.log2(freq / 440.0) + 69)
    return CHROMATIC[midi % 12] + str(midi // 12 - 1)

def detect(sig):
    rms = float(np.sqrt(np.mean(sig**2)))
    if rms < RMS_THRESH:
        return None, 0

    w = sig * np.hanning(len(sig))
    n = len(w)

    fft = np.fft.rfft(w, n=n*2)
    acf = np.fft.irfft(fft * np.conj(fft))[:n]
    acf /= (acf[0] + 1e-9)

    lo = int(SAMPLE_RATE / MAX_FREQ)
    hi = min(int(SAMPLE_RATE / MIN_FREQ), n-1)

    idx = np.argmax(acf[lo:hi])
    conf = float(acf[lo + idx])
    freq = SAMPLE_RATE / (lo + idx)

    return (freq if conf >= MIN_CONF else None), conf

# ── state ──────────────────────────────────────────────
ring        = np.zeros(BLOCK_SIZE, dtype=np.float32)
freq_hist   = deque(maxlen=SMOOTH)
live        = [None]
history     = []
session_all = []
last_note   = [None]
stable_t    = [0.0]

# ── audio callback ─────────────────────────────────────
def callback(indata, frames, _t, _s):
    global ring

    chunk = indata[:,0]
    ring = np.roll(ring, -len(chunk))
    ring[-len(chunk):] = chunk

    freq, _ = detect(ring)
    if freq is None:
        live[0] = None
        stable_t[0] = 0
        return

    freq_hist.append(freq)
    if len(freq_hist) < 2:
        return

    name = closest_note(float(np.median(freq_hist)))
    live[0] = name

    now = time.time()

    if name != last_note[0]:
        if stable_t[0] == 0:
            stable_t[0] = now
        elif now - stable_t[0] > HOLD:
            history.insert(0, name)
            history[:] = history[:MAX_HIST]
            session_all.append(name)
            last_note[0] = name
            stable_t[0] = 0
    else:
        stable_t[0] = 0

# ── terminal helpers ───────────────────────────────────
def tw(): return shutil.get_terminal_size((80,24)).columns
def th(): return shutil.get_terminal_size((80,24)).lines

def fade(i):
    return 1.0 - abs(3.5 - i) / 6

# ── render ─────────────────────────────────────────────
def render():
    cols = tw()
    rows = th()

    cur = live[0]
    note = cur[:-1] if cur else None
    letter = note[0] if note else None
    sharp = note and "#" in note if note else False
    octv = cur[-1] if cur else ""

    art = BIG.get(letter, [" "]*8)

    hist_w = 7
    gap = 3
    art_w = len(art[0])

    total_w = hist_w + gap + art_w + 6
    pad_x = max(0, (cols - total_w)//2)
    pad_y = max(0, (rows - 8)//2)

    lines = [" " * cols for _ in range(pad_y)]

    for i in range(8):

        # ── history (soft fade) ───────────────────
        if i < len(history):
            h = history[i][:-1]

            if i == 0:
                col = GLOW + B
            elif i == 1:
                col = CREAM
            elif i == 2:
                col = SOFT
            else:
                col = FADE

            hist = col + f"{h:<3}" + R
            hist = hist.ljust(hist_w + len(col) + len(R))
        else:
            hist = " " * hist_w

        # ── big letter glow gradient ──────────────
        f = fade(i)
        if f > 0.75:
            col = GLOW + B
        elif f > 0.55:
            col = CREAM
        else:
            col = SOFT

        big = col + art[i] + R
        sharp_part = (DIM + SHARP[i] + R) if sharp else ""

        octo = (GLOW + B + " " + octv + R) if i == 3 and octv else ""

        row = (
            " " * pad_x +
            hist +
            " " * gap +
            big +
            sharp_part +
            octo
        )

        lines.append(row.ljust(cols))

    return "\n".join(lines)

def draw(frame):
    sys.stdout.write("\033[H\033[J" + frame)
    sys.stdout.flush()

# ── git auto-fix (no upstream issues ever again) ──────
def save_and_push():
    if not session_all:
        return

    SESSIONS_DIR.mkdir(exist_ok=True)

    if not (SESSIONS_DIR / ".git").exists():
        subprocess.run(["git", "init", "-b", "main"], cwd=SESSIONS_DIR)

    subprocess.run(["git", "remote", "remove", "origin"], cwd=SESSIONS_DIR, capture_output=True)
    subprocess.run(["git", "remote", "add", "origin", REPO_URL], cwd=SESSIONS_DIR)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    f = SESSIONS_DIR / f"{ts}.txt"
    f.write_text(" ".join(session_all))

    subprocess.run(["git", "add", "."], cwd=SESSIONS_DIR)
    subprocess.run(["git", "commit", "-m", "session"], cwd=SESSIONS_DIR)

    subprocess.run(["git", "branch", "-M", "main"], cwd=SESSIONS_DIR)
    subprocess.run(["git", "push", "-u", "origin", "main"], cwd=SESSIONS_DIR)

# ── main loop ──────────────────────────────────────────
def main():
    sys.stdout.write("\033[?25l")
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=OVERLAP,
            channels=1,
            dtype="float32",
            callback=callback,
        ):
            while True:
                draw(render())
                time.sleep(0.03)
    except KeyboardInterrupt:
        pass
    finally:
        save_and_push()
        sys.stdout.write("\033[?25h\n")

if __name__ == "__main__":
    main()
