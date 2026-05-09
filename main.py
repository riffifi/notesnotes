#!/usr/bin/env python3
"""
notes.py — real-time note display
pip install numpy sounddevice
ctrl-c to quit and save session
"""

import sys, time, math, shutil, os, subprocess
from pathlib import Path
from datetime import datetime
import numpy as np
import sounddevice as sd
from collections import deque

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

R   = "\033[0m"
B   = "\033[1m"
DIM = "\033[2m"
def fg(r,g,b): return f"\033[38;2;{r};{g};{b}m"

CREAM  = fg(240, 235, 220)
GOLD   = fg(212, 175, 100)
MUTED  = fg(100,  95,  85)
FAINT  = fg(58,   54,  46)
FAINTER= fg(38,   35,  28,)

def note_color(n):
    return GOLD if '#' in n else CREAM

BIG = {
    'A': [
        "  ▄▄▄  ",
        " █   █ ",
        " █   █ ",
        " █████ ",
        " █   █ ",
        " █   █ ",
        " █   █ ",
        "       ",
    ],
    'B': [
        " ████  ",
        " █   █ ",
        " █   █ ",
        " ████  ",
        " █   █ ",
        " █   █ ",
        " ████  ",
        "       ",
    ],
    'C': [
        "  ████ ",
        " █     ",
        " █     ",
        " █     ",
        " █     ",
        " █     ",
        "  ████ ",
        "       ",
    ],
    'D': [
        " ████  ",
        " █   █ ",
        " █   █ ",
        " █   █ ",
        " █   █ ",
        " █   █ ",
        " ████  ",
        "       ",
    ],
    'E': [
        " █████ ",
        " █     ",
        " █     ",
        " ████  ",
        " █     ",
        " █     ",
        " █████ ",
        "       ",
    ],
    'F': [
        " █████ ",
        " █     ",
        " █     ",
        " ████  ",
        " █     ",
        " █     ",
        " █     ",
        "       ",
    ],
    'G': [
        "  ████ ",
        " █     ",
        " █     ",
        " █  ██ ",
        " █   █ ",
        " █   █ ",
        "  ████ ",
        "       ",
    ],
}

SHARP_ART = [
    " ▗▖  ",
    " ███ ",
    " ▝▘  ",
    " ███ ",
    " ▗▖  ",
    "     ",
    "     ",
    "     ",
]

def build_chromatic():
    t = {}
    for midi in range(24, 109):
        oct_ = midi // 12 - 1
        name = CHROMATIC[midi % 12] + str(oct_)
        freq = 440.0 * 2 ** ((midi - 69) / 12)
        if MIN_FREQ <= freq <= MAX_FREQ:
            t[name] = freq
    return t

NOTE_TABLE = build_chromatic()

def closest_note(freq):
    midi = round(12 * math.log2(freq / 440.0) + 69)
    return CHROMATIC[midi % 12] + str(midi // 12 - 1)

def detect(signal):
    rms = float(np.sqrt(np.mean(signal ** 2)))
    if rms < RMS_THRESH:
        return None, 0.0
    w   = signal * np.hanning(len(signal))
    n   = len(w)
    fft = np.fft.rfft(w, n=n*2)
    acf = np.fft.irfft(fft * np.conj(fft))[:n]
    acf /= (acf[0] + 1e-9)
    lo  = int(SAMPLE_RATE / MAX_FREQ)
    hi  = min(int(SAMPLE_RATE / MIN_FREQ), n-1)
    idx = np.argmax(acf[lo:hi])
    conf = float(acf[lo + idx])
    freq = SAMPLE_RATE / (lo + idx)
    return (freq if conf >= MIN_CONF else None), conf

ring        = np.zeros(BLOCK_SIZE, dtype=np.float32)
freq_hist   = deque(maxlen=SMOOTH)
live        = [None]
history     = []
last_logged = [None]
stable_t    = [0.0]
last_snap   = [None]
last_nlines = [0]
session_all = []

def callback(indata, frames, _t, _s):
    global ring
    chunk = indata[:,0]
    ring  = np.roll(ring, -len(chunk))
    ring[-len(chunk):] = chunk

    freq, conf = detect(ring)
    if freq is None:
        live[0] = None
        stable_t[0] = 0.0
        return

    freq_hist.append(freq)
    if len(freq_hist) < 2:
        return

    name = closest_note(float(np.median(freq_hist)))
    live[0] = name

    now = time.time()
    if name != last_logged[0]:
        if stable_t[0] == 0.0:
            stable_t[0] = now
        elif now - stable_t[0] >= HOLD:
            history.insert(0, name)
            if len(history) > MAX_HIST:
                history.pop()
            session_all.append(name)
            last_logged[0] = name
            stable_t[0] = 0.0
    else:
        stable_t[0] = 0.0

def tw(): return shutil.get_terminal_size((80,24)).columns
def th(): return shutil.get_terminal_size((80,24)).lines

ART_ROWS = 8
HIST_W   = 7

def render():
    cols = tw()
    rows = th()

    cur   = live[0]
    note  = cur[:-1] if cur else None
    oct_  = cur[-1] if cur else None
    sharp = note and '#' in note if note else False
    letter= note[0] if note else None

    letter_art = BIG.get(letter, [" " * 7] * ART_ROWS)
    art_w      = len(letter_art[0])
    sharp_w    = len(SHARP_ART[0]) if sharp else 0
    big_w      = art_w + sharp_w + 2

    hist_col_w = HIST_W
    gap = 3

    content_w = hist_col_w + gap + big_w
    lpad = max(0, (cols - content_w) // 2)
    tpad = max(0, (rows - ART_ROWS) // 2)

    lines = []

    # top padding
    for _ in range(tpad):
        lines.append(" " * cols)

    for i in range(ART_ROWS):
        pad = " " * lpad

        # history (fixed width, ANSI-safe padding)
        if i < len(history):
            hn = history[i][:-1]
            col = note_color(hn)

            raw = f"{hn:<3}  "
            hist_str = col + raw + R + (" " * (hist_col_w - len(raw)))
        else:
            hist_str = " " * hist_col_w

        sp = " " * gap

        # big letter
        if letter:
            col = note_color(note)
            big_part = col + B + letter_art[i] + R
            shp_part = (MUTED + SHARP_ART[i] + R) if sharp else ""
        else:
            big_part = " " * art_w
            shp_part = ""

        if i == 3 and oct_:
            o_part = MUTED + " " + oct_ + R
        else:
            o_part = ""

        row = pad + hist_str + sp + big_part + shp_part + o_part
        lines.append(row.ljust(cols))

    return "\n".join(lines)

def draw(frame):
    n = frame.count("\n") + 1
    if last_nlines[0]:
        sys.stdout.write(f"\033[{last_nlines[0]}A\033[J")
    sys.stdout.write(frame)
    sys.stdout.flush()
    last_nlines[0] = n

def save_and_push():
    if not session_all:
        return

    SESSIONS_DIR.mkdir(exist_ok=True)

    ts = datetime.now()
    fname = ts.strftime("%Y-%m-%d_%H-%M-%S") + ".txt"
    fpath = SESSIONS_DIR / fname

    fpath.write_text(" ".join(session_all))

    subprocess.run(["git", "add", fname], cwd=SESSIONS_DIR)
    subprocess.run(["git", "commit", "-m", "session"], cwd=SESSIONS_DIR)
    subprocess.run(["git", "push", "--force"], cwd=SESSIONS_DIR)

def main():
    sys.stdout.write("\033[2J\033[H\033[?25l")
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
