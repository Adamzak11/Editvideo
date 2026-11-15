"""Synthesize original background music + sound effects for the edit.

Everything is generated from scratch with numpy (no copyrighted samples).
Outputs: bgm.wav and sfx.wav (48 kHz stereo) in the output directory.
"""
import sys, wave
import numpy as np

SR = 48000
DUR = 29.4
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
rng = np.random.default_rng(7)


def t_arr(d):
    return np.arange(int(d * SR)) / SR


def env_ad(n, a=0.005, d=0.2):
    t = np.arange(n) / SR
    e = np.minimum(t / max(a, 1e-4), 1.0) * np.exp(-np.maximum(t - a, 0) / d)
    return e


def add(buf, sig, at, gain=1.0, pan=0.0):
    i = int(at * SR)
    if i >= len(buf):
        return
    sig = sig[: len(buf) - i]
    l = np.cos((pan + 1) * np.pi / 4)
    r = np.sin((pan + 1) * np.pi / 4)
    buf[i:i + len(sig), 0] += sig * gain * l * 1.414
    buf[i:i + len(sig), 1] += sig * gain * r * 1.414


def lowpass(x, cutoff):
    # simple one-pole, cutoff may be array
    a = np.exp(-2 * np.pi * np.asarray(cutoff) / SR) * np.ones_like(x)
    y = np.zeros_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc = (1 - a[i]) * x[i] + a[i] * acc
        y[i] = acc
    return y


def midi(n):
    return 440.0 * 2 ** ((n - 69) / 12)


# ---------------------------------------------------------------- instruments
def kick():
    t = t_arr(0.35)
    f = 50 + 110 * np.exp(-t * 30)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(ph) * np.exp(-t * 9) * 1.0


def clap():
    t = t_arr(0.25)
    n = rng.standard_normal(len(t))
    n = n - lowpass(n, 900)  # high-passed noise
    e = np.exp(-t * 18)
    # little flam
    for d in (0.0, 0.011, 0.022):
        e += (t >= d) * np.exp(-np.maximum(t - d, 0) * 120) * 0.6
    return n * e * 0.35


def hat(open_=False):
    t = t_arr(0.18 if open_ else 0.05)
    n = rng.standard_normal(len(t))
    n = n - lowpass(n, 6000)
    return n * np.exp(-t * (14 if open_ else 70)) * 0.18


def pluck(freq, d=0.45):
    # marimba-ish: fundamental + 4th harmonic, fast decay
    t = t_arr(d)
    s = np.sin(2 * np.pi * freq * t) + 0.25 * np.sin(2 * np.pi * freq * 4 * t) * np.exp(-t * 25)
    return s * env_ad(len(t), 0.003, 0.16) * 0.22


def bass(freq, d=0.5):
    t = t_arr(d)
    s = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * freq * 2 * t)
    e = np.minimum(t / 0.01, 1) * np.exp(-t * 3.5)
    return np.tanh(s * 1.4) * e * 0.33


def pad(freqs, d):
    t = t_arr(d)
    s = sum(np.sin(2 * np.pi * f * t + i) + 0.5 * np.sin(2 * np.pi * f * 1.003 * t) for i, f in enumerate(freqs))
    e = np.minimum(t / 0.4, 1) * np.minimum((d - t) / 0.4, 1)
    return s * e * 0.025


# ---------------------------------------------------------------- music
def make_bgm():
    buf = np.zeros((int(DUR * SR), 2))
    bpm = 100
    beat = 60 / bpm
    bar = beat * 4
    # C  G  Am F   (I V vi IV) - bright, friendly
    prog = [(48, [60, 64, 67]), (43, [59, 62, 67]), (45, [60, 64, 69]), (41, [60, 65, 69])]
    melody = [72, None, 76, 74, 72, None, 67, None, 71, None, 74, 72, 71, None, 67, None,
              72, None, 76, 79, 76, None, 74, 72, 72, None, 69, None, 67, None, None, None]
    nbars = int(DUR / bar) + 1
    drop = (18.75, 19.75)  # music drop for "Paham?"
    for b in range(nbars):
        t0 = b * bar
        root, chord = prog[b % 4]
        add(buf, pad([midi(n) for n in chord], bar + 0.2), t0, 1.0)
        for s in range(8):  # 8th notes with light swing
            ts = t0 + s * beat / 2 + (0.035 if s % 2 else 0)
            if drop[0] <= ts < drop[1]:
                continue
            intro = ts < 0.6
            if s in (0, 3, 4) and not intro:
                add(buf, kick(), ts, 0.9)
            if s in (2, 6) and not intro:
                add(buf, clap(), ts, 0.8, 0.1)
            add(buf, hat(open_=(s == 7)), ts, 0.9 if s % 2 else 0.6, -0.3)
            if s in (0, 3, 6):
                add(buf, bass(midi(root)), ts, 1.0)
            if s in (1, 3, 5, 7):
                for k, n in enumerate(chord):
                    add(buf, pluck(midi(n)), ts + k * 0.008, 0.55, 0.35)
            m = melody[(b % 4) * 8 + s]
            if m is not None and b >= 1:
                add(buf, pluck(midi(m), 0.6), ts, 0.75, -0.2)
    # fade in/out
    n = len(buf)
    t = np.arange(n) / SR
    fade = np.minimum(t / 0.25, 1) * np.clip((DUR - t) / 1.2, 0, 1)
    buf *= fade[:, None]
    return buf


# ---------------------------------------------------------------- sfx
def whoosh(d=0.45, up=True):
    t = t_arr(d)
    n = rng.standard_normal(len(t))
    x = t / d
    cut = 300 + 5000 * (x if up else 1 - x) ** 2
    s = lowpass(n, cut)
    s = s - lowpass(s, 200)
    e = np.sin(np.pi * x) ** 2
    return s * e * 0.9


def pop(f0=900):
    t = t_arr(0.12)
    f = f0 * (1 + 1.2 * np.exp(-t * 60))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 35) * 0.6


def ding(f=1568, d=0.9):
    t = t_arr(d)
    s = sum(a * np.sin(2 * np.pi * f * m * t) * np.exp(-t * k)
            for a, m, k in [(1, 1, 5), (0.5, 2.0, 8), (0.25, 3.01, 12), (0.15, 4.2, 18)])
    return s * np.minimum(t / 0.002, 1) * 0.35


def sparkle():
    out = np.zeros(int(0.7 * SR))
    for i, f in enumerate([2093, 2637, 3136, 4186]):
        s = ding(f, 0.5) * 0.4
        j = int(i * 0.05 * SR)
        out[j:j + len(s)] += s[: len(out) - j]
    return out


def buzzer():
    t = t_arr(0.4)
    s = np.sign(np.sin(2 * np.pi * 140 * t)) * 0.5 + np.sign(np.sin(2 * np.pi * 147 * t)) * 0.5
    s = lowpass(s, 1800)
    return s * np.minimum(t / 0.01, 1) * np.clip((0.4 - t) / 0.05, 0, 1) * 0.28


def boom():
    t = t_arr(1.2)
    f = 38 + 70 * np.exp(-t * 12)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 3.2)
    n = lowpass(rng.standard_normal(len(t)), 400) * np.exp(-t * 6) * 0.8
    return np.tanh((s + n) * 1.5) * 0.7


def click():
    t = t_arr(0.04)
    return (np.sin(2 * np.pi * 2500 * t) + rng.standard_normal(len(t)) * 0.4) * np.exp(-t * 200) * 0.5


def riser(d):
    t = t_arr(d)
    f = 200 + 900 * (t / d) ** 2
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.3
    n = lowpass(rng.standard_normal(len(t)), 500 + 6000 * (t / d) ** 2) * 0.5
    return (s + n) * (t / d) ** 1.5 * 0.5


def levelup():
    out = np.zeros(int(1.0 * SR))
    for i, n in enumerate([72, 76, 79, 84, 88]):
        s = ding(midi(n), 0.5) * 0.5
        j = int(i * 0.09 * SR)
        out[j:j + len(s)] += s[: len(out) - j]
    return out


def chime():
    out = np.zeros(int(1.0 * SR))
    for i, n in enumerate([84, 91]):
        s = ding(midi(n), 0.8) * 0.6
        j = int(i * 0.12 * SR)
        out[j:j + len(s)] += s[: len(out) - j]
    return out


def boing():
    t = t_arr(0.35)
    f = 300 + 250 * np.sin(2 * np.pi * 14 * t) * np.exp(-t * 6) + 200 * t
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 7) * 0.4


def make_sfx():
    buf = np.zeros((int(DUR * SR), 2))
    ev = [
        # --- clip 1: FYP tips
        (0.00, boom(), 0.55, 0), (0.05, pop(700), 0.8, 0),
        (1.50, pop(1100), 0.8, 0.2), (1.55, sparkle(), 0.5, 0.3),
        (2.15, whoosh(0.35), 0.6, -0.3), (2.40, pop(800), 0.8, 0),
        (3.22, ding(1319), 0.55, -0.4), (3.18, whoosh(0.3), 0.4, -0.4),
        (4.40, ding(1568), 0.55, 0), (4.36, whoosh(0.3), 0.4, 0),
        (6.22, ding(1760), 0.55, 0.4), (6.18, whoosh(0.3), 0.4, 0.4),
        (8.66, buzzer(), 0.75, 0),
        # --- transition 1
        (9.62, whoosh(0.45), 0.9, 0), (10.0, pop(600), 0.5, 0),
        # --- clip 2: example
        (10.05, pop(1000), 0.7, 0),
        (12.26, boing(), 0.8, 0), (12.28, pop(1300), 0.6, 0),
        (13.22, buzzer(), 0.5, 0),
        (14.05, whoosh(0.35), 0.6, 0.3), (14.30, ding(1568), 0.5, 0.3),
        (15.88, pop(900), 0.7, 0),
        (16.80, riser(1.4), 0.6, 0), (18.20, ding(2093, 1.2), 0.6, 0), (18.22, sparkle(), 0.4, 0),
        (19.00, boom(), 0.85, 0),
        # --- transition 2
        (19.32, whoosh(0.45), 0.9, 0), (19.7, pop(600), 0.5, 0),
        # --- clip 3: CTA
        (21.86, click(), 0.9, 0), (21.92, pop(1200), 0.8, 0), (21.95, sparkle(), 0.5, 0),
        (23.38, whoosh(0.35), 0.6, -0.3), (23.60, pop(1000), 0.7, 0),
        (25.74, levelup(), 0.6, 0),
        (27.40, whoosh(0.3), 0.5, 0), (27.62, click(), 0.9, 0), (27.70, chime(), 0.55, 0),
    ]
    for at, s, g, p in ev:
        add(buf, s, at, g, p)
    return buf


def write(path, buf):
    peak = np.max(np.abs(buf))
    if peak > 0.98:
        buf = buf * 0.98 / peak
    data = (buf * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data.tobytes())


if __name__ == "__main__":
    write(f"{OUT}/sfx.wav", make_sfx())
    write(f"{OUT}/bgm.wav", make_bgm())
    print("ok")
