"""Render motion graphics + animated captions over the concatenated base video.

usage: python3 render.py base.mov transcript.json out_video.mp4
Reads raw frames from ffmpeg, composites overlays with Pillow, pipes to ffmpeg.
"""
import sys, json, math, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

BASE, TRANS, OUT = sys.argv[1:4]
W, H, FPS = 1080, 1920, 24
OFFS = {"v1": 0.0, "v2": 10.0, "v3": 19.7}
CUTS = [10.0, 19.7]
TOTAL = 28.9167

F_BLACK = "/usr/share/fonts/opentype/inter/InterDisplay-Black.otf"
F_XB = "/usr/share/fonts/opentype/inter/InterDisplay-ExtraBold.otf"
F_EMOJI = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"

YEL = (255, 214, 10)
RED = (255, 59, 48)
GRN = (52, 199, 89)
PINK = (255, 45, 120)
CYAN = (0, 214, 255)
WHITE = (255, 255, 255)
DARK = (18, 18, 22)

_fonts = {}


def font(path, size):
    k = (path, size)
    if k not in _fonts:
        _fonts[k] = ImageFont.truetype(path, size)
    return _fonts[k]


# ------------------------------------------------------------------ easing
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def out_cubic(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def out_back(x, s=2.2):
    x = clamp(x)
    return 1 + (s + 1) * (x - 1) ** 3 + s * (x - 1) ** 2


def in_cubic(x):
    x = clamp(x)
    return x ** 3


# ------------------------------------------------------------------ sprite builders
def emoji(ch, size):
    f = font(F_EMOJI, 109)
    im = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((80, 80), ch, font=f, embedded_color=True, anchor="mm")
    im = im.crop(im.getbbox())
    r = size / max(im.size)
    return im.resize((max(1, int(im.width * r)), max(1, int(im.height * r))), Image.LANCZOS)


def shadowed(im, off=(0, 10), blur=14, alpha=150):
    pad = blur * 3
    out = Image.new("RGBA", (im.width + pad * 2, im.height + pad * 2), (0, 0, 0, 0))
    a = im.split()[3].point(lambda v: v * alpha // 255)
    sh = Image.new("RGBA", im.size, (0, 0, 0, 255))
    sh.putalpha(a)
    out.paste(sh, (pad + off[0], pad + off[1]), sh)
    out = out.filter(ImageFilter.GaussianBlur(blur))
    out.alpha_composite(im, (pad, pad))
    return out


def text_img(txt, size, fill=WHITE, stroke=8, stroke_fill=(0, 0, 0), fpath=F_BLACK):
    f = font(fpath, size)
    d = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    l, t, r, b = d.textbbox((0, 0), txt, font=f, stroke_width=stroke)
    im = Image.new("RGBA", (r - l + 4, b - t + 4), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((-l + 2, -t + 2), txt, font=f, fill=fill, stroke_width=stroke, stroke_fill=stroke_fill)
    return im


def pill(txt, size, bg, fg=WHITE, padx=34, pady=18, icon=None, radius=None, outline=None):
    f = font(F_BLACK, size)
    d = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    l, t, r, b = d.textbbox((0, 0), txt, font=f)
    tw, th = r - l, b - t
    ic = emoji(icon, int(size * 1.15)) if icon else None
    iw = ic.width + 16 if ic else 0
    w, h = tw + iw + padx * 2, th + pady * 2
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    dr = ImageDraw.Draw(im)
    dr.rounded_rectangle((0, 0, w - 1, h - 1), radius=radius or h // 2, fill=bg,
                         outline=outline, width=6 if outline else 0)
    x = padx
    if ic:
        im.alpha_composite(ic, (x, (h - ic.height) // 2))
        x += iw
    dr.text((x - l, pady - t), txt, font=f, fill=fg)
    return shadowed(im)


def stack(*ims, gap=10, align="center"):
    w = max(i.width for i in ims)
    h = sum(i.height for i in ims) + gap * (len(ims) - 1)
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    y = 0
    for i in ims:
        x = (w - i.width) // 2 if align == "center" else 0
        out.alpha_composite(i, (x, y))
        y += i.height + gap
    return out


def hstack(*ims, gap=14):
    w = sum(i.width for i in ims) + gap * (len(ims) - 1)
    h = max(i.height for i in ims)
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    x = 0
    for i in ims:
        out.alpha_composite(i, (x, (h - i.height) // 2))
        x += i.width + gap
    return out


# ------------------------------------------------------------------ compositing helpers
def place(frame, im, cx, cy, scale=1.0, alpha=1.0, rot=0.0):
    if scale <= 0.01 or alpha <= 0.01:
        return
    if abs(scale - 1) > 1e-3:
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.BILINEAR)
    if rot:
        im = im.rotate(rot, resample=Image.BICUBIC, expand=True)
    if alpha < 1:
        a = im.split()[3].point(lambda v: int(v * alpha))
        im = im.copy()
        im.putalpha(a)
    x, y = int(cx - im.width / 2), int(cy - im.height / 2)
    frame.alpha_composite(im, (max(x, 0), max(y, 0)),
                          (max(-x, 0), max(-y, 0)))


class El:
    """Sprite with an entrance/exit animation between t0 and t1."""

    def __init__(self, im, t0, t1, cx, cy, anim="pop", dur_in=0.32, dur_out=0.22, wobble=0.0, rot=0.0):
        self.__dict__.update(locals())
        del self.__dict__["self"]

    def draw(self, fr, t):
        if t < self.t0 or t > self.t1:
            return
        a = (t - self.t0) / self.dur_in
        o = (self.t1 - t) / self.dur_out
        cx, cy, s, al = self.cx, self.cy, 1.0, 1.0
        if self.anim == "pop":
            s = out_back(a)
            al = clamp(a * 3)
        elif self.anim == "left":
            cx = self.cx - (1 - out_cubic(a)) * (W * 0.9)
        elif self.anim == "right":
            cx = self.cx + (1 - out_cubic(a)) * (W * 0.9)
        elif self.anim == "drop":
            cy = self.cy - (1 - out_back(a, 1.6)) * 300
            al = clamp(a * 3)
        elif self.anim == "stamp":
            s = 1 + 1.6 * (1 - out_cubic(a))
            al = clamp(a * 2.5)
        if o < 1:
            s *= 0.6 + 0.4 * out_cubic(o)
            al *= clamp(o)
        if self.wobble:
            s *= 1 + self.wobble * math.sin((t - self.t0) * 7)
        place(fr, self.im, cx, cy, s, al, self.rot)


# ------------------------------------------------------------------ captions
STRESS = {"fyp", "3", "hook", "retention", "watch", "time", "bosan", "ceramah", "panjang", "soalan",
          "mengejutkan", "meleret", "isi", "menarik", "habis", "paham", "like", "ai", "basic", "pro",
          "link", "bio", "tips", "atuk"}


def load_words():
    d = json.load(open(TRANS))
    words = []
    for k, segs in d.items():
        for s in segs:
            for (a, b, w) in s["words"]:
                words.append([a + OFFS[k], b + OFFS[k], w.strip(), k])
    # group into short chunks (max 3 words, break on punctuation / gaps / clip changes)
    chunks, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        brk = (len(cur) >= 3 or w[2][-1] in ",.?!" or nxt is None or nxt[3] != w[3]
               or nxt[0] - w[1] > 0.35)
        if brk:
            chunks.append(cur)
            cur = []
    out = []
    for i, c in enumerate(chunks):
        start = c[0][0]
        end = c[-1][1] + 0.25
        if i + 1 < len(chunks):
            end = min(end, chunks[i + 1][0][0])
        if c[-1][3] != (chunks[i + 1][0][3] if i + 1 < len(chunks) else None):
            end = min(c[-1][1] + 0.3, {"v1": 10.0, "v2": 19.7, "v3": TOTAL}[c[-1][3]])
        out.append((start, end, c))
    return out


CHUNKS = load_words()
CAP_Y = 1470


def draw_caption(fr, t):
    for start, end, c in CHUNKS:
        if not (start <= t < end):
            continue
        a = (t - start) / 0.18
        base_s = 0.85 + 0.15 * out_back(a)
        imgs = []
        for (ws, we, w, _) in c:
            txt = w.upper().rstrip(",.")
            key = w.lower().strip(",.?!")
            active = ws <= t < max(we, ws + 0.2)
            spoken = t >= ws
            col = YEL if (active or key in STRESS and spoken) else WHITE
            if key in STRESS and spoken and not active:
                col = YEL
            im = text_img(txt, 86, fill=col, stroke=10)
            if active:
                p = (t - ws) / 0.15
                im = im.resize((int(im.width * (1 + 0.12 * (1 - out_cubic(p)) + 0.06)),
                                int(im.height * (1 + 0.12 * (1 - out_cubic(p)) + 0.06))), Image.BILINEAR)
            elif not spoken:
                im = im.copy()
                im.putalpha(im.split()[3].point(lambda v: int(v * 0.55)))
            imgs.append(im)
        # wrap to 2 lines if too wide
        line = hstack(*imgs, gap=22)
        if line.width > W - 80 and len(imgs) > 1:
            line = stack(hstack(*imgs[:-1], gap=22), imgs[-1], gap=0)
        line = shadowed(line, off=(0, 6), blur=10, alpha=170)
        place(fr, line, W / 2, CAP_Y, base_s, clamp(a * 2.5))
        break


# ------------------------------------------------------------------ graphics timeline
def build():
    E = []
    # ---- clip 1 -------------------------------------------------------------
    E.append(El(pill("NAK MASUK FYP?", 66, PINK, icon="🚀"), 0.05, 2.2, W / 2, 190, "drop"))
    E.append(El(emoji("🔥", 120), 1.52, 2.2, 880, 640, "pop", wobble=0.06, rot=-12))
    E.append(El(emoji("✨", 90), 1.6, 2.2, 200, 560, "pop", wobble=0.08))
    big3 = stack(text_img("JAGA", 70, fill=WHITE, stroke=9),
                 text_img("3 BENDA", 120, fill=YEL, stroke=12), gap=0)
    E.append(El(big3, 2.24, 3.15, W / 2, 210, "pop"))
    # 3-item checklist (custom drawn, see draw_checklist)
    # "ceramah panjang" = boring -> red X stamp + Zzz
    E.append(El(pill("CERAMAH PANJANG", 58, DARK, fg=WHITE, icon="😴", outline=RED), 7.54, 9.85, W / 2, 1180, "pop"))
    stamp = text_img("✗ BOSAN!", 110, fill=RED, stroke=12, stroke_fill=WHITE)
    E.append(El(stamp, 8.66, 9.85, W / 2, 1040, "stamp", rot=8))
    # ---- clip 2 -------------------------------------------------------------
    E.append(El(pill("CONTOH", 56, CYAN, fg=DARK, icon="💡"), 10.05, 13.1, W / 2, 170, "left"))
    E.append(El(text_img("SOALAN", 92, fill=WHITE, stroke=11), 11.94, 13.1, W / 2, 300, "pop"))
    E.append(El(emoji("😱", 170), 12.26, 13.1, 850, 700, "pop", wobble=0.05, rot=-10))
    E.append(El(text_img("?!", 200, fill=YEL, stroke=14), 12.26, 13.1, 210, 700, "pop", rot=12, wobble=0.07))
    mel = pill("MELERET", 64, (70, 70, 76), fg=(200, 200, 200), icon="🐌")
    E.append(El(mel, 13.22, 15.7, W / 2, 190, "pop"))
    E.append(El(pill("TERUS KE ISI", 64, GRN, icon="🎯"), 14.12, 15.7, W / 2, 330, "right"))
    E.append(El(pill("AKHIR SEKALI", 52, PINK, icon="🏁"), 15.88, 18.95, W / 2, 170, "left"))
    E.append(El(text_img("PAHAM?", 150, fill=YEL, stroke=14), 19.0, 19.68, W / 2, 1180, "stamp", rot=-6))
    E.append(El(emoji("👍", 150), 19.05, 19.68, 880, 900, "pop", rot=10))
    # ---- clip 3 -------------------------------------------------------------
    E.append(El(pill("SUKA TIPS ATUK?", 60, PINK, icon="👴"), 19.75, 23.2, W / 2, 180, "drop"))
    E.append(El(pill("BUAT VIDEO AI", 64, (124, 58, 237), icon="🤖"), 23.38, 25.6, W / 2, 190, "drop"))
    E.append(El(text_img("MACAM ATUK!", 80, fill=YEL, stroke=10), 24.9, 25.6, W / 2, 320, "pop"))
    E.append(El(pill("DARI BASIC KE PRO", 56, DARK, icon="📈", outline=YEL), 25.74, 27.6, W / 2, 190, "drop"))
    return E


def draw_checklist(fr, t):
    """3-item checklist: appears at 'Jaga 3 benda', items light up when mentioned."""
    t0, t1 = 2.9, 9.85
    if not (t0 <= t <= t1):
        return
    items = [("HOOK", "🪝", 3.22, RED), ("RETENTION", "🔁", 4.40, CYAN), ("WATCH TIME", "⏱️", 6.22, GRN)]
    out = clamp((t1 - t) / 0.25)
    title = text_img("3 BENDA PENTING", 58, fill=YEL, stroke=8)
    place(fr, shadowed(title), W / 2, 95, out_back((t - t0) / 0.3) * (0.6 + 0.4 * out), out)
    xs = [190, 540, 890]
    for i, (name, ic, ti, col) in enumerate(items):
        app = (t - (t0 + 0.08 * i)) / 0.3
        on = t >= ti
        im = _chk_cache.get((name, on))
        if im is None:
            num = f"{i + 1}"
            bg = col if on else (40, 40, 46)
            fg = WHITE if on else (150, 150, 155)
            f = font(F_BLACK, 40 if len(name) > 6 else 46)
            card = Image.new("RGBA", (320, 190), (0, 0, 0, 0))
            d = ImageDraw.Draw(card)
            d.rounded_rectangle((0, 0, 319, 189), radius=34, fill=bg + (235,),
                                outline=WHITE if on else (90, 90, 96), width=5)
            e = emoji(ic, 70)
            if not on:
                g = e.convert("LA").convert("RGBA")
                g.putalpha(e.split()[3].point(lambda v: v // 2))
                e = g
            card.alpha_composite(e, (160 - e.width // 2, 22))
            d.text((160, 140), name, font=f, fill=fg, anchor="mm")
            badge = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
            bd = ImageDraw.Draw(badge)
            bd.ellipse((0, 0, 63, 63), fill=YEL if on else (90, 90, 96), outline=DARK, width=4)
            bd.text((32, 33), "✓" if on else num, font=font(F_BLACK, 36) if not on else
                    ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 38),
                    fill=DARK, anchor="mm")
            full = Image.new("RGBA", (350, 215), (0, 0, 0, 0))
            full.alpha_composite(card, (0, 25))
            full.alpha_composite(badge, (276, 0))
            im = shadowed(full)
            _chk_cache[(name, on)] = im
        s = out_back(app) * (0.6 + 0.4 * out)
        if on:
            p = (t - ti) / 0.35
            s *= 1 + 0.18 * math.sin(clamp(p) * math.pi)
        place(fr, im, xs[i], 290, s * 0.95, clamp(app * 3) * out)


_chk_cache = {}


def draw_progress(fr, t):
    """'tengok sampai habis' -> watch progress bar filling to 100%."""
    t0, t1 = 16.88, 18.95
    if not (t0 <= t <= t1):
        return
    a = out_back((t - t0) / 0.3)
    out = clamp((t1 - t) / 0.2)
    p = clamp((t - 17.0) / 1.2) ** 1.4
    w, h = 820, 150
    im = Image.new("RGBA", (w, h + 90), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 70, w - 1, 70 + h - 1), radius=40, fill=(20, 20, 24, 230), outline=WHITE, width=5)
    fw = int((w - 24) * p)
    if fw > 40:
        d.rounded_rectangle((12, 82, 12 + fw, 70 + h - 13), radius=30, fill=GRN if p >= 1 else YEL)
    pct = int(round(p * 100))
    d.text((w / 2, 70 + h / 2), f"{pct}%", font=font(F_BLACK, 70), fill=WHITE, anchor="mm",
           stroke_width=6, stroke_fill=DARK)
    d.text((w / 2, 30), "TENGOK SAMPAI HABIS", font=font(F_BLACK, 52), fill=WHITE, anchor="mm",
           stroke_width=7, stroke_fill=DARK)
    s = a * (0.7 + 0.3 * out)
    if p >= 1:
        s *= 1 + 0.08 * math.sin(clamp((t - 18.2) / 0.3) * math.pi)
    place(fr, shadowed(im), W / 2, 330, s, out)
    if t >= 18.2:
        place(fr, emoji("🏆", 120), 930, 250, out_back((t - 18.2) / 0.3), out, rot=-10)


def draw_like(fr, t):
    """'tekan like dulu' -> big like button pressed + heart burst."""
    t0, t1 = 21.6, 23.25
    if not (t0 <= t <= t1):
        return
    tp = 21.88
    out = clamp((t1 - t) / 0.2)
    a = out_back((t - t0) / 0.3)
    pressed = t >= tp
    s = a * (0.7 + 0.3 * out)
    if pressed:
        s *= 1 - 0.15 * math.sin(clamp((t - tp) / 0.2) * math.pi)
    btn = Image.new("RGBA", (260, 260), (0, 0, 0, 0))
    d = ImageDraw.Draw(btn)
    d.ellipse((0, 0, 259, 259), fill=PINK if pressed else (35, 35, 40), outline=WHITE, width=8)
    h = emoji("❤️" if pressed else "🤍", 140)
    btn.alpha_composite(h, (130 - h.width // 2, 130 - h.height // 2 + 6))
    cx, cy = W / 2, 1160
    place(fr, shadowed(btn), cx, cy, s, out)
    # cursor
    ca = clamp((t - t0) / 0.28)
    hand = emoji("👆", 110)
    hx = cx + 60 + (1 - out_cubic(ca)) * 300
    hy = cy + 120 + (1 - out_cubic(ca)) * 250 + (12 if abs(t - tp) < 0.1 else 0)
    place(fr, hand, hx, hy, 1.0, out * clamp(ca * 3))
    # burst of hearts
    if pressed:
        k = (t - tp) / 0.9
        if k < 1:
            for i in range(10):
                ang = i / 10 * 2 * math.pi + 0.3
                r = 150 + 260 * out_cubic(k)
                place(fr, _heart_small, cx + math.cos(ang) * r, cy + math.sin(ang) * r - 120 * k,
                      0.6 + 0.4 * (1 - k), 1 - k ** 2)


def draw_level(fr, t):
    """'dari basic sampai pro' -> level meter BASIC -> PRO."""
    t0, t1 = 25.9, 27.6
    if not (t0 <= t <= t1):
        return
    out = clamp((t1 - t) / 0.2)
    a = out_back((t - t0) / 0.3)
    p = clamp((t - 26.1) / 0.9)
    p = out_cubic(p)
    w = 860
    im = Image.new("RGBA", (w, 210), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    segs = 5
    sw = (w - 40 - (segs - 1) * 14) / segs
    cols = [(255, 99, 71), (255, 159, 10), (255, 214, 10), (130, 220, 60), (52, 199, 89)]
    for i in range(segs):
        x = 20 + i * (sw + 14)
        lit = p * segs > i + 0.15
        d.rounded_rectangle((x, 80, x + sw, 150), radius=18, fill=cols[i] if lit else (45, 45, 50),
                            outline=WHITE, width=4)
    d.text((20, 40), "BASIC", font=font(F_BLACK, 50), fill=WHITE, anchor="lm", stroke_width=6, stroke_fill=DARK)
    d.text((w - 20, 40), "PRO", font=font(F_BLACK, 62), fill=YEL if p > 0.95 else WHITE, anchor="rm",
           stroke_width=6, stroke_fill=DARK)
    place(fr, shadowed(im), W / 2, 1180, a * (0.7 + 0.3 * out), out)
    if p > 0.95:
        place(fr, emoji("👑", 100), W / 2 + 330, 1060, out_back((t - 27.0) / 0.3), out, rot=12)


def draw_bio(fr, t):
    """'klik link dekat bio' -> pulsing CTA button + pointing hand."""
    t0, t1 = 27.62, TOTAL + 1
    if t < t0:
        return
    a = out_back((t - t0) / 0.3)
    pulse = 1 + 0.05 * math.sin((t - t0) * 9)
    btn = _bio_btn
    place(fr, btn, W / 2, 1170, a * pulse, clamp((t - t0) * 4))
    hand = emoji("👇", 120)
    bob = math.sin((t - t0) * 10) * 18
    place(fr, hand, W / 2 + 330, 1040 + bob, a, clamp((t - t0) * 4), rot=15)


def draw_global(fr, t):
    # watch-progress bar along the very top (retention trick)
    p = t / TOTAL
    d = ImageDraw.Draw(fr)
    d.rectangle((0, 0, W, 12), fill=(0, 0, 0, 90))
    d.rectangle((0, 0, int(W * p), 12), fill=YEL + (255,))
    # white flash on cuts
    for c in CUTS:
        dt = t - c
        if 0 <= dt < 0.17:
            al = int(200 * (1 - dt / 0.17))
            fr.alpha_composite(Image.new("RGBA", (W, H), (255, 255, 255, al)))


_heart_small = emoji("💖", 70)
_bio_btn = pill("LINK DI BIO", 78, PINK, icon="🔗", padx=50, pady=28)

ELEMENTS = build()


# ------------------------------------------------------------------ camera (zoom punch-ins)
PUNCH = [  # (time, strength, attack, hold)
    (1.52, 0.10, 0.12, 0.6), (3.22, 0.05, 0.1, 0.4), (8.66, 0.10, 0.08, 0.9),
    (12.26, 0.12, 0.1, 0.7), (19.0, 0.22, 0.08, 0.7), (21.88, 0.07, 0.1, 0.6),
    (27.78, 0.08, 0.12, 1.2),
]


def zoom_at(t):
    z = 1.0 + 0.03 * ((t % 10) / 10)  # slow push
    for (pt, s, att, hold) in PUNCH:
        if pt <= t < pt + att + hold + 0.35:
            if t < pt + att:
                k = out_cubic((t - pt) / att)
            elif t < pt + att + hold:
                k = 1
            else:
                k = 1 - out_cubic((t - pt - att - hold) / 0.35)
            z += s * k
    for c in CUTS:  # incoming clip zooms out from 1.25
        if c <= t < c + 0.4:
            z += 0.25 * (1 - out_cubic((t - c) / 0.4))
    shake = 0
    for c in CUTS:
        if c - 0.12 <= t < c:
            shake = 1
    return z, shake


def camera(img, t):
    z, shake = zoom_at(t)
    if z <= 1.0005 and not shake:
        return img
    cx, cy = W / 2, 760  # face area
    nw, nh = W / z, H / z
    x0 = clamp(cx - nw / 2, 0, W - nw)
    y0 = clamp(cy - nh / 2, 0, H - nh)
    if shake:  # whip blur before the cut
        img = img.filter(ImageFilter.BoxBlur(10))
    return img.resize((W, H), Image.BILINEAR, box=(x0, y0, x0 + nw, y0 + nh))


# ------------------------------------------------------------------ main loop
def main():
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", BASE, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                           stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                            "-vf", "eq=contrast=1.06:saturation=1.12:brightness=0.01",
                            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                            "-pix_fmt", "yuv420p", OUT], stdin=subprocess.PIPE)
    n = 0
    fsz = W * H * 3
    while True:
        raw = dec.stdout.read(fsz)
        if len(raw) < fsz:
            break
        t = n / FPS
        img = Image.frombuffer("RGB", (W, H), raw).copy()
        img = camera(img, t)
        fr = img.convert("RGBA")
        for e in ELEMENTS:
            e.draw(fr, t)
        draw_checklist(fr, t)
        draw_progress(fr, t)
        draw_like(fr, t)
        draw_level(fr, t)
        draw_bio(fr, t)
        draw_caption(fr, t)
        draw_global(fr, t)
        enc.stdin.write(fr.convert("RGB").tobytes())
        n += 1
        if n % 100 == 0:
            print("frame", n, flush=True)
    enc.stdin.close()
    enc.wait()
    print("done", n)


if __name__ == "__main__":
    if len(sys.argv) > 4:  # preview stills: render.py base trans out t1,t2,...
        ts = [float(x) for x in sys.argv[4].split(",")]
        for t in ts:
            raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", BASE, "-frames:v", "1",
                                  "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
            img = camera(Image.frombuffer("RGB", (W, H), raw).copy(), t)
            fr = img.convert("RGBA")
            for e in ELEMENTS:
                e.draw(fr, t)
            for f in (draw_checklist, draw_progress, draw_like, draw_level, draw_bio, draw_caption, draw_global):
                f(fr, t)
            fr.convert("RGB").resize((360, 640)).save(f"{OUT}_{t:05.2f}.png")
    else:
        main()
