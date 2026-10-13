"""
Colada Character Maker - Texture painter
========================================
Bakes one atlas texture from the spec: skin, face (eyes / brows / mouth as
separate curved-patch cells), hair gradient, clothing patterns and extras.

Every cell is rendered at `antialias` x and downsampled, so edges stay clean
at any of the three texture sizes.
"""
from __future__ import annotations

import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import atlas
from .mesh import MOUTH_PHI_RATIO, head_uv

TAU = math.pi * 2.0


# ---------------------------------------------------------------------------
# colour helpers
# ---------------------------------------------------------------------------

def hex2rgb(h: str) -> tuple[int, int, int]:
    h = h.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        h = "808080"
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore


def mix(a, b, t: float):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def shade(c, f: float):
    if f >= 1.0:
        return tuple(int(min(255, round(v * f))) for v in c)
    return tuple(int(max(0, round(v * f))) for v in c)


def lighten(c, t: float):
    return mix(c, (255, 255, 255), t)


def darken(c, t: float):
    return mix(c, (0, 0, 0), t)


def saturate(c, t: float):
    g = sum(c) / 3.0
    return tuple(int(max(0, min(255, g + (v - g) * (1.0 + t)))) for v in c)


def warm(c, t: float = 0.25):
    return (min(255, int(c[0] * (1 + t))), int(c[1] * (1 + t * 0.35)), int(c[2] * (1 - t * 0.3)))


def hue_shift(c, dh: int):
    """Cheap hue rotation used for pattern variation."""
    r, g, b = c
    if dh == 0:
        return c
    piv = dh % 3
    return (g, b, r) if piv == 1 else (b, r, g)


# ---------------------------------------------------------------------------
# patterns (all tile horizontally so the lathe seam never shows)
# ---------------------------------------------------------------------------

def draw_pattern(d: ImageDraw.ImageDraw, box, pattern: str, c1, c2, W: int, H: int, rng: random.Random):
    x0, y0, x1, y1 = box
    if pattern == "solid":
        d.rectangle(box, fill=c1)
        return

    if pattern == "gradient":
        for y in range(H):
            t = y / max(1, H - 1)
            d.line([(x0, y0 + y), (x1, y0 + y)], fill=mix(c1, c2, t))
        return

    d.rectangle(box, fill=c1)
    if pattern == "stripes":
        n = 7
        band = W / (n * 2)
        for i in range(n * 2):
            if i % 2 == 0:
                d.rectangle([x0 + int(i * band), y0, x0 + int((i + 1) * band), y1], fill=c2)
    elif pattern == "dots":
        step = W // 6
        for iy in range(0, H, step):
            for ix in range(0, W + step, step):
                off = (iy // step % 2) * step // 2
                cx = x0 + (ix + off) % (W + step)
                d.ellipse([cx - step * 0.19, y0 + iy + step * 0.30, cx + step * 0.19, y0 + iy + step * 0.70],
                          fill=c2)
    elif pattern == "plaid":
        n = 5
        band = W / n
        for i in range(n):
            d.rectangle([x0 + int(i * band), y0, x0 + int(i * band + band * 0.28), y1],
                        fill=shade(c2, 0.85))
            d.rectangle([x0, y0 + int(i * band), x1, y0 + int(i * band + band * 0.28)],
                        fill=shade(c2, 0.85))
        for i in range(n):
            d.rectangle([x0 + int(i * band + band * 0.55), y0, x0 + int(i * band + band * 0.72), y1],
                        fill=shade(c1, 0.86))
            d.rectangle([x0, y0 + int(i * band + band * 0.55), x1, y0 + int(i * band + band * 0.72)],
                        fill=shade(c1, 0.86))
    elif pattern == "stars":
        step = W // 4
        for iy in range(0, H + step, step):
            for ix in range(0, W + step, step):
                off = (iy // step % 2) * step // 2
                cx = x0 + (ix + off) % (W + step)
                cy = y0 + iy + step // 2
                star(d, cx, cy, step * 0.20, c2)
    elif pattern == "camo":
        for i in range(90):
            cx = x0 + rng.randint(0, W)
            cy = y0 + rng.randint(0, H)
            rx = rng.randint(W // 22, W // 9)
            ry = rng.randint(W // 26, W // 12)
            col = mix(c2, c1, rng.choice([0.0, 0.25, 0.5, 0.75]))
            d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=col)
    elif pattern == "tropical":
        for y in range(H):
            t = y / max(1, H - 1)
            d.line([(x0, y0 + y), (x1, y0 + y)], fill=mix(c1, c2, 0.15 + 0.35 * t))
        for i in range(14):
            cx = x0 + rng.randint(0, W)
            cy = y0 + rng.randint(0, H)
            r = rng.randint(W // 26, W // 14)
            for k in range(5):
                a = TAU * k / 5 + rng.random()
                px, py = cx + math.cos(a) * r * 0.9, cy + math.sin(a) * r * 0.9
                d.ellipse([px - r * 0.45, py - r * 0.45, px + r * 0.45, py + r * 0.45],
                          fill=rng.choice([lighten(c2, 0.1), lighten(c1, 0.25), darken(c2, 0.15)]))
            d.ellipse([cx - r * 0.3, cy - r * 0.3, cx + r * 0.3, cy + r * 0.3], fill=darken(c1, 0.3))
    elif pattern == "pineapple":
        # big tropical logo centred on the chest (u = 0.5 of the wrap)
        cx = x0 + W // 2
        cy = y0 + int(H * 0.52)
        R = int(H * 0.30)
        body = mix(c2, (255, 224, 102), 0.55)
        d.ellipse([cx - R, cy - int(R * 1.18), cx + R, cy + int(R * 1.18)], fill=body,
                  outline=darken(body, 0.35), width=max(1, R // 14))
        # cross hatch
        for i in range(-3, 4):
            d.line([(cx - R, cy + i * R * 0.34), (cx + R, cy + i * R * 0.34)], fill=darken(body, 0.22),
                   width=max(1, R // 22))
            d.line([(cx + i * R * 0.34 + R * 0.4, cy - int(R * 1.1)), (cx + i * R * 0.34 + R * 0.4, cy + int(R * 1.1))],
                   fill=darken(body, 0.22), width=max(1, R // 22))
        # crown
        for i in range(5):
            a = -math.pi / 2 + (i - 2) * 0.36
            tipx, tipy = cx + math.cos(a) * R * 1.05, cy - int(R * 1.15) + math.sin(a) * R * 0.85
            d.polygon([(cx + (i - 2) * R * 0.30, cy - int(R * 1.05)), (cx + (i - 1) * R * 0.30, cy - int(R * 1.05)),
                       (tipx, tipy)], fill=(52, 160, 90))
        # little umbrella + sun for flavour
        ux, uy = cx - int(W * 0.36), cy + int(H * 0.18)
        ur = int(W * 0.10)
        d.pieslice([ux - ur, uy - ur, ux + ur, uy + ur], 180, 360, fill=rng.choice([(240, 80, 90), (255, 209, 102)]))
        d.line([(ux, uy), (ux, uy + ur)], fill=(90, 62, 40), width=max(1, ur // 6))


def star(d: ImageDraw.ImageDraw, cx, cy, r, col):
    pts = []
    for i in range(10):
        a = -math.pi / 2 + i * math.pi / 5
        rr = r if i % 2 == 0 else r * 0.45
        pts.append((cx + math.cos(a) * rr, cy + math.sin(a) * rr))
    d.polygon(pts, fill=col)


# ---------------------------------------------------------------------------
# face art
# ---------------------------------------------------------------------------

def _patch_px(px_box, hw, hh, size_scale):
    """pixels-per-metre for a patch cell: (sx, sy, W, H)."""
    x0, y0, x1, y1 = px_box
    W, H = x1 - x0, y1 - y0
    sx = W / max(1e-6, 2.0 * hw)
    sy = H / max(1e-6, 2.0 * hh)
    return sx, sy, W, H


def draw_eye(img: Image.Image, draw: ImageDraw.ImageDraw, box, spec, eye_r, skin, flip: bool):
    """The eye graphic fills its own cell; the patch geometry is a bit larger."""
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    sx = W / max(1e-6, 2.0 * eye_r)
    sy = H / max(1e-6, 2.0 * eye_r * 0.72)

    def P(wx, wy):
        px = (wx * sx) + W / 2.0
        py = (wy * sy) + H / 2.0
        if flip:
            px = W - px
        return (x0 + px, y0 + py)

    def E(wx, wy, rw, rh):
        cx, cy = P(wx, wy)
        return [cx - rw * sx, cy - rh * sy, cx + rw * sx, cy + rh * sy]

    style = spec["eye_style"]
    iris_col = hex2rgb(spec["eye_color"])
    lashes = bool(spec["lashes"])
    white = (250, 250, 252)
    outline = darken(skin, 0.55)

    if style == "star":
        # solid colourful eye with a star highlight
        d = ImageDraw.Draw(img, "RGBA")
        d.ellipse(E(0, 0, 0.86 * eye_r, 0.66 * eye_r), fill=iris_col, outline=outline,
                  width=max(1, int(0.06 * sx * eye_r)))
        star(d, *P(0, 0), 0.44 * eye_r * sx, (255, 255, 255))
        d.ellipse(E(0.22 * eye_r, -0.16 * eye_r, 0.13 * eye_r, 0.13 * eye_r), fill=(255, 255, 255, 235))
    else:
        d = ImageDraw.Draw(img, "RGBA")
        lid = {"round": 0.0, "wide": -0.10, "almond": 0.16, "sleepy": 0.55}.get(style, 0.0)
        half_h = 0.66 * eye_r * (1.0 - lid)
        d.ellipse(E(0, 0, 0.88 * eye_r, half_h), fill=white, outline=outline,
                  width=max(1, int(0.045 * sx * eye_r)))
        # iris
        iris_r = 0.52 * eye_r * (0.92 if style != "wide" else 1.12)
        d.ellipse(E(0, 0.02 * eye_r, iris_r, iris_r * 1.02), fill=iris_col)
        d.ellipse(E(0, 0.02 * eye_r, iris_r * 0.98, iris_r), outline=darken(iris_col, 0.45),
                  width=max(1, int(0.035 * sx * eye_r)))
        # pupil
        d.ellipse(E(0, 0.02 * eye_r, iris_r * 0.46, iris_r * 0.50), fill=darken(iris_col, 0.86))
        # highlights
        d.ellipse(E(-0.24 * eye_r, -0.24 * eye_r, 0.17 * eye_r, 0.17 * eye_r), fill=(255, 255, 255, 245))
        d.ellipse(E(0.22 * eye_r, 0.20 * eye_r, 0.09 * eye_r, 0.09 * eye_r), fill=(255, 255, 255, 200))
        # lower lid light
        d.arc(E(0, 0, 0.86 * eye_r, 0.62 * eye_r), 30, 150, fill=darken(skin, 0.18),
              width=max(1, int(0.03 * sy * eye_r)))
        if style == "sleepy":
            d.rectangle(E(0, -0.30 * eye_r, 1.05 * eye_r, 0.34 * eye_r), fill=skin)

    # eyeliner + lashes
    if lashes or style != "sleepy":
        ul = ImageDraw.Draw(img, "RGBA")
        w = max(1, int(0.07 * sx * eye_r))
        ul.arc(E(0, 0, 0.90 * eye_r, 0.70 * eye_r), 185, 355, fill=darken(outline, 0.25), width=w)
        if lashes:
            for i, t in enumerate((-0.85, -0.55, -0.2, 0.2, 0.55, 0.85)):
                a = math.pi * (1.0 + t * 0.5)
                rr = 0.9 * eye_r
                bx, by = P(math.cos(a) * rr * 0.95, -math.sin(a) * rr * 0.68)
                ln = 0.26 * eye_r * sy * (1.0 - 0.35 * abs(t))
                ang = a - math.pi / 2
                ul.line([bx, by, bx + math.cos(ang) * ln * 0.5 - 0.5 * ln, by + abs(math.sin(ang)) * ln - ln * 0.15],
                        fill=darken(outline, 0.35), width=max(1, int(w * 0.8)))


def draw_brow(img, box, spec, eye_r, brow_color, flip):
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    style = spec["brow_style"]
    if style == "none":
        return
    sx = W / max(1e-6, 2.0 * eye_r * 1.55)
    d = ImageDraw.Draw(img, "RGBA")
    col = brow_color
    thick = {"soft": 0.26, "thin": 0.15, "bold": 0.40, "angry": 0.34, "raised": 0.24}.get(style, 0.25)
    th = max(2, int(thick * eye_r * sx * 0.8))
    # brow shape: arc from outer (low) to inner (high) ; mirrored per side via flip
    inner_y = {"soft": -0.06, "thin": -0.02, "bold": -0.10, "angry": 0.16, "raised": -0.16}.get(style, 0.0)
    outer_y = {"soft": 0.10, "thin": 0.04, "bold": 0.14, "angry": -0.10, "raised": -0.02}.get(style, 0.05)
    curve = {"soft": 0.16, "thin": 0.12, "bold": 0.18, "angry": 0.06, "raised": 0.30}.get(style, 0.14)

    def pt(t):  # t: 0 inner -> 1 outer
        wx = (0.5 - t) * 1.42 * eye_r
        wy = (inner_y + (outer_y - inner_y) * t - curve * math.sin(math.pi * t)) * eye_r
        px = wx * sx + W / 2.0
        if flip:
            px = W - px
        return (x0 + px, y0 + wy * sx + H / 2.0)

    pts = [pt(t / 12.0) for t in range(13)]
    d.line(pts, fill=col, width=th, joint="curve")
    for p in (pts[0], pts[-1]):
        d.ellipse([p[0] - th / 2, p[1] - th / 2, p[0] + th / 2, p[1] + th / 2], fill=col)


def draw_mouth(img, box, spec, eye_r, skin, mouth_color):
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    sx = W / max(1e-6, 2.55 * eye_r)
    sy = H / max(1e-6, 1.70 * eye_r)

    def P(wx, wy):
        return (x0 + wx * sx + W / 2.0, y0 + wy * sy + H / 2.0)

    def E(wx, wy, rw, rh):
        cx, cy = P(wx, wy)
        return [cx - rw * sx, cy - rh * sy, cx + rw * sx, cy + rh * sy]

    style = spec["mouth_style"]
    d = ImageDraw.Draw(img, "RGBA")
    lip = mouth_color
    dark = darken(mix(lip, (60, 20, 30), 0.5), 0.25)
    line_w = max(2, int(0.10 * sy * eye_r))

    if style == "smile":
        d.arc(E(0, -0.55 * eye_r, 0.92 * eye_r, 1.05 * eye_r), 25, 155, fill=dark, width=line_w)
    elif style == "grin":
        box_e = E(0, -0.10 * eye_r, 0.95 * eye_r, 0.62 * eye_r)
        d.chord(box_e, 0, 360, fill=(250, 250, 252))          # teeth
        d.chord(box_e, 0, 360, outline=dark, width=line_w)
        d.line([(box_e[0], box_e[1] + (box_e[3] - box_e[1]) * 0.45),
                (box_e[2], box_e[1] + (box_e[3] - box_e[1]) * 0.45)], fill=lighten(dark, 0.35), width=1)
    elif style == "smirk":
        d.arc(E(0.25 * eye_r, -0.30 * eye_r, 0.80 * eye_r, 0.80 * eye_r), 20, 150, fill=dark, width=line_w)
        d.ellipse(E(-0.72 * eye_r, -0.10 * eye_r, 0.10 * eye_r, 0.10 * eye_r), fill=dark)
    elif style == "neutral":
        d.line([P(-0.85 * eye_r, 0.05 * eye_r), P(0.85 * eye_r, 0.05 * eye_r)], fill=dark, width=line_w)
    elif style == "open":
        d.chord(E(0, 0.05 * eye_r, 0.62 * eye_r, 0.62 * eye_r), 0, 360, fill=darken(lip, 0.55))
        d.chord(E(0, 0.30 * eye_r, 0.40 * eye_r, 0.26 * eye_r), 0, 360, fill=(220, 110, 120))
        d.chord(E(0, 0.05 * eye_r, 0.62 * eye_r, 0.62 * eye_r), 0, 360, outline=dark, width=max(1, line_w // 2))
    elif style == "frown":
        d.arc(E(0, 0.85 * eye_r, 0.90 * eye_r, 0.85 * eye_r), 205, 335, fill=dark, width=line_w)
    elif style == "pout":
        d.ellipse(E(0, 0.05 * eye_r, 0.42 * eye_r, 0.30 * eye_r), fill=darken(lip, 0.2), outline=dark,
                  width=max(1, line_w // 2))
    # lower lip highlight
    if style in ("smile", "grin", "pout"):
        d.arc(E(0, -0.35 * eye_r, 0.62 * eye_r, 0.62 * eye_r), 20, 160, fill=lighten(skin, 0.25),
              width=max(1, line_w // 2))


# ---------------------------------------------------------------------------
# head (equirect) painting
# ---------------------------------------------------------------------------

def skin_field(theta: np.ndarray, phi: np.ndarray, spec) -> np.ndarray:
    """Shading applied to bare skin - shared by the head cell and every patch."""
    base = np.array(hex2rgb(spec["skin_tone"]), dtype=np.float64)
    detail = float(spec["skin_detail"])
    front = np.clip(np.cos(theta), -1.0, 1.0) * 0.5 + 0.5    # 1 at the face, 0 behind
    depth = np.cos(phi)                                       # +1 crown, -1 chin
    light = (1.0
             + 0.075 * front * detail
             + 0.05 * depth * detail
             - 0.10 * np.clip(-depth - 0.35, 0, 1) * detail)
    warmf = np.clip(front * detail, 0, 1) * 0.05
    warm3 = np.stack([1.0 + warmf, 1.0 + warmf * 0.3, 1.0 - warmf * 0.25], axis=-1)
    field = base[None, None, :] * light[..., None] * warm3
    return np.clip(field, 0, 255)


def _head_skin_field(W: int, H: int, spec) -> np.ndarray:
    """Per pixel colour for the head cell (equirect over the whole skull)."""
    xs = (np.arange(W) + 0.5) / W
    ys = (np.arange(H) + 0.5) / H
    U, V = np.meshgrid(xs, ys)
    theta = (U * TAU + math.pi) % TAU
    phi = np.arccos(np.clip(2.0 * (1.0 - V) - 1.0, -1.0, 1.0))
    return skin_field(theta, phi, spec)


def _patch_bg(W: int, H: int, th: float, ph: float, dth: float, dph: float, spec) -> Image.Image:
    """Background image for a face-feature patch: identical shading to the head
    at that spot on the skull, so the patch edges vanish into the face."""
    xs = np.linspace(th - dth, th + dth, W)
    ys = np.linspace(ph - dph, ph + dph, H)
    TH, PH = np.meshgrid(xs, ys)
    return Image.fromarray(skin_field(TH, PH, spec).astype(np.uint8), "RGB")


def _uv_to_px(theta, phi, W, H, box):
    u, v = head_uv(theta, phi)
    x0, y0, x1, y1 = box
    return x0 + u * (x1 - x0), y0 + (1.0 - v) * (y1 - y0)


def paint_head(img: Image.Image, box, spec, m, size, ss: int):
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    w2, h2 = W * ss, H * ss
    field = _head_skin_field(w2, h2, spec)
    layer = Image.fromarray(field.astype(np.uint8), "RGB")
    d = ImageDraw.Draw(layer, "RGBA")
    skin = hex2rgb(spec["skin_tone"])
    rng = random.Random(int(sum(skin)) + int(float(spec["blush"]) * 100))

    # blush
    blush = float(spec["blush"])
    if blush > 0.02:
        for side in (1, -1):
            th = side * 0.52
            ph = math.pi * (MOUTH_PHI_RATIO - 0.135)
            cx, cy = _uv_to_px(th, ph, w2, h2, (0, 0, w2, h2))
            rx = 0.30 * w2 / 6.0
            ry = rx * 0.72
            col = warm(mix(skin, (232, 92, 96), 0.55), 0.15)
            for k in range(6, 0, -1):
                a = int(26 * blush * (7 - k) / 6 * 255 / 26)
                d.ellipse([cx - rx * k / 6, cy - ry * k / 6, cx + rx * k / 6, cy + ry * k / 6],
                          fill=col + (a,))

    # freckles
    fr = float(spec["freckles"])
    if fr > 0.02:
        n = int(30 + 90 * fr)
        for _ in range(n):
            th = rng.uniform(-0.62, 0.62)
            ph = math.pi * (MOUTH_PHI_RATIO - rng.uniform(0.10, 0.30))
            cx, cy = _uv_to_px(th, ph, w2, h2, (0, 0, w2, h2))
            r = rng.uniform(0.0016, 0.0034) * w2 / 0.5 * 0.5
            r = max(1.0, m.H * rng.uniform(0.0011, 0.0022) * (w2 / (TAU * m.head_r[0])))
            col = darken(warm(skin, 0.18), 0.32)
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col + (int(90 + 120 * fr),))

    # nose shadow
    if spec["nose_style"] != "none":
        cx, cy = _uv_to_px(0.0, math.pi * (MOUTH_PHI_RATIO - 0.036), w2, h2, (0, 0, w2, h2))
        rx = 0.055 * w2
        ry = rx * 0.5
        d.ellipse([cx - rx, cy - ry * 0.6, cx + rx, cy + ry * 1.3], fill=darken(skin, 0.20) + (60,))

    # painted facial hair (stubble / colour)
    beard = spec["beard"]
    if beard != "none":
        bcol = hex2rgb(spec["beard_color"])
        if beard == "stubble":
            for _ in range(1400):
                th = rng.uniform(-0.95, 0.95)
                ph = math.pi * (MOUTH_PHI_RATIO + rng.uniform(-0.045, 0.10))
                cx, cy = _uv_to_px(th, ph, w2, h2, (0, 0, w2, h2))
                r = max(1.0, 0.0016 * w2 / 0.5)
                d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=bcol + (70,))
        elif beard in ("full", "goatee"):
            for _ in range(900):
                th = rng.uniform(-1.05, 1.05)
                ph = math.pi * (MOUTH_PHI_RATIO + rng.uniform(-0.02, 0.115))
                cx, cy = _uv_to_px(th, ph, w2, h2, (0, 0, w2, h2))
                r = max(1.0, 0.0022 * w2 / 0.5)
                d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=bcol + (55,))
        if beard in ("mustache", "full"):
            for _ in range(500):
                th = rng.uniform(-0.55, 0.55)
                ph = math.pi * (MOUTH_PHI_RATIO - rng.uniform(0.048, 0.075))
                cx, cy = _uv_to_px(th, ph, w2, h2, (0, 0, w2, h2))
                r = max(1.0, 0.0024 * w2 / 0.5)
                d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=bcol + (150,))

    layer = layer.resize((W, H), Image.LANCZOS)
    img.paste(layer, (x0, y0))


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def paint(spec: dict, m, layout: dict, size: int) -> Image.Image:
    ss = max(1, int(spec.get("antialias", 2)))
    img = Image.new("RGB", (size, size), (26, 30, 38))
    skin = hex2rgb(spec["skin_tone"])
    rng = random.Random(1337 + int(m.H * 1000) + sum(skin))

    def cell(name):
        box = atlas.px_box(name, size)
        x0, y0, x1, y1 = box
        return box, (x1 - x0), (y1 - y0)

    def make(name, draw_fn, ss_override=None):
        box, W, H = cell(name)
        s = ss if ss_override is None else ss_override
        layer = Image.new("RGB", (W * s, H * s), (0, 0, 0))
        dd = ImageDraw.Draw(layer, "RGBA")
        draw_fn(dd, (0, 0, W * s, H * s), W * s, H * s, s)
        layer = layer.resize((W, H), Image.LANCZOS)
        img.paste(layer, (box[0], box[1]))
        return layer

    # ---- head ------------------------------------------------------------
    paint_head(img, cell("head")[0], spec, m, size, ss)

    # ---- skin (body + limbs + nose) --------------------------------------
    def skin_fn(dd, box, W, H, s):
        top, bot = lighten(skin, 0.05), darken(skin, 0.10)
        for y in range(H):
            dd.line([(box[0], box[1] + y), (box[2], box[1] + y)], fill=mix(top, bot, y / max(1, H - 1)))
        # subtle limb shading streaks
        for i in range(26):
            x = rng.randint(0, W)
            dd.line([(box[0] + x, box[1]), (box[0] + x + rng.randint(-W // 8, W // 8), box[3])],
                    fill=darken(skin, 0.06) + (60,), width=max(1, W // 90))
    make("skin", skin_fn, ss_override=1)

    # ---- eyes / brows / mouth (each its own patch cell) ------------------
    # NOTE: both eyes share one cell; the right eye's patch UVs are mirrored by
    # the mesh builder so asymmetric art (angry brows, pupil highlights) reads
    # correctly on both sides.
    eye_r = m.eye_r
    s = max(2, ss * 2)
    eye_th, eye_ph, _, _, eye_dth, eye_dph = layout["eye.L"]
    box, W, H = cell("eyes")
    layer = _patch_bg(W * s, H * s, eye_th, eye_ph, eye_dth, eye_dph, spec)
    draw_eye(layer, ImageDraw.Draw(layer, "RGBA"), (0, 0, W * s, H * s), spec, eye_r, skin, False)
    img.paste(layer.resize((W, H), Image.LANCZOS), (box[0], box[1]))

    box, W, H = cell("brows")
    bth, bph, _, _, bdth, bdph = layout["brow.L"]
    layer = _patch_bg(W * s, H * s, bth, bph, bdth, bdph, spec)
    draw_brow(layer, (0, 0, W * s, H * s), spec, eye_r, hex2rgb(spec["brow_color"]), False)
    img.paste(layer.resize((W, H), Image.LANCZOS), (box[0], box[1]))

    box, W, H = cell("mouth")
    mth, mph, _, _, mdth, mdph = layout["mouth"]
    layer = _patch_bg(W * s, H * s, mth, mph, mdth, mdph, spec)
    lip = mix(skin, warm(darken(skin, 0.35), 0.4), 0.55)
    draw_mouth(layer, (0, 0, W * s, H * s), spec, eye_r, skin, lip)
    img.paste(layer.resize((W, H), Image.LANCZOS), (box[0], box[1]))

    # ---- hair ------------------------------------------------------------
    root = hex2rgb(spec["hair_color"])
    tip = hex2rgb(spec["hair_tip_color"])

    def hair_fn(dd, box, W, H, s):
        for y in range(H):
            t = y / max(1, H - 1)
            dd.line([(box[0], box[1] + y), (box[2], box[1] + y)], fill=mix(root, tip, t ** 0.8))
        # strand streaks
        for i in range(60):
            x = rng.randint(0, W)
            y0 = int(H * rng.uniform(0.0, 0.5))
            ln = int(H * rng.uniform(0.25, 0.6))
            col = lighten(mix(root, tip, rng.random()), 0.10) if i % 2 else darken(mix(root, tip, rng.random()), 0.18)
            dd.line([(box[0] + x, box[1] + y0), (box[0] + x + rng.randint(-W // 30, W // 30), box[1] + y0 + ln)],
                    fill=col + (120,), width=max(1, s))
        # shine band
        shine = float(spec["hair_shine"])
        if shine > 0.02:
            for y in range(H):
                t = y / max(1, H - 1)
                a = int(70 * shine * math.exp(-((t - 0.30) ** 2) / 0.012))
                if a > 2:
                    dd.line([(box[0], box[1] + y), (box[2], box[1] + y)], fill=(255, 255, 255, a))
    make("hair", hair_fn, ss_override=max(1, ss - 1))

    # ---- facial hair volume ---------------------------------------------
    bcol = hex2rgb(spec["beard_color"])

    def facial_fn(dd, box, W, H, s):
        for y in range(H):
            t = y / max(1, H - 1)
            dd.line([(box[0], box[1] + y), (box[2], box[1] + y)], fill=mix(darken(bcol, 0.15), lighten(bcol, 0.12), t))
        for i in range(220):
            x = rng.randint(0, W)
            y = rng.randint(0, H)
            dd.line([(box[0] + x, box[1] + y), (box[0] + x + rng.randint(-W // 40, W // 40), box[1] + y + H // 12)],
                    fill=mix(bcol, (255, 255, 255), rng.uniform(0, 0.25)) + (110,), width=max(1, s))
    make("facial", facial_fn, ss_override=1)

    # ---- top -------------------------------------------------------------
    c1, c2 = hex2rgb(spec["top_color"]), hex2rgb(spec["top_color2"])

    def top_fn(dd, box, W, H, s):
        draw_pattern(dd, box, spec["top_pattern"], c1, c2, W, H, rng)
        # collar + hem detailing painted in texture space (v: 0 hem -> 1 shoulder)
        collar = spec["collar"]
        if spec["top_style"] != "tank" and collar in ("crew", "v"):
            band = int(H * 0.075)
            dd.rectangle([box[0], box[1], box[2], box[1] + band], fill=darken(c1, 0.28))
            if collar == "v":
                cx = box[0] + W // 2
                dd.polygon([(cx - W * 0.20, box[1] + band), (cx + W * 0.20, box[1] + band),
                            (cx, box[1] + H * 0.34)], fill=skin)
        if spec["top_style"] == "hoodie":
            dd.rectangle([box[0], box[2] and box[1] + H - int(H * 0.05), box[2], box[3]], fill=darken(c1, 0.22))
            # kangaroo pocket
            px0, px1 = box[0] + W * 0.30, box[0] + W * 0.70
            dd.rounded_rectangle([px0, box[1] + H * 0.55, px1, box[1] + H * 0.86], radius=int(H * 0.04),
                                 outline=darken(c1, 0.35), width=max(1, int(s * 1.2)))
        if spec["top_style"] == "armor":
            dd.rectangle([box[0], box[1], box[2], box[1] + int(H * 0.06)], fill=hex2rgb(spec["top_color2"]))
            dd.line([(box[0], box[1] + H * 0.30), (box[2], box[1] + H * 0.30)], fill=darken(c2, 0.2), width=max(1, int(s * 1.4)))
        # shoulder seam
        dd.line([(box[0], box[1] + H - 2 * s), (box[2], box[1] + H - 2 * s)], fill=darken(c1, 0.30), width=s)
        dd.line([(box[0], box[1] + 2 * s), (box[2], box[1] + 2 * s)], fill=darken(c1, 0.22), width=s)
    make("top", top_fn)

    # ---- bottom ----------------------------------------------------------
    b1, b2 = hex2rgb(spec["bottom_color"]), hex2rgb(spec["bottom_color2"])

    def bottom_fn(dd, box, W, H, s):
        draw_pattern(dd, box, spec["bottom_pattern"], b1, b2, W, H, rng)
        # waistband + cuffs
        dd.rectangle([box[0], box[1], box[2], box[1] + int(H * 0.05)], fill=darken(b1, 0.30))
        if spec["bottom_style"] in ("pants", "capri"):
            dd.rectangle([box[0], box[3] - int(H * 0.04), box[2], box[3]], fill=darken(b1, 0.25))
        if spec["bottom_style"] == "skirt":
            for i in range(-1, 8):
                a = i * W / 6.0
                dd.polygon([(box[0] + a, box[3]), (box[0] + a + W / 14, box[3]), (box[0] + a + W / 6, box[1])],
                           fill=darken(b1, 0.10) + (90,))
    make("bottom", bottom_fn)

    # ---- shoes -----------------------------------------------------------
    s1, s2 = hex2rgb(spec["shoes_color"]), hex2rgb(spec["shoes_color2"])

    def shoes_fn(dd, box, W, H, s):
        dd.rectangle(box, fill=s1)
        for i in range(9):
            dd.line([(box[0], box[1] + i * H // 9), (box[2], box[1] + i * H // 9)], fill=darken(s1, 0.05) + (40,),
                    width=max(1, s))
        dd.rectangle([box[0], box[1] + int(H * 0.42), box[2], box[1] + int(H * 0.58)], fill=s2)
        # laces
        for i in range(4):
            y = box[1] + int(H * (0.20 + i * 0.08))
            dd.line([(box[0] + W * 0.30, y), (box[0] + W * 0.70, y + H * 0.03)], fill=lighten(s2, 0.5),
                    width=max(1, int(s * 1.5)))
    make("shoes", shoes_fn, ss_override=1)

    # ---- extras (fur) / extras_skin --------------------------------------
    def fur_fn(dd, box, W, H, s):
        for y in range(H):
            t = y / max(1, H - 1)
            dd.line([(box[0], box[1] + y), (box[2], box[1] + y)], fill=mix(shade(root, 0.86), tip, t))
        # inner ear flush
        dd.rectangle([box[0] + W * 0.38, box[1] + H * 0.30, box[0] + W * 0.62, box[1] + H * 0.85],
                     fill=mix(hex2rgb("#f4b8c0"), root, 0.35) + (200,))
    make("extras", fur_fn, ss_override=1)

    def extras_skin_fn(dd, box, W, H, s):
        for y in range(H):
            t = y / max(1, H - 1)
            dd.line([(box[0], box[1] + y), (box[2], box[1] + y)], fill=mix(lighten(skin, 0.04), darken(skin, 0.12), t))
    make("extras_skin", extras_skin_fn, ss_override=1)

    # ---- hat -------------------------------------------------------------
    h1, h2 = hex2rgb(spec["hat_color"]), hex2rgb(spec["hat_color2"])

    def hat_fn(dd, box, W, H, s):
        dd.rectangle(box, fill=h1)
        dd.rectangle([box[0], box[1] + H * 0.60, box[2], box[3]], fill=h2)
        for i in range(0, W, max(2, W // 12)):
            dd.line([(box[0] + i, box[1]), (box[0] + i, box[3])], fill=darken(h1, 0.08) + (50,), width=max(1, s))
    make("hat", hat_fn, ss_override=1)

    # ---- glasses ---------------------------------------------------------
    gcol = darken(mix(hex2rgb(spec["shoes_color2"]), (40, 44, 52), 0.5), 0.1)

    def glasses_fn(dd, box, W, H, s):
        dd.rectangle(box, fill=gcol)
        for i in range(6):
            dd.line([(box[0] + i * W // 6, box[1]), (box[0] + i * W // 6, box[3])],
                    fill=lighten(gcol, 0.08) + (60,), width=max(1, s))
    make("glasses", glasses_fn, ss_override=1)

    return img
