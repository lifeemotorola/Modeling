"""
Colada Character Maker - UV Atlas layout
========================================
One baked texture holds every material.  This module is the shared truth for
where each material lives, used by both the mesh builder (to remap UVs) and
the texture painter (to know which pixels to draw in).

Rects are declared top-left origin in fractions of the image (fractions of a
4x4 grid of 0.25 cells), which is the natural way to think about a canvas.
The head and top cells are 2x1 cells (512x256 @1024) which suits the
equirect / wrapping mappings they use.
"""
from __future__ import annotations

import numpy as np

# name: (x0, y0, x1, y1) top-left origin, fractions of the atlas
CELLS: dict[str, tuple[float, float, float, float]] = {
    "head":        (0.00, 0.00, 0.50, 0.25),   # skull, equirect
    "top":         (0.50, 0.00, 1.00, 0.25),   # shirt / hoodie / armor, wraps in u
    "skin":        (0.00, 0.25, 0.25, 0.50),   # body + limbs + nose
    "bottom":      (0.25, 0.25, 0.50, 0.50),   # shorts / pants / skirt
    "shoes":       (0.50, 0.25, 0.75, 0.50),
    "hair":        (0.75, 0.25, 1.00, 0.50),
    "eyes":        (0.00, 0.50, 0.25, 0.75),
    "mouth":       (0.25, 0.50, 0.50, 0.75),
    "brows":       (0.50, 0.50, 0.75, 0.75),
    "facial":      (0.75, 0.50, 1.00, 0.75),   # moustache / beard volume
    "extras":      (0.00, 0.75, 0.25, 1.00),   # fur ears / tail
    "extras_skin": (0.25, 0.75, 0.50, 1.00),   # elf / round ears
    "hat":         (0.50, 0.75, 0.75, 1.00),
    "glasses":     (0.75, 0.75, 1.00, 1.00),
}

MATERIALS = list(CELLS.keys())


def uv_rect(cell: str) -> tuple[float, float, float, float]:
    """(u0, v0, u1, v1) in UV space, v pointing up."""
    x0, y0, x1, y1 = CELLS[cell]
    return (x0, 1.0 - y1, x1, 1.0 - y0)


def px_box(cell: str, size: int) -> tuple[int, int, int, int]:
    """(left, top, right, bottom) in pixels for a `size` square atlas."""
    x0, y0, x1, y1 = CELLS[cell]
    return (int(round(x0 * size)), int(round(y0 * size)), int(round(x1 * size)), int(round(y1 * size)))


def map_uv(cell: str, u, v):
    """Map local part UVs (0..1) into the atlas cell."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    u0, v0, u1, v1 = uv_rect(cell)
    eps = 0.0012
    u = u0 + eps + (u1 - u0 - 2 * eps) * np.clip(u, 0.0, 1.0)
    v = v0 + eps + (v1 - v0 - 2 * eps) * np.clip(v, 0.0, 1.0)
    return u, v
