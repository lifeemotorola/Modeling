"""
Colada Character Maker - Mesh builder
=====================================
Turns a spec + skeleton into a single skinned mesh.

Everything is generated from primitives (lathes, parallel-transport tubes,
super-ellipsoids, surface patches) so the character is fully parametric and
the UVs land in known atlas cells that the texture painter also uses.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field

import numpy as np

from . import atlas
from .skeleton import Metrics, Rig, _norm

TAU = math.pi * 2.0


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def smoothstep(x):
    """Works for floats and numpy arrays."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def axis_frame(d: np.ndarray, up_hint: np.ndarray | None = None) -> np.ndarray:
    """Orthonormal 3x3 whose column 1 is `d`."""
    y = _norm(np.asarray(d, dtype=np.float64))
    ref = np.array([1.0, 0.0, 0.0]) if abs(y[0]) < 0.92 else np.array([0.0, 0.0, 1.0])
    if up_hint is not None:
        ref = np.asarray(up_hint, dtype=np.float64)
        if abs(float(ref @ y)) > 0.97:
            ref = np.array([1.0, 0.0, 0.0])
    z = _norm(np.cross(ref, y))
    x = _norm(np.cross(y, z))
    return np.stack([x, y, z], axis=1)


def orient_outward(verts: np.ndarray, faces: list) -> list:
    """Flip faces if the closed mesh has negative signed volume."""
    vol = 0.0
    for a, b, c in faces:
        vol += float(np.dot(verts[a], np.cross(verts[b], verts[c])))
    if vol < 0:
        return [(a, c, b) for a, b, c in faces]
    return faces


def compute_normals(verts: np.ndarray, faces: list) -> np.ndarray:
    nrm = np.zeros_like(verts)
    for a, b, c in faces:
        n = np.cross(verts[b] - verts[a], verts[c] - verts[a])
        nrm[a] += n
        nrm[b] += n
        nrm[c] += n
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    ln[ln < 1e-12] = 1.0
    return nrm / ln


# ---------------------------------------------------------------------------
# Part
# ---------------------------------------------------------------------------

@dataclass
class Part:
    name: str
    material: str
    verts: np.ndarray
    faces: list
    uvs: np.ndarray
    bones: list
    power: float = 3.2
    bias: dict = dc_field(default_factory=dict)

    def __post_init__(self):
        self.verts = np.asarray(self.verts, dtype=np.float64)
        self.uvs = np.asarray(self.uvs, dtype=np.float64)


def transform(verts: np.ndarray, R: np.ndarray, t) -> np.ndarray:
    return verts @ R.T + np.asarray(t, dtype=np.float64)[None, :]


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def lathe(profile, segs: int, theta_offset: float = math.pi,
          close_bottom: bool = True, close_top: bool = True):
    """profile: [(y, rx, rz), ...] bottom -> top.  Closed surface of revolution."""
    n = len(profile)
    y0, y1 = profile[0][0], profile[-1][0]
    yspan = max(1e-6, y1 - y0)
    verts, uvs, faces = [], [], []
    for i, (y, rx, rz) in enumerate(profile):
        v = (y - y0) / yspan
        for j in range(segs):
            th = theta_offset + TAU * j / segs
            verts.append((rx * math.sin(th), y, rz * math.cos(th)))
            uvs.append((j / segs, v))
    for i in range(n - 1):
        for j in range(segs):
            j2 = (j + 1) % segs
            a = i * segs + j
            b = i * segs + j2
            c = (i + 1) * segs + j2
            d = (i + 1) * segs + j
            faces.append((a, b, c))
            faces.append((a, c, d))
    if close_bottom:
        y, rx, rz = profile[0]
        ci = len(verts)
        verts.append((0.0, y, 0.0))
        uvs.append((0.5, 0.0))
        for j in range(segs):
            j2 = (j + 1) % segs
            faces.append((ci, j2, j))
    if close_top:
        y = profile[-1][0]
        ci = len(verts)
        verts.append((0.0, y, 0.0))
        uvs.append((0.5, 1.0))
        base = (n - 1) * segs
        for j in range(segs):
            j2 = (j + 1) % segs
            faces.append((ci, base + j, base + j2))
    verts = np.array(verts, dtype=np.float64)
    return verts, faces, np.array(uvs, dtype=np.float64)


def tube(path, radii, segs: int = 16, cap_start: str = "round", cap_end: str = "round",
         up_hint=None, cap_rings: int = 3):
    """Parallel transported tube.  radii: scalar | (rx, rz) | list per point."""
    path = np.asarray(path, dtype=np.float64)
    n = len(path)
    if np.isscalar(radii):
        radii = [(float(radii), float(radii))] * n
    else:
        radii = [((float(r), float(r)) if np.isscalar(r) else (float(r[0]), float(r[1]))) for r in radii]

    # tangents
    tangents = np.zeros_like(path)
    for i in range(n):
        if i == 0:
            d = path[1] - path[0]
        elif i == n - 1:
            d = path[-1] - path[-2]
        else:
            d = path[i + 1] - path[i - 1]
        tangents[i] = _norm(d)

    # parallel transport frames
    ref = np.array([0.0, 0.0, 1.0]) if up_hint is None else np.asarray(up_hint, dtype=np.float64)
    if abs(float(ref @ tangents[0])) > 0.95:
        ref = np.array([1.0, 0.0, 0.0])
    u0 = _norm(np.cross(tangents[0], ref))
    frames = []
    u = u0
    for i in range(n):
        t = tangents[i]
        u = _norm(u - t * float(u @ t))
        v = np.cross(t, u)
        frames.append((u, v, t))

    rings: list[tuple[np.ndarray, np.ndarray, float]] = []  # (center, uv_v)
    caps: list[tuple[np.ndarray, np.ndarray, float, float]] = []

    def make_ring(center, u, v, rx, rz, v_uv):
        pts = []
        uvs = []
        for j in range(segs):
            th = TAU * j / segs
            pts.append(center + u * (rx * math.cos(th)) + v * (rz * math.sin(th)))
            uvs.append((j / segs, v_uv))
        return np.array(pts), np.array(uvs)

    total_len = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))) or 1.0
    acc = 0.0
    # start cap rings (extend backwards)
    if cap_start == "round":
        r0 = radii[0]
        for k in range(cap_rings, 0, -1):
            a = (math.pi / 2) * (k / cap_rings)
            c = path[0] - frames[0][2] * (r0[0] * math.sin(a))
            pts, uvs = make_ring(c, frames[0][0], frames[0][1], r0[0] * math.cos(a), r0[1] * math.cos(a), 0.0)
            rings.append((pts, uvs, 0.0))
    for i in range(n):
        if i > 0:
            acc += float(np.linalg.norm(path[i] - path[i - 1]))
        u, v, t = frames[i]
        pts, uvs = make_ring(path[i], u, v, radii[i][0], radii[i][1], acc / total_len)
        rings.append((pts, uvs, acc / total_len))
    if cap_end == "round":
        r1 = radii[-1]
        for k in range(1, cap_rings + 1):
            a = (math.pi / 2) * (k / cap_rings)
            c = path[-1] + frames[-1][2] * (r1[0] * math.sin(a))
            pts, uvs = make_ring(c, frames[-1][0], frames[-1][1], r1[0] * math.cos(a), r1[1] * math.cos(a), 1.0)
            rings.append((pts, uvs, 1.0))

    verts = np.concatenate([r[0] for r in rings], axis=0)
    uvs = np.concatenate([r[1] for r in rings], axis=0)
    nring = len(rings)
    faces = []
    for i in range(nring - 1):
        for j in range(segs):
            j2 = (j + 1) % segs
            a = i * segs + j
            b = i * segs + j2
            c = (i + 1) * segs + j2
            d = (i + 1) * segs + j
            faces.append((a, b, c))
            faces.append((a, c, d))
    if cap_start == "flat":
        ci = len(verts)
        verts = np.vstack([verts, path[0]])
        uvs = np.vstack([uvs, [0.5, 0.0]])
        for j in range(segs):
            j2 = (j + 1) % segs
            faces.append((ci, j2, j))
    if cap_end == "flat":
        base = (nring - 1) * segs
        ci = len(verts)
        verts = np.vstack([verts, path[-1]])
        uvs = np.vstack([uvs, [0.5, 1.0]])
        for j in range(segs):
            j2 = (j + 1) % segs
            faces.append((ci, base + j, base + j2))
    return verts, faces, uvs


def blob(center, radii, p: float = 2.0, segs: int = 24, rings: int = 16,
         R: np.ndarray | None = None, theta_offset: float = 0.0):
    """Super-ellipsoid (p=2 sphere, p>3 rounded box). Rings go bottom -> top."""
    radii = np.asarray(radii, dtype=np.float64)
    verts, uvs, faces = [], [], []
    pole_b = np.array([0.0, -1.0, 0.0])
    pole_t = np.array([0.0, 1.0, 0.0])

    def sph(theta, y):
        r = math.sqrt(max(0.0, 1.0 - y * y))
        return np.array([r * math.sin(theta), y, r * math.cos(theta)])

    def shape(nvec):
        denom = (abs(nvec[0]) ** p + abs(nvec[1]) ** p + abs(nvec[2]) ** p) ** (1.0 / p)
        return nvec / max(denom, 1e-9) * radii

    ring_idx = []
    for i in range(1, rings):
        y = -1.0 + 2.0 * i / rings          # bottom -> top
        v = i / rings
        idx0 = len(verts)
        ring_idx.append(idx0)
        for j in range(segs):
            th = theta_offset + TAU * j / segs
            verts.append(shape(sph(th, y)))
            uvs.append((j / segs, v))
    i_b = len(verts)
    verts.append(shape(pole_b))
    uvs.append((0.5, 0.0))
    i_t = len(verts)
    verts.append(shape(pole_t))
    uvs.append((0.5, 1.0))
    for j in range(segs):
        j2 = (j + 1) % segs
        faces.append((i_b, ring_idx[0] + j2, ring_idx[0] + j))
        faces.append((i_t, ring_idx[-1] + j, ring_idx[-1] + j2))
    for k in range(len(ring_idx) - 1):
        for j in range(segs):
            j2 = (j + 1) % segs
            a = ring_idx[k] + j
            b = ring_idx[k] + j2
            c = ring_idx[k + 1] + j2
            d = ring_idx[k + 1] + j
            faces.append((a, b, c))
            faces.append((a, c, d))
    verts = np.array(verts, dtype=np.float64)
    if R is not None:
        verts = transform(verts, np.asarray(R, dtype=np.float64), center)
    else:
        verts = verts + np.asarray(center, dtype=np.float64)[None, :]
    return verts, faces, np.array(uvs, dtype=np.float64)


def ellipsoid_point(c, radii, theta, phi):
    """Equirect point on an axis aligned ellipsoid. phi: 0 = top."""
    return np.asarray(c, dtype=np.float64) + np.array([
        radii[0] * math.sin(phi) * math.sin(theta),
        radii[1] * math.cos(phi),
        radii[2] * math.sin(phi) * math.cos(theta),
    ])


def head_deform(v: np.ndarray, hc, hr) -> np.ndarray:
    """Sculpting pass over the raw skull ellipsoid (jaw, cheeks, forehead).

    Every head mounted part (face patches, hair, glasses, hats) runs through
    this so nothing ever sinks into the skull or floats off it.
    """
    hc = np.asarray(hc, dtype=np.float64)
    hr = np.asarray(hr, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    out = v.copy()
    rel = (v - hc) / hr
    low = np.clip(-rel[:, 1], 0.0, 1.0)
    k = smoothstep(low * 1.25)
    out[:, 0] = hc[0] + (v[:, 0] - hc[0]) * (1.0 - 0.20 * k)
    out[:, 2] = hc[2] + (v[:, 2] - hc[2]) * (1.0 - 0.16 * k)
    cheek = np.exp(-((rel[:, 1] + 0.05) ** 2) / 0.06) * np.clip(rel[:, 2], 0, 1)
    out[:, 0] = hc[0] + (out[:, 0] - hc[0]) * (1.0 + 0.045 * cheek)
    out[:, 2] = hc[2] + (out[:, 2] - hc[2]) * (1.0 + 0.05 * cheek)
    top = np.clip(rel[:, 1] - 0.4, 0, 1)
    out[:, 2] = hc[2] + (out[:, 2] - hc[2]) * (1.0 - 0.06 * smoothstep(top * 1.6))
    return out


class HeadSurface:
    """Deformed skull surface with exact normals - the anchor for anything
    that gets stuck onto the head."""

    def __init__(self, hc, hr):
        self.hc = np.asarray(hc, dtype=np.float64)
        self.hr = np.asarray(hr, dtype=np.float64)

    def point(self, theta: float, phi: float, lift: float = 0.0) -> np.ndarray:
        p = ellipsoid_point(self.hc, self.hr, theta, phi)
        p = head_deform(p[None, :], self.hc, self.hr)[0]
        if lift:
            p = p + self.normal(theta, phi) * lift
        return p

    def normal(self, theta: float, phi: float) -> np.ndarray:
        e = 1e-4
        du = head_deform(ellipsoid_point(self.hc, self.hr, theta + e, phi)[None, :], self.hc, self.hr)[0]
        dv = head_deform(ellipsoid_point(self.hc, self.hr, theta, phi + e)[None, :], self.hc, self.hr)[0]
        p = self.point(theta, phi)
        n = np.cross(dv - p, du - p)      # outward (theta x phi cross order matters)
        ln = np.linalg.norm(n)
        if ln < 1e-12:
            return (p - self.hc) / np.maximum(1e-9, self.hr)
        n = n / ln
        # guard against a degenerate frame flipping it inward
        if float(n @ (p - self.hc)) < 0:
            n = -n
        return n


def head_uv(theta: float, phi: float, theta_offset: float = math.pi) -> tuple[float, float]:
    """Equirect UV used by the head mesh and the face painter.

    Matches the mesh blob exactly: u wraps from the back centre, v follows
    cos(phi) so the poles compress like a real sphere map.
    """
    u = ((theta - theta_offset) % TAU) / TAU
    v = (math.cos(phi) + 1.0) * 0.5
    return u, v


def patch(surface: HeadSurface, theta0, theta1, phi0, phi1, nu: int, nv: int, lift: float = 0.0,
          flip_u: bool = False):
    """Curved quad stuck onto the deformed skull (eyes / mouth / brows / blush).

    `flip_u` mirrors the sampled art horizontally - used so the right eye and
    brow share the left one's texture cell and still look correct.
    """
    verts, uvs, faces = [], [], []
    for i in range(nv + 1):
        phi = phi0 + (phi1 - phi0) * i / nv
        for j in range(nu + 1):
            th = theta0 + (theta1 - theta0) * j / nu
            verts.append(surface.point(th, phi, lift))
            u = j / nu
            uvs.append((1.0 - u if flip_u else u, 1.0 - i / nv))
    stride = nu + 1
    for i in range(nv):
        for j in range(nu):
            a = i * stride + j
            b = i * stride + j + 1
            c = (i + 1) * stride + j + 1
            d = (i + 1) * stride + j
            faces.append((a, c, b))
            faces.append((a, d, c))
    return np.array(verts, dtype=np.float64), faces, np.array(uvs, dtype=np.float64)


def torus(center, R: float, r: float, seg_u: int = 24, seg_v: int = 10, R_mat=None):
    verts, uvs, faces = [], [], []
    for i in range(seg_u):
        th = TAU * i / seg_u
        for j in range(seg_v):
            ph = TAU * j / seg_v
            x = (R + r * math.cos(ph)) * math.cos(th)
            y = r * math.sin(ph)
            z = (R + r * math.cos(ph)) * math.sin(th)
            verts.append((x, y, z))
            uvs.append((i / seg_u, j / seg_v))
    for i in range(seg_u):
        i2 = (i + 1) % seg_u
        for j in range(seg_v):
            j2 = (j + 1) % seg_v
            a = i * seg_v + j
            b = i * seg_v + j2
            c = i2 * seg_v + j2
            d = i2 * seg_v + j
            faces.append((a, b, c))
            faces.append((a, c, d))
    verts = np.array(verts, dtype=np.float64)
    if R_mat is not None:
        verts = transform(verts, np.asarray(R_mat, dtype=np.float64), center)
    else:
        verts = verts + np.asarray(center, dtype=np.float64)[None, :]
    return verts, faces, np.array(uvs, dtype=np.float64)


def cap_shell(surface: HeadSurface, boundary, nu: int = 28, nv: int = 8, scale: float = 1.0,
              rim: float = 0.0, theta_offset: float = math.pi):
    """Hair-like shell over the top of the skull.

    boundary(theta) -> max phi covered at that azimuth.  Returns an open surface
    plus (optionally) a rim band folding underneath, giving the cap thickness.
    """
    verts, uvs, faces = [], [], []
    thetas = [(theta_offset + TAU * j / nu) for j in range(nu)]
    bounds = [boundary(t) for t in thetas]
    c = surface.hc
    radii = surface.hr

    def pt(th, phi, sc):
        p = surface.point(th, phi)
        return c + (p - c) * sc

    rings = []
    for i in range(1, nv + 1):
        idx0 = len(verts)
        rings.append(idx0)
        for j, th in enumerate(thetas):
            phi = bounds[j] * (i / nv)
            verts.append(pt(th, phi, scale))
            uvs.append((j / nu, i / nv))
    i_pole = len(verts)
    verts.append(pt(0.0, 0.0, scale))
    uvs.append((0.0, 0.0))
    for j in range(nu):
        j2 = (j + 1) % nu
        faces.append((i_pole, rings[0] + j, rings[0] + j2))
    for k in range(len(rings) - 1):
        for j in range(nu):
            j2 = (j + 1) % nu
            a = rings[k] + j
            b = rings[k] + j2
            cc = rings[k + 1] + j2
            d = rings[k + 1] + j
            faces.append((a, b, cc))
            faces.append((a, cc, d))
    if rim > 0.0:
        idx0 = len(verts)
        for j, th in enumerate(thetas):
            phi = bounds[j]
            verts.append(pt(th, phi, scale * (1.0 - rim)))
            uvs.append((j / nu, 1.0))
        base = rings[-1]
        for j in range(nu):
            j2 = (j + 1) % nu
            a = base + j
            b = base + j2
            cc = idx0 + j2
            d = idx0 + j
            faces.append((a, d, cc))
            faces.append((a, cc, b))
    return np.array(verts, dtype=np.float64), faces, np.array(uvs, dtype=np.float64)


# ---------------------------------------------------------------------------
# Body metrics helpers
# ---------------------------------------------------------------------------

def lower_body_radii(m: Metrics, y: float, pad: float = 1.0):
    """Ellipse that covers both thighs at height `y` - used by anything hip mounted."""
    leg_span = max(1e-6, m.hip_y - m.ankle_y)
    frac = max(0.0, (m.hip_y - y) / leg_span)
    if frac < 0.62:
        thigh_r = m.r_thigh * (1.0 - 0.22 * min(1.0, frac * 1.6))
    else:
        thigh_r = m.r_calf * max(0.6, 1.14 - 0.28 * (frac - 0.62))
    return m.hip_x + thigh_r * pad, max(m.r_hip * 1.02, thigh_r * 1.45 * pad)


def torso_radii_at(m: Metrics, y: float) -> tuple[float, float]:
    prof = m.profile
    if y <= prof[0][0]:
        return prof[0][1], prof[0][2]
    if y >= prof[-1][0]:
        return prof[-1][1], prof[-1][2]
    for i in range(len(prof) - 1):
        y0, rx0, rz0 = prof[i]
        y1, rx1, rz1 = prof[i + 1]
        if y0 <= y <= y1:
            k = (y - y0) / max(1e-9, y1 - y0)
            return rx0 + (rx1 - rx0) * k, rz0 + (rz1 - rz0) * k
    return prof[-1][1], prof[-1][2]


def leg_path(m: Metrics, side_x: float, y_top: float, y_bottom: float, n: int = 8):
    """Points along the thigh->shin centre line between two heights."""
    hip = np.array([side_x, m.hip_y, 0.0])
    knee = np.array([side_x * 0.92, m.knee_y, m.knee[2]])
    ankle = np.array([side_x * 0.92, m.ankle_y, m.ankle[2]])
    def lerp3(a, b, t):
        return a + (b - a) * t
    pts = []
    total = m.hip_y - m.ankle_y
    for i in range(n):
        t = i / (n - 1)
        y = y_top + (y_bottom - y_top) * t
        # find matching point along hip->knee->ankle
        if y >= m.knee_y:
            k = (y - m.knee_y) / max(1e-9, m.hip_y - m.knee_y)
            p = lerp3(knee, hip, k)
        else:
            k = (y - m.ankle_y) / max(1e-9, m.knee_y - m.ankle_y)
            p = lerp3(ankle, knee, k)
        pts.append(p)
    return pts


# ---------------------------------------------------------------------------
# Character assembly
# ---------------------------------------------------------------------------

def build_mesh(spec: dict, m: Metrics, rig: Rig):
    hc, hr = m.head_c, m.head_r
    surf = HeadSurface(hc, hr)
    detail = float(spec.get("poly_detail", 1.0))
    segs = max(8, int(round(20 * detail)))
    hsegs = max(10, int(round(28 * detail)))
    parts: list[Part] = []

    def add(name, material, prim, bones, power=3.2, bias=None, outward=True, conform=False):
        verts, faces, uvs = prim
        verts = np.asarray(verts, dtype=np.float64)
        faces = [tuple(f) for f in faces]
        if conform:
            # glue head mounted parts onto the sculpted skull so nothing floats
            verts = head_deform(verts, hc, hr)
        if outward and len(faces) > 4:
            faces = orient_outward(verts, faces)
        parts.append(Part(name, material, verts, faces, np.asarray(uvs, dtype=np.float64),
                          list(bones), power, dict(bias or {})))

    H = m.H
    # =====================================================================
    # TORSO / SPINE
    # =====================================================================
    torso_profile = [(y, rx, rz) for (y, rx, rz) in m.profile]
    add("torso", "top" if spec["top_style"] in ("tee", "tank", "tropical", "hoodie", "armor") else "skin",
        lathe(torso_profile, segs, close_bottom=True, close_top=True),
        ["Hips", "Spine", "Chest", "Neck", "Shoulder.L", "Shoulder.R"])

    # neck
    neck_path = [np.array([0, m.neck_base_y - H * 0.01, 0.0]), np.array([0, m.neck_top_y + H * 0.012, 0.0])]
    add("neck", "skin", tube(neck_path, [m.r_neck * 1.05, m.r_neck],
                             segs=max(8, segs // 2), cap_start="round", cap_end="flat"),
        ["Neck", "Chest", "Head"], power=2.6)

    # =====================================================================
    # HEAD
    # =====================================================================
    hverts, hfaces, huvs = blob(hc, hr, p=2.0, segs=hsegs, rings=max(10, hsegs // 2), theta_offset=math.pi)
    add("head", "head", (head_deform(hverts, hc, hr), hfaces, huvs), ["Head", "Neck"], power=2.4,
        bias={"Head": 3.0})

    # face feature patches -------------------------------------------------
    fl = face_layout(m, spec)
    lift = H * 0.0022
    for side in ("L", "R"):
        th, ph, w, h, dth, dph = fl[f"eye.{side}"]
        add(f"eye.{side}", "eyes",
            patch(surf, th - dth, th + dth, ph - dph, ph + dph, 10, 10, lift=lift,
                  flip_u=(side == "R")),
            ["Head"], power=2.0, outward=False)
        bth, bph, bw, bh, bdth, bdph = fl[f"brow.{side}"]
        add(f"brow.{side}", "brows",
            patch(surf, bth - bdth, bth + bdth, bph - bdph, bph + bdph, 8, 5, lift=lift,
                  flip_u=(side == "R")),
            ["Head"], power=2.0, outward=False)

    mth, mph, mw, mh, mdth, mdph = fl["mouth"]
    add("mouth", "mouth",
        patch(surf, mth - mdth, mth + mdth, mph - mdph, mph + mdph, 12, 8, lift=lift),
        ["Head"], power=2.0, outward=False)

    nth, nph, _, _, _, _ = fl["nose"]
    if spec["nose_style"] != "none":
        nsz = {"button": 0.021, "small": 0.015, "snub": 0.024, "wide": 0.032}[spec["nose_style"]]
        npos = surf.point(nth, nph)
        add("nose", "skin", blob(npos + (hc - npos) * 0.10, (H * nsz * 0.5, H * nsz * 0.42, H * nsz * 0.55),
                                 p=2.1, segs=12, rings=8), ["Head"], power=2.0)
    if spec["beard"] != "none":
        if spec["beard"] in ("goatee", "full"):
            bp = surf.point(0.0, mph + 0.17, lift * 0.5)
            add("beard", "facial", blob(bp, (hr[0] * 0.34, hr[1] * 0.24, hr[2] * 0.26), p=2.0, segs=14, rings=10),
                ["Head"], power=2.0)
        if spec["beard"] in ("mustache", "full", "goatee"):
            mp = surf.point(0.0, mph - 0.19, lift * 0.5)
            add("mustache", "facial", blob(mp, (hr[0] * 0.30, hr[1] * 0.085, hr[2] * 0.15), p=2.2, segs=14, rings=8),
                ["Head"], power=2.0)

    if spec["ears"] in ("cat", "fox", "round", "elf"):
        emat = "extras_skin" if spec["ears"] in ("round", "elf") else "extras"
        for side, s in (("L", 1.0), ("R", -1.0)):
            base = surf.point(s * 1.15, 0.92, -hr[0] * 0.02)
            if spec["ears"] == "round":
                add(f"ear.{side}", emat, blob(base + np.array([0, hr[1] * 0.20, 0]),
                                              (hr[0] * 0.24, hr[1] * 0.24, hr[2] * 0.11), p=2.0, segs=12, rings=8),
                    [f"Ear.{side}", "Head"], power=2.5, conform=True)
            elif spec["ears"] == "elf":
                tip = base + np.array([s * hr[0] * 0.55, hr[1] * 0.45, -hr[2] * 0.16])
                add(f"ear.{side}", emat, tube([base, base * 0.45 + tip * 0.55, tip],
                                              [hr[0] * 0.22, hr[0] * 0.13, hr[0] * 0.02], segs=10,
                                              cap_start="flat", cap_end="round"),
                    [f"Ear.{side}", "Head"], power=2.5, conform=True)
            else:
                height = hr[1] * (1.05 if spec["ears"] == "fox" else 0.78)
                tip = base + np.array([s * hr[0] * 0.22, height, -hr[2] * 0.10])
                path = [base, base * 0.5 + tip * 0.5, tip]
                rad = [(hr[0] * 0.20, hr[2] * 0.055), (hr[0] * 0.12, hr[2] * 0.035), (hr[0] * 0.015, hr[2] * 0.012)]
                add(f"ear.{side}", emat, tube(path, rad, segs=10, cap_start="flat", cap_end="round"),
                    [f"Ear.{side}", "Head"], power=2.5, conform=True)

    if spec["tail"] != "none":
        base = np.array([0.0, m.hip_y + m.torso_h * 0.16, -m.r_hip * 0.82])
        L = m.r_hip * (2.9 if spec["tail"] != "puff" else 1.15)
        if spec["tail"] == "puff":
            add("tail", "extras", blob(base + np.array([0, -L * 0.15, -L * 0.35]), (L * 0.42, L * 0.42, L * 0.42),
                                       p=2.0, segs=16, rings=12), ["Hips", "Tail1"], power=2.5)
        else:
            dirs = [np.array([0.0, 0.25, -0.95]), np.array([0.0, -0.30, -0.95]), np.array([0.0, -0.80, -0.60]), np.array([0.0, -0.70, 0.35])]
            pts, rads = [base], []
            r0 = m.r_hip * (0.34 if spec["tail"] == "fox" else 0.30)
            rads.append(r0 * 0.9)
            for i, d in enumerate(dirs):
                pts.append(pts[-1] + _norm(d) * (L / len(dirs)))
                rads.append(r0 * (1.0 if i == 0 else (1.55 if i == 1 else (1.3 if i == 2 else 0.8))))
            add("tail", "extras", tube(pts, rads, segs=14, cap_start="flat", cap_end="round"),
                ["Hips", "Tail1", "Tail2", "Tail3"], power=2.6)

    # =====================================================================
    # ARMS + HANDS
    # =====================================================================
    for side, s in (("L", 1.0), ("R", -1.0)):
        root = np.array([s * m.arm_root[0], m.arm_root[1], m.arm_root[2]])
        el = np.array([s * m.elbow[0], m.elbow[1], m.elbow[2]])
        wr = np.array([s * m.wrist[0], m.wrist[1], m.wrist[2]])
        ht = np.array([s * m.hand_tip[0], m.hand_tip[1], m.hand_tip[2]])
        # deltoid - biased to the upper arm so it follows shoulder rotation
        add(f"deltoid.{side}", "skin", blob(root, (m.r_upper_arm * 1.32, m.r_upper_arm * 1.30, m.r_upper_arm * 1.26),
                                            p=2.0, segs=14, rings=10),
            [f"Shoulder.{side}", f"UpperArm.{side}", "Chest"], power=3.0,
            bias={f"UpperArm.{side}": 2.2, "Chest": 0.35})
        arm_path = [root, root * 0.5 + el * 0.5, el, el * 0.5 + wr * 0.5, wr]
        arm_rad = [m.r_upper_arm, m.r_upper_arm * 0.86, m.r_upper_arm * 0.84, m.r_forearm * 1.04, m.r_forearm * 0.92]
        add(f"arm.{side}", "skin", tube(arm_path, arm_rad, segs=max(10, segs - 4), cap_start="round", cap_end="flat"),
            [f"Shoulder.{side}", f"UpperArm.{side}", f"LowerArm.{side}", f"Hand.{side}"])

        # hand: mitten + thumb
        d = _norm(ht - wr)
        Rm = axis_frame(d)
        hand_len = float(np.linalg.norm(ht - wr))
        add(f"hand.{side}", "skin",
            blob(wr + d * hand_len * 0.55, (m.r_hand * 0.62, hand_len * 0.62, m.r_hand * 0.80),
                 p=2.6, segs=14, rings=10, R=Rm),
            [f"Hand.{side}", f"LowerArm.{side}"], power=3.0)
        thumb_off = Rm @ np.array([-s * m.r_hand * 0.62, hand_len * 0.18, m.r_hand * 0.12])
        add(f"thumb.{side}", "skin",
            blob(wr + d * hand_len * 0.30 + thumb_off, (m.r_hand * 0.24, m.r_hand * 0.42, m.r_hand * 0.24),
                 p=2.0, segs=10, rings=8, R=Rm),
            [f"Hand.{side}", f"LowerArm.{side}"], power=3.0)

    # =====================================================================
    # LEGS + FEET
    # =====================================================================
    for side, s in (("L", 1.0), ("R", -1.0)):
        hip = np.array([s * m.hip_x, m.hip_y, 0.0])
        kn = np.array([s * m.knee[0], m.knee[1], m.knee[2]])
        an = np.array([s * m.ankle[0], m.ankle[1], m.ankle[2]])
        leg_path_p = [hip, hip * 0.5 + kn * 0.5, kn, kn * 0.5 + an * 0.5, an]
        leg_rad = [m.r_thigh, m.r_thigh * 0.94, m.r_thigh * 0.80, m.r_calf * 1.06, m.r_calf * 0.80]
        add(f"leg.{side}", "skin", tube(leg_path_p, leg_rad, segs=max(10, segs - 2), cap_start="flat", cap_end="round"),
            [f"Thigh.{side}", f"Shin.{side}", f"Foot.{side}", "Hips"], power=3.0)

        # foot
        fwd = 0.55 if s > 0 else 0.55
        fc = np.array([s * m.ankle[0] * 1.0, m.foot_h * 0.62, an[2] + m.foot_len * 0.30])
        add(f"foot.{side}", "skin",
            blob(fc, (m.foot_w * 0.52, m.foot_h * 0.82, m.foot_len * 0.62), p=3.0, segs=16, rings=12),
            [f"Foot.{side}", f"Shin.{side}"], power=3.0)
        if spec["shoes_style"] in ("barefoot", "sandals", "none"):
            tc = fc + np.array([0, -m.foot_h * 0.12, m.foot_len * 0.44])
            add(f"toes.{side}", "skin",
                blob(tc, (m.foot_w * 0.42, m.foot_h * 0.34, m.foot_len * 0.14), p=2.4, segs=14, rings=10),
                [f"Toe.{side}", f"Foot.{side}"], power=3.0)

    # =====================================================================
    # HAIR
    # =====================================================================
    hair = spec["hair_style"]
    if hair != "bald":
        vol = float(spec["hair_volume"])
        bangs = float(spec["bangs"])

        def boundary(th: float) -> float:
            """phi limit of the hair cap as a function of azimuth (front -> back)."""
            a = math.cos(th - math.pi)     # 1 at the front
            front = {"buzz": 0.85, "short": 0.95, "bob": 0.96, "long": 0.97, "ponytail": 1.00,
                     "spiky": 1.00, "curly": 0.98, "mohawk": 0.55, "topknot": 0.95}[hair]
            back = {"buzz": 0.95, "short": 1.55, "bob": 2.35, "long": 2.25, "ponytail": 1.75,
                    "spiky": 1.45, "curly": 1.65, "mohawk": 1.30, "topknot": 1.45}[hair]
            front -= 0.30 * bangs if hair in ("short", "bob", "long", "ponytail", "curly", "spiky") else 0.0
            k = (1.0 - a) * 0.5                       # 0 front, 1 back
            b = front + (back - front) * (k ** 0.85)
            return min(math.pi * 0.97, b)

        scale = 1.0 + 0.055 * vol
        add("hair_cap", "hair",
            cap_shell(surf, boundary, nu=max(16, hsegs), nv=8, scale=scale, rim=0.045),
            ["Head", "HairBack"], power=2.2, bias={"Head": 2.0}, outward=False, conform=True)

        # strand / volume add-ons
        rng = np.random.default_rng(abs(hash((hair, round(vol, 3)))) % (2 ** 31))

        def strand(center, length, dirv, w=0.34, mat="hair", bones=None, taper=0.06):
            d = _norm(np.asarray(dirv, dtype=np.float64))
            p0 = np.asarray(center, dtype=np.float64)
            p1 = p0 + d * (length * 0.55) + np.array([0, length * 0.1, 0])
            p2 = p0 + d * length + np.array([0, -length * 0.25, 0])
            r0 = hr[0] * w * 0.42
            add("strand", mat, tube([p0, p1, p2], [r0 * 1.1, r0 * 0.85, r0 * taper], segs=10,
                                    cap_start="flat", cap_end="round"),
                bones or ["Head", "HairBack"], power=2.4, outward=True, conform=True)

        if hair == "spiky":
            n = 13
            for i in range(n):
                a = TAU * i / n + 0.3
                rad = hr[0] * (0.15 + 0.62 * ((i % 3) / 2.0))
                base = surf.point(a, 0.62 + 0.22 * ((i % 3) / 2.0))
                out = _norm((base - hc) * np.array([1, 0.55, 1]))
                strand(base, hr[1] * (0.9 + 0.5 * rng.random()), out, w=0.55)
        elif hair == "curly":
            n = 26
            for i in range(n):
                a = TAU * (i * 0.618) % TAU
                u = rng.random()
                phi = 0.25 + 0.95 * u
                base = surf.point(a, phi) * 0.0 + surf.point(a, phi)
                add("curl", "hair", blob(base, (hr[0] * 0.30, hr[0] * 0.30, hr[0] * 0.30), p=2.0, segs=10, rings=8),
                    ["Head", "HairBack"], power=2.4, conform=True)
        elif hair == "mohawk":
            n = 7
            for i in range(n):
                t = (i / (n - 1)) * 2 - 1
                base = surf.point(math.pi if t > 0 else 0.0, 0.55 + 0.42 * t * t)
                h = hr[1] * (0.75 - 0.28 * abs(t)) * vol
                add("mohawk", "hair", tube([base, base + np.array([0, h, 0])],
                                           [hr[0] * 0.16, hr[0] * 0.10], segs=10, cap_start="flat", cap_end="round"),
                    ["Head", "HairBack"], power=2.4, conform=True)
        elif hair == "topknot":
            base = surf.point(math.pi, 0.10) * 0.0 + surf.point(math.pi, 0.10) + np.array([0, hr[1] * 0.55, 0])
            add("knot", "hair", blob(base, (hr[0] * 0.42 * vol, hr[1] * 0.34 * vol, hr[0] * 0.42 * vol),
                                     p=2.0, segs=16, rings=12), ["Head"], power=2.4, conform=True)
        elif hair in ("bob", "long"):
            # side curtains + skirt of hair
            top_y = hc[1] + hr[1] * 0.35
            skirt_bottom = (m.neck_base_y - m.torso_h * 0.24) if hair == "long" else (hc[1] - hr[1] * 0.92)
            n = max(18, hsegs)
            prof = []
            steps = 8
            for i in range(steps):
                t = i / (steps - 1)
                y = top_y + (skirt_bottom - top_y) * t
                # hugs the skull at the top, then falls with a gentle flare
                flare = 1.03 + (0.14 if hair == "long" else 0.05) * smoothstep(t * 1.1)
                shrink = 1.0 - 0.10 * t * t          # tuck in toward the tips
                prof.append((y, hr[0] * 1.01 * flare * shrink, hr[2] * 1.01 * flare * shrink))
            add("hair_skirt", "hair", lathe(prof, n, close_bottom=False, close_top=False),
                ["Head", "HairBack", "Neck"], power=2.4, outward=False, conform=True)
        elif hair == "ponytail":
            b = surf.point(math.pi, 1.95) + np.array([0, 0, -hr[2] * 0.10])
            mid = b + np.array([0.0, -hr[1] * 0.45, -hr[2] * 0.55])
            tip = mid + np.array([0.0, -hr[1] * 0.85, hr[2] * 0.15])
            add("ponytail", "hair",
                tube([b, mid, tip], [hr[0] * 0.34 * vol, hr[0] * 0.30 * vol, hr[0] * 0.10], segs=14,
                     cap_start="flat", cap_end="round"),
                ["Head", "HairBack", "HairTail"], power=2.4, conform=True)
        elif hair == "buzz":
            pass

        # bangs: strands hanging from the hairline (they stop above the brows)
        if bangs > 0.02 and hair not in ("buzz", "mohawk"):
            n = 9
            for i in range(n):
                t = (i / (n - 1)) * 2 - 1
                a = t * 0.80
                phi0 = 0.92 - 0.22 * bangs - 0.10 * abs(t)
                base = surf.point(a, phi0) + surf.normal(a, phi0) * hr[0] * 0.03
                length = hr[1] * (0.16 + 0.30 * bangs) * (1.0 - 0.30 * abs(t))
                tip = base + np.array([0.0, -length, hr[2] * 0.06])
                r = hr[0] * (0.26 - 0.08 * abs(t))
                mid = base * 0.45 + tip * 0.55
                add("bang", "hair", tube([base, mid, tip], [r * 1.05, r * 0.85, r * 0.30],
                                         segs=10, cap_start="flat", cap_end="round"),
                    ["Head", "HairBack"], power=2.4, conform=True)

    # =====================================================================
    # CLOTHING
    # =====================================================================
    top_style = spec["top_style"]
    if top_style != "none":
        hem_y = m.hip_y + m.torso_h * (0.30 if top_style in ("tank",) else (0.12 if top_style == "hoodie" else 0.22))
        top_y = m.shoulder_y + m.torso_h * 0.045
        profile = []
        steps = 12
        for i in range(steps):
            t = i / (steps - 1)
            y = hem_y + (top_y - hem_y) * t
            rx, rz = torso_radii_at(m, y)
            inf = 1.055 + (0.02 if top_style == "hoodie" else 0.0)
            profile.append((y, rx * inf, rz * inf))
        # tank: stop before the shoulder tips
        if top_style == "tank":
            profile = [(y, rx, rz) for (y, rx, rz) in profile if y < m.shoulder_y - m.torso_h * 0.03]
        add("top_body", "top", lathe(profile, segs, close_bottom=True, close_top=(top_style != "tank")),
            ["Hips", "Spine", "Chest", "Shoulder.L", "Shoulder.R"], power=2.9)

        # collar
        collar = spec["collar"]
        if collar in ("crew", "v") and top_style != "tank":
            cy = profile[-1][0]
            rneck = m.r_neck * (1.22 if collar == "crew" else 1.12)
            cp = [(cy - m.torso_h * 0.015, rneck, rneck),
                  (cy + m.torso_h * 0.030, rneck * 1.10, rneck * 1.10)]
            add("collar", "top", lathe(cp, max(12, segs // 2), close_bottom=False, close_top=False),
                ["Chest", "Neck"], power=3.4, bias={"Neck": 1.6}, outward=False)
        hood = None
        if top_style == "hoodie" or collar == "hood":
            hy = m.shoulder_y
            hp = np.array([0.0, hy + m.torso_h * 0.06, -m.r_chest * 0.62])
            hood = blob(hp, (m.r_chest * 0.78, m.torso_h * 0.34, m.r_chest * 0.62), p=2.4, segs=18, rings=13)
            add("hood", "top", hood, ["Chest", "Neck", "Head"], power=2.6)

        # sleeves / straps
        sleeve = spec["sleeve"]
        for side, s in (("L", 1.0), ("R", -1.0)):
            root = np.array([s * m.arm_root[0], m.arm_root[1], m.arm_root[2]])
            el = np.array([s * m.elbow[0], m.elbow[1], m.elbow[2]])
            wr = np.array([s * m.wrist[0], m.wrist[1], m.wrist[2]])
            if top_style == "tank":
                # shoulder strap
                a = np.array([s * m.shoulder_x * 0.45, m.shoulder_y + m.torso_h * 0.02, m.r_chest * 0.30])
                b = np.array([s * m.shoulder_x * 0.45, m.shoulder_y + m.torso_h * 0.02, -m.r_chest * 0.34])
                add(f"strap.{side}", "top", tube([a, a * 0.5 + b * 0.5 + np.array([s * 0.01, m.torso_h * 0.06, 0]), b],
                                                 [m.r_upper_arm * 0.55, m.r_upper_arm * 0.6, m.r_upper_arm * 0.55],
                                                 segs=10, cap_start="flat", cap_end="flat"),
                    ["Chest", f"Shoulder.{side}"], power=2.8)
                continue
            if sleeve == "none":
                continue
            end = el if sleeve == "short" else wr
            path = [root - (root - el) * 0.12, root * 0.5 + el * 0.5, el]
            rads = [m.r_upper_arm * 1.34, m.r_upper_arm * 1.16, m.r_upper_arm * 1.06]
            if sleeve == "long":
                path = [root - (root - el) * 0.12, root * 0.5 + el * 0.5, el, el * 0.5 + wr * 0.5, wr]
                rads = [m.r_upper_arm * 1.34, m.r_upper_arm * 1.16, m.r_upper_arm * 1.04, m.r_forearm * 1.10, m.r_forearm * 1.04]
            add(f"sleeve.{side}", "top", tube(path, rads, segs=max(10, segs - 4), cap_start="round", cap_end="flat"),
                [f"Shoulder.{side}", f"UpperArm.{side}", f"LowerArm.{side}"], power=2.9)

        if top_style == "armor":
            for side, s in (("L", 1.0), ("R", -1.0)):
                root = np.array([s * m.arm_root[0], m.arm_root[1], m.arm_root[2]])
                add(f"pad.{side}", "top", blob(root + np.array([0, m.r_upper_arm * 0.10, 0]),
                                               (m.r_upper_arm * 1.42, m.r_upper_arm * 1.05, m.r_upper_arm * 1.40),
                                               p=2.6, segs=16, rings=11),
                    [f"Shoulder.{side}", f"UpperArm.{side}", "Chest"], power=3.0)
            # chest plate: a shell wrapped around the front of the torso
            pv, pf, puv = [], [], []
            nu, nv = 14, 6
            y0, y1 = m.chest_y - m.torso_h * 0.12, m.chest_y + m.torso_h * 0.16
            for i in range(nv + 1):
                y = y0 + (y1 - y0) * i / nv
                rx, rz = torso_radii_at(m, y)
                for j in range(nu + 1):
                    th = -0.82 + 1.64 * (j / nu)          # front facing arc
                    k = 1.09 - 0.05 * abs(j / nu - 0.5) * 2
                    pv.append((rx * k * math.sin(th), y, rz * k * math.cos(th)))
                    puv.append((j / nu, i / nv))
            for i in range(nv):
                for j in range(nu):
                    a_ = i * (nu + 1) + j
                    b_ = a_ + 1
                    c_ = a_ + nu + 2
                    d_ = a_ + nu + 1
                    pf.append((a_, b_, c_))
                    pf.append((a_, c_, d_))
            add("plate", "top", (np.array(pv), pf, np.array(puv)), ["Chest", "Spine"], power=2.9, outward=False)

    # --- bottoms ----------------------------------------------------------
    bstyle = spec["bottom_style"]
    if bstyle != "none":
        bl = float(spec["bottom_length"])
        waist_y = m.hip_y + m.torso_h * 0.30
        if bstyle == "skirt":
            hem_y = m.hip_y - (m.hip_y - m.knee_y) * (0.35 + 0.75 * bl)
            steps = 10
            prof = []
            for i in range(steps):
                t = i / (steps - 1)
                y = waist_y - (waist_y - hem_y) * t
                tx, tz = torso_radii_at(m, y)
                lx, lz = lower_body_radii(m, y, 1.12)
                flare = 1.0 + 0.42 * t * t
                prof.append((y, max(tx, lx) * flare, max(tz, lz) * flare))
            add("skirt", "bottom", lathe(prof, segs, close_bottom=True, close_top=True),
                ["Hips", "Spine"], power=2.6)
        else:
            # hip block: A-line from the waist down over the crotch, then two legs
            leg_span = max(1e-6, m.hip_y - m.ankle_y)
            crotch_y = m.hip_y - leg_span * 0.22
            top_rx, top_rz = torso_radii_at(m, waist_y)
            thigh_r0 = m.r_thigh
            bot_rx = m.hip_x + thigh_r0 * 1.06
            bot_rz = max(m.r_hip * 1.05, thigh_r0 * 1.50)
            hip_prof = []
            steps = 10
            for i in range(steps):
                t = i / (steps - 1)
                y = crotch_y + (waist_y - crotch_y) * t
                # full width over the pelvis, tapering into the waist at the top
                k = smoothstep((t - 0.40) / 0.60)
                lx, lz = lower_body_radii(m, y, 1.06)
                rx = lx + (top_rx * 1.05 - lx) * k
                rz = lz + (top_rz * 1.05 - lz) * k
                hip_prof.append((y, rx, rz))
            add("pants_hip", "bottom", lathe(hip_prof, segs, close_bottom=True, close_top=True),
                ["Hips", "Spine", "Thigh.L", "Thigh.R"], power=2.8)
            for side, s in (("L", 1.0), ("R", -1.0)):
                y_top = crotch_y + leg_span * 0.10          # tucked up inside the hip block
                y_bot = m.hip_y - leg_span * bl
                y_bot = min(y_bot, m.hip_y - leg_span * 0.12)
                if y_bot <= m.ankle_y:
                    y_bot = m.ankle_y - m.foot_h * 0.45
                pts = leg_path(m, s * m.hip_x, y_top, y_bot, n=7)
                rads = []
                for pp in pts:
                    frac = (m.hip_y - pp[1]) / leg_span
                    r = m.r_thigh * (1.0 - 0.20 * frac) if frac < 0.5 else m.r_calf * (1.12 - 0.22 * (frac - 0.5))
                    rads.append(r * (1.13 - 0.06 * min(1.0, frac * 1.6)) + m.H * 0.002)
                add(f"pantleg.{side}", "bottom", tube(pts, rads, segs=max(10, segs - 4), cap_start="round", cap_end="flat"),
                    [f"Thigh.{side}", f"Shin.{side}", "Hips", f"Foot.{side}"], power=2.9)
        if spec["belt"] and bstyle in ("shorts", "pants", "capri"):
            by = waist_y - m.torso_h * 0.02
            rx, rz = torso_radii_at(m, m.hip_y)
            prof = [(by - m.torso_h * 0.022, rx * 1.10, rz * 1.10), (by + m.torso_h * 0.022, rx * 1.10, rz * 1.10)]
            add("belt", "bottom", lathe(prof, segs, close_bottom=False, close_top=False),
                ["Hips", "Spine"], power=2.8, outward=False)
            add("buckle", "glasses", blob(np.array([0.0, by, rz * 1.10]),
                                        (m.r_hip * 0.20, m.torso_h * 0.030, m.r_hip * 0.06), p=3.0, segs=10, rings=8),
                ["Hips", "Spine"], power=2.8)

    # --- shoes ------------------------------------------------------------
    sh = spec["shoes_style"]
    if sh in ("sneakers", "boots", "sandals"):
        for side, s in (("L", 1.0), ("R", -1.0)):
            an = np.array([s * m.ankle[0], m.ankle_y, m.ankle[2]])
            fc = np.array([an[0], m.foot_h * 0.60, an[2] + m.foot_len * 0.30])
            if sh == "sneakers":
                add(f"shoe.{side}", "shoes",
                    blob(fc + np.array([0, m.foot_h * 0.18, 0]),
                         (m.foot_w * 0.70, m.foot_h * 1.45, m.foot_len * 0.72), p=2.7, segs=18, rings=12),
                    [f"Foot.{side}", f"Toe.{side}"], power=3.0)
                add(f"sole.{side}", "shoes",
                    blob(fc + np.array([0, -m.foot_h * 0.45, 0]),
                         (m.foot_w * 0.76, m.foot_h * 0.62, m.foot_len * 0.76), p=3.4, segs=18, rings=9),
                    [f"Foot.{side}", f"Toe.{side}"], power=3.0)
                add(f"tongue.{side}", "shoes",
                    blob(fc + np.array([0, m.foot_h * 1.25, -m.foot_len * 0.18]),
                         (m.foot_w * 0.52, m.foot_h * 0.45, m.foot_len * 0.30), p=2.4, segs=14, rings=9),
                    [f"Foot.{side}", f"Shin.{side}"], power=3.0)
                if spec["socks"]:
                    pts = leg_path(m, s * m.hip_x, m.ankle_y + m.foot_h * 1.6, m.ankle_y + m.foot_h * 0.5, n=3)
                    add(f"sock.{side}", "shoes", tube(pts, [m.r_calf * 0.98, m.r_calf * 0.92, m.r_calf * 0.86],
                                                      segs=12, cap_start="flat", cap_end="flat"),
                        [f"Shin.{side}", f"Foot.{side}"], power=3.0)
            elif sh == "boots":
                add(f"boot.{side}", "shoes",
                    blob(fc + np.array([0, m.foot_h * 0.15, 0]),
                         (m.foot_w * 0.75, m.foot_h * 1.5, m.foot_len * 0.78), p=2.8, segs=18, rings=12),
                    [f"Foot.{side}", f"Toe.{side}"], power=3.0)
                shaft_top = m.ankle_y + (m.knee_y - m.ankle_y) * 0.42
                pts = leg_path(m, s * m.hip_x, shaft_top, m.ankle_y + m.foot_h * 0.6, n=4)
                rads = [m.r_calf * 1.12, m.r_calf * 1.06, m.r_calf * 1.02, m.r_calf * 1.05]
                add(f"bootshaft.{side}", "shoes", tube(pts, rads, segs=14, cap_start="flat", cap_end="flat"),
                    [f"Shin.{side}", f"Foot.{side}"], power=3.0)
            elif sh == "sandals":
                add(f"sole.{side}", "shoes",
                    blob(fc + np.array([0, -m.foot_h * 0.42, 0]),
                         (m.foot_w * 0.72, m.foot_h * 0.5, m.foot_len * 0.74), p=3.4, segs=18, rings=9),
                    [f"Foot.{side}", f"Toe.{side}"], power=3.0)
                a = np.array([an[0] - m.foot_w * 0.55, an[1] + m.foot_h * 0.5, an[2] + m.foot_len * 0.10])
                b = np.array([an[0] + m.foot_w * 0.55, an[1] + m.foot_h * 0.5, an[2] + m.foot_len * 0.10])
                add(f"strap.{side}", "shoes", tube([a, (a + b) / 2 + np.array([0, m.foot_h * 0.3, 0]), b],
                                                   [m.foot_w * 0.16, m.foot_w * 0.17, m.foot_w * 0.16], segs=8,
                                                   cap_start="flat", cap_end="flat"),
                    [f"Foot.{side}", f"Shin.{side}"], power=3.0)

    # --- glasses ----------------------------------------------------------
    gl = spec["glasses"]
    if gl != "none":
        for side, s in (("L", 1.0), ("R", -1.0)):
            ex = s * m.eye_x
            ez = hc[2] + hr[2] * 1.06
            if gl in ("round", "square"):
                Rl = m.eye_r * (1.9 if gl == "round" else 1.85)
                rr = m.eye_r * (0.16 if gl == "round" else 0.14)
                if gl == "round":
                    v, f, uv = torus(np.array([ex, m.eye_y, ez]), Rl, rr, 20, 8, R_mat=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1.0]]))
                else:
                    v, f, uv = torus(np.array([ex, m.eye_y, ez]), Rl, rr, 4, 8, R_mat=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1.0]]))
                    # square-ish: no smoothing needed, 4 segments gives a rounded quad frame
                add(f"glasses_frame.{side}", "glasses", (v, f, uv), ["Head"], power=2.0, conform=True)
                add(f"glasses_lens.{side}", "glasses", blob(np.array([ex, m.eye_y, ez]),
                                                            (Rl * 0.94, Rl * 0.94, m.eye_r * 0.10), p=2.0, segs=16, rings=10),
                    ["Head"], power=2.0, conform=True)
            else:  # shades
                add(f"shades.{side}", "glasses",
                    blob(np.array([ex, m.eye_y, ez * 0.995]), (m.eye_r * 2.05, m.eye_r * 1.55, m.eye_r * 0.5),
                         p=3.0, segs=16, rings=11), ["Head"], power=2.0, conform=True)
            # temple arm
            a = np.array([s * (m.eye_x + m.eye_r * 1.9), m.eye_y + m.eye_r * 0.2, ez])
            b = np.array([s * hr[0] * 0.92, m.eye_y + m.eye_r * 0.1, hc[2] - hr[2] * 0.15])
            add(f"temple.{side}", "glasses", tube([a, a * 0.5 + b * 0.5, b],
                                                  [m.eye_r * 0.18, m.eye_r * 0.16, m.eye_r * 0.14], segs=8,
                                                  cap_start="round", cap_end="flat"),
                ["Head"], power=2.0, conform=True)
        add("bridge", "glasses",
            blob(np.array([0.0, m.eye_y + m.eye_r * 0.2, hc[2] + hr[2] * 1.03]),
                 (m.eye_x * 0.5, m.eye_r * 0.16, m.eye_r * 0.16), p=2.0, segs=12, rings=8),
            ["Head"], power=2.0, conform=True)

    # --- hat --------------------------------------------------------------
    hat = spec["hat"]
    if hat != "none":
        top_y = hc[1] + hr[1]
        if hat == "cap":
            dome = blob(np.array([0.0, hc[1] + hr[1] * 0.44, 0.0]), (hr[0] * 1.10, hr[1] * 0.60, hr[2] * 1.10),
                        p=2.0, segs=max(16, hsegs), rings=12)
            add("cap", "hat", dome, ["Head"], power=2.0, conform=True)
            add("visor", "hat", blob(np.array([0.0, hc[1] + hr[1] * 0.28, hr[2] * 0.98]),
                                     (hr[0] * 0.92, hr[1] * 0.08, hr[2] * 0.55), p=2.6, segs=18, rings=9),
                ["Head"], power=2.0, conform=True)
        elif hat == "beanie":
            add("beanie", "hat", blob(np.array([0.0, hc[1] + hr[1] * 0.50, 0.0]),
                                      (hr[0] * 1.12, hr[1] * 0.66, hr[2] * 1.12), p=2.0, segs=max(16, hsegs), rings=13),
                ["Head"], power=2.0, conform=True)
            add("brim", "hat", torus(np.array([0.0, hc[1] + hr[1] * 0.24, 0.0]), hr[0] * 1.10, hr[0] * 0.10,
                                     24, 8), ["Head"], power=2.0, conform=True)
            add("pom", "hat", blob(np.array([0.0, top_y + hr[1] * 0.06, 0.0]),
                                   (hr[0] * 0.26, hr[0] * 0.26, hr[0] * 0.26), p=2.0, segs=14, rings=10),
                ["Head"], power=2.0, conform=True)
        elif hat == "sunhat":
            bv, bf, buv = lathe([(hc[1] + hr[1] * 0.74, hr[0] * 1.06, hr[2] * 1.06),
                                 (hc[1] + hr[1] * 0.68, hr[0] * 1.68, hr[2] * 1.68)],
                                max(20, hsegs), close_bottom=False, close_top=False)
            add("brim", "hat", (bv, bf, buv), ["Head"], power=2.0, outward=False)
            add("crown", "hat", blob(np.array([0.0, hc[1] + hr[1] * 0.98, 0.0]),
                                     (hr[0] * 1.04, hr[1] * 0.50, hr[2] * 1.04), p=2.0, segs=20, rings=12),
                ["Head"], power=2.0, conform=True)
            add("band", "hat", torus(np.array([0.0, hc[1] + hr[1] * 0.80, 0.0]), hr[0] * 1.05, hr[0] * 0.06, 22, 8),
                ["Head"], power=2.0, conform=True)
        elif hat == "wizard":
            base_y = hc[1] + hr[1] * 0.56
            prof = []
            steps = 8
            for i in range(steps):
                t = i / (steps - 1)
                prof.append((base_y + hr[1] * 2.6 * t, hr[0] * (1.05 - 1.00 * t ** 1.25), hr[0] * (1.05 - 1.00 * t ** 1.25)))
            add("wizcone", "hat", lathe(prof, max(18, hsegs), close_bottom=True, close_top=True),
                ["Head"], power=2.0, conform=True)
            add("wizbrim", "hat", lathe([(base_y - hr[1] * 0.02, hr[0] * 1.9, hr[2] * 1.9),
                                         (base_y + hr[1] * 0.06, hr[0] * 1.05, hr[2] * 1.05)],
                                        max(20, hsegs), close_bottom=False, close_top=False),
                ["Head"], power=2.0, outward=False)

    # =====================================================================
    # ASSEMBLE
    # =====================================================================
    all_verts, all_uvs, all_norms, all_faces, all_joints, all_weights = [], [], [], [], [], []
    ranges: list[tuple[str, int, int]] = []   # (material, index_start, index_count)
    offset = 0
    for part in parts:
        n_v = len(part.verts)
        u, v = atlas.map_uv(part.material, part.uvs[:, 0], part.uvs[:, 1])
        all_verts.append(part.verts.astype(np.float32))
        all_uvs.append(np.stack([u, v], axis=1).astype(np.float32))
        all_norms.append(compute_normals(part.verts, part.faces).astype(np.float32))
        idx, wts = rig.weights_for(part.verts, part.bones, power=part.power, bias=part.bias)
        all_joints.append(idx.astype(np.uint16))
        all_weights.append(wts)
        faces = np.array(part.faces, dtype=np.int64) + offset
        all_faces.append(faces)
        ranges.append((part.material, int(np.sum([len(f) for f in all_faces[:-1]])) * 3, len(part.faces) * 3))
        offset += n_v

    verts = np.concatenate(all_verts, axis=0)
    uvs = np.concatenate(all_uvs, axis=0)
    norms = np.concatenate(all_norms, axis=0)
    joints = np.concatenate(all_joints, axis=0)
    weights = np.concatenate(all_weights, axis=0)
    faces = np.concatenate(all_faces, axis=0).astype(np.uint32)

    # index ranges per material
    index_ranges: dict[str, list[tuple[int, int]]] = {}
    for material, start, count in ranges:
        index_ranges.setdefault(material, []).append((start, count))

    return MeshData(verts, uvs, norms, faces, joints, weights, index_ranges)


@dataclass
class MeshData:
    verts: np.ndarray
    uvs: np.ndarray
    normals: np.ndarray
    faces: np.ndarray
    joints: np.ndarray
    weights: np.ndarray
    index_ranges: dict


# ---------------------------------------------------------------------------
# Face feature placement shared with the painter
# ---------------------------------------------------------------------------

MOUTH_PHI_RATIO = 0.646   # as a fraction of pi, from the top of the skull


def _surf(local: np.ndarray) -> tuple[float, float]:
    n = local / max(1e-9, float(np.linalg.norm(local)))
    th = math.atan2(float(n[0]), float(n[2]))
    ph = math.acos(float(np.clip(n[1], -1.0, 1.0)))
    return th, ph


def face_layout(m: Metrics, spec: dict) -> dict:
    """Where every face feature sits on the head ellipsoid.

    Returns patches as (theta_center, phi_center, half_width_m, half_height_m,
    theta_half_rad, phi_half_rad).  Shared by the mesh builder (geometry) and
    the texture painter (art), so the eyes and the eyelash art always agree.
    """
    hc, hr = m.head_c, m.head_r
    out: dict = {}
    for side, s in (("L", 1.0), ("R", -1.0)):
        ex = s * m.eye_x
        ey = m.eye_y
        ez = hc[2] + hr[2] * math.sqrt(max(0.05, 1.0 - (m.eye_x / hr[0]) ** 2 - ((m.eye_y - hc[1]) / hr[1]) ** 2))
        th, ph = _surf((np.array([ex, ey, ez]) - hc) / hr)
        hw = m.eye_r * (0.95 + 0.30 * (1.0 - float(spec["eye_spacing"])))
        hh = m.eye_r * 0.68
        out[f"eye.{side}"] = (th, ph, hw, hh, hw / hr[0], hh / hr[1])
        bh = m.eye_r * 0.26
        bw = hw * 1.30
        brow_ph = ph - (hh * 1.25 + bh * 1.7) / hr[1]
        out[f"brow.{side}"] = (th, brow_ph, bw, bh, bw / hr[0], bh / hr[1])
        out[f"blush.{side}"] = (th + s * 0.14, ph + (hh * 1.35) / hr[1],
                                m.eye_r * 1.6, m.eye_r * 0.8, 0.0, 0.0)

    mph = math.pi * MOUTH_PHI_RATIO
    mhw = m.eye_r * 2.05
    mhh = m.eye_r * 0.72
    out["mouth"] = (0.0, mph, mhw, mhh, mhw / hr[0], mhh / hr[1])
    out["nose"] = (0.0, mph - (m.eye_r * 1.15) / hr[1], m.eye_r * 0.9, m.eye_r * 0.6, 0.0, 0.0)
    return out
