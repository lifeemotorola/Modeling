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
    if not faces:
        return faces
    f = np.asarray(faces, dtype=np.int64)
    c = verts.mean(axis=0)
    v = verts - c
    vol = float(np.sum(np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]]))))
    if vol < 0:
        return [(int(a), int(cc), int(b)) for a, b, cc in faces]
    return faces


def compute_normals(verts: np.ndarray, faces: list) -> np.ndarray:
    if not faces:
        return np.zeros_like(verts)
    f = np.asarray(faces, dtype=np.int64)
    fn = np.cross(verts[f[:, 1]] - verts[f[:, 0]], verts[f[:, 2]] - verts[f[:, 0]])
    # Weld coincident seam vertices so UV seams share smooth normals
    _, inv = np.unique(np.round(verts, 5), axis=0, return_inverse=True)
    n_uniq = int(inv.max()) + 1 if inv.size else 0
    accum = np.zeros((n_uniq, 3), dtype=np.float64)
    np.add.at(accum, inv[f[:, 0]], fn)
    np.add.at(accum, inv[f[:, 1]], fn)
    np.add.at(accum, inv[f[:, 2]], fn)
    nrm = accum[inv]
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
    normals: np.ndarray | None = None

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
    stride = segs + 1
    verts, uvs, faces = [], [], []
    for i, (y, rx, rz) in enumerate(profile):
        v = (y - y0) / yspan
        for j in range(stride):
            th = theta_offset + TAU * (j % segs) / segs
            verts.append((rx * math.sin(th), y, rz * math.cos(th)))
            uvs.append((j / segs, v))
    for i in range(n - 1):
        for j in range(segs):
            j2 = j + 1
            a = i * stride + j
            b = i * stride + j2
            c = (i + 1) * stride + j2
            d = (i + 1) * stride + j
            faces.append((a, b, c))
            faces.append((a, c, d))
    if close_bottom:
        y, rx, rz = profile[0]
        ci = len(verts)
        verts.append((0.0, y, 0.0))
        uvs.append((0.5, 0.0))
        for j in range(segs):
            faces.append((ci, j + 1, j))
    if close_top:
        y = profile[-1][0]
        ci = len(verts)
        verts.append((0.0, y, 0.0))
        uvs.append((0.5, 1.0))
        base = (n - 1) * stride
        for j in range(segs):
            faces.append((ci, base + j, base + j + 1))
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
    stride = segs + 1

    def make_ring(center, u, v, rx, rz, v_uv):
        pts = []
        uvs = []
        for j in range(stride):
            th = TAU * (j % segs) / segs
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
            j2 = j + 1
            a = i * stride + j
            b = i * stride + j2
            c = (i + 1) * stride + j2
            d = (i + 1) * stride + j
            faces.append((a, b, c))
            faces.append((a, c, d))
    if cap_start == "flat":
        ci = len(verts)
        verts = np.vstack([verts, path[0]])
        uvs = np.vstack([uvs, [0.5, 0.0]])
        for j in range(segs):
            faces.append((ci, j + 1, j))
    if cap_end == "flat":
        base = (nring - 1) * stride
        ci = len(verts)
        verts = np.vstack([verts, path[-1]])
        uvs = np.vstack([uvs, [0.5, 1.0]])
        for j in range(segs):
            faces.append((ci, base + j, base + j + 1))
    return verts, faces, uvs


def blob(center, radii, p: float = 2.0, segs: int = 24, rings: int = 16,
         R: np.ndarray | None = None, theta_offset: float = 0.0):
    """Super-ellipsoid (p=2 sphere, p>3 rounded box). Rings go bottom -> top."""
    radii = np.asarray(radii, dtype=np.float64)
    verts, uvs, faces = [], [], []
    pole_b = np.array([0.0, -1.0, 0.0])
    pole_t = np.array([0.0, 1.0, 0.0])
    stride = segs + 1

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
        for j in range(stride):
            th = theta_offset + TAU * (j % segs) / segs
            verts.append(shape(sph(th, y)))
            uvs.append((j / segs, v))
    i_b = len(verts)
    verts.append(shape(pole_b))
    uvs.append((0.5, 0.0))
    i_t = len(verts)
    verts.append(shape(pole_t))
    uvs.append((0.5, 1.0))
    for j in range(segs):
        faces.append((i_b, ring_idx[0] + j + 1, ring_idx[0] + j))
        faces.append((i_t, ring_idx[-1] + j, ring_idx[-1] + j + 1))
    for k in range(len(ring_idx) - 1):
        for j in range(segs):
            j2 = j + 1
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
    verts, uvs, norms, faces = [], [], [], []
    for i in range(nv + 1):
        phi = phi0 + (phi1 - phi0) * i / nv
        for j in range(nu + 1):
            th = theta0 + (theta1 - theta0) * j / nu
            verts.append(surface.point(th, phi, lift))
            norms.append(surface.normal(th, phi))
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
    return (np.array(verts, dtype=np.float64), faces,
            np.array(uvs, dtype=np.float64), np.array(norms, dtype=np.float64))


def torus(center, R: float, r: float, seg_u: int = 24, seg_v: int = 10, R_mat=None):
    verts, uvs, faces = [], [], []
    stride_u = seg_u + 1
    stride_v = seg_v + 1
    for i in range(stride_u):
        th = TAU * (i % seg_u) / seg_u
        for j in range(stride_v):
            ph = TAU * (j % seg_v) / seg_v
            x = (R + r * math.cos(ph)) * math.cos(th)
            y = r * math.sin(ph)
            z = (R + r * math.cos(ph)) * math.sin(th)
            verts.append((x, y, z))
            uvs.append((i / seg_u, j / seg_v))
    for i in range(seg_u):
        i2 = i + 1
        for j in range(seg_v):
            j2 = j + 1
            a = i * stride_v + j
            b = i * stride_v + j2
            c = i2 * stride_v + j2
            d = i2 * stride_v + j
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
    stride = nu + 1
    thetas = [(theta_offset + TAU * (j % nu) / nu) for j in range(stride)]
    bounds = [boundary(t) for t in thetas]
    c = surface.hc

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
            uvs.append((j / nu, 1.0 - i / nv))
    i_pole = len(verts)
    verts.append(pt(0.0, 0.0, scale))
    uvs.append((0.5, 1.0))
    for j in range(nu):
        faces.append((i_pole, rings[0] + j + 1, rings[0] + j))
    for k in range(len(rings) - 1):
        for j in range(nu):
            j2 = j + 1
            a = rings[k] + j
            b = rings[k] + j2
            cc = rings[k + 1] + j2
            d = rings[k + 1] + j
            faces.append((a, cc, b))
            faces.append((a, d, cc))
    if rim > 0.0:
        idx0 = len(verts)
        for j, th in enumerate(thetas):
            phi = bounds[j]
            verts.append(pt(th, phi, scale * (1.0 - rim)))
            uvs.append((j / nu, 0.0))
        base = rings[-1]
        for j in range(nu):
            j2 = j + 1
            a = base + j
            b = base + j2
            cc = idx0 + j2
            d = idx0 + j
            faces.append((a, cc, b))
            faces.append((a, d, cc))
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

def remap_uv_box(prim, u0: float, v0: float, u1: float, v1: float):
    """Remap a primitive's [0..1] UVs into a sub-rectangle [u0..u1, v0..v1] of its cell."""
    verts, faces, uvs = prim[0], prim[1], np.asarray(prim[2], dtype=np.float64).copy()
    uvs[:, 0] = u0 + (u1 - u0) * np.clip(uvs[:, 0], 0.0, 1.0)
    uvs[:, 1] = v0 + (v1 - v0) * np.clip(uvs[:, 1], 0.0, 1.0)
    return (verts, faces, uvs) if len(prim) == 3 else (verts, faces, uvs, prim[3])


def build_mesh(spec: dict, m: Metrics, rig: Rig):
    hc, hr = m.head_c, m.head_r
    surf = HeadSurface(hc, hr)
    detail = float(spec.get("poly_detail", 1.0))
    segs = max(8, int(round(20 * detail)))
    hsegs = max(12, int(round(28 * detail)))
    parts: list[Part] = []

    def add(name, material, prim, bones, power=3.2, bias=None, outward=True, conform=False):
        if len(prim) == 4:
            verts, faces, uvs, norms = prim
            norms = np.asarray(norms, dtype=np.float64)
        else:
            verts, faces, uvs = prim
            norms = None
        verts = np.asarray(verts, dtype=np.float64)
        faces = [tuple(f) for f in faces]
        if conform:
            # glue head mounted parts onto the sculpted skull so nothing floats
            verts = head_deform(verts, hc, hr)
        if outward and len(faces) > 4 and norms is None:
            faces = orient_outward(verts, faces)
        parts.append(Part(name, material, verts, faces, np.asarray(uvs, dtype=np.float64),
                          list(bones), power, dict(bias or {}), normals=norms))

    H = m.H
    # =====================================================================
    # TORSO / SPINE
    # =====================================================================
    torso_profile = [(y, rx, rz) for (y, rx, rz) in m.profile]
    add("torso", "top" if spec["top_style"] in ("tee", "tropical", "hoodie", "armor") else "skin",
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
    hrings = max(16, int(round(hsegs * 0.75)))
    hverts, hfaces, huvs = blob(hc, hr, p=2.0, segs=hsegs, rings=hrings, theta_offset=math.pi)
    hverts_def = head_deform(hverts, hc, hr)
    # Analytical surface normals on the head match the face patches seamlessly
    hnorms = np.zeros_like(hverts_def)
    stride_h = hsegs + 1
    for i in range(1, hrings):
        y = -1.0 + 2.0 * i / hrings
        phi = math.acos(max(-1.0, min(1.0, y)))
        base_idx = (i - 1) * stride_h
        for j in range(stride_h):
            th = math.pi + TAU * (j % hsegs) / hsegs
            hnorms[base_idx + j] = surf.normal(th, phi)
    hnorms[-2] = np.array([0.0, -1.0, 0.0])
    hnorms[-1] = np.array([0.0, 1.0, 0.0])
    add("head", "head", (hverts_def, hfaces, huvs, hnorms), ["Head", "Neck"], power=2.4,
        bias={"Head": 3.0})

    # face feature patches -------------------------------------------------
    fl = face_layout(m, spec)
    lift = H * 0.0009
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
        add("nose", "skin", blob(npos + (hc - npos) * 0.12, (H * nsz * 0.48, H * nsz * 0.40, H * nsz * 0.50),
                                 p=2.1, segs=12, rings=8), ["Head"], power=2.0)
    if spec["beard"] != "none":
        if spec["beard"] in ("goatee", "full"):
            bp = surf.point(0.0, mph + 0.22, -hr[2] * 0.10)
            bw_r = hr[0] * (0.58 if spec["beard"] == "full" else 0.28)
            bh_r = hr[1] * (0.24 if spec["beard"] == "full" else 0.18)
            bz_r = hr[2] * 0.20
            add("beard", "facial", blob(bp, (bw_r, bh_r, bz_r), p=2.2, segs=14, rings=10),
                ["Head"], power=2.0)
        if spec["beard"] in ("mustache", "full", "goatee"):
            mp = surf.point(0.0, mph - 0.11, -hr[2] * 0.03)
            add("mustache", "facial", blob(mp, (hr[0] * 0.26, hr[1] * 0.055, hr[2] * 0.09), p=2.2, segs=14, rings=8),
                ["Head"], power=2.0)

    if spec["ears"] in ("cat", "fox", "round", "elf"):
        emat = "extras_skin" if spec["ears"] in ("round", "elf") else "extras"
        for side, s in (("L", 1.0), ("R", -1.0)):
            if spec["ears"] in ("cat", "fox"):
                base = surf.point(s * 0.92, 0.58, -hr[0] * 0.03)
                height = hr[1] * (0.68 if spec["ears"] == "fox" else 0.52)
                tip = base + np.array([s * hr[0] * 0.24, height, -hr[2] * 0.06])
                path = [base, base * 0.5 + tip * 0.5, tip]
                rad = [(hr[0] * 0.22, hr[2] * 0.08), (hr[0] * 0.14, hr[2] * 0.05), (hr[0] * 0.02, hr[2] * 0.018)]
                add(f"ear.{side}", emat, tube(path, rad, segs=10, cap_start="flat", cap_end="round"),
                    [f"Ear.{side}", "Head"], power=2.5)
            elif spec["ears"] == "round":
                base = surf.point(s * 1.48, 1.45, -hr[0] * 0.02)
                add(f"ear.{side}", emat, blob(base + np.array([s * hr[0] * 0.06, 0, 0]),
                                              (hr[0] * 0.16, hr[1] * 0.18, hr[2] * 0.09), p=2.0, segs=12, rings=8),
                    [f"Ear.{side}", "Head"], power=2.5)
            elif spec["ears"] == "elf":
                base = surf.point(s * 1.45, 1.45, -hr[0] * 0.02)
                tip = base + np.array([s * hr[0] * 0.48, hr[1] * 0.28, -hr[2] * 0.14])
                add(f"ear.{side}", emat, tube([base, base * 0.45 + tip * 0.55, tip],
                                              [hr[0] * 0.16, hr[0] * 0.10, hr[0] * 0.02], segs=10,
                                              cap_start="flat", cap_end="round"),
                    [f"Ear.{side}", "Head"], power=2.5)

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
    sleeved_top = spec["top_style"] not in ("none", "tank") and spec["sleeve"] != "none"
    for side, s in (("L", 1.0), ("R", -1.0)):
        root = np.array([s * m.arm_root[0], m.arm_root[1], m.arm_root[2]])
        el = np.array([s * m.elbow[0], m.elbow[1], m.elbow[2]])
        wr = np.array([s * m.wrist[0], m.wrist[1], m.wrist[2]])
        ht = np.array([s * m.hand_tip[0], m.hand_tip[1], m.hand_tip[2]])
        # deltoid - biased to the upper arm so it follows shoulder rotation
        delt_prim = blob(root, (m.r_upper_arm * 1.28, m.r_upper_arm * 1.26, m.r_upper_arm * 1.24),
                         p=2.0, segs=14, rings=10)
        if sleeved_top:
            delt_prim = remap_uv_box(delt_prim, 0.78, 0.25, 0.96, 0.80)
        add(f"deltoid.{side}", "top" if sleeved_top else "skin", delt_prim,
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
            a = math.cos(th)               # +1 at the front (th=0), -1 at the back (th=pi)
            front = {"buzz": 0.78, "short": 0.82, "bob": 0.80, "long": 0.80, "ponytail": 0.82,
                     "spiky": 0.82, "curly": 0.80, "mohawk": 0.52, "topknot": 0.82}[hair]
            back = {"buzz": 2.10, "short": 2.18, "bob": 2.32, "long": 2.38, "ponytail": 2.18,
                    "spiky": 2.08, "curly": 2.20, "mohawk": 2.00, "topknot": 2.15}[hair]
            front += 0.16 * bangs if hair in ("short", "bob", "long", "ponytail", "curly", "spiky") else 0.0
            k = (1.0 - a) * 0.5                       # 0 front (th=0), 1 back (th=pi)
            b = front + (back - front) * (k ** 0.80)
            return min(math.pi * 0.92, b)

        scale = 1.03 + 0.055 * vol
        add("hair_cap", "hair",
            cap_shell(surf, boundary, nu=max(16, hsegs), nv=10, scale=scale, rim=0.035),
            ["Head", "HairBack"], power=2.2, bias={"Head": 2.0}, outward=False)

        # deterministic seed for strand / volume add-ons
        seed_val = sum((i + 1) * ord(c) for i, c in enumerate(hair)) + int(round(vol * 1000))
        rng = np.random.default_rng(seed_val)

        def strand(center, length, dirv, w=0.34, mat="hair", bones=None, taper=0.06):
            d = _norm(np.asarray(dirv, dtype=np.float64))
            p0 = np.asarray(center, dtype=np.float64)
            p1 = p0 + d * (length * 0.55) + np.array([0, length * 0.1, 0])
            p2 = p0 + d * length + np.array([0, -length * 0.25, 0])
            r0 = hr[0] * w * 0.42
            add("strand", mat, tube([p0, p1, p2], [r0 * 1.1, r0 * 0.85, r0 * taper], segs=10,
                                    cap_start="flat", cap_end="round"),
                bones or ["Head", "HairBack"], power=2.4, outward=True)

        if hair == "spiky":
            n = 13
            for i in range(n):
                a = TAU * i / n + 0.3
                base = surf.point(a, 0.52 + 0.22 * ((i % 3) / 2.0))
                out = _norm((base - hc) * np.array([1.0, 0.75, 1.0]))
                strand(base, hr[1] * (0.72 + 0.38 * rng.random()) * vol, out, w=0.50)
        elif hair == "curly":
            n = 28
            for i in range(n):
                a = (TAU * (i * 0.618034)) % TAU
                u = rng.random()
                max_p = boundary(a) * 0.95
                phi = 0.20 + (max_p - 0.20) * u
                base = surf.point(a, phi, hr[0] * 0.04 * vol)
                rc = hr[0] * (0.24 + 0.08 * (i % 3) / 2.0) * (0.85 + 0.25 * vol)
                add("curl", "hair", blob(base, (rc, rc, rc), p=2.0, segs=10, rings=8),
                    ["Head", "HairBack"], power=2.4)
        elif hair == "mohawk":
            n = 7
            for i in range(n):
                t = i / (n - 1)                # 0 front -> 1 back along crown
                th_m = 0.0 if t < 0.45 else math.pi
                phi_m = abs(t - 0.45) * 1.65
                base = surf.point(th_m, phi_m)
                out_d = _norm(surf.normal(th_m, phi_m) + np.array([0.0, 0.65, 0.0]))
                h = hr[1] * (0.68 - 0.20 * abs(t - 0.4)) * vol
                add("mohawk", "hair", tube([base - out_d * (hr[0] * 0.05), base + out_d * h],
                                           [hr[0] * 0.16, hr[0] * 0.08], segs=10, cap_start="flat", cap_end="round"),
                    ["Head", "HairBack"], power=2.4)
        elif hair == "topknot":
            base = surf.point(math.pi, 0.22, hr[1] * 0.24 * vol)
            add("knot", "hair", blob(base, (hr[0] * 0.36 * vol, hr[1] * 0.30 * vol, hr[0] * 0.36 * vol),
                                     p=2.0, segs=16, rings=12), ["Head"], power=2.4)
        elif hair in ("bob", "long"):
            # Open curtain wrapping around the sides and back of the head (leaving the face open)
            nu_s = max(18, hsegs)
            nv_s = 9
            th0, th1 = 0.34 * math.pi, 1.66 * math.pi
            top_y = hc[1] + hr[1] * 0.28
            skirt_bottom = (m.neck_base_y - m.torso_h * 0.22) if hair == "long" else (hc[1] - hr[1] * 0.88)
            sv, suv, sf = [], [], []
            for i in range(nv_s + 1):
                t = i / nv_s
                y = top_y + (skirt_bottom - top_y) * t
                flare = (1.03 + 0.05 * vol) + (0.14 if hair == "long" else 0.05) * smoothstep(t * 1.1)
                shrink = 1.0 - (0.06 if hair == "long" else 0.12) * t * t
                rx_s = hr[0] * flare * shrink
                rz_s = hr[2] * flare * shrink
                for j in range(nu_s + 1):
                    u_j = j / nu_s
                    th = th0 + (th1 - th0) * u_j
                    # taper inward slightly at the front temple edges
                    edge = math.sin(math.pi * u_j) ** 0.35
                    r_mult = 0.94 + 0.06 * edge
                    sv.append((hc[0] + rx_s * r_mult * math.sin(th), y, hc[2] + rz_s * r_mult * math.cos(th)))
                    suv.append((u_j, 1.0 - t))
            stride_s = nu_s + 1
            for i in range(nv_s):
                for j in range(nu_s):
                    a_ = i * stride_s + j
                    b_ = a_ + 1
                    c_ = a_ + stride_s + 1
                    d_ = a_ + stride_s
                    # Both outer and inner faces so back/side hair is visible from front and back
                    sf.append((a_, c_, b_))
                    sf.append((a_, d_, c_))
                    sf.append((a_, b_, c_))
                    sf.append((a_, c_, d_))
            add("hair_skirt", "hair", (np.array(sv), sf, np.array(suv)),
                ["Head", "HairBack", "Neck"], power=2.4, outward=False)
        elif hair == "ponytail":
            b = surf.point(math.pi, 1.45, hr[2] * 0.04)
            mid = b + np.array([0.0, -hr[1] * 0.35, -hr[2] * 0.45])
            tip = mid + np.array([0.0, -hr[1] * 0.78, -hr[2] * 0.10])
            add("ponytail", "hair",
                tube([b, mid, tip], [hr[0] * 0.30 * vol, hr[0] * 0.26 * vol, hr[0] * 0.09], segs=14,
                     cap_start="round", cap_end="round"),
                ["Head", "HairBack", "HairTail"], power=2.4)
        elif hair == "buzz":
            pass

        # bangs: subtle fringe strands emerging from the front hairline of hair_cap
        if bangs > 0.02 and hair not in ("buzz", "mohawk"):
            n = 7
            cap_lift = hr[0] * (scale - 1.0)
            for i in range(n):
                t = (i / (n - 1)) * 2 - 1
                a = t * 0.58
                b_phi = boundary(a)
                phi0 = max(0.38, b_phi - 0.22)
                phi1 = b_phi + (0.06 + 0.08 * bangs) * (1.0 - 0.35 * abs(t))
                phi_mid = 0.5 * (phi0 + phi1)
                p0 = surf.point(a * 0.90, phi0, cap_lift * 0.65)
                p1 = surf.point(a * 0.96, phi_mid, cap_lift * 1.02)
                p2 = surf.point(a, phi1, cap_lift * 0.45)
                r = hr[0] * (0.14 - 0.025 * abs(t))
                bprim = tube([p0, p1, p2], [(r * 0.85, r * 0.32), (r * 0.95, r * 0.36), (r * 0.35, r * 0.16)],
                             segs=10, cap_start="round", cap_end="round")
                add("bang", "hair", remap_uv_box(bprim, 0.15, 0.08, 0.85, 0.55),
                    ["Head", "HairBack"], power=2.4, outward=True)

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
        if collar == "crew" and top_style != "tank":
            cy = profile[-1][0]
            rneck = m.r_neck * 1.22
            cp = [(cy - m.torso_h * 0.015, rneck, rneck),
                  (cy + m.torso_h * 0.030, rneck * 1.10, rneck * 1.10)]
            cprim = remap_uv_box(lathe(cp, max(12, segs // 2), close_bottom=False, close_top=False),
                                 0.02, 0.93, 0.98, 0.99)
            add("collar", "top", cprim,
                ["Chest", "Neck"], power=3.4, bias={"Neck": 1.6}, outward=False)
        hood = None
        if top_style == "hoodie" or collar == "hood":
            hy = m.shoulder_y
            hp = np.array([0.0, hy + m.torso_h * 0.06, -m.r_chest * 0.62])
            hood = remap_uv_box(blob(hp, (m.r_chest * 0.78, m.torso_h * 0.34, m.r_chest * 0.62), p=2.4, segs=18, rings=13),
                                0.76, 0.15, 0.98, 0.85)
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
                sprim = remap_uv_box(tube([a, a * 0.5 + b * 0.5 + np.array([s * 0.01, m.torso_h * 0.06, 0]), b],
                                          [m.r_upper_arm * 0.55, m.r_upper_arm * 0.6, m.r_upper_arm * 0.55],
                                          segs=10, cap_start="flat", cap_end="flat"),
                                     0.78, 0.20, 0.96, 0.80)
                add(f"strap.{side}", "top", sprim,
                    ["Chest", f"Shoulder.{side}"], power=2.8)
                continue
            if sleeve == "none":
                continue
            path = [root, root * 0.5 + el * 0.5, el]
            rads = [m.r_upper_arm * 1.18, m.r_upper_arm * 1.12, m.r_upper_arm * 1.05]
            if sleeve == "long":
                path = [root, root * 0.5 + el * 0.5, el, el * 0.5 + wr * 0.5, wr]
                rads = [m.r_upper_arm * 1.18, m.r_upper_arm * 1.12, m.r_upper_arm * 1.03, m.r_forearm * 1.10, m.r_forearm * 1.04]
            sl_prim = remap_uv_box(tube(path, rads, segs=max(10, segs - 4), cap_start="round", cap_end="flat"),
                                   0.76, 0.15, 0.98, 0.85)
            add(f"sleeve.{side}", "top", sl_prim,
                [f"Shoulder.{side}", f"UpperArm.{side}", f"LowerArm.{side}"], power=2.9)

        if top_style == "armor":
            for side, s in (("L", 1.0), ("R", -1.0)):
                root = np.array([s * m.arm_root[0], m.arm_root[1], m.arm_root[2]])
                pprim = remap_uv_box(blob(root + np.array([0, m.r_upper_arm * 0.10, 0]),
                                          (m.r_upper_arm * 1.42, m.r_upper_arm * 1.05, m.r_upper_arm * 1.40),
                                          p=2.6, segs=16, rings=11),
                                     0.76, 0.10, 0.98, 0.95)
                add(f"pad.{side}", "top", pprim,
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
            steps = 12
            prof = []
            w_rx, w_rz = torso_radii_at(m, waist_y)
            hip_t = (waist_y - m.hip_y) / max(1e-6, waist_y - hem_y)
            # lathe expects bottom -> top
            for i in range(steps):
                t = 1.0 - i / (steps - 1)             # 1 at hem (bottom) -> 0 at waist (top)
                y = waist_y - (waist_y - hem_y) * t
                tx, tz = torso_radii_at(m, y)
                lx, lz = lower_body_radii(m, min(y, m.hip_y), 1.12)
                k = smoothstep(min(1.0, t / max(0.08, hip_t * 0.80)))
                flare = 1.0 + 0.36 * (max(0.0, t - hip_t * 0.4) ** 1.3)
                rx = (w_rx * 1.06) * (1.0 - k) + max(tx, lx) * flare * k
                rz = (w_rz * 1.06) * (1.0 - k) + max(tz, lz) * flare * k
                prof.append((y, rx, rz))
            add("skirt", "bottom", lathe(prof, segs, close_bottom=True, close_top=True),
                ["Hips", "Spine", "Thigh.L", "Thigh.R"], power=2.6)
        else:
            # hip block: A-line from the waist down over the crotch, then two legs
            leg_span = max(1e-6, m.hip_y - m.ankle_y)
            crotch_y = m.hip_y - leg_span * 0.22
            top_rx, top_rz = torso_radii_at(m, waist_y)
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
        eth_l, eph_l, _, _, _, _ = fl["eye.L"]
        epos_l = surf.point(eth_l, eph_l)
        ez = float(epos_l[2]) + m.eye_r * 0.32
        for side, s in (("L", 1.0), ("R", -1.0)):
            ex = s * m.eye_x
            if gl in ("round", "square"):
                Rl = m.eye_r * (1.32 if gl == "round" else 1.28)
                rr = m.eye_r * (0.12 if gl == "round" else 0.11)
                R_tor = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
                if gl == "round":
                    v, f, uv = torus(np.array([ex, m.eye_y, ez]), Rl, rr, 20, 8, R_mat=R_tor)
                else:
                    v, f, uv = torus(np.array([ex, m.eye_y, ez]), Rl, rr, 4, 8,
                                     R_mat=R_tor @ np.array([[0.7071, 0, -0.7071], [0, 1, 0], [0.7071, 0, 0.7071]]))
                add(f"glasses_frame.{side}", "glasses", (v, f, uv), ["Head"], power=2.0)
            else:  # shades
                add(f"shades.{side}", "glasses",
                    blob(np.array([ex, m.eye_y, ez]), (m.eye_r * 1.40, m.eye_r * 1.02, m.eye_r * 0.16),
                         p=2.8, segs=16, rings=10), ["Head"], power=2.0)
            # temple arm
            a = np.array([s * (m.eye_x + m.eye_r * 1.32), m.eye_y + m.eye_r * 0.15, ez])
            b = surf.point(s * 1.35, eph_l, m.eye_r * 0.05)
            add(f"temple.{side}", "glasses", tube([a, a * 0.5 + b * 0.5, b],
                                                  [m.eye_r * 0.12, m.eye_r * 0.11, m.eye_r * 0.09], segs=8,
                                                  cap_start="round", cap_end="flat"),
                ["Head"], power=2.0)
        add("bridge", "glasses",
            blob(np.array([0.0, m.eye_y + m.eye_r * 0.14, ez + m.eye_r * 0.04]),
                 (max(m.eye_r * 0.25, m.eye_x - m.eye_r * 1.15), m.eye_r * 0.12, m.eye_r * 0.12), p=2.0, segs=12, rings=8),
            ["Head"], power=2.0)

    # --- hat --------------------------------------------------------------
    hat = spec["hat"]
    if hat != "none":
        top_y = hc[1] + hr[1]
        if hat == "cap":
            dome = blob(np.array([0.0, hc[1] + hr[1] * 0.44, 0.0]), (hr[0] * 1.12, hr[1] * 0.62, hr[2] * 1.12),
                        p=2.0, segs=max(16, hsegs), rings=12)
            add("cap", "hat", dome, ["Head"], power=2.0, conform=True)
            add("visor", "hat", blob(np.array([0.0, hc[1] + hr[1] * 0.28, hc[2] + hr[2] * 0.96]),
                                     (hr[0] * 0.88, hr[1] * 0.07, hr[2] * 0.52), p=2.6, segs=18, rings=9),
                ["Head"], power=2.0)
        elif hat == "beanie":
            add("beanie", "hat", blob(np.array([0.0, hc[1] + hr[1] * 0.50, 0.0]),
                                      (hr[0] * 1.14, hr[1] * 0.68, hr[2] * 1.14), p=2.0, segs=max(16, hsegs), rings=13),
                ["Head"], power=2.0, conform=True)
            add("brim", "hat", torus(np.array([0.0, hc[1] + hr[1] * 0.24, hc[2]]), hr[0] * 1.06, hr[0] * 0.10,
                                     24, 8), ["Head"], power=2.0)
            add("pom", "hat", blob(np.array([0.0, top_y + hr[1] * 0.14, hc[2]]),
                                   (hr[0] * 0.24, hr[0] * 0.24, hr[0] * 0.24), p=2.0, segs=14, rings=10),
                ["Head"], power=2.0)
        elif hat == "sunhat":
            bv, bf, buv = lathe([(hc[1] + hr[1] * 0.66, hr[0] * 1.02, hr[2] * 1.02),
                                 (hc[1] + hr[1] * 0.61, hr[0] * 1.65, hr[2] * 1.65),
                                 (hc[1] + hr[1] * 0.65, hr[0] * 1.65, hr[2] * 1.65),
                                 (hc[1] + hr[1] * 0.70, hr[0] * 1.02, hr[2] * 1.02)],
                                max(20, hsegs), close_bottom=False, close_top=False)
            bv[:, 2] += hc[2]
            add("brim", "hat", (bv, bf, buv), ["Head"], power=2.0, outward=False)
            add("crown", "hat", blob(np.array([0.0, hc[1] + hr[1] * 0.86, hc[2]]),
                                     (hr[0] * 1.02, hr[1] * 0.42, hr[2] * 1.02), p=2.2, segs=20, rings=12),
                ["Head"], power=2.0)
            add("band", "hat", torus(np.array([0.0, hc[1] + hr[1] * 0.70, hc[2]]), hr[0] * 1.03, hr[0] * 0.055, 22, 8),
                ["Head"], power=2.0)
        elif hat == "wizard":
            base_y = hc[1] + hr[1] * 0.56
            prof = []
            steps = 8
            for i in range(steps):
                t = i / (steps - 1)
                prof.append((base_y + hr[1] * 2.2 * t, hr[0] * (1.04 - 0.99 * t ** 1.25), hr[2] * (1.04 - 0.99 * t ** 1.25)))
            wv, wf, wuv = lathe(prof, max(18, hsegs), close_bottom=True, close_top=True)
            wv[:, 2] += hc[2]
            add("wizcone", "hat", (wv, wf, wuv), ["Head"], power=2.0)
            wbv, wbf, wbuv = lathe([(base_y + hr[1] * 0.04, hr[0] * 1.02, hr[2] * 1.02),
                                    (base_y - hr[1] * 0.03, hr[0] * 1.75, hr[2] * 1.75),
                                    (base_y + hr[1] * 0.01, hr[0] * 1.75, hr[2] * 1.75),
                                    (base_y + hr[1] * 0.08, hr[0] * 1.02, hr[2] * 1.02)],
                                   max(20, hsegs), close_bottom=False, close_top=False)
            wbv[:, 2] += hc[2]
            add("wizbrim", "hat", (wbv, wbf, wbuv), ["Head"], power=2.0, outward=False)

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
        nrm = part.normals if part.normals is not None else compute_normals(part.verts, part.faces)
        all_norms.append(nrm.astype(np.float32))
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
