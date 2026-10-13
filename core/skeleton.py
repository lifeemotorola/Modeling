"""
Colada Character Maker - Skeleton
=================================
Body metrics, bone hierarchy and the procedural "auto-skinning" maths that
turns a spec into a rig.

Conventions
-----------
* Y is up, character faces +Z (glTF friendly, meters).
* Every bone stores a rest world matrix.  Local (parent relative) transforms
  are derived from those, so the exported skin binds perfectly.
* Bone local axes: +Y points head->tail (Blender friendly).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field

import numpy as np

# ---------------------------------------------------------------------------


def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.array([0.0, 1.0, 0.0])


def bone_axes(head: np.ndarray, tail: np.ndarray) -> np.ndarray:
    """Rotation matrix (columns = x, y, z axes) for a bone pointing head->tail."""
    y = _norm(tail - head)
    ref = np.array([1.0, 0.0, 0.0]) if abs(y[0]) < 0.92 else np.array([0.0, 0.0, 1.0])
    z = _norm(np.cross(ref, y))
    x = _norm(np.cross(y, z))
    return np.stack([x, y, z], axis=1)


def rest_matrix(head: np.ndarray, tail: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = bone_axes(head, tail)
    m[:3, 3] = head
    return m


# ---------------------------------------------------------------------------
# Body metrics
# ---------------------------------------------------------------------------

@dataclass
class Metrics:
    """Every measurement the mesh builder needs, derived from a spec."""
    H: float = 1.7
    foot_h: float = 0.07
    ankle_y: float = 0.08
    knee_y: float = 0.47
    hip_y: float = 0.86
    torso_h: float = 0.5
    waist_y: float = 1.02
    chest_y: float = 1.2
    shoulder_y: float = 1.3
    neck_base_y: float = 1.36
    neck_top_y: float = 1.42
    head_base_y: float = 1.42
    head_top_y: float = 1.68
    head_c: np.ndarray = dc_field(default_factory=lambda: np.zeros(3))
    head_r: np.ndarray = dc_field(default_factory=lambda: np.ones(3))
    hip_x: float = 0.1
    shoulder_x: float = 0.18
    arm_root: np.ndarray = dc_field(default_factory=lambda: np.zeros(3))
    elbow: np.ndarray = dc_field(default_factory=lambda: np.zeros(3))
    wrist: np.ndarray = dc_field(default_factory=lambda: np.zeros(3))
    hand_tip: np.ndarray = dc_field(default_factory=lambda: np.zeros(3))
    knee: np.ndarray = dc_field(default_factory=lambda: np.zeros(3))
    ankle: np.ndarray = dc_field(default_factory=lambda: np.zeros(3))
    toe: np.ndarray = dc_field(default_factory=lambda: np.zeros(3))
    toe_tip: np.ndarray = dc_field(default_factory=lambda: np.zeros(3))
    # radii
    r_chest: float = 0.2
    r_waist: float = 0.16
    r_hip: float = 0.17
    r_neck: float = 0.05
    r_upper_arm: float = 0.05
    r_forearm: float = 0.04
    r_hand: float = 0.05
    r_thigh: float = 0.09
    r_calf: float = 0.07
    foot_len: float = 0.24
    foot_w: float = 0.09
    eye_r: float = 0.032
    eye_x: float = 0.058
    eye_y: float = 0.0   # filled relative to head centre
    eye_z: float = 0.0
    profile: list = dc_field(default_factory=list)  # torso lathe profile [(y, rx, rz)]


def build_metrics(spec: dict) -> Metrics:
    m = Metrics()
    H = float(spec["height"])
    g = lambda k: float(spec[k])  # noqa: E731

    head_s, torso_s, neck_s = g("head_scale"), g("torso_scale"), g("neck_scale")
    leg_s, arm_s = g("leg_length"), g("arm_length")
    arm_t, leg_t = g("arm_thick"), g("leg_thick")
    fat, mus = g("body_fat"), g("muscle")
    shoulder_s, hip_s = g("shoulder_scale"), g("hip_scale")
    chest, waist = g("chest"), g("waist")
    foot_s, hand_s = g("foot_size"), g("hand_size")

    m.H = H
    head_h = 0.152 * H * head_s
    neck_h = 0.032 * H * neck_s
    m.foot_h = 0.042 * H * foot_s
    m.ankle_y = 0.048 * H * foot_s

    # ground -> hip, then hip -> crown fills exactly `height`
    m.hip_y = 0.470 * H * leg_s
    m.knee_y = m.ankle_y + (m.hip_y - m.ankle_y) * 0.48
    torso_h = max(0.12 * H, H - head_h - neck_h - m.hip_y)
    m.torso_h = torso_h

    m.neck_base_y = m.hip_y + torso_h
    m.neck_top_y = m.neck_base_y + neck_h
    m.head_base_y = m.neck_top_y - neck_h * 0.35
    m.head_top_y = m.neck_top_y + head_h
    m.chest_y = m.hip_y + torso_h * 0.62
    m.waist_y = m.hip_y + torso_h * 0.34
    m.shoulder_y = m.hip_y + torso_h * 0.94

    # --- head -----------------------------------------------------------------
    # the skull ellipsoid fills exactly one head-height above the neck tip
    rx = head_h * 0.42
    ry = head_h * 0.50
    rz = head_h * 0.44
    m.head_r = np.array([rx, ry, rz])
    m.head_c = np.array([0.0, m.neck_top_y + head_h * 0.50, -0.02 * head_h])

    # --- widths (values tuned against real anthropometry for each height) -----
    m.hip_x = 0.055 * H * hip_s * (1.0 + 0.18 * fat)
    m.shoulder_x = 0.095 * H * shoulder_s * (1.0 + 0.10 * mus + 0.08 * fat)
    girth = 1.0 + 0.42 * fat - 0.06 * mus
    m.r_chest = 0.082 * H * shoulder_s * girth
    m.r_waist = m.r_chest * (1.0 - 0.26 * waist)
    m.r_hip = 0.080 * H * hip_s * girth * (1.0 + 0.10 * (1.0 - waist))
    m.r_neck = 0.030 * H * (0.85 + 0.3 * neck_s) * (1.0 + 0.45 * fat)
    m.r_upper_arm = 0.026 * H * arm_t * (1.0 + 0.30 * mus + 0.45 * fat)
    m.r_forearm = 0.022 * H * arm_t * (1.0 + 0.26 * mus + 0.42 * fat)
    m.r_hand = 0.030 * H * hand_s * (1.0 + 0.25 * fat)
    m.r_thigh = 0.048 * H * leg_t * (1.0 + 0.22 * mus + 0.55 * fat)
    m.r_calf = 0.036 * H * leg_t * (1.0 + 0.22 * mus + 0.55 * fat)

    # --- arms (A-pose) --------------------------------------------------------
    up_len = 0.175 * H * arm_s
    lo_len = 0.150 * H * arm_s
    hand_len = 0.055 * H * hand_s
    root = np.array([m.shoulder_x, m.shoulder_y, 0.0])
    a1 = math.radians(37.0 + 4.0 * fat)
    a2 = math.radians(31.0 + 4.0 * fat)
    elbow = root + up_len * np.array([math.sin(a1), -math.cos(a1), 0.10])
    wrist = elbow + lo_len * np.array([math.sin(a2), -math.cos(a2), 0.16])
    hand_tip = wrist + hand_len * np.array([math.sin(a2), -math.cos(a2), 0.18])
    m.arm_root, m.elbow, m.wrist, m.hand_tip = root, elbow, wrist, hand_tip

    # --- legs -----------------------------------------------------------------
    m.knee = np.array([m.hip_x * 0.92, m.knee_y, 0.004 * H])
    m.ankle = np.array([m.hip_x * 0.92, m.ankle_y, -0.008 * H])
    m.foot_len = 0.135 * H * foot_s
    m.foot_w = 0.062 * H * foot_s * (1.0 + 0.25 * fat)
    toe = np.array([m.ankle[0] * 1.06, m.foot_h * 0.55, -0.008 * H + m.foot_len * 0.42])
    m.toe = toe
    m.toe_tip = toe + np.array([0.0, -0.002 * H, m.foot_len * 0.55])

    # --- eyes -----------------------------------------------------------------
    m.eye_r = 0.020 * H * head_s * float(spec["eye_size"])
    m.eye_x = rx * 0.42 * float(spec["eye_spacing"])
    m.eye_y = m.head_c[1] + ry * 0.10
    m.eye_z = -rz * 0.03

    # --- torso lathe profile (y, rx, rz) -------------------------------------
    prof = []
    n = 9
    for i in range(n):
        t = i / (n - 1)
        y = m.hip_y - m.hip_y * 0.02 + t * (torso_h * 1.02)
        # blend hip -> waist -> chest -> shoulder
        if t < 0.3:
            k = t / 0.3
            r = m.r_hip * (1 - k) + m.r_waist * k
        elif t < 0.62:
            k = (t - 0.3) / 0.32
            r = m.r_waist * (1 - k) + m.r_chest * k
        else:
            k = (t - 0.62) / 0.38
            r = m.r_chest * (1 - k * 0.18)
        # ellipse: chest is deeper than wide? keep slightly oval
        rx_i = r
        rz_i = r * (0.62 + 0.22 * chest)
        prof.append((y, rx_i, rz_i))
    m.profile = prof
    return m


# ---------------------------------------------------------------------------
# Bones
# ---------------------------------------------------------------------------

@dataclass
class Bone:
    name: str
    parent: str | None
    head: np.ndarray
    tail: np.ndarray
    deform: bool = True
    group: str = "body"


def build_bones(spec: dict, m: Metrics) -> list[Bone]:
    bones: list[Bone] = []
    add = lambda *a, **k: bones.append(Bone(*a, **k))  # noqa: E731

    spine_top = np.array([0.0, m.hip_y + m.torso_h * 0.94, 0.0])
    hips_tail = np.array([0.0, m.hip_y + m.torso_h * 0.22, 0.0])
    spine_mid = np.array([0.0, m.hip_y + m.torso_h * 0.44, 0.0])
    chest_head = np.array([0.0, m.hip_y + m.torso_h * 0.58, 0.0])

    add("Root", None, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.1, 0.0]), deform=False, group="root")
    add("Hips", "Root", np.array([0.0, m.hip_y + m.torso_h * 0.10, 0.0]), hips_tail, group="spine")
    add("Spine", "Hips", hips_tail, spine_mid, group="spine")
    add("Chest", "Spine", spine_mid, spine_top, group="spine")
    add("Neck", "Chest", spine_top, np.array([0.0, m.neck_top_y, 0.0]), group="head")
    add("Head", "Neck", np.array([0.0, m.neck_top_y, 0.0]), np.array([0.0, m.head_top_y, 0.0]), group="head")

    for side, s in (("L", 1.0), ("R", -1.0)):
        sh_head = np.array([s * m.shoulder_x * 0.28, m.shoulder_y - m.torso_h * 0.02, 0.0])
        root = np.array([s * m.arm_root[0], m.arm_root[1], m.arm_root[2]])
        el = np.array([s * m.elbow[0], m.elbow[1], m.elbow[2]])
        wr = np.array([s * m.wrist[0], m.wrist[1], m.wrist[2]])
        ht = np.array([s * m.hand_tip[0], m.hand_tip[1], m.hand_tip[2]])
        add(f"Shoulder.{side}", "Chest", sh_head, root, group="arm")
        add(f"UpperArm.{side}", f"Shoulder.{side}", root, el, group="arm")
        add(f"LowerArm.{side}", f"UpperArm.{side}", el, wr, group="arm")
        add(f"Hand.{side}", f"LowerArm.{side}", wr, ht, group="hand")
        if spec.get("finger_bones"):
            d = _norm(ht - wr)
            for fi, spread in enumerate((-0.014, 0.0, 0.014)):
                fh = ht + np.array([s * spread, 0, -abs(spread) * 0.6])
                ft = fh + d * (m.r_hand * 2.1)
                add(f"Finger{fi + 1}.{side}", f"Hand.{side}", fh, ft, group="hand")

        hip = np.array([s * m.hip_x, m.hip_y, 0.0])
        kn = np.array([s * m.knee[0], m.knee[1], m.knee[2]])
        an = np.array([s * m.ankle[0], m.ankle[1], m.ankle[2]])
        toe = np.array([s * m.toe[0], m.toe[1], m.toe[2]])
        tip = np.array([s * m.toe_tip[0], m.toe_tip[1], m.toe_tip[2]])
        add(f"Thigh.{side}", "Hips", hip, kn, group="leg")
        add(f"Shin.{side}", f"Thigh.{side}", kn, an, group="leg")
        add(f"Foot.{side}", f"Shin.{side}", an, toe, group="foot")
        add(f"Toe.{side}", f"Foot.{side}", toe, tip, group="foot")

    # hair / tail secondary motion
    hb0 = np.array([0.0, m.head_top_y, -m.head_r[2] * 0.85])
    hb1 = hb0 + np.array([0.0, -m.head_r[1] * 0.55, -m.head_r[2] * 0.5])
    add("HairBack", "Head", hb0, hb1, group="hair")
    if spec.get("hair_style") == "ponytail":
        t0 = hb1 + np.array([0.0, -m.head_r[1] * 0.9, -m.head_r[2] * 0.6])
        add("HairTail", "HairBack", hb1, t0, group="hair")

    tail = spec.get("tail", "none")
    if tail != "none":
        base = np.array([0.0, m.hip_y + m.torso_h * 0.16, -m.r_hip * 0.85])
        L = m.r_hip * (2.6 if tail != "puff" else 1.2)
        seg = (np.array([0.0, 0.32, -0.95]), np.array([0.0, -0.22, -1.0]), np.array([0.0, -0.72, -0.6]), np.array([0.0, -0.86, 0.1]))
        prev = "Hips"
        p = base
        for i in range(3):
            d = _norm(seg[i])
            nxt = p + d * (L / 3.0)
            add(f"Tail{i + 1}", prev, p, nxt, group="tail")
            prev, p = f"Tail{i + 1}", nxt

    for side, s in (("L", 1.0), ("R", -1.0)):
        if spec.get("ears", "none") != "none":
            base = np.array([s * m.head_r[0] * 0.52, m.head_c[1] + m.head_r[1] * 0.42, -m.head_r[2] * 0.12])
            tip = base + np.array([s * m.head_r[0] * 0.3, m.head_r[1] * (0.75 if spec["ears"] != "fox" else 0.95), -m.head_r[2] * 0.12])
            add(f"Ear.{side}", "Head", base, tip, group="head")
    return bones


# ---------------------------------------------------------------------------
# Skinning
# ---------------------------------------------------------------------------

def segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(ab @ ab)
    t = 0.0 if denom < 1e-12 else float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


class Rig:
    """Bone list + rest matrices + fast skinning weight solver."""

    def __init__(self, spec: dict, metrics: Metrics):
        self.spec = spec
        self.metrics = metrics
        self.bones = build_bones(spec, metrics)
        self.name_to_index = {b.name: i for i, b in enumerate(self.bones)}
        self.rest_world = np.stack([rest_matrix(b.head, b.tail) for b in self.bones])
        self.deform_bones = [b for b in self.bones if b.deform]
        self.deform_index = {b.name: i for i, b in enumerate(self.deform_bones)}

    # -- local transforms ---------------------------------------------------
    def local_transforms(self) -> tuple[np.ndarray, np.ndarray]:
        """(translations, rotation matrices) for each bone relative to its parent."""
        T = np.zeros((len(self.bones), 3))
        R = np.zeros((len(self.bones), 3, 3))
        for i, b in enumerate(self.bones):
            if b.parent is None:
                T[i] = self.rest_world[i][:3, 3]
                R[i] = self.rest_world[i][:3, :3]
            else:
                p = self.name_to_index[b.parent]
                rel = np.linalg.inv(self.rest_world[p]) @ self.rest_world[i]
                T[i] = rel[:3, 3]
                R[i] = rel[:3, :3]
        return T, R

    def inverse_bind(self) -> np.ndarray:
        return np.stack([np.linalg.inv(self.rest_world[self.name_to_index[b.name]]) for b in self.deform_bones])

    # -- weights ------------------------------------------------------------
    def weights_for(self, points: np.ndarray, allowed: list[str], power: float = 3.2,
                    max_infl: int = 3, bias: dict[str, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Distance based skinning over an explicit whitelist of bones."""
        idx = [self.deform_index[n] for n in allowed if n in self.deform_index]
        names = [n for n in allowed if n in self.deform_index]
        heads = np.stack([self.deform_bones[i].head for i in idx])
        tails = np.stack([self.deform_bones[i].tail for i in idx])
        A, B = heads[None, :, :], tails[None, :, :]
        P = points[:, None, :]
        AB = B - A
        denom = np.sum(AB * AB, axis=-1)
        denom = np.where(denom < 1e-12, 1e-12, denom)
        t = np.clip(np.sum((P - A) * AB, axis=-1) / denom, 0.0, 1.0)
        closest = A + t[..., None] * AB
        d = np.linalg.norm(P - closest, axis=-1) + 1e-4
        w = 1.0 / np.power(d, power)
        if bias:
            for k, n in enumerate(names):
                w[:, k] *= float(bias.get(n, 1.0))
        # keep top-k
        if w.shape[1] > max_infl:
            order = np.argsort(-w, axis=1)
            keep = order[:, :max_infl]
            mask = np.zeros_like(w, dtype=bool)
            np.put_along_axis(mask, keep, True, axis=1)
            w = np.where(mask, w, 0.0)
        w = w / np.sum(w, axis=1, keepdims=True)
        # compact to 4 slots (glTF JOINTS_0/WEIGHTS_0), padding with the best bone
        n_infl = min(4, w.shape[1])
        idx_arr = np.asarray(idx, dtype=np.int32)[None, :].repeat(points.shape[0], 0)
        top = np.argsort(-w, axis=1)[:, :n_infl]
        w4 = np.zeros((points.shape[0], 4))
        idx4 = np.zeros((points.shape[0], 4), dtype=np.int32)
        for c in range(4):
            src = min(c, n_infl - 1)
            w4[:, c] = np.take_along_axis(w, top[:, src: src + 1], axis=1)[:, 0]
            idx4[:, c] = np.take_along_axis(idx_arr, top[:, src: src + 1], axis=1)[:, 0]
        for c in range(n_infl, 4):
            w4[:, c] = 0.0
        s = w4.sum(axis=1, keepdims=True)
        s[s < 1e-9] = 1.0
        w4 = w4 / s
        return idx4, w4.astype(np.float32)
