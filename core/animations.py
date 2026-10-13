"""
Colada Character Maker - Animations
===================================
Poses are authored in WORLD axes (x = pitch/forward-back, y = swing/twist,
z = raise/spread) which is far easier to reason about than bone local space.
They are converted to bone local quaternions here, using each bone's rest
matrix, so the same numbers work for three.js and for Blender.

Sign conventions (character faces +Z, right handed, Y up):
  arm  : raise -> +z on the left, -z on the right  (handled by the helpers)
         forward/back swing -> y
         elbow bend -> -x (forearm comes forward)
  leg  : forward swing -> -x, knee bend -> +x, out to the side -> +-z
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .skeleton import Rig

TAU = math.pi * 2.0
D = math.radians


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def euler_q(x=0.0, y=0.0, z=0.0) -> np.ndarray:
    return rot_z(z) @ rot_y(y) @ rot_x(x)


def smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def lerp(a, b, t):
    return a + (b - a) * t


# ---------------------------------------------------------------------------

MIRROR = {"L": 1.0, "R": -1.0}


@dataclass
class Pose:
    """A single frame: bone -> world-axis euler triple (radians) + offset."""
    rot: dict = field(default_factory=dict)
    off: dict = field(default_factory=dict)

    def s(self, bone: str, x=0.0, y=0.0, z=0.0):
        """Set absolute world-axis rotation for a bone."""
        self.rot[bone] = (x, y, z)

    def add(self, bone: str, x=0.0, y=0.0, z=0.0):
        px, py, pz = self.rot.get(bone, (0.0, 0.0, 0.0))
        self.rot[bone] = (px + x, py + y, pz + z)

    def move(self, bone: str, v):
        self.off[bone] = self.off.get(bone, np.zeros(3)) + np.asarray(v, dtype=np.float64)

    # --- semantic helpers -------------------------------------------------
    def arm(self, side: str, raise_=0.0, fwd=0.0, twist=0.0):
        m = MIRROR[side]
        self.s(f"UpperArm.{side}", x=twist, y=-fwd * m, z=raise_ * m)

    def elbow(self, side: str, bend=0.0, twist=0.0):
        self.s(f"LowerArm.{side}", x=-bend, y=0.0, z=twist * MIRROR[side])

    def hand(self, side: str, x=0.0, y=0.0, z=0.0):
        m = MIRROR[side]
        self.s(f"Hand.{side}", x=x, y=y * m, z=z * m)

    def shoulder(self, side: str, up=0.0, fwd=0.0):
        m = MIRROR[side]
        self.s(f"Shoulder.{side}", z=up * m, y=-fwd * m)

    def leg(self, side: str, fwd=0.0, spread=0.0, twist=0.0):
        m = MIRROR[side]
        self.s(f"Thigh.{side}", x=-fwd, y=twist * m, z=spread * m)

    def knee(self, side: str, bend=0.0):
        self.s(f"Shin.{side}", x=bend)

    def foot(self, side: str, pitch=0.0, roll=0.0):
        self.s(f"Foot.{side}", x=pitch, z=roll * MIRROR[side])

    def spine(self, pitch=0.0, roll=0.0, yaw=0.0, weight=1.0):
        """Distribute a spine curve over Hips / Spine / Chest."""
        parts = (("Hips", 0.34), ("Spine", 0.33), ("Chest", 0.33))
        for name, w in parts:
            self.s(name, x=pitch * w * weight, y=roll * w * weight, z=yaw * w * weight)

    def head(self, pitch=0.0, roll=0.0, yaw=0.0, neck=0.4):
        self.s("Neck", x=pitch * neck, y=roll * neck, z=yaw * neck)
        self.s("Head", x=pitch * (1 - neck), y=roll * (1 - neck), z=yaw * (1 - neck))


# ---------------------------------------------------------------------------
# clips - every function returns a Pose for normalised (u) or linear (t) time
# ---------------------------------------------------------------------------

def clip_tpose(u: float, pose: Pose, d):
    for s in ("L", "R"):
        pose.arm(s, raise_=D(53.0), fwd=0.0)
    pose.s("LowerArm.L", x=-D(3.0))
    pose.s("LowerArm.R", x=-D(3.0))


def clip_idle(u: float, pose: Pose, d):
    a = TAU * u
    breath = math.sin(a)
    sway = math.sin(a * 0.5)
    pose.move("Root", (0, 0.0045 * breath - 0.002, 0))
    pose.spine(pitch=D(2.5) + D(1.2) * breath, roll=D(1.4) * sway, yaw=D(2.0) * math.sin(a * 0.5 + 0.7))
    pose.head(pitch=D(-1.5) - D(1.0) * breath, yaw=D(3.0) * math.sin(a * 0.5 + 1.2), roll=D(2.0) * sway)
    pose.arm("L", raise_=D(-3.0) + D(2.2) * breath, fwd=D(2.0) * math.sin(a + 0.4))
    pose.arm("R", raise_=D(-3.0) + D(2.2) * math.sin(a + 0.6), fwd=D(2.0) * math.sin(a + 0.9))
    pose.elbow("L", bend=D(9.0) + D(3.0) * breath)
    pose.elbow("R", bend=D(9.0) + D(3.0) * math.sin(a + 0.6))
    pose.hand("L", z=D(4.0))
    pose.hand("R", z=D(4.0))
    pose.leg("L", fwd=D(0), spread=D(1.5))
    pose.leg("R", fwd=D(0), spread=D(1.5))
    pose.knee("L", bend=D(4.0) + D(1.5) * breath)
    pose.knee("R", bend=D(4.0) + D(1.5) * math.sin(a + 0.5))
    pose.s("HairBack", x=D(3.0) * math.sin(a + 0.3))
    pose.s("Tail1", x=D(6.0) * math.sin(a * 2.0))


def _step(pose: Pose, phase: float, cfg):
    """Shared walk/run machinery. phase in [0,1) for one full cycle (2 steps)."""
    a = TAU * phase
    swing = cfg["swing"]
    lift = cfg["lift"]
    for side, ph in (("L", 0.0), ("R", 0.5)):
        p = a + TAU * ph
        fwd = math.cos(p) * swing
        # knees bend during the swing phase, straighten on contact
        bend = cfg["knee_base"] + max(0.0, math.sin(p + cfg["knee_phase"])) ** 1.4 * lift
        pose.leg(side, fwd=fwd, spread=D(cfg["spread"]))
        pose.knee(side, bend=bend)
        pose.foot(side, pitch=cfg["ankle"] * math.sin(p + 1.1) + cfg.get("ankle_base", 0.0),
                  roll=D(1.0))
        # arms counter-swing
        arm_fwd = -math.cos(p) * cfg["arm_swing"]
        pose.arm(side, raise_=D(cfg["arm_out"]), fwd=arm_fwd)
        pose.elbow(side, bend=cfg["elbow"] + cfg["elbow_swing"] * max(0.0, math.cos(p)))
    bounce = -abs(math.sin(a)) * cfg["bounce"] + cfg["bounce"] * 0.35
    pose.move("Root", (0, bounce, 0))
    pose.spine(pitch=cfg["lean"], roll=D(cfg["roll"]) * math.sin(a), yaw=D(cfg["yaw"]) * math.sin(a))
    pose.head(pitch=-cfg["lean"] * 0.55, yaw=D(cfg["yaw"]) * 0.35 * math.sin(a + 0.6))
    pose.shoulder("L", up=D(1.5) * math.sin(a))
    pose.shoulder("R", up=D(1.5) * math.sin(a + math.pi))
    pose.s("HairBack", x=D(6.0) * math.sin(a + 0.8))


def clip_walk(u: float, pose: Pose, d):
    _step(pose, u, dict(
        swing=D(26.0), lift=D(48.0), knee_base=D(5.0), knee_phase=0.4, spread=1.5,
        ankle=D(12.0), ankle_base=D(2.0), arm_swing=D(17.0), arm_out=-4.0,
        elbow=D(12.0), elbow_swing=D(10.0), bounce=0.028, lean=D(3.0), roll=2.5, yaw=3.5))


def clip_run(u: float, pose: Pose, d):
    _step(pose, u, dict(
        swing=D(46.0), lift=D(92.0), knee_base=D(14.0), knee_phase=0.35, spread=2.0,
        ankle=D(16.0), ankle_base=D(-4.0), arm_swing=D(40.0), arm_out=-6.0,
        elbow=D(64.0), elbow_swing=D(26.0), bounce=0.055, lean=D(11.0), roll=2.0, yaw=5.0))


def clip_jump(u: float, pose: Pose, d):
    t = u  # 0..1 across the clip
    crouch = smoothstep((t - 0.02) / 0.16) * (1 - smoothstep((t - 0.22) / 0.10))
    launch = smoothstep((t - 0.26) / 0.10) * (1 - smoothstep((t - 0.36) / 0.08))
    air = smoothstep((t - 0.32) / 0.08) * (1 - smoothstep((t - 0.74) / 0.12))
    land = smoothstep((t - 0.76) / 0.10) * (1 - smoothstep((t - 0.90) / 0.10))
    knee = D(70) * crouch + D(52) * land + D(30) * air + D(6)
    hips = -0.19 * crouch - 0.13 * land + 0.22 * launch
    airtime = max(0.0, math.sin(math.pi * max(0.0, min(1.0, (t - 0.30) / 0.52)))) * 0.42
    pose.move("Root", (0, hips + airtime, 0))
    pose.spine(pitch=D(14) * crouch + D(-4) * air + D(10) * land, roll=0, yaw=0)
    pose.head(pitch=D(-8) * crouch + D(6) * air)
    for side in ("L", "R"):
        pose.leg(side, fwd=D(-26) * crouch + D(16) * air + D(-8) * land, spread=D(4) * crouch)
        pose.knee(side, bend=knee)
        pose.foot(side, pitch=D(20) * crouch - D(24) * air + D(16) * land)
        pose.arm(side, raise_=D(16) * crouch + D(58) * air + D(10) * land,
                 fwd=D(-30) * crouch + D(20) * air)
        pose.elbow(side, bend=D(34) * crouch + D(18) * air + D(46) * land)
    pose.knee("R", bend=knee * 0.94)


def clip_wave(u: float, pose: Pose, d):
    """2.2s: raise the right arm and wave, weight on one leg."""
    raise_up = smoothstep(u / 0.18)
    down = 1.0 - smoothstep((u - 0.86) / 0.14)
    k = min(raise_up, down)
    wave = math.sin(TAU * 3.0 * max(0.0, u - 0.16) / 0.7)
    pose.arm("R", raise_=lerp(D(-2), D(80), k), fwd=lerp(D(2), D(24), k))
    pose.elbow("R", bend=lerp(D(10), D(46), k))
    pose.hand("R", z=lerp(D(4), D(26) * wave, k), x=D(6) * wave * k)
    pose.shoulder("R", up=D(7) * k)
    pose.arm("L", raise_=D(-4), fwd=D(2))
    pose.elbow("L", bend=D(11))
    pose.spine(pitch=D(2.5), roll=D(2.5), yaw=D(-5) * k)
    pose.head(pitch=D(-3) - D(4) * k, roll=D(-4) * k, yaw=D(-9) * k)
    pose.leg("L", spread=D(2))
    pose.leg("R", spread=D(3))
    pose.knee("L", bend=D(5))
    pose.knee("R", bend=D(9) + D(4) * k)
    pose.move("Root", (0, -0.004 - 0.008 * k + 0.006 * abs(wave) * k, 0))
    pose.s("HairBack", x=D(4) + D(4) * math.sin(TAU * 2 * u))


def clip_dance(u: float, pose: Pose, d):
    a = TAU * u
    b = TAU * u * 2
    pose.move("Root", (0.02 * math.sin(a), -0.015 - 0.02 * abs(math.sin(b)), 0))
    pose.spine(pitch=D(4), roll=D(9) * math.sin(a), yaw=D(11) * math.sin(a + 0.5))
    pose.head(pitch=D(-4) + D(5) * math.sin(b), roll=D(6) * math.sin(a), yaw=D(13) * math.sin(a))
    for side, ph in (("L", 0.0), ("R", math.pi)):
        s = math.sin(a + ph)
        pose.arm(side, raise_=D(38) + D(46) * s, fwd=D(14) * math.cos(a + ph))
        pose.elbow(side, bend=D(56) + D(26) * s)
        pose.hand(side, z=D(16) * s)
        pose.knee(side, bend=D(16) + D(14) * abs(math.sin(b + ph * 0.5)))
        pose.leg(side, spread=D(5) + D(3) * math.sin(a + ph))
        pose.shoulder(side, up=D(6) * s)
    pose.s("Tail1", x=D(16) * math.sin(b))
    pose.s("Tail2", x=D(20) * math.sin(b - 0.7))
    pose.s("HairBack", x=D(8) * math.sin(b))


def clip_kick(u: float, pose: Pose, d):
    wind = smoothstep(u / 0.30) * (1 - smoothstep((u - 0.30) / 0.16))
    strike = smoothstep((u - 0.28) / 0.14) * (1 - smoothstep((u - 0.58) / 0.22))
    back = smoothstep((u - 0.66) / 0.28)
    swing = -D(28) * wind + D(96) * strike - D(20) * back
    pose.leg("L", fwd=swing, spread=D(6))
    pose.knee("L", bend=D(58) * wind + D(20) * strike + D(30) * back)
    pose.foot("L", pitch=D(24) * wind - D(26) * strike)
    pose.leg("R", fwd=D(-6) * wind, spread=D(2))
    pose.knee("R", bend=D(12) + D(16) * wind + D(8) * strike)
    pose.arm("L", raise_=D(24) * wind + D(46) * strike, fwd=D(-16) * wind + D(16) * strike)
    pose.arm("R", raise_=D(36) * wind + D(52) * strike, fwd=D(20) * wind - D(10) * strike)
    pose.elbow("L", bend=D(30) + D(20) * strike)
    pose.elbow("R", bend=D(46) + D(24) * strike)
    pose.spine(pitch=D(-4) * wind + D(12) * strike, yaw=D(-10) * wind + D(14) * strike,
               roll=D(-4) * strike)
    pose.head(pitch=D(4) * strike, yaw=D(6) * strike)
    pose.move("Root", (0, -0.02 * wind - 0.02 * strike + 0.03 * strike, 0))
    pose.s("Tail1", x=D(20) * strike)


def clip_cheer(u: float, pose: Pose, d):
    a = TAU * u
    hop = max(0.0, math.sin(TAU * u * 2.0))
    pose.move("Root", (0, 0.05 * hop - 0.01, 0))
    pose.spine(pitch=D(-6) - D(4) * hop, roll=D(4) * math.sin(a))
    pose.head(pitch=D(-10) - D(6) * hop, roll=D(5) * math.sin(a + 1.0))
    for side, ph in (("L", 0.0), ("R", math.pi)):
        s = 0.5 + 0.5 * math.sin(a * 2 + ph)
        pose.arm(side, raise_=D(78) + D(16) * s, fwd=D(10))
        pose.elbow(side, bend=D(28) + D(20) * s)
        pose.hand(side, z=D(20) * s)
        pose.knee(side, bend=D(8) + D(24) * hop)
        pose.leg(side, spread=D(4))
        pose.shoulder(side, up=D(10))
    pose.s("Tail1", x=D(12) * math.sin(a * 2))


def clip_zombie(u: float, pose: Pose, d):
    a = TAU * u
    drag = math.sin(a)
    for side, ph in (("L", 0.0), ("R", math.pi)):
        p = a + ph
        pose.leg(side, fwd=D(16) * math.cos(p), spread=D(3))
        pose.knee(side, bend=D(9) + D(14) * max(0.0, math.sin(p + 1.2)))
        pose.foot(side, pitch=D(-10))
        pose.arm(side, raise_=D(72) + D(4) * math.sin(p), fwd=D(6))
        pose.elbow(side, bend=D(16) + D(10) * math.sin(p + 0.5))
        pose.hand(side, x=D(18), z=D(-8))
    pose.spine(pitch=D(16), roll=D(5) * drag, yaw=D(6) * math.sin(a * 0.5))
    pose.head(pitch=D(-14) + D(4) * math.sin(a * 2), roll=D(13) * math.sin(a * 1.3 + 0.6))
    pose.move("Root", (0.03 * drag, -0.03 - 0.012 * abs(math.sin(a * 2)), 0))
    pose.s("HairBack", x=D(10) + D(6) * math.sin(a))
    pose.s("Tail1", x=D(10) * math.sin(a * 1.5))


CLIPS: dict[str, dict] = {
    "idle":   {"fn": clip_idle,   "duration": 4.0,  "loop": True,  "label": "Idle"},
    "walk":   {"fn": clip_walk,   "duration": 1.1,  "loop": True,  "label": "Walk"},
    "run":    {"fn": clip_run,    "duration": 0.72, "loop": True,  "label": "Run"},
    "jump":   {"fn": clip_jump,   "duration": 1.5,  "loop": False, "label": "Jump"},
    "wave":   {"fn": clip_wave,   "duration": 2.4,  "loop": False, "label": "Wave"},
    "dance":  {"fn": clip_dance,  "duration": 2.4,  "loop": True,  "label": "Dance"},
    "kick":   {"fn": clip_kick,   "duration": 1.3,  "loop": False, "label": "Kick"},
    "cheer":  {"fn": clip_cheer,  "duration": 2.0,  "loop": True,  "label": "Cheer"},
    "zombie": {"fn": clip_zombie, "duration": 2.4,  "loop": True,  "label": "Zombie"},
    "tpose":  {"fn": clip_tpose,  "duration": 0.4,  "loop": False, "label": "T-Pose"},
}

DEFAULT_CLIP_ORDER = ["idle", "walk", "run", "jump", "wave", "dance", "kick", "cheer", "zombie", "tpose"]


# ---------------------------------------------------------------------------
# sampling -> local TRS tracks
# ---------------------------------------------------------------------------

def _continuity(quats: np.ndarray) -> np.ndarray:
    """Keep quaternion samples in the same hemisphere for LINEAR interpolation."""
    q = quats.copy()
    for i in range(1, len(q)):
        d = np.sum(q[i] * q[i - 1], axis=-1)
        flip = d < 0
        q[i][flip] *= -1
    return q


def _mat_to_quat(m: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion (x, y, z, w)."""
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    n = np.linalg.norm(q)
    return q / n if n > 1e-9 else np.array([0.0, 0.0, 0.0, 1.0])


class Sampler:
    """Turns Pose functions into (translation, quaternion) tracks per bone."""

    def __init__(self, rig: Rig):
        self.rig = rig
        rest_T, rest_R = rig.local_transforms()
        self.rest_T, self.rest_R = rest_T, rest_R
        # parent rest rotations for the world -> local conversion
        self.parent_R = []
        for b in rig.bones:
            if b.parent is None:
                self.parent_R.append(np.eye(3))
            else:
                self.parent_R.append(rig.rest_world[rig.name_to_index[b.parent]][:3, :3])

    def pose_to_local(self, pose: Pose):
        """-> (T[n,3], R[n,3,3]) local transforms for every bone."""
        n = len(self.rig.bones)
        T = self.rest_T.copy()
        R = np.zeros((n, 3, 3))
        for i, b in enumerate(self.rig.bones):
            Ri = self.rig.rest_world[i][:3, :3]
            Rp = self.parent_R[i]
            q = euler_q(*pose.rot.get(b.name, (0.0, 0.0, 0.0)))
            R[i] = Rp.T @ q @ Ri
            if b.name in pose.off:
                off = pose.off[b.name]
                T[i] = T[i] + (Rp.T @ off if b.parent is not None else off)
        return T, R

    def sample(self, clip_name: str, fps: int = 30) -> dict:
        info = CLIPS[clip_name]
        dur = info["duration"]
        frames = max(2, int(round(dur * fps)) + (1 if not info["loop"] else 0))
        times, quats, trans = [], [], []
        for f in range(frames):
            if info["loop"]:
                u = f / (frames - 1) if frames > 1 else 0.0      # last frame == first
            else:
                u = f / (frames - 1)
            pose = Pose()
            info["fn"](u, pose, dur)
            T, R = self.pose_to_local(pose)
            quats.append(np.stack([_mat_to_quat(R[i]) for i in range(len(self.rig.bones))]))
            trans.append(T)
            times.append(u * dur)
        return {
            "name": clip_name,
            "duration": dur,
            "fps": fps,
            "times": np.array(times, dtype=np.float64),
            "quats": np.stack(quats),          # (frames, bones, 4)
            "trans": np.stack(trans),          # (frames, bones, 3)
            "loop": info["loop"],
        }
