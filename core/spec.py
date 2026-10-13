"""
Colada Character Maker - Character specification
=================================================
A single source of truth for every tunable feature of a character.

The same schema is served to the HTML front-end (`/api/schema`) so that the
Python engine and the web UI can never drift apart.  Add a field here and it
shows up in the app automatically.

Coordinate system: Y-up, meters, character faces +Z (glTF friendly).
"""
from __future__ import annotations

import copy
import random
from typing import Any

# ---------------------------------------------------------------------------
# Schema ---------------------------------------------------------------------
# Each field: key, label, type, default, plus type specific metadata.
#   type: range | color | select | bool | checkboxes | text
# ---------------------------------------------------------------------------

SCHEMA: list[dict[str, Any]] = [
    {
        "id": "body",
        "label": "Body",
        "icon": "🧍",
        "fields": [
            {"key": "height", "label": "Height", "type": "range", "min": 0.9, "max": 2.1, "step": 0.01, "default": 1.68, "unit": "m"},
            {"key": "head_scale", "label": "Head size", "type": "range", "min": 0.8, "max": 1.9, "step": 0.01, "default": 1.15},
            {"key": "body_fat", "label": "Chubbiness", "type": "range", "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.45},
            {"key": "muscle", "label": "Muscle", "type": "range", "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.5},
            {"key": "torso_scale", "label": "Torso length", "type": "range", "min": 0.8, "max": 1.3, "step": 0.01, "default": 1.0},
            {"key": "shoulder_scale", "label": "Shoulder width", "type": "range", "min": 0.7, "max": 1.5, "step": 0.01, "default": 1.0},
            {"key": "hip_scale", "label": "Hip width", "type": "range", "min": 0.7, "max": 1.5, "step": 0.01, "default": 1.0},
            {"key": "chest", "label": "Chest", "type": "range", "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.5},
            {"key": "waist", "label": "Waist taper", "type": "range", "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.5},
            {"key": "neck_scale", "label": "Neck", "type": "range", "min": 0.6, "max": 1.6, "step": 0.01, "default": 1.0},
            {"key": "arm_length", "label": "Arm length", "type": "range", "min": 0.75, "max": 1.3, "step": 0.01, "default": 1.0},
            {"key": "arm_thick", "label": "Arm thickness", "type": "range", "min": 0.7, "max": 1.6, "step": 0.01, "default": 1.0},
            {"key": "leg_length", "label": "Leg length", "type": "range", "min": 0.75, "max": 1.3, "step": 0.01, "default": 1.0},
            {"key": "leg_thick", "label": "Leg thickness", "type": "range", "min": 0.7, "max": 1.6, "step": 0.01, "default": 1.0},
            {"key": "hand_size", "label": "Hands", "type": "range", "min": 0.7, "max": 1.4, "step": 0.01, "default": 1.0},
            {"key": "foot_size", "label": "Feet", "type": "range", "min": 0.7, "max": 1.4, "step": 0.01, "default": 1.0},
        ],
    },
    {
        "id": "face",
        "label": "Face",
        "icon": "😊",
        "fields": [
            {"key": "skin_tone", "label": "Skin tone", "type": "color", "default": "#f0c69c"},
            {"key": "skin_detail", "label": "Skin shading", "type": "range", "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.5},
            {"key": "eye_style", "label": "Eye style", "type": "select", "options": ["round", "wide", "almond", "sleepy", "star"], "default": "round"},
            {"key": "eye_size", "label": "Eye size", "type": "range", "min": 0.6, "max": 1.5, "step": 0.01, "default": 1.05},
            {"key": "eye_spacing", "label": "Eye spacing", "type": "range", "min": 0.8, "max": 1.25, "step": 0.01, "default": 1.0},
            {"key": "eye_color", "label": "Eye color", "type": "color", "default": "#3f6fae"},
            {"key": "brow_style", "label": "Eyebrows", "type": "select", "options": ["soft", "thin", "bold", "angry", "raised", "none"], "default": "soft"},
            {"key": "brow_color", "label": "Brow color", "type": "color", "default": "#4a352a"},
            {"key": "mouth_style", "label": "Mouth", "type": "select", "options": ["smile", "grin", "smirk", "neutral", "open", "pout", "frown"], "default": "smile"},
            {"key": "expression", "label": "Expression macro", "type": "select", "options": ["custom", "happy", "neutral", "surprised", "angry", "sad", "sleepy", "smug"], "default": "custom", "help": "One click mood: drives mouth + brows + eyes. 'custom' keeps your own picks."},
            {"key": "blush", "label": "Blush", "type": "range", "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.25},
            {"key": "freckles", "label": "Freckles", "type": "range", "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.0},
            {"key": "nose_style", "label": "Nose", "type": "select", "options": ["button", "small", "snub", "wide", "none"], "default": "button"},
            {"key": "lashes", "label": "Eyelashes", "type": "bool", "default": True},
            {"key": "beard", "label": "Facial hair", "type": "select", "options": ["none", "stubble", "mustache", "goatee", "full"], "default": "none"},
            {"key": "beard_color", "label": "Facial hair color", "type": "color", "default": "#4a352a"},
        ],
    },
    {
        "id": "hair",
        "label": "Hair",
        "icon": "💇",
        "fields": [
            {"key": "hair_style", "label": "Style", "type": "select", "options": ["short", "buzz", "bob", "long", "ponytail", "spiky", "curly", "mohawk", "topknot", "bald"], "default": "bob"},
            {"key": "hair_color", "label": "Root color", "type": "color", "default": "#3b2a20"},
            {"key": "hair_tip_color", "label": "Tip color", "type": "color", "default": "#a8703f"},
            {"key": "hair_volume", "label": "Volume", "type": "range", "min": 0.5, "max": 1.6, "step": 0.01, "default": 1.0},
            {"key": "bangs", "label": "Bangs", "type": "range", "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.6},
            {"key": "hair_shine", "label": "Shine", "type": "range", "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.55},
        ],
    },
    {
        "id": "outfit",
        "label": "Outfit",
        "icon": "👕",
        "fields": [
            {"key": "top_style", "label": "Top", "type": "select", "options": ["tee", "tank", "tropical", "hoodie", "armor", "none"], "default": "tee"},
            {"key": "top_pattern", "label": "Top pattern", "type": "select", "options": ["pineapple", "solid", "stripes", "dots", "plaid", "stars", "tropical", "gradient", "camo"], "default": "pineapple"},
            {"key": "top_color", "label": "Top color", "type": "color", "default": "#20c4b5"},
            {"key": "top_color2", "label": "Top accent", "type": "color", "default": "#ffd166"},
            {"key": "sleeve", "label": "Sleeves", "type": "select", "options": ["short", "long", "none"], "default": "short"},
            {"key": "collar", "label": "Collar", "type": "select", "options": ["crew", "v", "hood", "none"], "default": "crew"},
            {"key": "bottom_style", "label": "Bottom", "type": "select", "options": ["shorts", "pants", "skirt", "capri", "none"], "default": "shorts"},
            {"key": "bottom_length", "label": "Bottom length", "type": "range", "min": 0.15, "max": 1.0, "step": 0.01, "default": 0.34, "help": "fraction of the way from hip to ankle"},
            {"key": "bottom_pattern", "label": "Bottom pattern", "type": "select", "options": ["solid", "stripes", "dots", "plaid", "stars", "camo", "gradient"], "default": "solid"},
            {"key": "bottom_color", "label": "Bottom color", "type": "color", "default": "#2b4a6f"},
            {"key": "bottom_color2", "label": "Bottom accent", "type": "color", "default": "#e9f1f7"},
            {"key": "belt", "label": "Belt", "type": "bool", "default": True},
            {"key": "shoes_style", "label": "Shoes", "type": "select", "options": ["sneakers", "boots", "sandals", "barefoot", "none"], "default": "sneakers"},
            {"key": "shoes_color", "label": "Shoe color", "type": "color", "default": "#e0503a"},
            {"key": "shoes_color2", "label": "Shoe accent", "type": "color", "default": "#20242c"},
            {"key": "socks", "label": "Socks", "type": "bool", "default": False},
        ],
    },
    {
        "id": "extras",
        "label": "Extras",
        "icon": "✨",
        "fields": [
            {"key": "ears", "label": "Ears", "type": "select", "options": ["none", "cat", "fox", "elf", "round"], "default": "none"},
            {"key": "tail", "label": "Tail", "type": "select", "options": ["none", "cat", "fox", "puff"], "default": "none"},
            {"key": "glasses", "label": "Glasses", "type": "select", "options": ["none", "round", "square", "shades"], "default": "none"},
            {"key": "hat", "label": "Hat", "type": "select", "options": ["none", "cap", "beanie", "sunhat", "wizard"], "default": "none"},
            {"key": "hat_color", "label": "Hat color", "type": "color", "default": "#7c4dff"},
            {"key": "hat_color2", "label": "Hat accent", "type": "color", "default": "#ffd166"},
        ],
    },
    {
        "id": "studio",
        "label": "Studio",
        "icon": "🎬",
        "fields": [
            {"key": "texture_size", "label": "Texture size", "type": "select", "options": ["512", "1024", "2048"], "default": "1024"},
            {"key": "antialias", "label": "Texture AA", "type": "range", "min": 1, "max": 3, "step": 1, "default": 2},
            {"key": "poly_detail", "label": "Mesh detail", "type": "range", "min": 0.6, "max": 1.6, "step": 0.05, "default": 1.0},
            {"key": "lighting", "label": "Lighting", "type": "select", "options": ["studio", "sunset", "neon", "night", "beach"], "default": "studio"},
            {"key": "ground", "label": "Show ground", "type": "bool", "default": True},
            {"key": "show_grid", "label": "Show grid", "type": "bool", "default": True},
            {"key": "show_skeleton", "label": "Show skeleton", "type": "bool", "default": False},
            {"key": "wireframe", "label": "Wireframe", "type": "bool", "default": False},
            {"key": "auto_rotate", "label": "Auto rotate", "type": "bool", "default": False},
            {"key": "head_track", "label": "Head follows camera", "type": "bool", "default": False},
        ],
    },
    {
        "id": "rig",
        "label": "Rig & Anim",
        "icon": "🦴",
        "fields": [
            {"key": "finger_bones", "label": "Finger bones", "type": "bool", "default": False},
            {"key": "ik_setup", "label": "IK targets (Blender)", "type": "bool", "default": True},
            {"key": "bone_size", "label": "Bone display size", "type": "range", "min": 0.2, "max": 2.0, "step": 0.05, "default": 1.0},
            {
                "key": "clips",
                "label": "BaKe animation clips",
                "type": "checkboxes",
                "options": ["idle", "walk", "run", "jump", "wave", "dance", "kick", "cheer", "zombie", "tpose"],
                "default": ["idle", "walk", "run", "jump", "wave", "dance", "kick", "cheer", "zombie", "tpose"],
            },
        ],
    },
]

# ---------------------------------------------------------------------------
# Derived helpers ------------------------------------------------------------
# ---------------------------------------------------------------------------

ALL_FIELDS: dict[str, dict[str, Any]] = {
    f["key"]: f for group in SCHEMA for f in group["fields"]
}

DEFAULT_SPEC: dict[str, Any] = {
    f["key"]: copy.deepcopy(f["default"]) for group in SCHEMA for f in group["fields"]
}


def defaults() -> dict[str, Any]:
    return copy.deepcopy(DEFAULT_SPEC)


def field(key: str) -> dict[str, Any]:
    return ALL_FIELDS[key]


def clamp_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Coerce an incoming spec to the schema: unknown keys dropped, values clamped."""
    out = defaults()
    for key, meta in ALL_FIELDS.items():
        if key not in spec:
            continue
        val = spec[key]
        t = meta["type"]
        try:
            if t == "range":
                val = float(val)
                val = max(float(meta["min"]), min(float(meta["max"]), val))
                if float(meta["step"]).is_integer() and float(meta["min"]).is_integer():
                    val = int(round(val))
            elif t == "bool":
                if isinstance(val, str):
                    val = val.strip().lower() in ("1", "true", "yes", "on")
                val = bool(val)
            elif t == "color":
                val = str(val)
                if not val.startswith("#") or len(val) not in (4, 7):
                    val = str(meta["default"])
            elif t in ("select", "text"):
                val = str(val)
                if t == "select" and val not in meta["options"]:
                    val = str(meta["default"])
            elif t == "checkboxes":
                if isinstance(val, str):
                    val = [v.strip() for v in val.split(",") if v.strip()]
                val = [str(v) for v in val if str(v) in meta["options"]]
        except (TypeError, ValueError):
            val = copy.deepcopy(meta["default"])
        out[key] = val
    return _apply_expression(out)


# expression macros drive mouth + brows (+ eye shape) in one go
EXPRESSIONS: dict[str, dict] = {
    "happy": {"mouth_style": "grin", "brow_style": "raised", "eye_style": "wide"},
    "neutral": {"mouth_style": "neutral", "brow_style": "soft"},
    "surprised": {"mouth_style": "open", "brow_style": "raised", "eye_style": "wide", "eye_size": 1.3},
    "angry": {"mouth_style": "frown", "brow_style": "angry", "eye_style": "almond"},
    "sad": {"mouth_style": "frown", "brow_style": "soft", "eye_style": "round", "blush": 0.1},
    "sleepy": {"mouth_style": "pout", "brow_style": "soft", "eye_style": "sleepy"},
    "smug": {"mouth_style": "smirk", "brow_style": "bold", "eye_style": "almond"},
}


def _apply_expression(sp: dict) -> dict:
    macro = EXPRESSIONS.get(sp.get("expression", "custom"))
    if not macro:
        return sp
    for k, v in macro.items():
        meta = ALL_FIELDS.get(k)
        if meta and meta["type"] in ("select", "range"):
            sp[k] = v
        elif meta and meta["type"] == "color":
            sp[k] = v
    return sp


# ---------------------------------------------------------------------------
# Presets --------------------------------------------------------------------
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict[str, Any]] = {
    "colada": {
        "label": "Piña Colada",
        "icon": "🍍",
        "spec": {},
    },
    "chibi": {
        "label": "Chibi Cutie",
        "icon": "🐣",
        "spec": {
            "height": 1.15, "head_scale": 1.85, "body_fat": 0.6, "leg_length": 0.78,
            "arm_length": 0.85, "hand_size": 1.2, "foot_size": 1.15, "neck_scale": 0.7,
            "eye_size": 1.45, "blush": 0.6, "hair_style": "bob", "hair_color": "#f2b134",
            "hair_tip_color": "#ffd98e", "hair_volume": 0.85, "bangs": 0.4, "top_pattern": "pineapple", "top_color": "#39d3c8",
            "bottom_style": "shorts", "bottom_length": 0.3, "shoes_color": "#ff8fab",
        },
    },
    "athlete": {
        "label": "Athlete",
        "icon": "🏃",
        "spec": {
            "height": 1.86, "muscle": 0.95, "body_fat": 0.18, "head_scale": 0.95,
            "shoulder_scale": 1.22, "chest": 0.8, "waist": 0.85, "arm_thick": 1.2,
            "leg_thick": 1.18, "skin_tone": "#c98a5b", "hair_style": "buzz",
            "hair_color": "#20160f", "top_style": "tank", "top_pattern": "gradient",
            "top_color": "#1f6feb", "top_color2": "#0b1f3a", "bottom_style": "shorts",
            "bottom_length": 0.32, "bottom_color": "#141821", "shoes_style": "sneakers",
            "shoes_color": "#f5f7fa", "shoes_color2": "#1f6feb", "socks": True,
            "belt": False, "mouth_style": "grin",
        },
    },
    "hero": {
        "label": "Tall Hero",
        "icon": "🦸",
        "spec": {
            "height": 2.0, "muscle": 0.85, "head_scale": 0.9, "shoulder_scale": 1.35,
            "torso_scale": 1.1, "leg_length": 1.15, "hair_style": "spiky",
            "hair_color": "#1b1f3a", "hair_tip_color": "#5b6bff", "eye_color": "#54d1ff",
            "top_style": "armor", "top_pattern": "gradient", "top_color": "#2b3ea8",
            "top_color2": "#ffd166", "bottom_style": "pants", "bottom_length": 1.0,
            "bottom_color": "#20263a", "shoes_style": "boots", "shoes_color": "#c2410c",
            "belt": True, "brow_style": "bold", "mouth_style": "smirk",
        },
    },
    "grandma": {
        "label": "Grandma",
        "icon": "👵",
        "spec": {
            "height": 1.55, "head_scale": 1.1, "body_fat": 0.62, "shoulder_scale": 0.86,
            "hair_style": "topknot", "hair_color": "#cfd3d8", "hair_tip_color": "#eef1f4",
            "skin_tone": "#e8bb95", "skin_detail": 0.85, "lashes": False,
            "brow_style": "thin", "brow_color": "#b9bec6", "mouth_style": "neutral",
            "glasses": "round", "top_style": "tee", "top_pattern": "plaid",
            "top_color": "#b06ab3", "top_color2": "#e7d3f2", "sleeve": "long",
            "bottom_style": "skirt", "bottom_length": 0.6, "bottom_color": "#4d5a6b",
            "shoes_style": "sandals", "belt": False, "blush": 0.35,
        },
    },
    "tropical": {
        "label": "Tourist",
        "icon": "🌺",
        "spec": {
            "height": 1.74, "body_fat": 0.66, "muscle": 0.25, "head_scale": 1.2,
            "skin_tone": "#f6d0a6", "top_style": "tropical", "top_pattern": "tropical",
            "top_color": "#ff7b54", "top_color2": "#2ec4b6", "collar": "v",
            "bottom_style": "shorts", "bottom_length": 0.35, "bottom_pattern": "plaid",
            "bottom_color": "#f6f7f4", "bottom_color2": "#ff7b54", "shoes_style": "sandals",
            "hat": "sunhat", "hat_color": "#f6e6b4", "hat_color2": "#e0503a",
            "glasses": "shades", "hair_style": "short", "hair_color": "#3d2418",
            "mouth_style": "grin", "blush": 0.4,
        },
    },
    "ninja": {
        "label": "Ninja",
        "icon": "🥷",
        "spec": {
            "height": 1.72, "muscle": 0.7, "body_fat": 0.2, "head_scale": 1.0,
            "skin_tone": "#d8a276", "hair_style": "ponytail", "hair_color": "#161a24",
            "hair_tip_color": "#3b4a6b", "eye_color": "#2b2f3a", "brow_style": "angry",
            "mouth_style": "neutral", "top_style": "hoodie", "collar": "hood",
            "top_pattern": "solid", "top_color": "#161a24", "top_color2": "#c2410c",
            "sleeve": "long", "bottom_style": "pants", "bottom_length": 0.85,
            "bottom_color": "#1c2230", "shoes_style": "boots", "shoes_color": "#2d3340",
            "belt": True, "socks": True,
        },
    },
    "zombie": {
        "label": "Zombie",
        "icon": "🧟",
        "spec": {
            "height": 1.75, "body_fat": 0.3, "muscle": 0.4, "head_scale": 1.0,
            "skin_tone": "#9dbf86", "skin_detail": 0.9, "eye_color": "#c9d94a",
            "eye_style": "sleepy", "brow_style": "raised", "mouth_style": "open",
            "blush": 0.0, "freckles": 0.5, "hair_style": "spiky", "hair_color": "#4b5a3c",
            "hair_tip_color": "#7f9163", "top_style": "tee", "top_pattern": "camo",
            "top_color": "#6b7a4f", "top_color2": "#3d4a2e", "sleeve": "short",
            "bottom_style": "pants", "bottom_length": 0.75, "bottom_pattern": "camo",
            "bottom_color": "#4a5340", "shoes_style": "barefoot", "belt": False,
        },
    },
    "sea": {
        "label": "Beach Bum",
        "icon": "🏖️",
        "spec": {
            "height": 1.7, "body_fat": 0.5, "muscle": 0.55, "head_scale": 1.12,
            "skin_tone": "#b5764a", "skin_detail": 0.6, "hair_style": "long",
            "hair_color": "#2a1b12", "hair_tip_color": "#c98a4b", "eye_color": "#2e8b8b",
            "top_style": "none", "top_pattern": "solid", "bottom_style": "shorts",
            "bottom_length": 0.28, "bottom_pattern": "tropical", "bottom_color": "#ffd166",
            "bottom_color2": "#06d6a0", "shoes_style": "barefoot", "belt": False,
            "glasses": "shades", "blush": 0.45,
        },
    },
}

# ---------------------------------------------------------------------------
# Randomizer -----------------------------------------------------------------
# ---------------------------------------------------------------------------

_HAIR = ["short", "buzz", "bob", "long", "ponytail", "spiky", "curly", "mohawk", "topknot", "bald"]
_TOPS = ["tee", "tank", "tropical", "hoodie", "tee", "tee"]
_PATTERNS = ["pineapple", "solid", "stripes", "dots", "plaid", "stars", "tropical", "gradient", "camo"]
_EXTRAS = ["none", "none", "none", "cat", "fox", "puff"]


def random_spec(seed: int | None = None, base: dict[str, Any] | None = None) -> dict[str, Any]:
    """Tasteful random character. Correlates a few traits so results look intentional."""
    rng = random.Random(seed)
    spec = defaults() if base is None else clamp_spec(base)

    chibi = rng.random() < 0.22
    tall = rng.random() < 0.25
    if chibi:
        spec.update(
            height=round(rng.uniform(1.0, 1.3), 2), head_scale=round(rng.uniform(1.5, 1.9), 2),
            leg_length=round(rng.uniform(0.75, 0.9), 2), eye_size=round(rng.uniform(1.2, 1.5), 2),
            body_fat=round(rng.uniform(0.35, 0.7), 2), neck_scale=round(rng.uniform(0.65, 0.85), 2),
            arm_length=round(rng.uniform(0.82, 0.95), 2), hand_size=round(rng.uniform(1.0, 1.3), 2),
        )
    else:
        spec.update(
            height=round(rng.uniform(1.62, 1.95) if tall else rng.uniform(1.48, 1.78), 2),
            head_scale=round(rng.uniform(0.92, 1.25), 2),
            leg_length=round(rng.uniform(0.95, 1.25) if tall else rng.uniform(0.88, 1.12), 2),
        )

    spec["body_fat"] = round(rng.uniform(0.1, 0.8), 2)
    spec["muscle"] = round(rng.uniform(0.1, 0.95) if not chibi else rng.uniform(0.05, 0.5), 2)
    spec["shoulder_scale"] = round(rng.uniform(0.85, 1.35), 2)
    spec["hip_scale"] = round(rng.uniform(0.85, 1.25), 2)
    spec["chest"] = round(rng.uniform(0.25, 0.75), 2)
    spec["waist"] = round(rng.uniform(0.15, 0.8), 2)
    spec["arm_thick"] = round(rng.uniform(0.8, 1.35), 2)
    spec["leg_thick"] = round(rng.uniform(0.85, 1.4), 2)
    spec["torso_scale"] = round(rng.uniform(0.92, 1.12), 2)
    spec["skin_tone"] = rng.choice(["#f6d3b0", "#f0c69c", "#dfa87a", "#c98a5b", "#a86a42", "#7d4b2c", "#5a3421", "#f2c1a0", "#e8b48c"])
    spec["hair_style"] = rng.choice(_HAIR)
    spec["hair_color"] = rng.choice(["#3b2a20", "#1b1f3a", "#5a3421", "#f2b134", "#e0503a", "#7c4dff", "#2ec4b6", "#cfd3d8", "#ff8fab", "#161a24"])
    spec["hair_tip_color"] = rng.choice(["#a8703f", "#ffd166", "#e9f1f7", "#5b6bff", "#ff7b54", "#06d6a0", spec["hair_color"]])
    spec["hair_volume"] = round(rng.uniform(0.75, 1.35), 2)
    spec["bangs"] = round(rng.uniform(0.0, 1.0), 2)
    spec["eye_color"] = rng.choice(["#3f6fae", "#2e8b57", "#8b5e3c", "#6b4fbb", "#c9a227", "#3a3f47", "#2ec4b6", "#c74b50"])
    spec["eye_style"] = rng.choice(["round", "round", "wide", "almond", "sleepy", "star"])
    spec["eye_size"] = round(rng.uniform(0.85, 1.35), 2)
    spec["brow_style"] = rng.choice(["soft", "thin", "bold", "angry", "raised"])
    spec["mouth_style"] = rng.choice(["smile", "smile", "grin", "smirk", "neutral", "open"])
    spec["blush"] = round(rng.uniform(0.0, 0.6), 2)
    spec["freckles"] = round(rng.choice([0, 0, 0, rng.uniform(0.2, 0.8)]), 2)
    spec["nose_style"] = rng.choice(["button", "small", "snub", "wide"])
    spec["beard"] = rng.choice(["none", "none", "none", "none", "stubble", "mustache", "goatee", "full"])
    spec["beard_color"] = spec["hair_color"]

    spec["top_style"] = rng.choice(_TOPS)
    spec["top_pattern"] = rng.choice(_PATTERNS)
    spec["top_color"] = rng.choice(["#20c4b5", "#ff7b54", "#2b4a6f", "#f2b134", "#7c4dff", "#ff8fab", "#06d6a0", "#e9f1f7", "#161a24", "#c74b50"])
    spec["top_color2"] = rng.choice(["#ffd166", "#e9f1f7", "#161a24", "#06d6a0", "#c74b50", "#7c4dff"])
    spec["sleeve"] = rng.choice(["short", "long", "none"] if spec["top_style"] in ("tee", "tropical", "hoodie") else ["short", "long"])
    spec["collar"] = rng.choice(["crew", "v", "hood", "none"])
    spec["bottom_style"] = rng.choice(["shorts", "shorts", "pants", "skirt", "capri", "none"])
    spec["bottom_length"] = {"shorts": round(rng.uniform(0.25, 0.45), 2), "pants": round(rng.uniform(0.9, 1.0), 2),
                             "capri": round(rng.uniform(0.55, 0.75), 2), "skirt": round(rng.uniform(0.35, 0.7), 2),
                             "none": 0.4}[spec["bottom_style"]]
    spec["bottom_pattern"] = rng.choice(["solid", "solid", "stripes", "dots", "plaid", "stars", "camo", "gradient"])
    spec["bottom_color"] = rng.choice(["#2b4a6f", "#161a24", "#f6f7f4", "#c98a5b", "#4a5340", "#3f3f8f", "#ffd166"])
    spec["bottom_color2"] = rng.choice(["#e9f1f7", "#ffd166", "#2ec4b6", "#e0503a", "#20242c"])
    spec["belt"] = rng.random() < 0.5
    spec["shoes_style"] = rng.choice(["sneakers", "sneakers", "boots", "sandals", "barefoot"])
    spec["shoes_color"] = rng.choice(["#e0503a", "#f5f7fa", "#20242c", "#2ec4b6", "#ffd166", "#7c4dff"])
    spec["shoes_color2"] = rng.choice(["#f5f7fa", "#20242c", "#c2410c"])
    spec["socks"] = rng.random() < 0.35

    spec["ears"] = rng.choice(["none", "none", "none", "none", "cat", "fox", "elf", "round"])
    spec["tail"] = rng.choice(["none", "none", "none", "none", "cat", "fox", "puff"])
    if spec["ears"] != "none" or spec["tail"] != "none":
        spec["ears"] = spec["ears"] if spec["ears"] != "none" else rng.choice(["cat", "fox", "puff"])
    spec["glasses"] = rng.choice(["none", "none", "none", "round", "square", "shades"])
    spec["hat"] = rng.choice(["none", "none", "none", "none", "cap", "beanie", "sunhat", "wizard"])
    if spec["hat"] != "none":
        spec["hat_color"] = rng.choice(["#7c4dff", "#e0503a", "#20242c", "#2ec4b6", "#f2b134", "#f6e6b4"])
        spec["hat_color2"] = rng.choice(["#ffd166", "#e9f1f7", "#20242c"])
    spec["expression"] = "custom"
    return clamp_spec(spec)


def preset_spec(name: str) -> dict[str, Any]:
    """Preset specs are built on top of the randomizer baseline for variety."""
    p = PRESETS.get(name)
    if not p:
        return defaults()
    base = defaults()
    if name != "colada":
        # give non-default presets a deterministic random base for richness
        base = random_spec(seed=hash(name) % 10_000)
    base.update(copy.deepcopy(p["spec"]))
    return clamp_spec(base)
