# 🍍 Colada Character Maker

**An advanced procedural character creator: HTML studio + Python engine + Blender exporter.**

Build a fully featured 3D character — body, face, hair, outfit, accessories — with
**baked textures, an auto-generated rig, and ten animation clips**, then take it
into Blender (or any glTF app) ready to animate.

Everything is generated from a single JSON spec, so a character is reproducible,
shareable and tweakable forever.

```
spec (JSON) ─→ metrics ─→ skeleton ─→ skinned mesh ─→ texture atlas ─→ .glb + Blender script
```

---

## Quick start

```bash
# 1. the studio (no dependencies beyond numpy + pillow)
python3 web/server.py --port 8000
#    → open http://localhost:8000

# 2. or straight from the command line
python3 cli.py --preset colada --out build --blender --zip
python3 cli.py --random --seed 42 --preview

# 3. or build every preset into a gallery
python3 cli.py --all-presets --gallery
```

**Requirements:** Python 3.9+, `numpy`, `Pillow`. Nothing else — the web server
uses only the standard library and the vendored three.js is included.

---

## What it makes

| | |
|---|---|
| **Mesh** | one skinned mesh, 5–8k verts / 10–12k tris, parametric detail level |
| **Rig** | 22–24 deform bones (spine, neck/head, arms, hands, legs, feet, toes, hair/tail/ears) + optional finger bones |
| **Skinning** | distance-weighted auto skinning, max 4 influences, normalised weights |
| **Texture** | one baked 512/1024/2048 atlas — face, skin, hair, top, bottoms, shoes, extras |
| **Animations** | idle, walk, run, jump, wave, dance, kick, cheer, zombie, t-pose (30 fps, sampled) |
| **Exports** | `.glb` (mesh+rig+animations+materials), texture `.png`, `character_spec.json`, a **native Blender build script**, and a zip bundle |

Faces are painted procedurally: separate curved patches for eyes, eyebrows and
mouth means you get real eyelids, lashes, teeth, iris highlights and blushes —
not a flat decal.

---

## The studio

The HTML app is generated from the engine's own schema (`core/spec.py`), so the
UI can never drift from the generator. 74 controls across 7 groups:

* **Presets** — Piña Colada, Chibi, Athlete, Hero, Grandma, Tourist, Ninja, Zombie, Beach Bum
* **Body** — height, head size, chubbiness, muscle, torso/shoulders/hips, chest, waist, neck, arms, legs, hands, feet
* **Face** — skin tone + shading, 5 eye styles, brows, 7 mouths, expression macros (happy/angry/surprised/sad/sleepy/smug), blush, freckles, noses, lashes, 5 beards
* **Hair** — 10 styles (short, buzz, bob, long, ponytail, spiky, curly, mohawk, topknot, bald) with a root→tip gradient, volume, bangs, shine
* **Outfit** — 6 tops, 5 bottoms, 9 patterns (pineapple logo, tropical, plaid, camo, stars…), sleeves, collars, belts, 5 shoe types, socks
* **Extras** — cat/fox/elf/round ears, cat/fox/puff tails, glasses, 4 hats
* **Studio** — texture resolution, AA, mesh detail, 5 lighting rigs, grid/ground, wireframe, skeleton overlay, auto-rotate, head-track
* **Rig & Anim** — finger bones, IK targets, bone display size, which clips to bake

Viewport: three.js with orbit controls, shadowed studio lighting, live animation
transport (play / scrub / speed / loop), PNG snapshot and WEBM recording.

Shortcuts: `R` random · `T` tweak · `Space` play · `1‑9` clips · `G` grid ·
`B` bones · `W` wireframe · `Ctrl+Z` undo · `Ctrl+S` export glb.

---

## Taking it into Blender

Three routes, all included:

### 1. Import the `.glb`
`File > Import > glTF 2.0 > model.glb` — the character arrives with its
armature and every action as a separate animation. Press `Space`.

### 2. Run the generated build script
Each export ships `build_character_in_blender.py` (self-contained, data baked in
via zlib+base64). In Blender: **Scripting ▸ Open ▸ Run Script**. It builds
natively:

* real `Colada_Mesh` with vertex groups and one material slot per body part
* real `Colada_Rig` armature with correct bone rolls and deform flags
* real **Actions** (keyed pose bones) pushed onto NLA tracks
* muted **IK targets** for hands and feet (`influence = 0` → unmute to switch to IK)
* a studio light rig + tracked camera, so you can render immediately

### 3. Install the addon
`blender/colada_addon.py` → **Edit ▸ Preferences ▸ Add-ons ▸ Install…** then
enable *Colada Character Maker*. A **Colada** tab appears in the viewport
sidebar (N):

* **Import Colada character** — reads a `.zip` bundle or `.glb`, then wires up
  NLA tracks, IK targets, lights, camera and stores the spec on the scene
* **Run Colada build script** — executes a generated native build script in place
* clip browser + **Play clip** (sets the scene frame range and starts playback)
* **Rig helpers** — IK targets / studio / NLA on demand
* **Export Colada .glb** and **Normalise action names** for round trips

---

## Command line

```
python3 cli.py --preset colada --out build            # preset build
python3 cli.py --random --seed 42                     # reproducible random
python3 cli.py --spec char.json --set hair_style=spiky --set height=1.92
python3 cli.py --preset hero --clips idle,run,kick --texture-size 2048 --fps 60
python3 cli.py --all-presets --gallery                # every preset + contact sheet
python3 cli.py --validate build/colada.glb            # glb structural QA
python3 cli.py --verify  build/colada.glb             # skinning verifier
```

Outputs: `<name>.glb`, `<name>_texture.png`, `<name>_spec.json`,
`<name>_blender.py`, `<name>_preview.png`, `<name>_bundle.zip`.

> **Sample output already lives in `samples/`** — all nine presets as `.glb` +
> texture + spec, plus `colada_bundle.zip` (glb + texture + spec + Blender
> script) and `gallery.png`. They persist across restarts; `build/` and
> `.arena/` are working directories that the workspace snapshot ignores, so
> render into `samples/` (or any other name) when you want to keep results.

---

## HTTP API

| Route | Purpose |
|---|---|
| `GET /api/schema` | controls, presets, clips, defaults (drives the whole UI) |
| `POST /api/random` | `{seed?, base?}` → randomised spec |
| `POST /api/preset` | `{name}` → preset spec |
| `POST /api/build` | `{spec, clips?, preview?, thumbs?}` → glb/texture/script/zip + stats |
| `POST /api/preview` | `{spec}` → PNG rendered by the Python rasteriser |
| `GET /api/library`, `POST /api/library`, `DELETE /api/library/<id>` | character library |
| `GET /build/<id>/<file>` | generated assets |
| `GET /static/...` | UI + vendored three.js |

---

## Layout

```
colada_maker/
├── core/                  the Python engine
│   ├── spec.py            schema, defaults, presets, randomiser, expression macros
│   ├── skeleton.py        body metrics, bone hierarchy, weighted auto-skinning
│   ├── mesh.py            procedural geometry (lathes, tubes, super-ellipsoids, patches)
│   ├── atlas.py           UV atlas layout shared by mesh + painter
│   ├── textures.py        procedural texture painter (face art, patterns)
│   ├── animations.py      pose authoring → bone-local quaternion tracks
│   ├── gltf.py            glTF 2.0 / GLB writer (skin + animations)
│   ├── blender_export.py  generates the native Blender build script
│   ├── preview.py         pure-numpy software renderer (contact sheets, thumbs)
│   └── builder.py         spec → everything, plus the CLI entry point
├── web/server.py          zero-dependency studio server + API
├── static/                HTML studio (index.html, app.js, viewer.js, style.css)
├── blender/colada_addon.py  real Blender addon
├── tools/                 glb validator + skinning verifier
├── cli.py                 command line studio
└── library/               your saved characters
```

### Design notes

* **World-axis pose authoring.** Animations are written in intuitive world axes
  (raise / forward / twist) and converted to bone-local quaternions through each
  bone's rest matrix, so the same numbers work in three.js and Blender.
* **One surface, one truth.** The skull is sculpted by a deformation function;
  eyes, brows, mouth, hair, glasses and hats are all glued onto that same
  deformed surface, so nothing ever floats or sinks in, at any head size.
* **Atlas discipline.** `core/atlas.py` is the single source of truth for where
  every material lives; geometry and the painter both read it, so UVs and art
  can never disagree.
* **Deterministic.** Same spec → same character, byte for byte. The randomiser
  takes a seed and keeps a character's traits internally consistent (proportions
  follow the height, palettes stay tasteful).

---

## Tests / QA

```bash
python3 tools/validate_glb.py build/*.glb     # container, accessors, skin, anims, materials
python3 tools/verify_skin.py build/model.glb --clip idle --frame 30
```

`verify_skin` rebuilds a pose from the glb's own samplers, skins the mesh with
its inverse bind matrices and checks the result stays sane — i.e. that any glTF
consumer will show the same character the engine renders.

---

## Troubleshooting

* **`ImportError: numpy`** → `pip install numpy pillow`
* **Port already in use** → `python3 web/server.py --port 8010`
* **Preview images look flat in the studio** → the Python rasteriser is only used
  for contact sheets/thumbnails; the live viewport is three.js and needs a WebGL
  capable browser.
* **Blender 4.x and custom normals** → the generated script prints a warning if
  `normals_split_custom_set` is unavailable; shading still works with smooth +
  sharp edges.

MIT-style: use it, ship it, make characters. 🍹
