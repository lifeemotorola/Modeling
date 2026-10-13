"""
Colada Character Maker - software preview renderer
==================================================
Tiny z-buffered rasteriser used for CLI contact sheets.  The web UI renders
with three.js; this exists so characters can be eyeballed (and animation
frames checked) straight from Python with no GPU, no browser and no deps
beyond numpy + Pillow.
"""
from __future__ import annotations

import io
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _look_at(eye, target, up=np.array([0.0, 1.0, 0.0])):
    f = target - eye
    f = f / max(1e-9, np.linalg.norm(f))
    r = np.cross(f, up)
    r = r / max(1e-9, np.linalg.norm(r))
    u = np.cross(r, f)
    m = np.eye(4)
    m[0, :3] = r
    m[1, :3] = u
    m[2, :3] = -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def render(mesh, tex: Image.Image, size=(420, 620), yaw=0.0, pitch=-4.0, dist=4.2,
           target_y=None, bg=(24, 28, 36), ground=True, verts_override=None,
           light_dir=(-0.4, 0.75, 0.55)):
    W, H = size
    verts = np.asarray(verts_override if verts_override is not None else mesh.verts, dtype=np.float64)
    norms = np.asarray(mesh.norms if hasattr(mesh, "norms") else mesh.normals, dtype=np.float64)
    uvs = np.asarray(mesh.uvs, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    ymin, ymax = float(np.min(verts[:, 1])), float(np.max(verts[:, 1]))
    height = max(0.6, ymax - ymin)
    fov = math.radians(30.0)
    fit = (height * 1.10) / (2.0 * math.tan(fov * 0.5)) * (W / float(H))
    dist = max(fit, height * 0.6) * (dist / 4.2)
    center = (ymin + ymax) * 0.5 + height * 0.02
    ty = center if target_y is None else target_y

    a = math.radians(yaw)
    p = math.radians(pitch)
    eye = np.array([math.sin(a) * math.cos(p), math.sin(p) * 0.6, math.cos(a) * math.cos(p)]) * dist
    eye = eye + np.array([0.0, ty, 0.0])
    view = _look_at(eye, np.array([0.0, ty, 0.0]))
    f = 1.0 / math.tan(fov * 0.5)

    # camera space
    cam = verts @ view[:3, :3].T + view[:3, 3]
    cam[:, 2] = np.where(cam[:, 2] > -1e-4, -1e-4, cam[:, 2])          # in front of camera
    scr = np.empty((len(cam), 3))
    scr[:, 0] = (cam[:, 0] * f / -cam[:, 2]) * (W * 0.5) + W * 0.5
    scr[:, 1] = -(cam[:, 1] * f / -cam[:, 2]) * (W * 0.5) + H * 0.5
    scr[:, 2] = -cam[:, 2]

    tex_arr = np.asarray(tex.convert("RGB"), dtype=np.float64) / 255.0
    th, tw = tex_arr.shape[0], tex_arr.shape[1]

    img = np.zeros((H, W, 3), dtype=np.float64)
    # vertical gradient background
    for y in range(H):
        t = y / max(1, H - 1)
        img[y, :, :] = np.array(bg, dtype=np.float64) * (1.0 + 0.35 * (1 - t) * 0.9) / 255.0
    if ground:
        gy = int(H * 0.5 - (0.0 - ty) * f / dist * (W * 0.5))
        if 0 <= gy < H:
            img[gy:, :, :] *= 0.82
            img[gy:gy + 2, :, :] = np.array([0.35, 0.42, 0.5])

    zbuf = np.full((H, W), 1e9)

    # lighting
    L = np.array(light_dir, dtype=np.float64)
    L = L / np.linalg.norm(L)
    world_n = norms @ view[:3, :3].T          # normals in camera space

    tri = faces
    # backface + sub-pixel cull up front (the raster loop is the expensive part)
    p0, p1, p2 = scr[tri[:, 0]], scr[tri[:, 1]], scr[tri[:, 2]]
    area = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1])
    span = (np.maximum.reduce([p0[:, 0], p1[:, 0], p2[:, 0]]) - np.minimum.reduce([p0[:, 0], p1[:, 0], p2[:, 0]])) \
        * (np.maximum.reduce([p0[:, 1], p1[:, 1], p2[:, 1]]) - np.minimum.reduce([p0[:, 1], p1[:, 1], p2[:, 1]]))
    keep = (area < -0.35) & (span > 0.35)
    for i in np.nonzero(keep)[0]:
        a0, a1, a2 = p0[i], p1[i], p2[i]
        ar = area[i]
        xmin = max(0, int(min(a0[0], a1[0], a2[0])))
        xmax = min(W - 1, int(max(a0[0], a1[0], a2[0])) + 1)
        ymin_i = max(0, int(min(a0[1], a1[1], a2[1])))
        ymax_i = min(H - 1, int(max(a0[1], a1[1], a2[1])) + 1)
        if xmin > xmax or ymin_i > ymax_i:
            continue
        xs = np.arange(xmin, xmax + 1) + 0.5
        ys = np.arange(ymin_i, ymax_i + 1) + 0.5
        X, Y = np.meshgrid(xs, ys)
        w0 = ((a1[0] - X) * (a2[1] - Y) - (a2[0] - X) * (a1[1] - Y)) / ar
        w1 = ((a2[0] - X) * (a0[1] - Y) - (a0[0] - X) * (a2[1] - Y)) / ar
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        z = w0 * a0[2] + w1 * a1[2] + w2 * a2[2]
        sub = zbuf[ymin_i:ymax_i + 1, xmin:xmax + 1]
        hit = inside & (z < sub)
        if not hit.any():
            continue
        uv = (w0[..., None] * uvs[tri[i, 0]] + w1[..., None] * uvs[tri[i, 1]] + w2[..., None] * uvs[tri[i, 2]])
        tx = np.clip((uv[..., 0] * tw).astype(np.int64), 0, tw - 1)
        tyi = np.clip(((1.0 - uv[..., 1]) * th).astype(np.int64), 0, th - 1)
        col = tex_arr[tyi, tx]
        n = (w0[..., None] * world_n[tri[i, 0]] + w1[..., None] * world_n[tri[i, 1]] + w2[..., None] * world_n[tri[i, 2]])
        n = n / np.maximum(1e-9, np.linalg.norm(n, axis=-1, keepdims=True))
        diff = np.clip(n @ L, 0.0, 1.0)[..., None] * 0.72 + 0.30
        spec_ = np.clip((n @ L) ** 22, 0, 1)[..., None] * 0.12
        shaded = np.clip(col * diff + spec_, 0.0, 1.0)
        sub[hit] = z[hit]
        target = img[ymin_i:ymax_i + 1, xmin:xmax + 1]
        target[hit] = shaded[hit]

    out = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8), "RGB")
    return out


def contact_sheet(mesh, tex, size=(420, 620), yaws=(0, 40, 145, 250), labels=None, cols=4) -> Image.Image:
    views = [render(mesh, tex, size=size, yaw=y) for y in yaws]
    W, H = size
    sheet = Image.new("RGB", (W * cols, H + 26), (16, 18, 24))
    d = ImageDraw.Draw(sheet)
    for i, v in enumerate(views):
        sheet.paste(v, (i * W, 26))
        d.text((i * W + 10, 7), (labels[i] if labels else f"yaw {yaws[i]}"), fill=(220, 226, 235))
    return sheet


def pose_verts(mesh, rig, pose, sampler=None) -> np.ndarray:
    """Skinned vertex positions for a Pose (used for animation thumbnails)."""
    from .animations import Sampler
    sampler = sampler or Sampler(rig)
    T, R = sampler.pose_to_local(pose)
    # forward kinematics
    world = [None] * len(rig.bones)
    for i, bone in enumerate(rig.bones):
        if bone.parent is None:
            M = np.eye(4)
            M[:3, :3] = R[i]
            M[:3, 3] = T[i]
        else:
            M = world[rig.name_to_index[bone.parent]].copy()
            L = np.eye(4)
            L[:3, :3] = R[i]
            L[:3, 3] = T[i]
            M = M @ L
        world[i] = M
    skin_mats = np.stack([world[rig.name_to_index[b.name]] @ rig.inverse_bind()[k]
                          for k, b in enumerate(rig.deform_bones)])
    v = np.asarray(mesh.verts, dtype=np.float64)
    hom = np.concatenate([v, np.ones((len(v), 1))], axis=1)          # (N, 4)
    out = np.zeros_like(v)
    for k in range(4):
        j = mesh.joints[:, k].astype(np.int64)
        w = mesh.weights[:, k: k + 1]
        mats = skin_mats[j][:, :3, :]                                 # (N, 3, 4)
        out += (mats @ hom[:, :, None])[..., 0] * w
    return out
