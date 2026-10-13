/**
 * Colada Character Maker - viewport
 * three.js scene, glb loading, animation transport, capture helpers.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const THEMES = {
  studio: {
    bg: 0x101623, fog: 0x101623,
    hemi: [0xdfe8ff, 0x2a3242, 0.55],
    key: [0xffffff, 2.4, [2.6, 3.4, 3.2]],
    fill: [0x9fd0ff, 0.9, [-3.2, 1.8, 2.2]],
    rim: [0xffd9a0, 1.6, [-1.2, 2.4, -3.6]],
    ground: 0x1a2233, grid: 0x2c3a55,
  },
  sunset: {
    bg: 0x2a1626, fog: 0x2a1626,
    hemi: [0xffb27a, 0x3a1f33, 0.5],
    key: [0xffa15c, 2.8, [3.4, 2.2, 2.6]],
    fill: [0xff6f91, 1.1, [-3.0, 1.4, 2.0]],
    rim: [0x7a5cff, 1.4, [-1.0, 2.6, -3.4]],
    ground: 0x3a2130, grid: 0x5a3348,
  },
  neon: {
    bg: 0x0a0a14, fog: 0x0a0a14,
    hemi: [0x59f2ff, 0xff4fd8, 0.45],
    key: [0x59f2ff, 2.2, [2.8, 3.0, 3.0]],
    fill: [0xff4fd8, 1.6, [-3.0, 1.6, 2.4]],
    rim: [0xa6ff4f, 1.2, [-1.2, 2.2, -3.2]],
    ground: 0x11121f, grid: 0x353a5c,
  },
  night: {
    bg: 0x070a12, fog: 0x070a12,
    hemi: [0x8fa8ff, 0x121826, 0.32],
    key: [0xcfe0ff, 1.5, [2.2, 3.6, 2.8]],
    fill: [0x3a5cff, 0.7, [-3.2, 1.4, 2.2]],
    rim: [0xffffff, 1.1, [-1.0, 2.6, -3.4]],
    ground: 0x0d1220, grid: 0x222c44,
  },
  beach: {
    bg: 0x1e3340, fog: 0x1e3340,
    hemi: [0xfff2cf, 0x2b4a52, 0.85],
    key: [0xfff0c9, 2.9, [2.4, 4.2, 3.0]],
    fill: [0x7fd8ff, 0.9, [-3.4, 1.6, 2.4]],
    rim: [0xffc98a, 1.2, [-1.4, 2.4, -3.6]],
    ground: 0x2e4a4a, grid: 0x3f6360,
  },
};

export class Viewer {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(38, 1, 0.05, 120);
    this.camera.position.set(1.6, 1.25, 2.6);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = 0.5;
    this.controls.maxDistance = 14;
    this.controls.target.set(0, 0.95, 0);
    this.controls.autoRotateSpeed = 1.6;

    this.lights = new THREE.Group();
    this.scene.add(this.lights);
    this.ground = null;
    this.grid = null;
    this.model = null;
    this.skeletonHelper = null;
    this.mixer = null;
    this.actions = {};
    this.clipInfo = {};
    this.current = null;
    this.playing = false;
    this.loop = true;
    this.speed = 1;
    this.headTrack = false;
    this.wireframe = false;
    this._clock = new THREE.Clock();
    this.theme = 'studio';

    this.setTheme('studio');
    this._bindResize();
    this._tick = this._tick.bind(this);
    this.renderer.setAnimationLoop(this._tick);
  }

  // ---------------------------------------------------------------- scene
  setTheme(name, opts = {}) {
    const t = THEMES[name] || THEMES.studio;
    this.theme = name;
    this.scene.background = new THREE.Color(t.bg);
    if (this.scene.fog) this.scene.fog = null;

    this.lights.clear();
    const hemi = new THREE.HemisphereLight(t.hemi[0], t.hemi[1], t.hemi[2]);
    this.lights.add(hemi);
    const names = ['key', 'fill', 'rim'];
    this.lightRefs = {};
    for (const n of names) {
      const [col, intensity, pos] = t[n];
      const l = new THREE.DirectionalLight(col, intensity);
      l.position.set(...pos);
      l.castShadow = n === 'key';
      if (l.castShadow) {
        l.shadow.mapSize.set(1024, 1024);
        l.shadow.camera.near = 0.5;
        l.shadow.camera.far = 14;
        l.shadow.camera.left = -1.8;
        l.shadow.camera.right = 1.8;
        l.shadow.camera.top = 2.6;
        l.shadow.camera.bottom = -0.4;
        l.shadow.bias = -0.0016;
      }
      this.lights.add(l);
      this.lightRefs[n] = l;
    }
    if (this.ground) {
      this.ground.material.color.setHex(t.ground);
      if (this.grid) this.grid.material.color.setHex(t.grid);
    }
  }

  setGround(show) {
    if (show && !this.ground) {
      const t = THEMES[this.theme];
      const mat = new THREE.MeshStandardMaterial({ color: t.ground, roughness: 0.95, metalness: 0.0 });
      this.ground = new THREE.Mesh(new THREE.CircleGeometry(6, 64), mat);
      this.ground.rotation.x = -Math.PI / 2;
      this.ground.receiveShadow = true;
      this.scene.add(this.ground);
      const grid = new THREE.GridHelper(6, 24, t.grid, t.grid);
      grid.material.transparent = true;
      grid.material.opacity = 0.55;
      grid.position.y = 0.002;
      this.grid = grid;
      this.scene.add(grid);
    } else if (!show && this.ground) {
      this.scene.remove(this.ground, this.grid);
      this.ground.geometry.dispose();
      this.grid.geometry.dispose();
      this.ground = null;
      this.grid = null;
    }
    if (this.grid) this.grid.visible = this.showGrid;
  }

  setGridVisible(v) {
    this.showGrid = v;
    if (this.grid) this.grid.visible = v;
  }

  // ---------------------------------------------------------------- model
  async loadGLB(url) {
    const loader = new GLTFLoader();
    const gltf = await loader.loadAsync(url);
    if (this.skeletonHelper) {
      this.scene.remove(this.skeletonHelper);
      if (this.skeletonHelper.geometry) this.skeletonHelper.geometry.dispose();
      if (this.skeletonHelper.material) this.skeletonHelper.material.dispose();
      this.skeletonHelper = null;
    }
    if (this.model) {
      this.scene.remove(this.model);
      this.disposeTree(this.model);
    }
    this.mixer = null;
    this.actions = {};
    this.clipInfo = {};
    this.current = null;

    this.model = gltf.scene;
    this.restQuats = {};
    this.model.traverse((o) => {
      if (o.isBone || o.type === 'Bone' || o.name === 'Head') {
        this.restQuats[o.name] = o.quaternion.clone();
      }
    });
    this.model.traverse((o) => {
      if (o.isMesh || o.isSkinnedMesh) {
        o.castShadow = true;
        o.receiveShadow = true;
        o.frustumCulled = false;
        const mats = Array.isArray(o.material) ? o.material : [o.material];
        for (const m of mats) {
          if (m && m.map) {
            m.map.colorSpace = THREE.SRGBColorSpace;
            m.map.anisotropy = 4;
          }
          if (m) m.side = m.transparent ? THREE.DoubleSide : m.side;
        }
      }
    });
    this.scene.add(this.model);

    if (gltf.animations && gltf.animations.length) {
      this.mixer = new THREE.AnimationMixer(this.model);
      this.mixer.addEventListener('finished', () => {
        this.playing = false;
      });
      for (const clip of gltf.animations) {
        const action = this.mixer.clipAction(clip);
        const key = clip.name.replace(/^Colada_/, '').toLowerCase();
        this.actions[key] = action;
        this.clipInfo[key] = { duration: clip.duration, name: clip.name };
        action.clampWhenFinished = true;
      }
    }
    this.setWireframe(this.wireframe);
    if (this.showSkeleton) this.setSkeleton(true);
    return { clips: Object.keys(this.actions).map((k) => ({ name: k, duration: this.clipInfo[k].duration })) };
  }

  disposeTree(root) {
    root.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      const mats = Array.isArray(o.material) ? o.material : [o.material];
      for (const m of mats) {
        if (!m) continue;
        if (m.map && m.map.dispose) m.map.dispose();
        m.dispose && m.dispose();
      }
    });
  }

  fitCamera() {
    if (!this.model) return;
    const box = new THREE.Box3().setFromObject(this.model);
    const size = new THREE.Vector3();
    const center = new THREE.Vector3();
    box.getSize(size);
    box.getCenter(center);
    const h = Math.max(0.6, size.y);
    const fov = THREE.MathUtils.degToRad(this.camera.fov);
    const dist = (h * 0.62) / Math.tan(fov * 0.5) * 1.35;
    this.controls.target.set(center.x, center.y + h * 0.02, center.z);
    const dir = new THREE.Vector3(0.62, 0.16, 1).normalize();
    this.camera.position.copy(this.controls.target).addScaledVector(dir, dist);
    this.controls.update();
    if (this.lightRefs) {
      const key = this.lightRefs.key;
      if (key) key.shadow.camera.updateProjectionMatrix();
    }
  }

  // ----------------------------------------------------------- transport
  play(name, { restart = false } = {}) {
    if (!this.mixer) return;
    if (!name) name = this.current || Object.keys(this.actions)[0];
    if (!this.actions[name]) return;
    const action = this.actions[name];
    if (this.current && this.current !== name && this.actions[this.current]) {
      this.actions[this.current].fadeOut(0.18);
    }
    if (restart || this.current !== name) action.reset();
    action.paused = false;
    action.enabled = true;
    action.setLoop(this.loop ? THREE.LoopRepeat : THREE.LoopOnce, Infinity);
    action.clampWhenFinished = true;
    action.setEffectiveTimeScale(this.speed);
    action.setEffectiveWeight(1);
    action.fadeIn(0.18).play();
    this.current = name;
    this.playing = true;
  }

  pause() {
    this.playing = false;
  }

  resume() {
    if (!this.current || !this.actions[this.current]) return;
    const a = this.actions[this.current];
    const dur = this.duration();
    if (!this.loop && dur > 0 && a.time >= dur - 1e-3) {
      a.reset();
    }
    a.paused = false;
    a.enabled = true;
    a.play();
    this.playing = true;
  }

  toggle() { this.playing ? this.pause() : this.resume(); }

  seek(t) {
    if (!this.current) return;
    const a = this.actions[this.current];
    if (!a) return;
    a.enabled = true;
    a.paused = false;
    a.play();
    a.time = Math.max(0, Math.min(this.clipInfo[this.current].duration, t));
    if (this.mixer) this.mixer.update(0);
  }

  time() { return (this.current && this.actions[this.current]) ? this.actions[this.current].time : 0; }
  duration() { return (this.current && this.clipInfo[this.current]) ? this.clipInfo[this.current].duration : 0; }

  setSpeed(v) {
    this.speed = v;
    if (this.current) this.actions[this.current]?.setEffectiveTimeScale(v);
  }

  setLoop(v) {
    this.loop = v;
    if (this.current) {
      const a = this.actions[this.current];
      a.setLoop(v ? THREE.LoopRepeat : THREE.LoopOnce, Infinity);
      a.clampWhenFinished = true;
    }
  }

  // -------------------------------------------------------------- extras
  setWireframe(v) {
    this.wireframe = v;
    if (!this.model) return;
    this.model.traverse((o) => {
      const mats = Array.isArray(o.material) ? o.material : [o.material];
      for (const m of mats) if (m) m.wireframe = v;
    });
  }

  setSkeleton(v) {
    this.showSkeleton = v;
    if (!this.model) return;
    if (v && !this.skeletonHelper) {
      this.skeletonHelper = new THREE.SkeletonHelper(this.model);
      this.skeletonHelper.material.linewidth = 2;
      this.skeletonHelper.material.depthTest = false;
      this.skeletonHelper.renderOrder = 3;
      this.scene.add(this.skeletonHelper);
    }
    if (this.skeletonHelper) this.skeletonHelper.visible = v;
  }

  setAutoRotate(v) { this.controls.autoRotate = v; }
  setHeadTrack(v) { this.headTrack = v; }

  _tick() {
    const dt = this._clock.getDelta();
    this.controls.update();
    if (this.mixer && this.playing) this.mixer.update(dt * 1.0);
    if (this.headTrack && this.model) this._applyHeadTrack();
    this.renderer.render(this.scene, this.camera);
  }

  _applyHeadTrack() {
    // The Head bone's local +Z is the character's forward, so aiming +Z at the
    // camera in the parent bone's space makes the head look at the viewer.
    const bone = this.model.getObjectByName('Head');
    if (!bone || !bone.parent) return;
    const rest = this.restQuats && this.restQuats.Head;
    if (!rest) return;
    const headWorld = new THREE.Vector3();
    bone.getWorldPosition(headWorld);
    const dir = this.camera.position.clone().sub(headWorld).normalize();
    const pq = new THREE.Quaternion();
    bone.parent.getWorldQuaternion(pq).invert();
    const localDir = dir.applyQuaternion(pq).normalize();
    const restFwd = new THREE.Vector3(0, 0, 1).applyQuaternion(rest).normalize();
    const delta = new THREE.Quaternion().setFromUnitVectors(restFwd, localDir);
    const target = delta.multiply(rest).normalize();
    // never fully override the animation - blend 30% toward the viewer
    bone.quaternion.slerp(target, 0.30);
  }

  // ------------------------------------------------------------- capture
  screenshot() {
    this.renderer.render(this.scene, this.camera);
    return this.canvas.toDataURL('image/png');
  }

  startRecording(onDone) {
    const stream = this.canvas.captureStream(30);
    const types = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'];
    const mime = types.find((t) => MediaRecorder.isTypeSupported(t));
    if (!mime) { onDone(null); return null; }
    const rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 6_000_000 });
    const chunks = [];
    rec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
    rec.onstop = () => {
      const blob = new Blob(chunks, { type: 'video/webm' });
      onDone(URL.createObjectURL(blob));
    };
    rec.start();
    return rec;
  }

  loadTexturePreview(url) {
    return new THREE.TextureLoader().load(url);
  }

  _bindResize() {
    const resize = () => {
      const parent = this.canvas.parentElement;
      const w = parent.clientWidth;
      const h = parent.clientHeight;
      this.renderer.setSize(w, h, false);
      this.camera.aspect = w / Math.max(1, h);
      this.camera.updateProjectionMatrix();
    };
    this._resize = resize;
    window.addEventListener('resize', resize);
    if (window.ResizeObserver) new ResizeObserver(resize).observe(this.canvas.parentElement);
    resize();
  }
}
