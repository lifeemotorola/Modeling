/**
 * Colada Character Maker - studio app
 * Builds the control UI straight from the engine schema, talks to the Python
 * backend and drives the three.js viewport.
 */
import { Viewer } from './viewer.js';

const $ = (id) => document.getElementById(id);
const api = async (path, body) => {
  const res = await fetch(path, body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
};

const state = {
  schema: [], presets: {}, clipMeta: [], defaults: {},
  spec: {}, clips: [],
  activeTab: 'body',
  autoBuild: true,
  buildId: null,
  undo: [], redo: [],
  recorder: null,
  recordingUntil: 0,
  fitted: false,
};

const viewer = new Viewer($('view'));

/* ------------------------------------------------------------------- UI */

function toast(msg, isError = false) {
  const el = $('toast');
  el.textContent = msg;
  el.classList.toggle('err', isError);
  el.classList.add('show');
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.remove('show'), 2600);
}

function busy(on, text = 'building character…') {
  $('busy-text').textContent = text;
  $('busy').classList.toggle('show', on);
}

function fmt(v, field) {
  const dec = field.step && String(field.step).includes('.') ? String(field.step).split('.')[1].length : 0;
  return Number(v).toFixed(Math.min(2, dec));
}

function buildTabs() {
  const tabs = $('tabs');
  tabs.innerHTML = '';
  for (const group of state.schema) {
    const b = document.createElement('button');
    b.className = 'tab' + (group.id === state.activeTab ? ' active' : '');
    b.innerHTML = `${group.icon || ''} ${group.label}`;
    b.onclick = () => { state.activeTab = group.id; buildTabs(); buildControls(); };
    tabs.appendChild(b);
  }
}

function buildControls() {
  const host = $('controls');
  host.innerHTML = '';
  const group = state.schema.find((g) => g.id === state.activeTab);
  if (!group) return;
  const title = document.createElement('div');
  title.className = 'group-title';
  title.textContent = group.label;
  host.appendChild(title);

  for (const f of group.fields) {
    const wrap = document.createElement('div');
    wrap.className = 'field';
    const cur = state.spec[f.key];
    const label = f.type === 'bool' ? null : `<div class="row"><label>${f.label}</label>`
      + (f.type === 'range' ? `<span class="val" id="v-${f.key}">${fmt(cur, f)}${f.unit || ''}</span>` : '')
      + '</div>';

    if (f.type === 'range') {
      wrap.innerHTML = `${label}<input type="range" min="${f.min}" max="${f.max}" step="${f.step}" value="${cur}">`
        + (f.help ? `<div class="help">${f.help}</div>` : '');
      const inp = wrap.querySelector('input');
      const paint = () => {
        const pct = ((inp.value - f.min) / (f.max - f.min)) * 100;
        inp.style.setProperty('--fill', pct + '%');
        $('v-' + f.key).textContent = fmt(inp.value, f) + (f.unit || '');
      };
      inp.oninput = () => { state.spec[f.key] = parseFloat(inp.value); paint(); schedule(); };
      inp.onchange = () => pushUndo();
      paint();
    } else if (f.type === 'color') {
      wrap.innerHTML = `${label}<div class="swatch"><input type="color" value="${cur}"><code>${cur}</code></div>`;
      const inp = wrap.querySelector('input');
      inp.oninput = () => { state.spec[f.key] = inp.value; wrap.querySelector('code').textContent = inp.value; schedule(); };
      inp.onchange = () => pushUndo();
    } else if (f.type === 'select') {
      const opts = f.options.map((o) => `<option value="${o}"${o === cur ? ' selected' : ''}>${o}</option>`).join('');
      wrap.innerHTML = `${label}<select>${opts}</select>` + (f.help ? `<div class="help">${f.help}</div>` : '');
      const sel = wrap.querySelector('select');
      sel.onchange = () => { state.spec[f.key] = sel.value; pushUndo(); schedule(true); };
    } else if (f.type === 'bool') {
      wrap.innerHTML = `<label class="toggle"><input type="checkbox"${cur ? ' checked' : ''}><span>${f.label}</span></label>`;
      const inp = wrap.querySelector('input');
      inp.onchange = () => { state.spec[f.key] = inp.checked; pushUndo(); schedule(true); };
    } else if (f.type === 'checkboxes') {
      wrap.innerHTML = `<div class="row"><label>${f.label}</label></div><div class="chips"></div>`;
      const chips = wrap.querySelector('.chips');
      for (const o of f.options) {
        const c = document.createElement('button');
        c.className = 'chip' + ((cur || []).includes(o) ? ' on' : '');
        c.textContent = o;
        c.onclick = () => {
          const list = new Set(state.spec[f.key] || []);
          list.has(o) ? list.delete(o) : list.add(o);
          state.spec[f.key] = f.options.filter((x) => list.has(x));
          c.classList.toggle('on');
          pushUndo();
          toast('clips updated — hit Build to re-bake animations');
        };
        chips.appendChild(c);
      }
    }
    host.appendChild(wrap);
  }
}

function buildPresets() {
  const host = $('presets');
  host.innerHTML = '';
  for (const [key, p] of Object.entries(state.presets)) {
    const b = document.createElement('button');
    b.className = 'preset';
    b.innerHTML = `<span>${p.icon}</span>${p.label}`;
    b.onclick = async () => {
      pushUndo();
      const r = await api('/api/preset', { name: key });
      applySpec(r.spec, { keepClips: true });
      toast(`preset: ${p.label}`);
      build();
    };
    host.appendChild(b);
  }
}

function refreshUI() {
  buildTabs();
  buildControls();
}

/* --------------------------------------------------------------- state */

function applySpec(spec, { keepClips = false } = {}) {
  state.spec = { ...state.spec, ...spec };
  if (!keepClips) state.spec.clips = spec.clips || state.spec.clips;
  try { localStorage.setItem('colada.spec', JSON.stringify(state.spec)); } catch (e) { /* ignore */ }
  refreshUI();
  syncTools();
}

function pushUndo() {
  state.undo.push(JSON.stringify(state.spec));
  if (state.undo.length > 60) state.undo.shift();
  state.redo.length = 0;
}

function undo() {
  if (!state.undo.length) return;
  state.redo.push(JSON.stringify(state.spec));
  state.spec = JSON.parse(state.undo.pop());
  refreshUI(); syncTools(); schedule(true);
}

function redo() {
  if (!state.redo.length) return;
  state.undo.push(JSON.stringify(state.spec));
  state.spec = JSON.parse(state.redo.pop());
  refreshUI(); syncTools(); schedule(true);
}

/* --------------------------------------------------------------- build */

let buildTimer = null;
let buildInFlight = false;
let buildQueued = false;

function schedule(immediate = false) {
  if (!state.autoBuild && !immediate) return;
  clearTimeout(buildTimer);
  buildTimer = setTimeout(build, immediate ? 120 : 650);
}

async function build() {
  // never let a slider drag stack up builds: keep only the newest request
  if (buildInFlight) { buildQueued = true; return; }
  buildInFlight = true;
  busy(true);
  $('badge-text').textContent = 'baking…';
  try {
    const t0 = performance.now();
    const meta = await api('/api/build', { spec: state.spec, clips: state.spec.clips });
    state.buildId = meta.id;
    applyMeta(meta);
    const el = ((performance.now() - t0) / 1000).toFixed(2);
    $('badge-text').textContent = `ready · ${el}s round trip`;
    $('engine-line').textContent = `engine ${meta.stats.vertices} verts · ${meta.stats.triangles} tris · `
      + `${meta.stats.glb_kb} kB glb · baked in ${meta.seconds}s`;
  } catch (err) {
    console.error(err);
    toast('build failed: ' + err.message, true);
    $('badge-text').textContent = 'error';
  } finally {
    buildInFlight = false;
    busy(false);
    if (buildQueued) { buildQueued = false; build(); }
  }
}

function applyMeta(meta) {
  const s = meta.stats;
  $('badge-verts').textContent = s.vertices.toLocaleString();
  $('badge-tris').textContent = s.triangles.toLocaleString();
  $('badge-bones').textContent = s.bones;
  $('badge-clips').textContent = s.clips.length;

  const grid = $('stats');
  const rows = [
    ['Height', `${meta.spec.height.toFixed(2)} m`],
    ['Vertices', s.vertices.toLocaleString()],
    ['Triangles', s.triangles.toLocaleString()],
    ['Materials', s.materials],
    ['Bones', `${s.deform_bones} deform / ${s.bones}`],
    ['Texture', s.texture],
    ['GLB size', `${s.glb_kb} kB`],
    ['Bake time', `${meta.seconds}s`],
  ];
  grid.innerHTML = rows.map(([k, v]) => `<div class="stat"><b>${v}</b><span>${k}</span></div>`).join('');

  const files = meta.files;
  $('x-glb').href = files.glb;
  $('x-zip').href = files.zip;
  $('x-tex').href = files.texture;
  $('x-blend').href = files.blender;
  $('x-spec').href = files.spec;
  $('x-glb').setAttribute('download', `colada_${meta.id}.glb`);
  $('x-zip').setAttribute('download', `colada_${meta.id}.zip`);
  $('x-tex').setAttribute('download', `colada_${meta.id}_texture.png`);
  $('x-blend').setAttribute('download', 'build_character_in_blender.py');
  $('x-spec').setAttribute('download', 'character_spec.json');

  buildClips(meta.clips);
  loadModel(files.glb);
}

async function loadModel(url) {
  try {
    // build ids are content addressed, so the url is a safe cache key
    const info = await viewer.loadGLB(url);
    if (!viewer.model) return;
    const existing = viewer.current;
    if (!existing && info.clips.length) viewer.play(info.clips[0].name, { restart: true });
    else if (existing) viewer.play(existing, { restart: false });
    viewer.setSkeleton($('t-bones').classList.contains('on'));
    viewer.setWireframe($('t-wire').classList.contains('on'));
    if (!state.fitted) {                 // frame the character once, then respect the user's camera
      viewer.fitCamera();
      state.fitted = true;
    } else {
      // characters are planted at y=0 and stand `height` tall
      viewer.controls.target.y = (state.spec.height || 1.7) * 0.5;
    }
    syncClipUI();
  } catch (err) {
    console.error(err);
    toast('could not load model', true);
  }
}


function buildClips(clips) {
  const host = $('clips');
  host.innerHTML = '';
  state.clipMeta = clips;
  clips.forEach((c, i) => {
    const b = document.createElement('button');
    b.className = 'clip';
    b.dataset.name = c.name;
    b.innerHTML = `${c.label || c.name} <span style="opacity:.5">${c.duration.toFixed(1)}s</span>`;
    b.onclick = () => selectClip(c.name);
    host.appendChild(b);
  });
  syncClipUI();
}

function selectClip(name) {
  viewer.play(name, { restart: true });
  viewer.loop = state.clipMeta.find((c) => c.name === name)?.loop ?? true;
  $('tr-loop').classList.toggle('on', viewer.loop);
  syncClipUI();
}

function syncClipUI() {
  const cur = viewer.current;
  for (const b of document.querySelectorAll('.clip')) b.classList.toggle('active', b.dataset.name === cur);
  const info = state.clipMeta.find((c) => c.name === cur);
  $('anim-name').textContent = info ? (info.label || cur) : '–';
  $('tr-dur').textContent = info ? `${info.duration.toFixed(2)}s` : '–';
}

function syncTools() {
  const map = {
    't-wire': ['wireframe', (v) => viewer.setWireframe(v)],
    't-bones': ['show_skeleton', (v) => viewer.setSkeleton(v)],
    't-grid': ['show_grid', (v) => { viewer.setGridVisible(v); viewer.setGround(v); }],
    't-rot': ['auto_rotate', (v) => viewer.setAutoRotate(v)],
    't-head': ['head_track', (v) => viewer.setHeadTrack(v)],
  };
  for (const [id, [key, fn]] of Object.entries(map)) {
    const on = !!state.spec[key];
    $(id).classList.toggle('on', on);
    fn(on);
  }
  const theme = state.spec.lighting || 'studio';
  viewer.setTheme(theme);
}

/* ------------------------------------------------------------ transport */

function bindTransport() {
  $('tr-play').onclick = () => {
    viewer.toggle();
    $('tr-play').textContent = viewer.playing ? '⏸' : '▶';
  };
  $('tr-first').onclick = () => { viewer.seek(0); viewer.play(viewer.current, { restart: true }); };
  $('tr-loop').onclick = () => {
    viewer.setLoop(!viewer.loop);
    $('tr-loop').classList.toggle('on', viewer.loop);
  };
  $('tr-speed').onchange = (e) => viewer.setSpeed(parseFloat(e.target.value));
  const scrub = $('tr-scrub');
  let scrubbing = false;
  const wasPlaying = () => viewer.playing;
  scrub.addEventListener('pointerdown', () => { scrubbing = true; viewer.pause(); });
  window.addEventListener('pointerup', () => {
    if (!scrubbing) return;
    scrubbing = false;
    if (wasPlaying()) viewer.resume();
    $('tr-play').textContent = viewer.playing ? '⏸' : '▶';
  });
  scrub.addEventListener('input', () => {
    const d = viewer.duration() || 1;
    viewer.seek((scrub.value / 1000) * d);
    $('tr-time').textContent = viewer.time().toFixed(2) + 's';
  });
  setInterval(() => {
    if (scrubbing) return;
    const d = viewer.duration();
    if (!d) return;
    scrub.value = Math.round((viewer.time() / d) * 1000);
    scrub.style.setProperty('--fill', (scrub.value / 10) + '%');
    $('tr-time').textContent = viewer.time().toFixed(2) + 's';
  }, 90);
}

/* --------------------------------------------------------------- tools */

function bindTools() {
  const toggle = (id, key, after) => {
    $(id).onclick = () => {
      state.spec[key] = !state.spec[key];
      $(id).classList.toggle('on', !!state.spec[key]);
      after && after(state.spec[key]);
    };
  };
  toggle('t-wire', 'wireframe', (v) => viewer.setWireframe(v));
  toggle('t-bones', 'show_skeleton', (v) => viewer.setSkeleton(v));
  toggle('t-grid', 'show_grid', (v) => { viewer.setGridVisible(v); viewer.setGround(v); });
  toggle('t-rot', 'auto_rotate', (v) => viewer.setAutoRotate(v));
  toggle('t-head', 'head_track', (v) => viewer.setHeadTrack(v));
  $('t-fit').onclick = () => viewer.fitCamera();

  $('t-shot').onclick = () => {
    const url = viewer.screenshot();
    const a = document.createElement('a');
    a.href = url;
    a.download = `colada_${state.buildId || 'shot'}.png`;
    a.click();
  };

  $('t-rec').onclick = () => {
    if (state.recorder) {
      state.recorder.stop();
      state.recorder = null;
      $('t-rec').classList.remove('on');
      $('t-rec').textContent = '⏺ Record';
      return;
    }
    const rec = viewer.startRecording((url) => {
      if (!url) return toast('recording not supported here', true);
      const a = document.createElement('a');
      a.href = url;
      a.download = `colada_${viewer.current || 'clip'}.webm`;
      a.click();
    });
    if (!rec) return toast('recording not supported here', true);
    state.recorder = rec;
    state.recordingUntil = Date.now() + 12000;
    $('t-rec').classList.add('on');
    toast('recording 12s of ' + (viewer.current || 'clip') + '…');
    setTimeout(() => { if (state.recorder) $('t-rec').click(); }, 12000);
  };

  $('x-save').onclick = async () => {
    try {
      busy(true, 'saving to library…');
      const meta = await api('/api/library', { spec: state.spec, meta: { label: `${state.spec.hair_style} ${state.spec.top_style}` } });
      toast('saved: ' + meta.id);
      refreshLibrary();
    } catch (e) { toast('save failed', true); } finally { busy(false); }
  };
}

async function refreshLibrary() {
  try {
    const { items } = await api('/api/library');
    const host = $('library');
    $('lib-count').textContent = items.length ? `${items.length}` : '';
    if (!items.length) { host.innerHTML = '<div class="lib-empty">nothing saved yet</div>'; return; }
    host.innerHTML = '';
    for (const it of items.slice(0, 12)) {
      const div = document.createElement('div');
      div.className = 'lib-item';
      div.innerHTML = `<img src="${it.files.portrait}" alt=""><button class="del" title="delete">✕</button>`
        + `<button class="load">load</button>`;
      div.querySelector('.load').onclick = () => {
        pushUndo();
        applySpec(it.spec, { keepClips: true });
        build();
        toast('loaded ' + it.label);
      };
      div.querySelector('.del').onclick = async () => {
        await fetch('/api/library/' + it.id, { method: 'DELETE' });
        refreshLibrary();
      };
      host.appendChild(div);
    }
  } catch (e) { /* ignore */ }
}

/* ------------------------------------------------------------ shortcuts */

function bindKeys() {
  window.addEventListener('keydown', (e) => {
    const tag = (e.target.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'select' || tag === 'textarea') return;
    if (e.ctrlKey || e.metaKey) {
      if (e.key.toLowerCase() === 'z') { e.preventDefault(); e.shiftKey ? redo() : undo(); }
      if (e.key.toLowerCase() === 's') { e.preventDefault(); $('x-glb').click(); }
      return;
    }
    const k = e.key.toLowerCase();
    if (k === 'r') randomize(false);
    else if (k === 't') randomize(true);
    else if (k === ' ') { e.preventDefault(); $('tr-play').click(); }
    else if (k === 'g') $('t-grid').click();
    else if (k === 'b') $('t-bones').click();
    else if (k === 'w') $('t-wire').click();
    else if (k >= '1' && k <= '9') {
      const i = parseInt(k, 10) - 1;
      const c = state.clipMeta[i];
      if (c) selectClip(c.name);
    }
  });
}

async function randomize(tweakOnly) {
  pushUndo();
  try {
    const base = tweakOnly ? state.spec : undefined;
    const r = await api('/api/random', { base });
    applySpec(r.spec, { keepClips: true });
    toast(tweakOnly ? 'tweaked hair + outfit' : 'rolled a fresh character 🎲');
    build();
  } catch (e) { toast('randomize failed', true); }
}

/* --------------------------------------------------------------- boot */

async function boot() {
  busy(true, 'loading engine…');
  const s = await api('/api/schema');
  state.schema = s.schema;
  state.presets = s.presets;
  state.defaults = s.defaults;
  state.clipMeta = s.clips;
  state.spec = state.defaults;

  let saved = null;
  try { saved = JSON.parse(localStorage.getItem('colada.spec') || 'null'); } catch (e) { /* ignore */ }
  if (saved && saved.height) state.spec = { ...state.defaults, ...saved };

  buildPresets();
  refreshUI();
  bindTransport();
  bindTools();
  bindKeys();
  refreshLibrary();

  $('btn-build').onclick = () => build();
  $('btn-random').onclick = () => randomize(false);
  $('btn-random-tweak').onclick = () => randomize(true);
  $('btn-undo').onclick = undo;
  $('btn-redo').onclick = redo;
  $('btn-reset').onclick = () => { pushUndo(); applySpec(s.defaults); build(); };
  $('tr-loop').classList.toggle('on', viewer.loop);

  busy(false);
  await build();
  $('engine-line').textContent += ` · ${state.schema.length} control groups loaded`;
}

boot().catch((e) => {
  console.error(e);
  busy(false);
  toast('startup failed: ' + e.message, true);
});
