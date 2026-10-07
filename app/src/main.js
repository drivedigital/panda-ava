import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { SpringBoneSystem, CapsuleCollider } from './springbone.js';
import { installQA } from './qa.js';

const $ = (id) => document.getElementById(id);
const errbox = $('errbox');
function showError(msg) {
  errbox.style.display = 'block';
  errbox.textContent += msg + '\n';
  console.error(msg);
}

// ---------------------------------------------------------------- renderer
const stage = $('stage');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(stage.clientWidth, stage.clientHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
stage.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x14161c);
scene.fog = new THREE.Fog(0x14161c, 9, 26);

const camera = new THREE.PerspectiveCamera(38, stage.clientWidth / stage.clientHeight, 0.05, 100);
camera.position.set(2.5, 1.75, 3.1);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 1.0, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 0.9;
controls.maxDistance = 12;
controls.maxPolarAngle = Math.PI * 0.52;

// environment for PBR
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
scene.environmentIntensity = 0.85;

// key light + fill
const key = new THREE.DirectionalLight(0xfff1e0, 2.4);
key.position.set(3.2, 5.2, 2.6);
key.castShadow = true;
key.shadow.mapSize.set(2048, 2048);
key.shadow.camera.left = -3.2; key.shadow.camera.right = 3.2;
key.shadow.camera.top = 4.2; key.shadow.camera.bottom = -2.2;
key.shadow.camera.near = 0.5; key.shadow.camera.far = 14;
key.shadow.bias = -0.00035;
key.shadow.normalBias = 0.02;
scene.add(key);
const rim = new THREE.DirectionalLight(0x8fb4ff, 1.1);
rim.position.set(-3.5, 3.0, -3.0);
scene.add(rim);

// ground
const ground = new THREE.Mesh(
  new THREE.CircleGeometry(14, 64),
  new THREE.MeshStandardMaterial({ color: 0x1b1e26, roughness: 0.95, metalness: 0 }),
);
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);
const grid = new THREE.GridHelper(14, 28, 0x2e3342, 0x232733);
grid.position.y = 0.002;
scene.add(grid);

// backdrop disc
const disc = new THREE.Mesh(
  new THREE.RingGeometry(2.05, 2.1, 96),
  new THREE.MeshBasicMaterial({ color: 0x3a2a1e, side: THREE.DoubleSide, transparent: true, opacity: 0.9 }),
);
disc.rotation.x = -Math.PI / 2;
disc.position.y = 0.004;
scene.add(disc);

// ---------------------------------------------------------------- state
const state = {
  mixer: null,
  actions: new Map(),   // name -> AnimationAction
  current: null,
  currentName: '',
  idleName: 'Muay Thai · Guard Stance',
  speed: 1,
  physics: true,
  debug: false,
  turntable: false,
  model: null,
  springs: null,
  debugHelpers: [],
  clips: [],
};

const LOOPING = /Guard Stance|Guard Position|CrouchIdle|Idle|Draw Stance|Walk|Run|T-Pose/i;

function actionFor(name) {
  return state.actions.get(name);
}

function playAnimation(name, { fade = 0.28, once } = {}) {
  const next = actionFor(name);
  if (!next) return false;
  const prev = state.current;
  const loop = once ?? !LOOPING.test(name);
  next.reset();
  next.setLoop(loop ? THREE.LoopOnce : THREE.LoopRepeat, Infinity);
  next.clampWhenFinished = true;
  next.enabled = true;
  next.timeScale = 1;
  if (prev && prev !== next) {
    next.crossFadeFrom(prev, fade, true);
  }
  next.play();
  state.current = next;
  state.currentName = name;
  $('now-playing').textContent = name.replace(' · ', ' — ');
  document.querySelectorAll('.anim-btn').forEach((b) => b.classList.toggle('active', b.dataset.name === name));
  return true;
}

// auto-return to idle after one-shot clips
function onFinished(e) {
  const finishedName = [...state.actions.entries()].find(([, a]) => a === e.action)?.[0];
  if (!finishedName) return;
  if (finishedName !== state.idleName && !LOOPING.test(finishedName)) {
    playAnimation(state.idleName, { fade: 0.4, once: false });
  }
}

// ---------------------------------------------------------------- model load
const loader = new GLTFLoader();
loader.load(
  'tiger.glb',
  (gltf) => {
    const model = gltf.scene;
    model.traverse((o) => {
      if (o.isMesh || o.isSkinnedMesh) {
        o.castShadow = true;
        o.receiveShadow = false;
        o.frustumCulled = false;
        if (o.material) {
          o.material.roughness = Math.min(0.95, o.material.roughness ?? 0.9);
          // authored metallicFactor=1.0 looks too dark w/o strong env; clamp
          o.material.metalness = Math.min(0.35, o.material.metalness ?? 0);
          o.material.needsUpdate = true;
        }
      }
    });
    scene.add(model);
    state.model = model;

    state.mixer = new THREE.AnimationMixer(model);
    state.mixer.addEventListener('finished', onFinished);
    state.clips = gltf.animations;
    for (const clip of gltf.animations) state.actions.set(clip.name, state.mixer.clipAction(clip));

    setupPhysics(model);
    buildUI(gltf.animations);
    setupDebug(model);

    state.springs.reset();
    playAnimation(state.idleName, { once: false });
    $('loading').classList.add('done');

    installQA({ state, playAnimation, scene, camera, renderer, controls });
  },
  (ev) => {
    if (ev.total) $('load-bar').style.width = `${Math.round((ev.loaded / ev.total) * 100)}%`;
  },
  (err) => showError('Failed to load tiger.glb: ' + (err?.message || err)),
);

// ---------------------------------------------------------------- physics
function setupPhysics(model) {
  const bones = {};
  model.traverse((o) => { if (o.isBone) bones[o.name] = o; });
  const sbs = new SpringBoneSystem(model);

  const chain = (names, opts) => {
    const bs = names.map((n) => bones[n]).filter(Boolean);
    if (bs.length) sbs.addChain(bs, opts);
  };

  // tail: heavy, muscular — slow spring, low gravity effect (it holds its shape)
  chain(['tail_01', 'tail_02', 'tail_03', 'tail_04', 'tail_05'],
    { radius: 0.028, stiffness: 26, damping: 5.5, gravity: new THREE.Vector3(0, -3.2, 0), windResponse: 0.25 });
  // front sash (cloth belt ends): light, flappy
  chain(['sash_front_01', 'sash_front_02', 'sash_front_03'],
    { radius: 0.02, stiffness: 60, damping: 6.5, gravity: new THREE.Vector3(0, -9.8, 0), windResponse: 1.4 });
  // back cloak panels
  chain(['cloak_back_01', 'cloak_back_02', 'cloak_back_03'],
    { radius: 0.024, stiffness: 42, damping: 6, gravity: new THREE.Vector3(0, -9.8, 0), windResponse: 1.0 });
  // side skirt panels: stiffer cloth
  chain(['skirt_left_01'], { radius: 0.024, stiffness: 90, damping: 9, gravity: new THREE.Vector3(0, -9.8, 0), windResponse: 0.8 });
  chain(['skirt_right_01'], { radius: 0.024, stiffness: 90, damping: 9, gravity: new THREE.Vector3(0, -9.8, 0), windResponse: 0.8 });

  // body colliders — capsule per limb segment (anatomy-aware)
  const cap = (a, b, r) => { if (bones[a]) sbs.addCollider(new CapsuleCollider(bones[a], bones[b], r)); };
  cap('pelvis', null, 0.20);                       // hips sphere-ish
  cap('pelvis', 'spine_03', 0.185);                // torso
  cap('thigh_l', 'calf_l', 0.145);
  cap('thigh_r', 'calf_r', 0.145);
  cap('calf_l', 'foot_l', 0.105);
  cap('calf_r', 'foot_r', 0.105);
  cap('upperarm_l', 'lowerarm_l', 0.085);
  cap('upperarm_r', 'lowerarm_r', 0.085);

  state.springs = sbs;
}

// ---------------------------------------------------------------- debug view
function setupDebug(model) {
  const bones = [];
  model.traverse((o) => { if (o.isBone) bones.push(o); });
  const skel = new THREE.SkeletonHelper(model);
  skel.visible = false;
  scene.add(skel);
  state.debugHelpers.push(skel);

  // collider debug meshes
  const colliderGroup = new THREE.Group();
  colliderGroup.visible = false;
  scene.add(colliderGroup);
  state.colliderGroup = colliderGroup;
}

function refreshColliderDebug() {
  const g = state.colliderGroup;
  g.clear();
  if (!state.springs) return;
  for (const c of state.springs.colliders) {
    const len = Math.max(c.a.distanceTo(c.b), 0.001);
    const geo = new THREE.CapsuleGeometry(c.radius, len, 6, 12);
    const mat = new THREE.MeshBasicMaterial({ color: 0x27e0a4, wireframe: true, transparent: true, opacity: 0.5 });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.copy(c.a).lerp(c.b, 0.5);
    mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3().subVectors(c.b, c.a).normalize());
    mesh.userData.collider = c;
    g.add(mesh);
  }
  for (const j of state.springs.joints) {
    const s = new THREE.Mesh(
      new THREE.SphereGeometry(j.radius, 8, 8),
      new THREE.MeshBasicMaterial({ color: 0xff7a2f, wireframe: true, transparent: true, opacity: 0.6 }),
    );
    s.userData.joint = j;
    g.add(s);
  }
}

function updateDebug() {
  const on = state.debug;
  for (const h of state.debugHelpers) h.visible = on;
  if (state.colliderGroup) {
    state.colliderGroup.visible = on;
    if (on) {
      if (state.colliderGroup.children.length === 0) refreshColliderDebug();
      for (const m of state.colliderGroup.children) {
        if (m.userData.collider) {
          const c = m.userData.collider;
          m.position.copy(c.a).lerp(c.b, 0.5);
          const len = Math.max(c.a.distanceTo(c.b), 0.001);
          m.scale.set(1, len / (m.geometry.parameters?.length || len), 1);
          m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3().subVectors(c.b, c.a).normalize());
        } else if (m.userData.joint) {
          m.position.copy(m.userData.joint.pos);
        }
      }
    }
  }
}

// ---------------------------------------------------------------- UI
function categoryOf(name) {
  if (name.startsWith('Unreal')) return 'Bonus (pre-baked)';
  if (name.startsWith('Reference')) return 'Reference';
  return name.split(' · ')[0];
}

function buildUI(clips) {
  const list = $('anim-list');
  list.innerHTML = '';
  const cats = new Map();
  for (const clip of clips) {
    const cat = categoryOf(clip.name);
    if (!cats.has(cat)) cats.set(cat, []);
    cats.get(cat).push(clip);
  }
  const order = ['Taekwondo', 'Muay Thai', 'Sword', 'Staff', 'Spear', 'Wrestling', 'Judo', 'Bonus (pre-baked)', 'Reference'];
  const sorted = [...cats.entries()].sort((a, b) => order.indexOf(a[0]) - order.indexOf(b[0]));
  for (const [cat, catClips] of sorted) {
    const div = document.createElement('div');
    div.className = 'cat';
    if (cat === 'Reference') div.classList.add('closed');
    div.innerHTML = `<div class="cat-head"><span class="chev">▼</span> ${cat} <span style="margin-left:auto">${catClips.length}</span></div>`;
    const body = document.createElement('div');
    body.className = 'cat-body';
    for (const clip of catClips) {
      const btn = document.createElement('button');
      btn.className = 'anim-btn';
      btn.dataset.name = clip.name;
      const short = clip.name.includes(' · ') ? clip.name.split(' · ').slice(1).join(' · ') : clip.name;
      btn.innerHTML = `${short}<span class="dur">${clip.duration.toFixed(1)}s</span>`;
      btn.onclick = () => playAnimation(clip.name);
      body.appendChild(btn);
    }
    div.appendChild(body);
    div.querySelector('.cat-head').onclick = () => div.classList.toggle('closed');
    list.appendChild(div);
  }
}

// toolbar wiring
$('speed').addEventListener('input', (e) => {
  state.speed = parseFloat(e.target.value);
  $('speed-val').textContent = state.speed.toFixed(2).replace(/0$/, '') + '×';
});
$('physics-on').addEventListener('change', (e) => {
  state.physics = e.target.checked;
  if (state.springs) state.springs.enabled = state.physics;
  if (state.physics && state.springs) state.springs.reset();
});
$('wind').addEventListener('input', (e) => { if (state.springs) state.springs.windAmount = parseFloat(e.target.value); });
$('debug-on').addEventListener('change', (e) => { state.debug = e.target.checked; });
$('turntable').addEventListener('change', (e) => { state.turntable = e.target.checked; });
$('menu-btn').addEventListener('click', () => $('sidebar').classList.toggle('open'));

// ---------------------------------------------------------------- AI command box
async function askAI(prompt) {
  const names = state.clips.map((c) => c.name);
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, animations: names }),
    });
    if (!res.ok) throw new Error(`api ${res.status}`);
    const data = await res.json();
    return data.animation;
  } catch {
    return keywordPick(prompt, names);
  }
}

function keywordPick(prompt, names) {
  const toks = prompt.toLowerCase().replace(/[^a-z0-9 ]/g, ' ').split(/\s+/).filter(Boolean);
  let best = null; let bestScore = 0;
  for (const name of names) {
    const words = name.toLowerCase().replace(/[^a-z0-9 ]/g, ' ').split(/\s+/);
    let score = 0;
    for (const t of toks) if (words.some((w) => w.startsWith(t) || t.startsWith(w))) score += 1 + Math.min(t.length, 5) * 0.1;
    if (score > bestScore) { bestScore = score; best = name; }
  }
  return bestScore > 0.5 ? best : null;
}

async function handleAI() {
  const input = $('ai-input');
  const prompt = input.value.trim();
  if (!prompt) return;
  $('ai-send').disabled = true;
  const name = await askAI(prompt);
  $('ai-send').disabled = false;
  if (name && playAnimation(name)) {
    input.value = '';
    $('ai-hint').textContent = `→ ${name}`;
  } else {
    $('ai-hint').textContent = 'No matching move — try “side kick”, “sword combo”, “hip throw”…';
  }
}
$('ai-send').addEventListener('click', handleAI);
$('ai-input').addEventListener('keydown', (e) => { if (e.key === 'Enter') handleAI(); });

// ---------------------------------------------------------------- loop
const clock = new THREE.Clock();
function tick() {
  requestAnimationFrame(tick);
  const dt = Math.min(clock.getDelta(), 0.05);
  if (state.mixer) state.mixer.update(dt * state.speed);
  if (state.springs && state.physics) state.springs.update(dt * state.speed);
  if (state.turntable) {
    const a = dt * 0.5;
    const x = camera.position.x, z = camera.position.z;
    camera.position.x = x * Math.cos(a) - z * Math.sin(a);
    camera.position.z = x * Math.sin(a) + z * Math.cos(a);
  }
  controls.update();
  updateDebug();
  renderer.render(scene, camera);
}
tick();

window.addEventListener('resize', () => {
  camera.aspect = stage.clientWidth / stage.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(stage.clientWidth, stage.clientHeight);
});
