// Extract FBX skeleton + animation curves to JSON using three.js FBXLoader.
// Usage: node fbx_extract.mjs <in.fbx> <out.json>
import { readFileSync, writeFileSync } from 'node:fs';
import { FBXLoader } from 'three/examples/jsm/loaders/FBXLoader.js';

globalThis.window = globalThis.window || {
  innerWidth: 1920,
  innerHeight: 1080,
  URL: { createObjectURL: () => 'blob:stub' },
};

const [, , inPath, outPath] = process.argv;
const buf = readFileSync(inPath);
const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
const loader = new FBXLoader();
const group = loader.parse(ab, '');

const bones = [];
const nodes = [];
group.traverse((o) => {
  if (o.isBone || o.children.some((c) => c.isBone)) {
    const rec = {
      name: o.name,
      type: o.isBone ? 'bone' : (o.isMesh ? 'mesh' : o.type),
      parent: o.parent ? o.parent.name : null,
      pos: [o.position.x, o.position.y, o.position.z],
      quat: [o.quaternion.x, o.quaternion.y, o.quaternion.z, o.quaternion.w],
      scale: [o.scale.x, o.scale.y, o.scale.z],
    };
    nodes.push(rec);
    if (o.isBone) bones.push(rec);
  }
});

const clips = (group.animations || []).map((clip) => ({
  name: clip.name,
  duration: clip.duration,
  tracks: clip.tracks.map((t) => {
    const dot = t.name.lastIndexOf('.');
    const bone = t.name.slice(0, dot);
    const path = t.name.slice(dot + 1);
    const stride = path === 'position' || path === 'scale' ? 3 : 4;
    const values = [];
    for (let i = 0; i < t.values.length; i += stride) values.push(t.values.slice(i, i + stride));
    return { bone, path, times: Array.from(t.times), values };
  }),
}));

const out = { source: inPath.split('/').pop(), root: group.name, nodes, bones, clips };
writeFileSync(outPath, JSON.stringify(out));
console.log(`${inPath.split('/').pop()}: ${bones.length} bones, ${clips.length} clips ` +
  `(${(clips[0]?.tracks.length ?? 0)} tracks, ${(clips[0]?.duration ?? 0).toFixed(2)}s)`);
console.log('non-bone nodes:', nodes.filter((n) => n.type !== 'bone').map((n) => `${n.name}<${n.type}> parent=${n.parent}`).join(', '));
