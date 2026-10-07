/**
 * QA harness — active when ?qa=1. Exposes window.QA for the automated
 * screenshot/metric crawler (Playwright) and shows on-screen QA readouts.
 */
import * as THREE from 'three';

export function installQA({ state, playAnimation, scene, camera, renderer, controls }) {
  const params = new URLSearchParams(location.search);
  if (!params.has('qa')) return;

  const qa = {
    ready: true,
    list: () => state.clips.map((c) => ({ name: c.name, duration: c.duration })),
    play: (name) => playAnimation(name, { fade: 0, once: false }),
    /** set normalized time [0..1] on the current action, settle physics */
    seek: (u) => {
      const a = state.current;
      if (!a) return 0;
      const clip = a.getClip();
      a.time = u * clip.duration;
      state.mixer.update(0);
      if (state.springs) {
        state.springs.reset();
        for (let i = 0; i < 90; i++) state.springs.update(1 / 60);
      }
      return clip.duration;
    },
    camera: (azDeg, elDeg, dist = 3.0) => {
      const az = THREE.MathUtils.degToRad(azDeg);
      const el = THREE.MathUtils.degToRad(elDeg);
      const t = new THREE.Vector3(0, 1.0, 0);
      camera.position.set(
        t.x + dist * Math.sin(az) * Math.cos(el),
        t.y + dist * Math.sin(el),
        t.z + dist * Math.cos(az) * Math.cos(el),
      );
      controls.target.copy(t);
      controls.update();
    },
    /** numeric health report for the current pose */
    report: () => {
      const bones = {};
      state.model.traverse((o) => { if (o.isBone) bones[o.name] = o; });
      scene.updateMatrixWorld(true);
      const v = new THREE.Vector3();
      const report = { bones: {}, issues: [] };

      // bone positions + ground penetration
      let minY = Infinity;
      for (const [name, b] of Object.entries(bones)) {
        b.getWorldPosition(v);
        report.bones[name] = [+v.x.toFixed(4), +v.y.toFixed(4), +v.z.toFixed(4)];
        if (!Number.isFinite(v.x + v.y + v.z)) report.issues.push(`NaN bone position: ${name}`);
        if (['foot_l', 'foot_r', 'ball_l', 'ball_r', 'hand_l', 'hand_r'].includes(name)) minY = Math.min(minY, v.y);
      }
      report.groundClearance = +minY.toFixed(4);
      if (minY < -0.03) report.issues.push(`feet/hands below ground: ${minY.toFixed(3)}`);

      // bone length preservation (stretch check) vs rest lengths captured at load
      if (!qa._restLengths) {
        qa._restLengths = {};
        for (const [name, b] of Object.entries(bones)) {
          const c = b.children.find((x) => x.isBone);
          if (c) qa._restLengths[name] = c.position.length();
        }
      }
      let worstStretch = 0; let worstBone = '';
      for (const [name, restLen] of Object.entries(qa._restLengths)) {
        const b = bones[name];
        const c = b.children.find((x) => x.isBone);
        if (!c) continue;
        const now = c.getWorldPosition(v).distanceTo(b.getWorldPosition(new THREE.Vector3()));
        const dev = Math.abs(now - restLen * _worldScale(b)) / Math.max(restLen * _worldScale(b), 1e-6);
        if (dev > worstStretch) { worstStretch = dev; worstBone = name; }
      }
      report.maxBoneStretch = +worstStretch.toFixed(4);
      if (worstStretch > 0.02) report.issues.push(`bone stretch ${(worstStretch * 100).toFixed(1)}% at ${worstBone}`);

      // world bounds of the skinned model (explosion detection)
      const box = new THREE.Box3();
      for (const b of Object.values(bones)) box.expandByPoint(b.getWorldPosition(v));
      report.bounds = [box.min.toArray().map((x) => +x.toFixed(3)), box.max.toArray().map((x) => +x.toFixed(3))];
      const size = box.getSize(v).length();
      if (size > 6) report.issues.push(`suspicious skeleton bounds: ${size.toFixed(2)}m`);

      return report;
    },
  };

  function _worldScale(bone) {
    const s = new THREE.Vector3();
    bone.parent.getWorldScale(s);
    return (s.x + s.y + s.z) / 3;
  }

  window.QA = qa;
}
