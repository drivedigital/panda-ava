/**
 * SpringBones — VRM-style verlet spring bone system for three.js.
 *
 * Each "joint" controls the world-space position of its child bone's head
 * (the bone tip). Chains are simulated after the animation mixer update and
 * written back as world-space rotations, so animation and physics compose.
 *
 * Also includes capsule/sphere colliders bound to body bones, plus a ground
 * plane, so cloth/tail chains collide with the body realistically.
 */
import * as THREE from 'three';

const _v1 = new THREE.Vector3();
const _v2 = new THREE.Vector3();
const _v3 = new THREE.Vector3();
const _q1 = new THREE.Quaternion();
const _q2 = new THREE.Quaternion();
const _m1 = new THREE.Matrix4();

export class CapsuleCollider {
  /** Capsule spanning from boneA head to boneB head (world space), radius r. */
  constructor(boneA, boneB, radius) {
    this.boneA = boneA;
    this.boneB = boneB || null;
    this.radius = radius;
    this.a = new THREE.Vector3();
    this.b = new THREE.Vector3();
  }
  update() {
    this.boneA.getWorldPosition(this.a);
    if (this.boneB) this.boneB.getWorldPosition(this.b);
    else this.b.copy(this.a);
  }
  /** Push `pos` (with radius pr) out of the capsule. Returns true if pushed. */
  pushOut(pos, pr) {
    const ab = _v1.subVectors(this.b, this.a);
    const len2 = ab.lengthSq();
    let t = 0;
    if (len2 > 1e-10) {
      t = THREE.MathUtils.clamp(_v2.subVectors(pos, this.a).dot(ab) / len2, 0, 1);
    }
    const closest = _v3.copy(this.a).addScaledVector(ab, t);
    const delta = _v2.subVectors(pos, closest);
    const dist = delta.length();
    const minDist = this.radius + pr;
    if (dist < minDist) {
      if (dist < 1e-7) delta.set(0, 1, 0);
      else delta.divideScalar(dist);
      pos.copy(closest).addScaledVector(delta, minDist);
      return true;
    }
    return false;
  }
}

export class SpringJoint {
  /**
   * @param {THREE.Bone} bone   bone whose tip (child head) is simulated
   * @param {THREE.Vector3} tipLocal  tip offset in the bone's local space (usually child.position)
   * @param {object} opts { radius, stiffness, damping, gravity, drag }
   */
  constructor(bone, tipLocal, opts) {
    this.bone = bone;
    this.tipLocal = tipLocal.clone();
    this.boneLength = tipLocal.length();
    this.radius = opts.radius ?? 0.02;
    this.stiffness = opts.stiffness ?? 60;
    this.damping = opts.damping ?? 8;
    this.gravity = opts.gravity ?? new THREE.Vector3(0, -9.8, 0);
    this.windResponse = opts.windResponse ?? 1;

    bone.updateWorldMatrix(true, false);
    const tipWorld = this.tipLocal.clone().applyMatrix4(bone.matrixWorld);
    this.pos = tipWorld.clone();
    this.prev = tipWorld.clone();
    this.restWorldQuat = new THREE.Quaternion();
    bone.getWorldQuaternion(this.restWorldQuat);
  }

  /**
   * One verlet sub-step. `animTipWorld` = where the tip would be under pure animation.
   */
  step(dt, animTipWorld, colliders, wind) {
    const pos = this.pos;
    const prev = this.prev;

    // verlet integration with damping
    _v1.subVectors(pos, prev);
    const dampFactor = Math.exp(-this.damping * dt);
    _v1.multiplyScalar(dampFactor);
    prev.copy(pos);

    // spring force toward animated pose
    _v2.subVectors(animTipWorld, pos).multiplyScalar(this.stiffness);
    // gravity + wind
    _v3.copy(this.gravity);
    if (wind) _v3.addScaledVector(wind, this.windResponse);
    _v2.add(_v3);

    pos.add(_v1).addScaledVector(_v2, dt * dt);

    // collisions
    for (const c of colliders) c.pushOut(pos, this.radius);
    if (pos.y < this.radius) pos.y = this.radius; // ground

    // rigid rod constraint: keep tip at boneLength from the bone head
    this.bone.getWorldPosition(_v3);
    _v2.subVectors(pos, _v3);
    const d = _v2.length();
    if (d > 1e-8) {
      pos.copy(_v3).addScaledVector(_v2, this.boneLength / d);
    }
  }

  /** Write simulated tip back as the bone's world rotation. */
  apply() {
    const bone = this.bone;
    bone.getWorldPosition(_v1); // head
    _v2.subVectors(this.pos, _v1).normalize(); // new world dir

    // rest tip dir in world under current parent orientation:
    bone.parent.updateWorldMatrix(true, false);
    _q1.setFromRotationMatrix(_m1.extractRotation(bone.parent.matrixWorld));
    // bone's animated local quat -> "animated" world quat
    _q2.copy(_q1).multiply(bone.quaternion);
    // animated world dir of the tip:
    _v3.copy(this.tipLocal).normalize().applyQuaternion(_q2);

    // swing from animated dir to simulated dir, preserving roll
    _q1.setFromUnitVectors(_v3, _v2);
    _q2.premultiply(_q1); // new world quat

    // convert world -> local
    bone.parent.getWorldQuaternion(_q1);
    bone.quaternion.copy(_q1.invert().multiply(_q2));
    bone.updateWorldMatrix(false, false);
    for (const c of bone.children) c.updateWorldMatrix(false, false);
  }
}

export class SpringBoneSystem {
  /**
   * @param {THREE.Object3D} root model root (for updateWorldMatrix)
   */
  constructor(root) {
    this.root = root;
    this.joints = [];
    this.colliders = [];
    this.wind = new THREE.Vector3();
    this.windAmount = 0;
    this.time = 0;
    this.enabled = true;
  }

  addChain(bones, opts) {
    const out = [];
    for (let i = 0; i < bones.length; i++) {
      const bone = bones[i];
      // tip = first child head, or extrapolated for the last bone in the chain
      let tipLocal;
      const child = bones[i + 1] || bone.children.find((c) => c.isBone);
      if (child) tipLocal = child.position.clone();
      else if (i > 0) tipLocal = bones[i].position.clone().normalize().multiplyScalar(this.joints.at(-1)?.boneLength ?? 0.08);
      else tipLocal = new THREE.Vector3(0, 0.08, 0);
      const j = new SpringJoint(bone, tipLocal, opts);
      this.joints.push(j);
      out.push(j);
    }
    return out;
  }

  addCollider(c) {
    this.colliders.push(c);
  }

  reset() {
    this.root.updateWorldMatrix(true, true);
    for (const j of this.joints) {
      const tipWorld = j.tipLocal.clone().applyMatrix4(j.bone.matrixWorld);
      j.pos.copy(tipWorld);
      j.prev.copy(tipWorld);
    }
  }

  update(dt) {
    if (!this.enabled || this.joints.length === 0) return;
    this.time += dt;
    // gentle procedural wind (gusty sine mix)
    const w = this.windAmount;
    this.wind.set(
      Math.sin(this.time * 0.9) * 0.6 + Math.sin(this.time * 2.3) * 0.25,
      0.12 * Math.sin(this.time * 1.7),
      Math.cos(this.time * 0.7) * 0.6 + Math.sin(this.time * 1.9) * 0.25,
    ).multiplyScalar(w * 6.0);

    this.root.updateWorldMatrix(true, true);
    for (const c of this.colliders) c.update();

    // fixed substeps for stability
    const sub = Math.max(1, Math.min(4, Math.ceil(dt / (1 / 120))));
    const sdt = dt / sub;
    for (let s = 0; s < sub; s++) {
      for (const j of this.joints) {
        const animTip = _v1.copy(j.tipLocal).applyMatrix4(j.bone.matrixWorld);
        j.step(sdt, animTip, this.colliders, this.windAmount > 0 ? this.wind : null);
      }
    }
    for (const j of this.joints) j.apply();
  }
}
