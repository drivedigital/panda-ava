"""Render specific animation frames of a GLB for visual QA.

Usage: Blender -b --factory-startup -noaudio --python render_anim.py -- <glb> <out_prefix> <anim_name> <time1,time2,...>
"""
import math
import sys

import bpy
import mathutils
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:]
glb, out_prefix, anim_name, times_s = argv[0], argv[1], argv[2], argv[3]
times = [float(x) for x in times_s.split(",")]

for ob in list(bpy.data.objects):
    bpy.data.objects.remove(ob, do_unlink=True)

bpy.context.scene.render.fps = 30
bpy.ops.import_scene.gltf(filepath=glb)

arm = [ob for ob in bpy.data.objects if ob.type == 'ARMATURE'][0]
act = None
for a in bpy.data.actions:
    if a.name == anim_name or a.name.startswith(anim_name):
        act = a
        break
if act is None:
    print("ACTIONS:", [a.name for a in bpy.data.actions])
    raise SystemExit("action not found: " + anim_name)
arm.animation_data_create()
arm.animation_data.action = act

scene = bpy.context.scene
try:
    scene.render.engine = 'BLENDER_EEVEE_NEXT'
except Exception:
    scene.render.engine = 'BLENDER_EEVEE'
scene.render.resolution_x = 420
scene.render.resolution_y = 520
world = bpy.data.worlds.new("W")
scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.16, 0.17, 0.2, 1)
sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
sun.data.energy = 4
sun.rotation_euler = (0.9, 0.2, 0.5)
scene.collection.objects.link(sun)
sun2 = bpy.data.objects.new("Sun2", bpy.data.lights.new("Sun2", "SUN"))
sun2.data.energy = 2
sun2.rotation_euler = (-0.4, -0.6, -2.2)
scene.collection.objects.link(sun2)
cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam"))
scene.collection.objects.link(cam)
scene.camera = cam
cam.data.clip_end = 100

def render_at(frame, tag):
    scene.frame_set(frame)
    dg = bpy.context.evaluated_depsgraph_get()
    pts = []
    for ob in bpy.data.objects:
        if ob.type == 'MESH':
            ev = ob.evaluated_get(dg)
            mesh = ev.to_mesh()
            mw = ev.matrix_world
            for v in mesh.vertices:
                co = mw @ v.co
                pts.append((co.x, co.y, co.z))
            ev.to_mesh_clear()
    pts = np.array(pts)
    lo, hi = pts.min(0), pts.max(0)
    c = (lo + hi) / 2
    r = float(np.linalg.norm(hi - lo))
    views = {"front": (0, 6), "side": (90, 6), "threeq": (35, 14)}
    for vtag, (az, el) in views.items():
        azr, elr = math.radians(az), math.radians(el)
        d = r * 1.35
        eye = c + d * np.array([math.sin(azr) * math.cos(elr), -math.cos(azr) * math.cos(elr), math.sin(elr)])
        cam.location = eye
        dirv = mathutils.Vector(c) - mathutils.Vector(eye)
        cam.rotation_euler = dirv.to_track_quat('-Z', 'Y').to_euler()
        scene.render.filepath = f"{out_prefix}_{tag}_{vtag}.png"
        bpy.ops.render.render(write_still=True)

for i, t in enumerate(times):
    render_at(int(round(t * 30)), f"t{i}")
print("RENDER DONE")
