"""Blender headless: import one FBX animation, export it as GLB (armature + anim only).

Usage:
  Blender -b --factory-startup -noaudio --python fbx2glb.py -- <src.fbx> <dst.glb>
"""
import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
src, dst = argv[0], argv[1]

# wipe default scene
for ob in list(bpy.data.objects):
    bpy.data.objects.remove(ob, do_unlink=True)

bpy.ops.import_scene.fbx(filepath=src)

# drop meshes (stickman lines etc.) - keep only the armature + its action
for ob in list(bpy.data.objects):
    if ob.type != 'ARMATURE':
        bpy.data.objects.remove(ob, do_unlink=True)

arm = [ob for ob in bpy.data.objects if ob.type == 'ARMATURE'][0]
# ensure the action is active so the exporter picks it up
if bpy.data.actions:
    arm.animation_data_create()
    arm.animation_data.action = bpy.data.actions[0]

bpy.ops.export_scene.gltf(
    filepath=dst,
    export_format='GLB',
    export_animations=True,
    export_animation_mode='ACTIONS',
    export_force_sampling=True,
    export_optimize_animation_size=True,
    export_skins=False,
    export_yup=True,
    export_apply=False,
    export_image_format='NONE',
)
print("FBX2GLB DONE", dst)
