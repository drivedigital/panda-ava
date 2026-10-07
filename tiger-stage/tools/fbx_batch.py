"""Blender headless: convert every FBX in a directory to GLB (armature + animation only).

Usage:
  Blender -b --factory-startup -noaudio --python fbx_batch.py -- <fbx_dir> <out_dir>
"""
import os
import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
src_dir, out_dir = argv[0], argv[1]
os.makedirs(out_dir, exist_ok=True)

fbx_files = sorted(f for f in os.listdir(src_dir) if f.lower().endswith(".fbx"))
print(f"CONVERTING {len(fbx_files)} files")

for fbx in fbx_files:
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    for a in list(bpy.data.actions):
        bpy.data.actions.remove(a)
    for arm_d in list(bpy.data.armatures):
        bpy.data.armatures.remove(arm_d)

    src = os.path.join(src_dir, fbx)
    dst = os.path.join(out_dir, os.path.splitext(fbx)[0] + ".glb")
    try:
        bpy.ops.import_scene.fbx(filepath=src)
        for ob in list(bpy.data.objects):
            if ob.type != 'ARMATURE':
                bpy.data.objects.remove(ob, do_unlink=True)
        arm = [ob for ob in bpy.data.objects if ob.type == 'ARMATURE'][0]
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
        print("OK", fbx)
    except Exception as e:  # noqa: BLE001
        print("FAIL", fbx, repr(e))

print("BATCH DONE")
