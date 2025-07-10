import os
import requests
import objaverse
import random
import multiprocessing
import shutil
import trimesh

processes = multiprocessing.cpu_count()

uids = objaverse.load_uids()
random_object_uids = random.sample(uids, 2)

local_paths = objaverse.load_objects(
    uids=random_object_uids,
    download_processes=processes
)

output_root = "./data"

for uid, glb_path in local_paths.items():
    try:
        print(f"Processing {uid}: {glb_path}")
        
        # 출력 폴더 생성
        export_dir = os.path.join(output_root, uid)
        os.makedirs(export_dir, exist_ok=True)
        
        # GLB 로드
        mesh = trimesh.load(glb_path, force='mesh')
        
        # OBJ로 내보내기
        obj_export_path = os.path.join(export_dir, "mesh.obj")
        mesh.export(obj_export_path)

        print(f"✅ Converted and saved to {export_dir}")

    except Exception as e:
        print(f"❌ Failed to process {uid}: {e}")