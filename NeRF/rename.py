import os
import subprocess
from ns_reconstruction import main as ns_reconstruction
from feature_fusion import main as feature_fusion
from material_proposal import main as material_proposal
# 변경하고자 하는 디렉토리 경로
directory = "logs/"
env = os.environ.copy()
env["QT_QPA_PLATFORM"] = "offscreen"
def main():

    for scene in os.listdir(directory):
        input_dir = os.path.join(directory, scene, "view_front")
        os.makedirs(os.path.join("/local_data_2/urp25su_hanuiseok/nerf/scenes", scene), exist_ok=True)
        output_dir = os.path.join("/local_data_2/urp25su_hanuiseok/nerf/scenes", scene)
        print("#######################################")
        print("process scene:", scene)
        subprocess.run([
            'ns-process-data', 'images',
            '--data', input_dir,
            '--output-dir', output_dir,
            "--no-gpu",
            ], env=env)
        
if __name__ == "__main__":
    print("Starting ns-process-data...")
    main()
    print("Processing complete. Starting NeRF reconstruction and feature fusion...")
    ns_reconstruction()
    print("NeRF reconstruction complete. Starting feature fusion...")
    feature_fusion()
    material_proposal()
    # 디렉토리 내 모든 파일 순회
    
    # ns-process-data images --data logs/20250709_chair_chair/view_front --output-dir logs/20250709_chair_chair/
