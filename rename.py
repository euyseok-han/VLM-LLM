import os
import subprocess
from NeRF.ns_reconstruction import main as ns_reconstruction
from NeRF.feature_fusion import main as feature_fusion
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
            ], env=env)
        
if __name__ == "__main__":
    main()
    ns_reconstruction()
    feature_fusion()

    # 디렉토리 내 모든 파일 순회
    
    # ns-process-data images --data logs/20250709_chair_chair/view_front --output-dir logs/20250709_chair_chair/
