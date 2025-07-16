import os
import subprocess

# 변경하고자 하는 디렉토리 경로
directory = "logs/"
env = os.environ.copy()
env["QT_QPA_PLATFORM"] = "offscreen"
# 디렉토리 내 모든 파일 순회
for dir in os.listdir(directory):
    for filename in os.listdir(os.path.join(directory, dir, "view_top")):
        if filename.endswith(".png"):
            old_path = os.path.join(directory, dir, "view_top", filename)

        # 확장자 제거하고 _top 붙이기
        name_only = filename[:-4]  # removes ".png"
        new_filename = f"{name_only}_top.png"
        new_path = os.path.join(directory, dir, "view_front", new_filename )

        # 이름 변경
        os.rename(old_path, new_path)
    

for dir in os.listdir(directory):
    input_dir = os.path.join(directory, dir, "view_front")
    output_dir = os.path.join(directory, dir, "ns")
    subprocess.run([
        'ns-process-data', 'images',
        '--data', input_dir,
        '--output-dir', output_dir
        ], env=env)
# ns-process-data images --data logs/20250709_chair_chair/view_front --output-dir logs/20250709_chair_chair/ns
