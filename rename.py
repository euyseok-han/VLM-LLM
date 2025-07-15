import os

# 변경하고자 하는 디렉토리 경로
directory = "logs/20250709_chair_chair/view_top"
new_directory = "logs/20250709_chair_chair/view_front"

# 디렉토리 내 모든 파일 순회
for filename in os.listdir(directory):
    if filename.endswith(".png"):
        old_path = os.path.join(directory, filename)
        
        # 확장자 제거하고 _top 붙이기
        name_only = filename[:-4]  # removes ".png"
        new_filename = f"{name_only}_top.png"
        new_path = os.path.join(new_directory, new_filename)

        # 이름 변경
        os.rename(old_path, new_path)