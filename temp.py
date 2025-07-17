import os
directory = "logs/"
for scene in os.listdir(directory):
    for filename in os.listdir(os.path.join(directory, scene, "view_front")):
        if filename.endswith("_top.png"):
            # 예: example_top.png
            base_name = filename[:-8]  # '_top.png' 길이 8 만큼 잘라내기
            new_name = f"top_{base_name}.png"
            old_path = os.path.join(directory, scene, "view_front", filename)
            new_path = os.path.join(directory, scene, "view_front", new_name)
            os.rename(old_path, new_path)
            print(f"Renamed {filename} -> {new_name}")