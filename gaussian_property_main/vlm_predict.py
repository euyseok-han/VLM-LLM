import os
import argparse
from gaussian_property_main.utils.vlm_utils import GeminiFlash, get_image_files, Qwen, GPT4V, Gemini
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import open3d as o3d
import matplotlib as mpl
import re

material_property = "Young’s Modulus(GPa)"

def clean_numeric_string(s):
    # 숫자와 점(.)만 남기고 모두 제거
    cleaned = re.sub(r'[^0-9.]', '', s)
    # 문자열 끝에 있는 점 제거
    cleaned = re.sub(r'\.+$', '', cleaned)
    return cleaned


def query_vlm(base_path, case_name, nir=False, vlm_type = "gpt4"):
    
    material_list = "wood, metal, plastic, glass, fabric, foam, food, ceramic, paper, leather, aluminum, brass, bronze, copper, steel, stainless steel, iron, cast iron, titanium, zinc, lead, gold, silver, platinum, nickel, chrome, magnesium, tin, carbon fiber, fiberglass, acrylic, polyethylene, polypropylene, polystyrene, polycarbonate, polyvinyl chloride, nylon, rubber, silicone, latex, plywood, MDF, particle board, cork, bamboo, concrete, cement, asphalt, brick, clay, porcelain, terracotta, marble, granite, limestone, sandstone, quartz, tempered glass, frosted glass, mirror, cardboard, suede, denim, cotton, wool, silk, linen, polyester, felt, velvet, mesh, canvas, fur, straw, jute, carbon, graphite, resin, wax, ice, snow, sand, soil, mud, chalk, plaster, gypsum, sponge, tar, vinyl, PVC, Teflon, Kevlar, quartzite, basalt, lava rock, obsidian, bone, horn, shell, pearl"
    material_list = material_list.split(", ")
    material_library = "{" + ", ".join(material_list) + "}"
    if nir:
        nir_library = """{
                    wood (light/dry): 20-30
                    metal (generic polished): 60-95
                    plastic (white): 15-20
                    plastic (black): 1-5
                    glass (clear): 4
                    fabric (white): 20-30
                    fabric (black): 5-15
                    foam (white): 40-60
                    food (generic): 40-60
                    ceramic (unglazed): 30-40
                    paper (white): 50-70
                    leather (white): 20-30
                    leather (black): 5-15
                    aluminum: 70
                    brass: 35
                    bronze: 30
                    copper: 40
                    steel: 15
                    stainless steel: 60
                    iron: 30
                    cast iron: 25
                    titanium: 30
                    zinc: 60
                    lead: 50
                    gold: 90
                    silver: 95
                    platinum: 65
                    nickel: 45
                    chrome: 65
                    tin: 70
                    carbon fiber: 5-10
                    fiberglass: 40-60
                    acrylic (clear): 4
                    polyethylene (white): 40-50
                    polyethylene (black): 1-5
                    polypropylene (white): 40-50
                    polypropylene (black): 1-5
                    polystyrene (white): 40-50
                    polystyrene (black): 1-5
                    polycarbonate (clear): 4
                    polyvinyl chloride / PVC (white): 15-20
                    polyvinyl chloride / PVC (black): 1-5
                    nylon (white): 20-30
                    nylon (black): 5-10
                    rubber (black): 5
                    silicone (white): 20-30
                    silicone (black): 5-10
                    latex (white): 20-30
                    latex (black): 5-10
                    plywood: 20-30
                    MDF: 15-25
                    cork: 25-35
                    bamboo: 25-35
                    concrete (dry): 20
                    cement (dry): 20
                    asphalt: 5
                    brick (red): 15
                    clay: 20-30
                    porcelain: 60-70
                    marble (white): 50-60
                    granite: 15-25
                    limestone: 40
                    sandstone: 20-30
                    quartz (clear): 4
                    tempered glass (clear): 4
                    frosted glass: 10-20
                    mirror (silvered): 90-95
                    cardboard: 30
                    suede (light): 20-30
                    suede (dark): 5-15
                    denim (blue): 10-20
                    cotton (white): 20-30
                    cotton (black): 5-15
                    wool (white): 20-30
                    wool (black): 5-15
                    silk (white): 20-30
                    silk (black): 5-15
                    linen (white): 20-30
                    linen (black): 5-15
                    polyester (white): 20-30
                    polyester (black): 5-15
                    felt (white): 20-30
                    felt (black): 5-15
                    velvet (light): 20-30
                    velvet (dark): 5-15
                    mesh (white synthetic): 20-30
                    mesh (black synthetic): 5-15
                    canvas (white): 20-30
                    canvas (black): 5-15
                    fur (light): 20-30
                    fur (dark): 5-15
                    straw: 25-35
                    jute: 25-35
                    carbon: 5-10
                    graphite: 5-10
                    resin (clear): 4
                    wax (white): 40-50
                    wax (colored): 10-30
                    ice: 50
                    snow (fresh): 80
                    sand (dry): 30-40
                    soil (dry): 20-30
                    soil (wet): 5-10
                    mud (wet): 5-10
                    chalk: 70-80
                    plaster: 60-70
                    sponge (natural, dry): 40-50
                    tar: 5
                    vinyl (white): 20-30
                    vinyl (black): 5-10
                    Teflon (white): 20-30
                    Teflon (black): 5-10
                    Kevlar (natural): 25-35
                    paint (white): 30-40
                    paint (black): 5-15
                    paint (metallic): 40-60}
                """
        prompt = f"""You are given a set of four images of an object:
1. Original Image
2. Mask Overlay in Blue (blue mask highlights the target part; note: the blue color is not the actual color of the part, only a visual marker to show the region)
3. Part Image
4. NIR Image

    Task:
    - Briefly caption the blue-masked part.
    - Identify the main material of the masked part using **NIR characteristics**.
    - Estimate the part's NIR reflectance (0-100) using the provided NIR material library: {nir_library}.
    - Provide {material_property} value of the part as low-high (one decimal place, no units or punctuation).

    Rules:
    - NIR: higher reflectance = brighter.
    - **Use NIR as the reference for material identification.**
    - Focus on the blue-masked part when identifying the material.
    - Use consistent {material_property} values for the same material across responses.
    Format:
    caption, material, NIR reflectance, {material_property}
    Only output in this format; do not add extra text."""
        
    else:
        prompt = f"""You are given three images of an object:
                1. Original Image
                2. Mask Overlay in Blue (blue mask highlights the target part; note: the blue color is not the actual color of the part, only a visual marker to show the region)
                3. Part Image

                Task:
                - Caption the blue-masked part.
                - Identify its main material (choose from {material_library}).
                - Estimate {material_property} as low-high (one decimal, no units).

                Rules:
                - Output only the pair, no extra text.
                - Property value format: 0.1-0.2 (low-high, one decimal).
                - Use consistent {material_property} values for the same material across responses.

                Format:
                caption, material, {material_property}
                Only output in this format; do not add extra text."""



    output_file = 'vlm_result.txt'
    results_file_path = os.path.join(base_path, 'scene', output_file)
    os.makedirs(os.path.join(base_path, case_name, "property_seg"), exist_ok=True)


    if not vlm_type:
        property_seg = []
        with open(results_file_path, 'r') as file:
             lines = [line.strip() for line in file]
        mask = None
        for line in lines:
            line = line.split(",")
            file = line[0]
            i = int(file[-9:-7])
            j = int(file[-6:-4])
            if not j:
                if mask is not None:
                    property_seg.append(mask)
                mask_path = os.path.join(base_path, case_name, "seg", str(i).zfill(3) + "_s.npy")
                mask = np.load(mask_path).astype(np.float32)
            property = line[-1] or line[-2]
            property_right = float(clean_numeric_string(property.split('-')[-1]))
            property = property_right
            mask[mask==j] = property    

        if mask is not None:
            property_seg.append(mask)
             
        save_path = os.path.join(base_path, case_name, "property_seg", "property_seg.npy")
        stacked = np.stack(property_seg, axis=0).astype(np.float32)  # 예시
        np.save(save_path, stacked)

        return
    if os.path.exists(results_file_path):
        os.remove(results_file_path)
        ...
    # results_file_path = os.path.join(base_path, case_name, output_file)
    # os.makedirs(os.path.dirname(results_file_path), exist_ok=True)
    input_image_path = os.path.join(base_path, case_name, "gpt_input")

    
         
    with open(results_file_path, 'a') as file:
        property_seg = []
        for i, image_files in enumerate(sorted(os.listdir(input_image_path))): #i: 0-35 views
            mask_path = os.path.join(base_path, case_name, "seg", str(i + 1).zfill(3) + "_s.npy")
            mask = np.load(mask_path).astype(np.float32)
            for j, image_file in enumerate(sorted(os.listdir(os.path.join(input_image_path, image_files)))): # traversing each part
                image_file = os.path.join(input_image_path, image_files, image_file)
                try:
                    if vlm_type == 'qwen':
                            message = str(Qwen(image_file, prompt))
                    elif vlm_type == 'gpt4':
                            message = str(GPT4V(image_file, prompt))
                    elif vlm_type == 'gemini':
                            message = str(Gemini(image_file, prompt))
                    elif vlm_type == 'gemini_flash':
                            message = str(GeminiFlash(image_file, prompt))
                    else:
                        raise NotImplementedError
                    
                except KeyError as e:
                    print(f"KeyError: {e} for image {image_file}")
                    raise e
                except Exception as e:
                    print(f"Exception: {e} for image {image_file}")
                    raise e
                write_msg = image_file + "," + message
                file.write(f"{write_msg}\n")
                file.flush()
                message_splitted = message.split(",")
                property = message_splitted[-1] or message_splitted[-2]
                property_right = float(clean_numeric_string(property.split('-')[-1]))
                property = property_right
                mask[mask==j] = property    

            # save the mask with property values
            property_seg.append(mask.copy())
            masked = np.ma.masked_where(mask == -1, mask)
    
            cmap = 'viridis'
            vmin, vmax = np.min(masked), np.max(masked)
            print(vmin, vmax)
            # 3. 시각화
            plt.figure(figsize=(6, 6))
            im = plt.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.axis('off')

            # 4. 컬러바 + 레이블 + tick 설정
            cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
            cbar.set_label('Value (scaled)', rotation=270, labelpad=15)
            cbar.ax.tick_params(labelsize=10)

            # 4. 이미지 저장
            # plt.savefig(os.path.join(base_path, case_name, "property_seg", str(i + 1).zfill(3) + ".png"), bbox_inches='tight', pad_inches=0.1)
            plt.close()
        save_path = os.path.join(base_path, case_name, "property_seg", "property_seg.npy")
        stacked = np.stack(property_seg, axis=0).astype(np.float32)  # 예시
        np.save(save_path, stacked)

    print("Messages have been written to", results_file_path)

def query_vlm_nir(base_path, case_name, vlm_type = "gpt4"):
    
    material_list = "wood, metal, plastic, glass, fabric, foam, food, ceramic, paper, leather, aluminum, brass, bronze, copper, steel, stainless steel, iron, cast iron, titanium, zinc, lead, gold, silver, platinum, nickel, chrome, magnesium, tin, carbon fiber, fiberglass, acrylic, polyethylene, polypropylene, polystyrene, polycarbonate, polyvinyl chloride, nylon, rubber, silicone, latex, plywood, MDF, particle board, cork, bamboo, concrete, cement, asphalt, brick, clay, porcelain, terracotta, marble, granite, limestone, sandstone, quartz, tempered glass, frosted glass, mirror, cardboard, suede, denim, cotton, wool, silk, linen, polyester, felt, velvet, mesh, canvas, fur, straw, jute, carbon, graphite, resin, wax, ice, snow, sand, soil, mud, chalk, plaster, gypsum, sponge, tar, vinyl, PVC, Teflon, Kevlar, quartzite, basalt, lava rock, obsidian, bone, horn, shell, pearl"
    material_list = material_list.split(", ")
    material_library = "{" + ", ".join(material_list) + "}"
    nir_library = """{
                    wood (light/dry): 20-30
                    metal (generic polished): 60-95
                    plastic (white): 15-20
                    plastic (black): 1-5
                    glass (clear): 4
                    fabric (white): 20-30
                    fabric (black): 5-15
                    foam (white): 40-60
                    food (generic): 40-60
                    ceramic (unglazed): 30-40
                    paper (white): 50-70
                    leather (white): 20-30
                    leather (black): 5-15
                    aluminum: 70
                    brass: 35
                    bronze: 30
                    copper: 40
                    steel: 15
                    stainless steel: 60
                    iron: 30
                    cast iron: 25
                    titanium: 30
                    zinc: 60
                    lead: 50
                    gold: 90
                    silver: 95
                    platinum: 65
                    nickel: 45
                    chrome: 65
                    tin: 70
                    carbon fiber: 5-10
                    fiberglass: 40-60
                    acrylic (clear): 4
                    polyethylene (white): 40-50
                    polyethylene (black): 1-5
                    polypropylene (white): 40-50
                    polypropylene (black): 1-5
                    polystyrene (white): 40-50
                    polystyrene (black): 1-5
                    polycarbonate (clear): 4
                    polyvinyl chloride / PVC (white): 15-20
                    polyvinyl chloride / PVC (black): 1-5
                    nylon (white): 20-30
                    nylon (black): 5-10
                    rubber (black): 5
                    silicone (white): 20-30
                    silicone (black): 5-10
                    latex (white): 20-30
                    latex (black): 5-10
                    plywood: 20-30
                    MDF: 15-25
                    cork: 25-35
                    bamboo: 25-35
                    concrete (dry): 20
                    cement (dry): 20
                    asphalt: 5
                    brick (red): 15
                    clay: 20-30
                    porcelain: 60-70
                    marble (white): 50-60
                    granite: 15-25
                    limestone: 40
                    sandstone: 20-30
                    quartz (clear): 4
                    tempered glass (clear): 4
                    frosted glass: 10-20
                    mirror (silvered): 90-95
                    cardboard: 30
                    suede (light): 20-30
                    suede (dark): 5-15
                    denim (blue): 10-20
                    cotton (white): 20-30
                    cotton (black): 5-15
                    wool (white): 20-30
                    wool (black): 5-15
                    silk (white): 20-30
                    silk (black): 5-15
                    linen (white): 20-30
                    linen (black): 5-15
                    polyester (white): 20-30
                    polyester (black): 5-15
                    felt (white): 20-30
                    felt (black): 5-15
                    velvet (light): 20-30
                    velvet (dark): 5-15
                    mesh (white synthetic): 20-30
                    mesh (black synthetic): 5-15
                    canvas (white): 20-30
                    canvas (black): 5-15
                    fur (light): 20-30
                    fur (dark): 5-15
                    straw: 25-35
                    jute: 25-35
                    carbon: 5-10
                    graphite: 5-10
                    resin (clear): 4
                    wax (white): 40-50
                    wax (colored): 10-30
                    ice: 50
                    snow (fresh): 80
                    sand (dry): 30-40
                    soil (dry): 20-30
                    soil (wet): 5-10
                    mud (wet): 5-10
                    chalk: 70-80
                    plaster: 60-70
                    sponge (natural, dry): 40-50
                    tar: 5
                    vinyl (white): 20-30
                    vinyl (black): 5-10
                    Teflon (white): 20-30
                    Teflon (black): 5-10
                    Kevlar (natural): 25-35
                    paint (white): 30-40
                    paint (black): 5-15
                    paint (metallic): 40-60}"""

    prompt = f"""
                        Provided four images:
            1. Original Image
            2. Partial segmentation (Mask Overlay, blue mask)
            3. Cropped segmented part (Part Image)
            4. Near-Infrared (NIR, ~850 nm) image

            Task:
            - Briefly caption the blue-masked part.
            - Identify the main material of the masked part.
            - Estimate the part's NIR reflectance (0-100) using the provided NIR material library: {nir_library}.
            - Provide {material_property} value of the part as low-high (one decimal place, no units or punctuation).

            Notes:
            - NIR: higher reflectance = brighter.
            - Use NIR as reference; RGB alone may be misleading.
            - Focus on the blue-masked part when identifying the material.
            - For painted/coated objects, identify them as "paint" or the specific coated material rather than the base material.

            Format:
            caption, material, NIR reflectance, {material_property}
            Only output the quadruple; do not add extra text.
            """

# Realworld -> futurework


#     if material_property == "density":
#         prompt = """You will be provided with captions that each describe an image of an object. The captions will be delimited with quotes ("). Based on the caption, give me 5 materials that the object might be made of, along with the mass densities (in kg/m^3) of each of those materials. You may provide a range of values for the mass density instead of a single value. Try to consider all the possible parts of the object. Do not include coatings like "paint" in your answer.

# Format Requirement:
# You must provide your answer as a list of 5 (material: mass density) pairs, each separated by a semi-colon (;). Do not include any other text in your answer, as it will be parsed by a code script later. Your answer must look like:
# (material 1: low-high kg/m^3);(material 2: low-high kg/m^3);(material 3: low-high kg/m^3);(material 4: low-high kg/m^3);(material 5: low-high kg/m^3)
# """
    # output_file = f'{case_name}.txt'


    output_file = 'vlm_result_nir.txt'
    results_file_path = os.path.join(base_path, 'scene', output_file)
    if not vlm_type:
        property_seg = []
        with open(results_file_path, 'r') as file:
             lines = [line.strip() for line in file]
        mask = None
        for line in lines:
            line = line.split(",")
            file = line[0]
            i = int(file[-9:-7])
            j = int(file[-6:-4])
            if not j:
                if mask is not None:
                    property_seg.append(mask)
                mask_path = os.path.join(base_path, case_name, "seg", str(i).zfill(3) + "_s.npy")
                mask = np.load(mask_path).astype(np.float32)
            property = line[-1] or line[-2]
            property_right = float(clean_numeric_string(property.split('-')[-1]))
            property = property_right
            mask[mask==j] = property    

        if mask is not None:
            property_seg.append(mask)
             
        save_path = os.path.join(base_path, case_name, "property_seg", "property_seg.npy")
        stacked = np.stack(property_seg, axis=0).astype(np.float32)  # 예시
        np.save(save_path, stacked)

        return
    if os.path.exists(results_file_path):
        os.remove(results_file_path)
        ...
    # results_file_path = os.path.join(base_path, case_name, output_file)
    # os.makedirs(os.path.dirname(results_file_path), exist_ok=True)
    input_image_path = os.path.join(base_path, case_name, "gpt_input_nir")

    
         
    with open(results_file_path, 'a') as file:
        property_seg = []
        for i, image_files in enumerate(sorted(os.listdir(input_image_path))): #i: 0-35 views
            mask_path = os.path.join(base_path, case_name, "seg", str(i + 1).zfill(3) + "_s.npy")
            mask = np.load(mask_path).astype(np.float32)
            for j, image_file in enumerate(sorted(os.listdir(os.path.join(input_image_path, image_files)))): # traversing each part
                image_file = os.path.join(input_image_path, image_files, image_file)
                try:
                    if vlm_type == 'qwen':
                            message = str(Qwen(image_file, prompt))
                    elif vlm_type == 'gpt4':
                            message = str(GPT4V(image_file, prompt))
                    elif vlm_type == 'gemini':
                            message = str(Gemini(image_file, prompt))
                    elif vlm_type == 'gemini_flash':
                            message = str(GeminiFlash(image_file, prompt))
                    else:
                        raise NotImplementedError
                    
                except KeyError as e:
                    print(f"KeyError: {e} for image {image_file}")
                    raise e
                except Exception as e:
                    print(f"Exception: {e} for image {image_file}")
                    raise e
                write_msg = image_file + "," + message
                file.write(f"{write_msg}\n")
                file.flush()
                message_splitted = message.split(",")
                property = message_splitted[-1] or message_splitted[-2]
                property_right = float(clean_numeric_string(property.split('-')[-1]))
                property = property_right
                mask[mask==j] = property    

            # save the mask with property values
            property_seg.append(mask.copy())
            masked = np.ma.masked_where(mask == -1, mask)
    
            cmap = 'viridis'
            vmin, vmax = np.min(masked), np.max(masked)
            print(vmin, vmax)
            # 3. 시각화
            plt.figure(figsize=(6, 6))
            im = plt.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.axis('off')

            # 4. 컬러바 + 레이블 + tick 설정
            cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
            cbar.set_label('Value (scaled)', rotation=270, labelpad=15)
            cbar.ax.tick_params(labelsize=10)

            # 4. 이미지 저장
            # plt.savefig(os.path.join(base_path, case_name, "property_seg", str(i + 1).zfill(3) + ".png"), bbox_inches='tight', pad_inches=0.1)
            plt.close()
        save_path = os.path.join(base_path, case_name, "property_seg", "property_seg.npy")
        stacked = np.stack(property_seg, axis=0).astype(np.float32)  # 예시
        np.save(save_path, stacked)
        
    print("Messages have been written to", results_file_path)


def run_vlm(base_path, nir=False, vlm_type = "gpt4"):
    all_cases = os.listdir(base_path) # only one case in my code

    for case_name in all_cases:
        save_path = os.path.join(base_path, case_name, "property_seg", "property_seg.npy")
        query_vlm(base_path, case_name, nir, vlm_type=vlm_type)
        backproject_to_blender(save_path)

def run_vlm_nir(base_path, nir=True, vlm_type = "gpt4"):
    all_cases = os.listdir(base_path) # only one case in my code

    for case_name in all_cases:
        save_path = os.path.join(base_path, case_name, "property_seg", "property_seg.npy")
        query_vlm(base_path, case_name, nir, vlm_type=vlm_type)
        backproject_to_blender(save_path)


def backproject_to_pcd(property_seg_path, pcd_idx_path="projected_views_ply/point_index_map_all.npy"):
    # === Load data ===
    property_seg = np.load(property_seg_path)  # shape: (num_of_views, H, W)
    point_index_map_all = np.load(pcd_idx_path) # "projected_views_ply/point_index_map_all.npy"  shape: (num_of_views, H, W)

    # === Flatten for processing ===
    view_count, H, W = property_seg.shape
    property_seg_flat = property_seg.reshape(view_count, -1)
    point_index_flat = point_index_map_all.reshape(view_count, -1)

    # === Accumulate property values for each 3D point ===
    point_values = defaultdict(list)
    for view in range(view_count):
        for i in range(H * W):
            point_idx = point_index_flat[view, i]
            prop_val = property_seg_flat[view, i]
            if point_idx != -1 and prop_val != -1:
                point_values[point_idx].append(prop_val)

    # === Load point cloud ===
    pcd_path = "logs/20250725_red_sofa_with_wooden_legs_be8fb6/pointcloud_from_mesh.ply"
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)
    n_points = len(points)
    print(n_points)
    # print(point_values)
    

    # === Average values and prepare final array ===
    point_property_array = np.full(n_points, -1.0, dtype=np.float32)
    for idx, values in point_values.items():
        counter = Counter(values)
        most_common_value, _ = counter.most_common(1)[0]
        point_property_array[idx] = most_common_value

    

    # === Normalize property values to [0, 1] for color mapping ===


    # === Apply colormap (e.g., viridis) ===
    cmap = plt.get_cmap('viridis')
    valid_mask = np.zeros(n_points, dtype=bool)
    valid_colors = np.ones((n_points, 3)) * 0.5  # default gray
    valid_indices = np.where((point_property_array != -1))[0]
    valid_mask[valid_indices] = True
    points = points[valid_mask]
    filtered_props = point_property_array[valid_mask]
    vmin, vmax = filtered_props.min(), filtered_props.max()
    norm_props = np.clip((filtered_props - vmin) / (vmax - vmin), 0, 1)

    cmap = plt.get_cmap('viridis')
    colors = np.array([cmap(val)[:3] for val in norm_props])


    pcd.colors = o3d.utility.Vector3dVector(colors)

    # === Save updated point cloud ===
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(points)
    filtered_pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud("projected_views_ply/pointcloud_with_property_color.ply", filtered_pcd)

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # 컬러바 이미지로 저장
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)

    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')
    cb.set_label(material_property)
    plt.savefig('projected_views_ply/color_legend.png')
    plt.close()

def backproject_to_blender(property_seg_path, pcd_idx_path="/home/han/workspace/VLM-LLM/projected_views_pcd/points.npy"):
    # === Load data ===
    property_seg = np.load(property_seg_path)  # shape: (num_of_views, H, W)
    point_index_map_all = np.load(pcd_idx_path) #  shape: (num_of_views, H, W, 3)

    # === Flatten for processing ===
    view_count, H, W = property_seg.shape
    property_seg_flat = property_seg.reshape(view_count, -1)
    point_index_flat = point_index_map_all.reshape(view_count, H * W, 3)

    # === Accumulate property values for each 3D point ===
    point_values = defaultdict(list)
    for view in range(view_count):
        for i in range(H * W):
            point_coord = point_index_flat[view, i]
            prop_val = property_seg_flat[view, i]
            if not np.isnan(point_coord[0]):
                point_values[tuple(point_coord)].append(prop_val)

    coords = []
    labels = []
    for coord, vals in point_values.items():
        most_common_val = Counter(vals).most_common(1)[0][0]
        coords.append(coord)
        labels.append(most_common_val)

    coords = np.array(coords)   # (N, 3)
    labels = np.array(labels)   # (N,)
    print (f"Number of points: {coords.shape}")
    print(coords)

    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=labels.min(), vmax=labels.max())
    colors = cmap(norm(labels))[:, :3]   # RGBA -> RGB

    # 3. Open3D PointCloud 생성
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 4. 시각화
    o3d.visualization.draw_geometries([pcd])  # show()

    # 5. 저장 (.ply)
    o3d.io.write_point_cloud("projected_views_ply/output_pointcloud.ply", pcd)

    # 6. 컬러바 저장
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)

    cb1 = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax, orientation="horizontal"
    )
    cb1.set_label("prop_val")
    plt.savefig("projected_views_ply/colorbar.png")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument('--vlm', type=str, default="qwen", help="gpt, qwen")
    parser.add_argument('--dataset_path', type=str, default="2d_output_dirs")
    args = parser.parse_args()
    run_vlm(args.dataset_path, args.vlm)

