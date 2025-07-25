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

def clean_numeric_string(s):
    # 숫자와 점(.)만 남기고 모두 제거
    cleaned = re.sub(r'[^0-9.]', '', s)
    # 문자열 끝에 있는 점 제거
    cleaned = re.sub(r'\.+$', '', cleaned)
    return cleaned


def query_vlm(base_path, case_name, vlm_type = "gpt4"):
    
    material_list = "wood, metal, plastic, glass, fabric, foam, food, ceramic, paper, leather, aluminum, brass, bronze, copper, steel, stainless steel, iron, cast iron, titanium, zinc, lead, gold, silver, platinum, nickel, chrome, magnesium, tin, carbon fiber, fiberglass, acrylic, polyethylene, polypropylene, polystyrene, polycarbonate, polyvinyl chloride, nylon, rubber, silicone, latex, plywood, MDF, particle board, cork, bamboo, concrete, cement, asphalt, brick, clay, porcelain, terracotta, marble, granite, limestone, sandstone, quartz, tempered glass, frosted glass, mirror, cardboard, suede, denim, cotton, wool, silk, linen, polyester, felt, velvet, mesh, canvas, fur, straw, jute, carbon, graphite, resin, wax, ice, snow, sand, soil, mud, chalk, plaster, gypsum, sponge, tar, vinyl, PVC, Teflon, Kevlar, quartzite, basalt, lava rock, obsidian, bone, horn, shell, pearl"
    material_list = material_list.split(", ")
    material_library = "{" + ", ".join(material_list) + "}"
    material_property = "density(g/cm^3)"
    prompt = f"""Provided a picture. The left image is the original picture of the object (Original Image), and the middle image is a partial segmentation diagram (Mask Overlay), mask is in red. The right image is a partial of the object.
    Based on the image, firstly provide a brief caption of the part. Secondly, describe what the part is made of (provide the major one), and {material_property} of the part. 
    
    Format Requirement:
    You must provide your answer as a (brief caption of the part, material of the part, {material_property}) pair.
    Do not include any other text in your answer.
    Common material library: {material_library}.
    Your answer must look like: caption, material, 0.1-0.2.
    Importantly, the property value must be in the form of low-high, using only one decimal place, with no units and no period or comma or parenthesis at the end. For example: 0.1-0.2.
    The material type must be chosen from the above common material library.
    For the same material, please provide consistent or similar {material_library} values across responses. For example, if the material is wood, please return a {material_library} similar to the one you previously provided for wood. Do not give different values for the same material.
    """#Make sure to use Shore A or Shore D hardness, not Mohs hardness."""

#     if material_property == "density":
#         prompt = """You will be provided with captions that each describe an image of an object. The captions will be delimited with quotes ("). Based on the caption, give me 5 materials that the object might be made of, along with the mass densities (in kg/m^3) of each of those materials. You may provide a range of values for the mass density instead of a single value. Try to consider all the possible parts of the object. Do not include coatings like "paint" in your answer.

# Format Requirement:
# You must provide your answer as a list of 5 (material: mass density) pairs, each separated by a semi-colon (;). Do not include any other text in your answer, as it will be parsed by a code script later. Your answer must look like:
# (material 1: low-high kg/m^3);(material 2: low-high kg/m^3);(material 3: low-high kg/m^3);(material 4: low-high kg/m^3);(material 5: low-high kg/m^3)
# """
    # output_file = f'{case_name}.txt'


    output_file = 'verdict.txt'
    results_file_path = os.path.join(base_path[:-5], output_file)
    if os.path.exists(results_file_path):
        os.remove(results_file_path)
        ...
    # results_file_path = os.path.join(base_path, case_name, output_file)
    # os.makedirs(os.path.dirname(results_file_path), exist_ok=True)
    input_image_path = os.path.join(base_path, case_name, "gpt_input")

    os.makedirs(os.path.join(base_path, case_name, "property_seg"), exist_ok=True)
    
    with open(results_file_path, 'a') as file:
        property_seg = []
        for i, image_files in enumerate(sorted(os.listdir(input_image_path))): #i: 0-35 views
            mask_path = os.path.join(base_path, case_name, "seg", str(i + 1).zfill(3) + "_s.npy")
            mask = np.load(mask_path)
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
            property_seg.append(mask)
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
            plt.savefig(os.path.join(base_path, case_name, "property_seg", str(i + 1).zfill(3) + ".png"), bbox_inches='tight', pad_inches=0.1)
            plt.close()
        save_path = os.path.join(base_path, case_name, "property_seg", "property_seg.npy")
        np.stack(property_seg, axis=0)
        np.save(save_path, property_seg)

    print("Messages have been written to", results_file_path)

def run_vlm(base_path, vlm_type = "gpt4"):
    all_cases = os.listdir(base_path) # only one case in my code

    for case_name in all_cases:
        save_path = os.path.join(base_path, case_name, "property_seg", "property_seg.npy")
        query_vlm(base_path, case_name, vlm_type=vlm_type)
        backproject_to_pcd(save_path, "projected_views_ply/point_index_map_all.npy")


def backproject_to_pcd(property_seg_path, pcd_idx_path="projected_views_ply/point_index_map_all.npy", pcd_path="logs/20250709_chair_chair/pointcloud_from_mesh.ply"):
    # === Load data ===
    property_seg = np.load(property_seg_path)  # shape: (36, H, W)
    point_index_map_all = np.load(pcd_idx_path) # "projected_views_ply/point_index_map_all.npy"  shape: (36, H, W)

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
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)
    n_points = len(points)

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
    o3d.io.write_point_cloud("logs/20250709_chair_chair/pointcloud_with_property_color.ply", filtered_pcd)

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # 컬러바 이미지로 저장
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)

    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')
    cb.set_label('Density (g/cm³)')
    plt.savefig('logs/20250709_chair_chair/color_legend.png')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument('--vlm', type=str, default="qwen", help="gpt, qwen")
    parser.add_argument('--dataset_path', type=str, default="2d_output_dirs")
    args = parser.parse_args()
    run_vlm(args.dataset_path, args.vlm)

