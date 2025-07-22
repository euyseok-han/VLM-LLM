import os
import argparse
from gaussian_property_main.utils.vlm_utils import get_image_files, Qwen, GPT4V, Qwen_prev
import numpy as np
from PIL import Image

def query_vlm(base_path, case_name, vlm_type = "gpt4"):
    input_image_path = os.path.join(base_path, case_name, "gpt_input")
    image_files = get_image_files(input_image_path)
    print(f"Processing case: {case_name}, list of images: {image_files}")
    mask_path = os.path.join(base_path, case_name, "seg", "001_s.npy")
    mask = np.load(mask_path)
    material_list = "wood, metal, plastic, glass, fabric, foam, food, ceramic, paper, leather, aluminum, brass, bronze, copper, steel, stainless steel, iron, cast iron, titanium, zinc, lead, gold, silver, platinum, nickel, chrome, magnesium, tin, carbon fiber, fiberglass, acrylic, polyethylene, polypropylene, polystyrene, polycarbonate, polyvinyl chloride, nylon, rubber, silicone, latex, plywood, MDF, particle board, cork, bamboo, concrete, cement, asphalt, brick, clay, porcelain, terracotta, marble, granite, limestone, sandstone, quartz, tempered glass, frosted glass, mirror, cardboard, suede, denim, cotton, wool, silk, linen, polyester, felt, velvet, mesh, canvas, fur, straw, jute, carbon, graphite, resin, wax, ice, snow, sand, soil, mud, chalk, plaster, gypsum, sponge, tar, vinyl, PVC, Teflon, Kevlar, quartzite, basalt, lava rock, obsidian, bone, horn, shell, pearl"
    material_list = material_list.split(", ")
    material_library = "{" + ", ".join(material_list) + "}"
    material_property = "density(g/cm^3)"
    prompt = f"""Provided a picture. The left image is the original picture of the object (Original Image), and the middle image is a partial segmentation diagram (Mask Overlay), mask is in red. The right image is a partial of the object. 
    Based on the image, firstly provide a brief caption of the part. Secondly, describe what the part is made of (provide the major one). Finally, we combine what the object is and the material of the object to predict the hardness of the part. Choose whether to use Shore A hardness or Shore D hardness depending on the material. You may provide a range of values for hardness instead of a single value. 

    Format Requirement:
    You must provide your answer as a (brief caption of the part, material of the part, {material_property}) pair. Do not include any other text in your answer, as it will be parsed by a code script later. 
    common material library: {material_library}. 
    Your answer must look like: caption, material, {material_property}. 
    The material type must be chosen from the above common material library. """ #Make sure to use Shore A or Shore D hardness, not Mohs hardness."""

#     if material_property == "density":
#         prompt = """You will be provided with captions that each describe an image of an object. The captions will be delimited with quotes ("). Based on the caption, give me 5 materials that the object might be made of, along with the mass densities (in kg/m^3) of each of those materials. You may provide a range of values for the mass density instead of a single value. Try to consider all the possible parts of the object. Do not include coatings like "paint" in your answer.

# Format Requirement:
# You must provide your answer as a list of 5 (material: mass density) pairs, each separated by a semi-colon (;). Do not include any other text in your answer, as it will be parsed by a code script later. Your answer must look like:
# (material 1: low-high kg/m^3);(material 2: low-high kg/m^3);(material 3: low-high kg/m^3);(material 4: low-high kg/m^3);(material 5: low-high kg/m^3)
# """
    # output_file = f'{case_name}.txt'
    output_file = 'verdict.txt'
    results_file_path = os.path.join(base_path, output_file)
    # results_file_path = os.path.join(base_path, case_name, output_file)
    # os.makedirs(os.path.dirname(results_file_path), exist_ok=True)

    with open(results_file_path, 'a') as file:
        for i, image_file in enumerate(image_files):
            try:
                if vlm_type == 'qwen':
                    message = str(Qwen(image_file, prompt))
                else:
                    message = str(GPT4V(image_file, prompt))
            except KeyError as e:
                print(f"KeyError: {e} for image {image_file}")
                raise e
            except Exception as e:
                print(f"Exception: {e} for image {image_file}")
                raise e
            write_msg = image_file + "," + message
            file.write(f"{write_msg}\n")
            file.flush()
            property = message.split(",")[-1]
            property_min = float(property.split('-')[0])
            property_max = float(property.split('-')[-1])
            property = (property_min + property_max) / 2
            mask[mask==i] = property
    
    print("Messages have been written to", results_file_path)


def run_vlm(base_path, vlm_type):
    all_cases = os.listdir(base_path)
    output_file = 'verdict.txt'
    results_file_path = os.path.join(base_path[:-5], output_file)
    if os.path.exists(results_file_path):
        os.remove(results_file_path)

    for case_name in all_cases:
        query_vlm(base_path, case_name, vlm_type=vlm_type)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument('--vlm', type=str, default="qwen", help="gpt, qwen")
    parser.add_argument('--dataset_path', type=str, default="2d_output_dirs")
    args = parser.parse_args()
    run_vlm(args.dataset_path, args.vlm)

