import openai
import base64
import json


PRED_CAND_MATS_DENSITY_SYS_MSG = """
Given a text file where each line is formatted like:
<file path>,<object name>,<material>,<material property>,
extract and analyze the material and the material property, where the property refers to the density of the material (e.g., metal, 7.85 g/cm³).

You must identify the most frequently occurring materials (up to 5) and their corresponding density ranges.

You must provide your answer as a list of (material: density) pairs, each separated by a semicolon (;). Do not include any other text in your answer, as it will be parsed by a code script later. Your answer must look exactly like:
(material 1: low-high g/cm³);(material 2: low-high g/cm³);(material 3: low-high g/cm³);(material 4: low-high g/cm³);(material 5: low-high g/cm³)
If there are fewer than 5 distinct materials, you may provide fewer than 5 pairs.

If a material appears with multiple density values, determine the lowest and highest values and present them as a range (e.g., 2.30–2.70 g/cm³). Round values to two decimal places. Use consistent units (g/cm³).
Do not include any other text in your answer. Do not include unnecessary words besides the material in the material name. 
"""

PRED_CAND_MATS_DENSITY_SYS_MSG_4V = """You will be given an image of an object. Based on the image, give me a short (5-10 words) description of what the object is, and also 5 materials (e.g. wood, plastic, foam) that the object might be made of, along with the mass densities (in kg/m^3) of each of those materials. You may provide a range of values for the mass density instead of a single value. Try to consider all the possible parts of the object. Do not include coatings like "paint" in your answer.

Format Requirement:
You must provide your answer in the following JSON format, as it will be parsed by a code script later. Your answer must look like:
{
    "description": description
    "materials": [
        {"name": material1, "mass density (kg/m^3)": low-high},
        {"name": material2, "mass density (kg/m^3)": low-high},
        {"name": material3, "mass density (kg/m^3)": low-high},
        {"name": material4, "mass density (kg/m^3)": low-high},
        {"name": material5, "mass density (kg/m^3)": low-high}
    ]
}
Do not include any other text in your answer. Do not include unnecessary words besides the material in the material name. 
"""


PRED_CAND_MATS_HARDNESS_SYS_MSG = """You will be provided with captions that each describe an image of an object. The captions will be delimited with quotes ("). Based on the caption, give me 3 materials that the object might be made of, along with the hardness of each of those materials. Choose whether to use Shore A hardness or Shore D hardness depending on the material. You may provide a range of values for hardness instead of a single value. Try to consider all the possible parts of the object.

Format Requirement:
You must provide your answer as a list of 3 (material: hardness, Shore A/D) tuples, each separated by a semi-colon (;). Do not include any other text in your answer, as it will be parsed by a code script later. Your answer must look like:
(material 1: low-high, <Shore A or Shore D>);(material 2: low-high, <Shore A or Shore D>);(material 3: low-high, <Shore A or Shore D>)
Make sure to use Shore A or Shore D hardness, not Mohs hardness.
"""

PRED_CAND_MATS_FRICTION_SYS_MSG = """You will be provided with captions that each describe an image. The captions will be delimited with quotes ("). Based on the caption, give me 3 materials that the surfaces in the image might be made of, along with the kinetic friction coefficient of each material when sliding against a fabric surface. You may provide a range of values for the friction coefficient instead of a single value. Try to consider all the possible surfaces.

Format Requirement:
You must provide your answer as a list of 3 (material: friction coefficient) pairs, each separated by a semi-colon (;). Do not include any other text in your answer, as it will be parsed by a code script later. Your answer must look like:
(material 1: low-high);(material 2: low-high);(material 3: low-high)
Try to provide as narrow of a range as possible for the friction coefficient.
"""

PRED_THICKNESS_SYS_MSG = """
You are given a text file in which each line follows the format:
<file path>,<object name>,<material>,<material property>,

Your task is to extract the material (the third item in each line) and infer a plausible thickness range (in centimeters) based solely on the material name — ignoring the material property field.

Use your knowledge of typical material applications and real-world usage to estimate a reasonable thickness range for each unique material.

Output Format:
Return a list of 5 (material: thickness) pairs.
Each thickness must be expressed as a range (e.g., "0.1-0.5 cm").
Each pair should be enclosed in parentheses and separated by a semicolon.

Important:

    Do not use the <material property> field in your analysis.

    Do not include any extra text before or after the list.

    The output must follow this format exactly:
    (material 1: low-high cm);(material 2: low-high cm);...;(material 5: low-high cm)
"""

PRED_THICKNESS_EXAMPLE_INPUT_1 = 'Caption: "a lamp with a white shade" Materials: "fabric, plastic, metal, ceramic, glass"'
PRED_THICKNESS_EXAMPLE_OUTPUT_1 = "(fabric: 0.1-0.2 cm);(plastic: 0.3-1.0 cm);(metal: 0.1-0.2 cm);(ceramic: 0.2-0.5 cm);(glass: 0.3-0.8 cm)"
PRED_THICKNESS_EXAMPLE_INPUT_2 = 'Caption: "a grey ottoman" Materials: "wood, fabric, foam, metal, plastic"'
PRED_THICKNESS_EXAMPLE_OUTPUT_2 = "(wood: 2.0-4.0 cm);(fabric: 0.2-0.5 cm);(foam: 5.0-15.0 cm);(metal: 0.1-0.2 cm);(plastic: 0.5-1.0 cm)"
PRED_THICKNESS_EXAMPLE_INPUT_3 = 'Caption: "a white frame" Materials: "plastic, wood, aluminum, steel, glass"'
PRED_THICKNESS_EXAMPLE_OUTPUT_3 = "(plastic: 0.1-0.3 cm);(wood: 1.0-1.5 cm);(aluminum: 0.1-0.3 cm);(steel: 0.1-0.2 cm);(glass: 0.2-0.5 cm)"
PRED_THICKNESS_EXAMPLE_INPUT_4 = 'Caption: "a metal rack with three shelves" Materials: "steel, aluminum, wood, plastic, iron"'
PRED_THICKNESS_EXAMPLE_OUTPUT_4 = "(steel: 0.1-0.2 cm);(aluminum: 0.1-0.3 cm);(wood: 1.0-2.0 cm);(plastic: 0.5-1.0 cm);(iron: 0.5-1.0 cm)"

import requests

def gpt_candidate_materials(caption, property_name='density', model_name='gpt-3.5-turbo', seed=100):

    if property_name == 'density':
        sys_msg = PRED_CAND_MATS_DENSITY_SYS_MSG
    elif property_name == 'hardness':
        sys_msg = PRED_CAND_MATS_HARDNESS_SYS_MSG
    elif property_name == 'friction':
        sys_msg = PRED_CAND_MATS_FRICTION_SYS_MSG
    else:
        raise NotImplementedError
    # response = openai.ChatCompletion.create(
    #   model=model_name,
    #     messages=[
    #         {"role": "system", "content": sys_msg},
    #         {"role": "user", "content": '"%s"' % caption},
    #     ],
    #     request_timeout=20,
    #     seed=seed,
    # )
    response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": "Bearer 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",
        "Content-Type": "application/json",
    },
    data=json.dumps({
        "model": "qwen/qwen-2.5-72b-instruct",
        "messages": [
        {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": sys_msg,
                    },
                ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": caption,
                },
            ],
        }
        ],
        
    })
    )
    return response.json()['choices'][0]['message']['content']


def gpt_thickness(caption, candidate_materials, mode='list', model_name='gpt-3.5-turbo', seed=100):

    if mode == 'list':
        mat_names, mat_vals = parse_material_list(candidate_materials)
    elif mode == 'json':
        caption, mat_names, mat_vals = parse_material_json(candidate_materials)
    else:
        raise NotImplementedError
    mat_names_str = ', '.join(mat_names)
    user_msg = 'Caption: "%s" Materials: "%s"' % (caption, mat_names_str)

    response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": "Bearer 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",
        "Content-Type": "application/json",
    },
    data=json.dumps({
        "model": "qwen/qwen-2.5-72b-instruct",
        "messages": [
        {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": PRED_THICKNESS_SYS_MSG,
                    },
                ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_msg,
                },
            ],
        }
        ],
        
    })
    )

    # response = openai.ChatCompletion.create(
    #   model=model_name,
    #     messages=[
    #         {"role": "system", "content": PRED_THICKNESS_SYS_MSG},
    #         # {"role": "user", "content": PRED_THICKNESS_EXAMPLE_INPUT_1},
    #         # {"role": "assistant", "content": PRED_THICKNESS_EXAMPLE_OUTPUT_1},
    #         # {"role": "user", "content": PRED_THICKNESS_EXAMPLE_INPUT_2},
    #         # {"role": "assistant", "content": PRED_THICKNESS_EXAMPLE_OUTPUT_2},
    #         # {"role": "user", "content": PRED_THICKNESS_EXAMPLE_INPUT_3},
    #         # {"role": "assistant", "content": PRED_THICKNESS_EXAMPLE_OUTPUT_3},
    #         # {"role": "user", "content": PRED_THICKNESS_EXAMPLE_INPUT_4},
    #         # {"role": "assistant", "content": PRED_THICKNESS_EXAMPLE_OUTPUT_4},
    #         {"role": "user", "content": user_msg},
    #     ],
    #     request_timeout=20,
    #     seed=seed,
    # )
    print(response)
    return response.json()['choices'][0]['message']['content']


def parse_material_list(matlist, max_n=5):
    #matlist: (material 1: low-high g/cm³);(material 2: low-high g/cm³);...
    elems = matlist.split(';')
    if len(elems) > max_n:
        print('too many materials %s' % matlist)
        return None
    
    mat_names = []
    mat_vals = []

    for elem in elems:
        elem_parts = elem.strip().split(':')
        if len(elem_parts) != 2: 
            print('bad format %s' % matlist)
            return None
        mat_name, values = elem_parts
        if not mat_name.startswith('(') or mat_name[1].isnumeric() or mat_name.startswith('(material 1'):
            print('bad format %s' % matlist)
            return None

        mat_name = mat_name[1:]
        mat_names.append(mat_name.lower())  # force lowercase

        values = values.strip().split(' ')[0]
        values = values.replace(",", "")
        if values[-1] == ')':
            values = values[:-1]

        # Value may or may not be a range
        splitted = values.split('-')
        try:
            float(splitted[0])
        except ValueError:
            print('value cannot be converted to float %s' % matlist)
            return None
        if len(splitted) == 2:
            mat_vals.append([float(splitted[0]), float(splitted[1])])
        elif len(splitted) == 1:
            mat_vals.append([float(splitted[0]), float(splitted[0])])
        else:
            print('bad format %s' % matlist)
            return None
        
    return mat_names, mat_vals


def parse_material_hardness(matlist, max_n=5):
    elems = matlist.split(';')
    if len(elems) > max_n:
        print('too many materials %s' % matlist)
        return None
    
    mat_names = []
    mat_vals = []

    for elem in elems:
        elem_parts = elem.strip().split(':')
        if len(elem_parts) != 2: 
            print('bad format %s' % matlist)
            return None
        mat_name, values = elem_parts
        if not mat_name.startswith('(') or mat_name[1].isnumeric() or mat_name.startswith('(material 1'):
            print('bad name %s' % matlist)
            return None

        mat_name = mat_name[1:]
        mat_names.append(mat_name.lower())  # force lowercase

        values = values.strip().split(',')
        units = values[-1].split(' ')[-1][:-1]
        if units not in ['A', 'D']:
            print('bad units %s' % matlist)
            return None
        values = values[0]
        values = values.replace(",", "")

        # Value may or may not be a range
        splitted = values.split('-')
        try:
            float(splitted[0])
        except ValueError:
            print('value cannot be converted to float %s' % matlist)
            return None
        if len(splitted) == 2:
            mat_vals.append([float(splitted[0]), float(splitted[1])])
        elif len(splitted) == 1:
            mat_vals.append([float(splitted[0]), float(splitted[0])])
        else:
            print('bad format %s' % matlist)
            return None
        
        if units == 'D':
            mat_vals[-1][0] += 100
            mat_vals[-1][1] += 100
        
    return mat_names, mat_vals


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    

def gpt4v_candidate_materials(image_path, property_name='density', seed=100):

    if property_name == 'density':
        sys_msg = PRED_CAND_MATS_DENSITY_SYS_MSG_4V
    else:
        raise NotImplementedError
    
    base64_image = encode_image(image_path)

    response = openai.ChatCompletion.create(
        model="gpt-4-vision-preview",
        messages=[
        {
            "role": "system",
            "content": sys_msg
        },
        {
            "role": "user",
            "content": [
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }
                },
            ]
        }
        ],
        request_timeout=30,
        max_tokens=300,
        seed=seed,
            # response_format={"type": "json_object"},
    )
    return response['choices'][0]['message']['content']


def parse_material_json(matjson, max_n=5, field_name='mass density (kg/m^3)'):
    desc_and_mats = json.loads(matjson)
    if 'description' not in desc_and_mats or 'materials' not in desc_and_mats:
        print('bad format %s' % matjson)
        return None
    mat_names = []
    mat_vals = []
    for mat in desc_and_mats['materials']:
        if 'name' not in mat or field_name not in mat:
            print('bad format %s' % matjson)
            return None
        mat_name = mat['name']
        mat_names.append(mat_name.lower())  # force lowercase
        values = mat[field_name]
        # Value may or may not be a range
        splitted = values.split('-')
        try:
            float(splitted[0])
        except ValueError:
            print('value cannot be converted to float %s' % matjson)
            return None
        if len(splitted) == 2:
            mat_vals.append([float(splitted[0]), float(splitted[1])])
        elif len(splitted) == 1:
            mat_vals.append([float(splitted[0]), float(splitted[0])])
        else:
            print('bad format %s' % matjson)
            return None
    return desc_and_mats['description'], mat_names, mat_vals