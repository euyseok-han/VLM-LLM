import numpy as np
import os
import random
import base64
from openai import OpenAI

import requests
import json

random.seed(123)  # Set random seed to 123

OPENROUTER_API_KEY = "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"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def GPT4V(image_path, prompt):
    client = OpenAI(api_key="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")


    # Getting the base64 string
    base64_image = encode_image(image_path)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
    )

    return response.choices[0]


    
def Qwen(image_path, prompt):

    base64_image = encode_image(image_path)
    response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    },
    data=json.dumps({
        "model": "qwen/qwen-2.5-72b-instruct",
        "messages": [
        {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
        
    })
    )
    return response.json()['choices'][0]['message']['content']

def Gemini(image_path, prompt):

    base64_image = encode_image(image_path)
    API_KEY = 'AIzaSyCBo9HY_AvmfdaNVZjz_VJwlm-ligO2W3I'
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

    headers = {
        'Content-Type': 'application/json',
    }

    payload = {
    "contents": [
        {
            "parts": [
                {
                    "text": prompt  # 프롬프트
                },
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": base64_image
                    }
                }
            ]
        }
    ]
}

    response = requests.post(url, headers=headers, json=payload)
    result = response.json()

    if response.ok:
        result = response.json()
        ret = result["candidates"][0]["content"]["parts"][0]["text"]
        print(ret)
        return ret
    else:
        print("Error:", response.status_code, response.text)
        raise ValueError

def GeminiFlash(image_path, prompt):
    # 2. API 설정
    base64_image = encode_image(image_path)
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    # 3. 요청 페이로드 설정
    payload = {
        "model": "google/gemini-2.5-flash-lite",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
    }

    # 4. 요청 보내기
    response = requests.post(url, headers=headers, json=payload)

    # 5. 결과 확인
    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    elif response.status_code == 408:
        return GeminiFlash(image_path, prompt)
    else:
        print("Error:", response.status_code, response.text)
        raise ValueError

def get_image_files(directory):
    image_files = []
    # Iterate over the directory
    num_image_files = len(os.listdir(directory))
    print("################################################", num_image_files)
    for i in range(1, num_image_files + 1):  # Modify range if more folders need to be processed
        sub_path = str(i).zfill(2)
        now_path = os.path.join(directory, sub_path)
        all_png = os.listdir(now_path)
        for png in all_png:
            img_path = os.path.join(now_path, png)
            image_files.append(img_path)

    image_files = sorted(image_files, key=lambda x: (x.split('/')[-2], x.split('/')[-1]))

    return image_files




