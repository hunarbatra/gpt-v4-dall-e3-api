from tenacity import retry, stop_after_attempt

import requests
import json

from typing import Optional

gpt4v_api_url = 'http://localhost:3000/'
headers = {
    'Content-Type': 'application/json',
}

@retry(stop=stop_after_attempt(3))
def gpt4v_runner(prompt: str, image: Optional[str] = None, model_name: str = 'gpt-4v'):
    data = {
        'prompt': prompt,
        'image': image,
    }
    response = requests.post(f'{gpt4v_api_url}{model_name}', headers=headers, data=json.dumps(data))
    response = response.json()
    return response["output"]

@retry(stop=stop_after_attempt(3))
def dalle3_runner(prompt: str, model_name: str = 'dall-e3'):
    data = {
        'prompt': prompt,
    }
    response = requests.post(f'{gpt4v_api_url}{model_name}', headers=headers, data=json.dumps(data))
    response = response.json()
    return response["images"]
    

