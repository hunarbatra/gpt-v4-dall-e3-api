from PIL import Image

import pandas as pd

import base64
import io
import os

SAVE_RESULTS_SCHEMA = {"image_path": [], "gpt4-v response": []}

def image_to_base64(image_path: str) -> str:
    with Image.open(image_path) as image:
        buffered = io.BytesIO()
        if image.mode in ("RGBA", "P"):  
            image = image.convert("RGB")

        image.save(buffered, format="JPEG")

        img_str = base64.b64encode(buffered.getvalue())
        return img_str.decode('utf-8')

def check_dir(dir_path: str) -> None:
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        
def save_csv(df, path):
    df.to_csv(path, index=False)

def load_csv(path):
    return pd.read_csv(path)

def load_df(path, filename):
    if not os.path.exists(path + filename):
        return pd.DataFrame(SAVE_RESULTS_SCHEMA)
    else:
        return load_csv(path + filename)