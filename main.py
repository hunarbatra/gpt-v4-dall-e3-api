from models import gpt4v_runner, dalle3_runner
from utils import image_to_base64, check_dir, load_df, save_csv

import pandas as pd
import argparse

def run_experiments(exp_dir: str, data_name: str = 'imagenet-1k-1000', data_cap: int = 100):
    images_path = get_dataset(data_name, data_cap)
    for i, image_path in enumerate(images_path):
        imageBase = image_to_base64(image_path)
        prompt = image_to_text
        image_info = gpt4v_runner(prompt, imageBase)
        print(image_info)
        df = load_df(f'experiments/{exp_dir}/', 'responses.csv')
        curr_row = {'image_path': image_path, 'gpt4-v response': image_info}
        df = pd.concat([df, pd.DataFrame(curr_row)],ignore_index=True)
        save_csv(df, f'experiments/{exp_dir}/responses.csv')

def run_experiments_test(exp_dir: str):
    filename = 'test.jpg'
    imageBase64 = image_to_base64(filename)
    prompt = "Describe what is in the given picture. Let's think segment by segment:"
    image_info = gpt4v_runner(prompt, imageBase64)
    print(image_info)
    df = load_df(f'experiments/{exp_dir}/', 'responses.csv')
    curr_row = {'image_path': [filename], 'gpt4-v response': [image_info]}
    df = pd.concat([df, pd.DataFrame(curr_row)], ignore_index=True)
    save_csv(df, f'experiments/{exp_dir}/responses.csv')    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, help='experiment directory')
    parser.add_argument('--data_name', type=str, default='imagenet-1k-1000', help='dataset name')
    parser.add_argument('--data_cap', type=int, default=100, help='dataset capacity')
    
    args = parser.parse_args()
    
    check_dir(f'experiments')
    check_dir(f'experiments/{args.exp_dir}')
    run_experiments_test(args.exp_dir)
    # run_experiments(args.exp_dir, args.data_name, args.data_cap)
    