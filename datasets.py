import os
import random
import shutil

# TODO: add some datasets here
def get_dataset(dataset_name: str = 'imagenet-1k-1000', dataset_cap: int = 100) -> list[str]:
    dir_path = f"./datasets/{dataset_name}"
    all_files = [os.path.join(dir_path, file) for file in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, file))]
    sampled_files = random.sample(all_files, dataset_cap)
    return sampled_files