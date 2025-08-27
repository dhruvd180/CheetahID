import os
import random
import shutil
from math import ceil
raw_dir = "data/raw"
test_dir = "data/test3"
os.makedirs(test_dir, exist_ok=True)
def split_test_train(splitratio):
    for folder in sorted(os.listdir(raw_dir)):
        folder_path = os.path.join(raw_dir, folder)
        if not os.path.isdir(folder_path) or not folder.startswith("individual_"):
            continue
        images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not images:
            continue
        splitratiodecimalized = splitratio/100
        num_to_pick = max(1, ceil(len(images) * splitratiodecimalized))
        selected_images = random.sample(images, num_to_pick)
        individual_num = folder.split("_")[-1] 
        individual_num_int = int(individual_num)
        for idx, img_name in enumerate(selected_images, start=1):
            src_path = os.path.join(folder_path, img_name)
            new_name = f"individual{individual_num_int} test _{idx:02d}{os.path.splitext(img_name)[1]}"
            dst_path = os.path.join(test_dir, new_name)
            shutil.copy2(src_path, dst_path)
    print(splitratio,"% of images copied to data/test3 with standardnames.")
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--split', type=int, default=20)     
args = parser.parse_args()   
split_test_train(args.split)