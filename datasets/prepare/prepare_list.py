import os
import cv2
from tqdm import tqdm

folder_path = "./datasets/SA-1B/images"
sa1b_list = "./SA1B_all.txt"
count = 0
with open(sa1b_list, 'w', encoding='utf-8') as txt_file:
    for filename in tqdm(os.listdir(folder_path)):
        if filename.lower().endswith('.jpg'):
            file_path = os.path.join(folder_path, filename)
            try:
                image = cv2.imread(file_path)
                height, width = image.shape[:2]
                if height > 0 and width > 0:
                    txt_file.write(f"{file_path}\n")
                else:
                    print(f"Wrong image size ({height}, {width}): {file_path}")
            except:
                print(f"Fail to load {file_path}")

print("Seperate into train and test")
import random
random.seed(42)

sa1b_list = "./SA1B_all.txt"
sa1b_train = "./SA1B_train.txt"
sa1b_val = "./SA1B_val.txt"

with open(sa1b_list, 'r') as f:
    filelist = f.read().splitlines()
random.shuffle(filelist)

train_size = int(0.9 * len(filelist))
# Split into 80% and 20%
list_train = filelist[:train_size]
list_val = filelist[train_size:]

with open(sa1b_train, 'w', encoding='utf-8') as txt_file:
    for item in list_train:
        txt_file.write(f"{item}\n")

with open(sa1b_val, 'w', encoding='utf-8') as txt_file:
    for item in list_val:
        txt_file.write(f"{item}\n")