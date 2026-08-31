import os
import urllib.request
from tqdm import tqdm

def load_file_to_dict(file_path):
    file_dict = {}
    with open(file_path, 'r') as file:
        # Skip the header line
        next(file)
        for line in file:
            # Split each line into parts
            parts = line.strip().split('\t')
            if len(parts) == 2:
                filename, cdn_link = parts
                file_dict[filename] = cdn_link
    return file_dict

file_path = './datasets/prepare/SA-1B.txt'
file_dict = load_file_to_dict(file_path)

data_size = 20
data_folder = "./datasets/SA-1B"
os.makedirs(data_folder, exist_ok=True)

for i in tqdm(range(data_size)):
    index = int(i * (1000 // data_size))
    name = "sa_" + str(index).zfill(6) + ".tar"
    url = file_dict[name]
    print(f"Downloading {name}")
    urllib.request.urlretrieve(url, os.path.join(data_folder, name))

# download ckpt
# sa_000000.tar
# sa_000050.tar
# sa_000100.tar
# ......
# sa_000950.tar