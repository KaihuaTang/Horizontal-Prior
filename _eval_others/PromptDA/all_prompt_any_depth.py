import os
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib
#import open3d as o3d
#from scipy.interpolate import CubicSpline

from argparse import ArgumentParser
from promptda.promptda import PromptDA
from promptda.utils.io_wrapper import load_image, load_depth, save_depth

def visualize_depth(depth: np.ndarray,
                    depth_min=None,
                    depth_max=None,
                    percentile=2,
                    ret_minmax=False,
                    cmap='Spectral'):
    if depth_min is None: depth_min = np.percentile(depth, percentile)
    if depth_max is None: depth_max = np.percentile(depth, 100 - percentile)
    if depth_min == depth_max:
        depth_min = depth_min - 1e-6
        depth_max = depth_max + 1e-6
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - depth_min) / (depth_max - depth_min)).clip(0, 1)
    img_colored_np = cm(depth[None], bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = (img_colored_np[0] * 255.0).astype(np.uint8)
    if ret_minmax:
        return img_colored_np, depth_min, depth_max
    else:
        return img_colored_np

def main(args):
    print(f"Loading model")
    DEVICE = 'cuda'
    model_path = "/home/couser/projects/PromptDA/checkpoints/large/model.ckpt"
    model = PromptDA.from_pretrained(model_path).to(DEVICE).eval()

    print(f"Start Inference")
    jpg_files = []
    dpt_files = []
    jpg_names = []
    for file in os.listdir(args.image_folder):
        if file.lower().endswith(args.image_type):
            jpg_files.append(os.path.join(args.image_folder, file))
            dpt_files.append(os.path.join(args.depth_folder, '.'.join(file.split('.')[:-1]) + '.png'))
            jpg_names.append('.'.join(file.split('.')[:-1]))

    for image_file, depth_file, image_name in tqdm(zip(jpg_files, dpt_files, jpg_names)):
        # load image
        image = load_image(image_file, max_size=1920).to(DEVICE)
        prompt_depth = load_depth(depth_file).to(DEVICE)
        # depth estimation
        image90 = torch.rot90(image, k=1, dims=(-2, -1))
        prompt_depth90 = torch.rot90(prompt_depth, k=1, dims=(-2, -1))
        depth90 = model.predict(image90, prompt_depth90)
        depth = torch.rot90(depth90, k=1, dims=(-1, -2))

        # Saving depth to npy
        os.makedirs(args.output_folder, exist_ok=True)
        np.save(os.path.join(args.output_folder, image_name + "_depth.npy"), depth.cpu().numpy())
        # Saving depth to image
        save_depth(depth, output_path=os.path.join(args.output_folder, image_name + "_depth.png"),)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--depth-folder", type=str, required=True)
    parser.add_argument("--output-folder", type=str, required=True)
    parser.add_argument("--image-type", type=str, default=".jpg")
    args = parser.parse_args()
    main(args)

# CUDA_VISIBLE_DEVICES=0 python all_depth_anything.py --image-folder ./samples/ --output-folder ./samples/
                                                                                                               
