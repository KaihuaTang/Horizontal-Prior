import argparse
import logging
import os
import cv2
import imageio
import time
import pprint
import random
from PIL import Image
from tqdm import tqdm
import matplotlib
import warnings
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision.transforms import Compose
import torch.nn.functional as F
from torchvision.transforms import functional as TF

from moge.model.v2 import MoGeModel

ckpt_path = "/home/couser/checkpoints/moge-2-vitl-normal/model.pt"

image_path = "/data/tangkaihua/datasets/open_scene/image/frame_00001.jpg"

size = (560, 560)
keep_aspect_ratio = False
model_name = "moge2" #"dav2" #"distill" #"rotate_pda" # "promptda"

os.makedirs(f"./visualization_{model_name}", exist_ok=True)

    
def visualize_depth(depth: np.ndarray, 
                    depth_min=None, 
                    depth_max=None, 
                    percentile=2, 
                    ret_minmax=False,
                    reverse=True,
                    cmap='Spectral'):
    if depth_min is None: depth_min = np.percentile(depth, percentile)
    if depth_max is None: depth_max = np.percentile(depth, 100 - percentile)
    if depth_min == depth_max:
        depth_min = depth_min - 1e-6
        depth_max = depth_max + 1e-6
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - depth_min) / (depth_max - depth_min)).clip(0, 1)
    if reverse:
        depth = 1.0 - depth
    img_colored_np = cm(depth[None], bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = (img_colored_np[0] * 255.0).astype(np.uint8)
    return Image.fromarray(img_colored_np)


def visualize_image(image_tensor):
    image = image_tensor.permute(1,2,0).numpy()
    image = (image * 255).astype(np.uint8)
    return Image.fromarray(image)


def multiple_of_14(x):
        y = (np.round(x / 14) * 14).astype(int)
        if y < size[0]:
            y = (np.ceil(x / 14) * 14).astype(int)
        return y
    
def tensor_resize_to_14multiple(image):
    _, height, width = image.shape
    scale = 1 # size[0] / min(height, width)
    new_size = (multiple_of_14(int(scale * height)), multiple_of_14(int(scale * width)))
    image = TF.resize(image, size=new_size, interpolation=TF.InterpolationMode.BILINEAR)
    return image

def tensor_rotate(image, angle):
    rotate_image = TF.rotate(image, angle=angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0, expand=True)
    return rotate_image

def crop_with_ratio(image, ratio, keep_aspect_ratio):
    _, height, width = image.shape
    crop_height = int(height * ratio)
    crop_width = int(width * ratio)
    if not keep_aspect_ratio:
        crop_height = min(crop_height, crop_width)
        crop_width = crop_height
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    return image[:, top:top+crop_height, left:left+crop_width]

def center_crop(image, orig_shape):
    _, height, width = image.shape
    _, orig_h, orig_w = orig_shape
    top = (height - orig_h) // 2
    left = (width - orig_w) // 2
    return image[:, top:top+orig_h, left:left+orig_w]


def get_image(rotate_angle, ratio, keep_aspect_ratio):
    # B, C, H, W
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    image = torch.tensor(image / 255, dtype=torch.float32).permute(2, 0, 1)
    image = tensor_rotate(image, 90)
    
    # rotate (padding only) and resize
    image = tensor_resize_to_14multiple(image)
    image = tensor_rotate(image, rotate_angle)
    image = crop_with_ratio(image, ratio=ratio, keep_aspect_ratio=keep_aspect_ratio)
    return image

device = torch.device("cuda")
model = MoGeModel.from_pretrained(ckpt_path).to(device).eval()

def get_input_image(rotate_angle):
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    image = torch.tensor(image / 255, dtype=torch.float32).permute(2, 0, 1)
    image = tensor_rotate(image, 90)

    orig_shape = image.shape
    
    # rotate (padding only) and resize
    image = tensor_rotate(image, rotate_angle)

    #rotate_image = crop_with_ratio(rotate_image, ratio=ratio, keep_aspect_ratio=keep_aspect_ratio)
    _, height, width = image.shape
    image = tensor_resize_to_14multiple(image)
    
    return image, height, width, orig_shape

def get_depth(rotate_angle, ratio, keep_aspect_ratio, ground_truth=None, return_org=False, display_angle=None):
    image, height, width, orig_shape = get_input_image(rotate_angle)
    # get depth prediction
    with torch.no_grad():
        output = model.infer(image.cuda().unsqueeze(0))
        pred = output["depth"].unsqueeze(0)
        #pred = model.predict(image.cuda().unsqueeze(0), prompt_depth.cuda().unsqueeze(0))
        pred = F.interpolate(pred, (height, width), mode='bilinear', align_corners=True)[0]
        pred = pred.cpu().numpy()

        if ground_truth is not None:
            ground_truth = torch.from_numpy(ground_truth).unsqueeze(0)
            rotate_gt = TF.rotate(ground_truth, angle=rotate_angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0, expand=True)
            _, height, width = pred.shape
            pred_masked = pred[rotate_gt > 0].reshape((-1, 1))
            gt_masked = rotate_gt[rotate_gt > 0].reshape((-1, 1))
            
            _ones = np.ones_like(pred_masked)
            A = np.concatenate([pred_masked, _ones], axis=-1)
            X = np.linalg.lstsq(A, gt_masked, rcond=None)[0]
            scale, shift = X
            aligned_pred = pred * scale + shift
            aligned_pred = aligned_pred.reshape(pred.shape)
            pred = aligned_pred
        if return_org:
            return pred[0]
        elif display_angle is not None:
            reverse_pred = TF.rotate(torch.from_numpy(pred), angle=-rotate_angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0, expand=True)
            crop_pred = center_crop(reverse_pred, orig_shape)
            rotate_pred = TF.rotate(crop_pred, angle=display_angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0, expand=True)
            final_pred = crop_with_ratio(rotate_pred, ratio=ratio, keep_aspect_ratio=keep_aspect_ratio)
            return final_pred[0].cpu().numpy()
        else:
            print(f"pred: {pred.shape}")
            #pred = TF.rotate(torch.from_numpy(pred), angle=-20, interpolation=TF.InterpolationMode.BILINEAR, fill=0, expand=True).numpy()
            crop_pred = crop_with_ratio(pred, ratio=ratio, keep_aspect_ratio=keep_aspect_ratio)
            return crop_pred[0]

def get_diff(pred, ground_truth, rotate_angle, ratio, keep_aspect_ratio):
    ground_truth = torch.from_numpy(ground_truth).unsqueeze(0)
    rotate_gt = TF.rotate(ground_truth, angle=rotate_angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0, expand=True)
    crop_gt = crop_with_ratio(rotate_gt, ratio=ratio, keep_aspect_ratio=keep_aspect_ratio)
    crop_gt = crop_gt.numpy()[0]

    height, width = pred.shape
    pred_masked = pred[crop_gt > 0].reshape((-1, 1))
    gt_masked = crop_gt[crop_gt > 0].reshape((-1, 1))
    
    _ones = np.ones_like(pred_masked)
    A = np.concatenate([pred_masked, _ones], axis=-1)
    X = np.linalg.lstsq(A, gt_masked, rcond=None)[0]
    scale, shift = X
    aligned_pred = pred * scale + shift
    aligned_pred = aligned_pred.reshape(pred.shape)
    #aligned_pred = torch.clamp(aligned_pred, min=testloader.dataset.min_disparity, max=testloader.dataset.max_disparity)
    pred_diff = np.abs(aligned_pred - crop_gt)
    return pred_diff

def ground_truth_norm(ground_truth):
    gt_min = ground_truth.min()
    gt_max = ground_truth.max()
    return (ground_truth - gt_min) / (gt_max - gt_min) * 500


# get ground truth first
ground_truth = get_depth(0, 1.0, True)
ground_truth = ground_truth_norm(ground_truth)
visualize_depth(ground_truth, reverse=False).save(f"./visualization_{model_name}/{model_name}_gt_horizon_depth.jpg")


for rotate_angle, ratio in ((0, 0.9), (-10, 0.7), (-15, 0.6), (-30, 0.45), (-45, 0.43), (-75, 0.6), (-90, 0.8)):
    pred = get_depth(rotate_angle, ratio, keep_aspect_ratio, ground_truth=ground_truth)
    result_depth = visualize_depth(pred, reverse=False)
    result_depth.save(f"./visualization_{model_name}/{model_name}_angle{int(-rotate_angle)}_ratio{int(ratio * 100)}_depth.jpg")

    image = get_image(rotate_angle, ratio, keep_aspect_ratio)
    result_image = visualize_image(image)
    result_image.save(f"./visualization_{model_name}/{model_name}_angle{int(-rotate_angle)}_ratio{int(ratio * 100)}.jpg")

    pred_diff = get_diff(pred, ground_truth, rotate_angle, ratio, keep_aspect_ratio)
    print(f"Prediction Difference: {pred_diff.max()}")
    result_diff = visualize_depth(pred_diff, 0, 350, cmap="gist_heat")
    result_diff.save(f"./visualization_{model_name}/{model_name}_angle{int(-rotate_angle)}_ratio{int(ratio * 100)}_diff.jpg")


    
