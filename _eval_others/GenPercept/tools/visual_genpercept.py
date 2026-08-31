# --------------------------------------------------------
# Diffusion Models Trained with Large Data Are Transferable Visual Models (https://arxiv.org/abs/2403.06090)
# Github source: https://github.com/aim-uofa/GenPercept
# Copyright (c) 2024 Zhejiang University
# Licensed under The CC0 1.0 License [see LICENSE for details]
# By Guangkai Xu
# Based on Marigold, diffusers codebases
# https://github.com/prs-eth/marigold
# https://github.com/huggingface/diffusers
# --------------------------------------------------------


import argparse
import os
import os.path as osp
from glob import glob
import logging
import cv2
import time
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
#from datasets import load_dataset

import safetensors
from diffusers import UNet2DConditionModel, AutoencoderKL

from genpercept.util.seed_all import seed_all
from genpercept.util.image_util import ResizeHard
from genpercept.pipeline_genpercept import GenPerceptPipeline

from genpercept.models import CustomUNet2DConditionModel # DPTHead, 
from genpercept.models.dpt_head_elu import DPTNeckHeadForUnetAfterUpsample

from datasets.logger import init_log


image_path = "/root/project/depth_data/frame_00001.jpg"
model_name = "genpercept"

IMG_SIZE = 560
SAME_AREA = False
size = (560, 560)
keep_aspect_ratio = False
os.makedirs("./visualization", exist_ok=True)


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
    scale = size[0] / min(height, width)
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
    image = Image.open(image_path).convert("RGB")
    image = np.asarray(image) / 255.0
    image = np.rot90(image, k=1)
    h, w = image.shape[:2]
    image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)
    
    # rotate (padding only) and resize
    image = tensor_resize_to_14multiple(image)
    image = tensor_rotate(image, rotate_angle)
    image = crop_with_ratio(image, ratio=ratio, keep_aspect_ratio=keep_aspect_ratio)
    return image


def get_input_image(rotate_angle):
    image = Image.open(image_path).convert("RGB")
    image = np.asarray(image) / 255.0
    image = np.rot90(image, k=1)
    h, w = image.shape[:2]
    image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)
    orig_shape = image.shape
    
    # rotate (padding only) and resize
    image = tensor_rotate(image, rotate_angle)
    #rotate_image = crop_with_ratio(rotate_image, ratio=ratio, keep_aspect_ratio=keep_aspect_ratio)
    _, height, width = image.shape
    image = tensor_resize_to_14multiple(image)
    
    return image, height, width, orig_shape

def get_depth(pipe, diffusion_config, rotate_angle, ratio, keep_aspect_ratio, ground_truth=None, return_org=False, display_angle=None):
    image, height, width, orig_shape = get_input_image(rotate_angle)
    # get depth prediction
    with torch.no_grad():
        pipe_out = pipe(
                visualize_image(image),
                processing_res=diffusion_config['processing_res'],
                match_input_res=diffusion_config['match_input_res'],
                batch_size=diffusion_config['batch_size'],
                color_map=diffusion_config['color_map'],
                show_progress_bar=True,
                mode='depth',
            )

        depth_pred: np.ndarray = pipe_out.pred_np
        depth_pred = F.relu(torch.from_numpy(depth_pred).float().cuda().unsqueeze(0).unsqueeze(0))
        pred = F.interpolate(depth_pred, (height, width), mode='bilinear', align_corners=True)[0]
        pred = pred.cpu().numpy()

        if ground_truth is not None:
            ground_truth = torch.from_numpy(ground_truth).unsqueeze(0)
            rotate_gt = TF.rotate(ground_truth, angle=rotate_angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0, expand=True)
            rotate_gt = rotate_gt.numpy()
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


#python script/depth/eval_depth.py --checkpoint prs-eth/marigold-depth-v1-1 --output_dir output --fp16

if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="GenPercept inference."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Checkpoint path or hub name.",
    )
    parser.add_argument(
        "--save_pcd",
        default=False,
        action="store_true",
        help="Save point cloud while evaluting depth.",
    )
    parser.add_argument(
        "--unet_ckpt_path",
        type=str,
        default=None,
        help="Checkpoint path for unet.",
    )
    parser.add_argument(
        "--vae_ckpt_path",
        type=str,
        default=None,
        help="Checkpoint path for vae.",
    )
    parser.add_argument(
        "--customized_head_name",
        type=str,
        default=None,
        help="Customized head to replace the VAE decoder",
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=['depth', 'seg', 'normal'],
        default="depth",
        help="inference mode.",
    )

    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )
    parser.add_argument(
        "--half_precision",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )

    # resolution setting
    parser.add_argument(
        "--processing_res",
        type=int,
        default=768,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 768.",
    )
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="When input is resized, output label at resized operating resolution. Default: False.",
    )

    # depth map colormap
    parser.add_argument(
        "--color_map",
        type=str,
        default="Spectral",
        help="Colormap used to render depth predictions.",
    )

    # other settings
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Inference batch size. Default: 0 (will be set automatically).",
    )
    parser.add_argument(
        "--apple_silicon",
        action="store_true",
        help="Flag of running on Apple Silicon.",
    )

    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    output_dir = args.output_dir

    half_precision = args.half_precision

    processing_res = args.processing_res
    match_input_res = not args.output_processing_res

    color_map = args.color_map
    seed = args.seed
    batch_size = args.batch_size
    apple_silicon = args.apple_silicon
    if apple_silicon and 0 == batch_size:
        batch_size = 1  # set default batchsize

    # -------------------- Preparation --------------------
    # Random seed
    if seed is None:
        import time
        seed = int(time.time())
        print('seed {}'.format(seed))

    seed_all(seed)

    # Output directories
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"output dir = {output_dir}")
    logger = init_log('global', output_dir, rank=0)
    logger.info(f"output dir = {output_dir}")

    # -------------------- Device --------------------
    if apple_silicon:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps:0")
        else:
            device = torch.device("cpu")
            logging.warning("MPS is not available. Running on CPU will be slow.")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            logging.warning("CUDA is not available. Running on CPU will be slow.")
    logger.info(f"device = {device}")


    # -------------------- Model --------------------
    if half_precision:
        dtype = torch.float16
        logger.info(f"Running with half precision ({dtype}).")
    else:
        dtype = torch.float32
    
    # unet
    unet_ckpt_path = args.unet_ckpt_path if args.unet_ckpt_path is not None else checkpoint_path
    unet = CustomUNet2DConditionModel.from_config(
        unet_ckpt_path, subfolder="unet", revision=args.non_ema_revision
    )
    try:
        load_ckpt_unet = safetensors.torch.load_file(osp.join(unet_ckpt_path, 'unet', 'diffusion_pytorch_model.safetensors'))
    except:
        load_ckpt_unet = safetensors.torch.load_file(osp.join(unet_ckpt_path, 'diffusion_pytorch_model.safetensors'))
    if not any('conv_out' in key for key in load_ckpt_unet.keys()):
        unet.conv_out = None
    if not any('conv_norm_out' in key for key in load_ckpt_unet.keys()):
        unet.conv_norm_out = None
    unet.load_state_dict(load_ckpt_unet)
    
    # vae
    vae_ckpt_path = args.vae_ckpt_path if args.vae_ckpt_path is not None else checkpoint_path
    vae = AutoencoderKL.from_config(
        vae_ckpt_path, subfolder="vae",
    )
    load_ckpt_vae = safetensors.torch.load_file(osp.join(vae_ckpt_path, 'vae', 'diffusion_pytorch_model.safetensors'))
    if not any('decoder' in key for key in load_ckpt_vae.keys()):
        vae.decoder = None
    if not any('post_quant_conv' in key for key in load_ckpt_vae.keys()):
        vae.post_quant_conv = None
    vae.load_state_dict(load_ckpt_vae)
    
    # customized head
    customized_head = None
    if args.customized_head_name is not None:
        if args.customized_head_name == 'dpt_head':
            cfgs = "configs_hf/dpt-sd2.1-unet-after-upsample"
            customized_head = DPTNeckHeadForUnetAfterUpsample.from_pretrained(checkpoint_path, subfolder="dpt_head")
        else:
            raise NotImplementedError
    
        customized_head = customized_head.to(device)
    
    empty_text_embed = torch.from_numpy(np.load("empty_text_embed.npy")).to(device, dtype)[None] # [1, 77, 1024]

    genpercept_params_ckpt = dict(
        unet=unet,
        vae=vae,
        empty_text_embed=empty_text_embed,
        customized_head=customized_head,
    )

    pipe = GenPerceptPipeline(**genpercept_params_ckpt)

    pipe = pipe.to(device).to(dtype)
    pipe.set_progress_bar_config(disable=True)

    try:
        import xformers
        pipe.enable_xformers_memory_efficient_attention()
    except:
        pass  # run without xformers


    # diffusion setting
    diffusion_config = {'processing_res' : processing_res,
                        'match_input_res' : match_input_res,
                        'batch_size' : batch_size,
                        'color_map' : color_map}

    # get ground truth first
    ground_truth = get_depth(pipe, diffusion_config, 0, 1.0, True)
    ground_truth = ground_truth_norm(ground_truth)
    visualize_depth(ground_truth, reverse=False).save(f"./visualization/{model_name}_gt_horizon_depth.jpg")


    for rotate_angle, ratio in ((0, 0.9), (-10, 0.7), (-15, 0.6), (-30, 0.45), (-45, 0.43), (-75, 0.6), (-90, 0.8)):
        pred = get_depth(pipe, diffusion_config, rotate_angle, ratio, keep_aspect_ratio, ground_truth=ground_truth)
        result_depth = visualize_depth(pred, reverse=False)
        result_depth.save(f"./visualization/{model_name}_angle{int(-rotate_angle)}_ratio{int(ratio * 100)}_depth.jpg")

        image = get_image(rotate_angle, ratio, keep_aspect_ratio)
        result_image = visualize_image(image)
        result_image.save(f"./visualization/{model_name}_angle{int(-rotate_angle)}_ratio{int(ratio * 100)}.jpg")

        pred_diff = get_diff(pred, ground_truth, rotate_angle, ratio, keep_aspect_ratio)
        print(f"Prediction Difference: {pred_diff.max()}")
        result_diff = visualize_depth(pred_diff, 0, 350, cmap="gist_heat")
        result_diff.save(f"./visualization/{model_name}_angle{int(-rotate_angle)}_ratio{int(ratio * 100)}_diff.jpg")
