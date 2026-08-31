# Copyright 2023-2025 Marigold Team, ETH Zürich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# More information about Marigold:
#   https://marigoldmonodepth.github.io
#   https://marigoldcomputervision.github.io
# Efficient inference pipelines are now part of diffusers:
#   https://huggingface.co/docs/diffusers/using-diffusers/marigold_usage
#   https://huggingface.co/docs/diffusers/api/pipelines/marigold
# Examples of trained models and live demos:
#   https://huggingface.co/prs-eth
# Related projects:
#   https://rollingdepth.github.io/
#   https://marigolddepthcompletion.github.io/
# Citation (BibTeX):
#   https://github.com/prs-eth/Marigold#-citation
# If you find Marigold useful, we kindly ask you to cite our papers.
# --------------------------------------------------------------------------

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import logging
import numpy as np
import os
import torch
import torch.nn.functional as F
from PIL import Image
from glob import glob
from tqdm.auto import tqdm
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF

from marigold import MarigoldDepthPipeline, MarigoldDepthOutput
from datasets.nyuv2 import NYUv2
from datasets.kitti import KITTI
from datasets.eth3d import ETH3D
from datasets.scannet import ScanNet
from datasets.diode import DIODE
from datasets.logger import init_log
from datasets.depth_metric import eval_depth

#python script/depth/visualize_depth.py --checkpoint prs-eth/marigold-depth-v1-1 --output_dir output --fp16

image_path = "/root/project/depth_data/frame_00001.jpg"
model_name = "marigold"

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
        pipe_out: MarigoldDepthOutput = pipe(
                visualize_image(image),
                denoising_steps=diffusion_config['denoise_steps'],
                ensemble_size=diffusion_config['ensemble_size'],
                processing_res=diffusion_config['processing_res'],
                match_input_res=diffusion_config['match_input_res'],
                batch_size=diffusion_config['batch_size'],
                color_map=diffusion_config['color_map'],
                show_progress_bar=True,
                resample_method=diffusion_config['resample_method'],
                generator=diffusion_config['generator'],
            )

        depth_pred: np.ndarray = pipe_out.depth_np
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



if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Marigold : Monocular Depth Estimation : Multi-image Inference"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="prs-eth/marigold-depth-v1-1",
        help="Checkpoint path or hub name.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=None,
        help="Diffusion denoising steps, more steps results in higher accuracy but slower inference speed. If set to "
        "`None`, default value will be read from checkpoint.",
    )
    parser.add_argument(
        "--processing_res",
        type=int,
        default=None,
        help="Resolution to which the input is resized before performing estimation. `0` uses the original input "
        "resolution; `None` resolves the best default from the model checkpoint. Default: `None`",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=1,
        help="Number of predictions to be ensembled. Default: `1`.",
    )
    parser.add_argument(
        "--half_precision",
        "--fp16",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="Setting this flag will output the result at the effective value of `processing_res`, otherwise the "
        "output will be resized to the input resolution.",
    )
    parser.add_argument(
        "--resample_method",
        choices=["bilinear", "bicubic", "nearest"],
        default="bilinear",
        help="Resampling method used to resize images and predictions. This can be one of `bilinear`, `bicubic` or "
        "`nearest`. Default: `bilinear`",
    )
    parser.add_argument(
        "--color_map",
        type=str,
        default="Spectral",
        help="Colormap used to visualize depth predictions.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Reproducibility seed. Set to `None` for randomized inference. Default: `None`",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Inference batch size. Default: 0 (will be set automatically).",
    )
    parser.add_argument(
        "--apple_silicon",
        action="store_true",
        help="Use Apple Silicon for faster inference (subject to availability).",
    )

    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    output_dir = args.output_dir

    denoise_steps = args.denoise_steps
    ensemble_size = args.ensemble_size
    if ensemble_size > 15:
        logging.warning("Running with large ensemble size will be slow.")
    half_precision = args.half_precision

    processing_res = args.processing_res
    match_input_res = not args.output_processing_res
    if 0 == processing_res and match_input_res is False:
        logging.warning(
            "Processing at native resolution without resizing output might NOT lead to exactly the same resolution, "
            "due to the padding and pooling properties of conv layers."
        )
    resample_method = args.resample_method

    color_map = args.color_map
    seed = args.seed
    batch_size = args.batch_size
    apple_silicon = args.apple_silicon
    if apple_silicon and 0 == batch_size:
        batch_size = 1  # set default batchsize

    # -------------------- Preparation --------------------
    # Output directories
    os.makedirs(output_dir, exist_ok=True)
    logger = init_log('global', output_dir, rank=0)
    logger.info(f"output dir = {output_dir}")

    # -------------------- Device --------------------
    if apple_silicon:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            logger.warning("MPS is not available. Running on CPU will be slow.")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            logger.warning("CUDA is not available. Running on CPU will be slow.")
    logger.info(f"device = {device}")

    # -------------------- Model --------------------
    if half_precision:
        dtype = torch.float16
        variant = "fp16"
        logging.info(
            f"Running with half precision ({dtype}), might lead to suboptimal result."
        )
    else:
        dtype = torch.float32
        variant = None

    pipe: MarigoldDepthPipeline = MarigoldDepthPipeline.from_pretrained(
        checkpoint_path, variant=variant, torch_dtype=dtype
    )

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except ImportError:
        pass  # run without xformers

    pipe = pipe.to(device)
    logging.info(
        f"Loaded depth pipeline: scale_invariant={pipe.scale_invariant}, shift_invariant={pipe.shift_invariant}"
    )

    # Print out config
    logging.info(
        f"Inference settings: checkpoint = `{checkpoint_path}`, "
        f"with denoise_steps = {denoise_steps or pipe.default_denoising_steps}, "
        f"ensemble_size = {ensemble_size}, "
        f"processing resolution = {processing_res or pipe.default_processing_resolution}, "
        f"seed = {seed}; "
        f"color_map = {color_map}."
    )

    # Random number generator
    if seed is None:
        generator = None
    else:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    # diffusion setting
    diffusion_config = {'denoise_steps' : denoise_steps,
                        'ensemble_size' : ensemble_size,
                        'processing_res' : processing_res,
                        'match_input_res' : match_input_res,
                        'batch_size' : batch_size,
                        'color_map' : color_map,
                        'resample_method' : resample_method,
                        'generator' : generator}

    
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


