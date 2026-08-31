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

from marigold import MarigoldDepthPipeline, MarigoldDepthOutput
from datasets.nyuv2 import NYUv2
from datasets.kitti import KITTI
from datasets.eth3d import ETH3D
from datasets.scannet import ScanNet
from datasets.diode import DIODE
from datasets.logger import init_log
from datasets.depth_metric import eval_depth


IMG_SIZE = 518
SAME_AREA = False

#python script/depth/eval_depth.py --checkpoint prs-eth/marigold-depth-v1-1 --output_dir output --fp16

def single_dataset_evaluation(dataset_name, testset, pipe, diffusion_config, logger, rotate_range, all_results, all_nsamples, output_dir):
    results = {'d1': torch.tensor([0.0]).cuda(), 'd2': torch.tensor([0.0]).cuda(), 'd3': torch.tensor([0.0]).cuda(), 
                'abs_rel': torch.tensor([0.0]).cuda(), 'sq_rel': torch.tensor([0.0]).cuda(), 'rmse': torch.tensor([0.0]).cuda(), 
                'rmse_log': torch.tensor([0.0]).cuda(), 'log10': torch.tensor([0.0]).cuda(), 'silog': torch.tensor([0.0]).cuda()}
    nsamples = torch.tensor([0.0]).cuda()
    invalid_samples = []

    for i, sample in tqdm(enumerate(testset)):
        
        img = sample['image'] # torch tensor
        depth = sample['depth'].float().cuda()
        valid_mask = sample['valid_mask'].cuda()

        # tensor to PIL Image
        img = np.transpose(img.numpy(), (1, 2, 0))
        img = (np.ascontiguousarray(img).astype(np.float32) * 255).astype(np.uint8)
        img = Image.fromarray(img)
        if i == 0:
            img.save(os.path.join(output_dir, f"{dataset_name}_{str(rotate_range)}.jpg"))

        with torch.no_grad():
            # Perform inference
            pipe_out: MarigoldDepthOutput = pipe(
                img,
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
            pred = F.interpolate(depth_pred, depth.shape[-2:], mode='bilinear', align_corners=True)[0, 0]
        
        valid_mask = (valid_mask == 1) & (depth > testset.min_depth) & (depth < testset.max_depth) 

        #pred = 1 / (pred + 1e-7)
        #depth = 1 / (depth + 1e-7)

        # less than 10 x 10 valid pixel image is meaningless
        if valid_mask.sum().item() < 100:
            invalid_samples.append((i, valid_mask.sum().item()))
            continue
        
        gt_masked = depth[valid_mask].reshape((-1, 1)).cpu().numpy()
        pred_masked = pred[valid_mask].reshape((-1, 1)).cpu().numpy()
        
        # numpy solver
        _ones = np.ones_like(pred_masked)
        A = np.concatenate([pred_masked, _ones], axis=-1)
        X = np.linalg.lstsq(A, gt_masked, rcond=None)[0]
        scale, shift = X
        aligned_pred = pred.squeeze() * torch.from_numpy(scale).to(pred.device) + torch.from_numpy(shift).to(pred.device)
        aligned_pred = aligned_pred.reshape(pred.shape)
        aligned_pred = torch.clamp(aligned_pred, min=testset.min_depth, max=testset.max_depth)
        
        cur_results = eval_depth(aligned_pred[valid_mask], depth[valid_mask])
        
        for k in results.keys():
            results[k] += cur_results[k]
        nsamples += 1
    

    # all results add gathered results from single test dataset
    for k in all_results.keys():
        all_results[k] += results[k]
    all_nsamples += nsamples
    
    logger.info(f"================ Evaluate Dataset: {dataset_name} With Rotation ({rotate_range}) ===============")
    logger.info(f"({dataset_name}) Valid Test Sample: {nsamples.item()}")
    #logger.info(f"Invalid Sample Index: {str(invalid_samples)}")
    logger.info('==========================================================================================')
    logger.info('{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}'.format(*tuple(results.keys())))
    logger.info('{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}'.format(*tuple([(v / nsamples).item() for v in results.values()])))
    logger.info('==========================================================================================')
    logger.info('\n')

    return all_results, all_nsamples



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

    # -------------------- Data --------------------
    dataset_list = ["DIODE", "ScanNet", "ETH3D", "KITTI", "NYU_V2"]

    for rotate_range in ((0, 0), (0, 15), (0, 45), (0, 90)):
        # evaluate different angle span
        all_results = {'d1': torch.tensor([0.0]).cuda(), 'd2': torch.tensor([0.0]).cuda(), 'd3': torch.tensor([0.0]).cuda(), 
                    'abs_rel': torch.tensor([0.0]).cuda(), 'sq_rel': torch.tensor([0.0]).cuda(), 'rmse': torch.tensor([0.0]).cuda(), 
                    'rmse_log': torch.tensor([0.0]).cuda(), 'log10': torch.tensor([0.0]).cuda(), 'silog': torch.tensor([0.0]).cuda()}
        all_nsamples = torch.tensor([0.0]).cuda()

        for dataset_name in dataset_list:
            size = (IMG_SIZE, IMG_SIZE)
            if dataset_name == "KITTI":
                testset = KITTI('./datasets/data_split/kitti/eigen_test_files_with_gt.txt', '/root/project/depth_data/kitti', 'test', size=size, rotate_range=rotate_range, same_area=SAME_AREA) 
            elif dataset_name == "ETH3D":
                testset = ETH3D('./datasets/data_split/eth3d/eth3d_filename_list.txt', '/root/project/depth_data/eth3d', 'test', size=size, rotate_range=rotate_range, same_area=SAME_AREA)
            elif dataset_name == "NYU_V2":
                testset = NYUv2('./datasets/data_split/nyu/labeled/filename_list_test.txt', '/root/project/depth_data/nyuv2', 'test', size=size, rotate_range=rotate_range, same_area=SAME_AREA)
            elif dataset_name == "ScanNet":
                testset = ScanNet("./datasets/data_split/scannet/scannet_val_sampled_list_800_1.txt", "/root/project/depth_data/scannet", size=size, rotate_range=rotate_range, same_area=SAME_AREA)
            elif dataset_name == "DIODE":
                testset = DIODE("./datasets/data_split/diode/diode_val_all_filename_list.txt", "/root/project/depth_data/diode", size=size, rotate_range=rotate_range, same_area=SAME_AREA)
            else:
                print(f"Invalid dataset: {dataset_name}")
            all_results, all_nsamples = single_dataset_evaluation(dataset_name, testset, pipe, diffusion_config, logger, rotate_range, all_results, all_nsamples, output_dir)

        logger.info(f"============================= Overall Eval Results With Rotation ({rotate_range}) ===================================")
        logger.info(f"All Evaluated Datasets: ({str(dataset_list)})")
        logger.info(f"All Valid Eval Sample: {all_nsamples.item()}")
        logger.info('{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}'.format(*tuple(all_results.keys())))
        logger.info('{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}'.format(*tuple([(v / all_nsamples).item() for v in all_results.values()])))
        logger.info('==========================================================================================')
        logger.info('\n')

    


