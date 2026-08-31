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
import matplotlib.pyplot as plt

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
import torch.nn.functional as F
from torchvision import transforms
#from datasets import load_dataset

import safetensors
from diffusers import UNet2DConditionModel, AutoencoderKL

from genpercept.util.seed_all import seed_all
from genpercept.util.image_util import ResizeHard
from genpercept.pipeline_genpercept import GenPerceptPipeline

from genpercept.models import CustomUNet2DConditionModel # DPTHead, 
from genpercept.models.dpt_head_elu import DPTNeckHeadForUnetAfterUpsample

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
        img = Image.fromarray(img).convert("RGB")
        if i == 0:
            img.save(os.path.join(output_dir, f"{dataset_name}_{str(rotate_range)}.jpg"))

        with torch.no_grad():
            # Perform inference
            pipe_out = pipe(
                img,
                processing_res=diffusion_config['processing_res'],
                match_input_res=diffusion_config['match_input_res'],
                batch_size=diffusion_config['batch_size'],
                color_map=diffusion_config['color_map'],
                show_progress_bar=True,
                mode='depth',
            )

            depth_pred: np.ndarray = pipe_out.pred_np
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

