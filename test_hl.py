# Author: Tang Kaihua
# Our evaluation code are built following:
# 1. https://github.com/isl-org/DPT/blob/main/EVALUATION.md#genral-purpose-models
# 2. https://github.com/DepthAnything/Depth-Anything-V2/blob/main/metric_depth/util/metric.py
# 3. https://github.com/aim-uofa/GenPercept
# 4. https://github.com/nianticlabs/monodepth2#-kitti-evaluation
# following Depth Anything to evaluate on disparity space
# https://github.com/LiheYoung/Depth-Anything/issues/174

import argparse
import csv
import logging
import os
import time
import pprint
import random
from tqdm import tqdm

import warnings
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from utils.utils_general import format_time_interval, load_from_checkpoint
from utils.utils_logger import init_log
from utils.utils_metric import eval_depth
from models.dav2 import DepthAnythingV2
from models.depthcue import DepthCueModel
from datasets.nyuv2 import NYUv2
from datasets.kitti import KITTI
from datasets.eth3d import ETH3D
from datasets.scannet import ScanNet
from datasets.diode import DIODE

parser = argparse.ArgumentParser(description='Depth Anything V2 Evaluation with Horizontal Leveling (HL)')

parser.add_argument('--encoder', default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
parser.add_argument('--img-size', default=518, type=int)
parser.add_argument('--angle-path', type=str, required=True)
parser.add_argument('--ckpt-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--testdata', default="all", type=str)

SAME_AREA = False

def main():
    args = parser.parse_args()
    
    os.makedirs(args.save_path, exist_ok=True)
    logger = init_log('global', args.save_path, rank=0)

    all_args = {**vars(args)}
    logger.info('{}\n'.format(pprint.pformat(all_args)))
    
    cudnn.enabled = True
    cudnn.benchmark = True
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    model = DepthAnythingV2(**model_configs[args.encoder])
    model = load_from_checkpoint(model, args.ckpt_path, logger)
    model = model.cuda().eval()

    # angle prediction model
    angle_model = DepthCueModel(encoder='vits', hidden_dim=768, layers=[8, 11], 
                                 task_configs={'names': ['angle'], 'losses' : ['l1_and_cos'], 'channels': [2], 'use_local_cue_teacher': False})
    angle_model = load_from_checkpoint(angle_model, args.angle_path, logger=logger)
    angle_model = angle_model.cuda().eval()

    logger.info(str(model))
    
    if args.testdata == "all":
        dataset_list = ["DIODE", "ScanNet", "ETH3D", "KITTI", "NYU_V2"]
    else:
        dataset_list = [args.testdata]
    logger.info(f"Start Evaluating Datasets: ({str(dataset_list)})")

    #for rotate_range in ((0, 0), (0, 15), (15, 30), (30, 45), (45, 60), (60, 75), (75, 90)):
    for rotate_range in ((0, 0), (0, 15), (0, 45), (0, 90)):
        # evaluate different angle span
        all_results = {'d1': torch.tensor([0.0]).cuda(), 'd2': torch.tensor([0.0]).cuda(), 'd3': torch.tensor([0.0]).cuda(),
                    'abs_rel': torch.tensor([0.0]).cuda(), 'sq_rel': torch.tensor([0.0]).cuda(), 'rmse': torch.tensor([0.0]).cuda(),
                    'rmse_log': torch.tensor([0.0]).cuda(), 'log10': torch.tensor([0.0]).cuda(), 'silog': torch.tensor([0.0]).cuda()}
        all_nsamples = torch.tensor([0.0]).cuda()
        # NEW: list of per-sample dicts, written to CSV at end of this rotate_range
        per_sample_records = []

        for dataset_name in dataset_list:
            size = (args.img_size, args.img_size)
            if dataset_name == "KITTI":
                testset = KITTI('./datasets/data_split/kitti/eigen_test_files_with_gt.txt', '/home/couser/datasets/depth_datasets/kitti', 'test', size=size, rotate_range=rotate_range, same_area=SAME_AREA)
            elif dataset_name == "ETH3D":
                testset = ETH3D('./datasets/data_split/eth3d/eth3d_filename_list.txt', '/home/couser/datasets/depth_datasets/eth3d', 'test', size=size, rotate_range=rotate_range, same_area=SAME_AREA)
            elif dataset_name == "NYU_V2":
                testset = NYUv2('./datasets/data_split/nyu/labeled/filename_list_test.txt', '/home/couser/datasets/depth_datasets/nyuv2', 'test', size=size, rotate_range=rotate_range, same_area=SAME_AREA)
            elif dataset_name == "ScanNet":
                testset = ScanNet("./datasets/data_split/scannet/scannet_val_sampled_list_800_1.txt", "/home/couser/datasets/depth_datasets/scannet", size=size, rotate_range=rotate_range, same_area=SAME_AREA)
            elif dataset_name == "DIODE":
                testset = DIODE("./datasets/data_split/diode/diode_val_all_filename_list.txt", "/home/couser/datasets/depth_datasets/diode", size=size, rotate_range=rotate_range, same_area=SAME_AREA)
            else:
                print(f"Invalid dataset: {dataset_name}")
            testloader = DataLoader(testset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False, shuffle=False)
            all_results, all_nsamples = single_dataset_evaluation(args, dataset_name, testloader, model, angle_model, logger, rotate_range, all_results, all_nsamples, per_sample_records)

        logger.info(f"============================= Overall Eval Results With Rotation ({rotate_range}) ===================================")
        logger.info(f"All Evaluated Datasets: ({str(dataset_list)})")
        logger.info(f"All Valid Eval Sample: {all_nsamples.item()}")
        logger.info('{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}'.format(*tuple(all_results.keys())))
        logger.info('{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}'.format(*tuple([(v / all_nsamples).item() for v in all_results.values()])))
        logger.info('==========================================================================================')
        logger.info('\n')

        # NEW: dump per-sample records for this rotate_range to CSV (for later bootstrap CI analysis)
        if len(per_sample_records) > 0:
            csv_name = f'per_sample_rotate_{rotate_range[0]}_{rotate_range[1]}.csv'
            csv_path = os.path.join(args.save_path, csv_name)
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(per_sample_records[0].keys()))
                writer.writeheader()
                writer.writerows(per_sample_records)
            logger.info(f"[per-sample] saved {len(per_sample_records)} rows to {csv_path}")
            logger.info('\n')



def single_dataset_evaluation(args, dataset_name, testloader, model, angle_model, logger, rotate_range, all_results, all_nsamples, per_sample_records=None):
    results = {'d1': torch.tensor([0.0]).cuda(), 'd2': torch.tensor([0.0]).cuda(), 'd3': torch.tensor([0.0]).cuda(),
                'abs_rel': torch.tensor([0.0]).cuda(), 'sq_rel': torch.tensor([0.0]).cuda(), 'rmse': torch.tensor([0.0]).cuda(),
                'rmse_log': torch.tensor([0.0]).cuda(), 'log10': torch.tensor([0.0]).cuda(), 'silog': torch.tensor([0.0]).cuda()}
    nsamples = torch.tensor([0.0]).cuda()
    invalid_samples = []
    if per_sample_records is None:
        per_sample_records = []  # caller didn't pass list; use a local one (won't be saved)

    for i, sample_i in tqdm(enumerate(testloader)):
        
        # original image used for angle prediction
        img = sample_i['image'].cuda().float()

        with torch.no_grad():
            pred_angle = angle_model(img)[0]
            #if args.use_activate == 'tanh':
            pred_angle = torch.tanh(pred_angle)
            pred_angle = pred_angle.detach().cpu().numpy()
            pred_cos = pred_angle[:, 0]
            pred_sin = pred_angle[:, 1]
            pred_deg = np.degrees(np.arctan2(pred_sin, pred_cos))
            # try to recover horizontal image        
            final_angles = sample_i['rotate_angle'].numpy() - pred_deg
            items = sample_i['item']

        # get horizontal image used for depth prediction
        sample = testloader.dataset.get_sample_with_angle(items[0].item(), float(final_angles[0]), force_mask=True)
        img = sample['image'].unsqueeze(0).cuda().float()
        depth = sample['depth'].cuda()
        valid_mask = sample['valid_mask'].cuda()

        with torch.no_grad():
            pred = model(img, nonzero=True) 
            pred = F.interpolate(pred[:, None], depth.shape[-2:], mode='bilinear', align_corners=True)[0, 0]
        
        depth = 1 / depth 

        valid_mask = (valid_mask == 1) & (depth > testloader.dataset.min_disparity) & (depth < testloader.dataset.max_disparity)

        # less than 10 x 10 valid pixel image is meaningless
        if valid_mask.sum().item() < 100:
            invalid_samples.append((i, valid_mask.sum().item()))
            continue
        
        gt_masked = depth[valid_mask].reshape((-1, 1)).cpu().numpy()
        pred_masked = pred[valid_mask].reshape((-1, 1)).cpu().numpy()
        
        # numpy solver
        # scale & shift invariance evaluation
        _ones = np.ones_like(pred_masked)
        A = np.concatenate([pred_masked, _ones], axis=-1)
        X = np.linalg.lstsq(A, gt_masked, rcond=None)[0]
        scale, shift = X
        aligned_pred = pred.squeeze() * torch.from_numpy(scale).to(pred.device) + torch.from_numpy(shift).to(pred.device)
        aligned_pred = aligned_pred.reshape(pred.shape)
        aligned_pred = torch.clamp(aligned_pred, min=testloader.dataset.min_disparity, max=testloader.dataset.max_disparity)
        
        cur_results = eval_depth(aligned_pred[valid_mask], depth[valid_mask])

        # NEW: per-sample record (one row per evaluated image) for later bootstrap CI analysis.
        # The (dataset, item_id, rotate_range_lo, rotate_range_hi) tuple is the pairing key
        # across different methods (baseline / IDCue-P / IDCue-C / HL etc.) when seeds are fixed.
        gt_rotate = float(sample_i['rotate_angle'].item()) if torch.is_tensor(sample_i['rotate_angle']) else float(sample_i['rotate_angle'][0])
        record = {
            'dataset': dataset_name,
            'sample_idx': i,
            'item_id': int(items[0].item()) if torch.is_tensor(items) else int(items[0]),
            'rotate_range_lo': rotate_range[0],
            'rotate_range_hi': rotate_range[1],
            'gt_rotate_angle_deg': gt_rotate,
            'pred_angle_deg': float(pred_deg[0]),
            'residual_angle_deg': float(final_angles[0]),  # gt - pred; what HL feeds to depth model
            'abs_residual_angle_deg': abs(float(final_angles[0])),
            'n_valid_pixels': int(valid_mask.sum().item()),
            'scale': float(scale.item()) if hasattr(scale, 'item') else float(scale),
            'shift': float(shift.item()) if hasattr(shift, 'item') else float(shift),
        }
        for k in cur_results.keys():
            record[k] = float(cur_results[k])
        per_sample_records.append(record)

        for k in results.keys():
            results[k] += cur_results[k]
        nsamples += 1
    

    # all results add gathered results from single test dataset
    for k in all_results.keys():
        all_results[k] += results[k]
    all_nsamples += nsamples
    
    logger.info(f"================ Evaluate Dataset: {dataset_name} With Rotation ({rotate_range}) ===============")
    logger.info(f"({dataset_name}) Valid Test Sample: {nsamples.item()}")
    logger.info('==========================================================================================')
    logger.info('{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}'.format(*tuple(results.keys())))
    logger.info('{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}'.format(*tuple([(v / nsamples).item() for v in results.values()])))
    logger.info('==========================================================================================')
    logger.info('\n')

    return all_results, all_nsamples

if __name__ == '__main__':
    main()