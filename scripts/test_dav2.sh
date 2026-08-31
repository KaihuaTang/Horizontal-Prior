#!/bin/bash

# fix windows editor prompt
# sed -i 's/\r$//' test.sh
now=$(date +"%Y%m%d_%H%M%S")

gpu_id=0
encoder=vitl
img_size=518 # 518 / 560
testdata=all # DIODE ETH3D

ckpt_path=./checkpoints/depth_anything_v2_${encoder}.pth
save_path=./eval/depth_anything_v2_${encoder}_size518

mkdir -p $save_path

CUDA_VISIBLE_DEVICES=$gpu_id python3 test.py --encoder $encoder \
    --img-size $img_size --testdata $testdata \
    --ckpt-path $ckpt_path --save-path $save_path 
