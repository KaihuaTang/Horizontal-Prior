#!/bin/bash

# fix windows editor prompt
# sed -i 's/\r$//' test.sh
now=$(date +"%Y%m%d_%H%M%S")

gpu_id=0
encoder=vitl
img_size=560 # 518 / 560
testdata=all # DIODE ETH3D

angle_path=./checkpoints/angle_prediction/latest.pth
ckpt_path=./exp/size560_bs8ep1_d1g1c0_baseline/latest.pth
save_path=./eval/size560_bs8ep1_d1g1c0_baseline 

mkdir -p $save_path

CUDA_VISIBLE_DEVICES=$gpu_id python3 test_hl.py --encoder $encoder \
    --img-size $img_size --testdata $testdata \
    --ckpt-path $ckpt_path --save-path $save_path \
    --angle-path $angle_path
