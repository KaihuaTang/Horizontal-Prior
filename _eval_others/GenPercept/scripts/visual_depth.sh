# CUDA_VISIBLE_DEVICES='0'
export PYTHONPATH=./

python tools/visual_genpercept.py \
--output_dir 'output/depth' \
--mode 'depth' \
--checkpoint "weights/v1" \
--unet_ckpt_path "weights/v1/unet_depth_v1" \
--batch_size 1 \
--half_precision

