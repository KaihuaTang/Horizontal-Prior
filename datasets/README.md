## Data Preprocess

The datasets we used includes 1) part of [SA-1B](https://ai.meta.com/datasets/segment-anything/) and [DepthCues](https://huggingface.co/datasets/danier97/depthcues) for training; 2) DIODE, ETH3D, KITTI, NYUv2, and ScanNet for evaluation.

### Training Data Prepare
- Please follow the "./prepare/download_sa1b.py" to download a subset of [SA-1B](https://ai.meta.com/datasets/segment-anything/) dataset, including "sa_000000.tar", "sa_000050.tar", ..., "sa_000950.tar". 
Then you need to unzip them and put them in a given folder. After that, use "./prepare/prepare_list.py" to convert all valid images into a txt file. You may need to change all the relative and absolute pathes in "./prepare/prepare_list.py". The generate txt file will be used for the dataloader.
- For DepthCues datasets, please following the instruction on the following [Huggingface Link](https://huggingface.co/datasets/danier97/depthcues) to download and prepare. You need to set all data paths in "./datasets/depthcue.py" to ensure the training samples of depthcues can be loaded correctly. 

### Evaluation Data Prepare
Since most of the existing monocular depth estimation codebases didn't release their evaluation code, e.g., Depth Anything V2, Distill Any Depth. We follow one open-sourced code from [GenPercept](https://github.com/aim-uofa/GenPercept) to prepare our evaluation dataset(Thank you so much for their great work!). The DIODE, ETH3D, KITTI, NYUv2, and ScanNet datasets can be downloaded from their [Hugging Face link](https://huggingface.co/datasets/guangkaixu/genpercept_datasets_eval/tree/main). Just keep their data somewhere and set the path in evaluation dataloader. Their sample lists are under "./data_split" in this repo.

