# Horizontal-Prior
Official implementation for the paper "Breaking the Horizontal Prior: From Long-Tailed Orientation Bias to Roll-Robust Monocular Depth Estimation"

# Getting Started
- Data: please following the "./datasets/README.md" to prepare all evaluation datasets
- Environment: we recommend setting up a virtual environment to ensure package compatibility. You can use [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) to set up the environment. The following steps show how to create and activate the environment, and install dependencies:
```
# Create a new conda environment with Python 3.10 
# (Other versions of python also work in most of the cases)
conda create -n idcue -y python=3.10

# Activate the created environment
conda activate idcue

# Install the required Python packages
pip install -r requirements.txt
```

# Checkpoints
For ease of management, we recommend placing all downloaded ckpt files in the "checkpoints" folder under this directory. However, as long as you specify the absolute path to the ckpt file in the corresponding test ".sh" scripts, you can actually store the checkpoints in any accessible location on the server.

- For Depth Anything V2, you can download all checkpoints from links in [their official github repo](https://github.com/DepthAnything/Depth-Anything-V2). We adopt Depth-Anything-V2-Large (335.3M) in our evaluation.

- For Distill Any Depth, their checkpoints can be downloaded from [their github repo](https://github.com/Westlake-AGI-Lab/Distill-Any-Depth) as well. 

# Evaluation

The evaluation scripts are located in the "./scripts" directory. Please note that neither Depth-Anything-v2 nor Distill-Any-Depth has released their official evaluation code. Consequently, we re-implemented these protocols; therefore, our results may differ slightly from those reported in the original papers.

1. Evaluate Depth-Anything-v2
```
bash ./scripts/test_dav2.sh
```

2. Evaluate reproduced baseline
```
bash ./scripts/test_baseline.sh
```

3. Evaluate baseline model with Re-balanced Augmentation
```
bash ./scripts/test_aug.sh
```

4. Evaluate baseline model with Horizontal Leveling
```
bash ./scripts/test_hl.sh
```


# Evaluation Other Methods
We also put our evaluation codes into other projects, such as GenPercept, Marigold, and PromptDA (visualization only). . Please refer to the "./_eval_others" folder for these scripts.


# Confidentiality Policy of the Company

Due to the confidentiality reasons of the company, all checkpoints and most of the codes are not allowed to release or send to the external network.


# Citation
```
@misc{tang2026breakinghorizontalpriorlongtailed,
      title={Breaking the Horizontal Prior: From Long-Tailed Orientation Bias to Roll-Robust Monocular Depth Estimation}, 
      author={Kaihua Tang and Ziqing Xia and Xiaoxu Zheng and Xiaoxue Zhang and Michael Bi Mi and Zhan Xu and Dave Zhenyu Chen},
      year={2026},
      eprint={2608.00678},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2608.00678}, 
}
```
