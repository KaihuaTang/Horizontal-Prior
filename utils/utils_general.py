import torch
from safetensors.torch import load_file

def format_time_interval(time_interval):
    hours = int(time_interval // 3600)
    minutes = int((time_interval % 3600) // 60)
    seconds = int(time_interval % 60)  # Keep seconds as float for precision
    if hours > 0:
        return f"{hours}h-{minutes}m-{seconds}s"
    elif minutes > 0:
        return f"{minutes}m-{seconds}s"
    else:
        return f"{seconds}s"


def load_from_checkpoint(model, ckpt_path, logger):
    if ckpt_path.endswith(".safetensors"):
        # load distill any depth checkpoints
        # https://github.com/Westlake-AGI-Lab/Distill-Any-Depth
        ckpt_init = load_file(ckpt_path)
        ckpt = {}
        for key, val in ckpt_init.items():
            if key.startswith("backbone."):
                ckpt[key.replace("backbone.", "pretrained.").replace("blocks.0", "blocks")] = val
            else:
                ckpt[key] = val
    else:
        ckpt = torch.load(ckpt_path, map_location='cpu')
    if "model" in ckpt:
        ckpt_post = {}
        for key, val in ckpt["model"].items():
            if key.startswith("module."):
                ckpt_post[key[7:]] = val
            else:
                ckpt_post[key] = val
        ckpt_matching(model, ckpt_post, logger)
        model.load_state_dict(ckpt_post, strict=False)
    elif "state_dict" in ckpt:
        ckpt_post = {}
        for key, val in ckpt["state_dict"].items():
            if key.startswith("pipeline."):
                ckpt_post[key[9:]] = val
            else:
                ckpt_post[key] = val
        ckpt_matching(model, ckpt_post, logger)
        model.load_state_dict(ckpt_post, strict=False)
    else:
        ckpt_matching(model, ckpt, logger)
        model.load_state_dict(ckpt, strict=False)
    return model

def ckpt_matching(model, ckpt, logger):
    param_names = []
    for name in model.state_dict().keys():
        param_names.append(name)
        if name not in ckpt:
            logger.info(f"Load from checkpoint: MISSING {name}")
    for name, param in ckpt.items():
        if name not in param_names:
            logger.info(f"Load from checkpoint: UNEXPECTED {name}")