import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose

from .dinov2 import DINOv2
from .modules import CueHead, DPTHeadTiny
from .modules import Resize, NormalizeImage, PrepareForNet

class DepthCueModel(nn.Module):
    def __init__(
        self, 
        encoder='vits', 
        hidden_dim=768, 
        layers=[8, 11],
        task_configs=None,
    ):
        super(DepthCueModel, self).__init__()
        
        self.encoder = encoder
        self.layers = layers
        self.task_configs = task_configs
        self.use_teacher = task_configs['use_local_cue_teacher']

        self.cue_backbone = DINOv2(model_name=encoder)
        self.cue_head = CueHead(self.cue_backbone.embed_dim, hidden_dim,
                                num_featmap=len(self.layers), 
                                task_configs=task_configs)
        if self.use_teacher:
            self.localcue_head = DPTHeadTiny(self.cue_backbone.embed_dim, out_dims=3, features=128, use_bn=False, out_channels=[256, 256], use_clstoken=False)
        
    
    def forward(self, x, task_masks=None, return_feat=False):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        
        features = self.cue_backbone.get_intermediate_layers(x, self.layers)
        
        task_outs = self.cue_head(features, task_masks, patch_h, patch_w)
        
        if self.use_teacher:
            local_cues = self.localcue_head(features, patch_h, patch_w)
            task_outs.append(local_cues)

        if return_feat:
            return task_outs, features
        else:
            return task_outs
