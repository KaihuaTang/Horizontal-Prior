import cv2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose

from .dpt_modules import FeatureFusionBlock

def _trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)
    tensor.uniform_(2 * l - 1, 2 * u - 1)
    tensor.erfinv_()
    tensor.mul_(std * math.sqrt(2.))
    tensor.add_(mean)
    tensor.clamp_(min=a, max=b)
    return tensor


class AttentionPoolLatent(nn.Module):
    """ Attention pooling w/ latent query
    """
    def __init__(
            self,
            in_features: int,
            out_features: int = None,
            embed_dim: int = None,
            num_heads: int = 8,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            latent_len: int = 1,
            pool_type: str = 'token',
            drop: float = 0.0,
    ):
        super().__init__()
        embed_dim = embed_dim or in_features
        out_features = out_features or in_features
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.pool = pool_type
        self.fused_attn = True

        self.pos_embed = None

        self.latent_dim = embed_dim
        self.latent_len = latent_len
        self.latent = nn.Parameter(torch.zeros(1, self.latent_len, embed_dim))

        self.q = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.kv = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(drop)

        self.mlp = nn.Sequential(
                    nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
                    nn.GELU(),
                    nn.Dropout(drop),
                    nn.Linear(int(embed_dim * mlp_ratio), out_features),
                    nn.Dropout(drop),
                )

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            self.latent = _trunc_normal_(self.latent, 0, 1.0, -2., 2.)
            self.latent.mul_(self.latent_dim ** -0.5)

    def forward(self, x, attn_mask = None):
        B, N, C = x.shape

        q_latent = self.latent.expand(B, -1, -1)
        q = self.q(q_latent).reshape(B, self.latent_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if attn_mask is not None:
                attn = attn + attn_mask
            attn = attn.softmax(dim=-1)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, self.latent_len, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x + self.mlp(x)

        # optional pool if latent seq_len > 1 and pooled output is desired
        if self.pool == 'token':
            x = x[:, 0]
        elif self.pool == 'avg':
            x = x.mean(1)
        return x
    

class MeanPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mean(1)

            
class CueHead(nn.Module):
    def __init__(
        self, 
        in_channels, 
        hidden_dim=768,
        num_featmap=2, 
        task_configs=None,
    ):
        super(CueHead, self).__init__()
        self.hidden_dim = hidden_dim
        self.task_configs = task_configs
        self.task_channels = task_configs["channels"]

        # get each task output
        for task_id, task_name in enumerate(self.task_configs['names']):
            if task_name == 'angle':
                self.angle_head = nn.Sequential(nn.Linear(in_channels * num_featmap, self.hidden_dim),
                                                AttentionPoolLatent(self.hidden_dim), #MeanPool(),
                                                nn.BatchNorm1d(self.hidden_dim, affine=False, eps=1e-6),
                                                nn.Linear(self.hidden_dim, self.task_channels[task_id]),
                                )
            
            elif task_name == 'elevation':
                self.elevation_head = nn.Sequential(nn.Linear(in_channels * num_featmap, self.hidden_dim),
                                                AttentionPoolLatent(self.hidden_dim),
                                                nn.BatchNorm1d(self.hidden_dim, affine=False, eps=1e-6),
                                                nn.Linear(self.hidden_dim, self.task_channels[task_id]),
                                )
            elif task_name == 'lightshadow':
                self.lightshadow_fc = nn.Linear(in_channels * num_featmap, self.hidden_dim)
                self.lightshadow_head = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                                      nn.GELU(),
                                                      nn.Linear(self.hidden_dim, self.task_channels[task_id]),)
            elif task_name == 'occlusion':
                self.occlusion_fc = nn.Linear(in_channels * num_featmap, self.hidden_dim)
                self.occlusion_head = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                                      nn.GELU(),
                                                      nn.Linear(self.hidden_dim, self.task_channels[task_id]),)
            elif task_name == 'size':
                self.size_fc = nn.Linear(in_channels * num_featmap, self.hidden_dim)
                self.size_head = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                                      nn.GELU(),
                                                      nn.Linear(self.hidden_dim, self.task_channels[task_id]),)
            elif task_name == 'texturegrad':
                self.texture_fc = nn.Linear(in_channels * num_featmap, self.hidden_dim)
                self.texture_head = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                                      nn.GELU(),
                                                      nn.Linear(self.hidden_dim, self.task_channels[task_id]),)
    
    def forward(self, out_features, task_masks=None, patch_h=None, patch_w=None):
        features = torch.cat(out_features, dim=-1)    # out : Batch, num_token, hidden_dim

        multi_task_outputs = []
        for task_id in range(len(self.task_configs['names'])):
            task_name = self.task_configs['names'][task_id]
            if task_name == 'angle': # angle head
                multi_task_outputs.append(self.angle_head(features))
            elif task_name == 'elevation':
                multi_task_outputs.append(self.elevation_head(features))
            elif task_name == 'lightshadow':
                ls_feat = self.lightshadow_fc(features)
                bs, _, height, width = task_masks.shape
                ls_feat = ls_feat.reshape(bs, patch_h, patch_w, -1).permute(0, 3, 1, 2).contiguous()
                ls_feat = nn.Upsample(size=(height, width), mode='bilinear')(ls_feat)
                red_mask = task_masks[:, 0, :, :].unsqueeze(1)
                green_mask = task_masks[:, 1, :, :].unsqueeze(1)
                x_red = (ls_feat * red_mask).sum(-1).sum(-1) / (red_mask.sum(-1).sum(-1) + 1e-5) # (B,C)
                x_green = (ls_feat * green_mask).sum(-1).sum(-1) / (green_mask.sum(-1).sum(-1) + 1e-5) # (B,C)
                multi_task_outputs.append(self.lightshadow_head(x_red - x_green))
            elif task_name == 'occlusion':
                oc_feat = self.occlusion_fc(features)
                bs, _, height, width = task_masks.shape
                oc_feat = oc_feat.reshape(bs, patch_h, patch_w, -1).permute(0, 3, 1, 2).contiguous()
                oc_feat = nn.Upsample(size=(height, width), mode='bilinear')(oc_feat)
                red_mask = task_masks[:, 0, :, :].unsqueeze(1)
                x_red = (oc_feat * red_mask).sum(-1).sum(-1) / (red_mask.sum(-1).sum(-1) + 1e-5) # (B,C)
                multi_task_outputs.append(self.occlusion_head(x_red))
            elif task_name == 'size':
                size_feat = self.size_fc(features)
                bs, _, height, width = task_masks.shape
                size_feat = size_feat.reshape(bs, patch_h, patch_w, -1).permute(0, 3, 1, 2).contiguous()
                size_feat = nn.Upsample(size=(height, width), mode='bilinear')(size_feat)
                red_mask = task_masks[:, 0, :, :].unsqueeze(1)
                green_mask = task_masks[:, 1, :, :].unsqueeze(1)
                x_red = (size_feat * red_mask).sum(-1).sum(-1) / (red_mask.sum(-1).sum(-1) + 1e-5) # (B,C)
                x_green = (size_feat * green_mask).sum(-1).sum(-1) / (green_mask.sum(-1).sum(-1) + 1e-5) # (B,C)
                multi_task_outputs.append(self.size_head(x_red - x_green))
            elif task_name == 'texturegrad':
                texture_feat = self.texture_fc(features)
                bs, _, height, width = task_masks.shape
                texture_feat = texture_feat.reshape(bs, patch_h, patch_w, -1).permute(0, 3, 1, 2).contiguous()
                texture_feat = nn.Upsample(size=(height, width), mode='bilinear')(texture_feat)
                red_mask = task_masks[:, 0, :, :].unsqueeze(1)
                green_mask = task_masks[:, 1, :, :].unsqueeze(1)
                x_red = (texture_feat * red_mask).sum(-1).sum(-1) / (red_mask.sum(-1).sum(-1) + 1e-5) # (B,C)
                x_green = (texture_feat * green_mask).sum(-1).sum(-1) / (green_mask.sum(-1).sum(-1) + 1e-5) # (B,C)
                multi_task_outputs.append(self.texture_head(x_red - x_green))

        return multi_task_outputs