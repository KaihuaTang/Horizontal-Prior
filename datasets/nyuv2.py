import os
import cv2
import tarfile
from PIL import Image 
from io import BytesIO
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from torchvision.transforms import functional as TF

from datasets.transform import Resize, NormalizeImage, PrepareForNet, Crop

class NYUv2(Dataset):
    def __init__(self, filelist_path, data_root, mode='test', size=(518, 518), rotate_range=(0, 0), same_area=False, need_mask=True):
        super().__init__()
        self.mode = mode
        self.size = size
        self.min_depth = 1e-3
        self.max_depth = 10.0
        self.min_disparity = 1.0 / self.max_depth
        self.max_disparity = 1.0 / self.min_depth
        self.data_root = data_root
        self.rotate_range = rotate_range
        self.same_area = same_area
        self.need_mask = need_mask

        with open(filelist_path, 'r') as f:
            self.filelist = f.read().splitlines()
        
        net_w, net_h = size
        self.transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=True if mode == 'train' else False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ] + ([Crop(size[0])] if self.mode == 'train' else []))
        
    def multiple_of_14(self, x):
        y = (np.round(x / 14) * 14).astype(int)
        if y < self.size[0]:
            y = (np.ceil(x / 14) * 14).astype(int)
        return y
    
    def __getitem__(self, item):
        # only used in evaluation for reproduction
        # otherwise, should use random sample instead
        angle_range = self.rotate_range[1] - self.rotate_range[0]
        if angle_range <= 0:
            rotate_angle = self.rotate_range[0]
        else:
            rotate_angle = item % angle_range + self.rotate_range[0] + 1
        if item % 2 == 1:
            rotate_angle = -rotate_angle
        
        sample = self.get_sample_with_angle(item, rotate_angle)

        sample['rotate_angle'] = rotate_angle
        sample['image_path'] = self.filelist[item].split(' ')[0]
        sample['item'] = item

        return sample
    
    def get_sample_with_angle(self, item, rotate_angle, force_mask=False):
        img_path = os.path.join(self.data_root, self.filelist[item].split(' ')[0])
        depth_path = os.path.join(self.data_root, self.filelist[item].split(' ')[2])
        
        image = Image.open(img_path).convert("RGB")
        image = np.asarray(image) / 255.0

        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype('float32')
        depth = depth / 1000

        sample = self.transform({'image': image, 'depth': depth})
        sample['image'] = torch.from_numpy(sample['image'])
        sample['depth'] = torch.from_numpy(sample['depth'])
        
        #valid region
        eval_mask = torch.zeros(sample['depth'].shape)
        eval_mask[45:471, 41:601] = 1
        # give invalid region large negative value for masking after rotate interpolation 
        sample['mask'] = (eval_mask == 1) & (sample['depth'] > self.min_depth) & (sample['depth'] < self.max_depth)
        sample['depth'] = torch.clamp(sample['depth'], min=self.min_depth, max=self.max_depth)
        sample['depth'][~sample['mask']] = -np.inf
                
        # rotate and resize
        if rotate_angle != 0:
            # rotate (padding only) and resize
            rotate_image = TF.rotate(sample['image'], angle=rotate_angle, interpolation=TF.InterpolationMode.NEAREST, fill=0, expand=True)
            _, height, width = rotate_image.shape
            if self.same_area:
                sample['depth'] = TF.rotate(sample['depth'].unsqueeze(0), angle=rotate_angle, interpolation=TF.InterpolationMode.NEAREST, fill=0, expand=True)
                sample['mask']  = TF.rotate(sample['mask'].unsqueeze(0), angle=rotate_angle, interpolation=TF.InterpolationMode.NEAREST, fill=0, expand=True)
                scale = ((self.size[0] * self.size[1]) / (height * width)) ** 0.5
                sample['image'] = self.center_crop_with_scale(rotate_image, scale, keep_multiple14=True)
                sample['depth'] = self.center_crop_with_scale(sample['depth'], scale)[0]
                sample['mask']  = self.center_crop_with_scale(sample['mask'], scale)[0]
            else:
                scale = self.size[0] / min(height, width)
                new_size = (self.multiple_of_14(int(scale * height)), self.multiple_of_14(int(scale * width)))
                sample['image'] = TF.resize(rotate_image, size=new_size, interpolation=TF.InterpolationMode.BILINEAR)
                # rotate (padding only)
                sample['depth'] = TF.rotate(sample['depth'].unsqueeze(0), angle=rotate_angle, interpolation=TF.InterpolationMode.NEAREST, fill=0, expand=True)[0]
                sample['mask']  = TF.rotate(sample['mask'].unsqueeze(0), angle=rotate_angle, interpolation=TF.InterpolationMode.NEAREST, fill=0, expand=True)[0]

        sample['valid_mask'] = (sample['mask'] > 0.5) & (sample['depth'] > self.min_depth) & (sample['depth'] < self.max_depth)
        return sample


    def center_crop_with_scale(self, image, scale, keep_multiple14=False):
        assert scale <= 1
        _, height, width = image.shape
        new_height = int(scale * height)
        new_width = int(scale * width)

        h_bg = (height - new_height) // 2
        w_bg = (width  - new_width) // 2

        if keep_multiple14:
            floor_height = (np.floor(new_height / 14) * 14).astype(int)
            floor_width = (np.floor(new_width / 14) * 14).astype(int)
            crop_img = image[:, h_bg:h_bg+new_height, w_bg:w_bg+new_width]
            return TF.resize(crop_img, size=(floor_height, floor_width), interpolation=TF.InterpolationMode.BILINEAR)
        else:
            return image[:, h_bg:h_bg+new_height, w_bg:w_bg+new_width]


    def __len__(self):
        return len(self.filelist)