import os
import cv2
import tarfile
from PIL import Image 
from io import BytesIO
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from torchvision.transforms import functional as TF
from datasets.transform import Resize, NormalizeImage, PrepareForNet, Crop

class ETH3D(Dataset):
    def __init__(self, filelist_path, data_root, mode='test', size=(518, 518), rotate_range=(0, 0), same_area=False, need_mask=True):
        super().__init__()
        self.mode = mode
        self.size = size
        self.min_depth = 1e-5
        self.max_depth = torch.inf
        self.min_disparity = 0.0
        self.max_disparity = 1.0 / self.min_depth
        self.HEIGHT = 4032
        self.WIDTH = 6048
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
        depth_path = os.path.join(self.data_root, self.filelist[item].split(' ')[1])
        
        #image = cv2.imread(img_path)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        cache_path = img_path[:-4] + f"_transformed_{self.size[0]}_{self.size[1]}.pth"
        if os.path.exists(cache_path):
            sample = torch.load(cache_path, weights_only=False)
        else:
            image = Image.open(img_path).convert("RGB")
            image = np.asarray(image) / 255.0
            
            with open(depth_path, "rb") as file:
                binary_data = file.read()
            depth_decoded = np.frombuffer(binary_data, dtype=np.float32).copy()
            depth_decoded[depth_decoded == torch.inf] = 0.0
            depth_decoded = depth_decoded.reshape((self.HEIGHT, self.WIDTH))
            
            sample = self.transform({'image': image, 'depth': depth_decoded})
            torch.save(sample, cache_path)

        sample['image'] = torch.from_numpy(sample['image'])

        if self.need_mask or force_mask:
            sample['depth'] = torch.from_numpy(sample['depth'])
            # set invalid pixel to -9999
            sample['mask'] = (sample['depth'] > self.min_depth) & (sample['depth'] < self.max_depth)
            sample['depth'] = torch.clamp(sample['depth'], min=self.min_depth, max=self.max_depth)
            sample['depth'][~sample['mask']] = -np.inf

        # rotate and resize
        if rotate_angle != 0:
            # rotate (padding only) and resize
            rotate_image = TF.rotate(sample['image'], angle=rotate_angle, interpolation=TF.InterpolationMode.NEAREST, fill=0, expand=True)
            _, height, width = rotate_image.shape
            if self.same_area:
                scale = ((self.size[0] * self.size[1]) / (height * width)) ** 0.5
                sample['image'] = self.center_crop_with_scale(rotate_image, scale, keep_multiple14=True)
                if self.need_mask or force_mask:
                    sample['depth'] = TF.rotate(sample['depth'].unsqueeze(0), angle=rotate_angle, interpolation=TF.InterpolationMode.NEAREST, fill=0, expand=True)
                    sample['mask']  = TF.rotate(sample['mask'].unsqueeze(0), angle=rotate_angle, interpolation=TF.InterpolationMode.NEAREST, fill=0, expand=True)
                    sample['depth'] = self.center_crop_with_scale(sample['depth'], scale)[0]
                    sample['mask']  = self.center_crop_with_scale(sample['mask'], scale)[0]
            else:
                scale = self.size[0] / min(height, width)
                new_size = (self.multiple_of_14(int(scale * height)), self.multiple_of_14(int(scale * width)))
                sample['image'] = TF.resize(rotate_image, size=new_size, interpolation=TF.InterpolationMode.BILINEAR)
                # rotate (padding only)
                if self.need_mask or force_mask:
                    sample['depth'] = TF.rotate(sample['depth'].unsqueeze(0), angle=rotate_angle, interpolation=TF.InterpolationMode.NEAREST, fill=0, expand=True)[0]
                    sample['mask']  = TF.rotate(sample['mask'].unsqueeze(0), angle=rotate_angle, interpolation=TF.InterpolationMode.NEAREST, fill=0, expand=True)[0]

        if self.need_mask or force_mask:
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