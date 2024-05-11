import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np
from pathlib import Path
import cv2

from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images

def pil_loader(path):
    return Image.open(path).convert('RGB')

class InpaintDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class UncroppingDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'manual':
            mask = bbox2mask(self.image_size, self.mask_config['shape'])
        elif self.mask_mode == 'fourdirection' or self.mask_mode == 'onedirection':
            mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode=self.mask_mode))
        elif self.mask_mode == 'hybrid':
            if np.random.randint(0,2)<1:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='onedirection'))
            else:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='fourdirection'))
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class ColorizationDataset(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[224, 224], loader=pil_loader):
        self.data_root = data_root
        flist = make_dataset(data_flist)
        if data_len > 0:
            self.flist = flist[:int(data_len)]
        else:
            self.flist = flist
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.flist[index]).zfill(5) + '.png'

        img = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'color', file_name)))
        cond_image = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'gray', file_name)))

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['path'] = file_name
        return ret

    def __len__(self):
        return len(self.flist)
    
    
class WoodblockDataset(data.Dataset):
    def __init__(self, data_root, mode="train", image_size=[512, 512]):
        self.data_root = data_root
        if mode != "valid":
            self.print_path_list = sorted(list(Path(os.path.join(self.data_root, mode, "print_512")).glob("*.png")), key=os.path.basename)
            self.depth_path_list = sorted(list(Path(os.path.join(self.data_root, mode, "np_depth_512")).glob("*.npy")), key=os.path.basename)
        else:
            self.print_path_list = sorted(list(Path(os.path.join(self.data_root, mode, "print_512")).glob("*.png")), key=os.path.basename)[:3]
            self.depth_path_list = sorted(list(Path(os.path.join(self.data_root, mode, "np_depth_512")).glob("*.npy")), key=os.path.basename)[:3]
        self.img_tfs = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda t: (t * 2) - 1)])
        
        self.image_size = image_size

    def __len__(self):
        return len(self.print_path_list)

    def preprocess_image(self, image_path):
        """Utility function that load an image an convert to torch."""
        # open image using OpenCV (HxWxC)
        img = Image.open(image_path).convert('L')
        # convert image to torch tensor (CxHxW)
        img_t: torch.Tensor = self.img_tfs(img)
        return img_t

    def preprocess_depth(self, depth_path):
        """Utility function that load an image an convert to torch."""
        # open image using OpenCV (HxWxC)
        img: np.ndarray = np.load(depth_path)
        img = np.asarray(img, dtype=np.float32)
        # mask = img != img.max()
        # mask = np.expand_dims(mask, axis=0)
        # t_mask = torch.from_numpy(mask)
        t_mask = torch.tensor(img != img.max(), dtype=torch.float16).unsqueeze(0)
        t_img = self.img_tfs(img)
        # # unsqueeze to make it 1xHxW
        # img = np.expand_dims(img, axis=0)
        # # cast type as np.float32
        # img = img.astype(np.float32)
        # # convert image to torch tensor (CxHxW)
        # img_t: torch.Tensor = torch.from_numpy(img)
        # t_mean_value = torch.mean(img_t)
        # # img_t = transforms.Compose([transforms.Normalize(mean=(t_mean_value, ), std=(1, ))])
        # # print("Before: ", img.shape)
        # img_t = img_t - t_mean_value
        # print("After: ", img_t.shape)
        return t_img, t_mask


    def __getitem__(self, index):
        print_path = self.print_path_list[index]
        depth_path = self.depth_path_list[index]
        print_img = self.preprocess_image(str(print_path))
        depth_matrix, t_mask = self.preprocess_depth(str(depth_path))
        ret = {
            'gt_image': depth_matrix,
            'cond_image': print_img,
            'mask': t_mask,
            'path': Path(print_path).name
        }
        return ret


