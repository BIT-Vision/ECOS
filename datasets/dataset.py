import os
import cv2
import numpy as np
import random
import json
import torch
from torch.utils import data

import datasets.transform as mytrans
import logging
from PIL import Image


class DataTrain(data.Dataset):
    r'''
    - root: data root path, str
    - output_size: output size of image and mask, tuple
    - imset: train sequence txt, str
    - clip_n: number of video clip for training, int
    - max_obj_n: maximum number of objects in a image for training, int
    '''
    def __init__(self,
                 root,
                 output_size,
                 imset='train.txt',
                 max_obj_n=11,
                 clip_n=8,
                 data_mode='event_5',
                 label_mode='event_label_format',
                 dim=5):
        self.root = root
        self.clip_n = clip_n
        self.output_size = output_size
        self.dim = dim
        self.data_mode = data_mode
        self.len_str = 'event_len'
        self.label_mode = label_mode

        with open(os.path.join(root, 'data.json')) as f:
            self.data = json.load(f)

        self.video_name_list = list()

        with open(os.path.join(root, imset), 'r') as lines:
            for line in lines:
                video_name = line.strip()
                if len(video_name) > 0:
                    length = self.data[video_name][self.len_str]
                    if length >= self.clip_n:
                        self.video_name_list.append(video_name)
                        self.data_len += 1

        self.logger = logging.getLogger(__name__)
        self.logger.info(f'load : {len(self.video_name_list)} videos.')

        self.random_affine = mytrans.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.95, 1.05), shear=10)
        self.resize = mytrans.Resize(output_size)
        self.to_onehot = mytrans.ToOnehot(max_obj_n, shuffle=True)
        self.max_obj_n = max_obj_n

    def __len__(self):
        return self.data_len * 5

    def __getitem__(self, idx):
        video_name = self.video_name_list[idx % self.data_len]

        voxel_list = self.data[video_name][self.data_mode]
        image_list = self.data[video_name][self.label_mode]

        img_n = self.data[video_name][self.len_str]

        idx_list = list()
        last_sample = -1
        sample_n = min(self.clip_n, img_n)
        for i in range(sample_n):
            if i == 0:
                last_sample = random.choice(range(0, img_n - sample_n + 1))
            else:
                last_sample += 1
            idx_list.append(last_sample)

        voxel = np.load(os.path.join(self.root, voxel_list[idx_list[0]]))

        voxels = torch.zeros((self.clip_n, self.dim, *self.output_size), dtype=torch.float)
        masks = torch.zeros((self.clip_n, self.max_obj_n, *self.output_size), dtype=torch.float)

        ret = None

        for i, frame_idx in enumerate(idx_list):
            mask = Image.open(os.path.join(self.root, image_list[frame_idx])).convert('P')

            voxel = np.load(os.path.join(self.root, voxel_list[frame_idx]))  #5 H W
            voxel = torch.from_numpy(voxel)
            voxel, mask = self.resize(voxel, mask)

            voxel, mask, ret = self.random_affine(voxel, mask, ret)
            mask = np.array(mask, np.uint8)

            if i == 0:
                mask, obj_list = self.to_onehot(mask)
                if len(obj_list) == 0:
                    obj_list = [1]
                obj_n = len(obj_list) + 1
            else:
                mask, _ = self.to_onehot(mask, obj_list)

            voxels[i] = voxel
            masks[i] = mask

        info = {'name': video_name, 'idx_list': idx_list}

        return voxels, masks, obj_n, info


class DataTest(data.Dataset):
    r'''
    - root: data root path, str
    - output_size: output size of image and mask, tuple
    - imset: test sequence txt, str
    - max_obj_n: maximum number of objects in a image for testing, int
    '''
    def __init__(self,
                 root,
                 output_size=None,
                 imset='test.txt',
                 max_obj_n=11,
                 data_mode='event_5',
                 label_mode='image',
                 dim=5):
        self.root = root
        self.video_name_list = list()
        self.output_size = output_size

        self.data_mode = data_mode
        self.len_str = 'event_len'
        self.label_mode = label_mode

        self.dim = dim

        with open(os.path.join(root, 'data.json')) as f:
            self.data = json.load(f)

        self.video_name_list = list()
        self.data_len = 0
        with open(os.path.join(root, imset), 'r') as lines:
            for line in lines:
                video_name = line.strip()
                if len(video_name) > 0:
                    length = self.data[video_name][self.len_str]
                    if length > 1:
                        self.video_name_list.append(video_name)
                        self.data_len += 1
        
        self.to_onehot = mytrans.ToOnehot(max_obj_n, shuffle=False)
        self.max_obj_n = max_obj_n
        self.resize = mytrans.Resize(self.output_size)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        video_name = self.video_name_list[idx]
        video_len = self.data[video_name][self.len_str]

        voxel_list = self.data[video_name][self.data_mode]
        mask_list = self.data[video_name][self.label_mode]

        first = np.load(os.path.join(self.root, voxel_list[0]))
        _, h, w = first.shape

        voxels = torch.zeros((video_len, self.dim, *self.output_size), dtype=torch.float)
        masks = torch.zeros((video_len, self.max_obj_n, *self.output_size), dtype=torch.float)

        for i in range(video_len):
            voxel = np.load(os.path.join(self.root, voxel_list[i]))
            voxel = torch.from_numpy(voxel)
            mask = Image.open(os.path.join(self.root, mask_list[0])).convert('P')
            voxel, mask = self.resize(voxel, mask)

            mask = np.array(mask, np.uint8)

            if i == 0:
                mask, obj_list = self.to_onehot(mask)
                if len(obj_list) == 0:
                    obj_list = [1]
                obj_n = len(obj_list) + 1
            else:
                mask, _ = self.to_onehot(mask, obj_list)

            voxels[i] = voxel
            masks[i] = mask

        info = {'name': video_name, 'original_size': (h, w)}
        return voxels, masks, obj_n, info


def build_dataset(cfg, mode='train', imset='event_train.txt'):
    if mode == 'train':
        return DataTrain(root=cfg.DATA.ROOT,
                         output_size=cfg.DATA.SIZE,
                         clip_n=cfg.DATA.TRAIN.FRAMES_PER_CLIP,
                         max_obj_n=cfg.DATA.TRAIN.MAX_OBJECTS,
                         data_mode=cfg.DATA.TRAIN.DATA_MODE,
                         label_mode=cfg.DATA.TRAIN.LABEL_MODE,
                         dim=cfg.MODEL.INPUT_DIM,
                         imset=imset)
    else:
        return DataTest(
            root=cfg.DATA.ROOT,
            output_size=cfg.DATA.SIZE,
            max_obj_n=11,
            data_mode=cfg.DATA.TRAIN.DATA_MODE,
            label_mode=cfg.DATA.TRAIN.LABEL_MODE,
            dim=cfg.MODEL.INPUT_DIM,
            imset=imset)
