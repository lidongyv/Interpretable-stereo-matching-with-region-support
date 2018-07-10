# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-03-19 13:33:07
# @Last Modified by:   yulidong
# @Last Modified time: 2018-07-06 16:51:51

import os
import torch
import numpy as np
from torch.utils import data
from rsden.utils import recursive_glob
import torchvision.transforms as transforms

class SceneFlow(data.Dataset):


    def __init__(self, root, split="train", is_transform=True, img_size=(540,960)):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        """
        self.is_transform = is_transform
        self.n_classes = 9  # 0 is reserved for "other"
        self.img_size = img_size if isinstance(img_size, tuple) else (540, 960)
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.stats={'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}
        self.files = {}
        self.datapath='/home/lidong/Documents/datasets/Driving/train_data_clean_pass/'
        self.files=os.listdir(self.datapath)
        self.files.sort()
        self.task='generation'
        if len(self.files)<1:
            raise Exception("No files for ld=[%s] found in %s" % (split, self.ld))

        print("Found %d in %s data" % (len(self.files), self.datapat))

    def __len__(self):
        """__len__"""
        return len(self.files)

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        data=np.load(os.path.join(self.datapath,self.files[index]))
        data=data[:540,:960,:]
        left=data[0:3]
        right=data[3:6]
        disparity=data[6]
        P=data[7:]
        if self.is_transform:
            img, region = self.transform(left, right,disparity,P)
        if self.task=='generation':
            return left, right,disparity,P
        else:
            return left, right,disparity,P
    def transform(self, left, right):
        """transform
        """
        trans=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**self.stats),
        ])
        left=trans(left)
        right=trans(right)
        disparity=torch.from_numpy(disparity).float()
        P=torch.from_numpy(P).float()
        return left,right,disparity,P
