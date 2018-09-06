# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-03-19 13:33:07
# @Last Modified by:   yulidong
# @Last Modified time: 2018-09-03 13:47:13

import os
import torch
import numpy as np
from torch.utils import data
from pssm.utils import recursive_glob
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
        self.left_files = {}
        self.datapath=root
        self.left_files=os.listdir(os.path.join(self.datapath,'left'))
        # self.match_files=os.listdir(os.path.join(self.datapath,'match'))
        # self.matching_files=os.listdir(os.path.join(self.datapath,'matching'))
        # self.aggregation_files=os.listdir(os.path.join(self.datapath,'aggregation'))
        self.left_files.sort()
        self.task='generation'
        if len(self.left_files)<1:
            raise Exception("No files for ld=[%s] found in %s" % (split, self.ld))

        print("Found %d in %s data" % (len(self.left_files), self.datapath))

    def __len__(self):
        """__len__"""
        return len(self.left_files)

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        data=np.load(os.path.join(self.datapath,'left',self.left_files[index]))
        data=data[:540,:960,:]
        left=data[...,0:3]
        #print(data.shape)
        right=data[...,3:6]

        disparity=data[...,6]
        P=data[...,7:]
        #pre_match=np.load(os.path.join(self.datapath,'match',self.left_files[index]))
        matching=np.load(os.path.join(self.datapath,'matching',self.left_files[index]))
        aggregation=np.load(os.path.join(self.datapath,'aggregation',self.left_files[index]))
        if self.is_transform:
            left, right,disparity,P,plane,s_plane,l_plane = self.transform(left, right,disparity,P,matching,aggregation)
        if self.task=='generation':
            return left, right,disparity,P,plane,s_plane,l_plane
        else:
            return left, right,disparity,P,plane,s_plane,l_plane
    def transform(self, left, right,disparity,P,matching,aggregation):
        """transform
        """
        # match=[s_match_x,
        #         s_match_y,
        #         s_shfit,
        #         l_match_x,
        #         l_match_y,
        #         l_shfit]
        # plane=[
        # plane_x_0,
        # plane_y_0,
        # plane_x_1,
        # plane_y_1,
        # plane_x_2,
        # plane_y_2,
        # plane_x_3,
        # plane_y_3,
        # np.array(plane_num_0),
        # np.array(plane_num_1),
        # np.array(plane_num_2),
        # np.array(plane_num_3)
        # ]

        # s_plane=[
        # s_plane_x_0,
        # s_plane_y_0,
        # s_plane_x_1,
        # s_plane_y_1,
        # s_plane_x_2,
        # s_plane_y_2,
        # s_plane_x_3,
        # s_plane_y_3,
        # np.array(s_plane_num_0),
        # np.array(s_plane_num_1),
        # np.array(s_plane_num_2),
        # np.array(s_plane_num_3)
        # ]

        # l_plane=[
        # l_plane_x_0,
        # l_plane_y_0,
        # l_plane_x_1,
        # l_plane_y_1,
        # l_plane_x_2,
        # l_plane_y_2,
        # l_plane_x_3,
        # l_plane_y_3,
        # np.array(l_plane_num_0),
        # np.array(l_plane_num_1),
        # np.array(l_plane_num_2),
        # np.array(l_plane_num_3)
        # ]
        # aggregation=[plane,s_plane,l_plane]        
        trans=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**self.stats),
        ])
        left=trans(left).float()
        right=trans(right).float()
        disparity=torch.from_numpy(disparity).float()
        P=torch.from_numpy(P).float()

        s_match_x=torch.from_numpy(matching[0])
        s_match_y=torch.from_numpy(matching[1])
        s_shfit=torch.from_numpy(matching[2])
        l_match_x=torch.from_numpy(matching[3])
        l_match_y=torch.from_numpy(matching[4])
        l_shfit=torch.from_numpy(matching[5])
        match=[s_match_x,
                s_match_y,
                s_shfit,
                l_match_x,
                l_match_y,
                l_shfit]
        plane=aggregation[0]
        plane_x_0=torch.from_numpy(plane[0])
        plane_y_0=torch.from_numpy(plane[1])
        plane_x_1=torch.from_numpy(plane[2])
        plane_y_1=torch.from_numpy(plane[3])
        plane_x_2=torch.from_numpy(plane[4])
        plane_y_2=torch.from_numpy(plane[5])
        plane_x_3=torch.from_numpy(plane[6])
        plane_y_3=torch.from_numpy(plane[7])
        plane_num_0=torch.from_numpy(plane[8])
        plane_num_1=torch.from_numpy(plane[9])
        plane_num_2=torch.from_numpy(plane[10])
        plane_num_3=torch.from_numpy(plane[11])
        plane=[
        plane_x_0,
        plane_y_0,
        plane_x_1,
        plane_y_1,
        plane_x_2,
        plane_y_2,
        plane_x_3,
        plane_y_3,
        plane_num_0,
        plane_num_1,
        plane_num_2,
        plane_num_3
        ]
        s_plane=aggregation[1]
        s_plane_x_0=torch.from_numpy(s_plane[0])
        s_plane_y_0=torch.from_numpy(s_plane[1])
        s_plane_x_1=torch.from_numpy(s_plane[2])
        s_plane_y_1=torch.from_numpy(s_plane[3])
        s_plane_x_2=torch.from_numpy(s_plane[4])
        s_plane_y_2=torch.from_numpy(s_plane[5])
        s_plane_x_3=torch.from_numpy(s_plane[6])
        s_plane_y_3=torch.from_numpy(s_plane[7])
        s_plane_num_0=torch.from_numpy(s_plane[8])
        s_plane_num_1=torch.from_numpy(s_plane[9])
        s_plane_num_2=torch.from_numpy(s_plane[10])
        s_plane_num_3=torch.from_numpy(s_plane[11])
        s_plane=[
        s_plane_x_0,
        s_plane_y_0,
        s_plane_x_1,
        s_plane_y_1,
        s_plane_x_2,
        s_plane_y_2,
        s_plane_x_3,
        s_plane_y_3,
        s_plane_num_0,
        s_plane_num_1,
        s_plane_num_2,
        s_plane_num_3
        ]
        l_plane=aggregation[2]
        l_plane_x_0=torch.from_numpy(l_plane[0])
        l_plane_y_0=torch.from_numpy(l_plane[1])
        l_plane_x_1=torch.from_numpy(l_plane[2])
        l_plane_y_1=torch.from_numpy(l_plane[3])
        l_plane_x_2=torch.from_numpy(l_plane[4])
        l_plane_y_2=torch.from_numpy(l_plane[5])
        l_plane_x_3=torch.from_numpy(l_plane[6])
        l_plane_y_3=torch.from_numpy(l_plane[7])
        l_plane_num_0=torch.from_numpy(l_plane[8])
        l_plane_num_1=torch.from_numpy(l_plane[9])
        l_plane_num_2=torch.from_numpy(l_plane[10])
        l_plane_num_3=torch.from_numpy(l_plane[11])
        l_plane=[
        l_plane_x_0,
        l_plane_y_0,
        l_plane_x_1,
        l_plane_y_1,
        l_plane_x_2,
        l_plane_y_2,
        l_plane_x_3,
        l_plane_y_3,
        l_plane_num_0,
        l_plane_num_1,
        l_plane_num_2,
        l_plane_num_3
        ]

        #print(pre1.shape)
        return left,right,disparity,P,match,plane,s_plane,l_plane
