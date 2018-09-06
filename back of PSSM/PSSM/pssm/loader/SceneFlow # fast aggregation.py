# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-03-19 13:33:07
# @Last Modified by:   yulidong
# @Last Modified time: 2018-09-04 15:57:49

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
            left, right,disparity,P,matching,plane,s_plane,l_plane = self.transform(left, right,disparity,P,matching,aggregation)
        if self.task=='generation':
            return left, right,disparity,P,matching,plane,s_plane,l_plane
        else:
            return left, right,disparity,P,matching,plane,s_plane,l_plane
    def transform(self, left, right,disparity,P,matching,aggregation):
        """transform
        """
    
        trans=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**self.stats),
        ])
        left=trans(left).float()
        right=trans(right).float()
        disparity=torch.from_numpy(disparity).float()
        P=torch.from_numpy(P).float()
        for k in range(len(matching)):
            if matching[k]==[]:
                matching[k]=np.concatenate([matching[k],-np.ones(1)])




        s_match_x=[]
        s_match_y=[]
        s_shfit=[]
        s_repeat=np.array(matching[3])
        l_match_x=[]
        l_match_y=[]
        l_shfit=[]
        l_repeat=np.array(matching[7])

        for i in range(s_repeat.shape[0]):
            #print(s_repeat[i].shape)
            shift=np.tile(matching[2][i],s_repeat[i][1])
            s_match_x_t=matching[0][i].repeat(s_repeat[i][0])
            s_match_y_t=matching[1][i].repeat(s_repeat[i][0])
            s_shfit=np.concatenate([s_shfit,shift])
            s_match_x=np.concatenate([s_match_x,s_match_x_t])
            s_match_y=np.concatenate([s_match_y,s_match_y_t])
        for i in range(l_repeat.shape[0]):
            shift=np.tile(matching[6][i],l_repeat[i][1])
            l_match_x_t=matching[4][i].repeat(l_repeat[i][0])
            l_match_y_t=matching[5][i].repeat(l_repeat[i][0])
            l_shfit=np.concatenate([l_shfit,shift])
            l_match_x=np.concatenate([l_match_x,l_match_x_t])
            l_match_y=np.concatenate([l_match_y,l_match_y_t])



        s_match_x=torch.from_numpy(np.array(s_match_x)).long()
        s_match_y=torch.from_numpy(np.array(s_match_y)).long()
        s_shfit=torch.from_numpy(np.array(s_shfit)).long()
        l_match_x=torch.from_numpy(np.array(l_match_x)).long()
        l_match_y=torch.from_numpy(np.array(l_match_y)).long()
        l_shfit=torch.from_numpy(np.array(l_shfit)).long()

        match=[s_match_x,
                s_match_y,
                s_shfit,
                l_match_x,
                l_match_y,
                l_shfit]
        plane=aggregation[0]
        for k in range(len(plane)):
            if plane[k]==[]:
                #print('non')
                plane[k]=np.concatenate([plane[k],-np.ones(1)])
        plane_x_0=torch.from_numpy(plane[0]).long()
        plane_y_0=torch.from_numpy(plane[1]).long()
        plane_x_1=torch.from_numpy(plane[2]).long()
        plane_y_1=torch.from_numpy(plane[3]).long()
        plane_x_2=torch.from_numpy(plane[4]).long()
        plane_y_2=torch.from_numpy(plane[5]).long()
        plane_x_3=torch.from_numpy(plane[6]).long()
        plane_y_3=torch.from_numpy(plane[7]).long()
        plane_num_0=torch.from_numpy(plane[8]).long().squeeze()
        plane_num_1=torch.from_numpy(plane[9]).long().squeeze()
        plane_num_2=torch.from_numpy(plane[10]).long().squeeze()
        plane_num_3=torch.from_numpy(plane[11]).long().squeeze()
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
        #print(s_plane[-1]==[])
        for k in range(len(s_plane)):
            #print(s_plane[k].shape,k)
            if s_plane[k].shape[0]==0:
                
                s_plane[k]=np.concatenate([s_plane[k],-np.ones(1)])
        s_plane_x_0=torch.from_numpy(s_plane[0]).long()
        s_plane_y_0=torch.from_numpy(s_plane[1]).long()
        s_plane_x_1=torch.from_numpy(s_plane[2]).long()
        s_plane_y_1=torch.from_numpy(s_plane[3]).long()
        s_plane_x_2=torch.from_numpy(s_plane[4]).long()
        s_plane_y_2=torch.from_numpy(s_plane[5]).long()
        s_plane_x_3=torch.from_numpy(s_plane[6]).long()
        s_plane_y_3=torch.from_numpy(s_plane[7]).long()
        s_plane_num_0=torch.from_numpy(s_plane[8]).long().squeeze()
        #print(s_plane_num_0.shape)
        s_plane_num_1=torch.from_numpy(s_plane[9]).long().squeeze()
        s_plane_num_2=torch.from_numpy(s_plane[10]).long().squeeze()
        s_plane_num_3=torch.from_numpy(s_plane[11]).long().squeeze()
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
        #print(l_plane[-1],l_plane[-1]==[])
        for k in range(len(l_plane)):
            if l_plane[k].shape[0]==0:
                #print('lnon'+str(k))
                l_plane[k]=np.concatenate([l_plane[k],-np.ones(1)])
        l_plane_x_0=torch.from_numpy(l_plane[0]).long()
        l_plane_y_0=torch.from_numpy(l_plane[1]).long()
        l_plane_x_1=torch.from_numpy(l_plane[2]).long()
        l_plane_y_1=torch.from_numpy(l_plane[3]).long()
        l_plane_x_2=torch.from_numpy(l_plane[4]).long()
        l_plane_y_2=torch.from_numpy(l_plane[5]).long()
        l_plane_x_3=torch.from_numpy(l_plane[6]).long()
        l_plane_y_3=torch.from_numpy(l_plane[7]).long()
        l_plane_num_0=torch.from_numpy(l_plane[8]).long().squeeze()
        l_plane_num_1=torch.from_numpy(l_plane[9]).long().squeeze()
        l_plane_num_2=torch.from_numpy(l_plane[10]).long().squeeze()
        l_plane_num_3=torch.from_numpy(l_plane[11]).long().squeeze()
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
