# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-03-19 13:33:07
# @Last Modified by:   yulidong
# @Last Modified time: 2018-09-09 16:40:05

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
        self.stats={'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}
        self.left_files = {}
        self.datapath=root
        self.left_files=os.listdir(os.path.join(self.datapath,'left'))
        self.match_files=os.listdir(os.path.join(self.datapath,'match'))
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
        index=0
        data=np.load(os.path.join(self.datapath,'left',self.left_files[index]))
        data=data[:540,:960,:]
        left=data[...,0:3]/255
        #print(data.shape)
        right=data[...,3:6]/255
        disparity=data[...,6]
        P=data[...,7:]
        pre_match=np.load(os.path.join(self.datapath,'match',self.left_files[index]))
        matching=np.load(os.path.join(self.datapath,'matching',self.left_files[index]))
        aggregation=np.load(os.path.join(self.datapath,'aggregation',self.left_files[index]))
        #print('load')
        if self.is_transform:
            left, right,disparity,P,pre_match,matching,aggregation = self.transform(left, right,disparity,P,pre_match,matching,aggregation)
        if self.task=='generation':
            #print('value')
            return left, right,disparity,P,pre_match,matching,aggregation
        else:
            return left, right,disparity,P,pre_match,matching,aggregation
    def transform(self, left, right,disparity,P,pre,matching,aggregation):
        """transform
        """
        trans=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**self.stats),
        ])
        # left=transforms.ToTensor()(left)
        # left=transforms.Normalize(**self.stats)(left)
     
        left=trans(left).float()
        right=trans(right).float()

        disparity=torch.from_numpy(disparity).float()
        P=torch.from_numpy(P).float()
        s_match_x=[]
        s_match_y=[]
        s_shfit=[]
        s_repeat=np.array(matching[3])
        l_match_x=[]
        l_match_y=[]
        l_shfit=[]
        l_repeat=np.array(matching[7])   
        # for m in range(len(matching)):
        #     for n in range(len(matching[m])):
        #         print(m,n,matching[m][n].shape)
        d=[]
        s_x=[]
        s_y=[]
        l_x=[]
        l_y=[]
        for i in range(s_repeat.shape[0]):
            #print(s_repeat[i].shape)
            #print(len(matching))
            d.append(torch.from_numpy(np.array(matching[2][i])).long())
            s_x.append(torch.from_numpy(np.array(matching[0][i])).long())
            s_y.append(torch.from_numpy(np.array(matching[1][i])).long())
            shift=np.tile(matching[2][i],s_repeat[i][1])
            s_match_x_t=matching[0][i].repeat(s_repeat[i][0])
            s_match_y_t=matching[1][i].repeat(s_repeat[i][0])
            # print(shift,s_match_x_t,s_match_y_t)
            # exit()
            s_shfit.append(torch.from_numpy(np.array(shift)).long())
            s_match_x.append(torch.from_numpy(np.array(s_match_x_t)).long())
            s_match_y.append(torch.from_numpy(np.array(s_match_y_t)).long())
        for i in range(l_repeat.shape[0]):
            #l_d.append(matching[6][i])
            l_x.append(torch.from_numpy(np.array(matching[4][i])).long())
            l_y.append(torch.from_numpy(np.array(matching[5][i])).long())
            shift=np.tile(matching[6][i],l_repeat[i][1])
            l_match_x_t=matching[4][i].repeat(l_repeat[i][0])
            l_match_y_t=matching[5][i].repeat(l_repeat[i][0])
            l_shfit.append(torch.from_numpy(np.array(shift)).long())
            l_match_x.append(torch.from_numpy(np.array(l_match_x_t)).long())
            l_match_y.append(torch.from_numpy(np.array(l_match_y_t)).long())

        matching=[s_match_x,s_match_y,s_shfit,l_match_x,l_match_y,l_shfit,s_x,s_y,l_x,l_y,d]
        plane=[]
        #print(len(aggregation))
        for i in range(len(aggregation[0])):
            plane_t=[]
            for j in range(len(aggregation[0][i])):
                plane_t.append(torch.from_numpy(np.array(aggregation[0][i][j])).long())
            plane.append(plane_t)
        plane_num=[]
        for i in range(len(aggregation[1])):
            plane_num.append(torch.from_numpy(aggregation[1][i]).long())
        s_plane=[]
        for i in range(len(aggregation[2])):
            plane_t=[]
            for j in range(len(aggregation[2][i])):
                plane_t.append(torch.from_numpy(np.array(aggregation[2][i][j])).long())
            s_plane.append(plane_t)
        s_plane_num=[]
        for i in range(len(aggregation[3])):
            s_plane_num.append(torch.from_numpy(aggregation[3][i]).long())
        l_plane=[]
        for i in range(len(aggregation[4])):
            plane_t=[]
            for j in range(len(aggregation[4][i])):
                plane_t.append(torch.from_numpy(np.array(aggregation[4][i][j])).long())
            l_plane.append(plane_t)
        l_plane_num=[]
        for i in range(len(aggregation[5])):
            l_plane_num.append(torch.from_numpy(aggregation[5][i]).long())
        aggregation=[plane,plane_num,s_plane,s_plane_num,l_plane,l_plane_num]




        #pre1=torch.from_numpy(pre[1,0]).float()
        #print(pre1.shape)
        #pre1=torch.cat([pre1,torch.zeros([pre1.shape[0],pre1.shape[1],100-pre1.shape[2]])],-1)
        #max 142
        pre2=torch.from_numpy(pre[0,0][0]).float()
        #print(pre2.shape)
        #pre2=torch.cat([pre2,torch.zeros([pre2.shape[0],100-pre2.shape[1],pre2.shape[2],pre2.shape[3],pre2.shape[4],pre2.shape[5],pre2.shape[6]])])

        #print(pre1.shape)
        return left,right,disparity,P,pre2,matching,aggregation
