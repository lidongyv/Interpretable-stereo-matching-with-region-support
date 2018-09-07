# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-07-17 10:44:43
# @Last Modified by:   yulidong
# @Last Modified time: 2018-09-06 16:34:27
# -*- coding: utf-8 -*-
# @Author: lidong
# @Date:   2018-03-20 18:01:52
# @Last Modified by:   yulidong
# @Last Modified time: 2018-07-16 22:16:14
import time
import torch
import numpy as np
import torch.nn as nn
import math
from math import ceil
from torch.autograd import Variable
from torch.nn.functional import cosine_similarity as cosine_s
from pssm import caffe_pb2
from pssm.models.utils import *
rsn_specs = {
    'scene': 
    {
         'n_classes': 9,
         'input_size': (540, 960),
         'block_config': [3, 4, 23, 3],
    },

}

group_dim=8

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""

    if stride==1:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
    if stride==2:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=2, bias=False) 
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.gn1 = nn.GroupNorm(group_dim,planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.gn2 = nn.GroupNorm(group_dim,planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
            # print(residual.shape)
            # print(out.shape)
        out += residual
        out = self.relu(out)

        return out
class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()
        self.inplanes = 32
        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1,1,1)

        self.branch1 = nn.Sequential(nn.AvgPool2d((54, 96), stride=(54,96)),
                                     nn.Conv2d(32, 8, 1, 1, 0, 1),
                                     nn.GroupNorm(4,8),
                                     nn.ReLU(inplace=False))

        self.branch2 = nn.Sequential(nn.AvgPool2d((27, 48), stride=(27,48)),
                                     nn.Conv2d(32, 8, 1, 1, 0, 1),
                                     nn.GroupNorm(4,8),
                                     nn.ReLU(inplace=False))

        self.branch3 = nn.Sequential(nn.AvgPool2d((36, 64), stride=(36,64)),
                                     nn.Conv2d(32, 8, 1, 1, 0, 1),
                                     nn.GroupNorm(4,8),
                                     nn.ReLU(inplace=False))

        self.branch4 = nn.Sequential(nn.AvgPool2d((18, 32), stride=(18,32)),
                                     nn.Conv2d(32, 8, 1, 1, 0, 1),
                                     nn.GroupNorm(4,8),
                                     nn.ReLU(inplace=False))
        self.branch5 = nn.Sequential(nn.AvgPool2d((9, 16), stride=(9,16)),
                                     nn.Conv2d(32, 8, 1, 1, 0, 1),
                                     nn.GroupNorm(4,8),
                                     nn.ReLU(inplace=False))
        self.branch6 = nn.Sequential(nn.AvgPool2d((3, 8), stride=(3,8)),
                                     nn.Conv2d(32, 8, 1, 1, 0, 1),
                                     nn.GroupNorm(4,8),
                                     nn.ReLU(inplace=False))


        self.lastconv = nn.Sequential(nn.Conv2d(80, 64, 3, 1, 1, 1),
                                      nn.GroupNorm(group_dim,64),
                                      nn.ReLU(inplace=False),
                                      nn.Conv2d(64, 32, 3, 1, 1, 1),
                                      nn.GroupNorm(group_dim,32),
                                      nn.ReLU(inplace=False),  
                                      )

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None

        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1))

        return nn.Sequential(*layers)

    def forward(self, x):
        # output = self.conv1(x)
        # output = self.gn1(output)
        # output = self.relu1(output)
        # output = self.conv2(output)
        # output = self.gn2(output)
        # output = self.relu2(output)
        # output = self.conv3(output)
        # output = self.gn3(output)
        # output = self.relu3(output)
        output_skip = self.layer1(x)
        # output_skip=x

        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.interpolate(output_branch1, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear',align_corners=True)

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.interpolate(output_branch2, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear',align_corners=True)

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.interpolate(output_branch3, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear',align_corners=True)

        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.interpolate(output_branch4, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear',align_corners=True)

        output_branch5 = self.branch5(output_skip)
        output_branch5 = F.interpolate(output_branch5, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear',align_corners=True)

        output_branch6 = self.branch6(output_skip)
        output_branch6 = F.interpolate(output_branch6, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear',align_corners=True)

        output_feature = torch.cat((output_skip, output_branch6, output_branch5, output_branch4, output_branch3, output_branch2, output_branch1), 1)
        output_feature = self.lastconv(output_feature)
        #print(output_feature.shape)
        return output_feature

class feature_extraction2(nn.Module):
    def __init__(self):
        super(feature_extraction2, self).__init__()
        self.inplanes = 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1,
                               bias=False,dilation=1)
        self.gn1 = nn.GroupNorm(group_dim,32)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1,
                               bias=False,dilation=1)
        self.gn2 = nn.GroupNorm(group_dim,32)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=7, stride=1, padding=6,
                               bias=False,dilation=2)
        self.gn3 = nn.GroupNorm(group_dim,32)
        self.relu3 = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(BasicBlock, 32, 1, 1,1,1)

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,))

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.gn1(output)
        output = self.relu1(output)
        output = self.conv2(output)
        output = self.gn2(output)
        output = self.relu2(output)
        output = self.conv3(output)
        output = self.gn3(output)
        output = self.relu3(output)
        #print(output.shape)
        output = self.layer1(output)


        return output





class rstereo(nn.Module):


    def __init__(self, 
                 n_classes=9, 
                 block_config=[3, 4, 6, 3], 
                 input_size= (480, 640), 
                 version='scene'):

        super(rstereo, self).__init__()
        self.feature_extraction=feature_extraction().cuda(0)
        self.feature_extraction2=feature_extraction2().cuda(0)
        self.softmax= nn.Softmax(dim=-1)
    def ss_argmin(self,x,index):
        one=torch.ones(1)
        zero=torch.zeros(1)
        x=self.softmax(x)
        #print(x)
        disparity= torch.sum(x*index.unsqueeze(0),dim=-1)
        return disparity     
    def cluster_vector(self,feature,x,y):
        one=torch.ones(1).cuda(1)
        zero=torch.zeros(1).cuda(1)
        cluster_feature=feature[...,x,y]
        mean=torch.sum(cluster_feature,dim=-1)/x.shape[0]
        mean=mean.view(cluster_feature.shape[0],cluster_feature.shape[1],1)
        #print(mean.shape)
        weights=torch.norm(cluster_feature-mean,dim=1)
        weights=torch.exp(-weights)
        return weights
    def forward(self, l,r,P,pre,matching,aggregation):

        start_time=time.time()
        with torch.no_grad():
          self.pre=pre.cuda(1)
        P1=P[...,0].cuda(1)
        P2=P[...,3].cuda(1)
        P3=P[...,1].cuda(1)
        P4=P[...,2].cuda(1)

        l_mask=P2-P1
        s_mask=P1

        l_sf=self.feature_extraction2(l)
        l_lf=self.feature_extraction(l_sf)


        r_sf=self.feature_extraction2(r)
        r_lf=self.feature_extraction(r_sf)


        disparity=torch.zeros([540,960]).cuda(0)
        one=torch.ones(1).cuda(1)
        zero=torch.zeros(1).cuda(1)

        l_lf=l_lf.cuda(1)
        r_lf=r_lf.cuda(1)
        r_sf=r_sf.cuda(1)
        l_sf=l_sf.cuda(1)

        count=0


        start_time=time.time()
        for i in range(torch.max(P3).type(torch.int32)):
            mask_object=torch.where(P3==i,one,zero)
            object_pixel_index=mask_object.nonzero()
            x1,y1,x2,y2=torch.min(object_pixel_index[:,0]),torch.min(object_pixel_index[:,1]) \
                             torch.max(object_pixel_index[:,0]),torch.max(object_pixel_index[:,1])
            #object_pixel_index=mask_object[...,x1:x2,y1:y2].nonzero()
            max_d=192*one
            min_d=zero


            s_feature=l_sf[...,object_pixel_index[:,0],object_pixel_index[:,1]]

            s_r_y=torch.max(matching[1][i]-matching[2][i],-torch.ones_like(matching[2][i]))
            s_r_o_t=r_sf[...,x1:x2,y1:y2][...,matching[0][i],s_r_y]
            s_cost=torch.where(s_r_y>=0,torch.sum(s_feature*s_r_o_t,1),-one*1e+8)
            #s_cost=torch.where(s_r_y>=0,cosine_s(s_feature,s_r_o_t),-one*1e+8)
            d=matching[2][i]-min_d
            #cost_volume[matching[0][i],matching[1][i],d]=s_cost
            disparity[x1:x2,y1:y2][matching[-5][i],matching[-4][i]]=self.ss_argmin(s_cost.view(1,matching[-5][i].shape[1],matching[-1][i].shape[1]).cuda(0),matching[-1][i].float().cuda(0))
            if torch.max(matching[3][i])>0:
                l_feature=l_lf[...,x1:x2,y1:y2][...,matching[3][i],matching[4][i]]
                l_r_y=torch.max(matching[4][i]-matching[5][i],-torch.ones_like(matching[5][i]))
                l_r_o_t=r_lf[...,x1:x2,y1:y2][...,matching[3][i],l_r_y]
                d=matching[5][i]-min_d
                l_cost=torch.where(l_r_y>=0,torch.sum(l_feature*l_r_o_t,1),-one*1e+8)
                #l_cost=torch.where(l_r_y>=0,cosine_s(l_feature,l_r_o_t),-one*1e+8)
                #cost_volume[matching[3][i],matching[4][i],d]=l_cost
                disparity[x1:x2,y1:y2][matching[-3][i],matching[-2][i]]=self.ss_argmin(l_cost.view(1,matching[-3][i].shape[1],matching[-1][i].shape[1]).cuda(0),matching[-1][i].float().cuda(0))
            #disparity[matching[-5][i],matching[-4][i]]=self.ss_argmin(cost_volume[matching[-5][i],matching[-4][i],:].cuda(0),matching[-1][i].float().cuda(0))
            #disparity[matching[-3][i],matching[-2][i]]=self.ss_argmin(cost_volume[matching[-3][i],matching[-2][i],:].cuda(0),matching[-1][i].float().cuda(0))
            #print(torch.max(disparity[matching[-5][i],matching[-4][i]]).item(),torch.min(disparity[matching[-5][i],matching[-4][i]]).item())
            #print(torch.max(disparity[matching[-3][i],matching[-2][i]]).item(),torch.min(disparity[matching[-3][i],matching[-2][i]]).item())            
            # print(matching[-1][i])
            # print(matching[-5][i],matching[-1][i],s_cost.shape,matching[2][i].shape)
            # print(s_cost.view(1,matching[-5][i].shape[1],matching[-1][i].shape[1]).shape)
            # disparity[matching[-5][i],matching[-4][i]]=self.ss_argmin(cost_volume[matching[-5][i],matching[-4][i],:].cuda(0),matching[-1][i].float().cuda(0))
            # disparity[matching[-3][i],matching[-2][i]]=self.ss_argmin(cost_volume[matching[-3][i],matching[-2][i],:].cuda(0),matching[-1][i].float().cuda(0))
        print(time.time()-start_time)
        #print(torch.max(torch.where(s_mask+l_mask>zero,disparity,zero)),torch.min(torch.where(s_mask+l_mask>zero,disparity,192*one)))
        #time.sleep(1000)
            
        return disparity


