# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-07-17 10:44:43
# @Last Modified by:   yulidong
# @Last Modified time: 2018-09-05 19:11:51
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
        self.relu = nn.LeakyReLU(inplace=True)
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
                                     nn.LeakyReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((27, 48), stride=(27,48)),
                                     nn.Conv2d(32, 8, 1, 1, 0, 1),
                                     nn.GroupNorm(4,8),
                                     nn.LeakyReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.AvgPool2d((36, 64), stride=(36,64)),
                                     nn.Conv2d(32, 8, 1, 1, 0, 1),
                                     nn.GroupNorm(4,8),
                                     nn.LeakyReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d((18, 32), stride=(18,32)),
                                     nn.Conv2d(32, 8, 1, 1, 0, 1),
                                     nn.GroupNorm(4,8),
                                     nn.LeakyReLU(inplace=True))
        self.branch5 = nn.Sequential(nn.AvgPool2d((9, 16), stride=(9,16)),
                                     nn.Conv2d(32, 8, 1, 1, 0, 1),
                                     nn.GroupNorm(4,8),
                                     nn.LeakyReLU(inplace=True))
        self.branch6 = nn.Sequential(nn.AvgPool2d((3, 8), stride=(3,8)),
                                     nn.Conv2d(32, 8, 1, 1, 0, 1),
                                     nn.GroupNorm(4,8),
                                     nn.LeakyReLU(inplace=True))


        self.lastconv = nn.Sequential(nn.Conv2d(80, 64, 3, 1, 1, 1),
                                      nn.GroupNorm(group_dim,64),
                                      nn.LeakyReLU(inplace=True),
                                      nn.Conv2d(64, 32, 3, 1, 1, 1),
                                      nn.GroupNorm(group_dim,32),
                                      nn.LeakyReLU(inplace=True),  
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
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1,
                               bias=False,dilation=1)
        self.gn2 = nn.GroupNorm(group_dim,32)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=7, stride=1, padding=6,
                               bias=False,dilation=2)
        self.gn3 = nn.GroupNorm(group_dim,32)
        self.relu3 = nn.LeakyReLU(inplace=True)
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

class ss_argmin(nn.Module):
    def __init__(self):
        super(ss_argmin, self).__init__()
        self.softmax = nn.Softmax(dim=-1)



    def forward(self,x,min,max):
        one=torch.ones(1)
        zero=torch.zeros(1)
        x=self.softmax(x)
        index=torch.ones_like(x)*torch.range(min,max)
        disparity= torch.sum(x*index,dim=-1)
        v,i=torch.topk(x,k=1,dim=-1)
        mask_1=torch.squeeze(torch.where(v>0.7,one,zero))
        v,i=torch.topk(x,k=5,dim=-1)
        v_sum=torch.sum(v,-1)
        mask_2=torch.squeeze(torch.where(v_s>0.7,one,zero))
        i_dis=torch.max(i,-1)[0]-torch.min(i,-1)[0]
        mask_3=torch.squeeze(torch.where(i_dis<6,one,zero))
        mask=mask_1+mask_2*mask_3
        mask=torch.where(mask>0,one,zero)
        return disparity*mask



class rstereo(nn.Module):


    def __init__(self, 
                 n_classes=9, 
                 block_config=[3, 4, 6, 3], 
                 input_size= (480, 640), 
                 version='scene'):

        super(rstereo, self).__init__()
        self.feature_extraction=feature_extraction().cuda(0)
        self.feature_extraction2=feature_extraction2().cuda(0)
        # self.aggregation_sparse=aggregation_sparse()
        # self.aggregation_dense=aggregation_dense()   
        self.ss_argmin=ss_argmin()
        # self.refinement_sparse=aggregation_sparse()
        # self.refinement_dense=aggregation_dense()        
    def cluster_vector(self,feature,x,y):
        one=torch.ones(1).cuda(1)
        zero=torch.zeros(1).cuda(1)
        cluster_feature=feature[...,x,y]
        mean=torch.sum(cluster_feature,dim=-2)/x.shape[0]
        weights=torch.norm(cluster_feature-mean,dim=0),zero
        weights=torch.exp(-weights),zero
        return weights
    def forward(self, l,r,P,pre,matching):
        #self.P=P[1,0]
        #0 l to r,1 min,2 max
        #[l_box,r_box,match],[min_d,max_d]
        start_time=time.time()
        with torch.no_grad():
          self.pre=pre.cuda(1)
        P1=P[...,0].cuda(1)
        P2=P[...,3].cuda(1)
        P3=P[...,1].cuda(1)
        P4=P[...,2].cuda(1)
        #feature extraction
        l_mask=P2-P1
        s_mask=P1
        #l_mask=l_mask.byte()
        #s_mask=s_mask.byte()
        #basic cuda 524
        #print(l.type)
         #1923
        #print(torch.cuda.memory_allocated(1))
        #2727
        l_sf=self.feature_extraction2(l)
        l_lf=self.feature_extraction(l_sf)

        #print(torch.cuda.memory_allocated(2))
        #the cuda won't copy the volume to the new gpu
        # a=l_lf.cuda(1)
        # b=l_lf.cuda(1)
        # c=l_sf.cuda(3)
        r_sf=self.feature_extraction2(r)
        r_lf=self.feature_extraction(r_sf)
        #print(torch.cuda.memory_allocated(1))
        #3267
        
        #print(torch.cuda.memory_allocated(2))
        #reshape the mask to batch and channel

        disparity=torch.zeros([540,960]).cuda(1)
        one=torch.ones(1).cuda(1)
        zero=torch.zeros(1).cuda(1)
        #cost_volume=[]
        #5710
        #print(value)
        l_lf=l_lf.cuda(1)
        r_lf=r_lf.cuda(1)
        r_sf=r_sf.cuda(1)
        l_sf=l_sf.cuda(1)
        #985
        #feature=torch.masked_select(l_sf,s_mask)
        #feature=torch.masked_select(l_lf,l_mask)+torch.masked_select(l_sf,s_mask)
        # feature=l_lf*l_mask+l_sf*s_mask
        # feature=torch.where((l_mask+s_mask)>0,feature,l_lf)


        # cost_s=[]
        # cost_l=[]
        # for m in range(10):
        count=0
        #start_time=time.time()
        #with torch.no_grad():
        for z in range(10):
          start_time=time.time()
          for i in range(torch.max(P3).type(torch.int32)):
            x1,y1,x2,y2,size=pre[0,i].long()
            max_d=torch.max(matching[-1][i])
            min_d=torch.min(matching[-1][i])
            
            cost_volume=torch.zeros(x2-x1,y2-y1,max_d-min_d+1).cuda(1)
            #ground 0-270, sky 0-40
            #intial 0.46, after 0.18,volume 0.3
            #cost computation intial 0.20,after 0.14,volume 0.3
            if torch.max(matching[0][i])>0:
              s_feature=l_sf[...,x1:x2,y1:y2][...,matching[0][i],matching[1][i]]
              s_r_y=torch.max(matching[1][i]-matching[2][i],-torch.ones_like(matching[2][i]))
              #print(s_r_y)
              s_r_o_t=r_sf[...,x1:x2,y1:y2][...,matching[0][i],s_r_y]
              #cost_volume[matching[0][i],matching[1][i],matching[2][i]-min_d]=torch.where(s_r_y>=0,cosine_s(s_feature,s_r_o_t),zero)
              s_cost=torch.where(s_r_y>=0,cosine_s(s_feature,s_r_o_t),zero)
              d=matching[2][i]-min_d
              cost_volume[matching[0][i],matching[1][i],d]=s_cost
            if torch.max(matching[3][i])>0:
              l_feature=l_lf[...,x1:x2,y1:y2][...,matching[3][i],matching[4][i]]
              l_r_y=torch.max(matching[4][i]-matching[5][i],-torch.ones_like(matching[5][i]))
              l_r_o_t=r_lf[...,x1:x2,y1:y2][...,matching[3][i],l_r_y]
              #print(min_d,torch.min(matching[2][i]),torch.min(matching[3][i]),torch.min(matching[4][i]))
              d=matching[5][i]-min_d
              #cost_volume[matching[3][i],matching[4][i],d]=torch.where(l_r_y>=0,cosine_s(l_feature,l_r_o_t),zero)
              l_cost=torch.where(l_r_y>=0,cosine_s(l_feature,l_r_o_t),zero)
              cost_volume[matching[3][i],matching[4][i],d]=l_cost
            for j in range(len(plane)):
              if s_plane[i][0].shape[0]>0:
                s_weights=self.cluster_vector(l_sf[...,x1:x2,y1:y2], s_plane[i][0], s_plane[i][1])
                mean_cost=torch.sum(cost_volume[s_plane[i][0], s_plane[i][1]]*s_weights,0,keepdim=True)/torch.sum(s_weights,0).cuda(2)
                cost_volume[s_plane[i][0], s_plane[i][1]]=mean_cost*s_weights+(1-s_weights)*costs

          print(time.time()-start_time)
        


            #print(time.time()-start_time)
        print(time.time()-start_time)
        #3s,4600mb
        #print(time.time()-start_time)  
        time.sleep(100)            
            # #ss_argmin
            # disparity[...,x1:x2,y1:y2]=ss_argmin(cost_volume,min_d,max_d)
            # #refinement
            # refine=torch.zeros_like(disparity)[...,x1:x2,y1:y2]
            # for j in range(min_r,max_r+1):
            #     plane_mask=torch.where(object_r==j,one,zero)[x1:x2,y1:y2]
            #     xp1,xp2,yp1,yp2=crop(plane_mask)
            #     plane_mask=plane_mask[xp1:xp2,yp1:yp2]
            #     s_plane_mask=plane_mask*s_mask[x1:x2,y1:y2][xp1:xp2,yp1:yp2]
            #     l_plane_mask=plane_mask*l_mask[x1:x2,y1:y2][xp1:xp2,yp1:yp2]
            #     plane_mask=plane_mask-torch.where(s_plane_mask+l_plane_mask>0,one,zero)
            #     plane=disparity[...,x1:x2,y1:y2][...,xp1:xp2,yp1:yp2]*plane_mask
            #     s_weights=self.cluster(l_sf[...,x1:x2,y1:y2][...,xp1:xp2,yp1:yp2],s_plane_mask)
            #     s_cost=torch.sum(torch.sum(plane*s_weights,-2,keepdim=True),-3,keepdim=True)/torch.sum(s_weights)
            #     l_weights=self.cluster(l_lf[...,x1:x2,y1:y2][...,xp1:xp2,yp1:yp2],l_plane_mask)
            #     l_cost=torch.sum(torch.sum(plane*l_weights,-2),-2)/torch.sum(l_weights)
            #     weights=self.cluster(l_lf[...,x1:x2,y1:y2][...,xp1:xp2,yp1:yp2],plane_mask)
            #     costs=torch.sum(torch.sum(plane*weights,-2,keepdim=True),-3,keepdim=True)/torch.sum(weights)
            #     plane=torch.where(s_plane_mask==1,s_cost*s_weights,plane)
            #     plane=torch.where(l_plane_mask==1,l_cost*l_weights,plane)
            #     plane=torch.where(plane_mask==1,cost*weights,plane)
            #     disparity[...,x1:x2,y1:y2][...,xp1:xp2,yp1:yp2]=plane
            
        return cost_volume


