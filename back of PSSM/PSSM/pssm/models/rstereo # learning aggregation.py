# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-07-17 10:44:43
# @Last Modified by:   yulidong
# @Last Modified time: 2018-08-24 22:21:40
# -*- coding: utf-8 -*-
# @Author: lidong
# @Date:   2018-03-20 18:01:52
# @Last Modified by:   yulidong
# @Last Modified time: 2018-07-16 22:16:14

import torch
import numpy as np
import torch.nn as nn
import math
from math import ceil
from torch.autograd import Variable
from torch.nn.functional import cosine_similarity as cosine_s
from rsden import caffe_pb2
from rsden.models.utils import *
rsn_specs = {
    'scene': 
    {
         'n_classes': 9,
         'input_size': (540, 960),
         'block_config': [3, 4, 23, 3],
    },

}


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation
                         if dilation > 1 else pad, dilation = dilation, bias=False),
                         nn.BatchNorm2d(out_planes))
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out
class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()
        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1,1,1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 1,1,1) 
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1,1,1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1,1,2)

        self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64,64)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32,32)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16,16)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8,8)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 32, kernel_size=1, padding=0, stride = 1, bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        output      = self.firstconv(x)
        output      = self.layer1(output)
        output_raw  = self.layer2(output)
        output      = self.layer3(output_raw)
        output_skip = self.layer4(output)


        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.upsample(output_branch1, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.upsample(output_branch2, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.upsample(output_branch3, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.upsample(output_branch4, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_feature = torch.cat((output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1)
        output_feature = self.lastconv(output_feature)

        return output_feature

class feature_extraction2(nn.Module):
    def __init__(self):
        super(feature_extraction2, self).__init__()
        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1,1,1)

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        output      = self.firstconv(x)
        output      = self.layer1(output)


        return output


class aggregation_sparse(nn.Module):
    def __init__(self):
        super(aggregation_sparse, self).__init__()
        self.s_conv = nn.Sequential(convbn(33, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 1, 3, 1, 1, 1)
                                       )
        self.l_conv = nn.Sequential(convbn(33, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))
        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1,1,1)

        self.l_conv2 = nn.Sequential(convbn(33, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 1, 3, 1, 1, 1))
    def forward(self, s,l,x):
        s_var = self.s_conv(torch.cat([s,x],1))+x
        l_var = self.l_conv(torch.cat([l,x],1))
        l_var = self.layer1(l_var)
        l_var = self.l_conv2(l_var)+x
        return s_var,l_var
class aggregation_dense(nn.Module):
    def __init__(self):
        super(aggregation, self).__init__()
        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(33, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1)
                                       )
        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1,1,1)
        self.layer3 = self._make_layer(BasicBlock, 64, 3, 1,1,1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1,1,2)
        self.lastconv = nn.Sequential(convbn(128, 64, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(32, 1, kernel_size=1, padding=0, stride = 1, bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)
    def forward(self, f,x):

        output = self.firstconv(torch.cat([f,x],1))
        output = self.layer1(output)
        output = self.layer3(output)
        output = self.layer4(output)
        dense = self.lastconv(output_feature)
        dense = torch.where(x>0,x,dense)
        return dense
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



class EDSNet(nn.Module):


    def __init__(self, 
                 P,
                 pre,
                 n_classes=9, 
                 block_config=[3, 4, 6, 3], 
                 input_size= (480, 640), 
                 version='scene'):

        super(EDSNet, self).__init__()
        self.feature_extraction=feature_extraction()
        self.feature_extraction2=feature_extraction2()
        self.aggregation_sparse=aggregation_sparse()
        self.aggregation_dense=aggregation_dense()   
        self.ss_argmin=ss_argmin()
        # self.refinement_sparse=aggregation_sparse()
        # self.refinement_dense=aggregation_dense()        
        self.P=P
        #0 l to r,1 min,2 max
        #[l_box,r_box,match],[min_d,max_d]
        self.pre=pre[1,0]
        self.pre2=pre[0,0]
    def crop(self,x):
        index=(x==1).nonzero()
        return torch.min(index[:,0]),torch.max(index[:,0])+1,torch.min(index[:,1]),torch.max(index[:,1]+1)
    def cluster(feature,mask):
        count=torch.sum(mask)
        mean=torch.sum(torch.sum(feature,dim=-1),dim=-1)/count
        weights=torch.where(mask==ones,torch.norm(feature-mean,dim=1),zeros)
        weights=weights/torch.sum(weights)
        return weights
    def forward(self, l,r):
        P1=self.P[...,0]
        P2=self.P[...,3]
        P3=self.P[...,1]
        P4=self.P[...,2]
        #feature extraction
        l_mask=P2-P1
        s_mask=P1
        l_lf=self.feature_extraction(l)
        l_sf=self.feature_extraction2(l)
        r_lf=self.feature_extraction(r)
        r_sf=self.feature_extraction2(r)
        #reshape the mask to batch and channel
        feature=l_lf*l_mask+l_sf*s_mask
        feature=torch.where((l_mask+s_mask)>0,feature,l_lf)
        disparity=torch.zeros([540,960]).cuda()
        one=torch.ones(1).cuda()
        zero=torch.zeros(1).cuda()
        cost_volume=[]
        #promotion
        #we can segment with bounding box and divide the whole image into many parts
        #each single bounding box will be managed through network not the whole image
        #matching cost computation
        for i in range(torch.max(self.P[:,:,1]).type(torch.int32)+1):
            min_d=self.pre[1,0][0,i]
            max_d=self.pre[1,0][1,i]
            object_mask=torch.where(P[:,:,1]==i,one,zero)
            #x1,y1,x2,y2=crop(object_mask)
            x1,y1,x2,y2,size=self.pre2[0]
            object_mask=object_mask[x1:x2,y1:y2]
            s_mask_o=object_mask*s_mask[x1:x2,y1:y2]
            l_mask_o=object_mask*l_mask[x1:x2,y1:y2]
            s_l_o=feature[...,x1:x2,y1:y2]*s_mask_o
            l_l_o=feature[...,x1:x2,y1:y2]*l_mask_o
            s_r_o=r_sf[...,x1:x2,torch.max(min_d,zero):torch.min(max_d,one*960)]
            l_r_o=r_lf[...,x1:x2,torch.max(min_d,zero):torch.min(max_d,one*960)]
            min_d=y2-max_d
            max_d=y1-min_d
            cost_s=[]
            cost_l=[]
            for i in range(0,max_d-min_d):
              if y2-y1-i>0:
                s_r_o_t=s_r_o[...,y2-y1-i:-i]
                cost_s.append(torch.where(s_mask_o==1,cosine_s(s_l_o,s_r_o_t),zero))
              else:
                s_r_o_t=torch.cat([torch.zeros_like(s_r_o[...,:y2-y1-i]),s_r_o[...,0:-i]],-1)
                cost_s.append(torch.where(s_mask_o==1,cosine_s(s_l_o,s_r_o_t),zero))
            cost_s=torch.stack(cost_s,-1)
            for i in range(0,max_d-min_d):
              if y2-y1-i>0:
                l_r_o_t=l_r_o[...,y2-y1-i:-i]
                cost_l.append(torch.where(l_mask_o==1,cosine_s(l_l_o,l_r_o_t),zero))
              else:
                l_r_o_t=torch.cat([torch.zeros_like(l_r_o[...,:y2-y1-i]),l_r_o[...,0:-i]],-1)
                cost_l.append(torch.where(l_mask_o==1,cosine_s(l_l_o,l_r_o_t),zero))                
            cost_l=torch.stack(cost_l,-1)
            cost_volume=cost_s+cost_l
            #aggregation
            a_volume=torch.zeros_like(cost_volume)
            object_r=torch.where(P3==i,P4,zero)
            max_r=torch.max(object_r)
            object_r=torch.where(P3==i,P4,max_r+1)
            min_r=torch.min(object_r)
            for j in range(min_r,max_r+1):
                plane_mask=torch.where(object_r==j,one,zero)[x1:x2,y1:y2]
                xp1,xp2,yp1,yp2=crop(plane_mask)
                #xp1,xp2,yp1,yp2.r_size=self.pre[0,0][1]
                plane_mask=plane_mask[xp1:xp2,yp1:yp2]
                plane=cost_volume[...,xp1:xp2,yp1:yp2,:]
                for m in range(planes.shape[-1])
                    s_var,l_var=self.aggregation_sparse(l_sf[...,x1:x2,y1:y2][...,xp1:xp2,yp1:yp2],l_lf[...,x1:x2,y1:y2][...,xp1:xp2,yp1:yp2],plane[...,m]*plane_mask)
                    plane[...,m]=s_var*s_mask[x1:x2,y1:y2][xp1:xp2,yp1:yp2]+l_var*l_mask[x1:x2,y1:y2][xp1:xp2,yp1:yp2]
                    plane[...,m]=plane[...,m]*plane_mask
                    plane[...,m]=self.aggregation_dense(feature[...,x1:x2,y1:y2][...,xp1:xp2,yp1:yp2],plane[...,m])*plane_mask
                cost_volume[...,xp1:xp2,yp1:yp2,:]=plane
            #ss_argmin
            disparity[...,x1:x2,y1:y2]=ss_argmin(cost_volume,min_d,max_d)
            #refinement
            refine=torch.zeros_like(disparity)[...,x1:x2,y1:y2]
            for j in range(min_r,max_r+1):
                plane_mask=torch.where(object_r==j,one,zero)[x1:x2,y1:y2]
                xp1,xp2,yp1,yp2=crop(plane_mask)
                plane_mask=plane_mask[xp1:xp2,yp1:yp2]
                plane=disparity[...,x1:x2,y1:y2][...,xp1:xp2,yp1:yp2]*plane_mask
                plane=self.aggregation_dense(feature[...,x1:x2,y1:y2][...,xp1:xp2,yp1:yp2],plane)*plane_mask
                disparity[...,x1:x2,y1:y2][...,xp1:xp2,yp1:yp2]=plane
                
        return disparity


