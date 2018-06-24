# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-06-20 14:37:27
# @Last Modified by:   yulidong
# @Last Modified time: 2018-06-22 21:46:22
import cupy
import cv2
import numpy
import os
from python_pfm import *
import time
p_left_image=r'/home/lidong/Documents/datasets/Driving/generation_test/left/'
p_right_image=r'/home/lidong/Documents/datasets/Driving/generation_test/right/'
p_disparity=r'/home/lidong/Documents/datasets/Driving/generation_test/disparity/'
p_sematnic=r'/home/lidong/Documents/datasets/Driving/generation_test/semantic/'
visual_dir=r'/home/lidong/Documents/datasets/Driving/generation_test/visual/'
n_l_image=os.listdir(p_left_image)
n_r_image=os.listdir(p_right_image)
n_disparity=os.listdir(p_disparity)
n_semantic=os.listdir(p_sematnic)
length=len(n_l_image)

for i in range(length):
    labels=[]
    start=time.time()
    l_image=np.array(cv2.imread(os.path.join(p_left_image,n_l_image[i])))
    r_image=np.array(cv2.imread(os.path.join(p_right_image,n_r_image[i])))
    disparity=np.array(readPFM(os.path.join(p_disparity,n_disparity[i]))[0])
    semantic=np.array(readPFM(os.path.join(p_sematnic,n_semantic[i]))[0])
    #cv2.imwrite(os.path.join(visual_dir,n_l_image[i]),semantic)
    #print(semantic)
    if i==i:
        j=0
        end=np.max(semantic)
        #print(end)
        while(j<end+1):
            count=np.sum(np.where(semantic==j))
            if count>0:
                if not j in (labels):
                    labels.append(j)
                #print(j,count)
            j=j+1
        # #visual
        # for k in range(len(labels)):
        #     semantic=np.where(semantic==labels[k],k,semantic)
        # semantic=semantic*np.floor((255/np.max(semantic)))
        # cv2.imwrite(os.path.join(visual_dir,n_l_image[i]),semantic)
        print(time.time()-start)
        print(labels)   
        print(len(labels))