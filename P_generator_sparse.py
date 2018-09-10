# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-06-20 14:37:27
# @Last Modified by:   yulidong
# @Last Modified time: 2018-09-09 23:04:57

import cv2
import numpy
import os
import sys
from python_pfm import *
from scipy import stats
import matplotlib.pyplot as plt
from skimage.filters import roberts, sobel, scharr, prewitt
#from skimage.feature import hog
from skimage import exposure
import time
division=20
P_dir=r'//home/lidong/Documents/datasets/Driving/train_data_clean_pass/left'
P=np.load(os.path.join(P_dir,'0.npy'))
for i in range(1):
    labels=[]
    start=time.time()
    l_image=P[...,0:3]
    r_image=P[...,3:6]
    disparity=P[...,6]
    P1=P[...,7]
    P3=P[...,8]
    P4=P[...,9]
    P2=P[...,10]

P1=np.zeros_like(P1)
discriminative=np.zeros_like(P1)
P3=P3.astype(np.int)
P2=np.zeros_like(P1)
for i in range(np.max(P3)+1):
    region=np.where(P3==i,1.0,0.0)   
    if np.sum(region)<100:
        P3=np.where(P3==i,P3+1,P3)
count=-1
for i in range(np.max(P3)+1):
    region=np.where(P3==i,1.0,0.0)  
    if np.sum(region)>0:
        P3=np.where(P3==i,count,P3)
        count-=1
P3=-P3
for i in range(np.max(P3)+1):
    region=np.where(P3==i,P4,0.0)
    for j in range(1,np.max(region).astype(np.int)):
        plane=np.where(region==j,1.0,0.0)
        if np.sum(plane)<36:
            region=np.where(region==j,region+1,region)
    count=-1
    for j in range(1,np.max(region).astype(np.int)):
        plane=np.where(region==j,1.0,0.0)
        if np.sum(plane)>0:        
            region=np.where(region==j,count,region)
            count-=1
    P4=np.where(P3==i,region,P4)    

for i in range(np.max(P3)+1):
    region=np.where(P3==i,1.0,0.0)
    if np.sum(region)<100:
        P3=np.where(P3==i,P3+1,P3)
    edge_sobel = sobel(region)
    edges=edge_sobel
    edges=np.where(edges>0.001,1.0,0.0)
    discriminative=np.where(edges+region==2,1.0,0.0)
    P1+=discriminative
    region=np.where(P3==i,P4,0.0)
    for j in range(1,np.max(region).astype(np.int)):
        plane=np.where(region==j,1.0,0.0)
        edge_sobel = sobel(plane)
        edges=edge_sobel
        edges=np.where(edges>0.001,1.0,0.0)
        discriminative=np.where(edges+plane==2,1.0,0.0)
        P2+=discriminative


P1=np.where(P1>0,1,0)
P2=np.where(P2>0,1,0)
P2=P2-P1
P2=np.where(P2>0,1,0)
