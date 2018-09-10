# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-07-18 18:49:15
# @Last Modified by:   yulidong
# @Last Modified time: 2018-09-09 23:23:41
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from multiprocessing import Process,Lock
from multiprocessing import Pool
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
thread_num=10

def pre_processing(start,end):
    output_dir=r'/home/lidong/Documents/datasets/Driving/train_data_clean_pass/left_re/'
    P_dir=r'/home/lidong/Documents/datasets/Driving/train_data_clean_pass/left/'
    P_files=os.listdir(P_dir)
    P_files.sort()    
    for f in range(int(start),int(end)):
        P=np.load(os.path.join(P_dir,P_files[f]))

        labels=[]
        start_time=time.time()
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
        data=np.concatenate([l_image,
                        r_image,
                        np.reshape(disparity,[disparity.shape[0],disparity.shape[1],1]),
                        np.reshape(P1,[disparity.shape[0],disparity.shape[1],1]),
                        np.reshape(P2,[disparity.shape[0],disparity.shape[1],1]),
                        np.reshape(P3,[disparity.shape[0],disparity.shape[1],1]),
                        np.reshape(P4,[disparity.shape[0],disparity.shape[1],1]),
                        ],
                        axis=2)
        np.save(os.path.join(output_dir,P_files[f]),data)
        print(time.time()-start_time)
        print(os.path.join(output_dir,P_files[f]))
    output_dir=r'/home/lidong/Documents/datasets/Driving/train_data_clean_pass/right_re/'
    P_dir=r'/home/lidong/Documents/datasets/Driving/train_data_clean_pass/right/'
    P_files=os.listdir(P_dir)
    P_files.sort()    
    for f in range(int(start),int(end)):
        P=np.load(os.path.join(P_dir,P_files[f]))

        labels=[]
        start_time=time.time()
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
        data=np.concatenate([l_image,
                        r_image,
                        np.reshape(disparity,[disparity.shape[0],disparity.shape[1],1]),
                        np.reshape(P1,[disparity.shape[0],disparity.shape[1],1]),
                        np.reshape(P2,[disparity.shape[0],disparity.shape[1],1]),
                        np.reshape(P3,[disparity.shape[0],disparity.shape[1],1]),
                        np.reshape(P4,[disparity.shape[0],disparity.shape[1],1]),
                        ],
                        axis=2)
        np.save(os.path.join(output_dir,P_files[f]),data)
        print(time.time()-start)
        print(os.path.join(output_dir,P_files[f]))
        
process = []
left_dir=r'/home/lidong/Documents/datasets/Driving/train_data_clean_pass/left/'
left_files=os.listdir(left_dir)
left_files.sort()
length=len(left_files)
start=[]
end=[]
p = Pool(thread_num)
for z in range(thread_num):
    start.append(z*length/10)
    end.append((z+1)*length/10)
for z in range(thread_num):
    p.apply_async(pre_processing, args=(start[z],end[z]))

p.close()
p.join()
#pre_processing(0,1)
print('end')
