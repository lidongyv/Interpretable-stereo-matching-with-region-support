# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-06-20 14:37:27
# @Last Modified by:   yulidong
# @Last Modified time: 2018-09-19 22:39:45

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
from skimage.filters import roberts, sobel, scharr, prewitt
#from skimage.feature import hog
from skimage import exposure
division=10
division2=50
thread_num=8
def pre_processing(start,end):
    output_dir=r'/home/dataset2/flying3d/train/'
    train=np.load(os.path.join(output_dir,'train.npy'))
    test=np.load(os.path.join(output_dir,'test.npy'))
    left=train[0]
    right=train[1]
    length=len(left)
    for f in range(start,end):
        labels=[]
        start_time=time.time()
        l_image=np.array(cv2.imread(left[f][0]))[:,:,::-1]
        r_image=np.array(cv2.imread(right[f][0]))[:,:,::-1]
        disparity=np.array(readPFM(left[f][1])[0])
        object=np.array(readPFM(left[f][2])[0])

        # #object statistics
        j=0
        max_index=np.max(object)
        while(j<max_index+1):
            count=np.sum(np.where(object==j))
            if count>0:
                if j not in (labels):
                    labels.append(j)
            j=j+1
        #visual
        for k in range(len(labels)):
            object=np.where(object==labels[k],k,object)
        P2=object
        #P1

        image=disparity/np.max(disparity)
        edge_roberts = roberts(image)
        edge_sobel = sobel(image)
        edge_prewitt = prewitt(image)
        edge_scharr = scharr(image)
        edge1=edge_roberts+edge_sobel+edge_prewitt+edge_scharr

        image=object/np.max(object)
        edge_roberts = roberts(image)
        edge_sobel = sobel(image)
        edge_prewitt = prewitt(image)
        edge_scharr = scharr(image)
        edge2=edge_roberts+edge_sobel+edge_prewitt+edge_scharr


        edge1=np.where(edge1>0.1,1,0)
        edge2=np.where(edge2>0.01,1,0)
        #edge3=np.where(edge3>0.04,1,0)
        edges=edge1+edge2
        edges=np.where(edges>=1,1,0)
        P1=edges
        #print(np.sum(np.where(P1>0,1,0))/(object.shape[0]*object.shape[1]))
        #print(time.time()-start_time)
        #P3

        image=P2
        depth=disparity
        segmentation=P2.astype(np.int32)
        P4=np.zeros([segmentation.shape[0],segmentation.shape[1]])
        P3=np.zeros([segmentation.shape[0],segmentation.shape[1]])
        P3_v=np.zeros([segmentation.shape[0],segmentation.shape[1]])
        for i in range(np.max(segmentation)+1):
            rbox=np.argwhere(segmentation==i)
            if rbox.shape[0]<=0: 
                continue
            rx_min=np.min(rbox[:,0])
            rx_max=np.max(rbox[:,0])
            ry_min=np.min(rbox[:,1])
            ry_max=np.max(rbox[:,1])
            image=segmentation[rx_min:rx_max+1,ry_min:ry_max+1]
            depth=disparity[rx_min:rx_max+1,ry_min:ry_max+1]
            region=np.where(image==i,depth,0)

            max=np.floor(np.max(region)).astype(np.int32)
            region=np.where(region==0,max+1,region)
            min=np.floor(np.min(region)).astype(np.int32)
            #print(time.time()-start_time)
            i_region=0
            for j in range(min,max+1):
                
                region_b=np.where(np.logical_and(region>=j,region<j+1),1.0,0.0)
                edge_size=np.sum(region_b)
                box=np.argwhere(region_b==1)


                if box.shape[0]>0:
                    i_region+=1    
                    P3[rx_min:rx_max+1,ry_min:ry_max+1]+=region_b*i_region
           
                    #P4
                    #print(box.shape[0])
                    x_min=np.min(box[:,0])
                    x_max=np.max(box[:,0])
                    y_min=np.min(box[:,1])
                    y_max=np.max(box[:,1])
                    b_size=(x_max-x_min)*(y_max-y_min)
                    if b_size>0:
                        image=region_b[x_min:x_max+1,y_min:y_max+1]
                        edge_sobel = sobel(image)
                        edges=edge_sobel
                        edges=np.where(edges>0.01,1,0)
                        if np.sum(edges)<edge_size:
                            region_b[x_min:x_max+1,y_min:y_max+1]=edges
                            P4[rx_min+x_min:rx_min+x_max+1,ry_min+y_min:ry_min+y_max+1]+=region_b[x_min:x_max+1,y_min:y_max+1]
                        else:
                            for m in range(x_min,x_max+1):
                                line=np.argwhere(region_b[m,y_min:y_max+1]>0)
                                for n in range(np.ceil(line.shape[0]/division).astype(np.int32)):
                                    P4[rx_min+m,ry_min+y_min+line[n,0]]=1
                                if line.shape[0]>0:
                                    P4[rx_min+m,ry_min+y_min+line[-1,0]]=1                       

                    else:
                        #get 10% pixels as representive pixels
                        for m in range(((y_max+1-y_min)/division2).astype(np.int32)):
                            P4[rx_min+x_max,np.floor(ry_min+y_min+m*division2).astype(np.int32)]=1
                        P4[rx_min+x_max,ry_min+y_max]=1
            P3_v=P3_v+np.where(segmentation==i,P3,0)*255/i_region

        P4=np.where(P4>0,1,0)

        data=np.concatenate([l_image,
                        r_image,
                        np.reshape(disparity,[disparity.shape[0],disparity.shape[1],1]),
                        np.reshape(P1,[disparity.shape[0],disparity.shape[1],1]),
                        np.reshape(P2,[disparity.shape[0],disparity.shape[1],1]),
                        np.reshape(P3,[disparity.shape[0],disparity.shape[1],1]),
                        np.reshape(P4,[disparity.shape[0],disparity.shape[1],1]),
                        ],
                        axis=2)
        
        np.save(os.path.join(output_dir,r'left',str(f)+'.npy'),data)
        print(f,start,end,time.time()-start_time)
        print(os.path.join(output_dir,r'left',str(f)+'.npy'))

    #right
    length=len(right)
    for f in range(start,end):
        labels=[]
        start_time=time.time()
        l_image=np.array(cv2.imread(left[f][0]))[:,:,::-1]
        r_image=np.array(cv2.imread(right[f][0]))[:,:,::-1]
        disparity=np.array(readPFM(right[f][1])[0])
        object=np.array(readPFM(right[f][2])[0])
        # #object statistics
        j=0
        max_index=np.max(object)
        #print(end)
        while(j<max_index+1):
            count=np.sum(np.where(object==j))
            if count>0:
                if j not in (labels):
                    labels.append(j)
            j=j+1
        #visual
        for k in range(len(labels)):
            object=np.where(object==labels[k],k,object)
        P2=object
        #P1
        image=disparity/np.max(disparity)
        edge_roberts = roberts(image)
        edge_sobel = sobel(image)
        edge_prewitt = prewitt(image)
        edge_scharr = scharr(image)
        edge1=edge_roberts+edge_sobel+edge_prewitt+edge_scharr

        image=object/np.max(object)
        edge_roberts = roberts(image)
        edge_sobel = sobel(image)
        edge_prewitt = prewitt(image)
        edge_scharr = scharr(image)
        edge2=edge_roberts+edge_sobel+edge_prewitt+edge_scharr


        edge1=np.where(edge1>0.1,1,0)
        edge2=np.where(edge2>0.01,1,0)
        #edge3=np.where(edge3>0.04,1,0)
        edges=edge1+edge2
        edges=np.where(edges>=1,1,0)
        P1=edges
        #print(np.sum(np.where(P1>0,1,0))/(object.shape[0]*object.shape[1]))
        #print(time.time()-start_time)
        #P3

        image=P2
        depth=disparity
        segmentation=P2.astype(np.int32)
        P4=np.zeros([segmentation.shape[0],segmentation.shape[1]])
        P3=np.zeros([segmentation.shape[0],segmentation.shape[1]])
        P3_v=np.zeros([segmentation.shape[0],segmentation.shape[1]])
        for i in range(np.max(segmentation)+1):
            rbox=np.argwhere(segmentation==i)
            if rbox.shape[0]<=0: 
                continue
            rx_min=np.min(rbox[:,0])
            rx_max=np.max(rbox[:,0])
            ry_min=np.min(rbox[:,1])
            ry_max=np.max(rbox[:,1])
            image=segmentation[rx_min:rx_max+1,ry_min:ry_max+1]
            depth=disparity[rx_min:rx_max+1,ry_min:ry_max+1]
            region=np.where(image==i,depth,0)

            max=np.floor(np.max(region)).astype(np.int32)
            region=np.where(region==0,max+1,region)
            min=np.floor(np.min(region)).astype(np.int32)
            #print(time.time()-start_time)
            i_region=0
            for j in range(min,max+1):
                
                region_b=np.where(np.logical_and(region>=j,region<j+1),1.0,0.0)
                edge_size=np.sum(region_b)
                box=np.argwhere(region_b==1)


                if box.shape[0]>0:
                    i_region+=1    
                    P3[rx_min:rx_max+1,ry_min:ry_max+1]+=region_b*i_region
           
                    #P4
                    #print(box.shape[0])
                    x_min=np.min(box[:,0])
                    x_max=np.max(box[:,0])
                    y_min=np.min(box[:,1])
                    y_max=np.max(box[:,1])
                    b_size=(x_max-x_min)*(y_max-y_min)
                    if b_size>0:
                        image=region_b[x_min:x_max+1,y_min:y_max+1]
                        edge_sobel = sobel(image)
                        edges=edge_sobel
                        edges=np.where(edges>0.01,1,0)
                        if np.sum(edges)<edge_size:
                            region_b[x_min:x_max+1,y_min:y_max+1]=edges
                            P4[rx_min+x_min:rx_min+x_max+1,ry_min+y_min:ry_min+y_max+1]+=region_b[x_min:x_max+1,y_min:y_max+1]
                        else:
                            for m in range(x_min,x_max+1):
                                line=np.argwhere(region_b[m,y_min:y_max+1]>0)
                                for n in range(np.ceil(line.shape[0]/division).astype(np.int32)):
                                    P4[rx_min+m,ry_min+y_min+line[n,0]]=1
                                if line.shape[0]>0:
                                    P4[rx_min+m,ry_min+y_min+line[-1,0]]=1                       

                    else:
                        #get 10% pixels as representive pixels
                        for m in range(((y_max+1-y_min)/division2).astype(np.int32)):
                            P4[rx_min+x_max,np.floor(ry_min+y_min+m*division2).astype(np.int32)]=1
                        P4[rx_min+x_max,ry_min+y_max]=1
            P3_v=P3_v+np.where(segmentation==i,P3,0)*255/i_region

        P4=np.where(P4>0,1,0)

        data=np.concatenate([l_image,
                        r_image,
                        np.reshape(disparity,[disparity.shape[0],disparity.shape[1],1]),
                        np.reshape(P1,[disparity.shape[0],disparity.shape[1],1]),
                        np.reshape(P2,[disparity.shape[0],disparity.shape[1],1]),
                        np.reshape(P3,[disparity.shape[0],disparity.shape[1],1]),
                        np.reshape(P4,[disparity.shape[0],disparity.shape[1],1]),
                        ],
                        axis=2)
        np.save(os.path.join(output_dir,r'right',str(f)+'.npy'),data)
        print(f,start,end,time.time()-start_time)
        print(os.path.join(output_dir,r'right',str(f)+'.npy'))



if __name__=='__main__':
    process = []
    output_dir=r'/home/lidong/Documents/datasets/monkey/train/'
    train=np.load(os.path.join(output_dir,'train.npy'))
    test=np.load(os.path.join(output_dir,'test.npy'))
    left=train[0]
    right=train[1]
    length=len(left)
    start=[]
    end=[]
    p = Pool(thread_num)
    for z in range(thread_num):
        start.append(int(np.floor(z*length/thread_num)))
        end.append(int(np.ceil((z+1)*length/thread_num)))
    for z in range(thread_num):
        p.apply_async(pre_processing, args=(start[z],end[z]))

    p.close()
    p.join()
    #pre_processing(0,1)
    print('end')