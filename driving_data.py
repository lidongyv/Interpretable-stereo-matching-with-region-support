# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-06-20 14:37:27
# @Last Modified by:   yulidong
# @Last Modified time: 2018-08-10 21:32:24

import cv2
import numpy as np
import os
from python_pfm import *
from scipy import stats
import matplotlib.pyplot as plt
from skimage.filters import roberts, sobel, scharr, prewitt
#from skimage.feature import hog
from skimage import exposure
import time
division=10
division2=50
# p_left_image=r'/home/lidong/Documents/datasets/Driving/generation_test/left/'
# p_right_image=r'/home/lidong/Documents/datasets/Driving/generation_test/right/'
# p_disparity=r'/home/lidong/Documents/datasets/Driving/generation_test/disparity/'
# p_sematnic=r'/home/lidong/Documents/datasets/Driving/generation_test/object/'
# #visual_dir=r'/home/lidong/Documents/datasets/Driving/generation_test/visual/'
# output_dir=r'/home/lidong/Documents/datasets/Driving/generation_test/train/'
# n_l_image=os.listdir(p_left_image)
# n_l_image.sort()
# n_r_image=os.listdir(p_right_image)
# n_r_image.sort()
# n_disparity=os.listdir(p_disparity)
# n_disparity.sort()
# n_object=os.listdir(p_sematnic)
# n_object.sort()
pathl=[

        r'/home/lidong/Documents/datasets/Driving/frames_cleanpass/15mm_focallength/scene_forwards/slow/left',

        r'/home/lidong/Documents/datasets/Driving/frames_cleanpass/15mm_focallength/scene_forwards/fast/left',

        r'/home/lidong/Documents/datasets/Driving/frames_cleanpass/15mm_focallength/scene_backwards/slow/left',

        r'/home/lidong/Documents/datasets/Driving/frames_cleanpass/15mm_focallength/scene_backwards/fast/left',

        r'/home/lidong/Documents/datasets/Driving/frames_cleanpass/35mm_focallength/scene_forwards/slow/left',

        r'/home/lidong/Documents/datasets/Driving/frames_cleanpass/35mm_focallength/scene_forwards/fast/left',

        r'/home/lidong/Documents/datasets/Driving/frames_cleanpass/35mm_focallength/scene_backwards/slow/left',

        r'/home/lidong/Documents/datasets/Driving/frames_cleanpass/35mm_focallength/scene_backwards/fast/left'      
      ]
pathl.sort()
pathr=[
        r'/home/lidong/Documents/datasets/Driving/frames_cleanpass/15mm_focallength/scene_forwards/slow/right',

        r'/home/lidong/Documents/datasets/Driving/frames_cleanpass/15mm_focallength/scene_forwards/fast/right',

        r'/home/lidong/Documents/datasets/Driving/frames_cleanpass/15mm_focallength/scene_backwards/slow/right',

        r'/home/lidong/Documents/datasets/Driving/frames_cleanpass/15mm_focallength/scene_backwards/fast/right',

        r'/home/lidong/Documents/datasets/Driving/frames_cleanpass/35mm_focallength/scene_forwards/slow/right',

        r'/home/lidong/Documents/datasets/Driving/frames_cleanpass/35mm_focallength/scene_forwards/fast/right',

        r'/home/lidong/Documents/datasets/Driving/frames_cleanpass/35mm_focallength/scene_backwards/slow/right',

        r'/home/lidong/Documents/datasets/Driving/frames_cleanpass/35mm_focallength/scene_backwards/fast/right',
     
      ]
pathr.sort()
pathd=[

        r'/home/lidong/Documents/datasets/Driving/disparity/15mm_focallength/scene_forwards/slow/left',

        r'/home/lidong/Documents/datasets/Driving/disparity/15mm_focallength/scene_forwards/fast/left',

        r'/home/lidong/Documents/datasets/Driving/disparity/15mm_focallength/scene_backwards/slow/left',

        r'/home/lidong/Documents/datasets/Driving/disparity/15mm_focallength/scene_backwards/fast/left',

        r'/home/lidong/Documents/datasets/Driving/disparity/35mm_focallength/scene_forwards/slow/left',

        r'/home/lidong/Documents/datasets/Driving/disparity/35mm_focallength/scene_forwards/fast/left',

        r'/home/lidong/Documents/datasets/Driving/disparity/35mm_focallength/scene_backwards/slow/left',

        r'/home/lidong/Documents/datasets/Driving/disparity/35mm_focallength/scene_backwards/fast/left'      
      ]
pathd.sort()
paths=[

        r'/home/lidong/Documents/datasets/Driving/object_index/15mm_focallength/scene_forwards/slow/left',

        r'/home/lidong/Documents/datasets/Driving/object_index/15mm_focallength/scene_forwards/fast/left',

        r'/home/lidong/Documents/datasets/Driving/object_index/15mm_focallength/scene_backwards/slow/left',

        r'/home/lidong/Documents/datasets/Driving/object_index/15mm_focallength/scene_backwards/fast/left',

        r'/home/lidong/Documents/datasets/Driving/object_index/35mm_focallength/scene_forwards/slow/left',

        r'/home/lidong/Documents/datasets/Driving/object_index/35mm_focallength/scene_forwards/fast/left',

        r'/home/lidong/Documents/datasets/Driving/object_index/35mm_focallength/scene_backwards/slow/left',

        r'/home/lidong/Documents/datasets/Driving/object_index/35mm_focallength/scene_backwards/fast/left'      
      ]
paths.sort()
p_left_image=[]
p_right_image=[]
p_disparity=[]
p_semantic=[]
output_dir=r'/home/lidong/Documents/datasets/Driving/train_data_clean_pass/left/'
path=pathd
for p in range(len(path)):
    file=os.listdir(path[p])
    file.sort()
    for f in range(len(file)):
        p_disparity.append(os.path.join(path[p],file[f]))
        #print(os.path.join(path[p],file[f]))
path=paths
for p in range(len(path)):
    file=os.listdir(path[p])
    file.sort()
    for f in range(len(file)):
        p_semantic.append(os.path.join(path[p],file[f]))
        #print(os.path.join(path[p],file[f]))
path=pathl
for p in range(len(path)):
    file=os.listdir(path[p])
    file.sort()
    for f in range(len(file)):
        p_left_image.append(os.path.join(path[p],file[f]))
        #print(os.path.join(path[p],file[f]))
path=pathr
for p in range(len(path)):
    file=os.listdir(path[p])
    file.sort()
    for f in range(len(file)):
        p_right_image.append(os.path.join(path[p],file[f]))
        #print(os.path.join(path[p],file[f]))
length=len(p_left_image)

for f in range(length):
    print(f)
    labels=[]
    start=time.time()
    l_image=np.array(cv2.imread(p_left_image[f]))[:,:,::-1]
    r_image=np.array(cv2.imread(p_right_image[f]))[:,:,::-1]
    disparity=np.array(readPFM(p_disparity[f])[0])
    object=np.array(readPFM(p_semantic[f])[0])
    # #object statistics
    j=0
    end=np.max(object)
    #print(end)
    while(j<end+1):
        count=np.sum(np.where(object==j))
        if count>0:
            if not j in (labels):
                labels.append(j)
            #print(j,count)
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
    #print(time.time()-start)
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
        #print(time.time()-start)
        i_region=0
        for j in range(min,max+1):
            i_region+=1
            region_b=np.where(np.logical_and(region>=j,region<j+1),1.0,0.0)
            edge_size=np.sum(region_b)
            box=np.argwhere(region_b==1)


            if box.shape[0]>0:       
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
    np.save(os.path.join(output_dir,str(f)+'.npy'),data)
    print(time.time()-start)
    print(os.path.join(output_dir,str(f)+'.npy'))
# print(np.sum(np.where(P4>0,1,0))/(P4.shape[0]*P4.shape[1]))
# overlap=P4-P1
# overlap=np.where(overlap>0,1,0)    
# fig, ax = plt.subplots(nrows=5,ncols=1, sharex=True, sharey=True,figsize=(32, 64))
# ax[0].imshow(P1, cmap=plt.cm.gray)
# ax[1].imshow(P2, cmap=plt.cm.gray)
# ax[2].imshow(P3_v, cmap=plt.cm.gray)
# ax[3].imshow(P4, cmap=plt.cm.gray)
# ax[4].imshow(overlap, cmap=plt.cm.gray)

